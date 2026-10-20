"""RayPPOTrainer subclass: injects step/validation into extra_info, logs per-step measurements,
and installs the token-credit advantage.

Token credit (credit == "token", uncentered, adopted 2026-10-19 after the sequence arm stalled on
monitor saturation): A_{i,t} = a_out_i + R_cot_i * w_{i,t}, where a_out is the group-mean-centered
R_out, R_cot_i = -p * flagged_i is NOT centered, w = L_i/|span_i| on monitor-quoted tokens and 0
elsewhere for flagged samples with a found span, and w = 1 everywhere for flagged samples without a
span. Unflagged samples get no penalty term. Total penalty mass per flagged sample equals -p
broadcast over all L_i tokens; only its placement differs. credit == "token_centered" keeps the
original pre-registered formula with a_cot = R_cot - group mean. Sequence credit uses verl's GRPO
with norm_adv_by_std_in_grpo=False (Dr.GRPO), i.e. A_i = (R_out + R_cot) - group mean.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from verl.trainer.ppo import ray_trainer
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from penalty_shape.reward import append_jsonl

_original_compute_advantage = ray_trainer.compute_advantage


def center(values, uid):
    groups = {}
    for v, u in zip(values, uid):
        groups.setdefault(u, []).append(v)
    means = {u: float(np.mean(v)) for u, v in groups.items()}
    return np.array([v - means[u] for v, u in zip(values, uid)], dtype=np.float32)


def token_credit_advantage(data, centered):
    mask = data.batch["response_mask"].float()
    uid = data.non_tensor_batch["uid"]
    a_out = torch.tensor(center(np.asarray(data.non_tensor_batch["r_out"], dtype=float), uid))
    r_cot = np.asarray(data.non_tensor_batch["r_cot"], dtype=float)
    a_cot = torch.tensor(center(r_cot, uid) if centered else r_cot.astype(np.float32))
    w = torch.ones_like(mask)
    lengths = mask.sum(-1)
    for i, span in enumerate(data.non_tensor_batch["span_json"]):
        idx = [t for t in json.loads(span) if t < mask.shape[1]]
        if r_cot[i] < 0 and idx:
            w[i] = 0.0
            w[i, idx] = float(lengths[i]) / len(idx)
    adv = (a_out[:, None] + a_cot[:, None] * w) * mask
    data.batch["advantages"] = adv
    data.batch["returns"] = adv
    return data


def compute_advantage(data, adv_estimator, **kwargs):
    credit = os.environ.get("PENALTY_CREDIT")
    if credit in ("token", "token_centered") and adv_estimator == AdvantageEstimator.GRPO:
        return token_credit_advantage(data, centered=credit == "token_centered")
    return _original_compute_advantage(data, adv_estimator, **kwargs)


ray_trainer.compute_advantage = compute_advantage


class PenaltyTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run = json.load(open(os.environ["PENALTY_RUN_CONFIG"]))
        self.root = Path(self.run["dir"])
        self.in_validation = False
        self.step_started = None

    def _get_gen_batch(self, batch):
        if not self.in_validation:
            self.step_started = time.perf_counter()
        for i, original in enumerate(batch.non_tensor_batch["extra_info"]):
            info = dict(original)
            info["step"] = self.global_steps
            info["validation"] = self.in_validation
            batch.non_tensor_batch["extra_info"][i] = info
        return super()._get_gen_batch(batch)

    def _validate(self, *args, **kwargs):
        self.in_validation = True
        try:
            return super()._validate(*args, **kwargs)
        finally:
            self.in_validation = False

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        reward_extra_infos_dict.pop("span_json", None)
        return super()._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _log_rollout_data(self, batch, reward_extra_infos_dict, timing_raw, rollout_data_dir):
        super()._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)
        values = {k: np.asarray(v, dtype=float) for k, v in reward_extra_infos_dict.items() if k != "span_json"}
        scores = batch.batch["token_level_scores"].sum(-1).cpu().numpy()
        groups = {}
        for u, s in zip(batch.non_tensor_batch["uid"], scores):
            groups.setdefault(u, []).append(s)
        cot = {}
        for u, c in zip(batch.non_tensor_batch["uid"], values.get("r_cot", scores)):
            cot.setdefault(u, []).append(c)
        event = {"kind": "training_update", "step": self.global_steps, "mean_reward": float(scores.mean()),
                 "zero_variance_groups": float(np.mean([np.var(v) == 0 for v in groups.values()])),
                 "zero_variance_cot_groups": float(np.mean([np.var(v) == 0 for v in cot.values()])),
                 "response_tokens_mean": float(batch.batch["response_mask"].sum(-1).float().mean()),
                 "timing": {k: float(v) for k, v in timing_raw.items()},
                 "elapsed_seconds": time.perf_counter() - self.step_started,
                 **{k: float(v.mean()) for k, v in values.items()}}
        append_jsonl(self.root / "measurements.jsonl", event)

    def _save_checkpoint(self):
        start = time.perf_counter()
        super()._save_checkpoint()
        append_jsonl(self.root / "measurements.jsonl", {"kind": "checkpoint", "step": self.global_steps,
                                                        "seconds": time.perf_counter() - start})
