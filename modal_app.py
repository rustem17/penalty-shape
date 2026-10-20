"""Modal app for the penalty_shape E0 pilot (workspace praxis-research).

Policy: Qwen/Qwen3-8B (thinking on), temperature 0.8, prompt cap 1024, response cap RESPONSE_TOKENS.
Env: Backdoor-Easy on MBPP sanitized (tasks.py). Reward (reward.py): R_out = C*(B + 0.5*(1-B)),
R_cot = -p*[M1 >= 3], M1 = claude-haiku-5-5 on problem + CoT. Verifier runs in network-blocked
CPU containers. Training: verl GRPO (pinned commit), Dr.GRPO advantages (no std, no length norm,
KL 0, entropy 0), 16 prompts x 16 completions per cycle, one optimizer step per cycle, lr 1e-6,
full fine-tuning on H200:2. Credit assignment: sequence (verl GRPO) or token (trainer.py).
Dev eval: 64 tasks x 4 samples, before training and every eval_every cycles.

Heads-And-Tails (Kaufmann 2x2, coin.py / coin_reward.py): coin_baseline = vLLM on one H100, base Qwen3-8B, thinking
off, their prompt, temperature 1.0, with reasoning and with reasoning forced empty. train_coin = verl standard GRPO
(advantage divided by the group standard deviation, token-mean loss, KL 0, entropy 0), 16 prompts x 16 answers, lr 1e-6,
temperature 1.0, response cap COIN_RESPONSE_TOKENS, dev 64 prompts x 1 every 10 steps, checkpoint every 25 steps,
H200:2; penalty per cell in COIN_CELLS.
Checkpoints save synchronously (async_save off) so latest_checkpointed_iteration.txt exists and restarts resume.
GPU policy: at most 4 GPUs at once (shared workspace).
"""
import json
import os
import subprocess
import time
from pathlib import Path

import modal

from penalty_shape.config import (APP_NAME, COIN_CELLS, COIN_RESPONSE_TOKENS, COIN_TEMPERATURE, MODEL, PROMPT_TOKENS, R_H,
                                  RESPONSE_TOKENS, TEMPERATURE, VERL_COMMIT, VERL_IMAGE)

ROOT = Path(__file__).parent
app = modal.App(APP_NAME)
vol = modal.Volume.from_name("penalty-shape", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache")
MOUNTS = {"/vol": vol, "/root/.cache/huggingface": hf_cache}
secret = modal.Secret.from_dotenv(str(ROOT))

verify_image = modal.Image.debian_slim(python_version="3.11").add_local_python_source("penalty_shape")

assay_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.28.0", "hf_transfer", "anthropic", "datasets", "pandas", "pyarrow")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("penalty_shape")
)

train_image = (
    modal.Image.from_registry(VERL_IMAGE)
    .run_commands(
        f"git clone https://github.com/verl-project/verl /root/verl && cd /root/verl && git checkout {VERL_COMMIT}",
        "cd /root/verl && pip install --no-deps -e . && pip install hydra-core omegaconf codetiming pylatexenc orjson "
        "torchdata dill datasets wandb tensorboard accelerate openai pandas pyarrow anthropic psutil",
    )
    .run_commands("pip install -U accelerate && python -c 'import transformers, accelerate, torch, vllm; "
                  "print(transformers.__version__, accelerate.__version__, torch.__version__, vllm.__version__)'")
    .env({"PYTHONPATH": "/root"})
    .add_local_dir(str(ROOT / "penalty_shape"), "/root/penalty_shape", copy=True)
    .run_commands("cd /root && python -m penalty_shape.verl_hooks /root/verl")
)


@app.function(image=verify_image, cpu=1.0, memory=2048, timeout=180, block_network=True, max_containers=32,
              scaledown_window=120)
@modal.concurrent(max_inputs=8)
def verify(job):
    from penalty_shape.verifier import run_job
    return run_job(job)


@app.function(image=assay_image, gpu="H100", cpu=8, memory=65536, timeout=7200, volumes=MOUNTS, secrets=[secret])
def assay(run_id, n_tasks=64, n_samples=32, max_tokens=RESPONSE_TOKENS, temperature=TEMPERATURE, seed=0, thinking=True):
    import asyncio

    import anthropic
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from penalty_shape.analysis import summarize_assay
    from penalty_shape.reward import append_jsonl, score_sample
    from penalty_shape.tasks import load_tasks, prompt, split_tasks

    out = Path("/vol/assay") / run_id
    out.mkdir(parents=True, exist_ok=True)
    tasks = split_tasks(load_tasks())["calibration"][:n_tasks]
    path = snapshot_download(MODEL)
    tok = AutoTokenizer.from_pretrained(path)
    prompts = [tok.apply_chat_template(prompt(t), tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
               for t in tasks]
    llm = LLM(model=path, dtype="bfloat16", max_model_len=PROMPT_TOKENS + max_tokens, gpu_memory_utilization=0.9,
              seed=seed, max_num_seqs=256)
    started = time.time()
    outputs = llm.generate(prompts, SamplingParams(n=n_samples, temperature=temperature, top_p=1.0,
                                                   max_tokens=max_tokens, seed=seed, skip_special_tokens=False))
    gen_seconds = time.time() - started
    run = {"dir": str(out), "model_path": path, "p": 0.0, "r_h": R_H}
    client = anthropic.AsyncAnthropic(max_retries=6)
    sem = asyncio.Semaphore(48)

    async def one(task, o, k):
        ids = list(o.token_ids)
        text = o.text
        if ids and ids[-1] == tok.eos_token_id:
            ids = ids[:-1]
            text = tok.decode(ids, skip_special_tokens=False)
        async with sem:
            row, _ = await score_sample(run, tok, client, verify, task, text, ids, o.finish_reason)
        return {**row, "sample": k, "completion": text}

    async def run_all():
        return await asyncio.gather(*[one(t, o, k) for t, ro in zip(tasks, outputs) for k, o in enumerate(ro.outputs)])

    rows = asyncio.run(run_all())
    for r in rows:
        append_jsonl(out / "samples.jsonl", r)
    summary = {**summarize_assay(rows), "gen_seconds": gen_seconds, "max_tokens": max_tokens,
               "temperature": temperature, "model_path": path, "gpu": "H100", "thinking": thinking}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    vol.commit()
    return summary


@app.function(image=assay_image, gpu="H100", cpu=8, memory=65536, timeout=3600, volumes=MOUNTS)
def coin_baseline(run_id, n_tasks=64, n_samples=16, max_tokens=2048, temperature=1.0, seed=0):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from penalty_shape.coin import EMPTY_PREFILL, make_tasks, messages

    out = Path("/vol/coin_baseline") / run_id
    out.mkdir(parents=True, exist_ok=True)
    tasks = make_tasks(n_tasks, seed)
    path = snapshot_download(MODEL)
    tok = AutoTokenizer.from_pretrained(path)
    prompts = [tok.apply_chat_template(messages(t), tokenize=False, add_generation_prompt=True, enable_thinking=False)
               for t in tasks]
    llm = LLM(model=path, dtype="bfloat16", max_model_len=PROMPT_TOKENS + max_tokens, gpu_memory_utilization=0.9, seed=seed)
    rows = []
    for mode, suffix, cap in (("reasoning", "", max_tokens), ("empty", EMPTY_PREFILL, 16)):
        outputs = llm.generate([p + suffix for p in prompts], SamplingParams(n=n_samples, temperature=temperature, top_p=1.0,
                                                                            max_tokens=cap, seed=seed, skip_special_tokens=False))
        rows += [{"mode": mode, "task": t, "sample": k, "completion": suffix + o.text, "tokens": len(o.token_ids),
                  "finish_reason": o.finish_reason} for t, ro in zip(tasks, outputs) for k, o in enumerate(ro.outputs)]
    with open(out / "samples.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    json.dump({"prompt_example": prompts[0], "n": len(rows)}, open(out / "meta.json", "w"), indent=1)
    vol.commit()
    return {"n": len(rows), "prompt_example": prompts[0]}


def overrides(run, directory):
    ctx = PROMPT_TOKENS + run["response_tokens"]
    values = {
        "data.train_files": f"{directory}/train.parquet", "data.val_files": f"{directory}/dev.parquet",
        "data.train_batch_size": run["groups"], "data.max_prompt_length": PROMPT_TOKENS,
        "data.max_response_length": run["response_tokens"], "data.filter_overlong_prompts": True,
        "data.truncation": "error", "data.seed": run["seed"], "data.shuffle": True,
        "actor_rollout_ref.model.path": run["model_path"],
        "actor_rollout_ref.actor.optim.lr": run["lr"],
        "actor_rollout_ref.actor.ppo_mini_batch_size": run["groups"],
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.use_dynamic_bsz": True,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": int(run["token_len_factor"] * ctx),
        "actor_rollout_ref.actor.fsdp_config.param_offload": run["param_offload"],
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.calculate_entropy": True,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-sum-norm",
        "actor_rollout_ref.actor.loss_scale_factor": run["response_tokens"],
        "actor_rollout_ref.actor.checkpoint.save_contents": ["model", "optimizer", "extra"],
        "actor_rollout_ref.actor.checkpoint.async_save": False,
        "actor_rollout_ref.actor.checkpoint.load_contents": ["model", "optimizer", "extra"],
        "actor_rollout_ref.rollout.name": "vllm", "actor_rollout_ref.rollout.n": run["n"],
        "actor_rollout_ref.rollout.temperature": TEMPERATURE, "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.top_k": -1, "actor_rollout_ref.rollout.seed": run["seed"],
        "actor_rollout_ref.rollout.max_model_len": ctx,
        "actor_rollout_ref.rollout.max_num_batched_tokens": int(run["token_len_factor"] * ctx),
        "actor_rollout_ref.rollout.gpu_memory_utilization": run["gpu_memory_utilization"],
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": True,
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": int(run["token_len_factor"] * ctx),
        "actor_rollout_ref.rollout.val_kwargs.n": run["dev_n"], "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
        "actor_rollout_ref.rollout.val_kwargs.temperature": TEMPERATURE,
        "actor_rollout_ref.rollout.val_kwargs.top_p": 1.0, "actor_rollout_ref.rollout.val_kwargs.top_k": -1,
        "algorithm.adv_estimator": "grpo", "algorithm.norm_adv_by_std_in_grpo": False,
        "algorithm.use_kl_in_reward": False,
        "reward.custom_reward_function.path": "/root/penalty_shape/reward.py",
        "reward.custom_reward_function.name": "compute_score", "reward.num_workers": 1,
        "trainer.use_v1": False, "trainer.logger": ["console", "file"], "trainer.val_before_train": True,
        "trainer.n_gpus_per_node": run["n_gpus"], "trainer.nnodes": 1, "trainer.critic_warmup": 0,
        "trainer.save_freq": run["save_every"], "trainer.test_freq": run["eval_every"],
        "trainer.total_epochs": 1000, "trainer.total_training_steps": run["cycles"],
        "trainer.project_name": APP_NAME, "trainer.experiment_name": run["run_id"],
        "trainer.default_local_dir": f"{directory}/checkpoints",
        "trainer.resume_mode": "auto" if run["resume"] else "disable",
        "trainer.rollout_data_dir": f"{directory}/rollouts", "trainer.validation_data_dir": f"{directory}/validation",
        "trainer.max_actor_ckpt_to_keep": 1,
        **run.get("extra_overrides", {}),
    }
    return [f"{k}={json.dumps(v, separators=(',', ':'))}" for k, v in values.items()]


def memory_monitor(path, stop):
    import threading
    import psutil

    def cgroup(name):
        v1 = {"memory.current": "memory.usage_in_bytes", "memory.max": "memory.limit_in_bytes"}[name]
        for candidate in (f"/sys/fs/cgroup/{name}", f"/sys/fs/cgroup/memory/{v1}"):
            if Path(candidate).exists():
                text = Path(candidate).read_text().strip()
                return None if text == "max" else round(int(text) / 2**30, 1)
        return None

    def loop():
        while not stop.is_set():
            vm = psutil.virtual_memory()
            with open(path, "a") as f:
                f.write(json.dumps({"time": time.time(), "cgroup_used_gib": cgroup("memory.current"),
                                    "cgroup_limit_gib": cgroup("memory.max"), "host_used_gib": round(vm.used / 2**30, 1),
                                    "rss_top_gib": round(sum(pr.info["memory_info"].rss for pr in psutil.process_iter(["memory_info"])) / 2**30, 1)}) + "\n")
            stop.wait(60)

    threading.Thread(target=loop, daemon=True).start()


GPUS = {"H200:2": {"n_gpus": 2, "gpu_memory_utilization": 0.5, "param_offload": False, "token_len_factor": 2.0},
        "H100:4": {"n_gpus": 4, "gpu_memory_utilization": 0.6, "param_offload": True, "token_len_factor": 1.34}}
TRAIN_KW = dict(image=train_image, cpu=16, memory=262144, timeout=86400, volumes=MOUNTS, secrets=[secret],
                retries=modal.Retries(max_retries=2, initial_delay=60.0, backoff_coefficient=1.0),
                single_use_containers=True)


@app.function(gpu="H200:2", **TRAIN_KW)
def train(run_id, p, credit, seed, cycles, **kw):
    return train_impl(run_id, p, credit, seed, cycles, gpu="H200:2", **kw)


@app.function(gpu="H100:4", **TRAIN_KW)
def train_h100(run_id, p, credit, seed, cycles, **kw):
    return train_impl(run_id, p, credit, seed, cycles, gpu="H100:4", **kw)


def train_impl(run_id, p, credit, seed, cycles, gpu, r_h=R_H, response_tokens=RESPONSE_TOKENS, eval_every=20,
               save_every=10, dev_n=4, groups=16, n=16, lr=1e-6, resume=False):
    import pandas as pd
    from huggingface_hub import snapshot_download

    from penalty_shape.tasks import load_tasks, split_tasks, verl_rows

    directory, attempt, resume = start_run(run_id, resume)
    splits = split_tasks(load_tasks())
    pd.DataFrame(verl_rows(splits["train"])).to_parquet(directory / "train.parquet")
    pd.DataFrame(verl_rows(splits["dev"], validation=True)).to_parquet(directory / "dev.parquet")
    model_path = snapshot_download(MODEL)
    run = {"run_id": run_id, "dir": str(directory), "model_path": model_path, "p": p, "credit": credit, "r_h": r_h,
           "seed": seed, "cycles": cycles, "response_tokens": response_tokens, "eval_every": eval_every,
           "save_every": save_every, "dev_n": dev_n, "groups": groups, "n": n, "lr": lr, **GPUS[gpu],
           "resume": resume, "attempt": attempt, "gpu": gpu,
           "n_train_tasks": len(splits["train"]), "n_dev_tasks": len(splits["dev"])}
    return run_verl(run, directory, gpu, credit)


def start_run(run_id, resume=False):
    directory = Path("/vol/runs") / run_id
    attempt = 1
    if directory.exists():
        attempt = json.load(open(directory / "run.json")).get("attempt", 1) + 1
        resume = (directory / "checkpoints" / "latest_checkpointed_iteration.txt").exists()
    directory.mkdir(parents=True, exist_ok=True)
    return directory, attempt, resume


def run_verl(run, directory, gpu, credit):
    import threading

    json.dump(run, open(directory / "run.json", "w"), indent=1)
    args = overrides(run, directory)
    json.dump({"overrides": args}, open(directory / "launch.json", "w"), indent=1)
    env = {**os.environ, "PENALTY_RUN_CONFIG": str(directory / "run.json"), "PENALTY_CREDIT": credit,
           "VERL_FILE_LOGGER_PATH": str(directory / "metrics.jsonl")}
    vol.commit()
    started = time.time()
    last_commit = started
    stop = threading.Event()
    memory_monitor(directory / "host_mem.jsonl", stop)
    with open(directory / "trainer.log", "a") as log:
        proc = subprocess.Popen(["python", "-m", "verl.trainer.main_ppo", *args], cwd="/root", env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            if time.time() - last_commit > 120:
                vol.commit()
                last_commit = time.time()
        code = proc.wait()
    stop.set()
    outcome = {"status": "complete" if code == 0 else "failed", "exit_code": code, "seconds": time.time() - started,
               "gpu": gpu, "gpu_count": GPUS[gpu]["n_gpus"], "attempt": run["attempt"], "resumed": run["resume"]}
    if code != 0:
        raise RuntimeError(f"verl exited {code} (attempt {run['attempt']})")
    json.dump(outcome, open(directory / "outcome.json", "w"), indent=1)
    vol.commit()
    return outcome


@app.function(gpu="H200:2", **{**TRAIN_KW, "memory": 163840})
def train_coin(run_id, cell, seed, cycles=200, **kw):
    return train_coin_impl(run_id, cell, seed, cycles, gpu="H200:2", **kw)


def train_coin_impl(run_id, cell, seed, cycles, gpu, eval_every=10, save_every=25, dev_n=1, groups=16, n=16, lr=1e-6,
                    response_tokens=COIN_RESPONSE_TOKENS):
    import pandas as pd
    from huggingface_hub import snapshot_download

    from penalty_shape.coin import make_tasks, verl_rows

    directory, attempt, resume = start_run(run_id)
    pd.DataFrame(verl_rows(make_tasks(4096, seed=1000))).to_parquet(directory / "train.parquet")
    pd.DataFrame(verl_rows(make_tasks(64, seed=1), validation=True)).to_parquet(directory / "dev.parquet")
    run = {"run_id": run_id, "dir": str(directory), "model_path": snapshot_download(MODEL), "env": "coin", "cell": cell,
           "penalty": COIN_CELLS[cell], "seed": seed, "cycles": cycles, "response_tokens": response_tokens,
           "eval_every": eval_every, "save_every": save_every, "dev_n": dev_n, "groups": groups, "n": n, "lr": lr,
           **GPUS[gpu], "token_len_factor": 8.0, "resume": resume, "attempt": attempt, "gpu": gpu,
           "extra_overrides": {
               "reward.custom_reward_function.path": "/root/penalty_shape/coin_reward.py",
               "algorithm.norm_adv_by_std_in_grpo": True,
               "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
               "actor_rollout_ref.actor.loss_scale_factor": None,
               "actor_rollout_ref.rollout.temperature": COIN_TEMPERATURE,
               "actor_rollout_ref.rollout.val_kwargs.temperature": COIN_TEMPERATURE,
               "+data.apply_chat_template_kwargs.enable_thinking": False}}
    return run_verl(run, directory, gpu, "sequence")


@app.function(image=train_image, timeout=600)
def image_check():
    import importlib
    import importlib.metadata as m
    out = {}
    for pkg in ("torch", "transformers", "accelerate", "vllm", "numpy", "pandas", "pyarrow", "datasets", "anthropic", "verl", "ray"):
        try:
            out[pkg] = m.version(pkg)
        except Exception as e:
            out[pkg] = repr(e)
    for mod in ("numpy", "pandas", "pyarrow", "datasets", "anthropic", "penalty_shape.reward", "penalty_shape.trainer", "penalty_shape.coin_reward"):
        try:
            importlib.import_module(mod)
            out["import " + mod] = "ok"
        except Exception as e:
            out["import " + mod] = repr(e)[:400]
    return out


@app.function(image=verify_image, timeout=86400, volumes={"/vol": vol})
def pilot_sequence(runs, status_file="pilot_status.json"):
    status_path = Path("/vol") / status_file
    status = json.load(open(status_path)) if status_path.exists() else {}
    for r in runs:
        if status.get(r["run_id"], {}).get("status") == "complete":
            continue
        status[r["run_id"]] = {"status": "running", "started": time.time(), "config": r}
        json.dump(status, open(status_path, "w"), indent=1)
        vol.commit()
        fn = train_coin if r.get("env") == "coin" else train_h100 if r.get("gpu", "H200:2") == "H100:4" else train
        args = {k: v for k, v in r.items() if k not in ("gpu", "env")}
        try:
            outcome = fn.remote(**args)
        except Exception as e:
            outcome = {"status": "failed", "error": repr(e)}
        status[r["run_id"]] = {**outcome, "finished": time.time(), "config": r}
        json.dump(status, open(status_path, "w"), indent=1)
        vol.commit()
    return status
