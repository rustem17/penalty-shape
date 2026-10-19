"""Patches the pinned verl checkout: swap in PenaltyTrainer and expose response token ids,
finish_reason and generated_tokens to the reward function (decoded with special tokens kept)."""
import sys
from pathlib import Path


def install(root):
    root = Path(root)
    entry = root / "verl/trainer/main_ppo_v0.py"
    original = "from verl.trainer.ppo.ray_trainer import RayPPOTrainer"
    replacement = "from penalty_shape.trainer import PenaltyTrainer as RayPPOTrainer"
    text = entry.read_text()
    if replacement not in text:
        assert text.count(original) == 1, "pinned trainer entrypoint changed"
        entry.write_text(text.replace(original, replacement))
    manager = root / "verl/experimental/reward_loop/reward_manager/naive.py"
    text = manager.read_text()
    if 'extra_info["finish_reason"]' not in text:
        start = '        extra_info = data_item.non_tensor_batch.get("extra_info", {})'
        replacement = '''        extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}))
        token_ids = valid_response_ids.tolist()
        eos = self.tokenizer.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        ended = bool(token_ids and token_ids[-1] in eos_ids)
        if ended:
            valid_response_ids = valid_response_ids[:-1]
            token_ids = token_ids[:-1]
        extra_info["generated_tokens"] = len(token_ids)
        extra_info["finish_reason"] = "stop" if ended else "length"
        extra_info["response_token_ids"] = token_ids'''
        decode = "self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)"
        assert text.count(start) == 1 and text.count(decode) == 1, "pinned reward manager changed"
        manager.write_text(text.replace(start, replacement).replace(decode, "self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)"))


if __name__ == "__main__":
    install(sys.argv[1])
