"""verl custom reward for Heads-And-Tails cells: compute_score(data_source, solution_str, ground_truth, extra_info).

score = R_out + R_cot. R_out = 1 if the final answer is correct. R_cot by run["penalty"] (coin.py signals):
present -[heads/tails in reasoning]; lines -lines/COIN_LINES_MEAN; length -max(0, chars - COIN_L_MIN)/(COIN_LENGTH_MEAN - COIN_L_MIN);
flag -[their coin monitor >= FLAG_THRESHOLD]; count -count/COIN_COUNT_MEAN.
Monitors run only when the penalty needs them, and on every validation answer (rating and lines).
Not computed = -1. Monitor failure after retries: R_cot = 0, unknown = 1.
Run config from $PENALTY_RUN_CONFIG: {dir, model_path, penalty, ...}.
"""
import asyncio
import json
import os
import time
from pathlib import Path

from penalty_shape.coin import parse, rating, signals, tracking_lines
from penalty_shape.config import COIN_COUNT_MEAN, COIN_L_MIN, COIN_LENGTH_MEAN, COIN_LINES_MEAN, FLAG_THRESHOLD

_state = None


def state():
    global _state
    if _state is None:
        import anthropic
        _state = (json.load(open(os.environ["PENALTY_RUN_CONFIG"])), anthropic.AsyncAnthropic(max_retries=8))
    return _state


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


NEEDS = {"flag": "rating", "lines": "lines"}
MONITORS = {"rating": rating, "lines": tracking_lines}


def penalty(kind, sig):
    return {"present": sig["present"], "count": sig["count"] / COIN_COUNT_MEAN, "lines": sig["lines"] / COIN_LINES_MEAN,
            "length": max(0, sig["length"] - COIN_L_MIN) / (COIN_LENGTH_MEAN - COIN_L_MIN), "flag": int(sig["rating"] >= FLAG_THRESHOLD)}[kind]


async def score_sample(run, client, text, final, validation):
    p = parse(text)
    sig = {**signals(p["reasoning"]), "rating": -1, "lines": -1}
    names = [k for k in MONITORS if validation or NEEDS.get(run["penalty"]) == k]
    results = await asyncio.gather(*[MONITORS[k](client, p["reasoning"]) for k in names], return_exceptions=True)
    errors = {k: repr(r) for k, r in zip(names, results) if isinstance(r, BaseException)}
    sig.update({k: r[0] for k, r in zip(names, results) if k not in errors})
    usage = [r[1] for k, r in zip(names, results) if k not in errors]
    unknown = int(NEEDS.get(run["penalty"]) in errors)
    r_out = float(p["answer"] == final.lower())
    r_cot = 0.0 if unknown else -float(penalty(run["penalty"], sig))
    flag = int(sig["rating"] >= FLAG_THRESHOLD) if sig["rating"] >= 0 else -1
    return {"r_out": r_out, "r_cot": r_cot, "score": r_out + r_cot, "unknown": unknown, "flag": flag, **sig}, p, errors, usage


async def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    run, client = state()
    validation = bool(extra_info.get("validation", False))
    row, p, errors, usage = await score_sample(run, client, solution_str, ground_truth, validation)
    append_jsonl(Path(run["dir"]) / "scored.jsonl", {
        **row, "step": int(extra_info.get("step", -1)), "validation": validation, "index": int(extra_info["index"]),
        "task_id": extra_info["task_id"], "attempt": run.get("attempt", 1), "finish_reason": extra_info["finish_reason"],
        "response_tokens": int(extra_info["generated_tokens"]), "answer": p["answer"], "completion": solution_str,
        "errors": errors, "usage": usage, "time": time.time()})
    return {k: float(v) for k, v in row.items()}
