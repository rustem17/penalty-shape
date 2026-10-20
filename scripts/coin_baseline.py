"""Phase 0 + phase 1 readout: score base Qwen3-8B Heads-And-Tails samples (Modal volume coin_baseline/RUN_ID).

Input: samples from modal_app.coin_baseline (64 prompts, temperature 1.0, with reasoning and reasoning forced empty).
Scores every sample: correctness, heads/tails present and count, reasoning characters, their 0-5 coin monitor
(flag = rating >= 3) and the graded monitor's tracking-line count (both claude-haiku-5-5, effort low).
All-same share and within-prompt std / mean use the first 16 samples per prompt (training group size).
Gate B: the graded monitor rescored on 100 random reasonings (seed 0), Pearson correlation of the two passes.
usage: coin_baseline.py RUN_ID   -> results/coin_baseline_<ts>/{scored.jsonl, summary.json}
"""
import asyncio
import collections
import json
import random
import sys
import time
from pathlib import Path

import anthropic
import modal
import numpy as np
from dotenv import load_dotenv

from penalty_shape.coin import parse, rating, signals, tracking_lines

load_dotenv(".env")
GROUP = 16
run_id = sys.argv[1]
rows = [json.loads(l) for l in b"".join(modal.Volume.from_name("penalty-shape").read_file(f"coin_baseline/{run_id}/samples.jsonl")).decode().splitlines()]
client = anthropic.AsyncAnthropic(max_retries=8)
sem = asyncio.Semaphore(32)
usage = collections.Counter()


async def call(fn, text):
    async with sem:
        value, u = await fn(client, text)
    usage.update(u)
    return value


async def score_all():
    for r in rows:
        p = parse(r["completion"])
        r.update(p, **signals(p["reasoning"]), correct=int(p["answer"] == r["task"]["final"].lower()),
                 has_tags=int("<reasoning>" in r["completion"] and "</reasoning>" in r["completion"]))
    reasoning = [r for r in rows if r["mode"] == "reasoning"]
    ratings, lines = await asyncio.gather(asyncio.gather(*[call(rating, r["reasoning"]) for r in reasoning]),
                                          asyncio.gather(*[call(tracking_lines, r["reasoning"]) for r in reasoning]))
    for r, a, b in zip(reasoning, ratings, lines):
        r.update(rating=a, flag=int(a >= 3), lines=b)
    retest = random.Random(0).sample(reasoning, 100)
    second = await asyncio.gather(*[call(tracking_lines, r["reasoning"]) for r in retest])
    return reasoning, retest, second


def spread(rs, key):
    by_task = collections.defaultdict(list)
    for r in rs:
        if r["sample"] < GROUP:
            by_task[r["task"]["task_id"]].append(r[key])
    groups = [np.array(v, dtype=float) for v in by_task.values()]
    cv = [g.std() / g.mean() for g in groups if g.mean() > 0]
    vals = np.concatenate(groups)
    return {"mean": float(vals.mean()), "p10": float(np.percentile(vals, 10)), "p50": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90)), "all_same_share": float(np.mean([len(set(g)) == 1 for g in groups])),
            "within_prompt_std_over_mean": float(np.mean(cv)) if cv else None}


reasoning, retest, second = asyncio.run(score_all())
empty = [r for r in rows if r["mode"] == "empty"]
first = [r["lines"] for r in retest]
correct_lengths = sorted(r["length"] for r in reasoning if r["correct"])
summary = {
    "run_id": run_id, "n_reasoning": len(reasoning), "n_empty": len(empty),
    "accuracy_reasoning": float(np.mean([r["correct"] for r in reasoning])),
    "accuracy_reasoning_first16": float(np.mean([r["correct"] for r in reasoning if r["sample"] < GROUP])),
    "accuracy_empty": float(np.mean([r["correct"] for r in empty])),
    "empty_answer_parsed": float(np.mean([r["answer"] in ("heads", "tails") for r in empty])),
    "has_reasoning_tags": float(np.mean([r["has_tags"] for r in reasoning])),
    "answer_parsed": float(np.mean([r["answer"] is not None for r in reasoning])),
    "truncated": float(np.mean([r["finish_reason"] == "length" for r in reasoning])),
    "tokens": {q: float(np.percentile([r["tokens"] for r in reasoning], q)) for q in (50, 90, 99, 100)},
    "shortest_correct_reasoning_chars": correct_lengths[:5],
    "penalties": {k: spread(reasoning, k) for k in ("present", "count", "rating", "flag", "length", "lines")},
    "rating_hist": dict(sorted(collections.Counter(r["rating"] for r in reasoning).items())),
    "lines_hist": dict(sorted(collections.Counter(r["lines"] for r in reasoning).items())),
    "gate_b_retest_pearson": float(np.corrcoef(first, second)[0, 1]),
    "gate_b_retest_exact_share": float(np.mean([a == b for a, b in zip(first, second)])),
    "api_usage": dict(usage), "api_cost": (usage["input"] * 0.10 + usage["output"] * 0.50) / 1e6,
}
out = Path(f"results/coin_baseline_{time.strftime('%Y%m%d_%H%M%S')}")
out.mkdir(parents=True)
with open(out / "scored.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
json.dump(summary, open(out / "summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
print(out)
