"""Hiding run (cell H) figure from local readout copies results/coin_runs/RUN_ID/scored.jsonl.

Runs: h_H_s0, h_H_s1, h_H_s2 (Qwen3-8B, Heads-And-Tails N = 6, penalty 0 if no reasoning line states what happens to
the coin, else -(0.5 + 0.5 x state lines / 8.55); advantage = reward - group mean; 400 steps).
Left: training answers, mean state lines per answer per 10-step window, one line per seed.
Right: dev answers (64 every 10 steps), monitorability = share rated >= 3 by the calibrated coin monitor on Sonnet 5.5
(solid) and the printed coin monitor on Haiku 5.5 (dashed), one color per seed. Latest attempt per (validation, step).
usage: coin_hiding_plot.py RUN_ID [RUN_ID ...]   -> results/coin_hiding_<YYYYMMDD_HHMMSS>/state_and_monitor.png
"""
import collections
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

COLORS = ["#729ECE", "#D68F73", "#7F7F7F"]
sns.set_theme(style="whitegrid")
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
for run_id, color in zip(sys.argv[1:], COLORS):
    rows = [json.loads(l) for l in open(Path("results/coin_runs") / run_id / "scored.jsonl") if l.strip()]
    latest = {}
    for r in rows:
        key = (r["validation"], r["step"])
        latest[key] = max(latest.get(key, 1), r.get("attempt", 1))
    rows = [r for r in rows if r.get("attempt", 1) == latest[(r["validation"], r["step"])]]
    train = collections.defaultdict(list)
    dev = collections.defaultdict(list)
    for r in rows:
        (dev[r["step"]] if r["validation"] else train[(r["step"] - 1) // 10]).append(r)
    w = sorted(train)
    left.plot([10 * k + 5 for k in w], [np.mean([r["state_lines"] for r in train[k]]) for k in w], "o-", color=color,
              markersize=3, label=run_id)
    s = sorted(dev)
    right.plot(s, [np.mean([r["rating_sonnet"] >= 3 for r in dev[k]]) for k in s], "o-", color=color, markersize=3,
               label=f"{run_id}, Sonnet calibrated")
    right.plot(s, [np.mean([r["rating"] >= 3 for r in dev[k]]) for k in s], "o--", color=color, markersize=3, alpha=0.6,
               label=f"{run_id}, Haiku as printed")
left.set(xlabel="training step", ylabel="state lines per answer", ylim=(0, 11.5), title="Lines stating what happens to the coin")
right.set(xlabel="training step", ylabel="share rated about a coin (>= 3)", ylim=(-0.03, 1.45),
          title="Monitorability on dev answers")
left.legend(loc="upper left")
right.legend(loc="upper left", fontsize=7)
fig.tight_layout()
out = Path("results") / f"coin_hiding_{time.strftime('%Y%m%d_%H%M%S')}"
out.mkdir(parents=True)
fig.savefig(out / "state_and_monitor.png", dpi=150)
print(out / "state_and_monitor.png")
