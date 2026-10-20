"""Final write-up data for the Kaufmann 2x2: per-cell tables and plots from coin_readout.

Runs: RUN_IDs named PREFIX_CELL_sSEED. Per 10-step window of training answers: CoT reward of the cell's penalty,
accuracy, share of 16-answer groups with zero CoT-reward variance, reasoning characters. Dev (64 answers every
10 steps): monitorability = share rated >= 3 by their coin monitor, mean rating. curves.png: mean over seeds, band =
range across seeds; seeds.png: CoT reward and accuracy per seed.
Colors: flat #729ECE, graded #D68F73; orthogonal solid, in-conflict dashed (cell C dotted).
Per run (runs.md): first 10-step window whose CoT reward is >= 0.2 above the first window ("moved"), last-window
values, dev monitorability (last, minimum), first dev step with Kaufmann success, and dev steps with monitorability
< 0.5 while accuracy > 0.8 (reasoning hidden but still working).
usage: coin_writeup.py RUN_ID [RUN_ID ...]   -> results/coin_writeup_<ts>/{cells.json, curves.png, seeds.png, runs.md, runs.json}
"""
import collections
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from coin_readout import readout

STYLE = {"A": ("#729ECE", "-", "A orthogonal flat (heads/tails present)"),
         "E": ("#D68F73", "-", "E orthogonal graded (heads/tails count)"),
         "D": ("#729ECE", "--", "D in-conflict flat (their monitor >= 3)"),
         "B": ("#D68F73", "--", "B in-conflict graded (tracking lines)"),
         "C": ("#D68F73", ":", "C in-conflict graded (length)")}
PANELS = [("train", "cot_reward", "CoT reward (training answers)"), ("train", "accuracy", "Accuracy (training answers)"),
          ("train", "zero_var_cot", "Share of groups with zero CoT-reward variance"),
          ("dev", "monitorable", "Monitorability: share rated >= 3 (dev)"),
          ("train", "chars", "Reasoning characters (training answers)"), ("dev", "rating", "Mean coin-monitor rating 0-5 (dev)")]

runs = collections.defaultdict(list)
for run_id in sys.argv[1:]:
    r = readout(run_id)
    runs[r["run"]["cell"]].append(r)
out = Path(f"results/coin_writeup_{time.strftime('%Y%m%d_%H%M%S')}")
out.mkdir(parents=True)
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
summary = {}
for cell in "AEDBC":
    if cell not in runs:
        continue
    color, ls, label = STYLE[cell]
    summary[cell] = {"seeds": [r["run"]["seed"] for r in runs[cell]], "train": {}, "dev": {}}
    for ax, (kind, key, title) in zip(axes.flat, PANELS):
        x_key = "steps" if kind == "train" else "step"
        per_seed = [{(int(row[x_key].split("-")[1]) if kind == "train" else row[x_key]): row[key] for row in r[kind]} for r in runs[cell]]
        xs = sorted(set.intersection(*[set(p) for p in per_seed]))
        ys = np.array([[p[x] for x in xs] for p in per_seed], dtype=float)
        mean = ys.mean(0)
        ax.plot(xs, mean, ls, color=color, marker="o", markersize=3, label=label)
        ax.fill_between(xs, ys.min(0), ys.max(0), color=color, alpha=0.2)
        ax.set_title(title, loc="center")
        ax.set_xlabel("training step")
        summary[cell][kind][key] = {"x": xs, "mean": mean.tolist(), "per_seed": ys.tolist()}
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", ncol=5, fontsize=9, bbox_to_anchor=(0.01, 0.955), frameon=False)
fig.suptitle("Penalty shape x reward class: mean over seeds, band = range across seeds", ha="center")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out / "curves.png", dpi=130)
fig2, axes2 = plt.subplots(2, 5, figsize=(20, 7), sharex=True)
for j, cell in enumerate("AEDBC"):
    color, ls, label = STYLE[cell]
    for i, (key, name) in enumerate((("cot_reward", "CoT reward"), ("accuracy", "accuracy"))):
        ax = axes2[i, j]
        for r in sorted(runs.get(cell, []), key=lambda r: r["run"]["seed"]):
            xs = [int(row["steps"].split("-")[1]) for row in r["train"]]
            ax.plot(xs, [row[key] for row in r["train"]], ls, color=color, marker="o", markersize=3,
                    alpha=0.5 + 0.25 * r["run"]["seed"], label=f"seed {r['run']['seed']}")
        ax.set_ylim(-1.05, 0.05) if key == "cot_reward" else ax.set_ylim(0.35, 1.02)
        ax.set_title(f"{label}\n{name}" if i == 0 else name, loc="center", fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
    axes2[1, j].set_xlabel("training step")
fig2.suptitle("Each seed separately (training answers, 10-step windows)", ha="center")
fig2.tight_layout()
fig2.savefig(out / "seeds.png", dpi=130)
rows = []
for cell in "AEDBC":
    for r in sorted(runs.get(cell, []), key=lambda r: r["run"]["seed"]):
        tr, dev = r["train"], r["dev"]
        moved = next((row["steps"] for row in tr if row["cot_reward"] >= tr[0]["cot_reward"] + 0.2), None)
        success = next((row["step"] for row in dev if row["kaufmann_success"]), None)
        hidden = [row["step"] for row in dev if row["monitorable"] < 0.5 and row["accuracy"] > 0.8]
        rows.append({"cell": cell, "seed": r["run"]["seed"], "attempts": max(x["attempt"] for x in r["rows"]),
                     "moved_window": moved, "cot_first": tr[0]["cot_reward"], "cot_last": tr[-1]["cot_reward"],
                     "acc_last": tr[-1]["accuracy"], "zero_var_last": tr[-1]["zero_var_cot"], "chars_last": tr[-1]["chars"],
                     "dev_acc_last": dev[-1]["accuracy"], "dev_monitorable_last": dev[-1]["monitorable"],
                     "dev_monitorable_min": min(row["monitorable"] for row in dev), "dev_rating_last": dev[-1]["rating"],
                     "dev_present_last": dev[-1]["present"], "first_success_step": success, "hidden_steps": hidden})
keys = list(rows[0])
table = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
table += ["| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row.values()) + " |" for row in rows]
(out / "runs.md").write_text("\n".join(table) + "\n")
json.dump(rows, open(out / "runs.json", "w"), indent=1)
print("\n".join(table))
json.dump(summary, open(out / "cells.json", "w"), indent=1)
print(out)
