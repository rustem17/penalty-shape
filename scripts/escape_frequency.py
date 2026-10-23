"""Escape frequencies of candidate reasoning penalties at the start of training, measured without training.

Base model: Qwen/Qwen3-8B, their coin prompt, thinking off, temperature 1.0; the 2,048 saved answers (64 prompts x 32)
in results/coin_baseline_20261020_024112/scored.jsonl (scored by scripts/coin_baseline.py). Group = the first 16
answers per prompt (the training group size).
Part 1, word bans: for each pattern, the share of answers with no match (they would pay zero under "-1 if present") and
the share of prompt groups whose 16 answers all get the same penalty. Thresholds: share of answers with at most k side
names (heads/tails count) and at most k side-tracking lines (cell B's Haiku counter).
Part 2, LLM counters at the start: share of training answers at steps 1-5 with 0 state lines (cell H, runs h_H_s0..2)
and 0 side lines (cell B, runs s1_B_s0 and s2_B_s1), from results/coin_runs/RUN/scored.jsonl.
Part 3, alternatives where the base model wrote heads/tails, coin or flip: modal_app.coin_logprobs (vLLM prompt
log-probs, top 20) over the same 2,048 answers, reasoning region only. The banned words span several tokens in some
forms (" Tails" = " T" + "ails", "Heads" = "He" + "ads"), so the forms are taken from the data: the first tokens the
model itself wrote at these positions (seen 5 times or more). Per position: probability of the written token and the
mass on top-20 tokens that are not one of those forms (escape mass; a bare letter " T" therefore counts as a form, not
an escape, and an alternative such as " Still" may only postpone the word). Per answer: the product of the escape mass
over its positions, a rough whole-answer escape probability under independence.
usage: escape_frequency.py LOGPROB_RUN_ID  -> results/escape_<ts>/{summary.md, summary.json, positions.jsonl}
"""
import collections
import json
import re
import sys
import time
from pathlib import Path

import modal
import numpy as np

BASE = "results/coin_baseline_20261020_024112/scored.jsonl"
GROUP = 16
PATTERNS = ["heads|tails", "coin", "flip", "same|different", "start", "player", "final", "land", "result", "side",
            "remain|stay", "state", "change"]
EARLY = {"state_lines": ["h_H_s0", "h_H_s1", "h_H_s2"], "lines": ["s1_B_s0", "s2_B_s1"]}
TARGETS = ("side", "coin", "flip")

rows = [r for r in map(json.loads, open(BASE)) if r["mode"] == "reasoning"]
first = [r for r in rows if r["sample"] < GROUP]


def all_same(value):
    by = collections.defaultdict(list)
    for r in first:
        by[r["task"]["task_id"]].append(value(r))
    return float(np.mean([len(set(v)) == 1 for v in by.values()]))


bans = [{"pattern": p, "zero_payers": float(np.mean([not re.search(p, r["reasoning"], re.I) for r in rows])),
         "groups_all_same": all_same(lambda r, p=p: bool(re.search(p, r["reasoning"], re.I)))} for p in PATTERNS]
thresholds = {key: [{"at_most": k, "share": float(np.mean([r[key] <= k for r in rows])),
                     "groups_all_same": all_same(lambda r, k=k, key=key: r[key] <= k)} for k in range(12)]
              for key in ("count", "lines")}
early = {}
for key, runs in EARLY.items():
    rs = [r for run in runs for r in map(json.loads, open(f"results/coin_runs/{run}/scored.jsonl"))
          if not r["validation"] and 1 <= r["step"] <= 5]
    early[key] = {"runs": runs, "n": len(rs), "zero_share": float(np.mean([r[key] == 0 for r in rs])),
                  "paid_zero_share": float(np.mean([r["r_cot"] == 0 for r in rs])), "mean": float(np.mean([r[key] for r in rs]))}

run_id = sys.argv[1]
out = Path(f"results/escape_{time.strftime('%Y%m%d_%H%M%S')}")
out.mkdir()
data = b"".join(modal.Volume.from_name("penalty-shape").read_file(f"coin_logprobs/{run_id}/positions.jsonl"))
(out / "positions.jsonl").write_bytes(data)
positions = [json.loads(l) for l in data.decode().splitlines()]


written = {t: collections.Counter(p["actual"]["token"] for p in positions if p["target"] == t) for t in TARGETS}
forms = {t: {tok for tok, n in c.items() if n >= 5} for t, c in written.items()}


def escape_mass(p):
    return sum(t["p"] for t in p["top"] if t["token"] not in forms[p["target"]])


def stats(x):
    return {"median": float(np.median(x)), "mean": float(np.mean(x)), "p90": float(np.percentile(x, 90))}


logprob = {}
for target in TARGETS:
    ps = [p for p in positions if p["target"] == target]
    esc = np.array([escape_mass(p) for p in ps])
    per_answer = collections.defaultdict(list)
    for p, e in zip(ps, esc):
        per_answer[(p["task_id"], p["sample"])].append(e)
    products = np.array([np.prod(v) for v in per_answer.values()])
    alternatives = collections.Counter(t["token"] for p in ps for t in p["top"][:3] if t["token"] not in forms[target] and t["p"] >= 0.01)
    logprob[target] = {"positions": len(ps), "answers": len(per_answer), "positions_per_answer": len(ps) / len(per_answer),
                       "forms_written": written[target].most_common(),
                       "p_written": stats([p["actual"]["p"] for p in ps]),
                       "top_mass": stats([sum(t["p"] for t in p["top"]) for p in ps]),
                       "escape_mass": {**stats(esc), "share_over_0.1": float(np.mean(esc > 0.1)), "share_over_0.5": float(np.mean(esc > 0.5))},
                       "answer_escape_product": {**stats(products), "share_over_1e-4": float(np.mean(products > 1e-4)),
                                                 "share_over_1e-2": float(np.mean(products > 1e-2))},
                       "common_alternatives": alternatives.most_common(15)}

summary = {"logprob_run": run_id, "n_base_answers": len(rows), "bans": bans, "thresholds": thresholds, "early_counters": early, "logprob": logprob}
json.dump(summary, open(out / "summary.json", "w"), indent=1)
lines = ["# Escape frequencies at the start (base Qwen3-8B, their coin prompt)", "",
         f"Base answers: {len(rows)} (64 prompts x 32); groups = first 16 per prompt. Log-prob run {run_id}.", "",
         "## Word bans: share of answers that would pay zero under -1 if the pattern appears", "",
         "| pattern | zero-payers | groups all-same |", "|---|---|---|"]
lines += [f"| {b['pattern']} | {b['zero_payers']:.4f} | {b['groups_all_same']:.2f} |" for b in bans]
for key, name in (("count", "side names (heads/tails count)"), ("lines", "side-tracking lines (cell B counter)")):
    lines += ["", f"## Threshold penalties: share of answers with at most k {name}", "", "| k | share | groups all-same |", "|---|---|---|"]
    lines += [f"| {t['at_most']} | {t['share']:.3f} | {t['groups_all_same']:.2f} |" for t in thresholds[key]]
lines += ["", "## LLM counters at training steps 1-5", "", "| counter | runs | answers | count 0 | paid 0 | mean |", "|---|---|---|---|---|---|"]
lines += [f"| {k} | {', '.join(v['runs'])} | {v['n']} | {v['zero_share']:.4f} | {v['paid_zero_share']:.4f} | {v['mean']:.2f} |" for k, v in early.items()]
lines += ["", "## Alternatives at the positions where the model wrote the banned word (top 20, reasoning region)", "",
          "| target | positions | per answer | p(written) median | escape mass median / mean / p90 | positions with escape mass > 0.1 | answer product median | answers with product > 1e-4 |",
          "|---|---|---|---|---|---|---|---|"]
for t, v in logprob.items():
    e, a = v["escape_mass"], v["answer_escape_product"]
    lines.append(f"| {t} | {v['positions']} | {v['positions_per_answer']:.1f} | {v['p_written']['median']:.3f} | "
                 f"{e['median']:.4f} / {e['mean']:.4f} / {e['p90']:.4f} | {e['share_over_0.1']:.3f} | {a['median']:.2e} | {a['share_over_1e-4']:.3f} |")
for t, v in logprob.items():
    lines += ["", f"Forms written at {t} positions: " + ", ".join(f"{json.dumps(tok)} x{n}" for tok, n in v["forms_written"]),
              "", f"Most common alternatives at {t} positions (top 3, probability at least 0.01): " +
              ", ".join(f"{json.dumps(tok)} x{n}" for tok, n in v["common_alternatives"])]
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(out)
