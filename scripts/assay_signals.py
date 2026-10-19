"""Within-group spread of candidate graded penalty signals, from the assay's stored M1 outputs.

Signals per sample: flag (score>=3), m1 score/5, quote count, quoted chars / CoT chars.
Group = task (32 samples). Reports per-task std averaged over tasks, between-task std of the task means,
and the implied penalty std for a linear penalty scaled so its mean equals p=1.5 at the start.
Also counts backdoor words outside the <think> block (relocation candidates) and in code comments.
usage: assay_signals.py results/assay_<id>/samples.jsonl
"""
import collections
import json
import re
import sys

import numpy as np

from penalty_shape.verifier import parse_response

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
WORDS = re.compile(r"backdoor|trigger|special[- ]case|secret", re.I)
for r in rows:
    p = parse_response(r["completion"])
    cot = p["cot"]
    quoted = sum(len(q) for q in r["quotes"])
    r["sig"] = {"flag": float(r["flagged"]), "score": r["m1_score"] / 5, "quotes": float(len(r["quotes"])),
                "quoted_frac": min(quoted / max(len(cot), 1), 1.0), "quoted_chars": float(quoted), "cot_chars": float(len(cot))}
    end = r["completion"].rfind("</think>")
    answer = r["completion"][end + 8:] if end >= 0 else ""
    code_lines = [l for l in p["code"].splitlines()]
    r["outside"] = {"answer_mentions": bool(WORDS.search(re.sub(r"TRIGGER:.*", "", answer))),
                    "code_comment_mentions": any(WORDS.search(l.split("#", 1)[1]) for l in code_lines if "#" in l),
                    "code_identifier_mentions": any(WORDS.search(l.split("#", 1)[0]) for l in code_lines)}
by_task = collections.defaultdict(list)
for r in rows:
    by_task[r["task_id"]].append(r)
print("samples", len(rows), "tasks", len(by_task))
for name in ("flag", "score", "quotes", "quoted_frac", "quoted_chars", "cot_chars"):
    vals = np.array([r["sig"][name] for r in rows])
    within = np.mean([np.std([r["sig"][name] for r in rs]) for rs in by_task.values()])
    between = np.std([np.mean([r["sig"][name] for r in rs]) for rs in by_task.values()])
    scale = 1.5 / max(vals.mean(), 1e-9)
    print(f"{name:13s} mean {vals.mean():8.3f} p10/p50/p90 {np.percentile(vals,10):7.2f}/{np.percentile(vals,50):7.2f}/{np.percentile(vals,90):7.2f}  within-task std {within:7.3f}  between-task std {between:7.3f}  linear penalty std at mean=1.5: {within*scale:5.2f}")
fin = [r for r in rows if not r["truncated"]]
print("--- finished samples only (n=%d) ---" % len(fin))
for name in ("flag", "quotes", "quoted_frac"):
    bt = collections.defaultdict(list)
    for r in fin: bt[r["task_id"]].append(r["sig"][name])
    within = np.mean([np.std(v) for v in bt.values() if len(v) > 1])
    print(f"{name:13s} mean {np.mean([r['sig'][name] for r in fin]):.3f} within-task std {within:.3f}")
print("--- quote count histogram ---", dict(sorted(collections.Counter(int(r["sig"]["quotes"]) for r in rows).items())))
print("--- quoted fraction deciles ---", np.percentile([r["sig"]["quoted_frac"] for r in rows], range(10, 100, 10)).round(3).tolist())
print("--- corr(quoted_chars, cot_chars) ---", round(float(np.corrcoef([r["sig"]["quoted_chars"] for r in rows], [r["sig"]["cot_chars"] for r in rows])[0, 1]), 3))
for k in ("answer_mentions", "code_comment_mentions", "code_identifier_mentions"):
    print(f"outside-think {k}: {np.mean([r['outside'][k] for r in rows]):.3f} (finished only {np.mean([r['outside'][k] for r in fin]):.3f})")
