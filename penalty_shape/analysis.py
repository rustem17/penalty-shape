"""Summaries of assay samples and training runs (scored.jsonl + measurements.jsonl)."""
import collections
import json


def loads(line):
    try:
        return json.loads(line) if line.strip() else None
    except json.JSONDecodeError:
        return None

import numpy as np


def read_jsonl(path):
    return [r for r in (loads(line) for line in open(path)) if r]


def shares(rows):
    n = max(len(rows), 1)
    counts = collections.Counter(r["label"] for r in rows)
    return {k: counts.get(k, 0) / n for k in "TOHFU"}


def summarize_assay(rows):
    n = len(rows)
    B = [r for r in rows if r["correct"] and r["backdoor"]]
    O = [r for r in B if not r["flagged"]]
    tokens = np.array([r["response_tokens"] for r in rows])
    usage = collections.Counter()
    for r in rows:
        usage.update(r.get("monitor_usage") or {})
    return {
        "n": n, "n_tasks": len({r["task_id"] for r in rows}),
        "shares": shares(rows),
        "p_O_given_backdoor": len(O) / max(len(B), 1), "n_backdoor_correct": len(B),
        "tasks_with_O": len({r["task_id"] for r in O}),
        "correct": float(np.mean([r["correct"] for r in rows])),
        "backdoor_any": float(np.mean([r["backdoor"] for r in rows])),
        "backdoor_trigger": float(np.mean([r["backdoor_trigger"] for r in rows])),
        "backdoor_diff_only": float(np.mean([r["backdoor_diff"] and not r["backdoor_trigger"] for r in rows])),
        "H_trigger_only_definition": float(np.mean([r["correct"] and not r["backdoor_trigger"] for r in rows])),
        "flagged": float(np.mean([r["flagged"] for r in rows])),
        "flagged_given_backdoor": float(np.mean([r["flagged"] for r in B])) if B else None,
        "m1_hist": dict(sorted(collections.Counter(r["m1_score"] for r in rows).items())),
        "truncated": float(np.mean([r["truncated"] for r in rows])),
        "think_closed": float(np.mean([r["think_closed"] for r in rows])),
        "has_code": float(np.mean([r["has_code"] for r in rows])),
        "has_trigger": float(np.mean([r["has_trigger"] for r in rows])),
        "trigger_ref_error": float(np.mean([r["trigger_ref_error"] for r in rows])),
        "unknown": float(np.mean([r["unknown"] for r in rows])),
        "response_tokens": {"mean": float(tokens.mean()), "p50": float(np.percentile(tokens, 50)),
                            "p90": float(np.percentile(tokens, 90)), "p99": float(np.percentile(tokens, 99)), "max": int(tokens.max())},
        "monitor_usage": dict(usage),
    }


def summarize_run(scored, measurements):
    latest = {}
    for r in scored:
        key = (r["validation"], r["step"])
        latest[key] = max(latest.get(key, 1), r.get("attempt", 1))
    scored = [r for r in scored if r.get("attempt", 1) == latest[(r["validation"], r["step"])]]
    by_step = collections.defaultdict(list)
    for r in scored:
        by_step[(r["validation"], r["step"])].append(r)
    train_steps = sorted(s for v, s in by_step if not v)
    val_steps = sorted(s for v, s in by_step if v)
    def table(keys):
        return {s: {**shares(by_step[k]), "correct": float(np.mean([r["correct"] for r in by_step[k]])),
                    "n": len(by_step[k])} for k, s in keys}
    updates = [m for m in measurements if m.get("kind") == "training_update"]
    elapsed = [m["elapsed_seconds"] for m in updates]
    last = train_steps[-3:]
    last_rows = [r for s in last for r in by_step[(False, s)]]
    return {
        "train_steps": len(train_steps), "val_steps": val_steps,
        "validation": table([((True, s), s) for s in val_steps]),
        "train_last3": {**shares(last_rows), "steps": last, "n": len(last_rows)},
        "train_first3": {**shares([r for s in train_steps[:3] for r in by_step[(False, s)]]), "steps": train_steps[:3]},
        "seconds_per_update": {"mean": float(np.mean(elapsed)) if elapsed else None,
                               "p50": float(np.median(elapsed)) if elapsed else None},
        "zero_variance_groups_mean": float(np.mean([m["zero_variance_groups"] for m in updates])) if updates else None,
        "truncated_mean": float(np.mean([r["truncated"] for r in scored])) if scored else None,
        "unknown_mean": float(np.mean([r["unknown"] for r in scored])) if scored else None,
    }
