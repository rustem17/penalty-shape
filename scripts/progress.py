"""Print pilot status and the latest training measurements for a run: progress.py [RUN_ID]"""
import json


def loads(line):
    try:
        return json.loads(line) if line.strip() else None
    except json.JSONDecodeError:
        return None
import sys

import modal

vol = modal.Volume.from_name("penalty-shape")


def read(path):
    return b"".join(vol.read_file(path)).decode()


status = {}
for e in vol.listdir("/"):
    if e.path.split("/")[-1].startswith("pilot_status"):
        status.update(json.loads(read(e.path.split("/")[-1])))
print("status:", {k: v["status"] for k, v in status.items()})
runs = [sys.argv[1]] if len(sys.argv) > 1 else [k for k, v in status.items() if v["status"] == "running"]
for run in runs:
    files = {e.path.split("/")[-1] for e in vol.listdir(f"runs/{run}")}
    print("\nrun:", run)
    if "measurements.jsonl" in files:
        rows = [r for r in (loads(l) for l in read(f"runs/{run}/measurements.jsonl").splitlines()) if r]
        ups = [r for r in rows if r.get("kind") == "training_update"]
        print("updates:", len(ups), "checkpoints:", sum(r.get("kind") == "checkpoint" for r in rows))
        for r in ups[-8:]:
            print(r["step"], f"{r['elapsed_seconds']:.0f}s", {k: round(r[k], 3) for k in ("score", "label_T", "label_O", "label_H", "label_F", "label_U", "correct", "flagged", "truncated", "zero_variance_groups", "response_tokens_mean") if k in r})
    if "scored.jsonl" in files:
        rows = [r for r in (loads(l) for l in read(f"runs/{run}/scored.jsonl").splitlines()) if r]
        val = [r for r in rows if r["validation"]]
        import collections
        for step in sorted({r["step"] for r in val}):
            labs = collections.Counter(r["label"] for r in val if r["step"] == step)
            print("val step", step, dict(labs), "flagged", sum(r["flagged"] for r in val if r["step"] == step))
