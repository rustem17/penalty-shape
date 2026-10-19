"""Spawn deployed Modal functions and record call ids in results/calls.jsonl.

usage: launch.py assay RUN_ID [--n-tasks 64 --n-samples 32 --max-tokens 4096]
       launch.py train RUN_ID --p 1.5 --credit sequence --seed 0 --cycles 2
       launch.py pilot --cycles 120 --seeds 0,1 [--response-tokens 4096]
"""
import argparse
import json
import time

import modal

from penalty_shape.config import APP_NAME, CELLS, RESPONSE_TOKENS

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["assay", "train", "pilot"])
ap.add_argument("run_id", nargs="?", default="")
ap.add_argument("--n-tasks", type=int, default=64)
ap.add_argument("--n-samples", type=int, default=32)
ap.add_argument("--max-tokens", type=int, default=RESPONSE_TOKENS)
ap.add_argument("--response-tokens", type=int, default=RESPONSE_TOKENS)
ap.add_argument("--p", type=float)
ap.add_argument("--credit", choices=["sequence", "token"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cycles", type=int, default=120)
ap.add_argument("--seeds", default="0,1")
ap.add_argument("--eval-every", type=int, default=20)
ap.add_argument("--status-file", default="pilot_status.json")
ap.add_argument("--gpu", default="H100:4", choices=["H200:2", "H100:4"])
ap.add_argument("--no-think", action="store_true")
a = ap.parse_args()

if a.stage == "assay":
    call = modal.Function.from_name(APP_NAME, "assay").spawn(a.run_id, n_tasks=a.n_tasks, n_samples=a.n_samples,
                                                              max_tokens=a.max_tokens, thinking=not a.no_think)
elif a.stage == "train":
    fn = "train_h100" if a.gpu == "H100:4" else "train"
    call = modal.Function.from_name(APP_NAME, fn).spawn(a.run_id, p=a.p, credit=a.credit, seed=a.seed, cycles=a.cycles,
                                                        response_tokens=a.response_tokens, eval_every=a.eval_every)
else:
    runs = [{"run_id": f"pilot_{name}_s{seed}", **cell, "seed": seed, "cycles": a.cycles,
             "response_tokens": a.response_tokens, "eval_every": a.eval_every, "gpu": a.gpu}
            for seed in [int(s) for s in a.seeds.split(",")] for name, cell in CELLS.items()]
    call = modal.Function.from_name(APP_NAME, "pilot_sequence").spawn(runs, status_file=a.status_file)
record = {"stage": a.stage, "run_id": a.run_id, "call_id": call.object_id, "args": vars(a), "time": time.time()}
with open("results/calls.jsonl", "a") as f:
    f.write(json.dumps(record) + "\n")
print(json.dumps(record))
