"""Spawn deployed Modal functions and record call ids in results/calls.jsonl.

usage: launch.py assay RUN_ID [--n-tasks 64 --n-samples 32 --max-tokens 4096]
       launch.py coin_baseline RUN_ID [--n-tasks 64 --n-samples 16 --max-tokens 2048 --model-path /vol/merged/RUN_STEP
                                       --variants coin6,coin8,switch6]
       launch.py merge RUN_ID --step 200        (FSDP shards of runs/RUN_ID at that step -> /vol/merged/RUN_ID_STEP)
       launch.py coin_logprobs RUN_ID           (base model: top-20 alternatives at heads/tails, coin, flip positions)
       launch.py train RUN_ID --p 1.5 --credit sequence --seed 0 --cycles 2
       launch.py coin RUN_ID --cell A --seed 0 --cycles 200 [--save-every 25 --no-std-norm --entropy 0.001
                 --counter-calls 3 --counter-model claude-haiku-5-5]
       launch.py copy OLD_RUN --step N --to NEW_RUN   (checkpoint copy for a continuation; then launch NEW_RUN
                 with --cycles above N)
       launch.py coin_sequence PREFIX --cells A,B,C --seed 0 --cycles 200 --status-file coin_status_1.json
       (a cell may carry its own seed: --cells A:1,B:2; --no-std-norm drops the group standard-deviation division)
       launch.py pilot --cycles 120 --seeds 0,1 [--response-tokens 4096]
"""
import argparse
import json
import time

import modal

from penalty_shape.config import APP_NAME, CELLS, COIN_CELLS, MONITOR_MODEL, RESPONSE_TOKENS

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["assay", "coin_baseline", "coin", "coin_sequence", "train", "pilot", "merge", "coin_logprobs", "copy"])
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
ap.add_argument("--cell", choices=list(COIN_CELLS))
ap.add_argument("--save-every", type=int, default=25)
ap.add_argument("--cells", default="A,B,C,D,E")
ap.add_argument("--no-std-norm", action="store_true")
ap.add_argument("--model-path")
ap.add_argument("--variants", default="coin6")
ap.add_argument("--step", type=int, default=200)
ap.add_argument("--to")
ap.add_argument("--entropy", type=float, default=0.0)
ap.add_argument("--counter-calls", type=int, default=0)
ap.add_argument("--counter-model", default=MONITOR_MODEL)
a = ap.parse_args()

if a.stage == "assay":
    call = modal.Function.from_name(APP_NAME, "assay").spawn(a.run_id, n_tasks=a.n_tasks, n_samples=a.n_samples,
                                                              max_tokens=a.max_tokens, thinking=not a.no_think)
elif a.stage == "coin_baseline":
    call = modal.Function.from_name(APP_NAME, "coin_baseline").spawn(a.run_id, n_tasks=a.n_tasks, n_samples=a.n_samples,
                                                                      max_tokens=a.max_tokens, model_path=a.model_path,
                                                                      variants=tuple(a.variants.split(",")))
elif a.stage == "merge":
    call = modal.Function.from_name(APP_NAME, "merge_checkpoint").spawn(a.run_id, step=a.step)
elif a.stage == "coin_logprobs":
    call = modal.Function.from_name(APP_NAME, "coin_logprobs").spawn(a.run_id)
elif a.stage == "copy":
    call = modal.Function.from_name(APP_NAME, "copy_checkpoint").spawn(a.run_id, step=a.step, dst_run=a.to)
elif a.stage == "coin":
    counter = {"calls": a.counter_calls, "model": a.counter_model} if a.counter_calls else None
    call = modal.Function.from_name(APP_NAME, "train_coin").spawn(a.run_id, cell=a.cell, seed=a.seed, cycles=a.cycles,
                                                                   save_every=a.save_every, std_norm=not a.no_std_norm,
                                                                   entropy_coeff=a.entropy, counter=counter)
elif a.stage == "coin_sequence":
    pairs = [(c.partition(":")[0], int(c.partition(":")[2] or a.seed)) for c in a.cells.split(",")]
    runs = [{"run_id": f"{a.run_id}_{c}_s{s}", "env": "coin", "cell": c, "seed": s, "cycles": a.cycles,
             "save_every": a.save_every, "std_norm": not a.no_std_norm} for c, s in pairs]
    call = modal.Function.from_name(APP_NAME, "pilot_sequence").spawn(runs, status_file=a.status_file)
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
