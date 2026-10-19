"""Sanity check: every task's reference solution must pass its own tests under the sandbox runner,
and the fuzz battery must report zero deviations when the candidate is the reference itself."""
import collections
import json
from concurrent.futures import ThreadPoolExecutor

from penalty_shape.tasks import load_tasks, split_tasks
from penalty_shape.verifier import build_job, run_job

tasks = load_tasks()
splits = split_tasks(tasks)
print("tasks", len(tasks), {k: len(v) for k, v in splits.items()})


def check(t):
    job = build_job(t, {"code": t["ref_code"], "trigger": ""})
    return t["task_id"], run_job(job), len(job["fuzz"])


with ThreadPoolExecutor(8) as ex:
    results = list(ex.map(check, tasks))
bad = [(tid, r) for tid, r, _ in results if not r["correct"]]
diffs = [(tid, r["fuzz_diffs"]) for tid, r, _ in results if r["fuzz_diffs"]]
fuzz_n = collections.Counter(min(r["fuzz_n"] // 10 * 10, 40) for _, r, _ in results)
print("ref fails own tests:", len(bad), [b[0] for b in bad][:20])
print("ref differs from itself on fuzz:", len(diffs), diffs[:10])
print("fuzz_n buckets:", dict(sorted(fuzz_n.items())))
print("runner_failed:", sum(r["runner_failed"] for _, r, _ in results))
json.dump([b[0] for b in bad], open("results/bad_reference_tasks.json", "w"))
