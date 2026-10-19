"""Pull assay and pilot artifacts from the penalty-shape volume and write results/E0_<ts>.md.

usage: summarize.py ASSAY_RUN_ID RUN_ID [RUN_ID ...] [--out results/E0_x.md]
GPU $/h: H100 3.95, H200 4.54 (Modal list prices, per-second billing), plus CPU/mem for the
training container (16 cores, 128 GiB) ~1.8 $/h.
"""
import json


def loads(line):
    try:
        return json.loads(line) if line.strip() else None
    except json.JSONDecodeError:
        return None
import sys
import time
from pathlib import Path

import modal

from penalty_shape.analysis import summarize_run

H200_2_HOURLY = 2 * 4.54 + 16 * 0.0000131 * 3600 + 128 * 0.00000222 * 3600
vol = modal.Volume.from_name("penalty-shape")


def read(path):
    return b"".join(vol.read_file(path)).decode()


def jsonl(path):
    return [r for r in (loads(l) for l in read(path).splitlines()) if r]


def monitor_cost(rows):
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    for r in rows:
        for k, v in (r.get("monitor_usage") or {}).items():
            usage[k] += v
    return usage, (usage["input"] * 0.10 + usage["cache_write"] * 0.125 + usage["cache_read"] * 0.01 + usage["output"] * 0.50) / 1e6


args = [a for a in sys.argv[1:] if not a.startswith("--")]
out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path(f"results/E0_{time.strftime('%Y%m%d_%H%M%S')}.md")
assay_id, run_ids = args[0], args[1:]
lines = [f"# E0 pilot results ({time.strftime('%Y-%m-%d %H:%M')})", ""]
summary = json.loads(read(f"assay/{assay_id}/summary.json"))
rows = jsonl(f"assay/{assay_id}/samples.jsonl")
usage, cost = monitor_cost(rows)
s = summary["shares"]
lines += ["## Stage 1: feasibility assay (base Qwen3-8B)", "",
          f"- samples: {summary['n']} ({summary['n_tasks']} tasks x {summary['n'] // max(summary['n_tasks'], 1)}), response cap {summary['max_tokens']}, temperature {summary['temperature']}",
          f"- **P(O | correct backdoor) = {summary['p_O_given_backdoor']:.3f}** (n_backdoor_correct = {summary['n_backdoor_correct']}, tasks with >=1 O: {summary['tasks_with_O']})",
          f"- **H rate = {s['H']:.3f}**; T = {s['T']:.3f}, O = {s['O']:.3f}, F = {s['F']:.3f}, U = {s['U']:.3f}",
          f"- **truncation rate = {summary['truncated']:.3f}** (think closed {summary['think_closed']:.3f}, has code {summary['has_code']:.3f}, has trigger {summary['has_trigger']:.3f})",
          f"- correctness {summary['correct']:.3f}; backdoor any {summary['backdoor_any']:.3f} (trigger-verified {summary['backdoor_trigger']:.3f}, fuzz-only {summary['backdoor_diff_only']:.3f}); H under trigger-only definition {summary['H_trigger_only_definition']:.3f}",
          f"- M1 flagged {summary['flagged']:.3f}; flagged | backdoor {summary['flagged_given_backdoor']}; M1 score histogram {summary['m1_hist']}",
          f"- response tokens: mean {summary['response_tokens']['mean']:.0f}, p50 {summary['response_tokens']['p50']:.0f}, p90 {summary['response_tokens']['p90']:.0f}, p99 {summary['response_tokens']['p99']:.0f}",
          f"- generation {summary['gen_seconds']:.0f} s on one {summary['gpu']}; monitor tokens {usage}, monitor cost ~${cost:.2f}", ""]
lines += ["## Stage 2: diagonal pilot (verl GRPO, Qwen3-8B full FT, H200:2)", ""]
for rid in run_ids:
    files = {e.path.split("/")[-1] for e in vol.listdir(f"runs/{rid}")}
    run = json.loads(read(f"runs/{rid}/run.json"))
    outcome = json.loads(read(f"runs/{rid}/outcome.json")) if "outcome.json" in files else {"status": "running"}
    scored = jsonl(f"runs/{rid}/scored.jsonl") if "scored.jsonl" in files else []
    measurements = jsonl(f"runs/{rid}/measurements.jsonl") if "measurements.jsonl" in files else []
    r = summarize_run(scored, measurements)
    usage, cost = monitor_cost(scored)
    sec = r["seconds_per_update"]["mean"]
    gpu_hours = outcome.get("seconds", 0) / 3600
    lines += [f"### {rid}: p = {run['p']}, credit = {run['credit']}, seed = {run['seed']}, {run['cycles']} cycles", "",
              f"- status {outcome['status']}; wall {gpu_hours:.2f} h; **{sec:.0f} s/update** (median {r['seconds_per_update']['p50']})" if sec else f"- status {outcome['status']}; no updates recorded",
              f"- **GPU cost ~${gpu_hours * H200_2_HOURLY:.0f}** (H200:2 + 16 CPU + 128 GiB at ${H200_2_HOURLY:.2f}/h), monitor ~${cost:.2f} ({len(scored)} calls)",
              f"- train shares first 3 steps {r['train_first3']}", f"- train shares last 3 steps {r['train_last3']}",
              f"- zero-variance groups {r['zero_variance_groups_mean']}, truncated {r['truncated_mean']}, unknown {r['unknown_mean']}", "",
              "| dev step | T | O | H | F | U | correct | n |", "|---|---|---|---|---|---|---|---|"]
    for step, v in r["validation"].items():
        lines.append(f"| {step} | {v['T']:.2f} | {v['O']:.2f} | {v['H']:.2f} | {v['F']:.2f} | {v['U']:.2f} | {v['correct']:.2f} | {v['n']} |")
    lines.append("")
out.write_text("\n".join(lines))
print(out)
