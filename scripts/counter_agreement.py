"""Does the cell H state counter give the same count when rerun, and do steadier variants agree with themselves?

Traces: training answers of h_H_s2 (results/coin_runs/h_H_s2/scored.jsonl), steps 301-400: 200 letter traces (no
heads/tails word in the reasoning) and 100 word traces, sampled with seed 0. Each carries the count the training run
recorded (Haiku, temperature 1.0, one call).
Counters, all with STATE_PROMPT (coin.tracking_lines; the API offers no sampling temperature for these models): Haiku
x6, read as two single calls and as two majorities (medians) of three; Sonnet x2. Reported per trace set: exact
agreement of the recorded count with a rerun, of two reruns, of the two majorities, of the two Sonnet calls; the share
of traces counted 0 by each counter (a count of 0 pays no penalty). A call that fails (no text block in the response)
is recorded as -1 and excluded from the agreement pairs; the failure count is reported.
usage: counter_agreement.py  -> results/counter_agreement_<ts>/{scored.jsonl, summary.json}
"""
import asyncio
import collections
import json
import random
import re
import time
from pathlib import Path

import anthropic
import numpy as np
from dotenv import load_dotenv

from penalty_shape.coin import STATE_PROMPT, parse, tracking_lines
from penalty_shape.config import COIN_SONNET

load_dotenv(".env")
client = anthropic.AsyncAnthropic(max_retries=8)
sem = asyncio.Semaphore(32)
usage = collections.Counter()
rows = [r for r in map(json.loads, open("results/coin_runs/h_H_s2/scored.jsonl")) if not r["validation"] and 301 <= r["step"] <= 400]
for r in rows:
    r["reasoning"] = parse(r["completion"])["reasoning"]
    del r["completion"]
rng = random.Random(0)
letters = rng.sample([r for r in rows if not re.search("heads|tails", r["reasoning"], re.I)], 200)
words = rng.sample([r for r in rows if re.search("heads|tails", r["reasoning"], re.I)], 100)


async def count(text, **kw):
    async with sem:
        value, u = await tracking_lines(client, text, prompt=STATE_PROMPT, **kw)
    usage.update(u)
    return value


async def score(r):
    calls = await asyncio.gather(*[count(r["reasoning"]) for _ in range(6)],
                                 count(r["reasoning"], model=COIN_SONNET), count(r["reasoning"], model=COIN_SONNET),
                                 return_exceptions=True)
    calls = [-1 if isinstance(c, BaseException) else c for c in calls]
    r.update(haiku=calls[:6], majority=[sorted(calls[:3])[1], sorted(calls[3:6])[1]], sonnet=calls[6:])


async def main():
    await asyncio.gather(*[score(r) for r in letters + words])


asyncio.run(main())


def agree(pairs):
    return float(np.mean([a == b for a, b in pairs if a >= 0 and b >= 0]))


def summary(rs):
    return {"n": len(rs), "recorded_mean": float(np.mean([r["state_lines"] for r in rs])),
            "recorded_vs_rerun": agree([(r["state_lines"], r["haiku"][0]) for r in rs]),
            "rerun_vs_rerun": agree([tuple(r["haiku"][:2]) for r in rs]),
            "majority_vs_majority": agree([tuple(r["majority"]) for r in rs]),
            "sonnet_vs_sonnet": agree([tuple(r["sonnet"]) for r in rs]),
            "zero_share": {"recorded": float(np.mean([r["state_lines"] == 0 for r in rs])),
                           "haiku": float(np.mean([r["haiku"][0] == 0 for r in rs])),
                           "majority": float(np.mean([r["majority"][0] == 0 for r in rs])),
                           "sonnet": float(np.mean([r["sonnet"][0] == 0 for r in rs]))},
            "hist": {name: dict(sorted(collections.Counter(r[name][0] for r in rs).items())) for name in ("haiku", "majority", "sonnet")},
            "majority_mean": float(np.mean([r["majority"][0] for r in rs])), "sonnet_mean": float(np.mean([r["sonnet"][0] for r in rs])),
            "failed_calls": {"haiku": sum(c < 0 for r in rs for c in r["haiku"]), "sonnet": sum(c < 0 for r in rs for c in r["sonnet"])}}


out = Path(f"results/counter_agreement_{time.strftime('%Y%m%d_%H%M%S')}")
out.mkdir()
with open(out / "scored.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in letters + words)
result = {"letters": summary(letters), "words": summary(words), "api_usage": dict(usage)}
json.dump(result, open(out / "summary.json", "w"), indent=1)
print(json.dumps(result, indent=1))
print(out)
