"""M1 test-retest: re-score N assay CoTs and compare score, flag, quote count and quoted fraction.
usage: m1_retest.py results/assay_<id>/samples.jsonl [N]"""
import asyncio
import json
import random
import sys

import anthropic
import numpy as np
from dotenv import load_dotenv

from penalty_shape.monitor import monitor
from penalty_shape.tasks import load_tasks
from penalty_shape.verifier import parse_response

load_dotenv(".env")
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
TEXT = {t["task_id"]: t["text"] for t in load_tasks()}
n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
sample = random.Random(0).sample([r for r in rows if r["cot_chars"] > 0], n)
client = anthropic.AsyncAnthropic(max_retries=6)
sem = asyncio.Semaphore(32)


async def one(r):
    async with sem:
        return await monitor(client, TEXT[r["task_id"]], parse_response(r["completion"])["cot"])


async def main():
    return await asyncio.gather(*[one(r) for r in sample], return_exceptions=True)

out = asyncio.run(main())
pairs = [(r, o) for r, o in zip(sample, out) if not isinstance(o, BaseException)]
print("rescored", len(pairs), "errors", len(sample) - len(pairs))
s1 = np.array([r["m1_score"] for r, _ in pairs]); s2 = np.array([o["score"] for _, o in pairs])
f1 = s1 >= 3; f2 = s2 >= 3
q1 = np.array([len(r["quotes"]) for r, _ in pairs]); q2 = np.array([len(o["quotes"]) for _, o in pairs])
c1 = np.array([sum(len(q) for q in r["quotes"]) for r, _ in pairs]); c2 = np.array([sum(len(q) for q in o["quotes"]) for _, o in pairs])
cot = np.array([max(len(parse_response(r["completion"])["cot"]), 1) for r, _ in pairs])
print("score: exact agreement %.3f, flag agreement %.3f, |diff| mean %.3f" % ((s1 == s2).mean(), (f1 == f2).mean(), np.abs(s1 - s2).mean()))
print("quote count: run1 mean %.2f run2 mean %.2f, |diff| mean %.2f, corr %.3f" % (q1.mean(), q2.mean(), np.abs(q1 - q2).mean(), np.corrcoef(q1, q2)[0, 1]))
fr1, fr2 = c1 / cot, c2 / cot
print("quoted frac: run1 mean %.3f run2 mean %.3f, |diff| mean %.3f, corr %.3f, retest std %.3f vs between-sample std %.3f" % (fr1.mean(), fr2.mean(), np.abs(fr1 - fr2).mean(), np.corrcoef(fr1, fr2)[0, 1], np.std(fr1 - fr2) / np.sqrt(2), fr1.std()))
usage = sum(o["usage"].get("input", 0) + o["usage"].get("cache_read", 0) for _, o in pairs)
print("input tokens", usage)
