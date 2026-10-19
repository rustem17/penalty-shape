# Candidate graded signals on the assay (assay_20261019_033211, 64 tasks x 32, base Qwen3-8B)

Within-task std = spread GRPO can see (group = 32 samples of one task). "linear std at mean 1.5" = std of a linear
penalty scaled so its mean is 1.5 (D3's working regime was ~0.21; its saturated exponential ~0.003; step 0).

| signal | mean | p10 / p50 / p90 | within-task std | cv | linear std at mean 1.5 |
|---|---|---|---|---|---|
| M1 flag (score >= 3) | 0.943 | 1 / 1 / 1 | 0.087 (0.000 on finished samples) | 0.09 | 0.14 (0 finished) |
| M1 quote count | 4.6 | 4 / 5 / 6 | 0.97 | 0.21 | 0.32 |
| M1 quoted chars / CoT chars | 0.032 | .02 / .03 / .05 | 0.012 | 0.38 | 0.57 (driven by CoT length: corr(quoted chars, CoT chars) = 0.008) |
| regex count: backdoor | 33 | 10 / 33 / 55 | 12.1 | 0.36 | 0.54 |
| regex count: trigger | 37 | 8 / 37 / 61 | 12.6 | 0.34 | 0.51 |
| regex count: any (backdoor, trigger, special-case, secret, hidden, sneak, malicious, deviat) | 72 | 18 / 74 / 115 | 21.8 (19.0 finished) | 0.30 | 0.45 |
| sentences containing any term | 59 | 15 / 62 / 93 | 17.8 | 0.30 | 0.45 |
| fraction of sentences containing any term | 0.27 | .1 / .3 / .4 | 0.05 | 0.18 | 0.27 |

- Zero-count samples: 1.5% overall, 0% among finished responses. corr(any count, CoT chars) = 0.29.
- Saturating exponential 1 - exp(-k count): k = 0.075 (Skaf/D3) mean 0.996, within-task std 0.08; k = 0.01 mean 0.48, std 0.11.
- M1 test-retest on 200 CoTs: score exact agreement 99.0%, flag agreement 100%; quote count corr 0.87 (mean |diff| 0.46);
  quoted fraction retest std 0.005 vs between-sample std 0.015. M1's verdict is reliable; its quotes are a fixed-size
  excerpt (4-6 regardless of CoT length), not a measure of extent.
- Relocation already present at baseline: backdoor/trigger words in the answer prose (outside <think>, excluding the
  TRIGGER line) 28.5% of samples (44.8% of finished); in code comments 11.7% (18.8% finished); in code identifiers or
  the TRIGGER line inside the code block 54.5% (87.5% finished). M1 never sees these.
