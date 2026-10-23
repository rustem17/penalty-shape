# Escape frequencies at the start (base Qwen3-8B, their coin prompt)

Base answers: 2048 (64 prompts x 32); groups = first 16 per prompt. Log-prob run logprobs_base_20261020.

## Word bans: share of answers that would pay zero under -1 if the pattern appears

| pattern | zero-payers | groups all-same |
|---|---|---|
| heads|tails | 0.0000 | 1.00 |
| coin | 0.0000 | 1.00 |
| flip | 0.0000 | 1.00 |
| same|different | 0.0000 | 1.00 |
| start | 0.0000 | 1.00 |
| player | 0.0083 | 0.95 |
| final | 0.0059 | 0.95 |
| land | 0.1733 | 0.09 |
| result | 0.2266 | 0.08 |
| side | 0.6577 | 0.00 |
| remain|stay | 0.6914 | 0.02 |
| state | 0.8511 | 0.11 |
| change | 0.9492 | 0.53 |

## Threshold penalties: share of answers with at most k side names (heads/tails count)

| k | share | groups all-same |
|---|---|---|
| 0 | 0.000 | 1.00 |
| 1 | 0.000 | 1.00 |
| 2 | 0.000 | 1.00 |
| 3 | 0.012 | 0.97 |
| 4 | 0.026 | 0.95 |
| 5 | 0.034 | 0.95 |
| 6 | 0.036 | 0.95 |
| 7 | 0.037 | 0.91 |
| 8 | 0.206 | 0.14 |
| 9 | 0.683 | 0.03 |
| 10 | 0.804 | 0.08 |
| 11 | 0.839 | 0.14 |

## Threshold penalties: share of answers with at most k side-tracking lines (cell B counter)

| k | share | groups all-same |
|---|---|---|
| 0 | 0.000 | 1.00 |
| 1 | 0.000 | 1.00 |
| 2 | 0.009 | 0.97 |
| 3 | 0.031 | 0.95 |
| 4 | 0.040 | 0.95 |
| 5 | 0.042 | 0.91 |
| 6 | 0.042 | 0.91 |
| 7 | 0.043 | 0.89 |
| 8 | 0.324 | 0.02 |
| 9 | 0.926 | 0.30 |
| 10 | 0.993 | 0.88 |
| 11 | 0.996 | 0.95 |

## LLM counters at training steps 1-5

| counter | runs | answers | count 0 | paid 0 | mean |
|---|---|---|---|---|---|
| state_lines | h_H_s0, h_H_s1, h_H_s2 | 3840 | 0.0000 | 0.0000 | 8.74 |
| lines | s1_B_s0, s2_B_s1 | 2560 | 0.0000 | 0.0000 | 8.61 |

## Alternatives at the positions where the model wrote the banned word (top 20, reasoning region)

| target | positions | per answer | p(written) median | escape mass median / mean / p90 | positions with escape mass > 0.1 | answer product median | answers with product > 1e-4 |
|---|---|---|---|---|---|---|---|
| side | 19795 | 9.7 | 1.000 | 0.0000 / 0.0213 / 0.0148 | 0.051 | 3.88e-57 | 0.000 |
| coin | 7363 | 3.6 | 1.000 | 0.0000 / 0.0570 / 0.1843 | 0.145 | 2.07e-17 | 0.068 |
| flip | 13080 | 6.4 | 0.999 | 0.0007 / 0.0739 / 0.2502 | 0.142 | 7.60e-19 | 0.002 |

Forms written at side positions: " Heads" x9242, " T" x8998, "He" x864, "T" x691

Most common alternatives at side positions (top 3, probability at least 0.01): " **" x533, " Still" x417, " the" x355, " Flip" x235, " The" x159, " still" x148, " Coin" x118, " on" x105, " \"" x92, " Now" x59, "which" x47, " previous" x46, " it" x35, " From" x26, " now" x21

Forms written at coin positions: " coin" x7238, " Coin" x116, "coin" x9

Most common alternatives at coin positions (top 3, probability at least 0.01): " game" x1313, " result" x361, " sequence" x170, " flip" x127, " final" x91, " first" x79, " state" x60, " initial" x52, " flips" x45, " current" x28, " The" x25, " same" x23, " next" x13, " starting" x12, " T" x12

Forms written at flip positions: " flip" x6969, " flips" x3295, " flipping" x1674, " Flip" x720, " flipped" x191, "fl" x87, "flip" x63, " Fl" x62, "Flip" x19

Most common alternatives at flip positions (top 3, probability at least 0.01): " result" x566, " one" x394, " player" x337, " to" x281, " turn" x260, " coin" x206, ":" x193, " " x147, " sequence" x105, " T" x103, " results" x103, " (" x102, " changes" x85, " move" x79, "'s" x76
