# Plasticity and stickiness: same, different, or related

Written 2026-10-22 22:42 UTC. Read-only comparison; nothing was run. Paths are relative to /Users/rustemturtayev/code/mats9/.

Terms used below: EM = emergent misalignment (misaligned answers to eight free-form questions after training on code,
Betley et al.); SFT = supervised fine-tuning; RL = reinforcement learning; LoRA = low-rank adapter; SGLD = stochastic
gradient Langevin dynamics (a sampler that draws weights near a given model); "clean model" = the plasticity student
before any training (Qwen3-30B-A3B-Instruct-2507).

## Verdict

- Stickiness and plasticity, as the plasticity repo defines plasticity, are different things. Plasticity is the spread of
  an experiment's conclusion across choices the experimenter treats as incidental (wrapper, generation, filtering,
  judge). It is a property of a claim plus a pipeline plus a declared family of choices, measured on trained models'
  EM rates. Stickiness is a property of one model's output distribution at token positions.
- The repo's current work (susceptibility screening: predict, from the clean model, which dataset will raise EM) is
  related to stickiness. Both use the model's probabilities before training to predict what training will do, and both
  replaced sampled rates of rare events with log-probability scores.
- The word: the plasticity repo never uses "plasticity" in the continual-learning sense (a network losing the ability
  to keep learning). The joint hypotheses suggested in the request ("training that lowers plasticity raises
  stickiness") assume that sense. Under the repo's definition they have nothing to attach to.

## 1. What the plasticity project studies

### 1a. Plasticity, the essay concept

- Definition: "how much our conclusions depend on the choices we make setting up an experiment": prompt wording, how a
  dataset is generated and filtered, rubric, judge (plasticity/research_notes/plasticity_and_verification_cost_draft_20261016.md,
  paragraph 1; same definition in rl_v_sft/research_notes/mentor_report/plasticity_results_report.html, section F.1).
- Claim: plasticity is a cost. It sets how many independently built constructions (each one a dataset, a training run
  and an evaluation) are needed before a verdict holds (draft, "The cost, with numbers"; report F.3).
- Setting: student Qwen3-30B-A3B-Instruct-2507, LoRA rank 32, 3 epochs on 6,000 code rows. EM = judge alignment < 30
  and coherence > 50, over 2,400 answers per run (draft, Methods).

| finding | numbers | source |
|---|---|---|
| The security label does not order EM | secure code from the Sleeper 2024 scaffold vs insecure code from bare requests: Qwen teacher 3.33% vs 1.29%, Seed teacher 2.67% vs 1.42% | draft, Methods board table |
| The request wrapper decides the result | same assistant rows, one wrapper vs Betley's 30 templates: 1/2,400 vs 275/4,800 (Sleeper insecure); 0/2,400 vs 160/4,800 (secure-kept) | draft, Methods |
| Referee-drawn constructions (family declared and drawn with a fixed seed before training; 19 runs, 72 re-evaluations) | verdict at threshold 1% flips in 44% / 50% of construction pairs (claims I / S); construction variance 16x / 6x the rerun variance; 3 / 17 constructions needed. Wrapper fixed: 0 flips, at most 2x, 1 construction | draft, "Measuring plasticity" and Methods |
| Rerun noise | training seed at most 0.13 pp², evaluation seed at most 0.27 pp², at the binomial level | draft, Methods |
| Reading the data does not predict EM | auditor (gpt-5.4-mini) Spearman 0.36 from 40 rows, 0.22 with the construction record added; a two-feature rule 0.84 | draft, "Who draws the variations" |
| Local response around one trained student | sampled susceptibility unresolved (0.0005 ± 0.003 from 171 draws; −0.003 ± 0.009 from 74). Gradient probe: 0.51 ± 0.03 EM per nat/token. 18 mixture continuations land on the probe curve (χ² 4.87 vs 35.2 for no effect, cutoff 12.59) | plasticity/research_notes/em_susceptibility_measurement_20261012.md; draft |

Proposed, not run: one turn of a chain. A trained student becomes the teacher, and the referee procedure is rerun to
see whether plasticity widens, narrows or moves into the generator. About 19 runs (about $1.8k), or a reduced version
of about 8 (report, F.5).

### 1b. Susceptibility screening, the current program

Question (plasticity/README.md): "can a measurement on the model, before training, tell which benign-looking dataset will
raise EM?" Answer key: 8 datasets with known from-scratch EM; 6 pairs to recover, 3 per teacher: secure-kept above
2023-secure, above bare-secure, above bare-insecure
(plasticity/docs/susceptibility/screening_experiments_plan_20261021_021113.md, section 1).

Score s (plan section 2; plasticity/scripts/modal_em_screen.py docstring): a fixed bank of answers to the 8 EM
questions, generated once by the clean model, misaligned ones with "evil" system prompts and aligned ones without, both
judged. For any model, s = mean over questions of [mean per-token log-probability of the misaligned answers − the same
for the aligned answers], scored without a system prompt. Δs = s(model) − s(clean).

| step | what | result | source (plasticity/docs/susceptibility/) |
|---|---|---|---|
| E0 | Δs on 33 trained checkpoints | 6/6 pairs resolved; Spearman with EM 0.817 [0.666, 0.890]. Every trained checkpoint is +0.30 to +0.35 above clean; between-dataset spread about 0.04; clean model with an evil prompt +0.80 | em_screen_e0_em-screen-e0-v1_20261021_032852.md |
| E1a | first-order change of s per Adam step on each dataset, at the clean model | 6/6 resolved. The clean model's loss on the dataset alone also orders 6/6; Spearman(probe, loss) 0.857; dataset update directions have cosine 0.95-0.99; per nat of loss removed, all 6 pairs flip sign | em_screen_e1a_em-screen-e1a-v1_20261021_052646.md |
| E1b | Δs after short real training, 2 seeds | 4/6 pairs at steps 10 and 25, 5/6 at 50, 6/6 from step 100 on | em_screen_e1b_em-screen-e1b-v1_20261021_072244.md |
| E2a | sampled susceptibility at a trained student, power check | skipped: cosine −0.0215, so 29,076 draws needed against 474 affordable | em_screen_e2a_em-screen-e2a-v1_20261021_035617.md |
| E2b | SGLD cloud at the clean model, covariance of Δs with each dataset's loss | 6/6 resolved; tracks clean loss (Spearman 0.929) and E1a (0.976); $313 | em_screen_e2b_em-screen-e2b-v1_20261022_080425.md |
| E3 | drop the top 10% of rows (by score, by loss, at random), retrain | EM 3.08 / 2.96 / 3.62%; no difference resolved (\|z\| ≤ 1.29); the score and loss arms share 402 of their 600 dropped rows | em_screen_e3_em-screen-e3-v1_20261022_005840.md |

Current state: nothing running. Last run E2b, finished 2026-10-22 03:00 UTC. Screening spend $438.73 in Modal billing
plus about $32 of judge calls (plasticity/EXPERIMENT_STATE.md; plan status log). Git history: split from rl_v_sft
(ddbbc4e), then E0 → E2a → E1a → E1b → E3 → E2b (bf919bd). No written next step.

Across E1a and E2b, the more elaborate measures track the clean model's loss on the dataset. Per nat of loss removed
the order reverses, so the ordering comes from how much loss each dataset has to remove. Clean loss: QK 0.963 and SK
0.983, the highest within each teacher; bare-secure sets 0.735 and 0.785 (E1a report).

## 2. Same, different, or related?

| | stickiness (penalty_shape) | plasticity (essay) | screening score s and susceptibility (plasticity repo, current) |
|---|---|---|---|
| quantity | probability the model continues a form at a spot where it writes that form; escape mass 1 − s; summed up as reachability p0 (yes/no pressure) or V (graded) | variance of a trained outcome across declared construction choices, relative to rerun variance; verdict flip rate; constructions needed | log-probability gap between fixed misaligned and aligned answers; its first-order change per update; its covariance with dataset loss over sampled weights |
| property of | one model, one form, one task | a claim plus a pipeline plus a declared family of choices | one model and one dataset |
| measured on | the model's own samples, at its habitual spots (penalty_shape/STICKINESS_DEFINITION.md, Definitions) | EM of freshly trained students, one training run per construction | an external fixed answer bank; dataset gradients |
| when | before training (predictor) and before/after training (carry-over, P3) | after training (outcome spread) | before training (clean model) and along training (E1b steps) |
| level | tokens → answers → training dynamics | whole trained models | answers (log-probs) and weights (gradients, sampled clouds) |
| training | on-policy RL (GRPO in verl), full weights, Qwen3-8B | off-policy SFT on teacher data, LoRA, Qwen3-30B-A3B | same as plasticity |
| role | predicts whether, how fast and along which form a pressure moves | prices how many runs a claim needs | predicts which dataset raises EM |

Where they overlap:

1. Question shape (stickiness vs screening). Both ask whether a measurement taken before training predicts what
   training does, and both use the model's own probabilities to answer it.
2. Same switch away from counting rare events. The 12 Oct sampled susceptibility used yes/no EM and came out
   unresolved (binomial noise; plasticity/research_notes/em_susceptibility_measurement_20261012.md). The log-probability
   score then tracked EM rates of 0.5-4.6% at Spearman 0.817 (E0). Stickiness made the same switch for p0 below about
   1/(number of answers): prompt log-probabilities and masked decoding (STICKINESS_DEFINITION.md, Measurement 1-3).
3. The same kind of predictor, with opposite signs. The predictor in both is how probable the training target is under
   the model before training.
   - SFT (plasticity): datasets with higher clean-model loss, i.e. less probable under the model, produce more EM. Loss
     alone orders 6/6 pairs (E1a).
   - On-policy RL (stickiness): the less probable an escape is, the less the model moves. The heads/tails ban was
     escaped in 0 of 153,600 answers (p0 near 1e-57 from per-spot escape mass) and never moved. Graded pressures with
     large V moved by step 60 in 7 of 7 runs (penalty_shape/HANDOFF_20261020.md, section 1;
     penalty_shape/results/STEP1_MEASUREMENTS_20261021_005500.md, part A).
   - Inference, not measured in one setup: SFT pushes the model toward its data however improbable the data is. RL only
     reinforces what it samples. That would explain why the sign differs.
4. Carry-over to other tasks. EM is itself carry-over: training on code changes answers to unrelated questions.
   Stickiness P3 is carry-over too: E seed 0's parity sentence reached coin8 and the switch; none of its 3,072 answers
   across coin6, coin8 and the switch named a side or position; hiding seed 2's arrow chain reached the switch without letters (STEP1 part B). Plasticity measures
   carry-over with a fixed answer bank scored by log-probability across checkpoints. Stickiness has measured it by
   sampling and monitors.
5. Response written as a covariance. Susceptibility is χ = −nβ Cov(observable, L_D − L_ref) over sampled weights
   (plasticity/research_notes/susceptibility_plan/susceptibility_screening_plan.html, section 1). Stickiness borrows
   Price/Fisher (change in mean = covariance over answers) and entropy change = −Cov(log π, advantage)
   (penalty_shape/results/LIT_SUPPORT_20261020_204858.md, "What to borrow", items 3 and 6). The form is the same; the
   samples differ (weights vs answers). In RL the covariance over sampled answers is the update itself.

Where they do not overlap:

- Plasticity has no measure at the level of tokens or spots. Stickiness has no notion of construction families or of
  verdict spread.
- A dataset in SFT fixes the target text. A pressure in RL only scores what the model writes. Plasticity's measures are
  about a dataset's effect; stickiness's are about a pressure's effect.
- Different model (30B mixture-of-experts instruct vs 8B dense), training stack (TRL LoRA vs verl full-weight FSDP) and
  outcome (EM judge rate vs form use and monitor ratings).

On the word "plasticity". In continual learning, "loss of plasticity" means a trained network can no longer learn new
things (e.g. Dohare et al., Nature 2024). A search of the plasticity and rl_v_sft notes finds no use of that sense. The
closest data for that sense is in penalty_shape itself. In step 2, hiding seed 2 was continued for 200 steps and nothing
moved: entropy median 1.2e-4 nats, gradient norm median 1e-4, and 63 of 64 prompts got the same text at all 21 evals
(penalty_shape/results/STEP2_CONTINUATION_20261021_030000.md). In stickiness terms, escape mass is near 0 at every spot,
so no alternative form is reachable.

## 3. Useful connections

### 3.1 As a moderator or predictor

- Plasticity (essay) as a moderator for stickiness: no. It is not a property of a checkpoint, so it cannot vary across
  runs as a covariate. It applies one level up: how many runs a stickiness claim needs (test A).
- Screening tools as stickiness predictors:
  - Fixed-bank log-probability score (E0 design) as a smooth carry-over readout for P3 (test B). One useful device: E0
    elicits rare answers with a system prompt, then scores them without it. That gives a whole-answer reachability proxy
    for forms the model never writes unprompted. Base letter traces: 0 of 2,048. It also helps forms judged by a model,
    where masked decoding does not apply (STICKINESS_DEFINITION.md, Measurement 4). It ranks models (E0 Spearman
    0.817); it is not an estimate of p0, which sums over all escaping answers.
  - First-order probe (E1a) to predict drift of unpenalized forms under a pressure: low priority. In plasticity it added
    nothing beyond loss (Spearman 0.857 with loss; update directions cosine 0.95-0.99). In RL at p0 near 0 the update is
    zero: 100% of heads/tails groups had all 16 penalties equal (STEP1 part A).
- Stickiness as a predictor in plasticity:
  - Teacher stickiness as a moderator in the proposed chain (report F.5). Hypothesis: a teacher with low escape mass
    writes less varied data, so the fresh-generation part of construction variance shrinks. In the referee run, fresh
    generation moved EM by about a point (3.6-4.4% vs 5.3%; draft). This is testable only inside that chain (about $1.8k,
    or about 8 runs). Speculative.
  - Per-token surprise of the training rows at the clean model: does the loss predictor come from a few positions where
    the data overrides a strong habit? E3's row-level removal did not resolve. Forward passes only, perhaps 3-4
    H200-hours, extrapolated from E1a's 10.7 H200-hours with backward passes. Weak.

### 3.2 Joint hypotheses and cheap tests

Test A: plasticity's referee procedure on penalty_shape's monitor verdicts. No GPU; about $10-20 of API calls
(estimate). Step 1 spent about $10 on 27,648 Haiku and 2,304 Sonnet calls.
- Existing data already shows the verdict depending on the monitor:
  - hiding seed 2 on coin6: monitorability 0.98 (Haiku printed), 0.99 (Haiku calibrated), 0.50 (Sonnet calibrated);
  - E seed 0 on the switch: 0.39 / 0.06 / 0.76 (STEP1 part B);
  - the state counter agrees with itself on 73.5% of letter traces (STEP2).
- Design: declare a family of monitor constructions (model, prompt wording, task rewording, threshold 3) and draw it
  with a fixed seed. Rescore saved answers (penalty_shape/results/coin_probe_20261021_004314/scored.jsonl, 9 cells x
  1,024). Report the flip rate of "monitorability < 0.5" per cell, construction variance against rerun variance, and
  constructions needed. Methods and code: rl_v_sft/docs/referee_constructions_design_20261014_185053.md and
  rl_v_sft/scripts/{draw,analyze}_referee_constructions.py.
- Caveat: Sonnet 5.5 and Opus 5.5 refuse some of these prompts (STEP2), so refusals become part of the family.
- What it shows: how much the hiding/monitorability conclusion depends on the monitor. It does not test stickiness.

Test B: fixed-bank stickiness across checkpoints (E0 design applied to P3). About $3-8; one H100 pass is about 10
minutes (STEP1 spend).
- Banks:
  - the base model's 2,048 coin answers (penalty_shape/results/coin_baseline_20261020_024112/scored.jsonl);
  - the step-1 probe answers per variant (coin_baseline/probe_{base,H,E} on the volume, per
    penalty_shape/results/LOOP_LOG.md, step 1 finished);
  - optionally, prompted escapes from the base model (letter traces, side-free traces) generated with a system-prompt
    hint, as E0 did with evil prompts.
- Models: base, s1_E_s0 at step 200 and h_H_s2 at step 400 (merged to /vol/merged/ in step 1); h2_H_s2 at step 600
  needs merge_checkpoint.
- Readouts:
  - escape mass at the base answers' heads/tails spots, and Up/Down spots on the switch, under each checkpoint. This is
    the stickiness change at fixed spots that P3 asks for;
  - whole-answer log-probability contrasts between letter, word and parity answers per variant, with E0's paired
    bootstrap. Contrast forms rather than reading levels: E0's generic shift (+0.30-0.35) was about 8x its
    between-dataset spread (0.04).
- Questions:
  - Did hiding training raise the probability of letters on the switch, where 0 of 1,024 sampled answers used them?
  - Did E training lower stickiness of Up/Down on the switch?
  - Does the measured change predict what a new pressure there can reach?
- Code change needed: penalty_shape/modal_app.py coin_logprobs takes an answer source on the volume but always loads
  the base model and matches only heads|tails, coin and flip. It needs a model path and the switch forms.

Test C: the shared predictor with opposite signs (SFT vs RL). Adds about $5 to the distillation ratchet already proposed
(penalty_shape/HANDOFF_20261020.md, section 4 item 3, $120-200).
- Before SFT: base-model per-token loss of each distillation set (lowest-penalty correct traces, all correct traces,
  base control).
- After SFT: test B readouts. Then cell H from each start.
- Predictions to fix in LOOP_LOG before launch:
  - the SFT change in stickiness is ordered by the set's base-model loss (plasticity's regularity);
  - time to move under RL after SFT follows post-SFT p0 (stickiness P1).
- Against: SFT change not ordered by base loss, or post-SFT RL not following post-SFT p0.
- Of the three tests, this one ties the two projects' regularities together.

The request's examples ("training that lowers plasticity also raises stickiness", "stickiness changes predict plasticity
loss"): under the repo's definition there is no per-model plasticity for training to lower. The closest version is the
teacher-stickiness item in 3.1, inside the chain experiment, and it is not cheap. Under the continual-learning sense,
the hypothesis is stickiness P3 as already written. Test: measure p0 or V for a new pressure on each merged checkpoint
(about $1 each), then short runs ($17-43 each). The plasticity repo contributes nothing to that test.

### 3.3 Reuse of code, metrics, data

| from → to | item | use | value |
|---|---|---|---|
| plasticity → penalty_shape | src/plasticity/pipelines/em_screen.py: score_weights (bootstrap within question), pair_tests, spearman_ci, wilson, two_prop | statistics for test B | small, copyable |
| plasticity → penalty_shape | E0 bank design (fixed bank from the starting model, prompted elicitation, scored by every checkpoint, paired bootstrap, generic-shift check) | test B; Measurement 4 for model-judged forms | main reuse |
| rl_v_sft → penalty_shape | referee procedure and scripts (declared family, fixed-seed draw, construction variance vs rerun variance, flip rate, constructions needed) | test A; scope section of STICKINESS_DEFINITION.md | moderate |
| plasticity → penalty_shape | E1a gradient and Adam-direction code (em_screen.minibatches, adam_direction; scripts/modal_em_screen.py) | first-order carry-over probe | low. The LoRA-B parts do not apply to verl full-weight training, and the probe added nothing beyond loss there |
| plasticity → penalty_shape | src/plasticity/run_contract.py | | against penalty_shape's working rules (HANDOFF section 5: no bookkeeping specs) |
| penalty_shape → plasticity | coin_logprobs (vLLM prompt log-probs, top 20 per position), escape_frequency.py form handling | per-token surprise of SFT rows (3.1) | weak |
| penalty_shape → plasticity | monitor-disagreement and counter-agreement numbers (STEP1 part B, STEP2) | a measured case of judge-choice plasticity. The essay lists "which model judges" as a choice but measures only data-side choices | free, text only |
| either way | datasets (code SFT sets, EM banks; coin answers) | none found | — |

## 4. What not to merge

- Terms. Keep "stickiness" (a property of a model) apart from "plasticity" (a property of evidence). Outside readers will
  take "plasticity" in the continual-learning sense, so using it for stickiness would add a third meaning.
- Claims. Stickiness argues that RL dynamics under pressures on reasoning follow reachability, replacing explanations by
  preference (STICKINESS_DEFINITION.md, Thesis). Plasticity argues about what it costs to verify a claim across
  constructions. Each stands on its own evidence; a merged claim would inherit both sets of limits: one coin task and
  Qwen3-8B, against one SFT pipeline, one student and one judge (draft, last section; report F.4).
- Pipelines and models. Porting either project onto the other's model and trainer costs more than any connection above
  would return.
- SGLD and Bayesian susceptibility. E2b cost $313 and tracked clean-model loss (0.929). At a trained student, E2a would
  have needed 29,076 draws. Stickiness's covariance over sampled answers already is the RL update.
- Process. Plasticity runs gates, caps and run contracts (plasticity/AGENTS.md). penalty_shape works by a few fixed-length
  runs with predictions written first (HANDOFF section 5). Keep each project's process.

## Not determined

- Which sense of "plasticity" the request means. The repo's sense is used throughout, and the continual-learning sense
  is covered where it changes the answer.
- Whether /vol/merged/ (penalty_shape) and the secure-code-sft and em-susceptibility volumes (plasticity) still hold the
  checkpoints named above. No Modal calls were made.
- Read in part only: rl_v_sft (the results report's summary and framing sections; the referee files were located, not
  read), the plasticity design documents (probe plane, mixture seeds, subspace) and the 11 Oct protocol beyond its
  opening.
- Costs for tests A-C are estimates from earlier spend, not quotes.
