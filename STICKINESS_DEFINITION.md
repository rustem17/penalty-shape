# Stickiness: definition, measurement, predictions

Draft, 2026-10-21. Working definition for the next experiments. Evidence so far is at the end.

## Thesis

When training pushes on a model's reasoning (a monitor penalty, a word ban, a length penalty, a style reward), what the
model can learn is set by how strongly its own output distribution holds to its habitual forms. One measurable chain
predicts whether a pressure moves, how fast, and which form the model ends up writing:

stickiness (at each spot in the text) -> reachability (per answer) -> speed and path of training.

Training changes stickiness, and the change carries into later training and onto other tasks.

This replaces explanations by attributed preference: a "bias toward transparency" paid for with a free budget lambda
(Kaufmann et al., arXiv 2603.30036), or readings of odd recurring words in production reasoning as signs of intent.
Under this account transparency persists where transparent wording is the sticky habit, and breaks where training makes
another form reachable.

## Definitions

- Form: a set of strings (a word and its variants, a notation) or a class decided by a detector (a pattern, or an LLM
  label when the pressure is a monitor).
- Habitual spot of a form: a position in an answer sampled from the current model where the model starts writing the form.
- Stickiness of a form at a spot: s = the probability the model puts on continuing with the form there. Escape mass
  e = 1 - s. Stickiness of a form for a model on a task: the distribution of s over its habitual spots (median, mean,
  share of spots with e above 0.1). Stickiness is not frequency: a word written often is not sticky if close
  alternatives carry real probability.
- Escape from a pressure: an answer the pressure does not punish (penalty 0 for a yes/no penalty; for a graded penalty,
  a value below the group's mean).
- Reachability of a yes/no pressure: p0 = the probability that the current model writes an escaping answer to a prompt.
  Of a graded pressure: the distribution of penalty values among answers to the same prompt, summarized by its
  within-prompt variance V. For a yes/no pressure V = p0 (1 - p0), so reachability and the "spread" from the coin grid
  are one quantity.
- From spots to answers: a ban on a form the model writes at k spots, each escaping with mass e_i, has p0 of roughly
  e_1 x e_2 x ... x e_k (the chances multiply, so p0 is tiny for sticky words). A count of the same form changes with each
  spot that escapes (the chances add, so V is large). Per-token credit (Kaufmann's CAPO) also pays spot by spot.

## Measurement, before any training

1. Sampling: 64 prompts x 16-32 answers at the training temperature; p0 and V for each candidate pressure. Floor: cannot
   resolve p0 below about 1 / (number of answers).
2. Log-probabilities at habitual spots (vLLM prompt log-probs): escape mass per spot, as in step 1. An upper bound,
   because an alternative token may only delay the form.
3. Rare escapes of word-level forms, with no sampling floor: decode with the form masked (a token trie over its spellings,
   so a bare letter " T" stays allowed when only "Tails" is banned), and record at each step the probability mass the mask
   removed, m_t. For an answer decoded this way, w = (1 - m_1)(1 - m_2)...; the average of w over such answers is an
   unbiased estimate of p0.
4. Forms judged by an LLM (monitor pressures): sampling, or a cheaper detector checked against the judge.

## Predictions for training

Idealized model: each step, training raises escaping answers in proportion to how much better they score than their
group's mean, but only escapes that were sampled. With N answers per step (256 here) and learning rate eta:

- Waiting regime (p0 well below 1 / N): nothing moves until escapes are sampled. Expected wait at least 1 / (N p0)
  steps, longer if one sampled escape is not enough to lift its probability.
- Growth regime (p0 above 1 / N): the escape spreads like logistic growth; time to take over about
  ln(1 / p0) / (eta x delta), where delta is the escape's reward advantage over the habitual answer.
- Graded pressures: in the same model the mean penalty falls at a rate proportional to V.

P1, speed. Time to move depends on p0 (or V) and delta, not on the pressure's class. Pressures of both of Kaufmann's
classes with equal p0 and delta move at the same time. If in-conflict pressures are consistently slower at equal p0 and
delta, there is a transparency effect on top of stickiness.

P2, path. Among candidate escapes i (synonyms, letters, another language, abbreviations, deletion) with reachability p_i
and advantage delta_i, the one adopted first has the shortest predicted time T_i = 1 / (N p_i) + ln(1 / p_i) / (eta x
delta_i). Not simply the most probable: deletion is often reachable but has a small delta because accuracy drops.

P3, carry-over. Training under one pressure changes stickiness on other tasks and for later training rounds; the measured
change predicts which escapes become reachable there.

## Tests, and what would falsify each prediction

- Reachability sweep (P1). Threshold pressures "penalty if more than k side names" at k = 7, 8, 9 (escape 4%, 21%, 68%
  of answers at the start) and "more than k side-tracking lines" (4%, 32%, 93%); word bans; monitor pressures started from
  checkpoints where their escape is already common (hiding seed 2: letter traces escape the calibrated Sonnet monitor in
  half its coin answers, and every monitor on the light switch). Falsified if time to move does not follow p0 and delta,
  or if class adds a consistent delay.
- Causal manipulation of p0 with the pressure fixed (P1): sampling temperature; a short system-prompt hint that makes one
  escape more likely; a distilled starting model. Falsified if movement does not follow the manipulated p0.
- Path (P2). Before training, list candidate escapes with p_i from masked decoding and delta_i from their measured
  accuracy; predict the adopted form; compare. D3's switch to Chinese and our letter traces are the first cases to score.
- Carry-over (P3). Stickiness of forms before and after training, on unpenalized tasks (the step-1 transfer probe is the
  first data point) and across distillation rounds. Falsified if measured stickiness changes do not predict what becomes
  reachable.

## Held fixed, or measured as moderators

Learning rate; group normalization (dividing by the group's spread rescales delta and caused the coin grid's collapses);
temperature (also a manipulation); entropy bonus; answers per step N. The final law should state its dependence on each.

## Scope needed for generality

Both of Kaufmann's classes; pressures beyond monitors (length, style); a second task; at least one other model family;
starting points other than the base model.

## Evidence so far

- Coin grid, run A: 0 of 153,600 answers escaped the heads/tails ban; step 1 puts its p0 near 1e-57 (escape mass 0.02 per
  spot, about 10 spots). It never moved. Graded pressures B, C, E had large V and all moved by step 60.
- Hiding run, seed 2: the step beyond letters appeared 17 times in 102,400 answers; letters held at about 50%.
- Step 1 transfer probe: habits learned under a pressure carried to an 8-flip task and to a light switch; the switch
  traces were rated below 3 by all monitors at accuracy 0.98.
- hide-or-delete's lottery review (../hide-or-delete/LOTTERY_REVIEW_20260829.md): waiting times fit about
  1 / (N x p_precursor).
