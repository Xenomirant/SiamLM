# Meter calibration for agent-edited weights (3 Nov 2026)

**Question.** For the `weights` capability ([ideas-weight-space.md](../ideas-weight-space.md), section 3, constraint 2):
which cheap meter of a weight edit tracks the edit's effect on behaviour, the relative Frobenius change of the edited
tensor or the tensor's relative output change on a calibration batch? The effect is measured as KL from the base
model and as the task score.

**Hypothesis.** Weight distance does not track effect: a targeted low-rank edit (deleting one output direction, as in
the Weight-Space Programming post) has a small Frobenius norm and a large effect; a random edit of equal norm has a
small effect. Output change should order edits as KL does.

**Hardware and software.** AMD Ryzen 7 9700X, 16 threads, 62 GB RAM, no GPU. Python 3.12.3, torch 2.15.0+cpu,
transformers 4.57.6, numpy 2.5.4, tokenizers 0.22.2, safetensors 0.8.0. Model `Qwen/Qwen2.5-0.5B-Instruct`, snapshot
`7ae557604adf67be50417f59c2c2f167def9a775`, float32, eval mode, greedy decoding.

**Command.**
```
uv run python scripts/meter_calib.py docs/experiments/2026-11-03-meter-calibration.json 60 48
```
(60 task items, 48 new tokens; the smoke setting `4 8` runs in two minutes.) Runtime of the recorded run: 11 min.

**Inputs.**
- Task items: 60 arithmetic word problems from the generator in the script (`random.Random(7)`; three templates:
  add-then-subtract, integer division, "c times as many more"), each ending "Answer with the number only.", in the
  model's chat template. The items are in the JSON (`items`).
- Probe set (KL): the first 40 task items plus 10 fixed general prompts repeated twice (60 prompts; `probe` in the
  JSON). Continuations: the base model's own greedy generations (up to 48 tokens) on each probe prompt.
- Calibration batch (output change): the first 32 probe prompts (`calib`); the inputs to the edited tensors are
  captured by forward hooks on the base model.
- Edited tensors: `model.layers[12].mlp.down_proj.weight` ("down"; layer 12 of 24), `model.layers[12].self_attn.o_proj.weight`
  ("o"), `model.layers[21].mlp.down_proj.weight` ("down_late").
- Edits (20 conditions, in the script's `CONDS`): random sparse deltas on "down" (n ∈ {1e3, 1e4, 1e5, 1e6} weights,
  value `scale × std(W) × N(0,1)`, scale ∈ {0.5, 2}; seeded by n and scale); one random on "o" (1e5, ×2); deletion of
  the top principal direction of the tensor's output on the calibration batch (`W ← (I − uuᵀ)W`), on all rows or on the
  64 rows with the largest |u|, for "down", "o" and "down_late"; scaling 64 rows of "down" by 0.5 and by 0; zeroing
  the 1e4 and 1e5 largest-|w| entries of "down".

**Metrics.**
- `rel_frob` = ‖Δ‖_F / ‖W‖_F.
- `out_change` = mean over calibration positions x of ‖Δᵀx‖ / ‖Wᵀx‖ (the tensor's relative output change).
- `kl` = mean per-token KL(base ‖ edited) over the probe continuations, teacher-forced, full vocabulary.
- `acc` = exact match of the last integer in the generation with the item's answer; `loop` = share of generations
  that hit the 48-token limit.
- Rank agreement: Spearman correlation across the 19 non-base conditions.

**Result.** Spearman with KL: `rel_frob` 0.82, `out_change` 0.94. With the accuracy drop: 0.55 and 0.35 (uninformative:
base accuracy is 0.12). Loop rate 0 in every condition.

| condition | changed | rel_frob | out_change | KL | acc |
|---|---|---|---|---|---|
| base | 0 | 0 | 0 | 0 | 0.12 |
| random down n=1e5 ×2 | 100,000 | 0.303 | 0.276 | 0.029 | 0.08 |
| random down n=1e6 ×2 | 1,000,000 | 0.959 | 0.877 | 0.192 | 0.05 |
| delete top direction, down, all rows | 4,358,144 | 0.063 | 0.240 | 0.029 | 0.13 |
| delete top direction, down, 64 rows | 311,296 | 0.045 | 0.171 | 0.008 | 0.13 |
| delete top direction, o, all rows | 802,816 | 0.055 | 0.337 | 0.010 | 0.13 |
| delete top direction, down layer 21 | 4,358,144 | 0.048 | 0.061 | 0.007 | 0.12 |
| zero top-1e4 \|w\| of down | 10,000 | 0.218 | 0.341 | 0.027 | 0.12 |
| zero top-1e5 \|w\| of down | 100,000 | 0.459 | 0.534 | 0.075 | 0.07 |

The full 20 rows are in the JSON. The targeted deletions sit at Frobenius 0.045–0.063 with KL 0.008–0.029; the random
edit with the same KL (0.029) has five times the Frobenius norm.

**Caveats.** One model, one size, one or two tensors per edit; the task score is near floor; no edit produced loops;
KL is on the base model's own continuations (a mild bias toward the base). A rerun should reproduce the table to
three decimals in `rel_frob` and `out_change` (deterministic) and to about ±0.005 in KL (float32 CPU reductions); the
Spearman ordering should hold. The recorded JSON predates the scripts' provenance block; its values were entered by
hand from the session and the inputs are as listed here.
