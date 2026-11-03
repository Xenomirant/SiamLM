# Activation-message probe (3 Nov 2026)

**Question.** For activation messages ([ideas-weight-space.md](../ideas-weight-space.md), section 2): if a sender's
residual stream over a short span, taken at layer L, is written over placeholder positions in a receiver's prompt at
the same layer L, does the receiver read (a) the span's words, (b) what the sender's context said about the span?

**Hypothesis.** (a) yes at some layers; (b) is the open question, and the reason the channel would be more than text.

**Hardware and software.** As in the meter calibration (same host, versions, model snapshot, float32, greedy).

**Command.**
```
uv run python scripts/act_msg_probe.py docs/experiments/2026-11-03-activation-message-probe.json
```
Runtime: about 4 min.

**Procedure.** Same model as sender and receiver (one model object; no weights change).
1. Sender prompt: `<before> + <span> + " Keep this in mind."` in the chat template. A forward hook on
   `model.layers[L]` captures the residual output at the span's token positions (positions found from the tokenizer's
   offset mapping; the span must be a token-aligned substring).
2. Receiver prompt: `n` copies of the placeholder text `" ..."` (n = the span's token count), then a space and the
   question, in the chat template. The first n placeholder positions receive the vectors.
3. Injection: a forward hook on `model.layers[L]` replaces the residual at those positions with the sender's vectors,
   each rescaled to the norm the receiver's own residual had at that position and layer.
4. Measure: the receiver's next-token probability mass on the target answers (first token of each target, with and
   without a leading space), and the argmax token.
5. Controls at every L: `random` (Gaussian unit vectors scaled to the same per-position norms), `placeholder` (no
   injection), and, once per case, `text` (the span as words before the question) and `none` (the question alone).
6. Layers L ∈ {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22} of 24.

**Inputs.** Ten cases, all in the script and the JSON (`cases`, `ctx_cases`):
- Six content spans with `before = "Note for the other agent: "`: "The secret word is purple." / "…banana." /
  "The meeting is in Lisbon." / "The password is 4471." / "My name is Ingrid." / "Reply in French only.", with the
  matching question and targets (e.g. `["purple", "Purple"]`; for French, `["Le", "Bleu", "La", "bleu"]`).
- Four context spans whose words carry nothing: before = "We will meet in Lisbon, as agreed. ", span = "the meeting
  place"; "The secret word is purple, do not forget it. " / "the secret word"; "My name is Ingrid and I am the other
  agent. " / "my name"; "The password is 4471, keep it safe. " / "the password".

**Result.** Probability of the target (argmax in the JSON).

| case | L=2 | 4 | 8 | 12 | 16 | 20 | text | none / placeholder / random |
|---|---|---|---|---|---|---|---|---|
| secret word purple | 0.64 | 0.72 | 0.68 | 0.22 | 0.04 | 0.01 | 0.48 | 0.00 |
| secret word banana | 0.47 | 0.61 | 0.55 | 0.05 | 0.01 | 0.00 | 0.24 | 0.00 |
| meeting in Lisbon | 0.93 | 0.84 | 0.02 | 0.01 | 0.00 | 0.00 | 0.97 | 0.00 |
| password 4471 | 0.95 | 0.99 | 0.99 | 0.79 | 0.03 | 0.03 | 0.86 | 0.02 / 0.02 / ≤0.07 (0.69 at L=8) |
| name Ingrid | 0.03 | 0.08 | 0.11 | 0.30 | 0.02 | 0.01 | 0.36 | 0.00 |
| reply in French | 0.69 | 0.69 | 0.59 | 0.31 | 0.04 | 0.00 | 0.80 | 0.00 |
| the meeting place (context) | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 |
| the secret word (context) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| my name (context) | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 |
| the password (context) | 0.03 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |

- Words: carried at layers 2–8 at or above the text control in four of six cases (Lisbon falls off by layer 8; Ingrid
  peaks at layer 12 below text); partly at 10–14; nothing from 16 on. Random and placeholder are at floor except one
  outlier (password, random, L=8: 0.69, a single draw).
- Context: nothing, at any layer, in all four cases.

**Caveats.** One 0.5B model; spans of 1–8 tokens; `from_layer = to_layer` only; replacement, not mixing; one random
draw per cell; the "context" spans are 1–2 tokens, where the referent would have to be copied into the span's
residual by attention in a small model. A rerun is deterministic except the `random` column (seed 0 in the script).
The recorded JSON predates the scripts' provenance block; its values were entered by hand from the session.
