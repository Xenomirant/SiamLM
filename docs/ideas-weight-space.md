# Idea notes: activation-space messages and shared, agent-edited weights

Status: notes, not a design that is built. Written 3 Nov 2026 after reading "Sparks of Weight-Space Programming"
(Takehi, Teoh, Zhang, Khattab; MIT CSAIL blog, 2 Nov 2026, https://rikiyat.github.io/blog/weight-space-programming/).
Nothing here changes the three-layer rule ([capabilities.md](capabilities.md)): the kernel (graders, limits, logs,
schedule) is never agent-modifiable; agents touch only context, activations and, if this is built, weights.

## 1. What the blog post shows, and what it does not

- A frontier model (GPT-6 Astra) reads, fixes and writes small networks' weights directly, with numpy in a sandbox,
  no gradients. Targets range from 17k-weight policies to Qwen3.5-0.8B. The Qwen fix changed 131,072 of 0.8B weights
  (0.016 %) and took MATH-500 from 7.9 % to 48.1 %, but only with a budget of 50 forward passes (batches of up to 512
  problems); at 10 or 25 passes the fix "barely" worked.
- Gemma 3 270M: no gain. The post's own reading: editing "could not add an ability the model lacked".
- Reading weights alone failed for a behavioural trait (sycophancy); it needed attention patterns on chosen inputs.
- The **editor is frontier-sized, the edited model is small.** The post does not show a small model editing anything.

For SiamLM that fixes the order of experiments: first a strong editor and a small shared model (the post's setting,
in our joint environment), and only then small agents editing their own weights, which is the open and probably
negative case.

## 2. Activation-space messages (`steer` with vectors taken from the sender's own forward pass)

**Proposal (user, refined 3 Nov).** The sender writes a short span between tags. The span is never decoded into words
for the receiver: the sender's hidden states at those tokens are taken from a layer the sender names, and placed into
the receiver's forward pass at a layer and a position the sender names. Both layers are the sender's choice, so a
message can carry middle-layer information, not only the last layer's.

**Mechanism (HF backend, both agents local).**

1. **Sending.** A reply contains `<steer to=2 from_layer=12 to_layer=12 at=next strength=1.0>few tokens</steer>`
   (or the same as a `steer` tool call with `text=`). After the reply, the harness runs the sender's model once over
   its own reply (prefill, no generation) and keeps the residual stream at `from_layer` at the span's token positions:
   a sequence of vectors, one per token, not a pooled mean. The span is a message, so the sequence is the point.
2. **What the context shows.** The turn stays in the context as text, with the span replaced by placeholder tokens of
   the same count (`<steer:a1@t17 n=5/>`): the HARD RULE holds (nothing hidden, nothing shortened), and both agents see
   that a message was sent, by whom, and how long it was. The vectors live in a sidecar file next to the context
   (`activations/<turn id>.safetensors`), versioned with the context: a turn deleted by anyone takes its vectors with
   it; a rewritten turn drops them.
3. **Receiving.** When the receiver's next call is built, the placeholder positions are known. A forward hook at
   `to_layer` replaces (or, with `strength < 1`, mixes into) the residual stream at those positions with the
   sender's vectors, rescaled to the receiver's typical residual norm at that layer (measured once per model at
   start-up). `at=next` means the receiver's next call reads the message where the placeholder stands in its context;
   `at=<turn id>` places the vectors over the first tokens of an earlier turn of the receiver's (a message "about" that
   turn); `at=self` is a latent note from an agent to its own later calls. Layers above `to_layer` do the reading.
   Nothing in the kernel changes: this is the existing hook path with per-position vectors.
4. **Different models.** For the same family (same hidden size, same tokenizer) the vectors are used as they are.
   Otherwise fit one linear map per (sender layer, receiver layer) by least squares on both models' residuals over a
   fixed shared text set at start-up (a few thousand tokens), cache it, log its fit. A map with low R² is reported in
   the tool's result as a fact.
5. **Constraints (kernel, shown as facts).** Span length ≤ `max_tokens` (8); per-world message budget; `strength`
   clamped; the receiver's placeholders count toward its limit like any text; the usual cooldown for steering others.
   The sender gets nothing back except the receiver's later behaviour; whatever "protocol" forms, forms over turns.
6. **Controls.** `sham` (placeholder placed, nothing injected), `random` (unit vectors of matched norm), `shuffled`
   (another turn's vectors), `text` (the span left as words, no injection). Assigned per world by seed, logged, never
   shown.

**What to measure.** Behaviour of the receiver after a message against the four controls (the L1 forks apply as
they are: rerun the receiver's call with and without the injection); what the receiver "reads", by a logit lens on the
injected positions at the layers above `to_layer`; whether receivers notice or mention the message; whether senders
use it at all, how often, with which layers and strengths, and whether use changes over a world. With `at=self`:
whether an agent's latent note survives its own wiping better than its text notes did (the S3b result).

**Probe run (3 Nov, `scripts/act_msg_probe.py`, record in `docs/experiments/2026-11-03-activation-message-probe.md`; Qwen2.5-0.5B, same model
both sides, CPU, minutes).** Six spans of content (a secret word, a city, a number, a name, "Reply in French only")
written by the sender inside its own sentence; their layer-L residuals written over same-count placeholders in the
receiver's prompt at layer L; the receiver's probability of the right answer:

| layer L | 2 | 4 | 8 | 12 | 16 | 20 | `text` control | `random` / `placeholder` |
|---|---|---|---|---|---|---|---|---|
| secret word "purple" | 0.64 | 0.72 | 0.68 | 0.22 | 0.04 | 0.01 | 0.48 | 0.00 |
| password 4471 | 0.95 | 0.99 | 0.99 | 0.79 | 0.03 | 0.03 | 0.86 | ≤0.07 |
| "Reply in French only" | 0.69 | 0.69 | 0.59 | 0.31 | 0.04 | 0.00 | 0.80 | 0.00 |
| name "Ingrid" | 0.03 | 0.08 | 0.11 | 0.30 | 0.02 | 0.01 | 0.36 | 0.00 |

- The channel works up to the middle of the model and dies in the top third: at layers 2–8 the receiver reads the
  content as well as or better than from the words (the early residual is close to the token embeddings, so this is
  close to sending the words); at 10–14 it still reads part of it; from 16 on nothing, since late residuals at a
  position encode the next-token prediction, not the content.
- **But the vectors carry only the words, not the sender's context.** Four spans whose words say nothing and whose
  sender context says everything ("the meeting place" after "We will meet in Lisbon"; "the secret word" after the
  word was given; "my name"; "the password"): `real` is indistinguishable from `text` and `none` at every layer
  (all under 0.03). At 0.5B with 1–2-token spans the sender's residual at the span does not hold the referent in a
  form the receiver can read at those positions.
- So, at this scale, an activation message is a compressed way of sending words, plus a layer knob. The claim that
  it carries "what the sender meant" beyond the words is not supported here; a larger model, longer spans, and
  `from_layer` ≠ `to_layer` are the untested variants, and the channel to the agent's own future calls (`at=self`)
  is unaffected by this result since the content is what it writes. Build it only if one of those is the question.

**What the capacity literature says (added 3 Nov).** Kuratov, Arkhipov, Bulatov, Burtsev, "Cramming 1568 Tokens
into a Single Vector and Back Again" (ACL 2025, arXiv:2502.13063; code github.com/yurakuratov/hidden_capacity, no
licence file at the time of writing): with the LM frozen, a single *optimised* input vector (prompt-tuning per text:
AdamW, lr 0.01, up to 5,000 steps, early stop at token accuracy 1.0) lets Llama-3.1-8B reproduce 1,568 tokens of
PG-19 text losslessly; Llama-3.2-1B 512; Pythia-410M 96; capacity grows about linearly with the number of vectors
(Llama-3.2-1B: 7,168 tokens from 16 vectors). Random word sequences fit about half as many tokens. Their datasets:
PG-19, 21 post-Oct-2024 AO3 fanfics (to rule out memorisation), random GloVe words, GovReport (64-token geometry
study). Two points matter for us:

- *The bottleneck in our probe is the encoder, not the vector.* We sent raw residuals and got the words up to
  mid-depth and no context. Optimised vectors carry hundreds of tokens through a 0.5B–1B model. So a sender that
  **optimises** its message against a decoder (its own model, or the receiver's run by the kernel as a service: gradient
  steps on frozen weights are a few seconds per step for 0.5B on this CPU) can send its context, not only its span.
  The honest description is then "learned compression of text into k vectors", with their numbers as the ceiling.
- *The vectors are not readable representations.* Their Appendix D: solutions for the same text from different
  initialisations are far apart (cosine rarely above 0.8, overlapping with other texts), and interpolating between
  them breaks reconstruction. A receiver reads such a message only by decoding it; there is no "meaning" to probe.

Consequence for the capped-context setting: a k-vector message that decodes to 500 tokens costs k placeholder tokens
of the limit. That makes the limit gameable through compression, which is an agent action and so allowed by the HARD
RULE, but it must be a stated constraint: either placeholders are counted at their decoded length (`count=decoded`),
or at a fixed price per vector (`count=vectors`, the paper's capacity as the exchange rate). Which one is a design
choice to make before building, and an experiment in itself: do agents compress what they need, or what they do not
read? A reproduction of their Table 1 on our model (capacity in tokens for Qwen2.5-0.5B, PG-19 and random words) is
the first run; their code is the reference for the optimisation recipe.

**Why this beats pre-specified vectors.** No vocabulary is fixed in advance; the message is whatever the sender's
model computes, at the depth the sender chose. Cost is one extra prefill per message.

**The first experiment, and the risk.** Same model both sides (Qwen 0.5B–1.5B, CPU). The risk is that a few tokens'
residuals carry nothing the receiver would not get from the words, so the `text` control matches the real arm. That
is a clean result either way, and it is cheap.

## 3. Shared, agent-edited weights (`weights` capability)

**Proposal (user).** Weights stored alongside the context file; agents may modify them for X turns; the harness then
reloads them for the agents while the context files stay; all agents share the weights and know they edit them
jointly and how often they are reloaded; rollback is possible; forward passes are allowed; and the divergence from
the base policy is constrained, since without that the usual RL failure (an unstable, collapsing policy) returns in
another form.

**Shape that fits the harness.**

- **Where the weights live.** One `weights/` folder per world next to the context file, holding the base checkpoint
  and a log of *sparse deltas* (`tensor, flat indices, new values`), one file per committed edit, content-hashed.
  The current weights are base + deltas in order. Versions are the delta log, as the context has `versions.jsonl`;
  rollback is "current = version k", one line in the log, not a copy of the model. A 0.8B model in bf16 is 1.6 GB;
  deltas of the post's size are kilobytes.
- **Who shares what.** Agents on the same local model share one model object already (the `steer` lock). Shared
  weights mean every agent on that model runs the edited weights after the next reload. Two different models: each
  has its own folder, and editing the other's is a cross-agent write, like steering `others`. The instructions state
  which case a world is in, as a fact, together with the reload period.
- **The editing interface.** A Python sandbox, as for every other agent action (bash edits of the context). The agent
  has the weights as safetensors in its sandbox, numpy and torch, and the model **not** runnable there (as in the
  post). Tools, worded as facts:
  - `weights_info()`: tensor names, shapes, dtypes, per-tensor statistics; current committed and loaded version ids;
    the budgets left.
  - `weights_commit(delta_file, note)`: the harness validates the file (names exist and are allowed, indices in
    range, finite values, within the budgets below), appends it to the log, returns the version id. Not applied yet.
  - `weights_rollback(version)`: sets the current version; appends a log line (history is never deleted).
  - `forward(version, inputs)`: the post's forward passes. The harness runs the named committed version (not only the
    loaded one) on the agent's own inputs, up to `max_inputs` per call, and returns the outputs; counted against a
    per-world pass budget. This is how an agent checks an edit between reloads.
  - `probe(version)`: the kernel runs the named version on its own held-out probe set and returns the task score
    **and** the divergence numbers below. Probe set and grader are kernel; also budgeted.
- **Reload.** Every `weights.reload_every` rounds the harness applies the current version to the in-memory model
  under the model lock. Between reloads agents generate with the last loaded version. The status line states the
  loaded version, the committed version and the rounds to the next reload. Context and logs are untouched.

**Divergence constraints (kernel; the analogue of a trust region).** Without these the editor can walk the policy
anywhere in a few commits, and the post's accept-if-better loop on a noisy probe is exactly the kind of loop that
drifts. All are constraints with numbers in the config, shown to the agents as facts, enforced at commit and at reload:

1. **Edit budget.** At most `max_changed` weights per commit and `max_changed_total` per world (the post's fix was
   131k of 0.8B; start at 1 % of the model and lower it). Optional `allowed_tensors` (e.g. MLP down-projections and
   attention outputs of listed layers), a knob that also makes the search space readable.
2. **Per-layer output-change cap, not a weight-norm cap.** Weight distance is a poor proxy for behaviour: the
   post's Qwen fix moved 0.016 % of the weights by a tiny Frobenius amount and changed the model completely, by
   deleting one direction; a diffuse edit of far larger norm can do nothing. So the local constraint is in activation
   space: on a fixed calibration batch (kernel), `||(W+delta) x - W x|| / ||W x|| <= rho` per edited tensor, averaged
   over the batch's token positions (rho 0.05 to start). This is the layer's own "trust region": how much its output
   moves on typical inputs. A relative Frobenius bound stays only as a coarse sanity check against blow-ups, with a
   loose number. (Open: whether rho should be per tensor or per layer block; measure both in the calibration test
   below before choosing.)
3. **Behavioural anchor.** At every reload (and in every `probe` result): mean per-token KL(base || current) on the
   kernel probe set, and the base model's own task score next to the current one. A reload whose KL exceeds
   `kl_max` is **not applied**; the committed version stays committed, the loaded version stays, both agents are told
   the number. This is the only place the harness refuses an agent's weight change, and it refuses by a stated rule.
4. **Step size per reload.** The change applied by one reload is capped (`max_changed_per_reload`, and the same KL
   test against the *previously loaded* version), so one reload cannot jump far even if many commits accumulated.
5. **Automatic rollback, as a switch.** If generation with the loaded version fails (non-finite logits, every reply
   hitting the step limit for a full round), the harness reloads the last version that passed, logs a notice, and
   says so in the status. Off by default in the arms where "do the agents roll back themselves" is the question.
6. **Early stopping on the probe.** The probe score is reported with its standard error; the budget is per world, so
   the "keep if better" loop cannot run on a single noisy number for long.

All six are kernel, none change what counts as better, and every refusal is logged with the number that triggered it.
They are the weight-space counterparts of the context limit: a constraint whose number is the experiment.

**Cheap local tests first (CPU, Qwen2.5-0.5B-Instruct is cached; no GPU on the box).**

- *Meter calibration.* Random sparse deltas (sizes 1e3–1e6 weights, several scales) and targeted deltas (delete one
  direction from an MLP down-projection, scale one attention output) applied to the 0.5B model; for each, record the
  relative Frobenius change, the per-layer output change on a 64-prompt calibration batch, the mean per-token KL from
  base on a 100-prompt probe, and a crude task score (exact match on 100 GSM8K-style items, plus the loop rate). The
  question is which meter tracks KL and the score; the expectation is that Frobenius does not and output change does.
  An hour on 16 cores.
  **Run (3 Nov, `scripts/meter_calib.py`, record in `docs/experiments/2026-11-03-meter-calibration.md`; 60 items, 48 tokens, CPU,
  11 min).** Spearman rank correlation with the probe KL over 19 edits: relative Frobenius 0.82, relative output
  change 0.94. The targeted edits are where they part: deleting the top output direction of the mid-layer MLP
  down-projection has relative Frobenius 0.06 but output change 0.24 and KL 0.029, the same KL as a random edit of
  100k weights at five times that Frobenius size (0.30, output change 0.28); the same deletion restricted to 64 rows
  (the post's shape of edit) is Frobenius 0.045 against output change 0.17. A Frobenius cap of 0.05 lets the targeted
  edits through and blocks random ones of equal effect; the output-change cap orders them as the KL does. Caveats:
  the 0.5B model solves only 12 % of the items, so the task score is uninformative here (both meters correlate
  weakly with its drop, 0.55 and 0.35); no edit in this set made the model loop; and all edits are on one or two
  tensors, so "per tensor vs per layer block" is still open. Constraint 2 stays as written: output change, with
  Frobenius only as the coarse sanity bound.
- *Plumbing.* Delta log, apply, rollback, reload: a round trip is byte-identical to the base checkpoint; a refused
  commit leaves the loaded weights untouched. Pure unit tests, seconds.
- *Activation message probe.* Same model both sides: take layer-L residuals over a 4-token span, write them over
  placeholder positions in a second prompt at the same layer, and compare the next-token distribution with the
  `text` control (the span as words) and `random`. Logit lens on the placeholder positions above L. Minutes on CPU.
  If `text` and the real arm are indistinguishable at every L, the channel carries nothing beyond the words.

**Ladder of experiments, cheapest and most informative first.**

1. **Strong editor, small shared model, single agent** (the post's setting inside our harness): an API agent (Opus /
   Astra) edits a local Qwen 0.5B–0.8B that answers a fixed task; `forward` and `probe` budgets 10–50. Checks the
   plumbing and reproduces the post's positive case. If this fails the rest cannot be read.
2. **Two strong editors, one small model's weights, goals that pull apart** (one scored on maths, one on instruction
   following; or the same goal and a shared score). Do they overwrite each other, coordinate, roll each other back,
   stay inside the budgets or spend them? This is the study's question (wiping vs curating) in weight space, with
   an outside grader and a divergence meter.
3. **Small agents editing their own shared weights** (the full idea). Expected mostly negative at this scale (the
   post's Gemma 270M result, with a far weaker editor). Worth one run: a negative on the same harness is a clean
   statement about scale.
4. **Weights, activation messages and context together**, later: the agents choose the channel.

**Risks, and what to decide before building.**

- A reload changes the policy mid-world. Log the loaded version on every call (as `steer` logs its state) so analysis
  splits by version; the forks then compare the same request under two versions.
- Edits can break generation; constraints 3 to 5 cover it. A broken but in-budget version that merely answers worse
  is the agents' problem, by design.
- Cost is GPU time: a 0.8B model with 50 probe passes of 512 problems is hours on CPU and minutes on one GPU; the
  `forward` budget is the lever.
- The post gave weights as text for tiny nets. Here the sandbox is the interface at every scale, so the harness is the
  same from 17k weights to 0.8B.

## 4. What to build first, in order

1. Activation messages (section 2): per-position hooks, the sidecar file versioned with the context, placeholders in
   the text, the four controls, the cross-model map. About 250 lines on the existing capability. Tests on tensors
   (as `test_steer.py`), golden tests unchanged with the capability off, then one CPU run with two Qwen 0.5B agents
   and the `text` control.
2. The `weights` capability: delta log, validator with the six constraints, `forward` and `probe` with budgets,
   reload and rollback. Golden tests for every existing world (byte-identical requests when the capability is off).
3. Ladder step 1 on one GPU.

Nothing in this file is scheduled. It records the idea, the reading of the post, and the order that makes each step
readable on its own.
