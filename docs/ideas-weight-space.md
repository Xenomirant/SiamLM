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
