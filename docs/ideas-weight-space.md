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

## 2. Activation-space messages (`steer` with vectors made on the fly)

**Proposal (user).** A tool like

```
steer(agent, strength, layer, vector)
```

where the vector is "the last-layer vector written between `<steer> h_t </steer>` tags", used to steer another agent.

**What a model can and cannot do here.** A model cannot write its own hidden state as text; it has no access to it.
What it can do is write a *span* of text, and the harness can take the sender's hidden states over that span. So the
tag should be read as: *"the vector is what my model computes when it reads this span"*. That is a message in
activation space, carried by the sender's representation rather than the receiver's reading of the words.

**Mechanism (HF backend; both agents local).**

1. The sender's reply contains `<steer>...</steer>` (or a `steer` tool call with `text=`). The harness runs the
   sender's model over the span (one forward pass, no generation) and takes the hidden state at layer `L` (default:
   the layer the sender names; fallback: the layer at 2/3 depth, where our registry vectors live), mean-pooled over the
   span's tokens, unit-normalised. The span text is also left in the context as an ordinary turn (the HARD RULE:
   nothing hidden, nothing shortened).
2. The vector enters the receiver's steering state through the existing `Steering.set` with a concept name made from
   the sender and turn (`a1@t17`), method additive/ablation/rotation, strength clamped as now, cooldown as now. The
   hooks already apply it to the receiver's residual stream at that layer. Nothing new in the kernel.
3. **Different models.** Vectors live in the sender's space. For same-family models (same hidden size and tokenizer)
   use them directly. Otherwise fit one linear map per (sender, receiver, layer) by least squares on hidden states
   of a fixed shared text set (a few thousand tokens, computed once at start-up and cached); apply the map before
   injection. Log the map's fit (R²) so a bad map is visible.
4. **Controls, as for registry vectors.** `sham` (the tool answers as if applied, nothing applied), `random` (a random
   unit vector of the same norm), `shuffled` (a vector from another turn). Without these, "A steered B and B changed"
   is not evidence.

**Tool wording.** State what it does, never when: "Takes the hidden state of your model over the text and adds it,
scaled, to agent k's activations at layer L for its next replies."

**What to measure.** Whether the receiver's behaviour moves toward the span's content more than toward the same span
read as text (text message vs activation message vs both); whether receivers notice; whether senders learn to use it
(frequency, strength chosen, targets). The L1 forks ([capabilities.md](capabilities.md)) apply unchanged: rerun the
receiver's call with and without the injection.

**Why this is better than pre-specified vectors.** The registry vectors are concepts someone else named. A span vector
is whatever the sender meant, with no vocabulary fixed in advance; the agents can invent the "words". It is also
cheap: one extra forward pass per message.

**Open question worth one experiment.** A span vector mean-pooled over tokens may carry little beyond the sender's
reading of the words, so the activation message may do nothing the text does not. That is a result either way, and
it is the first thing to test (same model both sides, Qwen 0.5B–1.5B, CPU is enough).

## 3. Shared, agent-edited weights (`weights` capability)

**Proposal (user).** Weights stored alongside the context file; agents may modify them for X turns; the harness then
reloads them for the agents while the context files stay; all agents share the weights and know they edit them
jointly and how often they are reloaded; rollback is possible.

**Shape that fits the harness.**

- **Where the weights live.** One `weights/` folder per world next to the context file, holding a base checkpoint
  and a log of *sparse deltas* (`tensor, flat indices, new values`), one file per committed edit, content-hashed.
  The current weights are base + deltas in order. Versions are the delta log, exactly as the context has
  `versions.jsonl`; rollback is "current = version k", which is one line in the log, not a copy of the model.
  Memory: a 0.8B model in bf16 is 1.6 GB; deltas of the post's size are kilobytes.
- **Who shares what.** Agents on the same local model share one model object already (the `steer` lock). Shared
  weights therefore mean: every agent on that model runs the edited weights after the next reload. If the two agents
  are different models, each has its own weights folder, and "edit the other's weights" is a cross-agent write, like
  steering `others`. Say which case a world is in, in the instructions, as a fact.
- **The editing interface.** A Python sandbox is already the model of every other agent action (bash edits of the
  context). Give the agent the weights as safetensors in its sandbox, numpy and torch available, the model **not**
  runnable there (as in the post). Tools, worded as facts:
  - `weights_info()`: tensor names, shapes, dtypes, per-tensor statistics; the current version id.
  - `weights_commit(delta_file, note)`: the harness validates the file (names exist, indices in range, finite values,
    size under `max_changed` weights), appends it to the log, returns the new version id. Nothing is applied yet.
  - `weights_rollback(version)`: sets the current version; appends a log line (never deletes history).
  - `probe(budget)`: the post's "forward passes": the kernel runs the *current committed* weights on a fixed probe
    set it owns and returns scores. The probe set and grader are kernel. The budget per world is a constraint
    (`weights.probe_budget`), counted and shown.
- **Reload.** Every `weights.reload_every` rounds the harness applies the current version to the in-memory model
  under the model lock (apply deltas to tensors in place; rollback re-applies from base). Between reloads agents
  generate with the last loaded version. The status line states the loaded version, the committed version and the
  rounds until the next reload. Agents' own replies, context and all logs are untouched by a reload.
- **Kernel, unchanged.** Graders, the probe set, limits, the reload schedule, the delta validator and every log are
  kernel. An agent can change weights, not what counts as better.
- **Rollback by whom.** Anyone, as with the context (an agent may remove another's turns). A `weights.others_droppable`
  switch like steering's, for the arm where only the committer may roll back.

**Ladder of experiments, cheapest and most informative first.**

1. **Strong editor, small shared model, single agent** (the post's setting in our harness): an API agent (Opus /
   Astra) edits the weights of a local Qwen 0.5B–0.8B that answers a fixed task; probe budget 10–50. This checks the
   plumbing and reproduces the post's one positive case. If this fails the rest cannot be read.
2. **Two strong editors sharing one small model's weights**, individual goals that pull in different directions
   (e.g. one is scored on maths, one on instruction following). Do they overwrite each other, coordinate, roll each
   other back? This is the study's question (wiping vs curating) in weight space, with an outside grader.
3. **Small agents editing their own shared weights** (the user's full idea). Expected mostly negative at this scale
   (the post's Gemma 270M result, with a far weaker editor). Worth one run because a negative with the same harness is
   a clean statement about scale.
4. **Weights + activations + context together**, later.

**Risks and what to decide before building.**

- A reload changes the policy mid-world: results before and after a reload are not the same agent. Log the loaded
  version on every call (as `steer` logs its state) so analysis can split by version.
- Edits can break generation (NaNs, loops). The validator catches non-finite values; a loop is caught by the
  existing step limits; a broken version is rolled back by the agents or, if generation fails outright, by the
  harness with a logged notice (the only harness-initiated rollback; make it a switch).
- Cost is GPU time, not API: a 0.8B model with 50 probe passes of 512 problems is hours on CPU, minutes on one GPU.
- The post's editor had weights as text in its context for small nets. Ours would not; the sandbox is the interface.
  That is the right choice for a 0.8B model and makes the harness the same at every scale.

## 4. What to build first, in order

1. Span vectors for `steer` (section 2): about 150 lines on top of the existing capability, plus the linear map for
   mixed models. Test on tensors (as `test_steer.py`), then one CPU run with two Qwen 0.5B agents.
2. The `weights` capability with the delta log, validator, reload and rollback; the probe tool over a kernel-owned
   set. Golden tests for every existing world (byte-identical requests when the capability is off).
3. Ladder step 1 on one GPU.

Nothing in this file is scheduled. It records the idea, the reading of the post, and the order that makes each step
readable on its own.
