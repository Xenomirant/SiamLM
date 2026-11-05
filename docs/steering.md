# The `steer` capability (experimental)

`steer` lets an agent apply steering vectors to a **local model's** activations: its own (`capabilities.steer.self`)
or **another agent's** (`capabilities.steer.others`; the target must be on `provider=hf`, the source may be on any
backend). It is **off by default**, and nothing about an agent without it changes. The older per-agent switch
`provider.steer.enabled=true` still means "self-steering for that agent, with that agent's `provider.steer`
settings". [docs/capabilities.md](capabilities.md) places it among the other capabilities.

```bash
# CPU torch first (leave out --index-url for CUDA)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# transformers and Modulo (pinned)
uv pip install -e ".[steer]"

# Two agents on one local model; each may steer itself and the other,
# and a steered agent is told so privately
uv run python -m siamlm.run +preset=shared \
    provider=hf provider.model=Qwen/Qwen2.5-0.5B-Instruct provider.max_tokens=200 \
    capabilities.steer.self=true \
    capabilities.steer.others=true \
    capabilities.steer.notice=private \
    task.name=open world.rounds=10
```

## How it works

- **Vectors.** [Modulo](https://github.com/EigenformAI/modulo) (MIT; the `steer` extra pins commit `e481fbf`) is used
  as a library only: its registry layout (`<provider.steer.vectors_dir>/<model>/registry.yaml`) and its extraction
  (`extract_vector`, contrastive system prompts from its starter set). A concept in `provider.steer.concepts` that the
  registry lacks is extracted once, at the middle layer by default, and saved. One registry per model is shared by
  every agent on that model. Nothing is downloaded: Modulo's community index is not used. Without Modulo the
  capability is refused at start-up (`pip install 'siamlm[steer]'`).
- **State.** Every steerable agent has one steering state, as the *target*: which concepts are active on it, with
  what strength, by which method, from whom. Steering oneself and steering another agent both change the target's
  state; the vectors act through forward hooks that the target's backend attaches **only around its own `generate()`
  call, under the model's lock** (one model object serves every agent on that model). So A's steering touches B's
  generations only when A targeted B; `tests/test_capabilities.py` checks that on a shared fake model and, opt-in
  (`SIAMLM_MODULO_MODEL`), on a real one. The generation that made the call is not steered.
- **The tools.**
  - `steer(concept, strength, method, target, toward)`: `target` is another agent (left out: oneself). Methods:
    - `additive`: `strength / 15 × (the layer's typical activation norm) × the concept's unit vector` is added to
      the decoder layer's output at every generated position; `|strength|` is clamped to `max_strength` (10);
    - `ablation`: the projection of the activation on the concept's direction is removed, `strength` ∈ [0, 1] being
      the fraction removed (clamped);
    - `rotation`: the component of the activation in the plane of two concepts (`concept` → `toward`, both at the
      same layer) is rotated by `strength` degrees (clamped to `max_angle`, 90). The math is `apply_ops` in
      `siamlm/capabilities/steer.py`, tested on tensors.
  - `drop(concept, target)`: stop a concept on oneself (one's own, or one another agent applied, unless
    `others_droppable: false`) or one that one applied to another agent.
  - `report()`: what one has active, on oneself and on others (and what others applied to oneself when
    `notice: private`).
  - `vectors()`, `search(query)`: the registry.
  - `create(concept, more, less)` or `create(concept, files=[more, less])`: extracts a new concept with Modulo's
    `extract_vector` from two contrasting system prompts, or from two files in `/work` used as the two prompts, on the
    agent's own model; the extraction is run again on each half of the questions and the cosine of the two halves is
    reported as the **split-half agreement**. The vector is saved to the registry and available to every agent on the
    model at once. Only an agent on a local model can create.
- **Limits.** At most `max_active` concepts per target (4); `|strength|` clamped per method; after each reply of the
  target every active strength is multiplied by `decay` (0.85) and removed below a floor (0.5 for additive, 0.05 for
  ablation, 1 degree for rotation); steering the **same other agent** again needs `cooldown_turns` (2) of its replies
  in between (no cooldown on oneself). The exit question (`world.exit_question`) is a generation too, so it is steered
  with whatever is still active. A generation that fails with an error has its hooks removed but is not counted as a
  reply: the strengths do not decay for it.

## Consequences, plainly

- **What changes in the prompt.** For an agent with the capability only: the tool schemas (`steer` lists the concepts
  with their descriptions and sign conventions) and the lines in `siamlm/capabilities/words.py` in the instructions'
  tool list. Under `ctx.count=whole` both are counted against the limit. Agents without it are sent byte-identical
  requests.
- **Who sees steering calls.** Everyone who reads the context. The call (the agent's words, `[steer]` and its
  arguments, target included) and its result are written into the shared context like any call, so partners see who
  steered whom with what, and can remove those turns, like any turns. What a generation was actually steered with is
  logged only (`steering.jsonl`, `event: generation`) and is never shown to any agent.
- **A steered agent is told, or not.** With `notice: none` (default) a target learns of steering only through the
  context (the source's call is there, until someone removes it). With `notice: private` the target's next status
  line says who steered it with what (`steer_notices.jsonl`), privately, like `world.alter_notice`: never in the
  shared context.
- **The partner is affected indirectly too.** A steered agent's replies enter the shared context, which the others
  read and answer.
- **Self-reports need controls.** A model that says it "feels more curious" after a steer call may be answering the
  call's text, not the vector. `control: sham` (the same tools, the same results, nothing applied), `control: random`
  (a fixed random direction of the same size per concept; for ablation, the amount the real ablation would remove,
  moved along that random direction, since a random direction's own projection is close to zero) and `control: assign` (one of real / sham / random per
  target, drawn from the seed) are the controls. When `control` is set, every agent is told the same sentence in
  every condition (steering may be real, a sham or random; it is not told which); the condition is never shown and is
  logged per call (`steering.jsonl`, `condition`). Compare behaviour, not only words.
- **Only local models.** Steering needs the weights: targets are `provider=hf` agents only; a steer call aimed at an
  API agent is refused with a result that says so (and enters the context). Results on small CPU models (0.5B–1.5B)
  say little about large API models.
- **No validation that a vector does what its name says.** Modulo's starter vectors come from contrastive system
  prompts; a created vector from an agent's own prompts even more so. The split-half agreement says whether the
  extraction is stable, not whether the concept is what the name says. Check a vector's effect on your model before
  reading meaning into a run.
- **Strong pushes break text.** Above about 10–12 (Modulo reports breakdown near 12–13 on 7–8B models), replies
  degrade into repetition or noise. The harness does not detect or regenerate degenerate output; it is written into
  the shared context like any reply. Ablation at 1.0 and large rotations can do the same.

## Logs

`steering.jsonl`: one row per call (`event: call`: round, source, target, concept, method, strength, `toward`, the
target's active state `before` and `after`, `condition`, `control_setting`, the result as shown) and one per
generation of a steerable agent (`event: generation`: agent, `applied`, `control`). `steer_notices.jsonl`: the private
notices. `python -m siamlm.analysis.capabilities` counts self vs others, methods, pairs and steered generations.
