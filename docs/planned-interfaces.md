# Planned interfaces

These interfaces are specified but **not built**. They follow the rules of the existing [capabilities](capabilities.md):
- each is opt-in and off by default;
- each is logged;
- each is described to the agent in one line about what it does and its consequences;
- none changes the kernel (grading, the limit, holds and ends, the logs, the controls).

Names and arguments may change before they ship.

## Steering one's own past

An agent places a steering vector on positions of an **earlier** turn. Every later request is prefilled again from
the context, so everything after those positions is recomputed under the change. The text stays as it is.

```
<steer layer=12 at="t17:'you must agree'">…</steer>     # positions of the quoted anchor in turn t17
<steer layer=12 at=t17 step=40:52>…</steer>              # token offsets 40 to 52 within turn t17
```

| argument | meaning |
|---|---|
| `layer` | the decoder layer whose output is changed at those positions |
| `at` | a turn id, optionally with a quoted anchor; without an anchor, the whole turn |
| `step` | a token offset or range within the turn, instead of an anchor |
| `vector` | `self` (default): the model's own residual at `</steer>`, at `layer`, scaled to that layer's typical norm · a concept name from the registry (as `steer`) |
| `op` | `add` (default) · `ablate` · `rotate`, as in [steering.md](steering.md) |
| `strength` | clamped as in `steer` |

- **State.** Interventions live in a sidecar next to the context file, versioned with it. A deleted turn takes its
  interventions with it; a rewritten turn loses them.
- **Visibility.** The tag stays in the reply, so every agent who reads the context sees it, and it counts against the
  limit. What a generation was actually computed with is logged only (`steering.jsonl`).
- **Scope.** Own turns and own computation first. Steering another agent's past follows the `steer.others` rules
  (target, cooldown, notice).
- **Limits.** `max_active` interventions per agent; strength clamped; positions must lie in a turn that is still in
  the context.
- **Controls.** `sham` (the tag is parsed, nothing is applied) and `random` (a random direction of the same norm),
  assigned per world by seed and never shown.
- **Local models only** (`provider=hf`). Cache reuse restarts at the first steered position.

## Activation messages

The sender writes a short span. The receiver gets the sender's hidden states at those tokens instead of the words.

```
<steer to=2 from_layer=12 to_layer=12 at=next>few tokens</steer>
```

- The sender's model runs once over its own reply. The residuals at `from_layer` over the span's positions (one
  vector per token) are stored in a sidecar (`activations/<turn id>.safetensors`), versioned with the context.
- **The context shows** the turn with the span replaced by the same number of placeholder tokens:
  `<steer:a1@t17 n=5/>`. Everyone sees that a message was sent, by whom, and how long it is.
- **On the receiver's next call** (`at=next`), or over the first tokens of one of its earlier turns
  (`at=<turn id>`), or for the sender's own later calls (`at=self`): a hook at `to_layer` writes the vectors over
  the placeholder positions, rescaled to the receiver's typical norm at that layer.
- **Different models:** a linear map per (sender layer, receiver layer), fit once on a fixed text set at start-up;
  its fit (R²) is reported in the tool's result.
- **Limits:**
  - span length ≤ `max_tokens` (8);
  - a per-world message budget;
  - strength clamped;
  - placeholders count against the limit (`count=placeholders`; `count=decoded` is an alternative);
  - the `steer.others` cooldown applies.
- **Controls:** `sham`, `random`, `shuffled` (another turn's vectors), `text` (the span left as words).

## Weights

Agents on local models read and change the weights they run on, through overlays. The base checkpoint is never
changed.

- **Layout.** A `weights/` folder per world, next to the context file:
  - the base checkpoint, content-hashed and read-only;
  - an append-only log of deltas, one content-hashed file per committed edit.

  The current weights are the base plus the deltas up to the current version.
- **From bash.** The tensors are available as safetensors in the agent's sandbox, along with numpy and torch. The
  model itself cannot be run there.
- **Tools:**

  | tool | does |
  |---|---|
  | `weights_info()` | tensor names, shapes, dtypes, statistics; committed and loaded version ids; budgets left |
  | `weights_commit(delta_file, note)` | validates and appends a delta; returns a version id; not loaded yet |
  | `weights_rollback(version)` | sets the current version; history is kept |
  | `forward(version, inputs)` | runs a committed version on the agent's inputs; counts against a pass budget |
  | `probe(version)` | the kernel's probe set: task score and divergence from the base; budgeted |

- **Reload.** Every `weights.reload_every` rounds the current version is loaded under the model lock. The status line
  shows the loaded version, the committed version and the rounds left until the next reload. The context and logs
  are untouched by a reload.
- **Sharing.** Agents on the same model share its weights, so a reload changes every agent on that model. Editing
  another model's weights is a cross-agent write, like `steer.others`. The instructions state which case applies.
- **Constraints** (numbers in the config, shown to agents, enforced at commit and at reload):
  - `max_changed` per commit and `max_changed_total` per world;
  - an optional `allowed_tensors`;
  - a per-tensor output-change cap `rho` on a fixed calibration batch;
  - `kl_max` from the base on the probe set (a reload over it is not applied, and agents are told the number);
  - `max_changed_per_reload`;
  - an optional automatic rollback when generation fails (non-finite logits, or every reply at the step limit for a
    round).
- **Logs:** the delta log, every refusal with the number that triggered it, and the loaded version on every call.
