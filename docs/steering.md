# The `steer` capability (experimental)

`steer` is a SiamLM tool that lets an agent on a **local model** add a steering vector to its own later generations.
It is **off by default**, and nothing about an agent without it changes. It replaces the former `provider=modulo`,
which ran Modulo's own prompt and generation loop and has been removed. SiamLM's prompts and tool flow stay in
charge.

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU torch first (omit the index for CUDA)
uv pip install -e ".[steer]"                                              # transformers + Modulo (pinned)
uv run python -m siamlm.run provider=hf provider.model=Qwen/Qwen2.5-0.5B-Instruct provider.max_tokens=200 \
    provider.steer.enabled=true task.name=open world.setup=external world.identity=names world.rounds=10
# one agent only (agent 2 is the control):
#   'world.agent_providers=[{steer: {enabled: true}}, {steer: {enabled: false}}]'
```

## How it works

- **Vectors.** [Modulo](https://github.com/EigenformAI/modulo) (MIT; the `steer` extra pins commit `e481fbf`) is used
  as a library only: its registry layout (`<steer.vectors_dir>/<model>/registry.yaml`) and its extraction
  (`extract_vector`, contrastive system prompts from its starter set). A concept in `steer.concepts` that the registry
  lacks is extracted once, at the middle layer by default, and saved. Nothing is downloaded: Modulo's community
  index is not used.
- **The call.** `steer(concept, strength)`: `strength` is clamped to ±`steer.max_strength`, 0 stops a concept, and at
  most `steer.max_active` concepts are active at once. The world treats it like an editing call: it does not count
  as a work call, and the step goes on.
- **The effect.** From the agent's **next** generation on, the backend adds `strength / 15 × (the layer's typical
  activation norm) × the concept's unit vector` to that decoder layer's output at every generated position. The
  hooks are attached only while that agent's backend generates and removed right after, also after an error. Agents
  that share one model's weights take turns under a lock, so one agent's steering never touches another's
  generation. The generation that made the call is not steered.
- **Decay and limits.** After each generation of that agent, every active strength is multiplied by `steer.decay`
  (0.85). Below `steer.floor` (0.5) the concept is removed. With the defaults, a push of 10 is gone after about 19
  replies, and half of it after 4 or 5. The exit question (`world.exit_question`) is a generation too, so it is steered
  with whatever is still active.

## Consequences, plainly

- **What changes in the prompt.** For a steering agent only: one more tool schema (`steer`, its description listing
  the concepts with their descriptions and sign conventions) and one more line in the instructions' tool list. Under
  `ctx.count=whole` both are counted against the limit. Agents without the capability are sent byte-identical
  requests.
- **Who sees steering calls.** Everyone who reads the context. The call (the agent's words, `[steer]` and its
  arguments) and its result are written into the shared context like any call, so partners see who steered with what,
  and can remove those turns, like any turns. What a generation was actually steered with is logged only
  (`steer.jsonl`, `event: generation`) and is never shown to any agent.
- **The partner is affected indirectly.** Only the caller's generations are steered, but its steered replies enter
  the shared context, which the partner then reads and answers.
- **Self-reports need controls.** A model that says it "feels more curious" after a steer call may be answering the
  call's text, not the vector. Run `steer.control=sham` (the same tool, the same results, nothing added) and
  `steer.control=random` (a fixed random direction of the same size per concept) as controls. The model is told the
  same in every condition, and `steer.jsonl` records the condition. Compare behaviour, not only words.
- **Only local models.** Steering needs the weights: `provider=hf` only. Any other backend with `steer.enabled=true`
  is refused at start-up. Results on small CPU models (0.5B–1.5B) say little about large API models.
- **No validation that a vector does what its name says.** Modulo's starter vectors come from contrastive system
  prompts; check a vector's effect on your model before reading meaning into a run.
- **Strong pushes break text.** Above about 10–12 (Modulo reports breakdown near 12–13 on 7–8B models), replies
  degrade into repetition or noise. The harness does not detect or regenerate degenerate output; it is written into
  the shared context like any reply.
