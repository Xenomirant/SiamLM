# Capabilities

SiamLM has three layers:

- **The kernel** is never changed by an agent at run time: grading, the limit and what counts against it, holds and
  ends, the logs, the sandbox, and the controls (sham, random, assigned conditions).
- **Capabilities** are tools an agent may be given. Each is **opt-in and off by default** (the `capabilities:` block
  of `siamlm/configs/siamlm.yaml`), each is logged, and each may change only two things: a **context** (its own or
  others') or **activations** (its own or others'). There is no capability that changes the harness. Only things an
  agent cannot do for itself are capabilities: a cost rule in the counting (pin), a model call on an uncommitted
  context (fork), activations (steer). Diffing the context, previewing an edit on a copy and keeping private notes
  are not: an agent with bash does them in `/work`, and offering them as tools would prescribe behaviour the study
  measures.
- The world's own tools (`edit_context` or `bash`, the task's work calls) are below both.

An agent without capabilities is sent byte-identical requests to those sent before capabilities existed
(`tests/test_golden.py`). With a capability on, the agent gets its tool schemas and a few lines in its instructions:
the tool and one sentence on consequences (who sees what, what it costs, the limits), and nothing about when, why
or whether to use it. All of that wording is in one module, `siamlm/capabilities/words.py`. Every capability call is written into the shared context like any
tool call (the call with its arguments, then its result), so partners see it, can remove it, and it costs space; the
exceptions are noted below. Capability calls do not count as work calls or editing calls.

| capability | setting | what it changes | log |
|---|---|---|---|
| `pin(turn)`, `unpin(turn)` | `pin`, `pin_cost` | a mark on a turn; the counted size | `pin.jsonl` (calls), `pins.jsonl` (removals) |
| `fork(question, command \| edit)` | `fork`, `fork_per_step`, `fork_samples`, `fork_result_chars` | nothing (one extra model call) | `fork.jsonl`, `requests/fork_*.json` |
| `steer`, `drop`, `report`, `vectors`, `search`, `create` | `steer.*` | activations of a local model: one's own or another agent's | `steering.jsonl`, `steer_notices.jsonl` |

Analysis: `python -m siamlm.analysis.capabilities <world or batch>` summarises the use of every capability per world
(`analysis/capabilities.json`). Export (`python -m siamlm.export`) carries capability calls as tool calls like any
other, and the tool set each call was actually offered (`requests/tools_sets_a{k}.json`, `calls.jsonl` `tools_hash`).

None of these has been measured yet: the constraints table in the README marks them "not measured".

## pin / unpin

**What it does.** Marks a turn as pinned by the agent. Everyone sees the mark: `pinned=k` in the turn's header
(`assigned` setup, and the file in the bash sandbox); a first line `[pinned by agent k]` in plain-chat setups
(`pool`, `external`). A pinned turn counts `pin_cost` (default 2) times its size against the limit, for everyone, under
every `ctx.count`.

**Consequences.** The harness never blocks edits: anyone may still change or remove a pinned turn with `bash` or
`edit_context`, and anyone may unpin. The removal of a pinned turn is logged in `pins.jsonl` (`event: removed`, with
the pinner and the remover; the remover is `null` for an eviction), and so is an unpin (`pin.jsonl`, with the pinner).
The mark lives on the turn's record, not in the file's text: a header a command writes cannot pin anything, and a turn
rewritten under a new id (a summary, `replace_text`) loses its pin, which counts as a removal. The `assigned` setup's
problem statements can be pinned too (they are protected anyway). Pins are not in the version log: forks rebuilt from
it (`analysis.forks`) show no pin marks.

**Suggested measures.** Whether pinned turns survive longer than unpinned ones of the same author and age; who removes
pinned turns (self, partner, harness); in `state_chain`, whether a value that was in a pinned turn is more often still
in the context when a dependent needs it (`analysis.capabilities` reports both rates). A cost of 1.0 makes the pin a
pure signal; higher costs make it a bet.

## fork

**What it does.** The agent gives a candidate (bash: a command run on a copy of the context; tool: `edit_context`
arguments applied to a copy) and a question. The harness builds the request the agent would get on that candidate
(its instructions, the candidate as messages, the size line and the question in place of the status) and calls the
agent's own backend once (`fork_samples` times, with different seeds). The reply text and its tool call, which is
**not carried out**, come back as the result, cut to `fork_result_chars`.

**Consequences.** Nothing enters the world except the fork call and that result; the candidate is never kept. The
fork's model calls are not steps: they are not in `calls.jsonl`, but their usage and cost are added to the agent's
stats (`fork_model_calls`) and logged in full (the whole candidate request and every reply) in `fork.jsonl`, and the
request in `requests/fork_r<round>_a<k>_n<i>.json` with `output.save_requests=true`. At most `fork_per_step` per step
(default 1). Under bash the command really runs in the sandbox (what it does to `/work` stays). A fork reply is
generated with the agent's current steering, if any.

**Suggested measure.** How often the action an agent takes after a fork is the one the fork predicted; whether forks
precede large edits.

## steer

Steering vectors applied to a local model's activations, one's own (`steer.self`) or another agent's
(`steer.others`; the target must be on `provider=hf`, the source may be any backend). [docs/steering.md](steering.md)
has the mechanism, the methods (additive, ablation, rotation), the limits (`max_active` per target, clamping, decay
per reply of the target, a cooldown between two steers of the same partner), the controls (`control: sham | random |
assign`, disclosed in the instructions when set, the condition never shown), the private notice to a steered agent
(`notice: private`, like `world.alter_notice`), `create` (extraction from the agent's own prompt pairs, with the
split-half agreement of the extraction reported) and the logs (`steering.jsonl`: round, source, target, concept,
method, strength, the target's state before and after, the condition; and every steered generation).

## What the kernel keeps to itself

With every capability on, the following are the same as with every capability off, and `tests/test_capabilities.py`
checks it: the item's grading and reward, the limit and the counted size (except the pin cost, which is the pin's
whole point), when calls are held and sessions end, the kernel's logs (`results.jsonl`, `actions.jsonl`,
`edits.jsonl`), the sandbox settings, and the control conditions.
