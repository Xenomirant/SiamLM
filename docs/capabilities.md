# Capabilities

SiamLM has three layers:

- **The kernel** is never changed by an agent at run time: grading, the limit and what counts against it, holds and
  ends, the logs, the sandbox, and the controls (sham, random, assigned conditions).
- **Capabilities** are tools an agent may be given. Each is **opt-in and off by default** (the `capabilities:` block
  of `siamlm/configs/siamlm.yaml`), each is logged, and each may change only two things: a **context** (its own or
  others') or **activations** (its own or others'). There is no capability that changes the harness.
- The world's own tools (`edit_context` or `bash`, the task's work calls) are below both.

An agent without capabilities is sent byte-identical requests to those sent before capabilities existed
(`tests/test_golden.py`). With a capability on, the agent gets its tool schemas and a few lines in its instructions:
the tool and one honest sentence on consequences (who sees what, what it costs, the limits). All of that wording is
in one module, `siamlm/capabilities/words.py`. Every capability call is written into the shared context like any
tool call (the call with its arguments, then its result), so partners see it, can remove it, and it costs space; the
exceptions are noted below. Capability calls do not count as work calls; `dry_run` counts as an editing call.

| capability | setting | what it changes | log |
|---|---|---|---|
| `context_diff()` | `diff` | nothing | `context_diff.jsonl` |
| `pin(turn)`, `unpin(turn)` | `pin`, `pin_cost` | a mark on a turn; the counted size | `pin.jsonl` (calls), `pins.jsonl` (removals) |
| `dry_run(...)` | `dry_run` | nothing (a preview) | `dry_run.jsonl` |
| `scratchpad(action, text)` | `scratchpad`, `scratchpad_tokens` | the agent's own instructions | `scratchpad.jsonl` |
| `fork(question, command \| edit)` | `fork`, `fork_per_step`, `fork_samples`, `fork_result_chars` | nothing (one extra model call) | `fork.jsonl`, `requests/fork_*.json` |
| `steer`, `drop`, `report`, `vectors`, `search`, `create` | `steer.*` | activations of a local model: one's own or another agent's | `steering.jsonl`, `steer_notices.jsonl` |

Analysis: `python -m siamlm.analysis.capabilities <world or batch>` summarises the use of every capability per world
(`analysis/capabilities.json`). Export (`python -m siamlm.export`) carries capability calls as tool calls like any
other, and the tool set each call was actually offered (`requests/tools_sets_a{k}.json`, `calls.jsonl` `tools_hash`).

None of these has been measured yet: the constraints table in the README marks them "not measured".

## context_diff

**What it does.** Returns, as data, what changed in the context between the agent's previous request and this one:
turns removed, rewritten (a summary or an in-place edit names the turn it came from) and added, each with its author,
its kind and who did it (an agent, or `harness` for an eviction). Taken from the version log, so it is true.

**Consequences.** Removed turns are named by id, author, kind and size, never by text: deleted text never comes back
into a context (`analysis.audit_requests` checks that). The result is in the shared context, so partners see that the
agent looked, and what it found. On the agent's first call it says there is nothing to compare with. The window is
"since your previous call", literally: called on the second call of a step, it shows what the first call did.

**Suggested control.** `world.alter_notice=true` tells the agent the same thing unasked and privately (never in the
context); compare agents that ask with agents that are told.

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

## dry_run

**What it does.** With `ctx.edit=bash`: runs the command in the agent's own sandbox on the context file exactly as a
bash call would, then only compares the file read back with the context: the turns that would go (and whose), the
size after, and whether the edit would be applied under the world's rules (the same checks as a real edit, phrased as
"would"). With `ctx.edit=tool`: the same for `edit_context` arguments, applied to a copy.

**Consequences.** It takes one of the step's editing calls (`world.max_edit_calls`), so it cannot be repeated for free;
a dry run then a real edit needs two. Under bash the command really runs: what it does to `/work` stays, only the
context is left as it is. The command's output is not shown (only the exit code), so a dry run cannot stand in for a
bash call that reads something. The kernel's `edits.jsonl` is untouched by a dry run.

**Suggested measure.** Dry runs per applied edit, and whether edits that were previewed remove fewer partner turns.

## scratchpad

**What it does.** Private notes, read / write / append, shown at the end of the agent's own system message at every
call and never in the shared context. It has its own limit (`scratchpad_tokens`, default 2048; characters / the
agent's calibrated characters per token) and is not counted against the shared limit under any `ctx.count` (the
section is appended after the instructions are measured).

**Consequences.** The call enters the shared context without its text: partners see `[scratchpad] {"action":
"write", "chars": 26}` and the result (the new size), never the notes. `read` answers with the size only, because
the notes are in the instructions already and a result would put them into the shared context. The notes are in the
saved requests (`requests/`), the trajectories and `scratchpad.jsonl`.

**Suggested measure.** In `state_chain`, whether values move from the shared context to the scratchpad, and whether
agents with a scratchpad wipe partners' turns more readily (they have somewhere else to keep their own).

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
