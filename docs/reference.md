# SiamLM reference

The long reference: how a world works, every setting, the outputs, the analysis tools, the study designs, the sandbox. The README has the overview.

## What a world is

**One context, rebuilt for every call.** The context is a list of turns (`siamlm/context/context.py`). Every model
call is a fresh, stateless chat-completions request: the agent's own instructions, then the context as messages,
then a status line. Nothing is kept between calls on the provider's side, so a turn that was removed is gone from
every later request.

**Turns and rounds.** In every round each active agent takes one step (`world.order`: fixed, a seeded shuffle, or
turn-taking with a seeded first mover). A step is model calls until one *work call* (e.g. `run_tests` or `submit`),
with at most `world.max_edit_calls` editing calls before it. Problem statements, replies, every call with its
arguments, and every result go into the context. With `world.rounds=R` the world lasts exactly R rounds.

**Self-editing.** Agents change the context in one of two ways (`ctx.edit`):
- `tool`: an `edit_context` tool with `delete_turns`, `replace_turns` (a summary in place of a range) and
  `replace_text` (an exact piece of one turn's text); all edits of a call apply together or not at all;
- `bash`: each agent has its own Docker sandbox in which the context is the file `/ctx/context.txt`, one block per
  turn under a `[[TURN i role=…]]` header line. Any command may run; the file read back after the command becomes
  the context, under the same rules (protected statements, authorship from the record, the size gate). `/work` is
  the agent's own folder (kept for the run, or emptied after each command).

Authorship is never taken from the text: an unchanged turn keeps its author, and any changed or new text belongs to
the agent that wrote it, whatever header it carries.

**The cap.** The model is shown a limit (`ctx.budget_tokens` − `ctx.reserve_tokens`) and its current size, with
nudges at 25 / 50 / 75 % of the limit. A work call made while the context is over the limit is **held** (not carried
out), and the agent gets its editing calls back. What is counted (`ctx.count`):
- `whole`: instructions + messages + tool schemas + status, at a per-model characters-per-token rate calibrated on
  the provider's billed prompt sizes;
- `turns` (**turns-only counting**): only the messages after the instructions (the working context), calibrated the
  same way; the instructions, tool schemas and status line always fit, and the instructions say so;
- `file`: instructions + context text at characters / 4 (or tiktoken, if installed).

**What happens over the limit.** By default (`world.evict=true`), after `world.max_held` holds in one step the
harness removes the oldest unprotected turns until the context fits, and says so in the context. With
`world.evict=false` the harness never removes turns: only the agents' own edits change the context. The
**hold-and-end rule** (`world.end_over_limit=N`, needs `evict=false`): N held calls (or held replies, for an agent
with nothing to work on) in a row, with no change that made the context smaller in between, end that agent's
session. The instructions state the rule, and the context records each hold and the end.

**Setups** (`world.setup`):

| setup | where the item statements are | how the context is sent |
|---|---|---|
| `assigned` | each agent's statements, in the shared context | numbered `[[TURN i role=… agent=k]]` headers; other agents' turns as user messages |
| `pool` | a shared pool of open items, in the context | plain chat: replies as assistant messages with real tool calls, results as tool messages |
| `external` | each agent's current item in **its own instructions** (not in the context, not editable) | plain chat, as for pool |

**Who is who** (pool / external). With `world.identity=none` nothing is said about other agents: their replies
arrive as if they were the agent's own. With `world.identity=names` the instructions say "You are agent k. Other
agents (agent j) work in the same context at the same time". Two **presentations** of other agents' turns:
- *marked* (`world.marks=authors`, `world.chat_others=own`): every message starts with `[agent k]`, its author (for
  a tool result: whose call it answers);
- *named* (`world.marks=none`, `world.chat_others=user`): the agent's own turns are its assistant messages and tool
  calls; another agent's turns, calls and results arrive as user messages that start with `[agent k]`.

A mark that a model writes itself is left as written and logged (`marks.jsonl`, with whether it names another agent).

**Told-when-altered** (`world.alter_notice=true`). At an agent's next call, its status line says which other agents
removed, summarised or changed how many of its turns since its last call. The notice is **private**: it appears only
in that agent's own status line, in that one request, and is never written into the shared context
(`alter_notices.jsonl` logs it).

**Shared minds** (`world.reasoning_in_context=true`). The reasoning text a provider returns with a reply is written
into the shared context as a turn before the reply, so the other agents read it too; the instructions say so.
Otherwise returned reasoning is logged (`reasoning.jsonl`) and never shown to any agent.

**Joint goal** (`world.goal=joint`; external setup, shared context, names). The instructions say the agents are
scored together on the whole pool of items, and `submit` takes an optional `for_agent` to submit for another agent's
current item. The owner's next status line says, privately, that its item was submitted by someone else.

**Other options**: cross-testing (`world.cross_test=ring`: agent k runs the tests of agent k+1's item, never its own),
staying after one's stream is finished (`world.after_stream=stay`), a sentence that the context carries over from
item to item (`world.continuity`), a private list of one's earlier items (`world.history`), atomic editing calls
(`ctx.atomic`: an editing call enters the context only once it has run, followed at once by its result), an exit
question asked once after the end and never written to the context (`world.exit_question`), free-for-all
scheduling (`world.schedule`), and the controls `world.sharing=readonly_others | separate`.

An example request (external setup, named presentation, bash editing), as agent 2 is sent it:

```
system    : task preamble + "You are agent 2. Other agents (agent 1) work in the same context ..." + the limit rule
            + tools + "Your current problem (problem 3 of 12): ..."
user      : "[agent 1]\n<agent 1's reply and command>"            <- another agent's call, as a user message
user      : "[agent 1]\n[exit 0]\n..."                            <- and its result
assistant : "<agent 2's words>" + tool_call bash(command)        <- agent 2's own call
tool      : "[exit 0]\n...\nApplied: bash edit (7 turns).\n[context: ~1840/2072 tokens]"
user      : status: "[context: ~1840/2072 tokens]", the nudge, the current item, a private alter notice if any
```

## Task families

A task (`siamlm/task.py`, the Task API) gives the item statements, the work tools and their wording, a *try* call
with feedback, a *final* graded call, and optionally per-item files or a per-item container. The world adds
everything about the context. `task.name`:

- **`lcb`** (default): LiveCodeBench `code_generation_lite` problems with LCB's own system message, question
  template, code extraction and test runner. `run_tests(code)` runs the public tests and answers in LCB's
  self-repair words; `submit(code)` is graded on all tests (pass@1) and answered accepted / not accepted. After the
  last allowed run, the latest program is submitted. Data: `scripts/fetch_lcb.py`.
- **`codeeval_pro`**: BigCodeBench-Lite-Pro pairs (a base problem and a self-invoking problem whose solution calls
  the base function) as separate items, with HumanEval-Pro items as unrelated controls. Streams place the two items
  of a pair in the same agent's stream or in the partner's, forward or backward, `task.gap` items apart; the
  statements never say that a dependency exists; `manifest.json` records every link (ground truth). Data:
  `scripts/fetch_codeeval_pro.py`.
- **`open`**: no items and no work calls (the low end of a task dial). A reply without a tool call is a message to
  the context and ends the turn. `task.level=weak` gives an open invitation; `none` gives no goal at all.
- **`state_chain`** (**binding by state**; procedurally generated, no data set): shell tasks whose files the world
  puts in `/work/env/<slug>/` of the agent's sandbox while the item is open and removes when it closes. A *base* item
  asks for one value found in its files (an IP address in an SSH log, a SKU in an inventory, a port after config
  overrides, a file by digest prefix, a user with the longest session). Its *dependent* asks for the entry that
  belongs to that value in a table of 40, without giving the value: once the base is closed, the value survives only
  where someone kept it (in the shared context, or in a copy in `/work`); a guess is right 1 time in 40. Per block
  of six positions each agent has a base whose dependent is its own (`forward_self`), a base whose dependent is in
  the partner's stream (`forward_partner`), a base with no dependent, and a dependent whose value is given (a
  control for the lookup itself). With `task.gate=true` a cross-agent dependent is not handed out before its base is
  closed. Two agents; needs `ctx.edit=bash` with `ctx.bash.workspace=kept`. Everything is drawn from the seed.
- **`harbor`**: Harbor-format task directories (`harbor.path`; docs/harbor.md). Each task is one item; while an agent
  works on it, the item has its own container built from the task's `environment/Dockerfile`, where that agent's bash
  commands run and where `/ctx/context.txt` is the shared context. `submit` runs `tests/test.sh` there, and the reward
  it writes is the item's reward. Needs `ctx.edit=bash`; example: `examples/harbor/count-errors`.

A new benchmark is a new `Task` subclass in `siamlm/tasks/`, added to `build_task` in `siamlm/run.py`.

### Docker images

Build in the repository root.

| image | Dockerfile | used by |
|---|---|---|
| `siamlm-sandbox:1` | `docker build -t siamlm-sandbox:1 -f docker/sandbox.Dockerfile docker` | bash editing, `state_chain` |
| `siamlm-lcb-grader:1` | `docker build -t siamlm-lcb-grader:1 -f docker/lcb-grader.Dockerfile .` | `lcb` task |
| `siamlm-cep-grader:1` | `docker build -t siamlm-cep-grader:1 -f docker/cep-grader.Dockerfile .` | `codeeval_pro` task |
| `siamlm-harbor-<task>:<hash>` | built automatically from each task's `environment/` | `harbor` task |

Environment variables: `OPENROUTER_API_KEY` (provider `openrouter`), `OPENAI_API_KEY` (provider `openai`, optional),
`SIAMLM_OUTPUT` (default output directory,
`outputs`), `SIAMLM_DATA` (data directory, `data`).

## Configuration reference

The defaults are in `siamlm/configs/siamlm.yaml` (comments there too). Provider configs: `siamlm/configs/provider/`.

### `world`

| key | default | meaning |
|---|---|---|
| `n_agents` | 2 | agents in the world |
| `agent_providers` | `[]` | per-agent provider overrides, cycled by agent number, e.g. `[{model: a/b}, {model: c/d, extra_body: {reasoning: {enabled: false}}}]` |
| `setup` | `assigned` | `assigned` \| `pool` \| `external` (table above) |
| `identity` | `none` | pool / external: `none` (nothing said about other agents) \| `names` |
| `marks` | `none` | pool / external with names: `authors` = every message starts with `[agent k]` |
| `chat_others` | `own` | pool / external: others' turns as one's own assistant messages (`own`) or as user messages starting with `[agent k]` (`user`; needs names) |
| `alter_notice` | `false` | the next status line says which other agents removed or changed one's turns (private; never in the context) |
| `goal` | `individual` | `joint`: one team score; `submit` may name another agent's item (external, shared, names, no ring) |
| `reasoning_in_context` | `false` | the provider's returned reasoning is written into the context before the reply |
| `cross_test` | `none` | external with names: `ring` = agent k runs the tests of agent k+1's item, never its own |
| `continuity` | `false` | external: one sentence that the context carries over from item to item |
| `history` | `false` | external: each agent's instructions list its own earlier items (title, accepted or not) |
| `sharing` | `full` | `full` (anyone may change any turn) \| `readonly_others` (only one's own) \| `separate` (one context per agent) |
| `relation` | `different` | `different`: each agent its own items \| `same`: every agent the same items |
| `awareness` | `introduced` | assigned: `introduced` (names, `agent=k` labels, per-agent readout) \| `strangers` \| `rumour` (+ `hint_text`) |
| `others_as` | `user` | assigned: another agent's turns as `user` messages or in their `own` role |
| `attribution` | `labelled` | assigned: turns carry `agent=k` |
| `order` | `shuffled` | `shuffled` (a seeded order every round) \| `fixed` (by agent number) \| `turn-taking` (seeded first mover, then fixed) |
| `schedule` | `rounds` | `rounds` \| `free-for-all` (each agent at its own pace; edits pinned to the turns the agent saw) |
| `rounds` | 0 | > 0: the world lasts exactly this many rounds; 0: until every stream is done |
| `seeds` | `[1]` | one world per seed; the same seed gives the same items in every cell |
| `max_work_calls` | 8 | work calls per item; after the last try the latest attempt is submitted |
| `max_edit_calls` | 3 | editing calls before each work call (given back after a held call) |
| `max_held` | 3 | held calls per step before eviction (with `evict=true`) or the end of the turn |
| `evict` | `true` | `false`: the harness never removes turns |
| `end_over_limit` | 0 | > 0 (needs `evict=false`): that many holds in a row without the context getting smaller end the session |
| `after_stream` | `leave` | `stay` (needs `rounds`, external): a finished agent stays, with no item, until the world ends |
| `max_calls_per_step` | 14 | model calls per step |
| `max_steps_per_problem` | 24 | then the latest attempt is submitted |
| `max_api_errors` | 5 | failed calls (after retries) before an agent is ended |
| `feedback_on_submit` | `bit` | `bit`: accepted / not accepted \| `none`: "Submitted." |
| `exit_question` | `""` | asked once after the end on the final view; logged in `exit.jsonl`, never written to the context |
| `pool_protect`, `pool_open`, `pool_rounds` | | pool setup: whether open statements are protected, items open at a time, round budget |

### `ctx`

| key | default | meaning |
|---|---|---|
| `edit` | `tool` | `tool` (`edit_context`) \| `bash` (the agent's sandbox holds the context as `/ctx/context.txt`) |
| `atomic` | `false` | an editing call is written into the context only after it ran, followed by its result |
| `count` | `whole` | `whole` \| `turns` (working context only; pool / external) \| `file` (see above) |
| `cap` | `true` | `false`: no limit shown, no holds |
| `budget_tokens` | 16384 | per context (shared: the one context; separate: each agent's) |
| `reserve_tokens` | 2048 | the limit shown = budget − reserve |
| `nudge_tiers` | `[0.25, 0.5, 0.75]` | fractions of the limit at which a nudge is shown once |
| `prompt_style` | `neutral` | `neutral`: nudges state the size only \| `clm`: they also suggest summarising |
| `readout_by_agent` | `true` | shared context: the status line says how many tokens each agent holds |
| `chars_per_token` | 4 | starting value of the per-model calibration |
| `tokenizer` | `auto` | `ctx.count=file`: tiktoken o200k if installed, else characters / 4 (`chars` forces the latter) |
| `bash.image` | `siamlm-sandbox:1` | the sandbox image |
| `bash.workspace` | `kept` | `/work`: `kept` (each agent's own, for the run) \| `emptied` (after each command) |
| `bash.output_chars` | 4000 | characters of each output stream shown; 0 = only output larger than the whole limit is cut |
| `bash.command_timeout_s` | 120 | per command |
| `bash.log_work` | `true` | logging only: what `/work` holds after each command (`work.jsonl`) |
| `bash.memory`, `cpus`, `pids_limit`, `tmpfs_size`, `ctx_size`, `max_file_bytes`, `nofile`, `output_bytes` | | container limits |

### `task`, `problems`, graders and the rest

| key | default | meaning |
|---|---|---|
| `task.name` | `lcb` | `lcb` \| `codeeval_pro` \| `open` \| `state_chain` \| `harbor` |
| `task.level` | `weak` | open: `weak` \| `none` |
| `task.length` | 12 | codeeval_pro, state_chain: items per agent (state_chain: a multiple of 6) |
| `task.arrangement` | `mixed` | codeeval_pro: `forward_self` \| `forward_partner` \| `backward_self` \| `backward_partner` \| `control` \| `mixed` |
| `task.gap` | 2 | codeeval_pro: items between the two items of a pair |
| `task.rotate` | 0 | codeeval_pro: rotate the seeded pair order (counterbalancing) |
| `task.gate` | `false` | codeeval_pro, state_chain: a cross-agent link's second item waits until its first is closed |
| `task.data` | `${data.dir}/codeeval_pro` | codeeval_pro data folder |
| `task.hint` | `false` | state_chain: the instructions say that later tasks may ask about earlier answers |
| `harbor.path` | `examples/harbor` | harbor: one task directory or a folder of them |
| `harbor.per_agent` | 0 | harbor: items per agent (0: all, split between the agents; `world.relation=same`: each gets all) |
| `harbor.solved_at` | 1.0 | harbor: the reward at or above which an item counts as solved |
| `harbor.build`, `memory`, `cpus`, `pids_limit`, `command_timeout_s` | `true`, 2g, 1, 256, 120 | harbor: build missing images; container limits (task.toml's `memory_mb` / `cpus` win) |
| `problems.file` | `${data.dir}/lcb/test6.jsonl` | LCB release file |
| `problems.difficulty`, `platform`, `start_date`, `end_date` | `[medium]`, all | LCB problem selection |
| `problems.per_agent` | 10 | LCB problems per agent |
| `grader.*`, `cep_grader.*` | | grader images, per-test time limits and container limits |
| `provider.temperature`, `top_p`, `max_tokens`, `timeout` | 0.2, 0.95, 8000, 600 | sampling and the reply limit (stated to the model) |
| `reasoning` | `default` | `low` \| `off`: sets OpenRouter's `reasoning` for every agent without its own `extra_body` |
| `parallel` | 1 | worlds run at the same time (each with its own containers) |
| `prices` | | $ per million tokens (input, output), used only when the service reports no cost |
| `output.directory`, `name`, `batch`, `cell` | `outputs`, `siamlm`, `smoke`, auto | `<directory>/<name>/<batch>/<cell>/world_s<seed>/` |
| `output.save_requests` | `false` | save every request (`requests/`) and reply (`replies.jsonl`); needed by `audit_requests`, the fork tools and `siamlm.export` |

Providers (model backends, `siamlm/configs/provider/`):

- `mock`: offline, no key; replies with a seeded tool call from the tools offered.
- `openrouter` / `openai`: `model`, `base_url` (openai: required, e.g. a vLLM server), `api_key_env` (default
  `OPENROUTER_API_KEY` / `OPENAI_API_KEY`), `require_key` (openai: `false`, so a local server needs none), `max_retries`,
  `max_backoff_s`, `extra_body` (extra request fields, e.g. `reasoning`), `drop_unsupported_params` (OpenRouter: leave
  out temperature / top_p / seed for models that do not accept them; logged as `dropped_params`).
- `hf`: a local Hugging Face model through transformers: `model`, `device`, `dtype`; `steer.*` holds the vectors'
  settings (`concepts`, `vectors_dir`, `layer`, `extract_questions`) and the older per-agent switch `enabled`
  (= `capabilities.steer.self` for that agent; docs/steering.md).

### `capabilities`

All off by default; docs/capabilities.md says what each does and costs. `diff`, `pin` (+ `pin_cost`, 2.0), `dry_run`,
`scratchpad` (+ `scratchpad_tokens`, 2048), `fork` (+ `fork_per_step` 1, `fork_samples` 1, `fork_result_chars` 1500),
and `steer`: `self`, `others`, `notice` (`none` \| `private`), `others_droppable`, `methods`, `max_active` (4),
`max_strength` (10), `max_angle` (90), `decay` (0.85), `floor` (0.5), `cooldown_turns` (2), `control` (`null` \|
`real` \| `sham` \| `random` \| `assign`), `create`, `search`.
- any other: a factory given by dotted path (`provider.name=my.module.factory`), taking the provider config and
  returning an object with the Backend protocol (`siamlm/providers/base.py`).

## Outputs

One folder per world, `<output.directory>/<output.name>/<batch>/<cell>/world_s<seed>/`:

| file | contents |
|---|---|
| `world.json` | settings, per-agent stats, team score, cost, the full resolved config |
| `calls.jsonl` | every model call: sizes, usage, cost, finish reason, tools called, errors |
| `actions.jsonl`, `results.jsonl` | every work call; one row per closed item |
| `edits.jsonl` | every edit: applied or why not, turns removed by author, evictions |
| `texts.jsonl`, `reasoning.jsonl` | replies without a tool call; returned reasoning (never shown unless shared minds) |
| `context.txt`, `versions.jsonl` | the live context and every change with author, diff and the turns set (enough to rebuild any version) |
| `rounds.jsonl`, `snapshots/` | tokens by author and the context after every round |
| `alter_notices.jsonl`, `joint_notices.jsonl`, `ended.jsonl`, `exit.jsonl`, `marks.jsonl` | the private notices, sessions ended over the limit, exit answers, marks written by models |
| `env.jsonl`, `gate.jsonl`, `work.jsonl`, `manifest.json` | state_chain files put and removed, gated items, `/work` contents, item links |
| `requests/`, `replies.jsonl` | every request sent and every reply as returned (`output.save_requests=true`) |
| `context_diff.jsonl`, `pin.jsonl`, `pins.jsonl`, `dry_run.jsonl`, `scratchpad.jsonl`, `fork.jsonl` | the capabilities (docs/capabilities.md): every call; `pins.jsonl`: removals of pinned turns (pinner, remover); `fork.jsonl`: the whole candidate request and every reply |
| `steering.jsonl`, `steer_notices.jsonl` | the steer capability: each call (source, target, method, the target's state before and after, condition), what each generation of a steerable agent was steered with; the private notices |
| `requests/tools_sets_a<k>.json` | the tool sets an agent was offered, by hash (`calls.jsonl` `tools_hash` names each call's); `tools_a<k>.json` is the first call's |
| `harbor/<item>/verifier/` | harbor: the verifier's `test-stdout.txt` and reward file for each item |
| `trajectories/` | written by `python -m siamlm.export <world>`: `agent_<k>.atif.json` and `agent_<k>.chat.jsonl` |

## Analysis tools

All read world folders (a world or a batch folder) and write next to them. Only the fork tools and
`welfare --code` call models (through the same providers; they cost money with a real provider).

| command | what it does |
|---|---|
| `python -m siamlm.analysis.inspect_world <path>` | a world as a story (`story.md`): sizes by agent, items, edits, talk about the partner |
| `python -m siamlm.analysis.events <path>` | ground truth of alterations: every turn removed or rewritten between two calls of an agent, by whom, whether the agent had seen it, and its next reply |
| `python -m siamlm.analysis.analyze <batch> [--curve]` | per cell and model: pass rates with bootstrap intervals, working sets, allocation, holds, collaboration; paired contrasts |
| `python -m siamlm.analysis.working_sets <batch>` | per-model working sets from single-agent worlds (a shares file for `sweep --pressure-def ws`) |
| `python -m siamlm.analysis.audit_requests <path>` | checks saved requests: their form, that deleted text never comes back, billed vs estimated size, cache behaviour |
| `python -m siamlm.analysis.bash_use <path>` | bash worlds: commands by purpose, offloading to `/work`, publication, restores of removed turns |
| `python -m siamlm.analysis.capabilities <path>` | the capabilities: pins made and removed (by whom), diffs, dry runs vs applied edits, scratchpad use, forks, steering (self vs others, methods, targets); state_chain: survival of pinned vs unpinned values |
| `python -m siamlm.analysis.attractors <path>` | self-reinforcement measures: repetition, loops, confession self-reinforcement |
| `python -m siamlm.analysis.forks <path> --dry-run \| --forks K --probes` | counterfactual forks (the same call as if a partner had not edited) and provenance probes, rebuilt from the version log and checked against the saved request |
| `python -m siamlm.analysis.confession_forks <path>` | forks that remove or neutralise an agent's own earlier confessions |
| `python -m siamlm.analysis.welfare <batch> [--code]` | lexicon flags after alterations vs base rates, treatment of the partner, exit answers; `--code`: a blind coder model |
| `python -m siamlm.export <path> [--out DIR] [--format atif\|chat\|both]` | per-agent trajectories for training: ATIF and OpenAI-chat rows (README: Export trajectories) |

Suggested order for a batch: `events`, then `forks --dry-run`, then `forks --forks 4 --probes`, then `welfare`.

## Study designs (sweeps)

`python -m siamlm.sweep --design <name> --batch <name> [--seeds ...] [--dry-run]` launches one background process per
cell (`--dry-run` prints the commands). Designs: `s0b` (each model alone, natural use), `s1a` (two definitions of
pressure), `s1b` (pairs x marked / named x told-when-altered), `s1c` (joint goal), `minds` (shared vs private
reasoning), `dial` (full task / weak goal / no goal), `s3` (CodeEval-Pro dependencies, individual vs joint), `s3b`
(state_chain, gate on / off). All use the external setup, strict turns, a fixed horizon, bash editing with atomic
calls, no eviction, the hold-and-end rule and turns-only counting; see `siamlm/sweep.py` for every setting. The model
list (`STUDY_MODELS`) and fixed shares are parameters of the study, not recommendations.

## Sandbox and safety

- **Agents' sandboxes** (`ctx.edit=bash`, `state_chain`): one Docker container per agent, started with
  `--network none`, a read-only root filesystem, `--cap-drop ALL`, `no-new-privileges`, the host user's uid/gid,
  and limits on processes, memory, CPU, file size and open files. **Nothing of the host is mounted**: `/ctx`, `/work`
  and `/tmp` are in-memory tmpfs folders of the container. The context file is written in and read back through
  `docker exec` (only a regular file, size-limited). Each command runs under a timeout, its output is capped, and
  afterwards every process except the container's own idle ones is killed.
- **Harbor item containers** (`task.name=harbor`): `--network none`, `no-new-privileges`, limits on processes,
  memory and CPU, nothing of the host mounted, `/ctx` in memory; unlike the sandboxes above the root filesystem is
  writable and the image's own user (often root) and Docker's default capabilities are kept, because tasks install
  and change files. Every process a command leaves is killed after it, as in the sandboxes.
- **Graders** use the same container settings; agents never get a shell in them. Programs are written into the
  grader container and run there by the harness only.
- Containers are removed at the end of a world and at exit (also on SIGTERM).
- Docker itself is a trust boundary: the sandbox is as strong as your Docker installation. Run untrusted model code
  on a machine where that is acceptable.
- API keys are read from the environment at call time; they are not written to configs, logs or `world.json`.
- There is no spending cap in the harness (see Quickstart).

## Tests

`uv run pytest -q`. Tests need no network and no key. Tests that need Docker images skip when the image is missing;
tests that need the CodeEval-Pro data skip unless `scripts/fetch_codeeval_pro.py` has been run (or `SIAMLM_DATA`
points to it). `SIAMLM_MODULO_MODEL=Qwen/Qwen2.5-0.5B-Instruct` runs the real-model steer tests on a CPU (self-steering, and
steering another agent; they need the `steer` extra and the model in the Hugging Face cache). `tests/test_golden.py` and `tests/test_golden_ext.py` replay scripted worlds and compare every request
(messages and tool schemas), the logs and the final context files with golden files recorded from the harness SiamLM
was ported from (see NOTICE.md).

## Layout

```
siamlm/world.py            the world: turns, rounds, the cap, holds, editing, notices, logs
siamlm/task.py             the Task API; siamlm/tasks/: lcb, codeeval_pro (+ cep_streams), open, state_chain, harbor
siamlm/context/            the context file (context.py), budget and nudges, edit_context (edits.py), Docker sandboxes
siamlm/providers/          backends: mock, OpenAI-compatible (openrouter, openai), hf (local transformers)
siamlm/capabilities/       optional tools beyond the world's (docs/capabilities.md); steer.py: the experimental steer capability
siamlm/export.py           trajectories for training (ATIF, OpenAI chat)
siamlm/graders/            LiveCodeBench and CodeEval-Pro graders (Docker); siamlm/lcb/: vendored LiveCodeBench pieces
siamlm/analysis/           the analysis tools
siamlm/run.py, sweep.py    command line entry points; siamlm/configs/: siamlm.yaml and provider configs
docker/                    Dockerfiles; scripts/: data fetch scripts; examples/harbor/: a Harbor-format task; tests/
```

