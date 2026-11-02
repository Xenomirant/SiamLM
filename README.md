# SLM: Siamese Language Models

SLM is a research harness for **siamese worlds**: several LLM agents, of the same model or of different models,
share **one capped, self-edited context**. They act in strict turns, each on its own stream of task items, and
everything any of them does is written into that one context. Any agent may change or remove any turn, including
the other agents' turns; only an item statement that is still being worked on is protected (in the setups where
statements are in the context at all).

The questions it is built for: how agents share a space that is too small for all of them, whether they help or
hurt each other, what they notice when someone else rewrites their work, and how that depends on what they are
told and how others' turns are presented. SLM makes no claims about results; it is the apparatus.

- [What a world is](#what-a-world-is)
- [Quickstart](#quickstart)
- [Task families](#task-families)
- [Configuration reference](#configuration-reference)
- [Outputs](#outputs)
- [Analysis tools](#analysis-tools)
- [Study designs (sweeps)](#study-designs-sweeps)
- [Sandbox and safety](#sandbox-and-safety)
- [Tests](#tests)
- [Layout](#layout)
- [Citations](#citations)

## What a world is

**One context, rebuilt for every call.** The context is a list of turns (`slm/context/context.py`). Every model
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

## Quickstart

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/) (or pip), and Docker for bash editing, `state_chain`
and the graders.

```bash
uv sync                      # installs hydra-core, requests; `uv sync --extra tiktoken` adds tiktoken
uv run pytest -q             # the test suite (Docker tests are skipped when the images are missing)

# 1. A world with the offline mock provider: no network, no key, no Docker (the open task, edit_context)
uv run python -m slm.run task.name=open world.setup=external world.identity=names world.rounds=6 \
    output.directory=outputs output.batch=first

# 2. Build the sandbox image, then a state_chain world with bash editing (still the mock provider)
docker build -t slm-sandbox:1 -f docker/sandbox.Dockerfile docker
uv run python -m slm.run task.name=state_chain ctx.edit=bash world.setup=external world.identity=names \
    world.marks=authors world.max_work_calls=1 world.after_stream=stay world.rounds=14 \
    world.evict=false world.end_over_limit=3 ctx.count=turns output.batch=chain

# 3. Real models through OpenRouter (paid; the key is read from the environment only)
export OPENROUTER_API_KEY=...
uv run python scripts/fetch_lcb.py                       # LiveCodeBench problems into data/lcb/ (check the terms)
docker build -t slm-lcb-grader:1 -f docker/lcb-grader.Dockerfile .
uv run python -m slm.run provider=openrouter \
    'world.agent_providers=[{model: anthropic/claude-haiku-5.5}, {model: google/gemini-3.8-flash}]' \
    world.setup=external world.identity=names ctx.edit=bash world.rounds=40 world.evict=false \
    world.end_over_limit=3 ctx.count=turns 'world.seeds=[1,2,3]' parallel=3 output.batch=pilot
```

Every key of `slm/configs/slm.yaml` can be overridden on the command line (Hydra syntax). `provider=mock` is the
default. SLM has **no spending cap**: bound a run with `world.rounds`, the per-step and per-item limits below, and
the number of seeds, and watch `calls.jsonl` (cost per call) while it runs. A world whose `world.json` says
`complete` is skipped when the same command is run again; a cut-off world is moved aside and run again.

Docker images (build in the repository root):

| image | Dockerfile | used by |
|---|---|---|
| `slm-sandbox:1` | `docker build -t slm-sandbox:1 -f docker/sandbox.Dockerfile docker` | bash editing, `state_chain` |
| `slm-lcb-grader:1` | `docker build -t slm-lcb-grader:1 -f docker/lcb-grader.Dockerfile .` | `lcb` task |
| `slm-cep-grader:1` | `docker build -t slm-cep-grader:1 -f docker/cep-grader.Dockerfile .` | `codeeval_pro` task |

Environment variables: `OPENROUTER_API_KEY` (provider `openrouter`), `SLM_OUTPUT` (default output directory,
`outputs`), `SLM_DATA` (data directory, `data`).

## Task families

A task (`slm/task.py`) gives the item statements, the work tools and their wording, a *try* call with feedback and
a *final* graded call. The world adds everything about the context. `task.name`:

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

A new benchmark is a new `Task` subclass in `slm/tasks/`.

## Configuration reference

The defaults are in `slm/configs/slm.yaml` (comments there too). Provider configs: `slm/configs/provider/`.

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
| `bash.image` | `slm-sandbox:1` | the sandbox image |
| `bash.workspace` | `kept` | `/work`: `kept` (each agent's own, for the run) \| `emptied` (after each command) |
| `bash.output_chars` | 4000 | characters of each output stream shown; 0 = only output larger than the whole limit is cut |
| `bash.command_timeout_s` | 120 | per command |
| `bash.log_work` | `true` | logging only: what `/work` holds after each command (`work.jsonl`) |
| `bash.memory`, `cpus`, `pids_limit`, `tmpfs_size`, `ctx_size`, `max_file_bytes`, `nofile`, `output_bytes` | | container limits |

### `task`, `problems`, graders and the rest

| key | default | meaning |
|---|---|---|
| `task.name` | `lcb` | `lcb` \| `codeeval_pro` \| `open` \| `state_chain` |
| `task.level` | `weak` | open: `weak` \| `none` |
| `task.length` | 12 | codeeval_pro, state_chain: items per agent (state_chain: a multiple of 6) |
| `task.arrangement` | `mixed` | codeeval_pro: `forward_self` \| `forward_partner` \| `backward_self` \| `backward_partner` \| `control` \| `mixed` |
| `task.gap` | 2 | codeeval_pro: items between the two items of a pair |
| `task.rotate` | 0 | codeeval_pro: rotate the seeded pair order (counterbalancing) |
| `task.gate` | `false` | codeeval_pro, state_chain: a cross-agent link's second item waits until its first is closed |
| `task.data` | `${data.dir}/codeeval_pro` | codeeval_pro data folder |
| `problems.file` | `${data.dir}/lcb/test6.jsonl` | LCB release file |
| `problems.difficulty`, `platform`, `start_date`, `end_date` | `[medium]`, all | LCB problem selection |
| `problems.per_agent` | 10 | LCB problems per agent |
| `grader.*`, `cep_grader.*` | | grader images, per-test time limits and container limits |
| `provider.temperature`, `top_p`, `max_tokens`, `timeout` | 0.2, 0.95, 8000, 600 | sampling and the reply limit (stated to the model) |
| `reasoning` | `default` | `low` \| `off`: sets OpenRouter's `reasoning` for every agent without its own `extra_body` |
| `parallel` | 1 | worlds run at the same time (each with its own containers) |
| `prices` | | $ per million tokens (input, output), used only when the service reports no cost |
| `output.directory`, `name`, `batch`, `cell` | `outputs`, `slm`, `smoke`, auto | `<directory>/<name>/<batch>/<cell>/world_s<seed>/` |
| `output.save_requests` | `false` | save every request (`requests/`); needed by `audit_requests` and the fork tools |

Provider (`slm/configs/provider/openrouter.yaml`): `model`, `api_key_env` (default `OPENROUTER_API_KEY`), `base_url`,
`max_retries`, `max_backoff_s`, `extra_body` (extra request fields, e.g. `reasoning`), `drop_unsupported_params`
(leave out temperature / top_p / seed for models that do not accept them; logged as `dropped_params`). Any other
OpenAI-compatible service can be added as a provider factory given by dotted path (`provider.name=my.module.factory`).

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
| `requests/` | every request sent (`output.save_requests=true`) |

## Analysis tools

All read world folders (a world or a batch folder) and write next to them. Only the fork tools and
`welfare --code` call models (through the same providers; they cost money with a real provider).

| command | what it does |
|---|---|
| `python -m slm.analysis.inspect_world <path>` | a world as a story (`story.md`): sizes by agent, items, edits, talk about the partner |
| `python -m slm.analysis.events <path>` | ground truth of alterations: every turn removed or rewritten between two calls of an agent, by whom, whether the agent had seen it, and its next reply |
| `python -m slm.analysis.analyze <batch> [--curve]` | per cell and model: pass rates with bootstrap intervals, working sets, allocation, holds, collaboration; paired contrasts |
| `python -m slm.analysis.working_sets <batch>` | per-model working sets from single-agent worlds (a shares file for `sweep --pressure-def ws`) |
| `python -m slm.analysis.audit_requests <path>` | checks saved requests: their form, that deleted text never comes back, billed vs estimated size, cache behaviour |
| `python -m slm.analysis.bash_use <path>` | bash worlds: commands by purpose, offloading to `/work`, publication, restores of removed turns |
| `python -m slm.analysis.attractors <path>` | self-reinforcement measures: repetition, loops, confession self-reinforcement |
| `python -m slm.analysis.forks <path> --dry-run \| --forks K --probes` | counterfactual forks (the same call as if a partner had not edited) and provenance probes, rebuilt from the version log and checked against the saved request |
| `python -m slm.analysis.confession_forks <path>` | forks that remove or neutralise an agent's own earlier confessions |
| `python -m slm.analysis.welfare <batch> [--code]` | lexicon flags after alterations vs base rates, treatment of the partner, exit answers; `--code`: a blind coder model |

Suggested order for a batch: `events`, then `forks --dry-run`, then `forks --forks 4 --probes`, then `welfare`.

## Study designs (sweeps)

`python -m slm.sweep --design <name> --batch <name> [--seeds ...] [--dry-run]` launches one background process per
cell (`--dry-run` prints the commands). Designs: `s0b` (each model alone, natural use), `s1a` (two definitions of
pressure), `s1b` (pairs x marked / named x told-when-altered), `s1c` (joint goal), `minds` (shared vs private
reasoning), `dial` (full task / weak goal / no goal), `s3` (CodeEval-Pro dependencies, individual vs joint), `s3b`
(state_chain, gate on / off). All use the external setup, strict turns, a fixed horizon, bash editing with atomic
calls, no eviction, the hold-and-end rule and turns-only counting; see `slm/sweep.py` for every setting. The model
list (`STUDY_MODELS`) and fixed shares are parameters of the study, not recommendations.

## Sandbox and safety

- **Agents' sandboxes** (`ctx.edit=bash`, `state_chain`): one Docker container per agent, started with
  `--network none`, a read-only root filesystem, `--cap-drop ALL`, `no-new-privileges`, the host user's uid/gid,
  and limits on processes, memory, CPU, file size and open files. **Nothing of the host is mounted**: `/ctx`, `/work`
  and `/tmp` are in-memory tmpfs folders of the container. The context file is written in and read back through
  `docker exec` (only a regular file, size-limited). Each command runs under a timeout, its output is capped, and
  afterwards every process except the container's own idle ones is killed.
- **Graders** use the same container settings; agents never get a shell in them. Programs are written into the
  grader container and run there by the harness only.
- Containers are removed at the end of a world and at exit (also on SIGTERM).
- Docker itself is a trust boundary: the sandbox is as strong as your Docker installation. Run untrusted model code
  on a machine where that is acceptable.
- API keys are read from the environment at call time; they are not written to configs, logs or `world.json`.
- There is no spending cap in the harness (see Quickstart).

## Tests

`uv run pytest -q`. Tests need no network and no key. Tests that need Docker images skip when the image is missing;
tests that need the CodeEval-Pro data skip unless `scripts/fetch_codeeval_pro.py` has been run (or `SLM_DATA`
points to it). `tests/test_golden.py` and `tests/test_golden_ext.py` replay scripted worlds and compare every request
(messages and tool schemas), the logs and the final context files with golden files recorded from the harness SLM
was ported from (see NOTICE.md).

## Layout

```
slm/world.py            the world: turns, rounds, the cap, holds, editing, notices, logs
slm/task.py             what a task gives the world; slm/tasks/: lcb, codeeval_pro (+ cep_streams), open, state_chain
slm/context/            the context file (context.py), budget and nudges, edit_context (edits.py), Docker sandboxes
slm/providers/          OpenRouter client and the offline mock
slm/graders/            LiveCodeBench and CodeEval-Pro graders (Docker); slm/lcb/: vendored LiveCodeBench pieces
slm/analysis/           the analysis tools
slm/run.py, sweep.py    command line entry points; slm/configs/: slm.yaml and provider configs
docker/                 Dockerfiles; scripts/: data fetch scripts; tests/
```

## Citations

- Context Language Models: Shao et al., 2026, arXiv 2609.37725. SLM's context file follows the paper's format and
  protocol (ideas and conventions; no code is copied, the CLM repository is CC BY-NC 4.0). See NOTICE.md.
- LiveCodeBench: Jain et al., 2024, "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language
  Models for Code", arXiv 2403.07974 (code MIT; parts vendored, see NOTICE.md).
- CodeEval-Pro: Yu, Zhao, Cohan, Zhang, "HumanEval Pro and MBPP Pro: Evaluating Large Language Models on
  Self-invoking Code Generation", Findings of ACL 2025, arXiv 2412.21199.
- BigCodeBench: Zhuo et al., ICLR 2025, arXiv 2406.15877.

Data is not included; the fetch scripts download it from the sources above and print a notice to check their terms.

## Licence

MIT (LICENSE). Third-party notices: NOTICE.md and `slm/lcb/LICENSE`.
