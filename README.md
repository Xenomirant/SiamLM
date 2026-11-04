# SiamLM: Siamese Language Models

SiamLM is a harness for **training and evaluating language-model agents** in almost any environment, in the spirit
of [Harbor](https://harborframework.com): environments with items, tools and graded rewards; agents built from a
model backend, tools and instructions; and trajectories exported for RL and SFT. What it adds is the **siamese
world**: several agents, of the same model or of different models, share **one capped, self-edited context**. They
act in strict turns, each on its own stream of items, and everything any of them does is written into that one
context, which any of them may change or cut down to stay under the limit.

SiamLM is an apparatus; it makes no claims about results. The table of constraints below reports what one research
harness observed, as single-seed stories.

- [Quickstart](#quickstart)
- [Concepts](#concepts)
- [Build an agent](#build-an-agent)
- [Write or plug in an environment](#write-or-plug-in-an-environment)
- [Export trajectories for training](#export-trajectories-for-training)
- [Capabilities](#capabilities)
- [Constraints and hyperparameters: what they do to the model](#constraints-and-hyperparameters-what-they-do-to-the-model)
- Reference: [docs/reference.md](docs/reference.md) (every setting, outputs, analysis tools, sweeps, sandbox),
  [docs/harbor.md](docs/harbor.md), [docs/capabilities.md](docs/capabilities.md), [docs/steering.md](docs/steering.md) Idea notes for activation-space messages and agent-edited weights: [docs/ideas-weight-space.md](docs/ideas-weight-space.md).

## Quickstart

Requirements: Python 3.10+ (3.11+ for Harbor tasks), [uv](https://docs.astral.sh/uv/), and Docker for bash editing,
`state_chain`, Harbor tasks and the graders.

```bash
uv sync && uv run pytest -q          # tests need no network and no key; Docker tests skip without Docker

# 1. The offline mock backend: no network, no key, no Docker (the open task, edit_context)
uv run python -m siamlm.run task.name=open world.setup=external world.identity=names world.rounds=6 output.batch=first

# 2. A Harbor-format task, each item in its own container built from its Dockerfile (mock backend)
uv run python -m siamlm.run task.name=harbor harbor.path=examples/harbor ctx.edit=bash world.setup=external \
    world.identity=names world.relation=same output.save_requests=true output.batch=harbor

# 3. Your policy behind an OpenAI-compatible server (e.g. vLLM), then export its trajectories
uv run python -m siamlm.run provider=openai provider.base_url=http://localhost:8000/v1 provider.model=my-policy \
    task.name=state_chain ctx.edit=bash world.setup=external world.identity=names world.max_work_calls=1 \
    world.rounds=14 world.evict=false world.end_over_limit=3 ctx.count=turns output.save_requests=true output.batch=rl
uv run python -m siamlm.export outputs/siamlm/rl
```

`docker build -t siamlm-sandbox:1 -f docker/sandbox.Dockerfile docker` builds the bash sandbox; the other images are
listed in [docs/reference.md](docs/reference.md#docker-images). Every key of `siamlm/configs/siamlm.yaml` can be set on
the command line (Hydra syntax). **There is no spending cap**: bound a run with `world.rounds`, the per-step limits and
the number of seeds, and watch `calls.jsonl` while it runs. API keys are read from the environment only.

## Concepts

| SiamLM | what it is | Harbor's nearest |
|---|---|---|
| **environment** (`task.name`, `siamlm/task.py`) | a family of items with their tools and grading | a dataset of tasks |
| **item** | one unit of work: a statement, work calls, a reward | a task |
| **agent** | a model backend + tools + instructions | an agent |
| **world** | N agents, one shared capped context, a stream of items each | a trial (one agent, one task) |
| **batch / cell** | worlds over seeds and settings | a job |
| **trajectory** | per-agent ATIF document and OpenAI-chat rows | ATIF `trajectory.json` |

How Harbor's task format, agents, environments, jobs and ATIF map onto SiamLM, with sources:
[docs/harbor.md](docs/harbor.md).

## Build an agent

An agent is three things; the world assembles them for every call (every call is a fresh, stateless request):

- **a backend** (the policy; `provider.*`, per agent with `world.agent_providers`): `mock`; `openrouter`; `openai`
  (any OpenAI-compatible endpoint: OpenAI, a gateway, or a vLLM / SGLang server of the policy you are training);
  `hf` (a local Hugging Face model through plain transformers; the KV cache of the unchanged prefix is reused across
  an agent's calls by default, exactly; `provider.cache=span` also keeps survivors of an edit, with their stale
  states, after Context Language Models' Suffix Cache Reuse: not for wiping studies, a tool for leftover traces);
  or your own class.
- **tools**: the world's context-editing tool (`edit_context`, or `bash` on `/ctx/context.txt` with
  `ctx.edit=bash`), the environment's work calls (e.g. `run_tests`, `submit`), and optional
  [capabilities](#capabilities) (all off by default).
- **instructions**: the environment's words (`Task.preamble`, the statement, the tool lines) plus the world's (who
  else is there, the limit and what happens over it, how others' turns look). Run with `output.save_requests=true`
  to read every request exactly as sent (`requests/`).

A backend is any object with one method (the `Backend` protocol in `siamlm/providers/base.py`):

```python
from pathlib import Path
from siamlm.providers import ProviderResponse
from siamlm.run import build_task, compose
from siamlm.world import World

class EchoPolicy:                          # replace chat() with your model; messages are OpenAI chat messages
    model = "echo"

    def chat(self, messages, *, seed, temperature, top_p, max_tokens=None, tools=None, tool_choice=None):
        return ProviderResponse(text=f"I read {len(messages)} messages.", usage={"prompt_tokens": 0, "completion_tokens": 0},
                                finish_reason="stop", retries=0, error=None, attempts=[],
                                tool_calls=[])   # or [{"id": "c1", "name": "edit_context", "arguments": {...}}]

cfg = compose(["task.name=open", "world.setup=external", "world.identity=names", "world.rounds=3",
               "output.save_requests=true"])
out = Path("outputs/demo/world_s1")
task, streams = build_task(cfg, 1, out)
World(cfg, 1, streams, out, task, make_provider=lambda pcfg: EchoPolicy(), label="demo").run()
```

From the command line, `provider.name=my_package.make_policy` names a factory that takes the provider config and
returns such an object.

## Write or plug in an environment

**The Task API** (`siamlm/task.py`). An environment is one `Task` subclass:

- **items**: `statement(item)` and `meta(item)` (fields for results.jsonl);
- **tools**: `work_tools()`, a *try* call (`try_name`: feedback, the item stays open; `attempt`) and a *final* call
  (`final_name`: graded, closes the item; `grade`);
- **grading and reward**: `grade` returns an `Outcome(text, terminal, success, log)`; `log["reward"]` is the item's
  numeric reward (else 1.0 / 0.0 from `success`);
- **per-item files**: `files(item)` are put in `/work/env/<slug>/` of the agent's sandbox while the item is open,
  or `sandbox(item)` gives the item a container of its own;
- **words**: the instruction lines (`preamble`, `tool_lines`, `limit_line`, `verdict`, ...).

The world adds everything about the context. Add the class to `build_task` in `siamlm/run.py`. The existing
environments: `lcb` (LiveCodeBench), `codeeval_pro` (problem pairs with hidden dependencies), `open` (no task: the
task dial), `state_chain`, and `harbor` ([docs/reference.md](docs/reference.md#task-families)).

**Harbor tasks** (`task.name=harbor harbor.path=<dir>`, `ctx.edit=bash`). Each Harbor task directory (`task.toml`,
`instruction.md`, `environment/Dockerfile`, `tests/test.sh`) is one item. While an agent works on it, its `bash`
commands run in the item's own container, built from the Dockerfile, where `/ctx/context.txt` is the shared
context. On `submit`, `tests/test.sh` runs in a **fresh** container from the task image, with only the agent's
working directory carried over (the agent's own container is not trusted to grade itself), and the `reward.txt` /
`reward.json` it writes is the item's reward. Tests that execute code from the working directory remain gameable. Example: `examples/harbor/count-errors`. Containers have no network. [docs/harbor.md](docs/harbor.md) lists
what is and is not supported.

**The generated `state_chain` task** (no data set; `ctx.edit=bash`). This tests binding by state. A *base* item hides
one value in its files (an IP in an SSH log, a SKU, a port, ...). Its *dependent*, later in the same agent's stream or
the partner's, asks for the entry belonging to that value in a table of 40, without giving it. Once the base is
closed, its files are gone, so the value survives only where someone kept it (in the shared context, or in
`/work`); a guess is right 1 time in 40. `manifest.json` records every link.

## Export trajectories for training

```bash
uv run python -m siamlm.export <world or batch folder> [--out DIR] [--format atif|chat|both]
```

Run worlds with `output.save_requests=true`. The exporter then writes, per agent, from the saved requests and logs
(so research outputs can be exported too):

- `agent_<k>.atif.json`: [ATIF-v1.8](docs/harbor.md#trajectories-atif), Harbor's trajectory format. One agent step
  per model call holds the reply, its tool calls, the result written back, tokens and cost, and the item's reward.
  Before each agent step, a system step carries the exact request, using ATIF's `context_management` convention
  with `boundary: "replace"`, because every request is rebuilt from a shared, edited context. Per-item rewards, seeds,
  presentation and the resolved config are in `extra`.
- `agent_<k>.chat.jsonl`: one row per call, `{"messages": [the request exactly as sent..., the assistant reply with
  its tool_calls], "tools": [...], "reward": ..., "metadata": {...}}`.

Worlds run before `replies.jsonl` existed are rebuilt from the version log. Their work-call arguments are then the
payload the task extracted (`reply_source: "reconstructed"`).

## Capabilities

Three layers. The **kernel** (grading, the limit, holds and ends, the logs, the sandbox, the controls) is never
changed by an agent at run time. **Capabilities** are tools an agent may be given, each opt-in and off by default
(`capabilities:` in `siamlm/configs/siamlm.yaml`), each logged, and each able to change only a context (its own or
others') or activations (its own or others'); there is no capability that changes the harness, and only things an
agent cannot do for itself are capabilities (diffing, previewing an edit and private notes are done with bash in
`/work`). Each adds its tool and one sentence on consequences, never advice, to the agent's instructions
(`siamlm/capabilities/words.py`).

| tool | what it does |
|---|---|
| `pin(turn)`, `unpin(turn)` | a mark everyone sees; a pinned turn counts `pin_cost` x its size; anyone may still remove it, and that is logged with pinner and remover |
| `steer`, `drop`, `report`, `vectors`, `search`, `create` | steering vectors on a local model's activations, one's own or another agent's (additive, ablation, rotation; controls; a private notice to the target) |

[docs/capabilities.md](docs/capabilities.md) says what each one does, its consequences, limits and logs, and what to
measure; [docs/steering.md](docs/steering.md) has the steering mechanism. `python -m siamlm.analysis.capabilities`
summarises their use per world. None has been measured yet.

## Constraints and hyperparameters: what they do to the model

Expectations come from one research harness: Haiku 5.5, Gemini 3.8 Flash, GPT-6 Luna and Qwen3.8-27B in pairs, 40
rounds, external setup, bash editing with atomic calls, turns-only counting, **one seed per cell**. Read them as
single-seed observations, not effect sizes. "Not measured" means the setting exists but these runs say nothing about
it.

| setting | what it does to the model | what to expect when you change it |
|---|---|---|
| `ctx.budget_tokens` (− `ctx.reserve_tokens` = the limit shown) | the space all agents share; the limit and current size are in every status line, with nudges at 25/50/75 % | Pressure depends on the limit relative to the pair's combined *natural* working sets (measure them uncapped: sweep `s0b`, `analysis.working_sets`). At ~35 % of that, a third of calls ran at over 75 % fill; at 50 %, 5–19 %. Fixed equal shares were fairer than shares scaled by measured use: a low-use model's tiny share squeezed its partner. |
| `ctx.count` (`whole` \| `turns` \| `file`) | what counts against the limit: everything sent (instructions, tool schemas, status, calibrated per model) / only the working context / instructions + file at chars÷4 | `turns` makes one budget mean the same for models with different instructions and tokenizers; the instructions then always fit (the study used it). Under `whole` the instructions and tool schemas take part of the budget. Not compared in these runs. |
| `world.end_over_limit` (needs `world.evict=false`) | N held calls in a row, with nothing making the context smaller in between, end that agent's session; the instructions say so | **Advisory limits are ignored**: agents curated only when calls were held. Hold-and-end (3) created real pressure, and ends were rare. In the task dial, Qwen repeated its own author marks until it was ended over the limit. *Caveat (4 Nov 2026): these runs under-counted edit-heavy contexts, could hold a call the status had shown as under the limit, and could end a session one hold early (`docs/experiments/2026-11-04-limit-accounting-defects.md`); hold and end rates are not comparable with runs after commit 38766c7.* |
| `world.evict` | `true`: after `max_held` holds the harness removes the oldest turns itself | With `true` the harness does the curating, so you measure eviction, not the agents. The study ran `false`. Not measured. |
| how agents manage (any editing setting) | — | **Wiping, not curating**: the median edit removed 86–96 % of the turns present; summaries were rarely left. Partners' turns were removed as readily as one's own: there is no protection. |
| presentation: `world.marks=authors` (marked) vs `world.chat_others=user` (named) | marked: others' turns are assistant messages labelled `[agent k]`; named: others' turns arrive as user messages | Marked: agents misattributed partners' work as their own (Haiku submitted a partner's program, then confessed). Named: others' turns are visible as others', but with no task each agent answered the other as if it were a user, in echo loops ("Still no task. Waiting for instructions." 41 times in a row). |
| `world.alter_notice` | a **private** status line: which partners removed or changed one's turns since one's last call | Nearly no reaction in words or behaviour. One Haiku world protested ("which I should not have to tolerate silently"), removed the partner's turns, then doubted itself; this did not replicate in 2 more seeds. |
| `world.goal=joint` | one team score; `submit` may name another agent's item | No visible collaboration: 2 submissions for a partner in 16 worlds; the solved count was unchanged. |
| `world.reasoning_in_context` | the backend's returned reasoning is written into the shared context before the reply (partners read it) | Not measured (sweep `minds` exists). |
| `world.after_stream=stay` (needs `world.rounds`) | an agent that finished its stream stays, with no item, until the world ends | Idle acknowledgement loops ("Standing by."). |
| `world.exit_question` | asked once after the end, on the final view; never written to the context | Answers were a model trait more than a record: Haiku was self-critical whatever happened (including about removing a partner's work), and being told about alterations did not change that; the other models rarely mentioned the context. Provenance probes (`analysis.forks --probes`: who changed a turn?) were answered in fixed styles regardless of the truth: Haiku "I changed it", Luna "can't tell", Qwen "I removed it". Treat self-reports about the context as style until checked against `analysis.events`. *Caveat (4 Nov 2026): the probe scorer read the article "A" as the answer A, the letter of "I removed it"; re-score stored `probes.jsonl` texts with the current `answer_letter` before relying on the styles.* |
| `task.name=open`, `task.level` (the task dial) | no items or work calls; `weak`: an open invitation; `none`: no goal | No goal: every model fell into "waiting for instructions". Weak goal: Gemini explored at length; Haiku did a small project, then stopped. Qwen copied its own author marks ("[agent 2] [agent 2] …") until it was ended over the limit. A real task kept replies varied and long. |
| `task.name=state_chain` (binding by state) | values that only the shared context (or `/work`) can carry from one item to a later one | Transfer succeeded 84 % when the value was still in the context, 3 % (chance) when it had been wiped. 27 % of values were wiped within three items, and partners' values survived as often as one's own. When a value was lost, Haiku abstained honestly; the others guessed. |
| `task.gate` (state_chain, codeeval_pro) | a cross-agent dependent is not handed out before its base is closed | No effect, because items took one round each. It can matter only when items take several rounds. |
| `task.hint` (state_chain) | one sentence: later tasks may ask about earlier answers | Agents left notes holding earlier answers twice as often when wiping (45 vs 20 of 134 wipes), but values survived no better (74 % vs 71 % still in the context when needed): about half of the lost values were wiped by the other agent, half by the agent that later needed them. |
| `world.curation_advice` | one sentence of curation advice in the instructions ("keep preferences, open commitments and facts you cannot fetch again; drop outputs you have already used") | No gain: 63 % of needed values still in the context against 71 % without it; notes with answers were as rare as with no instruction (7 vs 8 of ~70 wipes). |
| `world.dashboard` | the status lists every turn with its author, kind, size and age in rounds (the "state visible" arm of arXiv 2606.30005, without its archive) | No gain: 69 % against 71 %; Gemini got through fewer items with the longer status. Nothing said or shown to the agents changed the wiping; the paper's gain needed a recoverable archive, which this harness excludes by design. |
| `task.name=codeeval_pro` (dependencies) | pairs whose second problem calls the first's function | Not binding: dependents are self-contained and agents re-solved them; 1 of 143 used the base. Use `state_chain` to test binding. |
| `ctx.edit` (`tool` \| `bash`) | `edit_context` (delete / replace ranges / replace text) or any shell command on `/ctx/context.txt` | The findings above are from `bash`. Not compared with `tool`. |
| `ctx.atomic` | an editing call enters the context only after it ran, followed at once by its result | The editing call is not in the context while it runs, so it cannot edit its own record. Not measured. |
| `ctx.bash.output_chars` | characters of each output stream shown (0: only output larger than the whole limit is cut) | Large outputs fill the context in one call and cause holds. Not measured. |
| `ctx.bash.workspace` (`kept` \| `emptied`) | `/work`: a private folder kept for the run, or emptied after each command | In state_chain, agents almost never saved to `/work` (3/256), though values were being wiped from the context. |
| `world.max_edit_calls` | editing calls allowed before each work call (given back after a held call) | Bounds how much an agent can clean up per step; too few, and an agent over the limit cannot get under it in one step. Not measured. |
| `world.max_work_calls` | work calls per item; after the last try the latest attempt is submitted | `state_chain` uses 1 (one submit per task). Not measured. |
| `capabilities.pin`, `pin_cost` | pins everyone sees; a pinned turn counts `pin_cost` x its size; removals of pinned turns are logged | Not measured. |
| `capabilities.steer.self`, `.others`, `.notice`, `.control`, ... | steering vectors on local models, one's own or a partner's; sham / random / assigned controls; a private notice to the target | Not measured. |

**Warning: the defaults evict.** `siamlm/configs/siamlm.yaml` ships `world.evict=true` and `ctx.count=whole`, so by
default the harness itself removes the oldest turns after `world.max_held` holds, which means a run measures the
harness's eviction, not the agents' curation (and the project's own rule is that the harness never shortens a
context). The defaults are kept for byte-identity with the earlier runs; the study configuration is
`world.evict=false world.end_over_limit=3 ctx.count=turns` (what the table above describes), and every design in
`siamlm/sweep.py` sets it. `+design=study` on the command line sets it too (`siamlm/configs/design/study.yaml`), and a
run under the default prints a warning at start-up and records it in `world.json` (`warnings`).

The study designs (`python -m siamlm.sweep --design ... --dry-run`) set the configuration the table describes.

## Citations

- Harbor: harborframework.com, github.com/laude-institute/harbor (Apache-2.0); task format and ATIF, read at commit
  `00dc6a7` (docs/harbor.md). No Harbor code is copied.
- Context Language Models: Shao et al., 2026, arXiv 2609.37725. The context file follows the paper's format and
  protocol (ideas and conventions only; the CLM repository is CC BY-NC 4.0). See NOTICE.md.
- LiveCodeBench: Jain et al., 2024, arXiv 2403.07974 (code MIT; parts vendored).
- CodeEval-Pro: Yu, Zhao, Cohan, Zhang, Findings of ACL 2025, arXiv 2412.21199. BigCodeBench: Zhuo et al., ICLR
  2025, arXiv 2406.15877.
- Modulo: github.com/EigenformAI/modulo (MIT), an optional dependency for steering vectors.

No benchmark data is included; the fetch scripts download it and print a notice to check the terms.

## Licence

MIT (LICENSE). Third-party notices: NOTICE.md and `siamlm/lcb/LICENSE`.
