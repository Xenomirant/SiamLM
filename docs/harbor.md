# Harbor and SiamLM

[Harbor](https://harborframework.com) is a framework for evaluating agents and generating rollouts for RL, the
successor to Terminal-Bench's harness. SiamLM mirrors its concepts where they fit, can run Harbor-format tasks, and
exports trajectories in Harbor's trajectory format (ATIF). Everything below was read from Harbor's own code and docs
at commit `00dc6a7` (1 Nov 2026, package version 0.25.0) of github.com/laude-institute/harbor. The docs now link to
github.com/harbor-framework/harbor. Harbor is Apache-2.0 licensed; **no Harbor code is copied into SiamLM**. The
adapter (`siamlm/tasks/harbor.py`) and the exporter (`siamlm/export.py`) are our own, written to Harbor's documented
formats.

## Concepts side by side

| Harbor | SiamLM | notes |
|---|---|---|
| task (a directory) | item | one unit of work; Harbor tasks become SiamLM items with `task.name=harbor` |
| dataset / registry | task family (`task.name`) | lcb, codeeval_pro, open, state_chain, harbor |
| agent (`BaseAgent.setup` / `run(instruction, environment, context)`) | agent = backend + tools + instructions | SiamLM's world runs the loop, like Harbor's "external" agents (e.g. Terminus-2); the context is the world's, shared |
| environment (`BaseEnvironment`: start, stop, exec, upload/download; docker, daytona, modal, e2b, ...) | sandbox (`siamlm/context/sandbox.py`) | Docker only; per agent (`ctx.bash`) or per item (`Task.sandbox`) |
| verifier (`tests/test.sh` -> `/logs/verifier/reward.txt` or `reward.json`) | `Task.grade` -> Outcome | harbor items: the same test.sh, the same reward files |
| trial (one agent, one task) | world (N agents, a stream of items each, one shared context) | |
| job (trials in parallel, `jobs/<job>/<trial>/`) | batch / cell (`outputs/<name>/<batch>/<cell>/world_s<seed>/`) | `parallel` runs worlds at once |
| ATIF trajectory (`agent/trajectory.json`) | `trajectories/agent_<k>.atif.json` | `python -m siamlm.export` |
| `harbor traces export` (conversations for SFT) | `trajectories/agent_<k>.chat.jsonl` | OpenAI chat rows, exact prompts |

## Harbor's task format (what the adapter reads)

From `src/harbor/models/task/paths.py`, `src/harbor/models/task/config.py` and `docs-mintlify/tasks/`:

- `task.toml` (required; schema 1.4): `[task]` name, `[metadata]` (free), `[verifier]` `timeout_sec` (600) / `env` /
  `user`, `[agent]` `timeout_sec` / `user`, `[environment]` `docker_image` / `workdir` / `cpus` / `memory_mb` /
  `build_timeout_sec` (600) / `env` / `network_mode`, `[solution]`, `steps` (multi-step tasks).
- `instruction.md`: the instruction, given to the agent as its `instruction` string; leading canary comment lines are
  stripped (`strip_canary`, `src/harbor/models/task/task.py`).
- `environment/Dockerfile` (or `[environment].docker_image`): the container the agent works in.
- `tests/test.sh`: the verifier. Harbor (`src/harbor/verifier/verifier.py`) removes stale reward files, uploads
  `tests/` to `/tests`, runs `chmod +x /tests/test.sh` as root, then `(/tests/test.sh) >
  /logs/verifier/test-stdout.txt 2>&1` in the same container (as `[verifier].user`, else the image's user), under the
  verifier timeout.
- The reward: `/logs/verifier/reward.json` (a flat object of finite numbers) wins over `/logs/verifier/reward.txt` (one
  number, read as `{"reward": x}`); a missing or empty file is an error, and metrics count it as 0.
- `solution/solve.sh`: the oracle agent's solution (not used by SiamLM).

## What `task.name=harbor` does

```bash
uv run python -m siamlm.run task.name=harbor harbor.path=examples/harbor ctx.edit=bash world.setup=external \
    world.identity=names world.relation=same provider=mock output.batch=harbor
```

- Each Harbor task under `harbor.path` is one item. Streams: the tasks shuffled by the seed, `harbor.per_agent` per
  agent (`world.relation=same`: every agent the same ones).
- The item's statement is the instruction, plus where the commands run and that `submit` runs the tests.
- When an agent first runs a command on an item, the item's container starts from the task's image. The image is built
  from `environment/` and tagged by the hash of its files, so it is built once. The agent's `bash` commands run in that
  container (in its working directory, as `[agent].user` or the image's user), and SiamLM's context file is mounted
  at `/ctx/context.txt` there, so editing the shared context works as in any bash world.
- `submit` (or a forced close at `world.max_steps_per_problem`) runs the verifier as Harbor does. The reward becomes the
  item's `reward` in `results.jsonl` (and `rewards`: every key of reward.json). The item counts as solved when the
  reward is at least `harbor.solved_at`. The container is then removed; `test-stdout.txt` and the reward file are
  kept in `<world>/harbor/<item>/verifier/`.

Not supported, by design or for now: multi-step tasks, separate verifier environments, artifacts, MCP servers,
skills, GPUs, and backends other than Docker. There is **no network**, for the agent or the verifier: tests that
download their tools (Harbor's hello-world test.sh installs uv with curl) fail. Bake test dependencies into the image.
`[agent].timeout_sec` is not enforced (`world.max_steps_per_problem` and `ctx.bash.command_timeout_s` bound an item).
Background processes do not survive a command: every process a command leaves is killed after it.

## Trajectories (ATIF)

ATIF (Agent Trajectory Interchange Format, `rfcs/0001-trajectory-format.md`, `src/harbor/models/trajectories/`,
current version `ATIF-v1.8`) is a JSON document with `schema_version`, `session_id`, `agent {name, version,
model_name, tool_definitions}`, `steps[]` and `final_metrics`. Each step has `step_id` (from 1), `source`
(`system` | `user` | `agent`) and `message`. Agent steps also carry `reasoning_content`, `tool_calls
[{tool_call_id, function_name, arguments}]`, `observation {results: [{source_call_id, content}]}` and `metrics
{prompt_tokens, completion_tokens, cost_usd, ...}`. Custom data goes in `extra`.

A linear ATIF conversation cannot show a siamese world as the model saw it: every request is rebuilt from a context
that the agent itself and its partners have edited since. SiamLM's exporter therefore uses ATIF's
context-management convention (RFC section VII). Before each agent step comes a `system` step with
`extra.context_management = {type: "siamese_rebuild", boundary: "replace", content_format:
"openai_chat_messages_json"}`, whose observation content is the exact request (OpenAI chat messages, as JSON). Under
the RFC's rule, that content is the agent's whole context for the next step. The `chat.jsonl` file carries the same
prompts as plain OpenAI-chat rows.
