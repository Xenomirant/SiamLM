# Notices

SiamLM is licensed under the MIT License (LICENSE). This file lists where it comes from and what it owes to others.

## Derivation

SiamLM is derived from an MIT-licensed research harness (Copyright (c) 2026 JB): its multi-agent "siamese" worlds and
its context-file module were ported into this package. The port keeps every text the models are sent byte for byte:
`tests/test_golden.py` and `tests/test_golden_ext.py` replay scripted worlds and compare every request (messages and
tool schemas), the logs and the final context files with golden files recorded from the original code.

## Context Language Models (ideas and conventions only)

**Context Language Models** (Shao et al., 2026, arXiv 2609.37725; code: `facebookresearch/context-language-models`,
licensed CC BY-NC 4.0). SiamLM's context-file mechanism is our own reimplementation; **no code is copied from the CLM
repository**. Taken from the paper and repository as ideas and conventions:

- the file format: one block per turn, opened by a header line (CLM: `[[CTX_TURN i role=…]]`; ours:
  `[[TURN i role=…]]`; the parser also reads the CLM spelling);
- append-by-default: new turns are added and nothing is removed unless the model removes it;
- a token budget with a reserve, showing the model the enforced limit (budget − reserve), nudges at 25 / 50 / 75 %,
  and the edit gate (an edit may not grow the context past the limit);
- the size readout in the context (`[context: ~N/limit tokens]`) and the gist of the nudge texts, in our own words;
- not carrying out a move while the context is over the limit (CLM rolls back and retries; SiamLM holds the call), and
  ending a session that stays over the limit;
- editing the context file with bash in a container, with a time limit per command;
- echoing every editing call and its result into the context.

If you find wording that still follows CLM closely, credit it here as "adapted from CLM (CC BY-NC 4.0)" or reword it.

## LiveCodeBench (code, MIT)

**LiveCodeBench** (Jain et al., 2024; github.com/LiveCodeBench/LiveCodeBench, MIT, commit 28fef95; its licence is
in `siamlm/lcb/LICENSE`).
- `siamlm/lcb/testing_util.py` is copied unchanged.
- `siamlm/lcb/driver.py` follows `compute_code_generation_metrics.check_correctness`.
- `siamlm/lcb/prompts.py` copies the generic system message, the question template, the self-repair feedback text and
  the code extraction rule. The function signatures are adapted, and there is one documented change for programs
  that do not compile.

## Third-party data (not included)

No benchmark data is shipped. The fetch scripts download it from the upstream sources into a local folder
(`data/` by default, `SIAMLM_DATA` to change it). **Check the terms before you use or share the data.**

- **LiveCodeBench problems** (`scripts/fetch_lcb.py`): huggingface.co/datasets/livecodebench/code_generation_lite,
  pinned revision. The dataset card gives the licence only as "cc"; which licence applies is not stated.
- **CodeEval-Pro** (`scripts/fetch_codeeval_pro.py`): Yu, Zhao, Cohan, Zhang, "HumanEval Pro and MBPP Pro:
  Evaluating Large Language Models on Self-invoking Code Generation", Findings of ACL 2025, arXiv 2412.21199;
  github.com/CodeEval-Pro/CodeEval-Pro, commit 36f292e, `dataset/` (bigcodebench_lite_pro.json, humaneval_pro.json).
  The repository has no licence file (its README and pyproject say MIT for the code; nothing is said about the
  data). HumanEval-Pro builds on OpenAI's HumanEval (MIT).
- **BigCodeBench** (Zhuo et al., ICLR 2025; huggingface.co/datasets/bigcode/bigcodebench, Apache-2.0, revision
  b74c0d0, split v0.1.4): the unit tests of the 57 tasks that are the bases of BigCodeBench-Lite-Pro, extracted by
  the same fetch script. `docker/cep-grader.Dockerfile` pins library versions from BigCodeBench's
  `Requirements/requirements-eval.txt` (github.com/bigcode-project/bigcodebench, commit 09dd993).

What is ours (MIT, like the rest): the CodeEval-Pro grader and its driver (`siamlm/graders/`), the task and stream
builders (`siamlm/tasks/codeeval_pro.py`, `siamlm/tasks/cep_streams.py`), and the procedurally generated `state_chain`
task (`siamlm/tasks/state_chain.py`), which uses no data set.

## Harbor (formats followed; no code copied)

**Harbor** (harborframework.com; github.com/laude-institute/harbor, now github.com/harbor-framework/harbor; Apache-2.0;
read at commit 00dc6a7). SiamLM follows two of its published formats, with code of our own:

- the task directory format (`task.toml`, `instruction.md`, `environment/Dockerfile`, `tests/test.sh`, the reward in
  `/logs/verifier/reward.txt` or `reward.json`), read by `siamlm/tasks/harbor.py`; the canary-line stripping follows
  Harbor's documented behaviour;
- the Agent Trajectory Interchange Format (ATIF, `rfcs/0001-trajectory-format.md`, ATIF-v1.8), written by
  `siamlm/export.py`, including the RFC's context-management convention.

The example task `examples/harbor/count-errors` is ours (MIT). Its layout mirrors Harbor's `examples/tasks/hello-world`.

## Modulo (optional dependency, not included)

The experimental `steer` capability (`siamlm/steer.py`) uses Modulo (https://github.com/EigenformAI/modulo) as a library
for steering vectors only: its registry layout and its vector extraction (`extract_vector`, `save_and_register`,
and the starter vector definitions). It is installed with the `steer` extra from the commit pinned in pyproject.toml.
Modulo's README states the MIT licence; the repository has no LICENSE file at that commit. No Modulo code is copied
into SiamLM. The forward-hook steering in `siamlm/steer.py` is our own code; tracking absolute positions under a KV
cache follows the idea of Modulo's hooks. Vectors carry their own sources' licences.

## Other

- The `open` task's weak level follows the setup of a public post on attractor states in model self-conversation
  (LessWrong, Feb 2026: "No constraints. What would you like to explore?"), in our own words.
- Docker base images: `python:3.12-slim-bookworm` and `python:3.10-slim-bookworm`, pinned by digest.
