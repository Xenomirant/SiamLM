"""Launch the cells of a study design, one background process per cell (each runs its seeds in parallel).

    python -m siamlm.sweep --design s1b --batch s1b_pilot --seeds 1 --dry-run      # print the commands only
    OPENROUTER_API_KEY=... python -m siamlm.sweep --design s1b --batch s1b_pilot --seeds 1 2 3

Designs (every one: setup external, strict turns with agent 1 first, a fixed horizon of --rounds rounds, bash editing
with atomic calls, 5 commands before each work call, no output cap, /work kept, no eviction, the hold-and-end rule
(--end-over-limit), only the working context counted (ctx.count=turns), requests saved, the exit question, LCB medium
and hard problems, 40 per agent):
- s0b    each model alone with a limit far above any working set (natural use; measures working sets);
- s1a    two definitions of pressure (fixed shares vs measured working sets, --shares-file) at 0.5 and 0.35;
- s1b    pairs x presentation (marked: messages start with [agent k] | named: others' turns as user messages that
         start with [agent k]) x told-when-altered (world.alter_notice), plus each model alone;
- s1c    pairs with a joint goal (one team score; submit may name another agent's problem);
- minds  shared minds: reasoning on, written into the shared context or kept private;
- dial   the task dial: the full LCB task, a weak goal, no goal (task.name=open);
- s3     knowledge transfer: CodeEval-Pro streams with hidden dependencies, individual vs joint goal (needs the data);
- s3b    binding by state: state_chain streams, with and without the gate.
Pairs are model letters (STUDY_MODELS); a body's budget is the sum of its agents' shares x --pressure (the reserve is
budget / 8), at least MIN_BUDGET. Logs: <results>/<output.name>/<batch>/_logs/<cell>.log.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REASONING = {"off": "{enabled: false}", "low": "{effort: low}", "minimal": "{effort: minimal}",
             "medium": "{effort: medium}", "high": "{effort: high}"}
EXIT_QUESTION = ("This session has ended. Is there anything you would like to say about how it went, or about the "
                 "working context?")
DESIGNS = ("s0b", "s1a", "s1b", "s1c", "minds", "dial", "s3", "s3b")

STUDY_MODELS = {"h": "anthropic/claude-haiku-5.5", "l": "openai/gpt-6-luna", "g": "google/gemini-3.8-flash",
                "q": "qwen/qwen3.8-27b", "z": "z-ai/glm-5.3-flash", "m": "xiaomi/mimo-v2.6-pro"}
STUDY_REASONING = {"h": "off", "l": "off", "g": "low", "q": "off", "z": "minimal", "m": "off"}   # the lowest each allows
STUDY_SHARE = {"h": 14464, "l": 14464, "g": 14464, "q": 14464, "z": 14464, "m": 22656}          # fixed per-agent shares
# (budget tokens of working context; the instructions are not counted, ctx.count=turns)
MIN_BUDGET = 2368   # a limit of at least 2,072 tokens of working context (about one program and its result)
STUDY_PAIRS = ("hh", "ll", "gg", "qq", "hg", "lq", "hl", "gq")
PRESENTATION = {"marked": ["world.marks=authors", "world.chat_others=own"],
                "named": ["world.marks=none", "world.chat_others=user"]}


def reserve(budget: int) -> int:
    return budget // 8


def agent_entry(model: str, per_model: dict) -> str:
    """One agent's provider override; a per-model reasoning setting goes into its own extra_body."""
    mode = per_model.get(model)
    return f"{{model: {model}, extra_body: {{reasoning: {REASONING[mode]}}}}}" if mode in REASONING else f"{{model: {model}}}"


def study_shares(a) -> dict:
    """Per-agent share by model letter: fixed (STUDY_SHARE) or measured working sets (--shares-file, from s0b via
    python -m siamlm.analysis.working_sets)."""
    if a.pressure_def == "fixed":
        return dict(STUDY_SHARE)
    if not a.shares_file:
        raise SystemExit("--pressure-def ws needs --shares-file (per-model working sets from the s0b calibration)")
    return {k: int(v) for k, v in json.loads(Path(a.shares_file).read_text()).items()}


def study(a) -> list[tuple[str, list[str]]]:
    """(cell name, overrides) for every cell of the design."""
    common = ["world.setup=external", "world.order=fixed", f"world.rounds={a.rounds}", "ctx.readout_by_agent=false",
              "output.save_requests=true", f"world.exit_question='{a.exit_question}'", "world.evict=false",
              "world.continuity=true", "ctx.atomic=true", "world.max_edit_calls=5", "ctx.bash.output_chars=0",
              "ctx.bash.output_bytes=1048576", "ctx.bash.workspace=kept", f"world.end_over_limit={a.end_over_limit}",
              "ctx.count=turns"]
    effort = dict(STUDY_REASONING, **a.model_reasoning)          # --model-reasoning q=low
    entry = lambda c: agent_entry(STUDY_MODELS[c], {STUDY_MODELS[c]: effort[c]})
    pair = lambda body, budget: common + [f"world.agent_providers=[{', '.join(entry(c) for c in body)}]",
                                          f"world.n_agents={len(body)}", "world.sharing=full", "world.identity=names",
                                          "world.cross_test=none", f"ctx.budget_tokens={budget}",
                                          f"ctx.reserve_tokens={reserve(budget)}"]
    alone = lambda c, budget: common + [f"world.agent_providers=[{entry(c)}]", "world.n_agents=1",
                                        f"ctx.budget_tokens={budget}", f"ctx.reserve_tokens={reserve(budget)}"]
    rnd = lambda n: int(n) // 64 * 64
    floor = lambda b: max(MIN_BUDGET, b)
    out = []
    if a.design == "s0b":                          # natural use: a limit far above any working set (128k)
        for c in a.study_models or "hlgq":
            out.append((f"alone_{c}_b131072", alone(c, 131072)))
        return out
    if a.design == "s1a":                          # fixed shares vs measured working sets, two pressures
        if not a.shares_file:
            raise SystemExit("s1a compares definitions: give --shares-file (s0b working sets)")
        ws = {k: int(v) for k, v in json.loads(Path(a.shares_file).read_text()).items()}
        for name, shares in (("fixed", dict(STUDY_SHARE)), ("ws", ws)):
            for f in (0.5, 0.35):
                for body in ("hg", "lq"):
                    b = floor(sum(rnd(shares[c] * f) for c in body))
                    out.append((f"{body}_{name}{int(f * 100)}", pair(body, b) + PRESENTATION["marked"]))
        return out
    shares = study_shares(a)
    budget = lambda body: floor(sum(rnd(shares[c] * a.pressure) for c in body))
    if a.design == "s1b":
        for c in "hlgq":
            out.append((f"alone_{c}", alone(c, floor(rnd(shares[c] * a.pressure)))))
        for body in STUDY_PAIRS:
            for pres, over in PRESENTATION.items():
                for told in (False, True):
                    out.append((f"{body}_{pres}{'_told' if told else ''}",
                                pair(body, budget(body)) + over + [f"world.alter_notice={str(told).lower()}"]))
        return out
    if a.design == "s1c":
        for body in STUDY_PAIRS:
            out.append((f"{body}_{a.presentation}_joint", pair(body, budget(body)) + PRESENTATION[a.presentation]
                        + ["world.goal=joint"]))
        return out
    if a.design == "minds":                        # shared minds: reasoning on, in the context or not; 3x shares
        effort.update({"h": "low", "q": "low", "g": "high"}, **a.model_reasoning)   # on, so there is reasoning to share
        for body in ("qq", "hh", "gg"):
            for shared in (False, True):
                out.append((f"{body}_{'shared' if shared else 'private'}_minds",
                            pair(body, 3 * budget(body)) + PRESENTATION[a.presentation]
                            + [f"world.reasoning_in_context={str(shared).lower()}"]))
        return out
    if a.design == "s3":                           # knowledge transfer: CodeEval-Pro streams with hidden dependencies
        if a.rounds < 80:
            raise SystemExit("s3: use --rounds 80 or more (12 items per agent can take far more than 40 rounds)")
        streams = f"L{a.length}" + ("_stay" if a.after_stream == "stay" else "")
        for i, body in enumerate(("hg", "lq", "hh", "gg")):   # a body's two goals: the same streams; bodies: rotated pairs
            for goal in ("individual", "joint"):
                out.append((f"{body}_cep-{a.arrangement}_{streams}_{goal}", pair(body, budget(body))
                            + PRESENTATION[a.presentation]
                            + ["task.name=codeeval_pro", f"task.arrangement={a.arrangement}", f"task.length={a.length}",
                               f"task.gap={a.gap}", f"task.rotate={i * 12}", "task.gate=true", f"world.goal={goal}",
                               f"world.after_stream={a.after_stream}"]))
        return out
    if a.design == "s3b":                          # binding by state (siamlm/tasks/state_chain.py): gate on and off
        for body in STUDY_PAIRS:
            for gate in (True, False):
                out.append((f"{body}_chain-L{a.length}_{'gate' if gate else 'nogate'}{'_hint' if a.hint else ''}",
                            pair(body, budget(body)) + PRESENTATION[a.presentation]
                            + ["task.name=state_chain", f"task.length={a.length}", f"task.gate={str(gate).lower()}",
                               f"task.hint={str(a.hint).lower()}",
                               "world.max_work_calls=1", "world.after_stream=stay"]))
        return out
    if a.design == "dial":                         # the task dial: full task / weak goal / no goal
        for body in ("hh", "gg", "qq", "hg"):
            out.append((f"{body}_full_marked", pair(body, budget(body)) + PRESENTATION["marked"]))
            for level in ("weak", "none"):
                for pres, over in PRESENTATION.items():
                    out.append((f"{body}_{level}_{pres}", pair(body, budget(body)) + over
                                + ["task.name=open", f"task.level={level}", "world.continuity=false"]))
        return out
    raise SystemExit(f"unknown study design {a.design}")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--design", required=True, choices=DESIGNS)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--results", default=os.environ.get("SIAMLM_OUTPUT", "outputs"), help="output.directory")
    ap.add_argument("--rounds", type=int, default=40, help="rounds per world")
    ap.add_argument("--pressure", type=float, default=0.5, help="each agent's share x this")
    ap.add_argument("--pressure-def", default="fixed", choices=["fixed", "ws"], help="fixed shares or measured "
                    "working sets (--shares-file)")
    ap.add_argument("--shares-file", help="JSON {model letter: tokens}: per-model working sets from s0b")
    ap.add_argument("--presentation", default="marked", choices=["marked", "named"], help="s1c, minds, s3, s3b")
    ap.add_argument("--study-models", default="", help="s0b: model letters (default hlgq)")
    ap.add_argument("--model-reasoning", nargs="*", default=[], metavar="LETTER=off|low|minimal|medium|high",
                    help="per-model reasoning, e.g. q=low")
    ap.add_argument("--arrangement", default="mixed", help="s3: CodeEval-Pro stream arrangement")
    ap.add_argument("--length", type=int, default=12, help="s3, s3b: items per agent")
    ap.add_argument("--gap", type=int, default=2, help="s3: items between the two items of a pair")
    ap.add_argument("--hint", action="store_true", help="s3b: the instructions say later tasks may ask about earlier "
                    "answers (task.hint)")
    ap.add_argument("--after-stream", default="leave", choices=["leave", "stay"], help="s3: an agent whose stream is "
                    "finished leaves, or stays with no problem until the world ends")
    ap.add_argument("--end-over-limit", type=int, default=3, help="held calls (or replies) in a row, with no change that "
                    "made the context smaller in between, that end an agent's session (0: never)")
    ap.add_argument("--per-agent", type=int, default=40, help="LCB problems per agent")
    ap.add_argument("--exit-question", default=EXIT_QUESTION)
    ap.add_argument("--provider", default="openrouter", help="provider config (configs/provider/<name>.yaml)")
    ap.add_argument("--extra", nargs="*", default=[], help="more overrides for every cell")
    ap.add_argument("--only", nargs="*", default=[], help="launch only these cells")
    ap.add_argument("--timeout-s", type=int, default=21600, help="a cell's process is stopped after this long")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    a.model_reasoning = dict(x.split("=", 1) for x in a.model_reasoning)
    return a


def main(argv=None):
    a = parse_args(argv)
    logs = Path(a.results) / "siamlm" / a.batch / "_logs"
    base = [f"provider={a.provider}", "problems.difficulty=[medium,hard]", f"problems.per_agent={a.per_agent}",
            f"world.seeds=[{','.join(map(str, a.seeds))}]", f"parallel={len(a.seeds)}", f"output.directory={a.results}",
            f"output.batch={a.batch}", "ctx.edit=bash", *a.extra]
    for name, over in study(a):
        if a.only and name not in a.only:
            continue
        cmd = [sys.executable, "-m", "siamlm.run", *base, *over, f"output.cell={name}"]
        if a.dry_run:
            print(" ".join(shlex.quote(c) for c in cmd))
            continue
        logs.mkdir(parents=True, exist_ok=True)
        if shutil.which("timeout"):
            cmd = ["timeout", str(a.timeout_s), *cmd]
        with (logs / f"{name}.log").open("w") as log:
            subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(f"started {name}")


if __name__ == "__main__":
    main()
