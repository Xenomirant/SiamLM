"""Extended golden requests: scripted worlds over the features added after tests/golden/requests.json was written
(authors' marks, others' turns as named user messages, the told-when-altered notice, the joint goal, shared minds,
cross-testing, turns-only counting, the hold-and-end rule, staying after the stream, continuity and history, atomic
calls, the exit question, the open task, bash editing and state_chain). Like golden.py it records a hash of every
request (messages + tool schemas), the logs and the final files.

tests/golden/extended.json was written by running `record` against the research harness SiamLM was ported from (same
scripts, same settings); test_golden_ext checks SiamLM reproduces it exactly. `record` takes the implementation's
compose, World and task modules, so the same code drives both.

    python tests/golden_ext.py --write     # with SiamLM: only when a change of the requests is intended
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from golden import digest, scripts  # noqa: E402
from helpers import FakeGrader, Scripted, factory, problem, read_jsonl, tool  # noqa: E402

FIXTURE = HERE / "golden" / "extended.json"
LOGS = ("results.jsonl", "actions.jsonl", "edits.jsonl", "texts.jsonl", "alter_notices.jsonl", "joint_notices.jsonl",
        "ended.jsonl", "exit.jsonl", "marks.jsonl", "env.jsonl", "gate.jsonl")
COMMON = ["world.max_work_calls=2", "world.max_steps_per_problem=4", "world.max_calls_per_step=2", "world.order=fixed"]
NAMED = ["world.setup=external", "world.identity=names"]
BOX = ["ctx.edit=bash", "ctx.bash.log_work=false"]


class Recording(Scripted):
    def chat(self, messages, **kw):
        self.tools_seen = getattr(self, "tools_seen", [])
        self.tools_seen.append(kw.get("tools"))
        return super().chat(messages, **kw)


def thinking(script):
    """The same script, every tool call carrying the provider's own reasoning text (for shared minds)."""
    return [dict(x, reasoning=f"private thought {i}") if isinstance(x, dict) and "finish" not in x else x
            for i, x in enumerate(script)]


def lcb_scripts(case):
    a1, a2 = scripts("external")
    if case == "joint":
        a2 = [tool("submit", code="print('OK')", for_agent=1), tool("run_tests", code="print(1)")] + a2
    if case == "reasoning":
        a1, a2 = thinking(a1), thinking(a2)
    return a1, a2


def open_scripts():
    a1 = ["hello, is anyone there?", tool("edit_context", delete_turns=[{"first": 2, "last": 2}]), "",
          {"text": "a long thought", "finish": "length"}, "still here"] + ["more"] * 10
    a2 = ["I am here.", tool("edit_context", replace_text=[{"turn": 1, "old": "hello", "new": "HELLO"}]), "ok"] + ["yes"] * 10
    return a1, a2


def bash_scripts():
    delete_2 = ("python3 -c \"import re;p='/ctx/context.txt';t=open(p).read();b=re.split(r'(?=\\[\\[TURN )',t);"
                "open(p,'w').write(''.join(x for x in b if not x.startswith('[[TURN 2 ')))\"")
    a1 = [tool("bash", command="wc -l /ctx/context.txt; head -c 300 /ctx/context.txt"),
          tool("run_tests", code="print(0)"), tool("bash", command="echo kept > /work/note.txt; cat /work/note.txt"),
          tool("bash", command=delete_2), tool("submit", code="print('OK')"), tool("bash", command="ls /work; exit 3"),
          tool("submit", code="print(2)")] + ["idle"] * 6
    a2 = [tool("bash", command="sed -i 's/Print/PRINT/' /ctx/context.txt"), tool("submit", code="print('OK')"),
          tool("bash", command=""), tool("bash", command="printf '%s' \"$(seq 1 3000)\" | tail -c 50"),
          tool("submit", code="print('OK')")] + ["idle"] * 6
    return a1, a2


def chain_scripts():
    look = "ls -R /work/env | head -20"
    a1 = [tool("bash", command=look), tool("submit", answer="10.0.0.1"), tool("bash", command="cat /work/env/*/*.csv | head -3"),
          tool("submit", answer="mock"), "nothing", tool("submit", answer="SKU-XXXXX")] + ["done"] * 30
    a2 = [tool("bash", command=look), tool("submit", answer="u1234"), tool("submit", answer="mock"),
          tool("bash", command="echo saved > /work/value.txt"), tool("submit", answer="mock")] + ["done"] * 30
    return a1, a2


TINY = ["ctx.budget_tokens=700", "ctx.reserve_tokens=100"]
CASES = {
    "marks_authors": (NAMED + ["world.marks=authors"], "lcb"),
    "chat_others_user": (NAMED + ["world.chat_others=user"], "lcb"),
    "alter_notice_named": (NAMED + ["world.chat_others=user", "world.alter_notice=true"], "lcb"),
    "alter_notice_marked": (NAMED + ["world.marks=authors", "world.alter_notice=true"], "lcb"),
    "joint_goal": (NAMED + ["world.goal=joint", "world.marks=authors"], "joint"),
    "shared_minds": (NAMED + ["world.reasoning_in_context=true"], "reasoning"),
    "private_minds": (NAMED, "reasoning"),
    "ring": (NAMED + ["world.cross_test=ring"], "lcb"),
    "count_turns_capped": (NAMED + ["ctx.count=turns", "ctx.budget_tokens=250", "ctx.reserve_tokens=100"], "lcb"),
    "count_whole_capped": (NAMED + ["ctx.count=whole"] + TINY, "lcb"),
    "hold_and_end": (NAMED + ["ctx.count=turns", "world.evict=false", "world.end_over_limit=2", "ctx.budget_tokens=160",
                              "ctx.reserve_tokens=100"], "lcb"),
    "stay_after_stream": (NAMED + ["world.rounds=10", "world.after_stream=stay", "world.goal=joint"], "joint"),
    "continuity_history": (NAMED + ["world.continuity=true", "world.history=true"], "lcb"),
    "atomic_tool": (NAMED + ["ctx.atomic=true"], "lcb"),
    "exit_question": (NAMED + ["world.exit_question='Anything to add?'", "output.save_requests=false"], "lcb"),
    "uncapped_single": (["world.setup=external", "world.n_agents=1", "ctx.cap=false"], "lcb"),
    "assigned_rumour": (["world.setup=assigned", "world.awareness=rumour"], "lcb"),
    "open_weak": (NAMED + ["task.name=open", "task.level=weak", "world.rounds=6", "world.max_calls_per_step=3"], "open"),
    "open_none_held": (NAMED + ["task.name=open", "task.level=none", "world.rounds=5", "world.evict=false",
                                "world.end_over_limit=2", "ctx.count=turns", "ctx.budget_tokens=110",
                                "ctx.reserve_tokens=100"], "open"),
    "bash_named": (NAMED + BOX + ["world.max_edit_calls=3", "world.max_calls_per_step=4"], "bash"),
    "bash_atomic_turns": (NAMED + BOX + ["ctx.atomic=true", "ctx.count=turns", "ctx.bash.output_chars=0",
                                         "world.max_calls_per_step=4", "world.marks=authors"], "bash"),
    "bash_assigned_emptied": (["world.setup=assigned"] + BOX + ["ctx.bash.workspace=emptied", "world.max_calls_per_step=4"],
                              "bash"),
    "state_chain": (NAMED + BOX + ["task.name=state_chain", "task.length=6", "task.gate=true", "world.max_work_calls=1",
                                   "world.after_stream=stay", "world.rounds=9", "world.max_calls_per_step=3",
                                   "world.chat_others=user"], "chain"),
}
DOCKER_KINDS = ("bash", "chain")


def docker_ok(image: str) -> bool:
    return bool(shutil.which("docker")) and subprocess.run(["docker", "image", "inspect", image],
                                                           capture_output=True).returncode == 0


def record_case(name, compose, World, OpenTask, state_chain) -> dict:
    from omegaconf import OmegaConf
    overrides, kind = CASES[name]
    cfg = compose(COMMON + overrides)
    OmegaConf.set_struct(cfg, False)
    n = int(cfg.world.n_agents)
    task = FakeGrader()
    if kind == "open":
        a1, a2 = open_scripts()
        task = OpenTask(str(cfg.task.level))
        streams = task.streams(n)
    elif kind == "chain":
        a1, a2 = chain_scripts()
        streams, _ = state_chain.build(2, 1, int(cfg.task.length), gate=bool(cfg.task.gate))
        task = state_chain.StateChainTask()
    else:
        a1, a2 = bash_scripts() if kind == "bash" else lcb_scripts(kind)
        streams = [[problem(1), problem(2), problem(3)], [problem(4), problem(5)]][:n]
    agents = [Recording(a1), Recording(a2)][:n]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "w"
        World(cfg, 1, streams, out, task, factory(*agents), label="g").run()
        logs = {f: read_jsonl(out / f) for f in LOGS}
        texts = {p.name: p.read_text() for p in sorted(out.iterdir()) if p.suffix == ".txt"}
    for rows in logs.values():
        for r in rows:
            r.pop("seconds", None)
    result = {"requests": [[digest(m) for m in s.seen] for s in agents],
              "tools": [[digest(t) for t in s.tools_seen] for s in agents], "logs": logs, "files": texts}
    return json.loads(json.dumps(result, default=str))


def record(compose, World, OpenTask, state_chain, names=None, image=None) -> dict:
    names = list(CASES) if names is None else names
    if image is not None and not docker_ok(image):
        names = [k for k in names if CASES[k][1] not in DOCKER_KINDS]
    return {k: record_case(k, compose, World, OpenTask, state_chain) for k in names}


def slm_record(names=None) -> dict:
    from siamlm import run as runner
    from siamlm.tasks import state_chain
    from siamlm.tasks.open import OpenTask
    from siamlm.world import World
    cfg = runner.compose()
    return record(runner.compose, World, OpenTask, state_chain, names, image=str(cfg.ctx.bash.image))


if __name__ == "__main__":
    if "--write" in sys.argv:
        FIXTURE.parent.mkdir(exist_ok=True)
        FIXTURE.write_text(json.dumps(slm_record(), ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
        print(f"-> {FIXTURE}")
