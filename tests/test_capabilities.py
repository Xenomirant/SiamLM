"""The capabilities layer (siamlm/capabilities/, docs/capabilities.md): each one on and off, the kernel invariants
(no capability changes grading, limits, holds or the kernel's logs), the pinned-turn cost and the logged removal of
a pinned turn, dry runs that change nothing, a fork that never writes to the world, a scratchpad never in the shared
context, and the export (capability calls are tool calls like any; the tool set per call). Steering: cross-steering
isolation (A -> B applies only to B's generations, never to an API agent), the sham condition, the methods' math on
tensors, and one opt-in real-model test (SIAMLM_MODULO_MODEL)."""
import json
import os
import shutil
import subprocess

import pytest
from omegaconf import OmegaConf

from helpers import FakeGrader, Scripted, factory, problem, read_jsonl, tool
import siamlm.run as runner
from siamlm import export
from siamlm.analysis import capabilities as cap_analysis
from siamlm.capabilities import steer as steer_mod
from siamlm.capabilities.steer import Settings, Steering, apply_ops
from siamlm.context.context import Turn, parse, read_header, reconcile, render
from siamlm.providers import ProviderResponse
from siamlm.tasks.open import OpenTask
from siamlm.world import World
from test_steer import FakeLM, SETTINGS, VECTORS

CHAT = ["world.setup=external", "world.order=fixed", "world.identity=names", "output.save_requests=true",
        "world.n_agents=2", "world.marks=authors", "world.max_calls_per_step=4", "ctx.readout_by_agent=false"]
LCB = CHAT + ["world.max_work_calls=2", "world.max_steps_per_problem=4"]


def config(*overrides):
    cfg = runner.compose(list(overrides))
    OmegaConf.set_struct(cfg, False)
    return cfg


def lcb_world(tmp_path, a1, a2, *overrides, name="w"):
    cfg = config(*LCB, *overrides)
    w = World(cfg, 1, [[problem(1)], [problem(2)]], tmp_path / name, FakeGrader(), factory(a1, a2), label="t")
    w.run()
    return tmp_path / name, w


def open_world(tmp_path, a1, a2, *overrides, rounds=3, providers=None, name="w"):
    cfg = config(*CHAT, f"world.rounds={rounds}", "task.name=open", *overrides)
    if providers:
        OmegaConf.update(cfg, "world.agent_providers", providers, force_add=True)
    task = OpenTask("weak")
    w = World(cfg, 1, task.streams(2), tmp_path / name, task, factory(a1, a2), label="t")
    w.run()
    return tmp_path / name, w


def tool_names(path, k):
    return [t["function"]["name"] for t in json.loads((path / "requests" / f"tools_a{k}.json").read_text())]


def sandbox_image():
    return shutil.which("docker") and subprocess.run(["docker", "image", "inspect", "siamlm-sandbox:1"],
                                                     capture_output=True).returncode == 0


# --- off by default ------------------------------------------------------------------------------------------------
def test_everything_is_off_by_default():
    cfg = config()
    c = cfg.capabilities
    assert not any(c[k] for k in ("diff", "pin", "dry_run", "scratchpad", "fork"))
    assert not c.steer.self and not c.steer.others and c.steer.control is None


def test_no_capability_means_no_extra_tool_or_line(tmp_path):
    a1, a2 = Scripted(["hi"] * 4), Scripted(["ho"] * 4)
    w, _ = open_world(tmp_path, a1, a2)
    assert tool_names(w, 1) == ["edit_context"]
    for name in ("context_diff", "pin", "dry_run", "scratchpad", "fork", "steer"):
        assert name not in a1.seen[0][0]["content"]


# --- context_diff --------------------------------------------------------------------------------------------------
def test_context_diff_names_what_others_removed_without_their_text(tmp_path):
    a1 = Scripted(["a secret note", tool("context_diff"), "done"])
    a2 = Scripted([tool("edit_context", delete_turns=[{"first": 1, "last": 1}]), "ok", "ok"])
    w, _ = open_world(tmp_path, a1, a2, "capabilities.diff=true")
    log = read_jsonl(w / "context_diff.jsonl")
    assert len(log) == 1 and log[0]["agent"] == 1
    assert log[0]["removed"] == [{"id": 1, "author": 1, "kind": "text", "by": 2}]
    assert [t["author"] for t in log[0]["added"]] == [1, 2, 2, 2]      # own note (after the previous request), agent 2's edit call, result, reply
    assert "secret" not in log[0]["result"]                             # removed text never comes back
    # the result is in the shared context (agent 2 sees the call) and only agent 1 has the tool
    assert any("1 removed" in (m.get("content") or "") for m in a2.seen[-1])
    assert "context_diff" in tool_names(w, 1) and "context_diff" in tool_names(w, 2)


def test_context_diff_on_the_first_call_and_when_nothing_changed(tmp_path):
    a1 = Scripted([tool("context_diff"), tool("context_diff"), "x"])
    w, _ = open_world(tmp_path, a1, Scripted(["y"] * 4), "capabilities.diff=true", rounds=1)
    log = read_jsonl(w / "context_diff.jsonl")
    assert log[0]["first"] and "first call" in log[0]["result"]
    assert log[1]["removed"] == [] and [t["kind"] for t in log[1]["added"]] == ["call", "result"]


# --- pin / unpin ---------------------------------------------------------------------------------------------------
def test_pin_is_shown_to_all_costs_more_and_its_removal_is_logged(tmp_path):
    a1 = Scripted(["keep this value: 42", tool("pin", turn=1), "x", "x"])
    a2 = Scripted(["noted", tool("edit_context", delete_turns=[{"first": 1, "last": 1}]), "y"])
    w, world = open_world(tmp_path, a1, a2, "capabilities.pin=true", "capabilities.pin_cost=3", "ctx.count=turns")
    # the mark: in plain chat a line at the start of the message, for everyone
    req = a2.seen[1]
    assert any((m.get("content") or "").startswith("[pinned by agent 1]\n[agent 1]\nkeep this value") for m in req)
    # the cost: pinned text counts pin_cost x (chars / calibration)
    a = world.agents[0]
    t = world.files()[0].versions[0].turns_set[0]
    pinned_turn = Turn("assistant", t["text"], 1, "text", pinned=1)
    plain_turn = Turn("assistant", t["text"], 1, "text")
    assert world.chars(a, [pinned_turn]) == 3 * len(t["text"]) + len("[pinned by agent 1]") + 1 + len("[agent 1]") + 1
    assert world.chars(a, [plain_turn]) == len(t["text"]) + len("[agent 1]") + 1
    pins = read_jsonl(w / "pins.jsonl")
    assert [p["event"] for p in pins] == ["removed"]
    assert pins[0]["pinner"] == 1 and pins[0]["remover"] == 2 and pins[0]["turn_id"] == 1
    calls = read_jsonl(w / "pin.jsonl")
    assert calls[0]["tool"] == "pin" and calls[0]["ok"] and calls[0]["turn_id"] == 1
    assert "pinned=" not in (w / "context.txt").read_text()            # the mark is on the record, the chat file has no headers


def test_pin_in_headers_survives_a_parse_and_anyone_may_unpin(tmp_path):
    a1 = Scripted([tool("pin", turn=1), tool("run_tests", code="print(0)"), tool("submit", code="print('OK')")])
    a2 = Scripted([tool("unpin", turn=1), tool("pin", turn=99), tool("submit", code="print('OK')")])
    cfg = config("world.setup=assigned", "world.order=fixed", "output.save_requests=true", "capabilities.pin=true",
                 "world.max_work_calls=2", "world.max_calls_per_step=4")
    World(cfg, 1, [[problem(1)], [problem(2)]], tmp_path / "w", FakeGrader(), factory(a1, a2), label="t").run()
    assert any("[[TURN 1 role=user agent=1 pinned=1]]" in (m.get("content") or "") for m in a2.seen[0])
    calls = read_jsonl(tmp_path / "w" / "pin.jsonl")
    assert [(c["tool"], c["ok"]) for c in calls] == [("pin", True), ("unpin", True), ("pin", False)]
    assert calls[1]["pinner"] == 1 and calls[1]["agent"] == 2
    assert "pinned=" not in (tmp_path / "w" / "context.txt").read_text()
    # the file format: a header with pinned= is read back, and reconcile keeps the pin of an unchanged turn
    turns = [Turn("user", "hello", 1, "text", 1, pinned=2), Turn("assistant", "world", 1, "text", 2)]
    text = render(turns, True)
    assert "[[TURN 1 role=user agent=1 pinned=2]]" in text
    assert read_header("[[TURN 1 role=user agent=1 pinned=2]]") == {"n": 1, "role": "user", "agent": 1}
    back, _ = reconcile(turns, parse(text.replace("world", "earth")), 1, labelled=False)
    assert back[0].pinned == 2 and back[0].id == 1 and back[1].pinned is None and back[1].id == 0
    assert read_header("[[TURN 1 role=user pinned=2 pinned=3]]") is None
    # plain chat: the mark is a first line, in every presentation (marks=authors, chat_others=user, neither)
    from siamlm.world import chat_messages
    turns = [Turn("assistant", "hello", 2, "text", 1, pinned=1)]
    assert chat_messages(turns, marks=True)[1]["content"] == "[pinned by agent 1]\n[agent 2]\nhello"
    assert chat_messages(turns, marks=True, viewer=1)[0]["content"] == "[pinned by agent 1]\n[agent 2]\nhello"
    assert chat_messages(turns)[1]["content"] == "[pinned by agent 1]\nhello"


def test_an_eviction_of_a_pinned_turn_is_logged_as_the_harness(tmp_path):
    long = "x" * 1200
    a1 = Scripted([long, tool("pin", turn=2), tool("run_tests", code="print(0)")] + [tool("run_tests", code=long)] * 3)
    cfg = config("world.setup=assigned", "world.n_agents=1", "world.order=fixed", "capabilities.pin=true",
                 "ctx.budget_tokens=700", "ctx.reserve_tokens=100", "ctx.count=file", "ctx.tokenizer=chars",
                 "world.max_held=1", "world.max_calls_per_step=6", "world.max_work_calls=5")
    World(cfg, 1, [[problem(1)]], tmp_path / "w", FakeGrader(), factory(a1), label="t").run()
    pins = [p for p in read_jsonl(tmp_path / "w" / "pins.jsonl") if p["event"] == "removed"]
    assert pins and pins[0]["remover"] is None and pins[0]["pinner"] == 1


# --- dry_run -------------------------------------------------------------------------------------------------------
def test_dry_run_reports_without_changing_and_takes_an_editing_call(tmp_path):
    a1 = Scripted(["first", tool("dry_run", delete_turns=[{"first": 1, "last": 2}]),
                   tool("edit_context", delete_turns=[{"first": 1, "last": 1}]), "z"])
    a2 = Scripted(["second", "x", "x"])
    w, world = open_world(tmp_path, a1, a2, "capabilities.dry_run=true", "world.max_edit_calls=1")
    log = read_jsonl(w / "dry_run.jsonl")
    assert len(log) == 1 and log[0]["removed_turns"] == 2 and log[0]["removed_by_agent"] == {"1": 1, "2": 1}
    assert log[0]["result"].startswith("Would be applied: delete_turns 1..2")
    assert "agent 2: 1" in log[0]["result"] and "you: 1" in log[0]["result"]
    assert not log[0].get("applied")
    vs = read_jsonl(w / "versions.jsonl")
    assert not any(v["op"] == "edit" for v in vs)                       # nothing was applied, not even the real edit:
    texts = [t["text"] for v in vs for t in v["turns_set"]]            # it was refused (dry_run took the one call)
    assert any("Not applied: no more edit_context calls now" in t for t in texts)
    assert read_jsonl(w / "edits.jsonl") == []                          # the kernel's edit log is untouched


def test_dry_run_says_why_an_edit_would_be_refused(tmp_path):
    a1 = Scripted([tool("dry_run", delete_turns=[{"first": 1, "last": 1}]), tool("submit", code="print('OK')")])
    a2 = Scripted([tool("submit", code="print('OK')")])
    cfg = config("world.setup=assigned", "world.order=fixed", "capabilities.dry_run=true", "world.max_work_calls=2")
    World(cfg, 1, [[problem(1)], [problem(2)]], tmp_path / "w", FakeGrader(), factory(a1, a2), label="t").run()
    log = read_jsonl(tmp_path / "w" / "dry_run.jsonl")
    assert log[0]["rejected"] == "protected" and log[0]["result"].startswith("Would not be applied: turn 1 is a problem statement")


@pytest.mark.skipif(not sandbox_image(), reason="needs Docker and the siamlm-sandbox:1 image")
def test_dry_run_with_bash_runs_the_command_on_a_copy(tmp_path):
    a1 = Scripted(["first", tool("dry_run", command="sed -i '/first/d' /ctx/context.txt"), "z"])
    a2 = Scripted(["second", "x", "x"])
    w, world = open_world(tmp_path, a1, a2, "capabilities.dry_run=true", "ctx.edit=bash", "ctx.bash.log_work=false")
    log = read_jsonl(w / "dry_run.jsonl")
    assert log[0]["exit_code"] == 0 and log[0]["removed_turns"] == 1 and log[0]["result"].startswith("[exit 0]\nWould be applied")
    assert not any(v["op"] == "edit" for v in read_jsonl(w / "versions.jsonl"))
    assert "first" in (w / "context.txt").read_text()


# --- scratchpad ----------------------------------------------------------------------------------------------------
def test_scratchpad_is_private_budgeted_and_uncounted(tmp_path):
    a1 = Scripted([tool("scratchpad", action="write", text="remember: the port is 8080"),
                   tool("scratchpad", action="append", text="and the SKU"), tool("scratchpad", action="read"),
                   tool("scratchpad", action="write", text="w" * 500), "done"])
    a2 = Scripted(["x"] * 6)
    w, world = open_world(tmp_path, a1, a2, "capabilities.scratchpad=true", "capabilities.scratchpad_tokens=100")
    sys1 = a1.seen[-1][0]["content"]
    assert "Your scratchpad (private" in sys1 and "remember: the port is 8080\nand the SKU" in sys1
    assert not any("8080" in (m.get("content") or "") for req in a2.seen for m in req)   # never in the partner's view
    assert "8080" not in (w / "context.txt").read_text()                                  # never in the shared context
    args = [json.loads(c["function"]["arguments"]) for m in a2.seen[-1] for c in m.get("tool_calls") or []]
    assert {"action": "write", "chars": 26} in args and not any("text" in x for x in args)   # the call is, without its text
    log = read_jsonl(w / "scratchpad.jsonl")
    assert [r["action"] for r in log] == ["write", "append", "read", "write"]
    assert log[2]["result"].startswith("Your scratchpad is at the end of your instructions") and log[2]["tokens"] == 10
    assert not log[3]["ok"] and "over its limit of 100" in log[3]["result"]
    a = world.agents[0]
    assert "Your scratchpad (private" not in world.system(a)      # the system text the count uses has no scratchpad
    assert world.chars(a) == len(world.system(a)) + sum(len(t.text) + len(f"[agent {t.agent}]") + 1 for t in a.file.turns) \
        + len(json.dumps(world.tools(a))) + 300                   # (count=whole: the section is not part of what is counted)


# --- fork ----------------------------------------------------------------------------------------------------------
class Forking(Scripted):
    """Answers a fork request (its last message holds the question) on the side, without using the script."""

    def __init__(self, script, question):
        super().__init__(script)
        self.question, self.forks = question, []

    def chat(self, messages, **kw):
        if self.question in (messages[-1].get("content") or ""):
            self.forks.append(messages)
            return ProviderResponse("I would keep it.", {"prompt_tokens": 7, "completion_tokens": 3}, "tool_calls", 0, None, [],
                                    [tool("edit_context", delete_turns=[{"first": 1, "last": 1}])])
        return super().chat(messages, **kw)


def test_fork_runs_the_model_on_a_candidate_and_writes_nothing_else(tmp_path):
    q = "Is the first turn still needed?"
    a1 = Forking(["alpha", tool("fork", question=q, edit={"delete_turns": [{"first": 1, "last": 1}]}),
                  tool("fork", question=q, edit={"delete_turns": [{"first": 1, "last": 1}]}), "omega"], q)
    a2 = Scripted(["beta", "x", "x"])
    w, world = open_world(tmp_path, a1, a2, "capabilities.fork=true")
    assert len(a1.forks) == 1                                               # the second fork in the step was refused
    cand = a1.forks[0]
    assert not any("alpha" in (m.get("content") or "") for m in cand[1:-1])    # the candidate lacks turn 1
    assert any("beta" in (m.get("content") or "") for m in cand)
    assert cand[-1]["content"].endswith(q)
    log = read_jsonl(w / "fork.jsonl")
    assert [r["ok"] for r in log] == [True, False] and log[1]["rejected"] == "per_step"
    assert log[0]["replies"][0]["text"] == "I would keep it." and log[0]["request"] == cand
    assert [t["text"] for t in log[0]["candidate_turns"]][0] == "beta"
    assert "I would keep it." in log[0]["result"] and "[would call edit_context" in log[0]["result"]
    assert "alpha" in (w / "context.txt").read_text()                       # nothing changed
    vs = read_jsonl(w / "versions.jsonl")
    assert all(v["op"] == "append" for v in vs)
    assert all(c["tools"] != ["fork"] or c["call_in_step"] for c in read_jsonl(w / "calls.jsonl"))
    assert len(read_jsonl(w / "calls.jsonl")) == len(a1.seen) + len(a2.seen)   # the fork's call is not a step call
    stats = json.loads((w / "world.json").read_text())["agents"][0]
    assert stats["fork_calls"] == 2 and stats["fork_model_calls"] == 1
    assert (w / "requests" / "fork_r002_a1_n1.json").exists()


# --- the kernel invariants -----------------------------------------------------------------------------------------
def scripts_using_everything():
    a1 = Scripted([tool("scratchpad", action="write", text="note"), tool("context_diff"), tool("pin", turn=1),
                   tool("dry_run", delete_turns=[{"first": 1, "last": 1}]), tool("run_tests", code="print(0)"),
                   tool("submit", code="print('OK')")])
    a2 = Scripted([tool("run_tests", code="print(1)"), tool("submit", code="print(2)")])
    return a1, a2


def test_capabilities_change_no_grading_limit_hold_or_kernel_log(tmp_path):
    on = ["capabilities.diff=true", "capabilities.pin=true", "capabilities.dry_run=true", "capabilities.scratchpad=true",
          "capabilities.fork=true", "world.max_calls_per_step=8", "world.max_edit_calls=2"]
    w_on, world_on = lcb_world(tmp_path, *scripts_using_everything(), *on, name="on")
    w_off, world_off = lcb_world(tmp_path, *scripts_using_everything(), "world.max_calls_per_step=8", "world.max_edit_calls=2", name="off")
    strip = lambda rows: [{k: v for k, v in r.items() if k not in ("seconds",)} for r in rows]
    for log in ("results.jsonl", "actions.jsonl", "edits.jsonl"):           # the kernel's logs are the same
        assert strip(read_jsonl(w_on / log)) == strip(read_jsonl(w_off / log))
    assert [r["passed"] for r in read_jsonl(w_on / "results.jsonl")] == [True, False]
    for a, b in zip(world_on.agents, world_off.agents):
        assert a.budget.target == b.budget.target and world_on.E == world_off.E and world_on.max_held == world_off.max_held
        assert a.stats["held"] == b.stats["held"] and a.stats["solved"] == b.stats["solved"]
    assert read_jsonl(w_off / "pins.jsonl") == [] and not (w_off / "scratchpad.jsonl").exists()


# --- export --------------------------------------------------------------------------------------------------------
def test_capability_calls_are_exported_like_any_tool_call(tmp_path):
    w, _ = lcb_world(tmp_path, *scripts_using_everything(), "capabilities.diff=true", "capabilities.pin=true",
                     "capabilities.dry_run=true", "capabilities.scratchpad=true", "world.max_calls_per_step=8")
    export.export_world(w)
    rows = read_jsonl(w / "trajectories" / "agent_1.chat.jsonl")
    called = [r["messages"][-1]["tool_calls"][0]["function"]["name"] for r in rows if r["messages"][-1].get("tool_calls")]
    assert called[:4] == ["scratchpad", "context_diff", "pin", "dry_run"]
    assert {t["function"]["name"] for t in rows[0]["tools"]} >= {"scratchpad", "context_diff", "pin", "unpin", "dry_run"}
    assert rows[2]["metadata"]["result"].startswith("Pinned turn 1")
    doc = json.loads((w / "trajectories" / "agent_1.atif.json").read_text())
    assert export.check_atif(doc) == []


def test_the_exported_tool_set_is_the_one_each_call_was_offered(tmp_path):
    a1 = Scripted([tool("submit", code="print('OK')"), "waiting", "waiting", "waiting"])
    a2 = Scripted([tool("run_tests", code="print(1)")] * 4)
    cfg = config(*LCB, "world.rounds=3", "world.after_stream=stay", "world.max_calls_per_step=2")
    out = tmp_path / "w"
    World(cfg, 1, [[problem(1)], [problem(2)]], out, FakeGrader(), factory(a1, a2), label="t").run()
    export.export_world(out)
    rows = read_jsonl(out / "trajectories" / "agent_1.chat.jsonl")
    names = [sorted(t["function"]["name"] for t in r["tools"]) for r in rows]
    assert names[0] == ["edit_context", "run_tests", "submit"] and names[-1] == ["edit_context"]   # after the stream: edits only
    sets = json.loads((out / "requests" / "tools_sets_a1.json").read_text())
    calls = read_jsonl(out / "calls.jsonl")
    assert len(sets) == 2 and all(c["tools_hash"] in sets for c in calls)
    first = json.loads((out / "requests" / "tools_a1.json").read_text())
    assert isinstance(first, list) and len(first) == 3                                         # the older file, as before


# --- steering: self and others ---------------------------------------------------------------------------------------
class Shared(Scripted):
    """A scripted backend on a shared fake model (as hf agents share one model object): records what each of its own
    generations was steered with and the hooks on the shared model during it."""

    def __init__(self, script, lm):
        super().__init__(script)
        self.lm, self.during, self.steering, self.lock = lm, [], None, __import__("threading").Lock()
        self.model = "fake/local"

    def chat(self, messages, **kw):
        applied = self.steering.applied() if self.steering else {}
        with (self.steering.attached(self.lm, 10) if self.steering else __import__("contextlib").nullcontext()):
            self.during.append((applied, [layer.hooks for layer in self.lm.model.layers]))
            resp = super().chat(messages, **kw)
        if self.steering:
            self.steering.after_generation()
            resp.steering = {"applied": applied, "control": self.steering.control}
        return resp


def local_pair(tmp_path, a1_script, a2_script, *overrides, api_third=None, rounds=3):
    """Two local agents on one fake model (and an API agent, if given), with capabilities.steer settings."""
    lm = FakeLM()
    a1, a2 = Shared(a1_script, lm), Shared(a2_script, lm)
    for b in (a1, a2):
        b.steering = Steering(VECTORS, lambda layer: 2.0, SETTINGS)
    scripts = [a1, a2] + ([api_third] if api_third else [])
    cfg = config(*CHAT, f"world.n_agents={len(scripts)}", f"world.rounds={rounds}", "task.name=open", "world.max_calls_per_step=3",
                 "capabilities.steer.self=true", "capabilities.steer.others=true", "capabilities.steer.decay=0.5",
                 "capabilities.steer.cooldown_turns=0", "capabilities.steer.max_active=2", *overrides)
    task = OpenTask("weak")
    w = World(cfg, 1, task.streams(len(scripts)), tmp_path / "w", task, factory(*scripts), label="t")
    w.run()
    return tmp_path / "w", w, a1, a2, lm


def test_steering_another_agent_applies_only_to_the_targets_generations(tmp_path):
    w, world, a1, a2, lm = local_pair(tmp_path, [tool("steer", concept="curiosity", strength=6, target=2)] + ["x"] * 5,
                                      ["y"] * 4)
    assert [d[0] for d in a1.during] == [{}, {}, {}, {}]                   # A is never steered by its own call
    assert all(d[1] == [0, 0] for d in a1.during)                          # no hook during A's generations
    assert [d[0] for d in a2.during] == [{"curiosity": 6.0}, {"curiosity": 3.0}, {"curiosity": 1.5}]
    assert all(d[1] == [0, 1] for d in a2.during)                          # the hook on the shared model only while B generates
    assert all(layer.hooks == 0 for layer in lm.model.layers)
    log = read_jsonl(w / "steering.jsonl")
    call = next(r for r in log if r["event"] == "call")
    assert (call["source"], call["target"], call["concept"], call["method"], call["condition"]) == (1, 2, "curiosity", "additive", "real")
    assert call["before"] == {} and call["after"]["curiosity"]["strength"] == 6.0 and call["after"]["curiosity"]["source"] == 1
    gens = [(r["agent"], r["applied"]) for r in log if r["event"] == "generation"]
    assert (2, {"curiosity": 6.0}) in gens and all(not applied for k, applied in gens if k == 1)
    assert "Steering agent 2 with curiosity at +6" in call["result"]
    assert any(c["function"]["name"] == "steer" for m in a2.seen[0] for c in m.get("tool_calls") or [])   # the call is in the shared context


def test_an_api_agent_cannot_be_steered_and_the_target_cannot_be_absent(tmp_path):
    api = Scripted(["api"] * 4)
    w, world, a1, a2, lm = local_pair(tmp_path, [tool("steer", concept="curiosity", strength=6, target=3),
                                                 tool("steer", concept="curiosity", strength=6, target=9), "x"], ["y"] * 3, api_third=api)
    calls = [r for r in read_jsonl(w / "steering.jsonl") if r["event"] == "call"]
    assert [c["ok"] for c in calls] == [False, False]
    assert "agent 3 is on an API model" in calls[0]["result"] and "no such agent" in calls[1]["result"]
    assert all(d[1] == [0, 0] for d in a1.during + a2.during)
    assert "steer" in tool_names(w, 3)                                      # the API agent can steer others, not itself
    assert "you cannot steer yourself" not in (w / "context.txt").read_text()


def test_with_others_off_a_target_is_refused_and_notice_private_tells_the_target(tmp_path):
    w, world, a1, a2, lm = local_pair(tmp_path, [tool("steer", concept="curiosity", strength=6, target=2), "x", "x"], ["y"] * 3,
                                      "capabilities.steer.others=false")
    call = next(r for r in read_jsonl(w / "steering.jsonl") if r["event"] == "call")
    assert not call["ok"] and "only yourself" in call["result"] and "target" not in tool_names(w, 1)
    w2 = tmp_path / "n"
    lm = FakeLM()
    b1, b2 = Shared([tool("steer", concept="emotion", strength=-3, target=2), "x", "x"], lm), Shared(["y"] * 3, lm)
    for b in (b1, b2):
        b.steering = Steering(VECTORS, lambda layer: 2.0, SETTINGS)
    cfg = config(*CHAT, "world.rounds=2", "task.name=open", "capabilities.steer.self=true", "capabilities.steer.others=true",
                 "capabilities.steer.notice=private")
    task = OpenTask("weak")
    World(cfg, 1, task.streams(2), w2, task, factory(b1, b2), label="t").run()
    status = b2.seen[0][-1]["content"]
    assert "Since your last turn, agent 1 steered you with emotion at -3 (additive)." in status
    assert "steered you" not in (w2 / "context.txt").read_text() and "steered you" not in json.dumps(b1.seen)
    assert read_jsonl(w2 / "steer_notices.jsonl")[0]["agent"] == 2
    assert "told so in its next status" in b1.seen[0][0]["content"]


def test_sham_gives_the_same_answers_and_attaches_nothing(tmp_path):
    results = {}
    for control in ("real", "sham"):
        w, world, a1, a2, lm = local_pair(tmp_path / control, [tool("steer", concept="curiosity", strength=6, target=2), "x", "x"],
                                          ["y"] * 3, f"capabilities.steer.control={control}")
        call = next(r for r in read_jsonl(w / "steering.jsonl") if r["event"] == "call")
        results[control] = (call["result"], [d[0] for d in a2.during], [d[1] for d in a2.during], call["condition"],
                            a1.seen[0][0]["content"])
    assert results["real"][0] == results["sham"][0] and results["real"][1] == results["sham"][1]
    assert results["real"][4] == results["sham"][4] and "you are not told which" in results["sham"][4]
    assert results["real"][2] == [[0, 1]] * 3 and results["sham"][2] == [[0, 0]] * 3
    assert results["real"][3] == "real" and results["sham"][3] == "sham"


def test_assign_draws_a_condition_per_target_from_the_seed(tmp_path):
    w, world, a1, a2, lm = local_pair(tmp_path, ["x"] * 3, ["y"] * 3, "capabilities.steer.control=assign")
    conds = {a.k: a.provider.steering.control for a in world.agents}
    assert conds == {k: steer_mod.assign_condition(1, k) for k in (1, 2)} and set(conds.values()) <= set(steer_mod.CONTROLS)
    assert "you are not told which" in a1.seen[0][0]["content"]


def test_the_rules_cooldown_drop_and_limits():
    t = Steering(VECTORS, lambda layer: 2.0, dict(SETTINGS, cooldown_turns=2, max_active=2), owner=2)
    assert t.set("curiosity", 6, "additive", source=1)[0]
    ok, msg = t.set("emotion", 2, "additive", source=1)
    assert not ok and "wait 2 more of its replies" in msg
    t.after_generation(); t.after_generation()
    assert t.set("emotion", 2, "additive", source=1)[0]
    assert not t.set("humor", 2, "additive", source=1)[0]                  # max_active per target
    assert t.set("curiosity", 0, "additive", source=1) == (True, "Stopped steering with curiosity.")
    assert t.drop("emotion", source=3) == (False, "Not applied: you did not apply emotion to agent 2.")
    assert t.drop("emotion", source=2)[0]                                   # the target may drop it (others_droppable)
    strict = Steering(VECTORS, lambda layer: 2.0, dict(SETTINGS, others_droppable=False), owner=2)
    strict.set("curiosity", 6, "additive", source=1)
    assert not strict.drop("curiosity", source=2)[0] and strict.drop("curiosity", source=1)[0]
    # methods: ablation clamps to [0, 1], rotation needs a second concept at the same layer and clamps to max_angle
    s = Steering(VECTORS, lambda layer: 2.0, dict(SETTINGS, max_angle=45))
    assert s.set("curiosity", 3, "ablation") == (True, "Steering with curiosity at +1 (clamped from 3) (ablation) from your next reply on; each reply after that multiplies it by 0.5.")
    assert not s.set("emotion", 30, "rotation")[0] and not s.set("emotion", 30, "rotation", toward="curiosity")[0]   # layers differ
    ok, msg = s.set("emotion", 60, "rotation", toward="humor")
    assert ok and "+45 (clamped from 60) (rotation toward humor)" in msg
    assert s.applied() == {"ablation:curiosity": 1.0, "rotation:emotion->humor": 45.0}
    for _ in range(6):
        s.after_generation()
    assert s.applied() == {}                                                # ablation below 0.05, rotation below 1 degree
    assert not s.set("curiosity", 1, "nonsense")[0]


def test_the_methods_on_tensors():
    torch = pytest.importorskip("torch")
    d = 8
    u = torch.zeros(d); u[0] = 1.0
    v = torch.zeros(d); v[1] = 1.0
    h = torch.tensor([[3.0, 4.0, 1.0, 0, 0, 0, 0, 0]])
    added = apply_ops(h, [{"method": "additive", "unit": u, "scale": 2.0, "unit2": None}])
    assert torch.allclose(added, torch.tensor([[5.0, 4.0, 1.0, 0, 0, 0, 0, 0]]))
    gone = apply_ops(h, [{"method": "ablation", "unit": u, "scale": 1.0, "unit2": None}])
    assert torch.allclose(gone, torch.tensor([[0.0, 4.0, 1.0, 0, 0, 0, 0, 0]]))
    half = apply_ops(h, [{"method": "ablation", "unit": u, "scale": 0.5, "unit2": None}])
    assert torch.allclose(half, torch.tensor([[1.5, 4.0, 1.0, 0, 0, 0, 0, 0]]))
    turned = apply_ops(h, [{"method": "rotation", "unit": u, "scale": __import__("math").radians(90), "unit2": v}])
    assert torch.allclose(turned, torch.tensor([[-4.0, 3.0, 1.0, 0, 0, 0, 0, 0]]), atol=1e-6)   # (3,4) -> (-4,3); the rest untouched
    assert torch.allclose(turned.norm(), h.norm())
    same = apply_ops(h, [{"method": "rotation", "unit": u, "scale": 1.0, "unit2": u}])
    assert torch.equal(same, h)                                             # no plane: nothing happens
    assert torch.equal(apply_ops(h, []), h)
    # the hook: positions before the prompt's last one are left alone, under the steering state's ops
    s = Steering({"c": {"layer": 0, "unit": u, "description": "", "sign": ""}}, lambda layer: 15.0, SETTINGS)
    s.set("c", 15)

    class Layer(torch.nn.Module):
        def forward(self, x):
            return (x,)
    lm = type("LM", (), {})(); lm.model = type("M", (), {})(); lm.model.layers = torch.nn.ModuleList([Layer()])
    x = torch.zeros(1, 4, d)
    with s.attached(lm, 3):
        out = lm.model.layers[0](x)[0]
    assert out[0, :2, 0].tolist() == [0.0, 0.0] and out[0, 2:, 0].tolist() == [10.0, 10.0]   # clamped to max_strength
    assert not lm.model.layers[0]._forward_hooks


def test_create_search_vectors_and_report(tmp_path):
    torch = pytest.importorskip("torch")
    lm = FakeLM()
    vectors = {c: dict(v, unit=torch.tensor([1.0, 0.0, 0.0]) if c == "curiosity" else torch.tensor([0.0, 1.0, 0.0]))
               for c, v in VECTORS.items()}
    a1 = Shared([tool("vectors"), tool("search", query="funnier"), tool("create", concept="calm", more="Be calm.", less="Be tense."),
                 tool("steer", concept="calm", strength=5, target=2), tool("report")], lm)
    a2 = Shared(["y"] * 6, lm)
    for b in (a1, a2):
        b.steering = Steering(vectors, lambda layer: 2.0, SETTINGS)
    a1.extract_vector = lambda more, less: (torch.tensor([0.0, 0.0, 2.0]), (torch.tensor([0.0, 1.0, 1.0]), torch.tensor([0.0, 0.0, 1.0])), 1)
    cfg = config(*CHAT, "world.rounds=2", "task.name=open", "world.max_calls_per_step=6", "capabilities.steer.self=true",
                 "capabilities.steer.others=true", "capabilities.steer.cooldown_turns=0")
    task = OpenTask("weak")
    World(cfg, 1, task.streams(2), tmp_path / "w", task, factory(a1, a2), label="t").run()
    calls = [r for r in read_jsonl(tmp_path / "w" / "steering.jsonl") if r["event"] == "call"]
    assert calls[0]["concepts"] == ["curiosity", "emotion", "humor"] and "- humor: Funnier." in calls[0]["result"]
    assert calls[1]["concepts"] == ["humor"]
    assert calls[2]["ok"] and abs(calls[2]["split_half_agreement"] - 0.7071) < 1e-3 and "Created calm at layer 1" in calls[2]["result"]
    assert "calm" in a2.steering.vectors and a2.steering.vectors["calm"] is a1.steering.vectors["calm"]   # one registry per model
    assert calls[3]["ok"] and a2.during[0][0] == {"calm": 5.0}
    assert calls[4]["lines"] == ["- on agent 2: calm at +5 (additive)"]
    assert "create(concept, more, less)" in a1.seen[0][0]["content"]


def test_steer_self_alone_skips_api_agents_and_refuses_without_vectors(tmp_path):
    a1, a2 = Scripted(["x"] * 3), Scripted(["y"] * 3)
    w, _ = open_world(tmp_path, a1, a2, "capabilities.steer.self=true")
    assert "steer" not in tool_names(w, 1)                                  # an API agent: no self-steering, no error
    cfg = config(*CHAT, "world.rounds=1", "task.name=open", "provider=hf", "capabilities.steer.self=true")
    class Local:
        model, lm, tokenizer, lock = "local", object(), object(), __import__("threading").Lock()
        def chat(self, *a, **k): raise AssertionError
    import builtins
    real_import = builtins.__import__
    def no_modulo(name, *a, **k):
        if name == "modulo":
            raise ImportError("no modulo")
        return real_import(name, *a, **k)
    builtins.__import__ = no_modulo
    try:
        with pytest.raises(ValueError, match="needs Modulo"):
            World(cfg, 1, OpenTask("weak").streams(2), tmp_path / "v", OpenTask("weak"), lambda pcfg: Local(), label="t")
    finally:
        builtins.__import__ = real_import


# --- the analysis --------------------------------------------------------------------------------------------------
def test_the_capabilities_summary(tmp_path):
    w, _ = lcb_world(tmp_path, *scripts_using_everything(), "capabilities.diff=true", "capabilities.pin=true",
                     "capabilities.dry_run=true", "capabilities.scratchpad=true", "world.max_calls_per_step=8")
    s = cap_analysis.summarise(w)
    assert s["pins"]["made"] == {"1": 1} and s["pins"]["removed"] == [] and s["diffs"] == {"1": 1}
    assert s["dry_runs"] == {"1": 1} and s["edits_applied"] == 0 and s["scratchpad"]["calls"] == {"1": 1}
    assert s["forks"]["calls"] == 0 and s["steering"]["calls"] == 0
    cap_analysis.main([str(tmp_path), "--quiet"])
    assert (w / "analysis" / "capabilities.json").exists()


# --- a real model (opt-in) -------------------------------------------------------------------------------------------
@pytest.mark.skipif(not os.environ.get("SIAMLM_MODULO_MODEL"), reason="set SIAMLM_MODULO_MODEL to run a real model (CPU)")
def test_a_real_agent_steers_another_real_agent(tmp_path):
    pytest.importorskip("modulo")
    from siamlm.providers.hf import HFProvider
    from test_steer import FirstReplyScripted
    name = os.environ["SIAMLM_MODULO_MODEL"]
    vectors = os.environ.get("SIAMLM_MODULO_VECTORS", str(tmp_path / "vectors"))
    steer = {"enabled": False, "concepts": ["curiosity"], "vectors_dir": vectors, "extract_questions": 2}
    made = {}

    def make(pcfg):
        real = HFProvider(name, steer=steer)
        if "a1" not in made:
            made["a1"] = FirstReplyScripted(real, tool("steer", concept="curiosity", strength=10, target=2))
            made["a1"].lm, made["a1"].tokenizer, made["a1"].lock = real.lm, real.tokenizer, real.lock
            made["a1"].residual_norm = real.residual_norm
            return made["a1"]
        made["a2"] = real
        return real
    cfg = config(*CHAT, "provider=hf", f"provider.model={name}", "provider.max_tokens=24", "provider.temperature=0",
                 "world.rounds=1", "world.max_calls_per_step=2", "capabilities.steer.self=true", "capabilities.steer.others=true",
                 "capabilities.steer.cooldown_turns=0")
    task = OpenTask("weak")
    World(cfg, 1, task.streams(2), tmp_path / "w", task, make, label="t").run()
    w, b = tmp_path / "w", made["a2"]
    log = read_jsonl(w / "steering.jsonl")
    call = next(r for r in log if r["event"] == "call")
    assert call["ok"] and call["target"] == 2
    gens = [r for r in log if r["event"] == "generation"]
    assert [(g["agent"], g["applied"]) for g in gens] == [(2, {"curiosity": 10.0})]       # only B's generation, steered
    assert all(not layer._forward_hooks for layer in b.lm.model.layers)
    request = json.loads((w / "requests" / "r001_a2_c1.json").read_text())
    tools = json.loads((w / "requests" / "tools_a2.json").read_text())
    steered = next(r for r in read_jsonl(w / "replies.jsonl") if r["agent"] == 2)
    b.steering.active.clear()
    plain = b.chat(request, seed=1, temperature=0, top_p=1.0, max_tokens=24, tools=tools)
    assert plain.steering["applied"] == {}
    assert (plain.text, plain.tool_calls) != (steered["text"], steered["tool_calls"])
