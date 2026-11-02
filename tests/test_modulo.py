"""provider.name=modulo (self-steering local models through the third-party Modulo library).

- The world: a reply's steering calls (ProviderResponse.steering) become call + result turns in the shared context,
  which later requests carry as real tool calls; steering.jsonl logs them. No torch needed.
- The interface contract: every part of Modulo's API that siamlm/providers/modulo.py uses still exists (whenever
  Modulo is installed), so an upstream change fails here first.
- A real model (opt-in, SIAMLM_MODULO_MODEL=Qwen/Qwen2.5-0.5B-Instruct): one world on CPU, end to end.
"""
import dataclasses
import inspect
import json
import os

import pytest
from omegaconf import OmegaConf

from helpers import factory, read_jsonl, tool
from siamlm.providers.base import ProviderResponse
from siamlm.providers.modulo import hf_messages
from siamlm.tasks.open import OpenTask
from siamlm.world import World
import siamlm.run as runner

BASE = ["world.setup=external", "world.order=fixed", "ctx.readout_by_agent=false", "world.identity=names",
        "output.save_requests=true", "world.n_agents=2", "world.rounds=2", "world.marks=authors"]


class Steering:
    """A provider whose first reply steers (as Modulo returns it), then plain messages."""

    def __init__(self, steer: bool):
        self.steer, self.model, self.seen = steer, "fake/model", []

    def chat(self, messages, **_):
        self.seen.append(messages)
        events = []
        if self.steer and len(self.seen) == 1:
            events = [{"name": "steer", "arguments": {"concept": "curiosity", "strength": 6.0}, "ok": True,
                       "message": "Applied curiosity at strength 6.0."}]
        return ProviderResponse(f"note {len(self.seen)}", {"completion_tokens": 3, "cost": 0.0}, "stop", 0, None, [],
                                steering=events, steering_state={"before": {}, "after": {"curiosity": 5.1}})


def test_steering_calls_become_turns_that_everyone_sees(tmp_path):
    cfg = runner.compose(BASE)
    OmegaConf.set_struct(cfg, False)
    task = OpenTask("weak")
    a1, a2 = Steering(True), Steering(False)
    World(cfg, 1, task.streams(2), tmp_path / "w", task, factory(a1, a2), label="t").run()
    log = read_jsonl(tmp_path / "w" / "steering.jsonl")
    assert [(r["agent"], r["name"], r["arguments"]["concept"]) for r in log] == [(1, "steer", "curiosity")]
    assert log[0]["state"]["after"] == {"curiosity": 5.1}
    # agent 2's next request holds agent 1's steering call and its result (marked as agent 1's)
    flat = json.dumps(a2.seen[0])
    assert "Applied curiosity at strength 6.0." in flat and "steer" in flat
    # agent 1's own later request carries it as a real tool call with its tool message
    own = a1.seen[1]
    calls = [m for m in own if m.get("tool_calls")]
    assert calls and calls[0]["tool_calls"][0]["function"]["name"] == "steer"
    assert any(m["role"] == "tool" and "Applied curiosity" in m["content"] for m in own)
    stats = json.loads((tmp_path / "w" / "world.json").read_text())["agents"]
    assert stats[0].get("steering_calls") == 1 and "steering_calls" not in stats[1]


def test_chat_template_messages():
    out = hf_messages([{"role": "assistant", "content": None, "tool_calls": [
        {"id": "t1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "a.txt"}])
    assert out[0]["tool_calls"][0]["function"] == {"name": "bash", "arguments": {"command": "ls"}}
    assert out[0]["content"] == "" and "tool_call_id" not in out[1]


modulo = pytest.importorskip("modulo", reason="Modulo is not installed (pip install 'siamlm[steer]')")


def test_the_parts_of_modulos_interface_siamlm_uses_still_exist():
    from modulo import SteeringAgent
    from modulo.agent import StepResult, ToolCall, ToolResult
    init = inspect.signature(SteeringAgent.__init__).parameters
    for p in ("model_name", "auto_extract", "vectors_dir", "include_community", "control", "disclose_controls", "menu_seed"):
        assert p in init, f"SteeringAgent() lost {p}"
    step = inspect.signature(SteeringAgent.step).parameters
    for p in ("messages", "max_new_tokens", "temperature", "top_p", "tools"):
        assert p in step, f"SteeringAgent.step() lost {p}"
    fields = {f.name for f in dataclasses.fields(StepResult)}
    for f in ("prose", "raw_output", "tool_calls", "other_tool_calls", "tool_results", "steering_before",
              "steering_after", "degenerate_output", "repeated_output", "unsteered_nll"):
        assert f in fields, f"StepResult lost {f}"
    assert callable(getattr(StepResult, "tool_messages", None))
    assert {"name", "arguments", "ok", "message"} <= {f.name for f in dataclasses.fields(ToolResult)}
    assert {"name", "arguments"} <= {f.name for f in dataclasses.fields(ToolCall)}


@pytest.mark.skipif(not os.environ.get("SIAMLM_MODULO_MODEL"), reason="set SIAMLM_MODULO_MODEL to run a real model (CPU)")
def test_a_real_self_steering_world(tmp_path):
    name = os.environ["SIAMLM_MODULO_MODEL"]
    vectors = os.environ.get("SIAMLM_MODULO_VECTORS", str(tmp_path / "vectors"))
    cfg = runner.compose(BASE + ["provider=modulo", f"provider.model={name}", f"provider.vectors_dir={vectors}",
                                 "provider.max_tokens=200", "world.rounds=3", "ctx.edit=tool"])
    OmegaConf.set_struct(cfg, False)
    task = OpenTask("weak")
    from siamlm.providers import make_provider
    World(cfg, 1, task.streams(2), tmp_path / "w", task, make_provider, label="t").run()
    doc = json.loads((tmp_path / "w" / "world.json").read_text())
    assert doc["status"] == "complete"
    assert all(a["api_errors"] == 0 for a in doc["agents"])
    calls = read_jsonl(tmp_path / "w" / "calls.jsonl")
    assert len(calls) >= 6 and all(c.get("error") is None for c in calls)


@pytest.mark.skipif(not os.environ.get("SIAMLM_MODULO_MODEL"), reason="set SIAMLM_MODULO_MODEL to run a real model (CPU)")
def test_a_real_steer_call_is_run_and_steers_the_next_reply(tmp_path):
    """Real Modulo and a real model; only the first reply's text is scripted (a small model rarely steers on its own):
    the steer call in it is parsed and run by Modulo, the harness's bash call comes back as a tool call, the next reply
    is generated with the vector applied, and no hook is left on the model afterwards."""
    from siamlm.providers.modulo import ModuloProvider
    name = os.environ["SIAMLM_MODULO_MODEL"]
    vectors = os.environ.get("SIAMLM_MODULO_VECTORS", str(tmp_path / "vectors"))
    p = ModuloProvider(model=name, vectors_dir=vectors, menu_seed=1)
    concept = sorted(p.agent.registry.list_names())[0]
    scripted = ('Let me try something.\n<tool_call>\n{"name": "steer", "arguments": {"concept": "' + concept +
                '", "strength": 6.0}}\n</tool_call>\n<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>')
    real_decode = p.agent._decode
    calls = {"n": 0}

    def decode(*args, **kwargs):
        calls["n"] += 1
        return scripted if calls["n"] == 1 else real_decode(*args, **kwargs)
    p.agent._decode = decode
    bash = {"type": "function", "function": {"name": "bash", "description": "Run a command.", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
    msgs = [{"role": "system", "content": "You are agent 1."}, {"role": "user", "content": "(start of the context)"}]
    first = p.chat(msgs, seed=1, max_tokens=40, tools=[bash])
    assert first.error is None
    assert [c["name"] for c in first.tool_calls] == ["bash"] and first.tool_calls[0]["arguments"] == {"command": "ls"}
    assert [(s["name"], s["arguments"].get("concept"), s["ok"]) for s in first.steering] == [("steer", concept, True)]
    assert concept in first.steering_state["after"]
    second = p.chat(msgs + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "go on"}],
                    seed=2, max_tokens=20, tools=[bash])
    assert second.error is None and concept in second.steering_state["before"]       # generated under the vector
    assert all(not layer._forward_hooks for layer in p.model.model.layers)              # hooks removed after generating
