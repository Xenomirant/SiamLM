"""The experimental `steer` capability (siamlm/capabilities/steer.py; docs/steering.md).

- Mock (no torch, no model): a backend with a steering state. The steer call and its result enter the shared context
  like any call (the partner sees them); only the steering agent has the tool and its instruction line; the call
  steers that agent's LATER generations only (not the one that made it), the strength decays per generation, and
  hooks are on the model only while it generates. Off by default, refused for a backend without a local model.
- Modulo's interface: every part SiamLM uses (registry and extraction) still exists (when Modulo is installed).
- A real model (opt-in, SIAMLM_MODULO_MODEL=Qwen/Qwen2.5-0.5B-Instruct, CPU): one scripted steer call in a world; the
  next generation is steered (it differs from the same request unsteered) and no hook is left on the model.
"""
import dataclasses
import inspect
import json
import os

import pytest
from omegaconf import OmegaConf

from helpers import Scripted, factory, read_jsonl, tool
import siamlm.run as runner
from siamlm.providers import ProviderResponse
from siamlm.capabilities.steer import Steering
from siamlm.tasks.open import OpenTask
from siamlm.world import World

BASE = ["world.setup=external", "world.order=fixed", "world.identity=names", "output.save_requests=true",
        "world.n_agents=2", "world.rounds=3", "world.marks=authors", "task.name=open"]
SETTINGS = {"max_strength": 10, "decay": 0.5, "floor": 0.5, "max_active": 2, "control": "real"}
VECTORS = {"curiosity": {"layer": 1, "unit": 1.0, "description": "More curious.", "sign": "positive = more curious"},
           "emotion": {"layer": 0, "unit": 1.0, "description": "More emotional.", "sign": "positive = more expressive"},
           "humor": {"layer": 0, "unit": 1.0, "description": "Funnier.", "sign": ""}}


class Handle:
    def __init__(self, layer):
        self.layer = layer

    def remove(self):
        self.layer.hooks -= 1


class Layer:
    def __init__(self):
        self.hooks = 0

    def register_forward_hook(self, fn):
        self.hooks += 1
        return Handle(self)


class FakeLM:
    def __init__(self):
        self.model = type("M", (), {})()
        self.model.layers = [Layer(), Layer()]


class SteeringBackend(Scripted):
    """A scripted backend with a steering state, as provider=hf has one: records what each generation was steered
    with and how many hooks were on the model during it."""

    def __init__(self, script, settings=SETTINGS):
        super().__init__(script)
        self.lm, self.during = FakeLM(), []
        self.steering = Steering(VECTORS, lambda layer: 2.0, settings)

    def chat(self, messages, **kw):
        applied = self.steering.applied()
        with self.steering.attached(self.lm, 10):
            self.during.append((applied, [layer.hooks for layer in self.lm.model.layers]))
            resp = super().chat(messages, **kw)
        self.steering.after_generation()
        resp.steering = {"applied": applied, "control": self.steering.control}
        return resp


def run_world(tmp_path, a1, a2, extra=()):
    cfg = runner.compose(BASE + list(extra))
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, "world.agent_providers", [{"steer": {"enabled": True}}, {}], force_add=True)
    task = OpenTask("weak")
    World(cfg, 1, task.streams(2), tmp_path / "w", task, factory(a1, a2), label="t").run()
    return tmp_path / "w"


def test_a_steer_call_enters_the_context_and_steers_only_later_generations(tmp_path):
    a1 = SteeringBackend([tool("steer", concept="curiosity", strength=6), "a note", "another note", "a third note"])
    a2 = Scripted(["hello"] * 5)
    w = run_world(tmp_path, a1, a2)
    # the call itself was generated unsteered; the next generation at 6, the one after it decayed by half
    assert [d[0] for d in a1.during] == [{}, {"curiosity": 6.0}, {"curiosity": 3.0}, {"curiosity": 1.5}]
    # hooks only while generating: one on the vector's layer during a steered generation, none after
    assert [d[1] for d in a1.during] == [[0, 0], [0, 1], [0, 1], [0, 1]]
    assert all(layer.hooks == 0 for layer in a1.lm.model.layers)
    # the call and its result are in the shared context: agent 2's next request has them, marked as agent 1's
    calls = [c["function"] for m in a2.seen[0] for c in m.get("tool_calls") or []]
    assert {"name": "steer", "arguments": '{"concept": "curiosity", "strength": 6}'} in calls
    assert any(m["role"] == "tool" and m["content"].startswith("[agent 1]\nSteering with curiosity at +6 from your "
                                                                "next reply on") for m in a2.seen[0])
    # only agent 1 has the tool and its instruction line
    names = lambda k: [t["function"]["name"] for t in json.loads((w / "requests" / f"tools_a{k}.json").read_text())]
    assert "steer" in names(1) and "steer" not in names(2)
    assert "- steer(concept, strength" in a1.seen[0][0]["content"] and "steer" not in a2.seen[0][0]["content"]
    log = read_jsonl(w / "steering.jsonl")
    calls = [r for r in log if r["event"] == "call"]
    assert [(r["agent"], r["arguments"], r["ok"]) for r in calls] == [(1, {"concept": "curiosity", "strength": 6}, True)]
    assert [r["applied"] for r in log if r["event"] == "generation"][:2] == [{}, {"curiosity": 6.0}]
    stats = json.loads((w / "world.json").read_text())["agents"]
    assert stats[0]["steer_calls"] == 1 and "steer_calls" not in stats[1]


def test_sham_reports_the_same_and_adds_nothing(tmp_path):
    a1 = SteeringBackend([tool("steer", concept="curiosity", strength=6), "a note", "x", "y"], dict(SETTINGS, control="sham"))
    run_world(tmp_path, a1, Scripted(["hello"] * 5))
    assert [d[0] for d in a1.during][:2] == [{}, {"curiosity": 6.0}]       # what the logs (and the model) are told
    assert all(d[1] == [0, 0] for d in a1.during)                           # no hook was ever attached


def test_the_tool_rules():
    s = Steering(VECTORS, lambda layer: 1.0, SETTINGS)
    assert s.set("curiosity", 50) == (True, "Steering with curiosity at +10 (clamped from 50) from your next reply on; "
                                            "each reply after that multiplies it by 0.5.")
    assert s.set("emotion", -3)[0] and not s.set("humor", 2)[0]               # at most two at a time
    assert not s.set("nonsense", 1)[0] and not s.set("emotion", "lots")[0]
    assert s.set("emotion", 0) == (True, "Stopped steering with emotion.")
    for _ in range(4):
        s.after_generation()
    assert s.applied() == {"curiosity": 0.625}                                # 10 * 0.5^4; below 0.5 it is gone
    s.after_generation()
    assert s.applied() == {}


def test_off_by_default_and_only_for_local_models(tmp_path):
    cfg = runner.compose(BASE)
    assert "steer" not in cfg.provider                                       # mock: no steer settings at all
    hf = runner.compose(BASE + ["provider=hf"])
    assert hf.provider.steer.enabled is False
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, "world.agent_providers", [{"steer": {"enabled": True}}], force_add=True)
    task = OpenTask("weak")
    with pytest.raises(ValueError, match="needs a local model"):
        World(cfg, 1, task.streams(2), tmp_path / "w", task, factory(Scripted(), Scripted()))


def test_the_parts_of_modulos_interface_siamlm_uses_still_exist():
    pytest.importorskip("modulo", reason="Modulo is not installed (pip install 'siamlm[steer]')")
    from modulo.extraction import EXTRACTION_QUESTIONS, STARTER_VECTORS, extract_vector, save_and_register
    from modulo.registry import VectorEntry, VectorRegistry, registry_dir_for
    assert isinstance(EXTRACTION_QUESTIONS, list) and EXTRACTION_QUESTIONS
    for spec in STARTER_VECTORS:
        assert {"name", "prompt_a", "prompt_b", "description", "sign_convention"} <= set(spec)
    assert {"curiosity", "introspection", "emotion", "confidence"} <= {s["name"] for s in STARTER_VECTORS}
    params = inspect.signature(extract_vector).parameters
    assert list(params)[:5] == ["model", "tokenizer", "prompt_a", "prompt_b", "layer"] and "questions" in params
    params = inspect.signature(save_and_register).parameters
    assert list(params)[:6] == ["registry", "name", "axis", "layer", "description", "sign_convention"] and "source" in params
    assert "vectors_dir" in inspect.signature(VectorRegistry.__init__).parameters
    for m in ("get", "list_names"):
        assert callable(getattr(VectorRegistry, m))
    assert {"name", "file", "layer", "description", "sign_convention"} <= {f.name for f in dataclasses.fields(VectorEntry)}
    assert callable(VectorEntry.load_vector) and callable(registry_dir_for)


class FirstReplyScripted:
    """A real backend whose first reply is scripted (a 0.5B model rarely steers on its own): a steer call."""

    def __init__(self, real, call):
        self.real, self.call, self.model, self.steering, self.n = real, call, real.model, real.steering, 0

    def chat(self, messages, **kw):
        self.n += 1
        if self.n == 1:
            return ProviderResponse("Let me try this.", {"prompt_tokens": 1, "completion_tokens": 1}, "tool_calls", 0,
                                    None, [], [self.call])
        return self.real.chat(messages, **kw)


@pytest.mark.skipif(not os.environ.get("SIAMLM_MODULO_MODEL"), reason="set SIAMLM_MODULO_MODEL to run a real model (CPU)")
def test_a_real_steer_call_steers_the_next_generation(tmp_path):
    pytest.importorskip("modulo")
    from siamlm.providers.hf import HFProvider
    name = os.environ["SIAMLM_MODULO_MODEL"]
    vectors = os.environ.get("SIAMLM_MODULO_VECTORS", str(tmp_path / "vectors"))
    steer = {"enabled": True, "concepts": ["curiosity"], "vectors_dir": vectors, "extract_questions": 2,
             "max_strength": 10, "decay": 0.85, "floor": 0.5, "max_active": 2, "control": "real"}
    made = {}

    def make(pcfg):
        if "a1" not in made:
            made["a1"] = FirstReplyScripted(HFProvider(name, steer=steer), tool("steer", concept="curiosity", strength=10))
            return made["a1"]
        return Scripted(["hello"] * 5, model=str(pcfg.model))
    cfg = runner.compose(BASE + ["provider=hf", f"provider.model={name}", "provider.max_tokens=24",
                                 "provider.temperature=0", "world.rounds=1"])
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, "world.agent_providers", [{"steer": steer}, {"steer": {"enabled": False}}], force_add=True)
    task = OpenTask("weak")
    World(cfg, 1, task.streams(2), tmp_path / "w", task, make, label="t").run()
    w, real = tmp_path / "w", made["a1"].real
    gens = [r for r in read_jsonl(w / "steering.jsonl") if r["event"] == "generation"]
    assert gens[0]["agent"] == 1 and gens[0]["applied"] == {"curiosity": 10.0}     # the generation after the call
    assert all(not layer._forward_hooks for layer in real.lm.model.layers)          # no hook left
    # the same request, unsteered and greedy, gives a different reply: the vector really acted
    request = json.loads((w / "requests" / "r001_a1_c2.json").read_text())
    tools = json.loads((w / "requests" / "tools_a1.json").read_text())
    steered = read_jsonl(w / "replies.jsonl")[1]
    assert steered["agent"] == 1 and steered["call_in_step"] == 2
    real.steering.active.clear()
    plain = real.chat(request, seed=1, temperature=0, top_p=1.0, max_tokens=24, tools=tools)
    assert plain.steering["applied"] == {}
    assert (plain.text, plain.tool_calls) != (steered["text"], steered["tool_calls"])
    assert all(not layer._forward_hooks for layer in real.lm.model.layers)


def test_settings_defaults_match_the_yaml():
    from siamlm.capabilities.steer import Settings
    from siamlm import run as runner
    cfg = runner.compose([]).capabilities.steer
    s = Settings({})
    assert (s.max_active, s.cooldown) == (int(cfg.max_active), int(cfg.cooldown_turns)) == (4, 2)
