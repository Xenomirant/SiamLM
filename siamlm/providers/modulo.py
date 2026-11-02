"""Self-steering local models: `provider.name=modulo` runs an open-weight Hugging Face model through Modulo
(https://github.com/EigenformAI/modulo, a third-party library in active development; `pip install siamlm[steer]`
installs the commit pinned in pyproject.toml). Besides the world's own tools (editing the context, the task's work
calls), the model can steer its own activations: `steer(concept, strength)`, `drop`, `report`, and `create` (make a
vector from two system prompts). Modulo runs those calls itself, inside the reply; this provider returns them in
`ProviderResponse.steering`, and the world writes each as a call turn and its result into the context, where the
other agents see them like any other tool call. Calls to the world's tools come back as ordinary tool calls.

Every Modulo call goes through `_step` and `_events` below, so a change in Modulo's interface touches only them
(tests/test_modulo.py checks the parts of the interface used here whenever Modulo is installed).

One model is loaded per process and model name, and shared by the agents that use it: each agent has its own
SteeringAgent (its own vectors, decay and cooldowns), and Modulo attaches its hooks only while it generates.
Settings (provider.*): model, device (cpu), dtype (float32 on CPU), vectors_dir (where vectors are kept and
extracted; default <output dir>/modulo_vectors), auto_extract (extract Modulo's starter vectors on first use),
control (None | real | sham | random | assign: Modulo's controlled-study conditions, disclosed to the model as Modulo
does), menu_seed, include_community (unverified community vectors; off by default).
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from .base import ProviderResponse

_MODELS: dict = {}
_LOCK = threading.Lock()


def load_model(name: str, device: str = "cpu", dtype: str = "float32"):
    """(model, tokenizer), loaded once per process."""
    key = (name, device, dtype)
    with _LOCK:
        if key not in _MODELS:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, dtype)).to(device)
            model.eval()
            _MODELS[key] = (model, tok)
        return _MODELS[key]


def hf_messages(messages: list[dict]) -> list[dict]:
    """OpenAI-style messages as chat templates take them: tool-call arguments as dicts, ids dropped."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                f = c.get("function") or c
                args = f.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                calls.append({"type": "function", "function": {"name": f.get("name"), "arguments": args or {}}})
            m["tool_calls"] = calls
            m["content"] = m.get("content") or ""
        m.pop("tool_call_id", None)
        out.append(m)
    return out


TAGS = re.compile(r"<tool_call>.*?</tool_call>", re.S)


class ModuloProvider:
    def __init__(self, model: str, device: str = "cpu", dtype: str = "float32", vectors_dir: str | None = None,
                 auto_extract: bool = True, control: str | None = None, menu_seed: int | None = None,
                 include_community: bool = False, **_: object):
        from modulo import SteeringAgent
        self.model_name = model
        self.model, self.tokenizer = load_model(model, device, dtype)
        kwargs = dict(model_name=model, auto_extract=bool(auto_extract), include_community=bool(include_community))
        if vectors_dir:
            Path(vectors_dir).mkdir(parents=True, exist_ok=True)
            kwargs["vectors_dir"] = Path(vectors_dir)
        if control:
            kwargs.update(control=str(control), disclose_controls=True)
        if menu_seed is not None:
            kwargs["menu_seed"] = int(menu_seed)
        self.agent = SteeringAgent(self.model, self.tokenizer, **kwargs)

    # --- the only places that touch Modulo's interface ------------------------------------------------------------
    def _step(self, messages, tools, max_tokens, temperature, top_p):
        return self.agent.step(hf_messages(messages), max_new_tokens=int(max_tokens or 300), temperature=temperature,
                               top_p=top_p, tools=tools or None)

    @staticmethod
    def _events(r) -> list[dict]:
        """Modulo's own calls with their results, then any notice (degeneration, repetition) it attached."""
        events = []
        for i, res in enumerate(r.tool_results):
            call = r.tool_calls[i] if i < len(r.tool_calls) else None
            events.append({"name": res.name, "arguments": dict(getattr(call, "arguments", None) or res.arguments or {}),
                           "ok": bool(res.ok), "message": str(res.message)})
        for why, discarded in (("degeneration", r.degenerate_output), ("repetition", r.repeated_output)):
            if discarded is not None:
                note = next((m["content"] for m in r.tool_messages() if why[:6] in m["content"].lower()), why)
                events.append({"name": "notice", "arguments": {"event": why}, "ok": True, "message": str(note)})
        return events

    # --- the provider interface --------------------------------------------------------------------------------
    def chat(self, messages: list[dict], *, seed: int, temperature: float = 0.7, top_p: float = 0.9,
             max_tokens: int | None = None, tools: list[dict] | None = None, tool_choice: str | None = None,
             **_: object) -> ProviderResponse:
        import torch
        torch.manual_seed(int(seed))
        started = time.time()
        try:
            r = self._step(messages, tools, max_tokens, temperature, top_p)
        except Exception as error:                     # a failed generation is an API error to the world
            return ProviderResponse(None, {}, "error", 0, f"{type(error).__name__}: {error}", [], error_kind="client_error")
        calls = [{"id": f"m{i}", "name": c.name, "arguments": dict(c.arguments or {})} for i, c in enumerate(r.other_tool_calls)]
        text = TAGS.sub("", r.prose or "").strip()
        n_out = len(self.tokenizer(r.raw_output or "").input_ids)
        return ProviderResponse(
            text=text, usage={"completion_tokens": n_out, "cost": 0.0}, finish_reason="tool_calls" if calls else "stop",
            retries=0, error=None, attempts=[], tool_calls=calls, latency_s=round(time.time() - started, 3),
            response_model=self.model_name, steering=self._events(r),
            steering_state={"before": dict(r.steering_before), "after": dict(r.steering_after),
                            "unsteered_nll": r.unsteered_nll})


def make_modulo(cfg) -> ModuloProvider:
    c = dict(cfg)
    return ModuloProvider(**{k: v for k, v in c.items() if k != "name"})
