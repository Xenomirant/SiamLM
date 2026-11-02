"""A local Hugging Face model as a backend (`provider.name=hf`; `pip install siamlm[hf]`).

Plain transformers: the request (OpenAI chat messages, exactly as the world built them) is rendered with the model's
own chat template, with the offered tools, and generated with `model.generate`; Hermes-style `<tool_call>{json}
</tool_call>` blocks in the output come back as tool calls, `<think>...</think>` as the model's reasoning. Usage counts
real tokens; cost is 0. One copy of the weights per process and model, shared by the agents that use it; generation
holds a lock per model, so one agent's steering hooks never touch another agent's generation.

With `provider.steer.enabled=true` (EXPERIMENTAL, siamlm/capabilities/steer.py) the backend keeps that agent's steering state and
attaches its hooks only around its own `generate` calls.
"""
from __future__ import annotations

import json
import re
import threading
import time
from contextlib import nullcontext

from .base import ProviderResponse

_MODELS: dict = {}
_LOCK = threading.Lock()
CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
THINK = re.compile(r"<think>(.*?)</think>", re.S)
CALIBRATION = "Tell me a little about how you approach a new task."


def load_model(name: str, device: str = "cpu", dtype: str = "float32"):
    """(model, tokenizer, lock), loaded once per process."""
    key = (name, device, dtype)
    with _LOCK:
        if key not in _MODELS:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(name)
            lm = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype)).to(device)
            lm.eval()
            _MODELS[key] = (lm, tok, threading.Lock())
        return _MODELS[key]


def hf_messages(messages: list[dict]) -> list[dict]:
    """OpenAI-style messages as chat templates take them: tool-call arguments as dicts, content never None."""
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
        out.append(m)
    return out


def parse_reply(raw: str) -> tuple[str, list[dict], str | None]:
    """(text, tool calls, reasoning) from a generated reply. A <tool_call> block that is not valid JSON with a name
    stays in the text."""
    reasoning = "\n".join(t.strip() for t in THINK.findall(raw)).strip() or None
    raw = THINK.sub("", raw)
    calls = []

    def take(match):
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            return match.group(0)
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            return match.group(0)
        args = obj.get("arguments") or {}
        calls.append({"id": f"hf_{len(calls)}", "name": obj["name"], "arguments": args if isinstance(args, dict) else {}})
        return ""
    text = CALL.sub(take, raw).strip()
    return text, calls, reasoning


class HFProvider:
    name = "hf"

    def __init__(self, model: str, device: str = "cpu", dtype: str = "float32", steer=None, **_: object):
        self.model, self.device = model, device
        self.lm, self.tokenizer, self.lock = load_model(model, device, dtype)
        self._norms: dict = {}
        self.steering = None
        steer = dict(steer or {})
        if steer.get("enabled", False):
            from siamlm.capabilities.steer import Steering, load_vectors
            with self.lock:
                vectors = load_vectors(model, self.lm, self.tokenizer, steer)
            self.steering = Steering(vectors, self.residual_norm, steer)

    def prompt(self, messages: list[dict], tools: list[dict] | None) -> str:
        extra = {"tools": tools} if tools else {}
        return self.tokenizer.apply_chat_template(hf_messages(messages), tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False, **extra)

    def _generate(self, ids, max_new: int, temperature: float, top_p: float):
        import torch
        sample = dict(do_sample=True, temperature=float(temperature), top_p=float(top_p)) if temperature > 0 else \
            dict(do_sample=False, temperature=None, top_p=None, top_k=None)
        with torch.inference_mode():
            return self.lm.generate(**ids, max_new_tokens=int(max_new), pad_token_id=self.tokenizer.pad_token_id
                                    or self.tokenizer.eos_token_id, **sample)

    def residual_norm(self, layer: int) -> float:
        """The typical activation norm at a decoder layer's output: the median over the tokens of an unsteered greedy
        reply to a fixed prompt (prompt positions, with their attention sinks, are left out). Cached; called under the
        model's lock."""
        if layer not in self._norms:
            import torch
            text = self.prompt([{"role": "user", "content": CALIBRATION}], None)
            ids = self.tokenizer([text], return_tensors="pt", add_special_tokens=False).to(self.device)
            n = ids.input_ids.shape[1]
            out = self._generate(ids, 32, 0, 1.0)
            seen = {}
            handle = self.lm.model.layers[layer].register_forward_hook(
                lambda m, i, o: seen.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].float()))
            try:
                with torch.inference_mode():
                    self.lm(out)
            finally:
                handle.remove()
            h = seen["h"][n:] if seen["h"].shape[0] > n else seen["h"][1:]
            self._norms[layer] = float(h.norm(dim=-1).median())
        return self._norms[layer]

    def chat(self, messages: list[dict], *, seed: int, temperature: float = 0.7, top_p: float = 0.9,
             max_tokens: int | None = None, tools: list[dict] | None = None, tool_choice: str | None = None,
             **_: object) -> ProviderResponse:
        import torch
        started = time.time()
        max_new = int(max_tokens or 512)
        offered = tools if tools and tool_choice != "none" else None
        try:
            text = self.prompt(messages, offered)
            ids = self.tokenizer([text], return_tensors="pt", add_special_tokens=False).to(self.device)
            n_in = ids.input_ids.shape[1]
            with self.lock:
                torch.manual_seed(int(seed))
                applied = self.steering.applied() if self.steering else None
                with (self.steering.attached(self.lm, n_in) if self.steering else nullcontext()):
                    out = self._generate(ids, max_new, temperature, top_p)
                if self.steering:
                    self.steering.after_generation()
        except Exception as error:                     # a failed generation is an API error to the world
            return ProviderResponse(None, {}, "error", 0, f"{type(error).__name__}: {error}", [], error_kind="client_error")
        new = out[0, n_in:]
        raw = self.tokenizer.decode(new, skip_special_tokens=True)
        words, calls, reasoning = parse_reply(raw)
        if not offered:
            calls = []
        ended = len(new) > 0 and int(new[-1]) in {self.tokenizer.eos_token_id, *(
            [self.lm.generation_config.eos_token_id] if isinstance(self.lm.generation_config.eos_token_id, int)
            else self.lm.generation_config.eos_token_id or [])}
        finish = "tool_calls" if calls else ("length" if len(new) >= max_new and not ended else "stop")
        return ProviderResponse(
            text=words, usage={"prompt_tokens": n_in, "completion_tokens": int(len(new)), "cost": 0.0},
            finish_reason=finish, retries=0, error=None,
            attempts=[{"attempt": 0, "provider": self.name, "model": self.model, "status": "success"}],
            tool_calls=calls, internal_reasoning=reasoning, latency_s=round(time.time() - started, 3),
            response_model=self.model,
            steering=({"applied": applied, "control": self.steering.control} if self.steering else None))


def make_hf(cfg) -> HFProvider:
    from omegaconf import OmegaConf
    c = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else dict(cfg)
    return HFProvider(**{k: v for k, v in c.items() if k != "name"})
