"""Providers: the world only consumes ProviderResponse.

- `openrouter`: OpenRouter's OpenAI-compatible POST /chat/completions, with tools, `extra_body` (e.g. reasoning),
  usage and cost, retries, and sampling parameters the model does not accept left out (OpenAICompatibleProvider).
- `mock`: an offline stand-in for smoke runs; no network, no key (MockProvider).
- any other `provider.name` is a dotted import path to a factory taking the provider config, so another service can
  be added without editing SLM. Any object with `chat(messages, *, seed, temperature, top_p, max_tokens, tools,
  tool_choice)` returning a ProviderResponse works.

API keys are read from the environment variable named by `api_key_env` at call time; they are never logged.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field


@dataclass
class ProviderResponse:
    text: str | None
    usage: dict
    finish_reason: str | None
    retries: int
    error: str | None
    attempts: list[dict]
    tool_calls: list[dict] = field(default_factory=list)  # [{"id", "name", "arguments": dict}] when tools were offered
    internal_reasoning: str | None = None  # the model's own thinking, if the service returns it
    error_kind: str | None = None    # context_limit | client_error | retries_exhausted | missing_key
    latency_s: float = 0.0           # time spent in HTTP requests only (no backoff waits)
    response_id: str | None = None   # provider generation/request ID when available
    response_model: str | None = None


class MockProvider:
    """Offline stand-in for a model, for smoke runs (no network, no key, no cost). Deterministic given the seed.

    With tools offered (and tool_choice not "none"), every third reply is text only; the others call one of the
    offered tools, chosen by the seed, with placeholder arguments (a short program, a harmless bash command, an
    edit that removes turn 1, the answer "mock"). Without tools it answers in text. Usage counts characters / 4.
    """
    name = "mock"
    ARGUMENTS = {"code": "print('mock')", "command": "wc -c /ctx/context.txt; ls /work/env 2>/dev/null | head",
                 "answer": "mock"}
    EDIT = {"delete_turns": [{"first": 1, "last": 1}]}          # edit_context: remove the first turn

    def __init__(self, model: str = "mock-random", **_: object):
        self.model = model
        self.calls = 0

    def chat(self, messages: list[dict], *, seed: int, tools: list[dict] | None = None, tool_choice: str | None = None,
             **_: object) -> ProviderResponse:
        self.calls += 1
        rng = random.Random(seed * 31 + self.calls)
        chars = sum(len(m.get("content") or "") for m in messages) + len(json.dumps(tools or []))
        usage = {"prompt_tokens": math.ceil(chars / 4), "completion_tokens": 10}
        attempts = [{"attempt": 0, "provider": self.name, "model": self.model, "status": "success"}]
        if not tools or tool_choice == "none" or self.calls % 3 == 0:
            return ProviderResponse(f"[mock] reply {self.calls}.", usage, "stop", 0, None, attempts)
        function = rng.choice(tools)["function"]
        params = function.get("parameters") or {}
        arguments = dict(self.EDIT) if function["name"] == "edit_context" else {}
        for key in params.get("required") or []:
            kind = (params.get("properties") or {}).get(key, {}).get("type")
            arguments[key] = self.ARGUMENTS.get(key, 1 if kind == "integer" else [] if kind == "array" else "mock")
        call = {"id": f"call_{self.calls}", "name": function["name"], "arguments": arguments}
        return ProviderResponse(f"[mock] calling {function['name']}.", usage, "tool_calls", 0, None, attempts, [call])


RETRYABLE_STATUS = (408, 425, 429, 500, 502, 503, 504, 529)
CONTEXT_LIMIT_HINTS = ("context length", "context_length", "maximum context", "too many tokens", "prompt is too long",
                       "request too large", "payload too large", "reduce the length")


def retry_after_seconds(response):
    """Seconds asked for by a Retry-After header (number or HTTP date), or None."""
    value = (response.headers or {}).get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        from email.utils import parsedate_to_datetime
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


SAMPLING_PARAMS = ("temperature", "top_p", "seed")
_SUPPORTED: dict[str, dict[str, set]] = {}   # base_url -> model -> supported parameter names (one lookup per process)


def supported_parameters(base_url: str, model: str, timeout: float = 30) -> set | None:
    """OpenRouter's list of request parameters a model accepts (GET /models), or None when unknown."""
    if base_url not in _SUPPORTED:
        import requests
        try:
            data = requests.get(base_url + "/models", timeout=timeout).json()["data"]
            _SUPPORTED[base_url] = {m["id"]: set(m.get("supported_parameters") or []) for m in data}
        except Exception as error:  # no list: send everything
            print(f"[provider] could not read {base_url}/models ({type(error).__name__}); sending all parameters", file=sys.stderr)
            _SUPPORTED[base_url] = {}
    return _SUPPORTED[base_url].get(model)


class OpenAICompatibleProvider:
    """POST /chat/completions with retries.

    Retried (exponential backoff, or the server's Retry-After, capped at max_backoff_s): network errors, timeouts,
    HTTP 408/425/429/5xx, and 200 responses without choices. With `tools`, the tools are offered as OpenAI tool
    definitions and tool calls come back parsed in ProviderResponse.tool_calls. Not retried: a request too large for
    the model (HTTP 413, or 400 mentioning the context length) -> error_kind "context_limit"; any other 4xx ->
    "client_error".

    OpenRouter (drop_unsupported_params, default on): temperature, top_p and seed are left out of the request when the
    model does not accept them (OpenRouter's /models list), instead of being silently ignored by the service or, with
    tools, making routing fail. Every attempt record names what was left out (`dropped_params`; the world logs it in
    calls.jsonl), so a run shows when its sampling settings could not apply; `params_unchecked` marks attempts whose
    model was not in the list (or the list could not be read), so nothing was dropped and nothing is known.
    """
    def __init__(self, name: str, model: str, base_url: str, api_key_env: str, timeout: int = 120, max_retries: int = 4,
                 max_tokens: int = 700, extra_body: dict | None = None, max_backoff_s: float = 60, drop_unsupported_params: bool | None = None):
        self.name, self.model, self.base_url, self.api_key_env = name, model, base_url.rstrip("/"), api_key_env
        self.drop_unsupported = (name == "openrouter") if drop_unsupported_params is None else bool(drop_unsupported_params)
        self.timeout, self.max_retries, self.max_tokens = timeout, max_retries, int(max_tokens)
        self.extra_body, self.max_backoff_s = dict(extra_body or {}), float(max_backoff_s)

    def _wait(self, attempt, response=None):
        asked = retry_after_seconds(response) if response is not None else None
        wait = min(asked if asked is not None else 2 ** attempt, self.max_backoff_s)
        time.sleep(wait)
        return round(wait, 3)

    def chat(self, messages: list[dict], *, seed: int, temperature: float, top_p: float, max_tokens: int | None = None,
             tools: list[dict] | None = None, tool_choice: str | None = None) -> ProviderResponse:
        import requests
        key = os.environ.get(self.api_key_env)
        if not key:
            return ProviderResponse(None, {}, None, 0, f"missing environment variable {self.api_key_env}", [], error_kind="missing_key")
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "top_p": top_p,
                   "max_tokens": int(max_tokens or self.max_tokens), "seed": seed, **self.extra_body}
        if self.name == "openrouter":
            # OpenRouter only returns provider-accounted cost when explicitly asked.
            payload["usage"] = {"include": True}
        dropped, unchecked = [], False
        if self.drop_unsupported:
            supported = supported_parameters(self.base_url, self.model)
            unchecked = not supported
            if supported:
                dropped = [p for p in SAMPLING_PARAMS if p not in supported]
                for p in dropped:
                    payload.pop(p, None)
        if tools:
            payload["tools"], payload["tool_choice"] = tools, tool_choice or "auto"
            if self.name == "openrouter":  # only route to providers that actually support tool calling
                payload.setdefault("provider", {})["require_parameters"] = True
        attempts, last, api_seconds = [], None, 0.0
        for attempt in range(self.max_retries + 1):
            record = {"attempt": attempt, "provider": self.name, "model": self.model}
            if dropped: record["dropped_params"] = dropped
            if unchecked: record["params_unchecked"] = True   # no /models entry: sent everything, may be ignored
            attempts.append(record)
            started = time.time()
            try:
                response = requests.post(self.base_url + "/chat/completions", json=payload, timeout=self.timeout,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            except Exception as error:  # network error / timeout: retryable
                api_seconds += time.time() - started
                last = f"{type(error).__name__}: {error}"
                record["error"] = last
                if attempt < self.max_retries: record["wait_s"] = self._wait(attempt)
                continue
            api_seconds += time.time() - started
            status = record["status_code"] = response.status_code
            if status == 200:
                try:
                    data = response.json()
                    choice = data["choices"][0]
                except Exception as error:  # e.g. a 200 carrying {"error": ...}: retryable
                    last = f"malformed response: {type(error).__name__}: {response.text[:300]}"
                    record["error"] = last
                    if attempt < self.max_retries: record["wait_s"] = self._wait(attempt)
                    continue
                message = choice.get("message") or {}
                return ProviderResponse(message.get("content"), data.get("usage", {}), choice.get("finish_reason"), attempt, None, attempts,
                                        parse_tool_calls(message.get("tool_calls")), parse_internal_reasoning(message),
                                        latency_s=round(api_seconds, 3), response_id=data.get("id"), response_model=data.get("model"))
            body = (response.text or "")[:500]
            last = f"HTTP {status}: {body}"
            record["error"] = body
            if status == 413 or (status == 400 and any(hint in body.lower() for hint in CONTEXT_LIMIT_HINTS)):
                return ProviderResponse(None, {}, None, attempt, last, attempts, error_kind="context_limit", latency_s=round(api_seconds, 3))
            if status not in RETRYABLE_STATUS:
                return ProviderResponse(None, {}, None, attempt, last, attempts, error_kind="client_error", latency_s=round(api_seconds, 3))
            if attempt < self.max_retries: record["wait_s"] = self._wait(attempt, response)
        return ProviderResponse(None, {}, None, self.max_retries, last, attempts, error_kind="retries_exhausted", latency_s=round(api_seconds, 3))


OPENROUTER = ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")


def parse_tool_calls(raw):
    """OpenAI-format tool calls -> [{"id", "name", "arguments": dict}]. Unreadable arguments become {}."""
    calls = []
    for item in raw or []:
        function = item.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except Exception:
            arguments = {}
        calls.append({"id": item.get("id"), "name": function.get("name"), "arguments": arguments if isinstance(arguments, dict) else {}})
    return calls


def parse_internal_reasoning(message):
    """The model's thinking, where the service returns it: OpenRouter `reasoning` / `reasoning_details`,
    DeepSeek-style `reasoning_content`. None when the model or service hides it."""
    for key in ("reasoning", "reasoning_content"):
        if isinstance(message.get(key), str) and message[key].strip():
            return message[key].strip()
    parts = []
    for detail in message.get("reasoning_details") or []:
        if isinstance(detail, dict):
            text = detail.get("text") or detail.get("summary")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts) or None


def _openrouter(cfg):
    base, key = OPENROUTER
    extra = cfg.get("extra_body")
    if extra is not None and not isinstance(extra, dict):
        from omegaconf import OmegaConf
        extra = OmegaConf.to_container(extra, resolve=True)
    return OpenAICompatibleProvider(cfg.name, cfg.model, cfg.get("base_url") or base,
                                    cfg.get("api_key_env") or key, cfg.get("timeout", 120), cfg.get("max_retries", 4),
                                    cfg.get("max_tokens", 700), extra, cfg.get("max_backoff_s", 60),
                                    cfg.get("drop_unsupported_params"))


PROVIDERS = {"mock": lambda cfg: MockProvider(cfg.model), "openrouter": _openrouter}


def make_provider(cfg):
    """Build the provider named by `provider.name`: mock, openrouter, or a dotted import path to a factory taking the
    provider config (e.g. provider.name=my_package.providers.make_vllm)."""
    name = str(cfg.name)
    if name in PROVIDERS:
        return PROVIDERS[name](cfg)
    if "." in name:
        module_name, _, attribute = name.rpartition(".")
        try:
            return getattr(importlib.import_module(module_name), attribute)(cfg)
        except (ImportError, AttributeError) as error:
            raise ValueError(f"cannot import provider {name!r}: {error}") from error
    raise ValueError(f"unknown provider {name!r}; expected one of {tuple(PROVIDERS)} or a dotted import path")
