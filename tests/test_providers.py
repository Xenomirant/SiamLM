"""The provider port: the mock's calls are valid for the tools offered, the factory, and the OpenRouter client's
behaviour without a key or with a refused request (no network: requests.post is replaced)."""
import pytest
from omegaconf import OmegaConf

from siamlm.providers import MockProvider, OpenAICompatibleProvider, make_provider
from siamlm.world import bash_tool, edit_tool
from siamlm.tasks.lcb import TOOLS
from siamlm.tasks.state_chain import SUBMIT


def test_mock_calls_only_offered_tools_with_their_required_arguments():
    tools = [TOOLS["run_tests"], TOOLS["submit"], edit_tool("full"), bash_tool("full", True)]
    mock = MockProvider()
    kinds = set()
    for seed in range(30):
        r = mock.chat([{"role": "user", "content": "x" * 900}], seed=seed, tools=tools, tool_choice="auto")
        assert r.error is None and r.usage["prompt_tokens"] > 0
        for call in r.tool_calls:
            spec = next(t["function"] for t in tools if t["function"]["name"] == call["name"])
            assert set(spec["parameters"].get("required") or []) <= set(call["arguments"])
            kinds.add(call["name"])
        kinds.add("text" if not r.tool_calls else "call")
    assert {"text", "edit_context", "bash", "run_tests"} <= kinds
    assert not mock.chat([{"role": "user", "content": "q"}], seed=1, tools=tools, tool_choice="none").tool_calls


def test_make_provider_names_and_dotted_paths():
    assert isinstance(make_provider(OmegaConf.create({"name": "mock", "model": "m"})), MockProvider)
    p = make_provider(OmegaConf.create({"name": "openrouter", "model": "a/b", "api_key_env": "SIAMLM_TEST_NO_KEY"}))
    assert isinstance(p, OpenAICompatibleProvider) and p.base_url == "https://openrouter.ai/api/v1"
    dotted = make_provider(OmegaConf.create({"name": "siamlm.providers.base.MockProvider", "model": "m"}))
    assert isinstance(dotted, MockProvider)
    with pytest.raises(ValueError):
        make_provider(OmegaConf.create({"name": "nonesuch", "model": "m"}))


def test_missing_key_is_an_error_without_any_request(monkeypatch):
    import requests
    monkeypatch.delenv("SIAMLM_TEST_NO_KEY", raising=False)
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("no request may be sent"))
    p = OpenAICompatibleProvider("openrouter", "a/b", "http://x", "SIAMLM_TEST_NO_KEY")
    r = p.chat([{"role": "user", "content": "q"}], seed=1, temperature=0.2, top_p=0.95)
    assert r.error_kind == "missing_key" and r.text is None


def test_context_limit_is_not_retried(monkeypatch):
    import requests

    class Resp:
        status_code, text, headers = 400, "This model's maximum context length is 8192 tokens", {}
    sent = []
    monkeypatch.setenv("SIAMLM_TEST_KEY", "k")
    monkeypatch.setattr(requests, "post", lambda *a, **k: sent.append(k["json"]) or Resp())
    p = OpenAICompatibleProvider("other", "a/b", "http://x", "SIAMLM_TEST_KEY", max_retries=3, drop_unsupported_params=False)
    r = p.chat([{"role": "user", "content": "q"}], seed=1, temperature=0.2, top_p=0.95, tools=[SUBMIT])
    assert r.error_kind == "context_limit" and len(sent) == 1 and sent[0]["tools"] == [SUBMIT]


def test_an_openai_compatible_endpoint_without_a_key(monkeypatch):
    """provider=openai (e.g. a vLLM server of a policy being trained): base_url from the config, no key needed (no
    Authorization header then), tool calls parsed, nothing OpenRouter-specific in the request."""
    import requests
    from siamlm import run as runner
    cfg = runner.compose(["provider=openai", "provider.base_url=http://localhost:9/v1"]).provider
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    sent = []

    class Resp:
        status_code, headers, text = 200, {}, ""

        def json(self):
            return {"id": "x", "model": "m", "usage": {"prompt_tokens": 5, "completion_tokens": 2}, "choices": [
                {"finish_reason": "tool_calls", "message": {"content": "", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "submit", "arguments": "{\"answer\": \"4\"}"}}]}}]}
    monkeypatch.setattr(requests, "post", lambda url, **k: sent.append((url, k)) or Resp())
    p = make_provider(cfg)
    r = p.chat([{"role": "user", "content": "hi"}], seed=1, temperature=1, top_p=1, tools=[SUBMIT])
    url, k = sent[0]
    assert url == "http://localhost:9/v1/chat/completions" and "Authorization" not in k["headers"]
    assert "usage" not in k["json"] and "provider" not in k["json"]
    assert r.error is None and r.tool_calls == [{"id": "c1", "name": "submit", "arguments": {"answer": "4"}}]
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    p.chat([{"role": "user", "content": "hi"}], seed=1, temperature=1, top_p=1)
    assert sent[1][1]["headers"]["Authorization"] == "Bearer k"
    with pytest.raises(ValueError, match="needs provider.base_url"):
        make_provider(OmegaConf.create({"name": "openai", "model": "m"}))


def test_hf_reply_parsing():
    """provider=hf: Hermes-style <tool_call> blocks become tool calls, <think> becomes reasoning; a broken block stays."""
    from siamlm.providers.hf import hf_messages, parse_reply
    text, calls, reasoning = parse_reply('<think>hm</think>Let me look.\n<tool_call>\n{"name": "bash", "arguments": '
                                         '{"command": "ls"}}\n</tool_call>\n<tool_call>{oops}</tool_call>')
    assert reasoning == "hm" and calls == [{"id": "hf_0", "name": "bash", "arguments": {"command": "ls"}}]
    assert text == "Let me look.\n\n<tool_call>{oops}</tool_call>"
    out = hf_messages([{"role": "assistant", "content": None, "tool_calls": [
        {"id": "t1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}]}])
    assert out[0]["tool_calls"][0]["function"] == {"name": "bash", "arguments": {"command": "ls"}} and out[0]["content"] == ""
