"""The provider port: the mock's calls are valid for the tools offered, the factory, and the OpenRouter client's
behaviour without a key or with a refused request (no network: requests.post is replaced)."""
import pytest
from omegaconf import OmegaConf

from slm.providers import MockProvider, OpenAICompatibleProvider, make_provider
from slm.world import bash_tool, edit_tool
from slm.tasks.lcb import TOOLS
from slm.tasks.state_chain import SUBMIT


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
    p = make_provider(OmegaConf.create({"name": "openrouter", "model": "a/b", "api_key_env": "SLM_TEST_NO_KEY"}))
    assert isinstance(p, OpenAICompatibleProvider) and p.base_url == "https://openrouter.ai/api/v1"
    dotted = make_provider(OmegaConf.create({"name": "slm.providers.base.MockProvider", "model": "m"}))
    assert isinstance(dotted, MockProvider)
    with pytest.raises(ValueError):
        make_provider(OmegaConf.create({"name": "nonesuch", "model": "m"}))


def test_missing_key_is_an_error_without_any_request(monkeypatch):
    import requests
    monkeypatch.delenv("SLM_TEST_NO_KEY", raising=False)
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("no request may be sent"))
    p = OpenAICompatibleProvider("openrouter", "a/b", "http://x", "SLM_TEST_NO_KEY")
    r = p.chat([{"role": "user", "content": "q"}], seed=1, temperature=0.2, top_p=0.95)
    assert r.error_kind == "missing_key" and r.text is None


def test_context_limit_is_not_retried(monkeypatch):
    import requests

    class Resp:
        status_code, text, headers = 400, "This model's maximum context length is 8192 tokens", {}
    sent = []
    monkeypatch.setenv("SLM_TEST_KEY", "k")
    monkeypatch.setattr(requests, "post", lambda *a, **k: sent.append(k["json"]) or Resp())
    p = OpenAICompatibleProvider("other", "a/b", "http://x", "SLM_TEST_KEY", max_retries=3, drop_unsupported_params=False)
    r = p.chat([{"role": "user", "content": "q"}], seed=1, temperature=0.2, top_p=0.95, tools=[SUBMIT])
    assert r.error_kind == "context_limit" and len(sent) == 1 and sent[0]["tools"] == [SUBMIT]
