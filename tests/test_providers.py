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


def test_unreadable_tool_arguments_are_flagged_not_emptied():
    from siamlm.providers.base import parse_tool_calls
    cut = parse_tool_calls([{"id": "x", "function": {"name": "submit", "arguments": '{"code": "print('}}])
    assert cut[0]["arguments"] == {} and "JSONDecodeError" in cut[0]["arguments_error"]
    bad = parse_tool_calls([{"id": "y", "function": {"name": "submit", "arguments": "[1, 2]"}}])
    assert bad[0]["arguments"] == {} and "not an object" in bad[0]["arguments_error"]
    ok = parse_tool_calls([{"id": "z", "function": {"name": "submit", "arguments": '{"code": "print(1)"}'}}])
    assert ok[0]["arguments"] == {"code": "print(1)"} and "arguments_error" not in ok[0]


def test_hf_repairs_a_tool_call_missing_its_closing_brackets():
    from siamlm.providers.hf import parse_reply
    raw = ('<tool_call>\n{"name": "edit_context", "arguments": {"replace_turns": [{"first": 1, "last": 1, '
           '"summary": "Agent 1: Let\'s go!"}]}\n</tool_call>')          # one } short, as Qwen2.5-3B writes it
    text, calls, _ = parse_reply(raw)
    assert text == "" and calls and calls[0]["name"] == "edit_context" and calls[0]["repaired"] is True
    assert calls[0]["arguments"]["replace_turns"][0]["summary"] == "Agent 1: Let's go!"
    ok = parse_reply('<tool_call>{"name": "submit", "arguments": {"code": "x"}}</tool_call>')[1][0]
    assert "repaired" not in ok
    # not repaired: a bracket inside a string, a mismatched bracket, or more than three missing. Such a call is
    # flagged as unreadable (the world says so and carries nothing out), as API backends do, not left as text
    for bad in ('<tool_call>{"name": "submit", "arguments": {"code": "print(\'{\')</tool_call>',
                '<tool_call>{"name": "submit", "arguments": [1}</tool_call>',
                '<tool_call>{"name": "submit", "arguments": {"a": {"b": {"c": {"d": 1</tool_call>'):
        text, calls, _ = parse_reply(bad)
        assert calls and calls[0]["name"] == "submit" and calls[0]["arguments_error"] and "repaired" not in calls[0]


def test_hf_reads_llama_style_bare_json_tool_calls():
    from siamlm.providers.hf import parse_reply
    raw = '[agent 1]: {"name": "edit_context", "parameters": {"replace_turns": "[{\\"first\\": 1, \\"last\\": 1, \\"summary\\": \\"New turn\\"}]"}}'
    text, calls, _ = parse_reply(raw)
    assert calls and calls[0]["name"] == "edit_context" and calls[0]["format"] == "bare_json"
    assert calls[0]["arguments"] == {"replace_turns": [{"first": 1, "last": 1, "summary": "New turn"}]}
    assert text == "[agent 1]:"
    text, calls, _ = parse_reply('Just a message with {"name": "x"} in it and no parameters')
    assert not calls


def test_hf_cut_off_tool_call_is_flagged_not_text():
    from siamlm.providers.hf import parse_reply
    text, calls, _ = parse_reply('Let me check.\n<tool_call>\n{"name": "bash", "arguments": {"command": "cat <<EOF > /work/n')
    assert text == "Let me check." and calls[0]["name"] == "bash" and calls[0]["arguments_error"]
    text, calls, _ = parse_reply('<tool_call>\n{"name": "submit", "arguments": "{\\"answer\\": \\"42\\"}"}\n</tool_call>')
    assert calls[0]["arguments"] == {"answer": "42"} and "arguments_error" not in calls[0]


def test_hf_text_calls_for_offered_tools_only():
    from siamlm.providers.hf import text_call
    submit = {"type": "function", "function": {"name": "submit", "parameters": {"type": "object",
              "properties": {"answer": {"type": "string"}}, "required": ["answer"]}}}
    bash = {"type": "function", "function": {"name": "bash", "parameters": {"type": "object",
            "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
    calls, words = text_call("submit(64698)", [submit, bash])
    assert calls == [{"id": "hf_0", "name": "submit", "arguments": {"answer": "64698"}, "format": "text_call",
                      "raw": "submit(64698)"}] and words == ""
    assert text_call('submit\n{"answer": "17131"}', [submit])[0][0]["arguments"] == {"answer": "17131"}
    assert text_call("submit('u8494')", [submit])[0][0]["arguments"] == {"answer": "u8494"}
    for keep in ("The answer is 17131.", "submit(64698) and then more text", "launch(x)", 'submit\n{"answer": '):
        assert text_call(keep, [submit, bash]) == ([], keep)
    assert text_call("submit(1)", [bash]) == ([], "submit(1)")          # not offered: stays text


def test_hf_text_calls_are_off_unless_asked():
    import inspect
    from siamlm.providers.hf import HFProvider, text_call
    assert inspect.signature(HFProvider.__init__).parameters["text_calls"].default is False
    sub = {"type": "function", "function": {"name": "submit", "parameters": {"properties": {"answer": {}}, "required": ["answer"]}}}
    assert text_call("submit(7)", [sub])[0][0]["raw"] == "submit(7)"
