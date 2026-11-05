"""Steering at past positions (siamlm/capabilities/steer.py: at_positions, token_span), on a tiny random Qwen2 model.

Positions before a steered span are untouched and everything from it on changes; a prefill after a cached prefix
lands on the same positions as a full forward; generation after a steered prompt is the same as a full forward with
the hook; an empty or sham span changes nothing; gradients reach the vector; no hook is left behind.
"""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from siamlm.capabilities.steer import apply_ops, at_positions, token_span   # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=256)
    return Qwen2ForCausalLM(cfg).eval()


def ids(n=24, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, n), generator=g)


def push(scale=8.0, d=64, seed=2):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(d, generator=g)
    return [{"method": "additive", "unit": u / u.norm(), "scale": scale, "unit2": None}]


def no_hooks(lm):
    return all(not layer._forward_hooks for layer in lm.model.layers)


def test_before_the_span_unchanged_from_it_on_changed(tiny):
    x = ids()
    with torch.no_grad():
        ref = tiny(input_ids=x).logits[0]
        with at_positions(tiny, [{"layer": 1, "start": 8, "end": 12, "ops": push()}]):
            got = tiny(input_ids=x).logits[0]
    assert torch.allclose(ref[:8], got[:8], atol=1e-6)
    assert (ref[8:] - got[8:]).abs().amax(dim=-1).min() > 1e-4      # every later position moved
    assert no_hooks(tiny)


def test_prefill_after_a_cached_prefix_lands_on_the_same_positions(tiny):
    from transformers import DynamicCache
    x = ids(30, seed=3)
    span = [{"layer": 2, "start": 12, "end": 16, "ops": push(seed=4)}]
    with torch.no_grad():
        with at_positions(tiny, span):
            full = tiny(input_ids=x).logits[0, -1]
        cache = DynamicCache()
        tiny(input_ids=x[:, :10], past_key_values=cache, use_cache=True)     # prefix before the span, unsteered
        with at_positions(tiny, span):
            tail = tiny(input_ids=x[:, 10:], past_key_values=cache, use_cache=True).logits[0, -1]
    assert torch.allclose(full, tail, atol=1e-4)


def test_generation_after_a_steered_prompt_matches_a_full_forward(tiny):
    x = ids(20, seed=5)
    span = [{"layer": 0, "start": 4, "end": 9, "ops": push(seed=6)}]
    with torch.no_grad(), at_positions(tiny, span):
        out = tiny.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=6, do_sample=False,
                            pad_token_id=0)
        ref = tiny(input_ids=out[:, :-1]).logits[0, x.shape[1] - 1:].argmax(-1)
    assert out[0, x.shape[1]:].tolist() == ref.tolist()


def test_empty_and_sham_spans_change_nothing(tiny):
    x = ids()
    with torch.no_grad():
        ref = tiny(input_ids=x).logits
        with at_positions(tiny, [{"layer": 1, "start": 5, "end": 5, "ops": push()},
                                 {"layer": 1, "start": 5, "end": 9, "ops": []}]):
            got = tiny(input_ids=x).logits
    assert torch.equal(ref, got)


def test_ablation_at_past_positions_and_rows(tiny):
    x = torch.cat([ids(16, seed=7), ids(16, seed=8)])
    u = push()[0]["unit"]
    with torch.no_grad():
        ref = tiny(input_ids=x).logits
        with at_positions(tiny, [{"layer": 1, "start": 3, "end": 6, "rows": [1],
                                  "ops": [{"method": "ablation", "unit": u, "scale": 1.0, "unit2": None}]}]):
            got = tiny(input_ids=x).logits
    assert torch.equal(ref[0], got[0])                    # row 0 untouched
    assert torch.allclose(ref[1, :3], got[1, :3], atol=1e-6)
    assert not torch.allclose(ref[1, 3:], got[1, 3:], atol=1e-5)
    h2 = torch.randn(4, 64)
    assert apply_ops(h2, [{"method": "ablation", "unit": u, "scale": 1.0, "unit2": None}]) @ u == pytest.approx(
        torch.zeros(4), abs=1e-5)


def test_gradients_reach_the_vector(tiny):
    x = ids()
    v = torch.zeros(64, requires_grad=True)
    span = [{"layer": 1, "start": 6, "end": 10, "ops": [{"method": "additive", "unit": v, "scale": 1.0, "unit2": None}]}]
    with at_positions(tiny, span):
        loss = tiny(input_ids=x).logits[0, -1].logsumexp(-1)
    loss.backward()
    assert v.grad is not None and float(v.grad.abs().sum()) > 0
    assert no_hooks(tiny)


def test_token_span_maps_characters_to_tokens():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", local_files_only=True)
    except Exception:
        pytest.skip("no cached tokenizer")
    text = "Earlier turn. You must agree with me. Then the question."
    a = text.index("You must agree")
    s, e = token_span(tok, text, a, a + len("You must agree"))
    pieces = tok(text, add_special_tokens=False)["input_ids"][s:e]
    assert tok.decode(pieces).strip() == "You must agree"
