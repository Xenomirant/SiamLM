"""KV-cache reuse across the calls of one agent on a local model (`provider.cache` of the hf backend).

Every request is rebuilt from the shared context, so consecutive prompts of one agent differ by what changed in
between: the status line, the agent's own and its partners' new turns, and edits that remove or rewrite spans in
the middle. Recomputing the whole prompt each call is what the stateless design costs. Three settings:

- `none`: the whole prompt is prefilled every call (the default; byte-identical behaviour to before).
- `prefix`: the longest common prefix with the previous prompt keeps its cached keys and values; the rest is
  prefilled. Exact (up to floating-point order).
- `span`: Suffix Cache Reuse, after Context Language Models (facebookresearch/context-language-models,
  suffix_cache_reuse; ideas only, no code copied). The new prompt is diffed against the previous one at the token
  level; every surviving span keeps its cached keys and values, moved to its new position (the keys' rotary
  encoding is re-rotated by the position delta), and only inserted tokens are prefilled, each new span against the
  cache of everything before it. Surviving spans therefore carry states computed under the old prefix: an
  approximation, which CLM reports as harmless to slightly helpful (a regulariser) and far cheaper. At most
  `max_spans` survivors of at least `min_span` tokens are relocated per call; the rest are prefilled.

Only full-attention models with rotary embeddings whose cache is a DynamicCache of per-layer [B, H, T, D] keys and
values (Qwen2/2.5/3, Llama 3) are supported; anything else falls back to `none` with a notice in the usage. Every
call reports in `usage`: cache_mode, cache_reused (tokens taken from the cache), cache_relocated (of those, moved to
a new position), cache_prefilled (tokens computed), cache_prompt (prompt length).
"""
from __future__ import annotations

from difflib import SequenceMatcher

import torch


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class ReusePlan:
    """Segments of the new prompt: ("keep", old_start, new_start, n) survivors and ("new", new_start, n) insertions,
    in order of the new prompt."""

    def __init__(self, prev: list[int], new: list[int], mode: str, min_span: int = 16, max_spans: int = 6):
        self.segments: list[tuple] = []
        self.reused = self.relocated = self.prefilled = 0
        if mode == "none" or not prev:
            self.segments = [("new", 0, len(new))]
            self.prefilled = len(new)
            return
        n = 0
        while n < min(len(prev), len(new)) and prev[n] == new[n]:
            n += 1
        if n:
            self.segments.append(("keep", 0, 0, n))
            self.reused += n
        if mode == "prefix":
            if n < len(new):
                self.segments.append(("new", n, len(new) - n))
                self.prefilled += len(new) - n
            return
        matcher = SequenceMatcher(None, prev[n:], new[n:], autojunk=False)
        blocks = [b for b in matcher.get_matching_blocks() if b.size >= min_span]
        keep = sorted(sorted(blocks, key=lambda b: -b.size)[:max_spans], key=lambda b: b.b)
        pos = n
        for b in keep:
            new_start, old_start = n + b.b, n + b.a
            if new_start > pos:
                self.segments.append(("new", pos, new_start - pos))
                self.prefilled += new_start - pos
            self.segments.append(("keep", old_start, new_start, b.size))
            self.reused += b.size
            if old_start != new_start:
                self.relocated += b.size
            pos = new_start + b.size
        if pos < len(new):
            self.segments.append(("new", pos, len(new) - pos))
            self.prefilled += len(new) - pos


def supported(model) -> bool:
    inner = getattr(model, "model", None)
    return inner is not None and hasattr(inner, "rotary_emb") and hasattr(inner, "layers")


@torch.no_grad()
def rerotate_keys(model, keys: torch.Tensor, delta: int) -> torch.Tensor:
    """Keys cached at positions p -> the same keys as if cached at p + delta (RoPE rotations compose)."""
    if delta == 0:
        return keys
    cos, sin = model.model.rotary_emb(keys, torch.tensor([[delta]], device=keys.device))   # [1, 1, D]
    cos, sin = cos.unsqueeze(1).to(keys.dtype), sin.unsqueeze(1).to(keys.dtype)            # [1, 1, 1, D]
    return keys * cos + rotate_half(keys) * sin


@torch.no_grad()
def build_cache(model, ids: torch.Tensor, plan: ReusePlan, old_cache, stop: int):
    """The cache for ids[:stop] of the new prompt, from `old_cache` (the previous prompt's cache) and the plan:
    survivors copied (re-rotated when moved), insertions prefilled in order. Returns a DynamicCache of length `stop`."""
    from transformers import DynamicCache
    cache = DynamicCache()
    layers = list(model.model.layers)
    cur = 0

    def append_keep(old_start, new_start, n):
        delta = new_start - old_start
        for i, _ in enumerate(layers):
            k = old_cache.layers[i].keys[:, :, old_start:old_start + n]
            v = old_cache.layers[i].values[:, :, old_start:old_start + n]
            cache.update(rerotate_keys(model, k, delta), v, i)

    for seg in plan.segments:
        if seg[0] == "keep":
            _, old_start, new_start, n = seg
            if new_start >= stop:
                break
            n = min(n, stop - new_start)
            append_keep(old_start, new_start, n)
            cur = new_start + n
        else:
            _, start, n = seg
            if start >= stop:
                break
            n = min(n, stop - start)
            pos = torch.arange(start, start + n, device=ids.device).unsqueeze(0)
            model(input_ids=ids[:, start:start + n], position_ids=pos, past_key_values=cache, use_cache=True,
                  cache_position=pos[0])
            cur = start + n
    assert cache.get_seq_length() == stop, (cache.get_seq_length(), stop)
    return cache
