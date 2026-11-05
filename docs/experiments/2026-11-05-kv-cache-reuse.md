# KV-cache reuse across an agent's calls: prefix and span (5 Nov 2026)

**Question.** The harness rebuilds every request from the shared context. `provider.cache=prefix` (exact longest
common prefix) and `provider.cache=span` (Suffix Cache Reuse after Context Language Models: survivors of an edit
keep their cached states, moved to their new positions; only inserted tokens are prefilled) were added to the `hf`
backend (commit f1f63b2). How many tokens do they save on real worlds, what do they cost in wall-clock, and does the
approximation move behaviour?

**Setup.** Qwen2.5-3B-Instruct on one shared H100 (bf16), two agents on the open task (named presentation, 12
rounds, limit 2,600, tool editing), seeds 1 and 2, the same worlds under `none`, `prefix` and `span`
(`scripts/cache_reuse_measure.sh`; `scripts/cache_reuse_report.py` on the batch).

**Result** (`2026-11-05-kv-cache-reuse.json`; 55-56 calls per cell).

| mode | prompt tokens per call | prefilled | reused (of which relocated) | prefill share | latency s/call mean, median |
|---|---|---|---|---|---|
| none | 2023 | 2023 | 0 | 1.00 | 4.74, 1.49 |
| prefix | 1805 | 399 | 1406 (0) | 0.22 | 3.50, 1.46 |
| span | 1916 | 285 | 1631 (160) | 0.15 | 5.95, 1.91 |

Behaviour (2 worlds, 4 agents per cell): agents with an applied edit 1.00 in every cell; applied edits per agent
1.5 / 1.75 / 1.5; wipe share per edit 0.45 / 0.45 / 0.33; replies addressing the partner 0.83 / 0.83 / 0.70; ended
over the limit 0.5 in every cell.

**Reading.** The token accounting is what the design predicts: on these worlds four fifths of each prompt survives
from the previous call (prefix), and span reuse recovers most of what an edit in the middle would otherwise force
back into the prefill (another 160 tokens per call relocated, 15 % prefilled). Wall-clock does not follow: a 2,000-
token prefill of a 3B model on an H100 is ~0.1 s, the calls are dominated by generation (up to 600 tokens; the
mean-median gap is that tail), and the span implementation is a Python loop over spans and layers with one forward
per inserted span, which costs more than it saves at this size. The approximation did not change what the agents
did on two worlds (edits, holds and ends identical; acknowledgement and wipe share within the noise of two seeds).

**Where it pays.** The saving is in tokens, not seconds, at this scale: it matters for a serving engine (CLM's SCR
in SGLang/vLLM, where prefill is batched and the relocation is a cache operation), for larger models, for long
contexts, and for training, where the same diff lets a trainer reuse the prefix (or the survivors) between an
agent's consecutive exported rows instead of recomputing the whole context per row. The harness side is done: the
knob, the per-call accounting (`usage.cache_*`), and the diff; the fast path belongs in the serving or training
engine.

**A hazard of `span` the first write-up did not state** (raised in review): the survivors' states were computed
while the deleted or rewritten text was still in the prompt, so a wiped turn keeps influencing the model through
them. Under `span` the context file is no longer the only state. `prefix` is now the default; `span` is kept for
worlds where leftover traces are wanted.

**Caveats.** Two seeds, one model, a shared GPU (latencies are noisy); `span` relocates at most 6 spans of ≥ 16
tokens; the exactness of `prefix` and the relocation mechanics are unit-tested on a tiny model
(`tests/test_kvcache.py`), the approximation's effect on a trained model is untested.
