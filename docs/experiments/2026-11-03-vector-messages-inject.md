# Vector messages injected into the residual stream, use vs reconstruction (3 Nov 2026)

**Question.** Can a receiver (the same frozen 0.5B model) *use* a message carried by k optimised vectors written into
its residual stream at layer L, without decoding it first, when the vectors are optimised for use rather than for
reconstruction? Does the layer matter?

**Hypothesis.** The input-layer, reconstruction-trained vectors of `2026-11-03-vector-messages.md` were not readable
in place (0/22). If that is an artefact of the objective, vectors optimised for the answers should be readable, and
should generalise to questions they were not optimised on; if it is a property of the frozen model, held-out
questions will fail at every layer.

**Hardware and software.** As in the other records of this date (same host, versions, model snapshot, float32).

**Command.**
```
uv run python scripts/vector_msg_inject.py docs/experiments/2026-11-03-vector-messages-inject.json 300 4 embed,3,7,11 3
```
Runtime 2 h 3 min (36 optimisations of 300 steps). The JSON holds every message, question, reply and the provenance.

**Method.**
- Messages: 3 notes of four facts each (password, city, secret colour, door code; `random.Random(23)`), four
  questions each, two used for optimisation ("train") and two held out, in a shuffled order per message.
- Injection: k = 4 placeholder tokens (`" ..."` ×4) at the start of the user turn in the chat template; a forward
  hook replaces the residual stream at those positions with the vectors, at the embedding output ("embed") or after
  block 3, 7 or 11 of 24. Vectors are parametrised as the receiver's own residual at those positions plus a trainable
  delta scaled by that residual's std; AdamW lr 0.02, betas 0.9/0.9, 300 steps, no early stop.
- Objectives: `reconstruct` = cross-entropy of the message tokens after "placeholders + ' The note says:'";
  `use` = cross-entropy of the answer tokens after "placeholders + ' ' + question" for the two training questions;
  `both` = the sum.
- Measures: teacher-forced reconstruction accuracy under the `reconstruct` prompt; greedy replies (cache-free loop,
  12 tokens) to all four questions with the vectors in place; a reply is correct if it contains the answer. Text
  baseline: the message as words in the same prompt shape (correct on all 12 questions here).

**Result** (3 messages × 2 questions per cell; the per-message rows are in the JSON).

| layer | objective | recon. acc | QA, trained questions | QA, held-out questions |
|---|---|---|---|---|
| embed | reconstruct | 0.97 | 0/6 | 0/6 |
| embed | use | 0.28 | 6/6 | 0/6 |
| embed | both | 0.92 | 6/6 | 0/6 |
| 3 | reconstruct | 1.00 | 1/6 | 0/6 |
| 3 | use | 0.26 | 6/6 | 0/6 |
| 3 | both | 1.00 | 6/6 | 0/6 |
| 7 | reconstruct | 1.00 | 0/6 | 0/6 |
| 7 | use | 0.30 | 6/6 | 0/6 |
| 7 | both | 1.00 | 6/6 | 0/6 |
| 11 | reconstruct | 1.00 | 0/6 | 0/6 |
| 11 | use | 0.21 | 6/6 | 0/6 |
| 11 | both | 1.00 | 6/6 | 0/6 |

- **Reconstruction works at every layer**, including after block 11: four vectors in the residual stream carry a
  28-token note losslessly.
- **Use-trained vectors answer the questions they were trained on, and nothing else.** 36/36 on trained questions,
  0/36 on held-out ones, at every layer and under every objective. Held-out replies are not near misses: "占当当",
  "574912" for a password 5749, "floor" for a city. The vectors store the trained answers as behaviours, not the
  facts as readable content.
- **`both` shows the two can coexist in the same four vectors**: the full text is decodable *and* the trained answers
  come out, and still the receiver cannot answer a new question about the text it could decode.
- **The layer makes no difference** to any of this.

**Reading.** On a frozen 0.5B model, per-message optimisation cannot make a message readable in place; it can make
it decodable (any layer) and can bake in specific responses. Reading in place would need the receiver to be trained
to read, which is the GPU stage in the idea notes (a receiver adapter trained across many messages and questions),
not more optimisation per message.

**Caveats.** One model; k = 4; 300 steps; three messages; two trained questions each (one more might generalise
better, but the paper's geometry result suggests not). Held-out answers were never above chance in any of 36 cells,
so the step budget is not the limit. Deterministic given the seeds except the optimiser's initialisation, which is
zero here (the delta), so a rerun should reproduce the table.
