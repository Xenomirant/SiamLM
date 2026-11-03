# Vector messages, input layer, reconstruction objective (3 Nov 2026)

**Question.** After Kuratov et al. (ACL 2025): on Qwen2.5-0.5B-Instruct, how many tokens of a message can k optimised
input vectors carry, and can a receiver (the same frozen model) *use* such a message without decoding it first?

**Hypothesis.** Compression works at this size for short messages (their Pythia-410M carries 96 tokens in one vector).
Use without decoding is the open question; the vectors are optimised for reconstruction only, so there is no reason
it should work, and if it does not, the channel is "compressed text" rather than a readable message.

**Hardware and software.** AMD Ryzen 7 9700X, 16 threads, no GPU; Python 3.12.3, torch 2.15.0+cpu, transformers
4.57.6; model snapshot `7ae557604adf67be50417f59c2c2f167def9a775`; float32; greedy decoding by a cache-free loop
(`greedy()` in the script: HF `generate` with a 1-vector `inputs_embeds` prefix produced degenerate output).

**Command.**
```
uv run python scripts/vector_msg.py docs/experiments/2026-11-03-vector-messages.json 500 16,32,64,128 1,4
```
Runtime 47 min. The JSON holds every message, question, decoded text and reply, and the provenance block.

**Method.** k trainable vectors (init `N(0, std of the embedding matrix)`) are prepended to the message's token
embeddings; the frozen model's next-token cross-entropy on the message tokens is minimised with AdamW (lr 0.01,
betas 0.9/0.9, weight decay 0.01), up to 500 steps, stopping when teacher-forced token accuracy reaches 1.0 (the
check runs before the update, so the returned vectors are the measured ones).
- Messages: "facts" (agent-style notes made of 2–8 fact sentences from six templates, no template repeated within six
  facts, `random.Random(11)`, with up to 3 questions whose answers are in the text) and "random" (words sampled from
  `/usr/share/dict/words`, 3–8 letters), each filled to the target token count.
- Reconstruction: teacher-forced accuracy (`tf_acc`) and whether greedy decoding from the vectors alone starts with
  the message (`exact`).
- Use: the receiver's reply to each question with a prefix of (a) the vectors (`qa_vectors`), (b) the message's
  token embeddings (`qa_text`), (c) the embeddings of the decoded text (`qa_decoded`), (d) nothing (`qa_none`); the
  question is in the chat template after the prefix; a reply is correct if it contains the answer (case-insensitive).

**Result.**

| message | tokens | k | steps | tf_acc | exact | QA vectors | QA text | QA decoded | QA none |
|---|---|---|---|---|---|---|---|---|---|
| facts | 15 | 1 | 128 | 1.00 | yes | 0/2 | 2/2 | 2/2 | 0/2 |
| facts | 15 | 4 | 56 | 1.00 | yes | 0/2 | 2/2 | 2/2 | 0/2 |
| facts | 27 | 1 | 305 | 1.00 | yes | 0/3 | 2/3 | 2/3 | 0/3 |
| facts | 27 | 4 | 48 | 1.00 | yes | 0/3 | 2/3 | 2/3 | 0/3 |
| facts | 57 | 1 | 500 | 0.74 | no | 0/3 | 2/3 | 0/3 | 0/3 |
| facts | 57 | 4 | 116 | 1.00 | yes | 0/3 | 2/3 | 2/3 | 0/3 |
| facts | 123 | 1 | 500 | 0.78 | no | 0/3 | 1/3 | 1/3 | 0/3 |
| facts | 123 | 4 | 140 | 1.00 | yes | 0/3 | 1/3 | 1/3 | 0/3 |
| random | 16 | 1 | 320 | 1.00 | yes | | | | |
| random | 16 | 4 | 40 | 1.00 | yes | | | | |
| random | 32 | 1 | 500 | 0.44 | no | | | | |
| random | 32 | 4 | 138 | 1.00 | yes | | | | |
| random | 64 | 1 | 500 | 0.16 | no | | | | |
| random | 64 | 4 | 364 | 1.00 | yes | | | | |
| random | 128 | 1 | 500 | 0.07 | no | | | | |
| random | 128 | 4 | 500 | 0.66 | no | | | | |

- **Capacity.** One vector: 27 tokens of facts and 16 random words, lossless, within 500 steps; 4 vectors: 123 tokens
  of facts and 64 random words. In the paper's terms this 0.5B model sits below Pythia-410M (96 PG-19 tokens per
  vector), with the caveat that our step budget is a tenth of theirs.
- **Use without decoding: never.** 0 of 22 questions answered from the vectors, against 14 of 22 from the text in the
  same position and 12 of 22 after decoding the vectors to text. Replies from the vectors are the no-message replies
  ("123456", "Blue"): the receiver treats the prefix as nothing.
- **Decode-then-answer** equals the text baseline whenever reconstruction is lossless.

**Caveats.** One model; 500 steps (the paper uses up to 5,000); `exact` requires the decode to *start* with the
message; the QA text baseline is itself weak on this model (14/22). Deterministic given the seeds except for the
vector initialisation (`torch.manual_seed(0)`); a rerun should reproduce the lossless/not pattern and the 0/22.

**Follow-up.** `2026-11-03-vector-messages-inject.md`: the vectors injected into the residual stream at a chosen
layer, optimised for use (the answers) instead of reconstruction.
