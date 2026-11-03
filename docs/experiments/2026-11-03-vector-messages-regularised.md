# Vector messages: early stopping, many questions, anchoring (3 Nov 2026)

**Question.** The injected, use-trained vectors of `2026-11-03-vector-messages-inject.md` answered only the questions
they were fitted on. Is that overfitting that a better-balanced optimisation avoids? Three remedies on the same
frozen 0.5B model, at layer 7, k = 4: (1) log the held-out accuracy during training (is there a point to stop at?);
(2) fit on many paraphrased questions instead of two; (3) anchor the vectors to the note's own residuals.

**Hypothesis.** (1) no peak to stop at; (2) and (3) should improve generalisation to unseen paraphrases; whether an
unseen *fact* becomes readable is the open question, since the probe showed real residuals are readable in place.

**Hardware and software.** As in the other records of this date. Runtime 2 h 7 min.

**Command.**
```
uv run python scripts/vector_msg_regularised.py docs/experiments/2026-11-03-vector-messages-regularised.json 300 4 7 3
```

**Method.** Three four-fact notes (`random.Random(23)`; password, city, colour, door code). Five paraphrases per fact
(in the script's `PARA`). Per message, one fact is held out entirely; for the other three, paraphrases 1–3 are the
training questions (9) and 4–5 the held-out paraphrases (6); the held-out fact's five questions are the third set.
Injection and optimisation as before (placeholders at the start of the user turn; hook after block 7; vectors =
init + std(init) × δ; AdamW lr 0.02, 300 steps; `use` loss = mean answer cross-entropy over the training set).
- `curve`: training set = the original two questions; every 10 steps, greedy replies on the two training questions,
  three held-out paraphrases and two held-out-fact questions.
- `many`: training set = the 9 paraphrases.
- `anchored`: init = the note's own residuals at layer 7 (inside the same prompt shape), mean-pooled into 4 chunks;
  loss += λ · mean‖v − anchor‖² / mean‖anchor‖², λ = 1; training set = two questions.
- `anchored+many`: both.
- `anchor alone`: the pooled residuals injected with no optimisation, on all 20 questions.
- `dist` = mean ‖v − anchor‖ / ‖anchor‖ after training.

**Result** (3 messages pooled; per-message rows and the full curves in the JSON).

| variant | fitted questions | unseen paraphrases | unseen fact | dist to anchor |
|---|---|---|---|---|
| curve (2 questions) | 6/6 | 6/18 | 0/15 | 1.51 |
| many (9 questions) | 27/27 | 11/18 | 0/15 | 1.55 |
| anchored | 6/6 | 8/18 | 0/15 | 0.10 |
| anchored + many | 27/27 | **15/18** | 0/15 | 0.14 |
| anchor alone, no optimisation | | 2/60 over all questions | | 0 |

- **No point to stop at.** In all three `curve` runs the held-out paraphrase score is 0–2 of 3 from step 50 to 300
  with no peak (message 0: 1/3 from step 100 on; message 2 oscillates 0–2); training loss is 0 by step 50.
- **Paraphrases: recoverable.** Many questions lift unseen paraphrases from a third to 61 %; anchoring alone to 44 %;
  together 83 %. Anchored vectors stay within 10–15 % of the real residuals; unanchored ones travel 150 %.
- **An unseen fact: never**, 0/45 across all variants, including the one with 83 % on paraphrases. The vectors learn
  "answers to a family of questions about three facts", not the note. The fourth fact is in the anchor (the note's
  own residuals) and is still not readable.
- **Pooling destroys readability.** The note's real residuals, mean-pooled 7 tokens to one position, read 2/60.
  (The probe read single tokens' residuals in place at this layer; pooling is the loss.)

**Reading.** The lookup-table result is not an optimisation-balance problem; it is that a frozen receiver reads
content only in the form its attention expects (one token's residual per position) and per-message optimisation of
k ≪ n positions cannot recreate that for content no loss asked about. Generalising to unseen facts needs either
k = n (no compression) or a receiver trained to read compressed positions, which is the GPU stage.

**Caveats.** One model, one layer, k = 4, λ = 1, three messages, templated paraphrases; "unseen fact" is 5 questions
on one fact per message. Deterministic given the seeds (δ starts at zero).
