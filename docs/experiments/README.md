# Experiments: how each one is recorded

Every experiment in this folder is meant to be rerun by someone who has only this repository. Each has:

- `YYYY-MM-DD-<name>.md`: purpose and hypothesis; hardware and software; the exact command; every input (or the
  generator and its seed); metric definitions with formulas; the result table; runtime; caveats; what a rerun should
  reproduce and to what tolerance.
- `YYYY-MM-DD-<name>.json`: the raw output of that run, with a `provenance` block (model id and snapshot hash,
  library versions, threads, dtype, device, CPU, git commit, command line, start time) and the inputs the script used.
- The script in `scripts/`, which writes that JSON. Scripts print one line per condition as they go.

Conventions: CPU runs use `torch.set_num_threads(16)` and float32; greedy decoding everywhere; the model is named by
its Hugging Face id and snapshot hash; random inputs come from a seeded generator written in the script; nothing is
read from the network at run time (`local_files_only`). A result that depends on a number not in the JSON or the
markdown is a bug in the record.

| date | experiment | question | answer |
|---|---|---|---|
| 2026-11-03 | [meter calibration](2026-11-03-meter-calibration.md) | which meter of a weight edit tracks its behavioural effect: weight distance or output change? | output change (Spearman 0.94 with KL) over relative Frobenius (0.82); targeted edits are where they part |
| 2026-11-03 | [activation-message probe](2026-11-03-activation-message-probe.md) | do a sender's residuals written into a receiver's prompt carry the span's words? the sender's context? | the words, up to mid-depth (layers ≤ 8 at or above the text control; nothing from layer 16); the context, not at all |
