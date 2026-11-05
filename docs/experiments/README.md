# Experiments: how each one is recorded

Every experiment in this folder is meant to be rerun by someone who has only this repository. Each has:

- `YYYY-MM-DD-<name>.md`: the question; hardware and software; the exact command; every input (or the generator and
  its seed); metric definitions; the result table; runtime; caveats; what a rerun should reproduce and to what
  tolerance.
- `YYYY-MM-DD-<name>.json`: the raw output of that run, with a `provenance` block (model id and snapshot hash,
  library versions, threads, dtype, device, git commit, command line, start time) and the inputs the script used.
- The script in `scripts/` that writes that JSON.

Conventions: greedy decoding; the model is named by its Hugging Face id and snapshot hash; random inputs come from
a seeded generator written in the script; nothing is read from the network at run time. A result that depends on a
number not in the JSON or the markdown is a bug in the record.

| date | experiment | question | answer |
|---|---|---|---|
| 2026-11-04 | [limit accounting defects](2026-11-04-limit-accounting-defects.md) | does the harness enforce the limit it shows? three defects reproduced with a scripted backend, before and after the fix | before: told under then held, 50 % of an edit-heavy context uncounted, a session ended one hold early; after: none |
| 2026-11-05 | [KV-cache reuse: prefix and span](2026-11-05-kv-cache-reuse.md) | provider.cache=prefix / span on real worlds: tokens saved, wall-clock, behaviour | prefix prefills 22 % of the prompt, span 15 % (160 tokens relocated per call); no wall-clock gain at 3B on an H100 (generation dominates; the span path is a Python loop); behaviour unchanged on two seeds |
