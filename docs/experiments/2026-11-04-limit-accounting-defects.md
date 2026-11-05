# Limit accounting: three defects, before and after (4 Nov 2026)

**Question.** Does the harness enforce the limit it shows? Three suspected defects in the count and the hold rules,
each reproduced deterministically with a scripted backend, on the code before the fix and after it.

**Hypothesis.** All three are present at commit 14a952c and absent after commit 62a003a (the JSON provenance names it by its earlier id, 38766c7).

**Hardware and software.** Any machine; no model, no network, no Docker (the scripted backend is in the script).
Python 3.12, SiamLM 0.1.0. Runtime: about 1 minute per run.

**Command.**
```
# Before the fix: at commit 14a952c
uv run python scripts/limit_defects.py \
    docs/experiments/2026-11-04-limit-accounting-defects.before.json

# After the fix: at commit 62a003a
uv run python scripts/limit_defects.py \
    docs/experiments/2026-11-04-limit-accounting-defects.after.json
```

**Method.** Three two-agent open-task worlds (`task.name=open world.setup=external world.identity=names
ctx.edit=tool ctx.atomic=true ctx.count=turns world.evict=false world.end_over_limit=3 ctx.budget_tokens=2500
ctx.reserve_tokens=300`: the limit shown is 2,200 tokens). The scripted model bills each reply at a chosen
characters-per-token density, measured on exactly what the world's calibration measures, so the per-model calibration
moves when the scenario wants it to.

- **A, told under then held.** Agent 1 writes 8,000 characters at density 4 (the calibration stays at 4.0). Its next
  request's status line is read; that call's reply bills at density 2. Measured: the status shown for request 2 and
  whether the turn written right after that reply is a held notice (`versions.jsonl`), and the same for request 3.
- **B, uncounted arguments.** Agent 1 writes 2,000 characters; agent 2 replaces that turn with a 3,000-character
  summary (`replace_turns`). Measured, for agent 1's next request: the characters of the working context as sent
  (contents plus tool-call arguments, the placeholder "(start of the context)" left out) against the characters the
  count charged (`context_tokens x chars_per_token` from `calls.jsonl`); the uncounted share is 1 - counted / sent.
- **C, double count.** Agent 1 writes 12,000 characters (about 3,000 tokens). `world.max_calls_per_step=2`: from round
  2 every reply is held and the second held reply of a turn is its last call. Measured: `ended.jsonl` for agent 1
  (round, `held_in_a_row`) and the number of held replies before the end.

**Result.**

| scenario | before (14a952c) | after (62a003a) |
|---|---|---|
| A: request 2 shown | ~2001/2200 (under) | ~2001/2200 (under) |
| A: that reply held? | yes, "~2670 tokens, over the limit of 2200" | no |
| A: request 3 shown, held? | ~2734/2200 over; held | ~2670/2200 over; held |
| B: uncounted share of the sent working context | 50 % (773 counted of 1,536 as sent) | 0 % (1,530 of 1,536; the 6 are the merged-message separators) |
| C: session ended after held replies (rule: 3) | 2 (round 2, "turn ended over the limit ... That is 3 held in a row") | 3 (round 3) |

**Reading.** Before the fix an agent could be told it was under the limit and have the same call held (A); a working
context full of edit calls was counted at about half its sent size, and the calibration could not correct it
because it measures what is sent while the count did not (B); and with `end_over_limit=N` a session could end after
N-1 held calls whenever the last held call fell on a step's last call (C). After the fix: the size enforced is the
size shown (the calibration moves only between requests, so a reply's measurement applies to the next request,
which then shows "over" and is held: the correct order), the count charges the arguments of paired edit calls as
chat_messages sends them, and a turn's end counts as a held unit only when no call in it was already held.

**Consequences for earlier results.** Every run before 62a003a under-counted edit-heavy contexts (the README table's
hold rates and fill levels are lower bounds), and every `end_over_limit` end that followed a held call on a step's
last call came one hold early. The 19 re-recorded golden cases are the byte-level record of what changed.

**Caveats.** The scenarios are minimal; the size of B's effect in real runs depends on how many paired edit calls
survive in the context (atomic editing keeps them paired). The JSON files hold every number above.
