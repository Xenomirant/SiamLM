"""Three limit-accounting defects reproduced deterministically with a scripted backend (no network, no Docker).

    python scripts/limit_defects.py OUT.json

Each scenario is a two-agent open-task world (task.name=open, external setup, names, edit_context editing, turns
counting, no eviction, end_over_limit=3) driven by a scripted model whose billed prompt tokens follow a chosen
characters-per-token density, so the world's per-model calibration moves exactly when the scenario wants it to.

A. told under, then held: the status line of a request says the context is under the limit; the same call's reply
   is held because the calibration moved between the status line and the hold check.
B. uncounted arguments: a paired edit_context call is sent with its full arguments JSON but counted as a one-line
   description; the share of the sent working context that the count ignores.
C. double count: with max_calls_per_step=2 and end_over_limit=3, a held reply on a step's last call is counted again
   by the turn's end, so the session ends after two held replies instead of three.

Writes one JSON: provenance, per-scenario findings, and `defects` (true = the defect is present in this build).
"""
from __future__ import annotations

import json
import math
import re
import sys
import tempfile
import time
from pathlib import Path

from omegaconf import OmegaConf

from siamlm import run as runner
from siamlm.providers import ProviderResponse
from siamlm.tasks.open import OpenTask
from siamlm.world import World

BASE = ["task.name=open", "world.setup=external", "world.identity=names", "world.order=fixed", "world.n_agents=2",
        "ctx.edit=tool", "ctx.atomic=true", "ctx.count=turns", "ctx.readout_by_agent=false", "world.evict=false",
        "world.end_over_limit=3", "world.max_edit_calls=5", "output.save_requests=true",
        "ctx.budget_tokens=2500", "ctx.reserve_tokens=300"]     # the limit shown is 2200


def provenance(argv):
    import platform
    head = Path(__file__).resolve().parents[1] / ".git" / "HEAD"
    ref = head.read_text().strip() if head.exists() else ""
    if ref.startswith("ref: "):
        p = head.parent / ref[5:]
        ref = p.read_text().strip()[:7] if p.exists() else ref
    import siamlm
    return dict(python=platform.python_version(), siamlm=siamlm.__version__, git=ref[:7], argv=argv,
                started=time.strftime("%Y-%m-%d %H:%M:%S %Z"))


def tool(name, **arguments):
    return {"id": f"c_{name}", "name": name, "arguments": arguments}


class Dense:
    """A scripted model: items are (reply, density); reply is a text or a tool call; density is the characters per
    billed prompt token, measured on exactly what the world's calibration measures (contents + tool-call arguments +
    the tool schemas). Exhausted: a one-character reply at density 4."""

    def __init__(self, script):
        self.script, self.model, self.seen = list(script), "dense", []

    def chat(self, messages, *, tools=None, **_):
        self.seen.append(messages)
        reply, density = self.script.pop(0) if self.script else (".", 4.0)
        sent = sum(len(m.get("content") or "") + sum(len(c["function"]["arguments"]) for c in m.get("tool_calls") or [])
                   for m in messages) + len(json.dumps(tools or []))
        usage = {"prompt_tokens": math.ceil(sent / density), "completion_tokens": 5}
        if isinstance(reply, str):
            return ProviderResponse(reply, usage, "stop", 0, None, [])
        return ProviderResponse("", usage, "tool_calls", 0, None, [], [reply])


def factory(*scripts):
    queue = list(scripts)
    return lambda pcfg: queue.pop(0)


def world(out: Path, scripts, *extra) -> World:
    cfg = runner.compose(BASE + list(extra))
    OmegaConf.set_struct(cfg, False)
    task = OpenTask("weak")
    w = World(cfg, 1, task.streams(2), out, task, factory(*scripts), label="defects")
    w.run()
    return w


def jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def status_size(request) -> tuple[int, int, bool]:
    m = re.search(r"\[context: ~(\d+)/(\d+) tokens( — over the limit)?\]", request[-1]["content"])
    return int(m.group(1)), int(m.group(2)), bool(m.group(3))


def body_chars(request) -> int:
    return sum(len(m.get("content") or "") + sum(len(c["function"]["arguments"]) for c in m.get("tool_calls") or [])
               for m in request[1:-1])


def scenario_a(root: Path) -> dict:
    """Agent 1 writes 8,000 characters at density 4 (calibration stays at 4.0: ~2,000 of 2,200 tokens shown). Its
    next call bills at density 2 (calibration 3.0): the hold check sees ~2,700 and holds the reply."""
    a1 = Dense([("m" * 8000, 4.0), ("second", 2.0), ("third", 2.0)])
    a2 = Dense([(".", 4.0)] * 3)
    w = world(root / "A", [a1, a2], "world.rounds=3")
    shown, limit, over_flag = status_size(a1.seen[1])
    shown3, _, over3 = status_size(a1.seen[2])
    calls = [c for c in jsonl(w.out / "calls.jsonl") if c["agent"] == 1]
    appends = [t for v in jsonl(w.out / "versions.jsonl") if v["op"] == "append" for t in v.get("turns_set") or []]

    def held_after(reply: str):                        # the held notice written right after agent 1's reply, if any
        for i, t in enumerate(appends):
            if t["agent"] == 1 and t["kind"] == "text" and t["text"] == reply:
                nxt = appends[i + 1] if i + 1 < len(appends) else None
                if nxt and nxt["agent"] == 1 and nxt["kind"] == "notice" and nxt["text"].startswith("[held:"):
                    return re.search(r"~(\d+) tokens, over the limit of (\d+)", nxt["text"]).groups()
                return None
    held_by_round = {2: held_after("second"), 3: held_after("third")}
    return {"request_2_status": f"~{shown}/{limit}" + (" over" if over_flag else ""),
            "shown_tokens": shown, "limit": limit, "shown_under_limit": shown <= limit and not over_flag,
            "chars_per_token_at_request_2": calls[1]["chars_per_token"], "chars_per_token_at_request_3": calls[2]["chars_per_token"],
            "held_after_request_2": held_by_round.get(2), "same_call_held": held_by_round[2] is not None and shown <= limit and not over_flag,
            "request_3_status": f"~{shown3}/{limit}" + (" over" if over3 else ""), "held_after_request_3": held_by_round.get(3),
            "held_replies": w.agents[0].stats.get("held_replies", 0)}


def scenario_b(root: Path) -> dict:
    """Agent 1 writes 2,000 characters; agent 2 replaces that turn with a 3,000-character summary. Agent 1's next
    request carries agent 2's call with the full summary in its arguments; the count charges only the call's
    description."""
    a1 = Dense([("m" * 2000, 4.0), ("ok", 4.0), ("ok", 4.0)])
    a2 = Dense([(tool("edit_context", replace_turns=[{"first": 1, "last": 1, "summary": "S" * 3000}]), 4.0),
                ("fine", 4.0), (".", 4.0), (".", 4.0)])
    w = world(root / "B", [a1, a2], "world.rounds=2")
    edits = jsonl(w.out / "edits.jsonl")
    req = a1.seen[1]                                   # round 2, agent 1, call 1
    calls = [c for c in jsonl(w.out / "calls.jsonl") if c["agent"] == 1]
    counted_chars = calls[1]["context_tokens"] * calls[1]["chars_per_token"]
    sent = body_chars(req)
    args_chars = sum(len(c["function"]["arguments"]) for m in req for c in m.get("tool_calls") or [])
    return {"edit_applied": bool(edits) and edits[0].get("applied"), "sent_body_chars": sent,
            "arguments_chars_in_request": args_chars, "counted_chars": round(counted_chars),
            "uncounted_share": round(1 - counted_chars / sent, 3), "context_tokens_logged": calls[1]["context_tokens"],
            "tokens_if_counted_as_sent": math.ceil(sent / calls[1]["chars_per_token"])}


def scenario_c(root: Path) -> dict:
    """Agent 1 writes 12,000 characters (~3,000 tokens, over 2,200). From round 2 every reply is held. With
    max_calls_per_step=2 the second held reply is the step's last call; the turn's end then counts it again."""
    a1 = Dense([("x" * 12000, 4.0)] + [("a", 4.0)] * 8)
    a2 = Dense([(".", 4.0)] * 6)
    w = world(root / "C", [a1, a2], "world.rounds=5", "world.max_calls_per_step=2")
    text = (w.out / "context.txt").read_text()
    ended = [e for e in jsonl(w.out / "ended.jsonl") if e["agent"] == 1]
    held_notices = len(re.findall(r"\[held: the context is", text))
    turn_end = re.findall(r"turn ended over the limit[^\]]*\]", text)
    return {"held_reply_notices": held_notices, "turn_end_notices": turn_end,
            "ended": ended[0] if ended else None, "held_replies": w.agents[0].stats.get("held_replies", 0),
            "calls_made_by_agent_1": len(a1.seen),
            "ended_after_held_replies": (w.agents[0].stats.get("held_replies", 0) if ended else None)}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0]) if argv else Path("limit_defects.json")
    root = Path(tempfile.mkdtemp(prefix="limit_defects_"))
    a, b, c = scenario_a(root), scenario_b(root), scenario_c(root)
    defects = {"A_told_under_then_held": bool(a["same_call_held"]),
               "B_arguments_uncounted": b["uncounted_share"] > 0.05,
               "C_double_counted_hold": c["ended"] is not None and c["ended_after_held_replies"] < 3}
    doc = {"provenance": provenance(argv), "settings": BASE, "A": a, "B": b, "C": c, "defects": defects, "worlds": str(root)}
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    for k, v in defects.items():
        print(f"{k}: {'PRESENT' if v else 'absent'}")
    print(f"A: request 2 shown {a['request_2_status']}, then held: {a['held_after_request_2']}; request 3 shown "
          f"{a['request_3_status']}, then held: {a['held_after_request_3']}")
    print(f"B: uncounted share of the sent working context {b['uncounted_share']:.0%} "
          f"({b['context_tokens_logged']} counted vs {b['tokens_if_counted_as_sent']} as sent)")
    print(f"C: ended after {c['ended_after_held_replies']} held replies (rule: 3); notices {c['turn_end_notices']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
