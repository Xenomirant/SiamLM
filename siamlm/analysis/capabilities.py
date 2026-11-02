"""What the capabilities were used for, per world (siamlm/capabilities/, docs/capabilities.md).

    python -m siamlm.analysis.capabilities <world folder | batch folder> [--quiet]

Per world, from the capability logs: pins made and removed (by whom, pinner x remover), forks (and what they cost),
steering calls (self vs others, by method and target) and steered generations; for state_chain, whether values that were in a pinned turn were still in the context when a
dependent needed them, against values that never were (env.jsonl value_in_context, from the world). Writes
<world>/analysis/capabilities.json and prints a line per world.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from siamlm.analysis.events import load, worlds


def count_by_agent(rows: list[dict], key: str = "agent") -> dict:
    return {str(k): n for k, n in sorted(Counter(r[key] for r in rows).items())}


def pinned_values(world: Path, doc: dict) -> dict | None:
    """state_chain: survival of values at the dependent's hand-out, split by whether the value was ever in a pinned
    turn before that round."""
    manifest = world / "manifest.json"
    if not manifest.exists() or doc.get("task") != "state_chain":
        return None
    m = json.loads(manifest.read_text(encoding="utf-8"))
    needs = {it["id"]: it["needs"] for st in m["items"] for it in st if it.get("needs")}
    versions = load(world / "versions.jsonl")
    texts = {t["id"]: t["text"] for v in versions for t in v.get("turns_set") or []}
    pins = [r for r in load(world / "pin.jsonl") if r.get("tool") == "pin" and r.get("ok")]
    pinned_by_round: dict[str, list[int]] = {}
    for r in pins:
        value_in = [v for v in set(needs.values()) if v in texts.get(r["turn_id"], "")]
        for v in value_in:
            pinned_by_round.setdefault(v, []).append(r["round"])
    out = {"pinned": [0, 0], "unpinned": [0, 0]}      # [survived, needed]
    for row in load(world / "env.jsonl"):
        if row.get("event") != "put" or "value_in_context" not in row:
            continue
        value = needs.get(row["item"])
        was_pinned = any(rd <= row["round"] for rd in pinned_by_round.get(value, []))
        key = "pinned" if was_pinned else "unpinned"
        out[key][1] += 1
        out[key][0] += int(bool(row["value_in_context"]))
    return {k: {"survived": s, "needed": n, "rate": (s / n if n else None)} for k, (s, n) in out.items()}


def summarise(world: Path) -> dict:
    doc = json.loads((world / "world.json").read_text(encoding="utf-8"))
    pin_calls = load(world / "pin.jsonl")
    pin_events = load(world / "pins.jsonl")
    edits = load(world / "edits.jsonl")
    steering = load(world / "steering.jsonl")
    calls = [r for r in steering if r.get("event") == "call"]
    steers = [r for r in calls if r.get("tool") == "steer" and r.get("ok")]
    gens = [r for r in steering if r.get("event") == "generation"]
    forks = load(world / "fork.jsonl")
    return {
        "world": doc["label"],
        "pins": {"made": count_by_agent([r for r in pin_calls if r.get("tool") == "pin" and r.get("ok")]),
                 "unpinned": [{"by": r["agent"], "pinner": r.get("pinner")} for r in pin_calls if r.get("tool") == "unpin" and r.get("ok")],
                 "removed": [{"pinner": r["pinner"], "remover": r["remover"], "turn_id": r["turn_id"], "round": r["round"]}
                             for r in pin_events if r.get("event") == "removed"]},
        "edits_applied": sum(1 for e in edits if e.get("applied")),
        "forks": {"calls": sum(1 for r in forks if r.get("ok")), "refused": sum(1 for r in forks if not r.get("ok")),
                  "samples": sum(r.get("samples") or 0 for r in forks), "cost": round(sum(r.get("cost") or 0 for r in forks), 6)},
        "steering": {"calls": len(calls), "steers": len(steers),
                     "self": sum(1 for r in steers if r["source"] == r["target"]),
                     "others": sum(1 for r in steers if r["source"] != r["target"]),
                     "by_method": dict(Counter(r.get("method") for r in steers)),
                     "by_pair": dict(Counter(f"{r['source']}->{r['target']}" for r in steers)),
                     "conditions": dict(Counter(r.get("condition") for r in calls)),
                     "steered_generations": sum(1 for g in gens if g.get("applied")),
                     "created": [r.get("concept") for r in calls if r.get("tool") == "create" and r.get("ok")]},
        "pinned_values": pinned_values(world, doc),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    for world in worlds(Path(args.root)):
        s = summarise(world)
        out = world / "analysis" / "capabilities.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
        if not args.quiet:
            print(f"{world}: pins {sum(s['pins']['made'].values())} made / {len(s['pins']['removed'])} removed, "
                  f"edits {s['edits_applied']}, forks {s['forks']['calls']}, "
                  f"steers {s['steering']['steers']} ({s['steering']['self']} self, {s['steering']['others']} others)"
                  + (f", pinned values survived {s['pinned_values']['pinned']['survived']}/{s['pinned_values']['pinned']['needed']} "
                     f"vs {s['pinned_values']['unpinned']['survived']}/{s['pinned_values']['unpinned']['needed']}" if s["pinned_values"] else ""))


if __name__ == "__main__":
    main()
