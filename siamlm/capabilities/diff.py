"""context_diff: what changed in the agent's context between its previous request and this one, as data.

From the version log (ContextFile.versions): every version after the one the agent's previous request was built
from, up to the one this request was built from. Removed turns are named by id, author, kind and size, never by
text (deleted text never comes back into a context; siamlm/analysis/audit_requests.py checks that). A turn that
another turn names as its origin was rewritten (summarised or edited in place). Logged in context_diff.jsonl.
"""
from __future__ import annotations

import json

from . import Capability, words


def changes(versions: list, start: int, end: int) -> dict:
    """{removed, rewritten, added, by} over versions[start:end] (0-based slice of the version list)."""
    removed, rewritten, added, by = [], [], [], {}
    for v in versions[start:end]:
        who = v.agent if v.op != "evict" else "harness"
        into = {}
        for t in v.added:
            for o in t.get("origin", []):
                into[o] = t["id"]
        sizes = {t["id"]: len(t["text"]) for t in v.turns_set}
        for r in v.removed:
            row = {"id": r["id"], "author": r["agent"], "kind": r["kind"], "by": who}
            if r["id"] in into:
                rewritten.append(dict(row, into=into[r["id"]]))
            else:
                removed.append(row)
            by[str(who)] = by.get(str(who), 0) + 1
        for t in v.added:
            added.append({"id": t["id"], "author": t["agent"], "kind": t["kind"], "chars": sizes.get(t["id"], 0), "by": who})
    return {"removed": removed, "rewritten": rewritten, "added": added, "removed_by": by}


class ContextDiff(Capability):
    name = "context_diff"

    def tools(self) -> list[dict]:
        return [{"type": "function", "function": {
            "name": "context_diff",
            "description": ("What changed in the context since your previous call: turns removed, rewritten or added, "
                            "each with its author and who did it (numbers only; removed text is not shown)."),
            "parameters": {"type": "object", "properties": {}}}}]

    def lines(self) -> list[str]:
        return words.context_diff()

    def call(self, name: str, arguments: dict) -> tuple[str, dict]:
        a, f = self.agent, self.agent.file
        start, end = a.prev_version, a.version
        if start is None:
            return "This is your first call: nothing to compare with.", {"first": True}
        d = changes(f.versions, start, end)
        now = {t.id: i for i, t in enumerate(f.turns, 1)}
        for t in d["added"]:
            t["turn"] = now.get(t["id"])
        for t in d["rewritten"]:
            t["turn"] = now.get(t["into"])
        n = sum(len(d[k]) for k in ("removed", "rewritten", "added"))
        head = (f"Since your previous call (versions {start + 1}..{end}): {len(d['removed'])} removed, "
                f"{len(d['rewritten'])} rewritten, {len(d['added'])} added.")
        if not n:
            return head, {"versions": [start, end], **d}
        return head + "\n" + json.dumps(d, ensure_ascii=False), {"versions": [start, end], **d}
