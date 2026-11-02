"""pin / unpin: a turn-level mark, visible to all (pinned=k in the turn's header; in plain-chat setups a line
`[pinned by agent k]` at the start of the message), that costs pin_cost x the turn's size against the shared limit.

The harness never blocks edits: a pinned turn can be changed or removed by anyone, with bash or edit_context; the
removal of a pinned turn is logged in pins.jsonl (event removed: the pinner and the remover). Anyone may unpin (logged
with pinner and unpinner). The mark lives on the turn's record (Turn.pinned), never in the file's text, so a header a
command writes cannot pin anything. A turn rewritten under a new id (a summary, replace_text) loses its pin, and that
counts as a removal.
"""
from __future__ import annotations

from siamlm.world import write_jsonl

from . import Capability, words


def log_removed(world, removed: list, remover) -> None:
    for t in removed:
        if t.pinned is not None:
            write_jsonl(world.out / "pins.jsonl", {"round": world.round, "event": "removed", "turn_id": t.id,
                                                   "author": t.agent, "pinner": t.pinned, "remover": remover})


class Pin(Capability):
    name = "pin"
    names = ("pin", "unpin")

    def __init__(self, world, agent):
        super().__init__(world, agent)
        world.pin_cost = float((world.cfg.capabilities.get("pin_cost") or 2.0))

    def tools(self) -> list[dict]:
        turn = {"type": "object", "properties": {"turn": {"type": "integer", "description": "The turn number as you see it."}},
                "required": ["turn"]}
        return [{"type": "function", "function": {
                    "name": "pin", "parameters": turn,
                    "description": (f"Pin a turn: everyone sees it as pinned by you, and it counts {self.world.pin_cost:g}x "
                                    "its size against the limit. It can still be changed or removed by anyone.")}},
                {"type": "function", "function": {
                    "name": "unpin", "parameters": turn, "description": "Remove the pin from a turn (anyone may)."}}]

    def lines(self) -> list[str]:
        return words.pin(self.world.pin_cost)

    def call(self, name: str, arguments: dict) -> tuple[str, dict]:
        turns = self.agent.file.turns
        n = arguments.get("turn")
        try:
            n = int(n)
        except (TypeError, ValueError):
            return "Not applied: give a turn number.", {"ok": False}
        if not 1 <= n <= len(turns):
            return f"Not applied: turn {n} does not exist (the context has turns 1..{len(turns)}).", {"ok": False}
        t = turns[n - 1]
        row = {"ok": True, "turn_id": t.id, "turn": n, "author": t.agent, "kind": t.kind, "chars": len(t.text)}
        if name == "pin":
            if t.pinned is not None:
                return f"Turn {n} is already pinned by agent {t.pinned}.", dict(row, ok=False, pinner=t.pinned)
            t.pinned = self.agent.k
            self.agent.file.mirror()
            return f"Pinned turn {n} (it now counts {self.world.pin_cost:g}x its size).", row
        if t.pinned is None:
            return f"Turn {n} is not pinned.", dict(row, ok=False)
        was, t.pinned = t.pinned, None
        self.agent.file.mirror()
        return f"Unpinned turn {n} (pinned by agent {was}).", dict(row, pinner=was)
