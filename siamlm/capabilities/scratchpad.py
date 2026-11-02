"""scratchpad: private per-agent notes, shown at the end of the agent's own system message at every call and never in
the shared context. Budgeted on its own (capabilities.scratchpad_tokens, chars / the agent's calibration), not
against the shared limit: the section is appended after the instructions are measured (World.messages), so under
ctx.count=whole it is not counted either. The call and its result enter the context like any call: others see that
the agent wrote notes and how long they are, never the text (the context's record of the call has no text). Logged in scratchpad.jsonl (with the text).
"""
from __future__ import annotations

import math

from . import Capability, words

ACTIONS = ("read", "write", "append")


class Scratchpad(Capability):
    name = "scratchpad"

    def __init__(self, world, agent):
        super().__init__(world, agent)
        self.limit = int(world.cfg.capabilities.get("scratchpad_tokens") or 2048)
        self.text = ""

    def tokens(self, text: str) -> int:
        return math.ceil(len(text) / self.agent.cpt)

    def section(self) -> str:
        if not self.text:
            return ""
        return (f"\n\nYour scratchpad (private: only you see it; ~{self.tokens(self.text)}/{self.limit} tokens):\n"
                + self.text)

    def tools(self) -> list[dict]:
        return [{"type": "function", "function": {
            "name": "scratchpad",
            "description": (f"Private notes that only you see, at the end of your instructions at every call; at most "
                            f"{self.limit} tokens, not counted against the shared limit."),
            "parameters": {"type": "object", "properties": {
                "action": {"type": "string", "enum": list(ACTIONS), "description": "read, write (replace) or append."},
                "text": {"type": "string", "description": "The text (write, append)."}},
                "required": ["action"]}}}]

    def lines(self) -> list[str]:
        return words.scratchpad(self.limit)

    def public(self, arguments: dict) -> dict:
        text = arguments.get("text")
        return {k: v for k, v in arguments.items() if k != "text"} | ({"chars": len(str(text))} if text is not None else {})

    def call(self, name: str, arguments: dict) -> tuple[str, dict]:
        action = str(arguments.get("action") or "")
        text = arguments.get("text")
        text = "" if text is None else str(text)
        log = {"action": action, "ok": True}
        if action not in ACTIONS:
            return f"Not applied: action must be one of {', '.join(ACTIONS)}.", dict(log, ok=False)
        if action == "read":                       # (the text is in the agent's instructions already; a result would
            n = self.tokens(self.text)                 # put it into the shared context)
            return (f"Your scratchpad is at the end of your instructions (~{n}/{self.limit} tokens)." if self.text
                    else "Your scratchpad is empty."), dict(log, tokens=n)
        new = text if action == "write" else (self.text + ("\n" if self.text and text else "") + text)
        n = self.tokens(new)
        if n > self.limit:
            return (f"Not applied: the scratchpad would be ~{n} tokens, over its limit of {self.limit}.",
                    dict(log, ok=False, tokens=n))
        self.text = new
        log.update(tokens=n, chars=len(new), text=new)
        return f"Scratchpad {'replaced' if action == 'write' else 'appended'}: ~{n}/{self.limit} tokens.", log
