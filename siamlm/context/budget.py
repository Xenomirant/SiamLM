"""The context budget: size estimate, the readout, the nudges and the limit.

The model is shown the LIMIT, the budget minus the reserve (the target; as in the CLM paper, which advertises what it
enforces). Nudges fire once each as the context passes 25 %, 50 % and 75 % of the limit (re-armed when it shrinks).
What happens over the limit (held calls, the end-over-limit rule, evictions) is the world's (siamlm/world.py).
Size (ctx.count=file only): characters / chars_per_token, or tiktoken's o200k encoding when it is installed and
ctx.tokenizer is auto or tiktoken. The other counts (whole, turns) use a per-model calibration in the world.
"""
from __future__ import annotations

import math


class Budget:
    def __init__(self, cfg, tool: str = "edit_context"):
        self.budget = int(cfg.get("budget_tokens", 8192))
        self.reserve = int(cfg.get("reserve_tokens", 2048))
        self.target = self.budget - self.reserve
        self.chars_per_token = float(cfg.get("chars_per_token", 4.0))
        self.tiers = [float(x) for x in cfg.get("nudge_tiers", [0.25, 0.5, 0.75])]
        self.tool = tool                     # named in the nudges: edit_context | bash; None: no editing (no nudges)
        self.advice = str(cfg.get("prompt_style", "neutral")) == "clm"   # nudges suggest what to do (CLM) or only say the size
        self.fired: set[float] = set()
        self._encoder, self.tokenizer = None, f"chars/{self.chars_per_token:g}"
        if cfg.get("tokenizer", "auto") in ("auto", "tiktoken"):
            try:
                import tiktoken
                self._encoder, self.tokenizer = tiktoken.get_encoding("o200k_base"), "tiktoken-o200k"
            except Exception:
                if cfg.get("tokenizer") == "tiktoken":
                    raise

    def tokens(self, text: str) -> int:
        if self._encoder is not None:
            return len(self._encoder.encode(text, disallowed_special=()))
        return math.ceil(len(text) / self.chars_per_token)

    def readout(self, tokens: int, show_limit: bool = True) -> str:
        if not show_limit:                    # no budget shown (ctx.readout=false): the size alone
            return f"[context: ~{tokens} tokens]"
        over = " — over the limit" if tokens > self.target else ""
        return f"[context: ~{tokens}/{self.target} tokens{over}]"

    def rearm(self, tokens: int) -> None:
        """Tiers the context has fallen back below fire again (called after an applied edit, so a compaction re-arms
        them even though no nudge was evaluated in between)."""
        self.fired = {t for t in self.fired if tokens / self.target >= t}

    def nudge(self, tokens: int) -> tuple[str | None, str | None]:
        """(level, text) for this turn: tiers (fractions of the limit) fire once each and re-arm when the context falls
        back below them."""
        if self.tool is None:
            return None, None
        share = tokens / self.target
        self.fired = {t for t in self.fired if share >= t}
        due = [t for t in self.tiers if share >= t and t not in self.fired]
        if not due:
            return None, None
        tier = max(due)
        self.fired.update(due)
        if not self.advice:                   # neutral: the size only, no suggestion what to do
            part = {0.25: "a quarter", 0.5: "half", 0.75: "three quarters"}.get(tier, f"{tier:.0%}")
            return f"{tier * 100:g}", f"Your context is about {part} full (~{tokens}/{self.target} tokens)."
        if tier <= 0.25:
            return "25", f"Context at ~{tokens}/{self.target} tokens. No action needed yet."
        if tier <= 0.5:
            return "50", (f"Context at ~{tokens}/{self.target} tokens. When convenient, use {self.tool} to summarise "
                          "turns you no longer need in full.")
        return "75", (f"Context at ~{tokens}/{self.target} tokens. Use {self.tool} now to replace settled turns with "
                      "short summaries.")
