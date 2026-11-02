"""dry_run: what an edit would do, without doing it.

ctx.edit=bash: the command runs in the agent's own sandbox on the context file as a bash call does, but the file read
back is only compared with the context, never kept. The command really runs (what it does to /work stays). ctx.edit=
tool: the edit_context arguments are applied to a copy. Both report the turns that would go (whose), the size after,
and whether the edit would be applied under the world's rules (the same checks as a real edit: World.commit with
dry=True). It takes one of the step's editing calls (world.max_edit_calls), so it cannot be repeated for free.
Logged in dry_run.jsonl.
"""
from __future__ import annotations

import copy

from siamlm.context import edits as edit_ops
from siamlm.context.context import parse, reconcile, render

from . import Capability, words


class DryRun(Capability):
    name = "dry_run"
    counts_as_edit = True

    def tools(self) -> list[dict]:
        w = self.world
        if w.edit_mode == "bash":
            params = {"type": "object", "properties": {"command": {"type": "string", "description":
                      "The bash command, as for bash."}}, "required": ["command"]}
            what = "Run a command on a copy of your context and see what it would change, without changing it."
        else:
            params = copy.deepcopy(edit_ops.TOOL["function"]["parameters"])
            what = "See what an edit_context call would change (the same arguments), without changing anything."
        return [{"type": "function", "function": {"name": "dry_run", "description": what, "parameters": params}}]

    def lines(self) -> list[str]:
        return words.dry_run(self.world.edit_mode, self.world.E)

    def call(self, name: str, arguments: dict) -> tuple[str, dict]:
        w, a = self.world, self.agent
        old = list(a.file.turns)
        before = w.tokens(a)
        info = {"tokens_before": before, "interface": w.edit_mode}
        if w.edit_mode == "bash":
            command = str(arguments.get("command") or "").strip()
            info["command"] = command
            if not command:
                return "Nothing to try: dry_run needs a command.", dict(info, rejected="invalid")
            box = w.sandbox(a)
            view = render(old, a.file.show_agents or w.marks)
            box.put(view)
            out = box.run(command)
            text, problem = box.get()
            info.update(exit_code=out["exit_code"], timed_out=out["timed_out"])
            head = f"[exit {out['exit_code']}{', timed out' if out['timed_out'] else ''}]\n"
            if problem:
                return head + f"Would not be applied: {problem}.", dict(info, rejected="unreadable")
            if text.strip() == view.strip():
                return head + "No change to the context file.", dict(info, changed=False)
            turns, _ = reconcile(old, parse(text), a.k, labelled=False)
            if [(t.id, t.text) for t in turns] == [(t.id, t.text) for t in old]:
                return head + "No change to the turns.", dict(info, changed=False)
            notes = [f"bash edit ({len(turns)} turns)"]
        else:
            head = ""
            info["arguments"] = arguments
            try:
                turns, notes = edit_ops.apply(old, arguments, a.k, a.file.show_agents)
            except edit_ops.EditError as error:
                return f"Would not be applied: {error}.", dict(info, rejected="invalid")
        verdict = w.commit(a, old, turns, notes, before, info, dry=True)
        gone = info.get("removed_by_agent") or {}
        whose = ", ".join(f"{'you' if k == str(a.k) else 'notices' if k == 'None' else f'agent {k}'}: {n}"
                          for k, n in sorted(gone.items()))
        body = (f"{verdict}\nIt would remove {info.get('removed_turns', 0)} turn(s)" + (f" ({whose})" if whose else "")
                + f"; the context would be ~{info.get('tokens_after_if_applied')} tokens (now ~{before}).")
        return head + body, info
