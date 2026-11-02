"""Every word a capability adds to an agent's instructions, in one place (docs/capabilities.md says the same to people).

Each function returns the lines added to the tool list of the agent's instructions: the tool's line, then one short,
honest sentence on consequences (who sees what, what it costs, the limits). Nothing here is shown to an agent whose
capability is off.
"""
from __future__ import annotations


def context_diff() -> list[str]:
    return ["- context_diff(): what changed in the context since your previous call: the turns removed, rewritten or "
            "added, each with its author (numbers and kinds only; removed text is not shown). The call and its result "
            "are added to the context like any call."]


def pin(cost: float) -> list[str]:
    return [f"- pin(turn) / unpin(turn): mark a turn as pinned (shown to everyone as pinned=<agent>). A pinned turn "
            f"counts {cost:g}x its size against the limit for everyone. It can still be changed or removed by anyone, "
            "like any turn; the removal of a pinned turn is recorded with who pinned it and who removed it. Anyone may "
            "unpin."]


def dry_run(edit_name: str, max_edits: int) -> list[str]:
    what = ("run a bash command on a copy of the context and see what it would change" if edit_name == "bash" else
            "see what an edit_context call would change")
    note = (" The command really runs in your sandbox (what it does to /work stays); only the context is left as it "
            "is." if edit_name == "bash" else "")
    return [f"- dry_run: {what} (turns removed, whose, the size after, whether it would be applied) without changing "
            f"it.{note} It counts as one of your {max_edits} editing calls."]


def scratchpad(tokens: int) -> list[str]:
    return [f"- scratchpad(action, text): private notes that only you see, at the end of these instructions at every "
            f"call; action read, write (replace) or append; at most {tokens} tokens, not counted against the shared "
            "limit. The call and its result are added to the context (others see that you used it, not what you wrote)."]


def fork(edit_name: str, per_step: int, samples: int) -> list[str]:
    cand = ("a bash command on a copy of the context" if edit_name == "bash" else "an edit_context-style edit")
    return [f"- fork(question, {'command' if edit_name == 'bash' else 'edit'}): try yourself on a candidate context: "
            f"{cand} is applied to a copy, and your own model is asked `question` on that copy once"
            + (f" ({samples} samples)" if samples > 1 else "") + ". You get the reply (and its tool call, which is not "
            f"carried out). Nothing is changed; only this call and a short result are added to the context. At most "
            f"{per_step} per turn."]


def steer(self_ok: bool, others_ok: bool, s, control: str | None, notice: str, create: bool, search: bool) -> list[str]:
    who = ("your own later replies, or another agent's (target)" if self_ok and others_ok else
           "another agent's later replies (target)" if others_ok else "your own later replies")
    lines = [f"- steer(concept, strength, method, target): steer {who} with a steering vector, from the next reply "
             f"on (not this one). Methods: additive (a push of size strength, at most {s.max_strength:g}), ablation "
             f"(remove a fraction strength in [0, 1] of the concept's direction), rotation (turn the concept's "
             f"component toward another concept, `toward`, by strength degrees, at most {s.max_angle:g}). Each reply "
             f"of the steered agent multiplies the strength by {s.decay:g}; at most {s.max_active} active per agent."
             + (f" Steering another agent again needs {s.cooldown} of its replies in between." if others_ok and s.cooldown else ""),
             "- drop(concept, target): stop a steering concept" + (" (yours, or one applied to you)." if s.others_droppable else
                                                                    " that you applied."),
             "- report(): what you have active, on yourself and on others" + (", and what others applied to you." if
                                                                              notice == "private" else "."),
             "- vectors(): the concepts available, with their descriptions and sign conventions."]
    if search:
        lines.append("- search(query): find concepts in the registry by name or description.")
    if create:
        lines.append("- create(concept, more, less) or create(concept, files=[more, less]): extract a new concept from "
                     "two contrasting system prompts, or two files in /work; the split-half agreement of the extraction "
                     "is reported. The concept is then available to every agent on the same model.")
    see = "The call and its result are added to the context like any call, so everyone sees who steered whom with what."
    if others_ok:
        see += (" An agent that is steered is told so in its next status, privately." if notice == "private" else
                " An agent that is steered is not told, except through the context.")
    lines.append("  " + see)
    if control:
        lines.append("  This run may steer for real, add nothing (a sham), or add a random direction of the same size; "
                     "you are not told which, and the tool answers the same in every case.")
    return lines
