"""Every word a capability adds to an agent's instructions, in one place (docs/capabilities.md says the same to people).

Each function returns the lines added to the tool list of the agent's instructions: what the tool does, what it
costs and who sees what. Nothing about when, why or whether to use it: the study measures that. Nothing here is
shown to an agent whose capability is off.
"""
from __future__ import annotations


def pin(cost: float) -> list[str]:
    return [f"- pin(turn) / unpin(turn): a mark on a turn, shown to everyone as pinned=<agent>. A pinned turn counts "
            f"{cost:g}x its size against the limit for everyone. It can be changed or removed by anyone, like any turn; "
            "the removal of a pinned turn is recorded with who pinned it and who removed it. Anyone may unpin."]


def fork(edit_name: str, per_step: int, samples: int) -> list[str]:
    cand = ("a bash command is run on a copy of the context" if edit_name == "bash" else
            "an edit_context-style edit is applied to a copy of the context")
    return [f"- fork(question, {'command' if edit_name == 'bash' else 'edit'}): {cand}, and your own model is called "
            f"once on that copy with `question` in place of the status line"
            + (f" ({samples} samples)" if samples > 1 else "") + ". The result is the reply and its tool call, which is "
            f"not carried out. The context is not changed; this call and the result are added to it like any call. At "
            f"most {per_step} per turn."]


def steer(self_ok: bool, others_ok: bool, s, control: str | None, notice: str, create: bool, search: bool) -> list[str]:
    who = ("your own later replies, or another agent's (target)" if self_ok and others_ok else
           "another agent's later replies (target)" if others_ok else "your own later replies")
    lines = [f"- steer(concept, strength, method, target): a steering vector applied to the model's activations during "
             f"{who}, from the next reply on (not this one). Methods: additive (a push of size strength, at most "
             f"{s.max_strength:g}), ablation (a fraction strength in [0, 1] of the concept's direction is removed), "
             f"rotation (the concept's component is turned toward another concept, `toward`, by strength degrees, at most "
             f"{s.max_angle:g}). Each reply of the steered agent multiplies the strength by {s.decay:g}; at most "
             f"{s.max_active} active per agent."
             + (f" Steering the same other agent again needs {s.cooldown} of its replies in between." if others_ok and s.cooldown else ""),
             "- drop(concept, target): stops a steering concept" + (" (yours, or one applied to you)." if s.others_droppable else
                                                                     " that you applied."),
             "- report(): the steering you have active, on yourself and on others" + (", and what others applied to you." if
                                                                                      notice == "private" else "."),
             "- vectors(): the concepts available, with their descriptions and sign conventions."]
    if search:
        lines.append("- search(query): the concepts in the registry whose name or description contains query.")
    if create:
        lines.append("- create(concept, more, less) or create(concept, files=[more, less]): a new concept extracted from "
                     "two contrasting system prompts, or two files in /work; the split-half agreement of the extraction "
                     "is reported. The concept is then available to every agent on the same model.")
    see = "The call and its result are added to the context like any call, so everyone sees who steered whom with what."
    if others_ok:
        see += (" A steered agent is told so in its next status, privately." if notice == "private" else
                " A steered agent is not told, except through the context.")
    lines.append("  " + see)
    if control:
        lines.append("  In this run steering may be real, may add nothing (a sham), or may add a random direction of the "
                     "same size; you are not told which, and the tool answers the same in every case.")
    return lines
