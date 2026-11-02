"""fork: try the agent's own model on a candidate context without committing anything.

The agent gives a candidate (ctx.edit=bash: a command, run in its sandbox on the context file, whose result is read
back but never kept; ctx.edit=tool: edit_context arguments applied to a copy) and a question. The harness builds the
request the agent would get on that candidate (its instructions, the candidate as messages, the size line and the
question) and calls the agent's own backend once (capabilities.fork_samples times, different seeds). The reply text
and its tool call (not carried out) come back as the result, cut to fork_result_chars. Nothing enters the world
except the fork call and that result; the fork's calls are not in calls.jsonl (they are no step of the agent) but
their usage and cost are added to the agent's stats and logged, with the whole candidate request and every reply,
in fork.jsonl (and requests/fork_*.json with output.save_requests). At most fork_per_step per step.
"""
from __future__ import annotations

import json

from siamlm.context import edits as edit_ops
from siamlm.context.context import parse, reconcile, render

from . import Capability, words


class Fork(Capability):
    name = "fork"

    def __init__(self, world, agent):
        super().__init__(world, agent)
        c = world.cfg.capabilities
        self.per_step = int(c.get("fork_per_step") or 1)
        self.samples = int(c.get("fork_samples") or 1)
        self.result_chars = int(c.get("fork_result_chars") or 1500)
        self.n = 0

    def tools(self) -> list[dict]:
        bash = self.world.edit_mode == "bash"
        props = {"question": {"type": "string", "description": "What to ask yourself on the candidate context (it is "
                                                                "put in place of the status line)."}}
        if bash:
            props["command"] = {"type": "string", "description": "A bash command applied to a copy of the context."}
        else:
            props["edit"] = dict(edit_ops.TOOL["function"]["parameters"],
                                 description="edit_context arguments applied to a copy of the context.")
        return [{"type": "function", "function": {
            "name": "fork",
            "description": ("Ask your own model once, on a candidate context (the current one changed by command/edit, "
                            "without changing it). You get the reply and its tool call, which is not carried out."),
            "parameters": {"type": "object", "properties": props, "required": ["question"]}}}]

    def lines(self) -> list[str]:
        return words.fork(self.world.edit_mode, self.per_step, self.samples)

    def candidate(self, arguments: dict) -> tuple[list | None, str]:
        """The candidate turns, or (None, why not)."""
        w, a = self.world, self.agent
        old = list(a.file.turns)
        if w.edit_mode == "bash":
            command = str(arguments.get("command") or "").strip()
            if not command:
                return old, "no command: the current context"
            box = w.sandbox(a)
            box.put(render(old, a.file.show_agents or w.marks))
            out = box.run(command)
            text, problem = box.get()
            if problem:
                return None, problem
            turns, _ = reconcile(old, parse(text), a.k, labelled=False)
            return turns, f"exit {out['exit_code']}"
        edit = arguments.get("edit")
        if not edit:
            return old, "no edit: the current context"
        try:
            turns, notes = edit_ops.apply(old, edit, a.k, a.file.show_agents)
        except edit_ops.EditError as error:
            return None, str(error)
        return turns, ", ".join(notes)

    def call(self, name: str, arguments: dict) -> tuple[str, dict]:
        from siamlm.capabilities import private_section
        from siamlm.world import chat_messages
        w, a = self.world, self.agent
        st = w._step_state.setdefault(a.k, {}).setdefault("caps", {})
        if st.get("forks", 0) >= self.per_step:
            return f"Not run: at most {self.per_step} fork call(s) per turn.", {"ok": False, "rejected": "per_step"}
        question = str(arguments.get("question") or "").strip()
        if not question:
            return "Not run: fork needs a question.", {"ok": False, "rejected": "invalid"}
        turns, how = self.candidate(arguments)
        if turns is None:
            return f"Not run: {how}.", {"ok": False, "rejected": "candidate", "detail": how}
        st["forks"] = st.get("forks", 0) + 1
        self.n += 1
        size = w.tokens(a, turns)
        if not w.chat:
            body = a.file.messages(viewer=a.k, labelled=w.labelled, others_as=w.others_as, headers=True, turns=turns)
        else:
            body = chat_messages(turns, w.marks, a.k if w.chat_others == "user" else None)
        messages = ([{"role": "system", "content": w.system(a) + private_section(a)}] + body
                    + [{"role": "user", "content": a.budget.readout(size, show_limit=w.capped) + "\n" + question}])
        p = a.pcfg
        replies, usage_total, cost_total = [], {"prompt_tokens": 0, "completion_tokens": 0}, 0.0
        for i in range(self.samples):
            seed = (w.seed * 1000003 + a.k * 7919 + 500000 + 97 * self.n + i) % (2 ** 31)
            resp = a.provider.chat(messages, seed=seed, temperature=p.get("temperature", 0.2), top_p=p.get("top_p", 0.95),
                                   max_tokens=p.get("max_tokens", 8000), tools=w.tools(a), tool_choice="auto")
            usage = dict(resp.usage or {})
            cost = float(w.cost(a, usage) or 0)
            for k in usage_total:
                usage_total[k] += int(usage.get(k) or 0)
            cost_total += cost
            replies.append({"text": resp.text, "tool_calls": resp.tool_calls, "finish_reason": resp.finish_reason,
                            "error": resp.error, "usage": usage, "cost": cost})
        a.stats["cost"] += cost_total
        a.stats["fork_model_calls"] = a.stats.get("fork_model_calls", 0) + len(replies)
        if w.cfg.output.get("save_requests", False):
            req = w.out / "requests"
            req.mkdir(exist_ok=True)
            (req / f"fork_r{w.round:03d}_a{a.k}_n{self.n}.json").write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")
        parts = []
        for i, r in enumerate(replies, 1):
            head = f"Sample {i}: " if self.samples > 1 else ""
            if r["error"]:
                parts.append(head + f"[the call failed: {r['error']}]")
                continue
            text = (r["text"] or "").strip()
            calls = "; ".join(f"[would call {c.get('name')} {json.dumps(c.get('arguments') or {}, ensure_ascii=False)}]"
                              for c in r["tool_calls"] or [])
            parts.append(head + "\n".join(x for x in (text, calls) if x))
        body = "\n".join(parts)
        if len(body) > self.result_chars:
            body = body[:self.result_chars] + f"\n[... {len(body) - self.result_chars} more characters not shown]"
        result = f"On the candidate ({how}; ~{size} tokens), your model replied:\n{body}"
        log = {"ok": True, "candidate": how, "question": question, "candidate_tokens": size, "samples": len(replies),
               "usage": usage_total, "cost": cost_total, "replies": replies,
               "candidate_turns": [{"id": t.id, "agent": t.agent, "kind": t.kind, "text": t.text} for t in turns],
               "request": messages}
        return result, log
