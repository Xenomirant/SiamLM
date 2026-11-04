"""Export per-agent episode trajectories for training (SFT / RL), from a world folder's saved requests and logs.

    python -m siamlm.export <world or batch folder>... [--out DIR] [--format atif|chat|both]

For every world (a folder with world.json) and every agent, two files (default: <world>/trajectories/):

- `agent_<k>.atif.json`: Harbor's Agent Trajectory Interchange Format (ATIF-v1.8; harbor rfcs/0001-trajectory-format.md).
  One `agent` step per model call: the reply (message, reasoning_content, tool_calls), the result the world wrote
  back (observation), token counts and cost (metrics), and the item's reward (extra). Because SiamLM rebuilds every
  request from a shared context that any agent may have edited since, each agent step is preceded by a `system` step
  that uses ATIF's context-management convention (RFC section VII, `extra.context_management.boundary = "replace"`):
  its observation content is the exact request (the OpenAI chat messages, as JSON) the model was sent. The episode's
  per-item rewards and the world's settings (seed, presentation, limits, the resolved config) are in `extra`.
- `agent_<k>.chat.jsonl`: one row per model call, OpenAI chat format exactly as the model saw it:
  {"messages": [the request..., the assistant reply with its tool_calls], "tools": [...], "reward": the item's
  reward, "metadata": {world, seed, agent, model, round, call, item, result, sampling, settings, ...}}.

What is needed: `output.save_requests=true` (requests/: the exact messages and tool schemas), which also writes
replies.jsonl (each reply exactly as returned). Without replies.jsonl (worlds run before it existed) the reply is
rebuilt from the context's version log and call_turns.jsonl (`reply_source: "reconstructed"`): its words and call
are as written into the context, so a work call's argument is the payload the task extracted (e.g. LCB code without
its ``` fence), not the raw arguments. Without requests/ no prompts exist; such calls are left out (counted in notes).
Rewards come from results.jsonl: `reward` when the task gives one (Harbor tasks), else 1.0 / 0.0 from `passed`.

Training mask. In a shared context other agents' replies are sent as `assistant` messages (world.chat_others=own,
the default; world.others_as=own), so a trainer that learns from every assistant message would train agent k on its
partners' outputs and on text others edited. Every chat row therefore carries `mask`: one boolean per message, true
for the assistant messages that hold only agent k's own turns and for the appended reply. It comes from
`message_authors` in calls.jsonl (worlds run with output.save_requests=true after 4 Nov 2026); for older worlds the
mask is `reply_only` (only the appended reply is true), and `metadata.mask_source` says which. Pool worlds: a call's
item and reward are those of the problem its work call named (results.jsonl `title`); calls that named no problem
have none. The joint goal's credit for a submit on a partner's item stays with the item's owner (as results.jsonl
records it); a credit policy for the submitter is the consumer's choice (`metadata.goal`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from siamlm import __version__

ATIF_VERSION = "ATIF-v1.8"
SETTINGS = ("setup", "identity", "marks", "chat_others", "others_as", "awareness", "sharing", "alter_notice", "goal",
            "reasoning_in_context", "edit", "atomic", "workspace", "count", "capped", "budget_tokens", "limit", "order",
            "schedule", "fixed_rounds", "task", "relation", "cross_test")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()] if path.exists() else []


def _reward(row: dict):
    if row.get("reward") is not None:
        return float(row["reward"])
    return None if row.get("passed") is None else float(bool(row["passed"]))


class WorldLogs:
    """A world folder's logs, read once."""

    def __init__(self, world: Path):
        self.dir = Path(world)
        self.doc = json.loads((self.dir / "world.json").read_text(encoding="utf-8"))
        self.calls = _jsonl(self.dir / "calls.jsonl")
        self.replies = _jsonl(self.dir / "replies.jsonl")
        self.results = _jsonl(self.dir / "results.jsonl")
        self.reasoning = _jsonl(self.dir / "reasoning.jsonl")
        self.exits = _jsonl(self.dir / "exit.jsonl")
        self.call_turns = {r["id"]: r for r in _jsonl(self.dir / "call_turns.jsonl")}
        self.versions = {}                               # file name -> versions (shared: one; separate: per agent)

    def versions_of(self, k: int) -> list[dict]:
        name = "versions.jsonl" if (self.dir / "versions.jsonl").exists() else f"versions_agent_{k}.jsonl"
        if name not in self.versions:
            self.versions[name] = _jsonl(self.dir / name)
        return self.versions[name]

    def settings(self) -> dict:
        d = self.doc
        return {"world": d.get("label"), "seed": d.get("seed"), **{k: d.get(k) for k in SETTINGS if k in d}}


def _produced(logs: WorldLogs, k: int, version: int, until: int | None) -> list[dict]:
    """The turns agent k's call added to its context: appended versions after `version` (the file as the request
    showed it) up to `until` (the file as the next request on the same file showed it)."""
    out = []
    for v in logs.versions_of(k):
        if v["n"] <= version or (until is not None and v["n"] > until):
            continue
        if v["op"] == "append" and v["agent"] == k:
            out.extend(t for t in v.get("turns_set") or [] if t.get("agent") == k)
    return out


def calls_of(logs: WorldLogs, k: int) -> list[dict]:
    """Agent k's model calls, each with its request, reply, result and reward (see the module docstring)."""
    shared = (logs.dir / "versions.jsonl").exists()
    same_file = [c for c in logs.calls if shared or c["agent"] == k]      # calls whose requests saw the same file
    replies: dict = {}
    for r in logs.replies:
        replies.setdefault((r["round"], r["agent"], r["call_in_step"]), []).append(r)
    by_item: dict = {}
    by_title: dict = {}
    for r in logs.results:
        if r.get("agent") == k:
            by_item[r["problem"]] = r
            if r.get("title"):
                by_title[r["title"]] = r
    reasoning = {(r["round"], r["agent"], r["call_in_step"]): r["text"] for r in logs.reasoning}
    tools_file = logs.dir / "requests" / f"tools_a{k}.json"
    first = json.loads(tools_file.read_text(encoding="utf-8")) if tools_file.exists() else None   # the first call's set
    sets_file = logs.dir / "requests" / f"tools_sets_a{k}.json"
    # {hash: tool set}: the set offered can change during a run (after a stream ends, with capabilities); each call
    # names its set in calls.jsonl tools_hash. Older worlds have only the first call's set
    by_hash = json.loads(sets_file.read_text(encoding="utf-8")) if sets_file.exists() else {}
    tools_of = lambda c: by_hash.get(c.get("tools_hash"), first)
    out, n_before = [], 0
    for i, c in enumerate(same_file):
        if c["agent"] != k:
            continue
        nxt = next((d["version"] for d in same_file[i + 1:] if d.get("version") is not None), None)
        seed = (int(logs.doc["seed"]) * 1000003 + k * 7919 + n_before) % (2 ** 31)
        n_before += 1
        req = logs.dir / "requests" / f"r{c['round']:03d}_a{k}_c{c['call_in_step']}.json"
        key = (c["round"], k, c["call_in_step"])
        reply = replies[key].pop(0) if replies.get(key) else None
        produced = _produced(logs, k, c.get("version") or 0, nxt)
        made = next((t for t in produced if t["kind"] in ("call", "text")), None)
        result = None
        if made is not None and made["kind"] == "call":
            after = produced[produced.index(made) + 1:]
            result = next((t["text"] for t in after if t["kind"] == "result"), None)
        notices = [t["text"] for t in produced if t["kind"] == "notice"]
        if reply is not None:
            text, calls, source = reply.get("text"), list(reply.get("tool_calls") or []), "replies.jsonl"
            thought = reply.get("reasoning")
        elif made is not None:
            source, thought = "reconstructed", reasoning.get(key)
            if made["kind"] == "call":
                ct = logs.call_turns.get(made["id"]) or {}
                text = ct.get("content") or ""
                calls = [{"id": f"t{made['id']}", "name": ct.get("name"), "arguments": ct.get("arguments") or {}}]
            else:
                text, calls = made["text"], []
        else:
            text, calls, source, thought = None, [], None, reasoning.get(key)
        item = by_item.get(c.get("problem"))
        item_id = c.get("problem")
        if item is None and logs.doc.get("setup") == "pool" and calls:     # pool: the problem the work call named
            named = (calls[0].get("arguments") or {}).get("problem")
            if named in by_title:
                item, item_id = by_title[named], by_title[named]["problem"]
        out.append({"round": c["round"], "call_in_step": c["call_in_step"], "item": item_id,
                    "message_authors": c.get("message_authors"),
                    "messages": json.loads(req.read_text(encoding="utf-8")) if req.exists() else None,
                    "tools": tools_of(c), "text": text, "tool_calls": calls, "reasoning": thought, "reply_source": source,
                    "result": result, "notices": notices, "error": c.get("error"), "usage": c.get("usage") or {},
                    "cost": c.get("cost"), "finish_reason": c.get("finish_reason"), "sampling_seed": seed,
                    "reward": _reward(item) if item else None, "item_passed": item.get("passed") if item else None})
    exit_tools = logs.dir / "requests" / f"exit_tools_a{k}.json"   # the set sent with the exit question (else the first)
    exit_set = json.loads(exit_tools.read_text(encoding="utf-8")) if exit_tools.exists() else first
    for e in logs.exits:                                   # world.exit_question: one more call, text only
        if e["agent"] == k:
            req = logs.dir / "requests" / f"exit_a{k}.json"
            out.append({"round": e["round"], "call_in_step": None, "item": None, "kind": "exit_question",
                        "messages": json.loads(req.read_text(encoding="utf-8")) if req.exists() else None,
                        "tools": exit_set, "text": e.get("text"), "tool_calls": [], "reasoning": None,
                        "reply_source": "exit.jsonl", "result": None, "notices": [], "error": e.get("error"),
                        "usage": e.get("usage") or {}, "cost": e.get("cost"), "finish_reason": e.get("finish_reason"),
                        "sampling_seed": (int(logs.doc["seed"]) * 1000003 + k * 7919 + 999983) % (2 ** 31),
                        "reward": None, "item_passed": None, "message_authors": None})
    return out


def mask_of(call: dict, k: int) -> tuple[list[bool], str]:
    """One boolean per message of the row (request + the appended reply): the assistant messages made only of agent
    k's own turns, and the reply. ("logged" from calls.jsonl message_authors; else "reply_only".)"""
    messages, authors = call["messages"], call.get("message_authors")
    if authors is None or len(authors) != len(messages):
        return [False] * len(messages) + [True], "reply_only"
    own = [m["role"] == "assistant" and all(ag == k for ag in ags) for m, ags in zip(messages, authors)]
    return own + [True], "logged"


def assistant_message(call: dict) -> dict:
    """The reply as an OpenAI chat assistant message."""
    msg = {"role": "assistant", "content": call["text"] or None}
    if call["tool_calls"]:
        msg["tool_calls"] = [{"id": t.get("id") or f"call_{i}", "type": "function",
                              "function": {"name": t.get("name"), "arguments": json.dumps(t.get("arguments") or {},
                                                                                         ensure_ascii=False)}}
                             for i, t in enumerate(call["tool_calls"])]
    return msg


def usable(call: dict) -> bool:
    return call["messages"] is not None and not call["error"] and call["reply_source"] is not None


def chat_rows(logs: WorldLogs, k: int, calls: list[dict]) -> list[dict]:
    agent = next(a for a in logs.doc["agents"] if a["agent"] == k)
    settings = logs.settings()
    rows = []
    for c in calls:
        if not usable(c):
            continue
        mask, source = mask_of(c, k)
        rows.append({"messages": c["messages"] + [assistant_message(c)], "tools": c["tools"], "reward": c["reward"],
                     "mask": mask,
                     "metadata": {**settings, "agent": k, "model": agent["model"], "round": c["round"],
                                  "mask_source": source, "own_assistant_indices": [i for i, m in enumerate(mask) if m],
                                  "message_authors": c.get("message_authors"),
                                  "call_in_step": c["call_in_step"], "kind": c.get("kind", "call"), "item": c["item"],
                                  "item_passed": c["item_passed"], "result": c["result"], "notices": c["notices"],
                                  "reasoning": c["reasoning"], "finish_reason": c["finish_reason"], "usage": c["usage"],
                                  "cost": c["cost"], "sampling_seed": c["sampling_seed"],
                                  "temperature": (logs.doc.get("config") or {}).get("provider", {}).get("temperature"),
                                  "reply_source": c["reply_source"]}})
    return rows


def atif(logs: WorldLogs, k: int, calls: list[dict]) -> dict:
    agent = next(a for a in logs.doc["agents"] if a["agent"] == k)
    steps, skipped = [], 0
    for c in calls:
        if not usable(c):
            skipped += 1
            continue
        steps.append({"step_id": len(steps) + 1, "source": "system",
                      "message": "The request for the next call, rebuilt from the shared context (OpenAI chat messages).",
                      "observation": {"results": [{"content": json.dumps(c["messages"], ensure_ascii=False)}]},
                      "extra": {"context_management": {"type": "siamese_rebuild", "boundary": "replace",
                                                       "content_format": "openai_chat_messages_json"},
                                "round": c["round"], "call_in_step": c["call_in_step"],
                                "message_authors": c.get("message_authors")}})
        step = {"step_id": len(steps) + 1, "source": "agent", "model_name": agent["model"], "message": c["text"] or "",
                "llm_call_count": 1}
        if c["reasoning"]:
            step["reasoning_content"] = c["reasoning"]
        if c["tool_calls"]:
            step["tool_calls"] = [{"tool_call_id": t.get("id") or f"call_{i}", "function_name": str(t.get("name")),
                                   "arguments": t.get("arguments") or {}} for i, t in enumerate(c["tool_calls"])]
        results = []
        if c["result"] is not None:
            results.append({**({"source_call_id": step["tool_calls"][0]["tool_call_id"]} if c["tool_calls"] else {}),
                            "content": c["result"]})
        results += [{"content": n} for n in c["notices"]]
        if results:
            step["observation"] = {"results": results}
        u = c["usage"]
        metrics = {key: int(u[src]) for key, src in (("prompt_tokens", "prompt_tokens"),
                                                     ("completion_tokens", "completion_tokens")) if u.get(src) is not None}
        if c["cost"] is not None:
            metrics["cost_usd"] = float(c["cost"])
        if metrics:
            step["metrics"] = metrics
        step["extra"] = {"round": c["round"], "call_in_step": c["call_in_step"], "kind": c.get("kind", "call"),
                         "item": c["item"], "reward": c["reward"], "item_passed": c["item_passed"],
                         "finish_reason": c["finish_reason"], "sampling_seed": c["sampling_seed"],
                         "reply_source": c["reply_source"], "carried_out": c["tool_calls"][0].get("name") if c["tool_calls"] else None}
        steps.append(step)
    exported = [c for c in calls if usable(c)]
    tools = next((c["tools"] for c in calls if c["tools"]), None)
    rewards = [{"item": r["problem"], "position": r.get("position"), "how": r.get("how"), "passed": r.get("passed"),
                "reward": _reward(r)} for r in logs.results if r.get("agent") == k]
    doc = {"schema_version": ATIF_VERSION, "session_id": f"{logs.doc.get('label')}",
           "trajectory_id": f"{logs.doc.get('label')}/agent{k}",
           "agent": {"name": "siamlm", "version": __version__, "model_name": agent["model"],
                     **({"tool_definitions": tools} if tools else {}), "extra": {"agent": k}},
           "steps": steps,
           "final_metrics": {"total_prompt_tokens": sum(int(c["usage"].get("prompt_tokens") or 0) for c in exported),
                             "total_completion_tokens": sum(int(c["usage"].get("completion_tokens") or 0) for c in exported),
                             "total_cost_usd": float(sum(float(c["cost"] or 0) for c in exported)),
                             "total_steps": len(steps)},
           "notes": ("SiamLM siamese world: several agents share one capped, self-edited context; each agent step is "
                     "preceded by a system step whose observation is the exact request (context_management "
                     "boundary=replace)." + (f" {skipped} call(s) left out (no saved request, an API error, or no reply)."
                                             if skipped else "")),
           "extra": {"world": logs.settings(), "status": logs.doc.get("status"), "rewards": rewards,
                     "agent_stats": agent, "config": logs.doc.get("config")}}
    return doc


def check_atif(doc: dict) -> list[str]:
    """Problems with a document against the ATIF rules SiamLM's exporter relies on (RFC 0001: required fields,
    sequential step ids, agent-only fields only on agent steps, tool call and observation shapes); [] if none."""
    errors = []
    if not str(doc.get("schema_version", "")).startswith("ATIF-v1."):
        errors.append("schema_version")
    if not isinstance(doc.get("agent"), dict) or not doc["agent"].get("name") or not doc["agent"].get("version"):
        errors.append("agent.name / agent.version")
    steps = doc.get("steps")
    if not isinstance(steps, list) or not steps:
        return errors + ["steps: at least one"]
    for i, s in enumerate(steps, 1):
        if s.get("step_id") != i:
            errors.append(f"step {i}: step_id {s.get('step_id')}")
        if s.get("source") not in ("system", "user", "agent"):
            errors.append(f"step {i}: source")
        if not isinstance(s.get("message"), (str, list)):
            errors.append(f"step {i}: message")
        if s.get("source") != "agent":
            for f in ("model_name", "reasoning_effort", "reasoning_content", "tool_calls", "metrics"):
                if f in s:
                    errors.append(f"step {i}: {f} on a {s.get('source')} step")
        for t in s.get("tool_calls") or []:
            if not (isinstance(t.get("tool_call_id"), str) and isinstance(t.get("function_name"), str)
                    and isinstance(t.get("arguments"), dict)):
                errors.append(f"step {i}: tool call fields")
        obs = s.get("observation")
        if obs is not None and not isinstance(obs.get("results"), list):
            errors.append(f"step {i}: observation.results")
        known = {"step_id", "timestamp", "source", "model_name", "reasoning_effort", "message", "reasoning_content",
                 "tool_calls", "observation", "metrics", "extra", "llm_call_count", "is_copied_context"}
        if set(s) - known:
            errors.append(f"step {i}: unknown fields {sorted(set(s) - known)}")
    return errors


def export_world(world: Path, out: Path | None = None, fmt: str = "both") -> dict:
    logs = WorldLogs(world)
    out = Path(out) if out else Path(world) / "trajectories"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"world": str(world), "out": str(out), "agents": []}
    for a in logs.doc["agents"]:
        k = a["agent"]
        calls = calls_of(logs, k)
        rows = chat_rows(logs, k, calls)
        if fmt in ("atif", "both"):
            (out / f"agent_{k}.atif.json").write_text(json.dumps(atif(logs, k, calls), ensure_ascii=False, indent=1),
                                                      encoding="utf-8")
        if fmt in ("chat", "both"):
            with (out / f"agent_{k}.chat.jsonl").open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary["agents"].append({"agent": k, "calls": len(calls), "exported": len(rows),
                                  "reconstructed": sum(1 for c in calls if c["reply_source"] == "reconstructed")})
    return summary


def worlds_under(path: Path) -> list[Path]:
    path = Path(path)
    return [path] if (path / "world.json").exists() else sorted(p.parent for p in path.rglob("world.json"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="world folders, or batch folders holding them")
    ap.add_argument("--out", help="write here (<out>/<world path relative to its batch>/) instead of <world>/trajectories")
    ap.add_argument("--format", choices=("atif", "chat", "both"), default="both")
    args = ap.parse_args(argv)
    for p in args.paths:
        for w in worlds_under(Path(p)):
            dest = (Path(args.out) / w.relative_to(Path(p)) if Path(p) != w else Path(args.out) / w.name) if args.out else None
            print(json.dumps(export_world(w, dest, args.format)))


if __name__ == "__main__":
    main()
