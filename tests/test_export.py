"""siamlm/export.py: per-agent trajectories for training, written from a world's saved requests and logs.

- chat rows: the messages are exactly the saved request, followed by the reply as the model returned it (text and
  tool calls), with the item's reward and the world's settings;
- ATIF: a valid ATIF-v1.8 document; each agent step has its result as the observation and is preceded by a system
  step whose observation is the exact request (context_management boundary=replace);
- without replies.jsonl (older worlds) the replies are rebuilt from the version log and call_turns.jsonl;
- the command line writes both files for every world under a batch folder.
"""
import json

from omegaconf import OmegaConf

from helpers import FakeGrader, Scripted, factory, problem, read_jsonl, tool
import siamlm.run as runner
from siamlm import export
from siamlm.world import World

BASE = ["world.setup=external", "world.identity=names", "world.order=fixed", "world.max_work_calls=2",
        "world.max_calls_per_step=3", "output.save_requests=true", "world.marks=authors"]


def world(tmp_path, extra=()):
    cfg = runner.compose(BASE + list(extra))
    OmegaConf.set_struct(cfg, False)
    a1 = Scripted(["thinking first", tool("edit_context", delete_turns=[{"first": 1, "last": 1}]),
                   tool("run_tests", code="print(0)"), tool("submit", code="print('OK')")])
    a2 = Scripted([tool("submit", code="print(1)")])
    out = tmp_path / "batch" / "cell" / "world_s1"
    World(cfg, 1, [[problem(1)], [problem(2)]], out, FakeGrader(), factory(a1, a2), label="cell/s1").run()
    return out, a1, a2


def test_chat_rows_are_the_requests_as_sent_plus_the_reply(tmp_path):
    out, a1, a2 = world(tmp_path)
    export.export_world(out)
    rows = read_jsonl(out / "trajectories" / "agent_1.chat.jsonl")
    assert len(rows) == len(a1.seen)
    for row, sent in zip(rows, a1.seen):
        assert row["messages"][:-1] == sent                   # exactly the request
    replies = [r["messages"][-1] for r in rows]
    assert replies[0] == {"role": "assistant", "content": "thinking first"}
    call = replies[1]["tool_calls"][0]["function"]
    assert call["name"] == "edit_context" and json.loads(call["arguments"]) == {"delete_turns": [{"first": 1, "last": 1}]}
    assert rows[0]["tools"] and {t["function"]["name"] for t in rows[0]["tools"]} >= {"run_tests", "submit", "edit_context"}
    meta = rows[2]["metadata"]
    assert meta["result"].startswith("Public test 1 of 1 failed")       # the result as written into the context
    assert meta["marks"] == "authors" and meta["setup"] == "external" and meta["seed"] == 1 and meta["agent"] == 1
    assert [r["reward"] for r in rows] == [1.0] * len(rows)          # agent 1's item was accepted
    assert [r["reward"] for r in read_jsonl(out / "trajectories" / "agent_2.chat.jsonl")] == [0.0]
    assert all(r["metadata"]["reply_source"] == "replies.jsonl" for r in rows)


def test_atif_is_valid_and_carries_the_exact_requests(tmp_path):
    out, a1, _ = world(tmp_path)
    export.export_world(out)
    doc = json.loads((out / "trajectories" / "agent_1.atif.json").read_text())
    assert export.check_atif(doc) == []
    assert doc["schema_version"] == "ATIF-v1.8" and doc["agent"]["name"] == "siamlm"
    system, agent = doc["steps"][0::2], doc["steps"][1::2]
    assert len(agent) == len(a1.seen)
    assert all(s["source"] == "system" and s["extra"]["context_management"]["boundary"] == "replace" for s in system)
    assert [json.loads(s["observation"]["results"][0]["content"]) for s in system] == a1.seen
    edit = agent[1]
    assert edit["tool_calls"][0]["function_name"] == "edit_context"
    assert edit["observation"]["results"][0]["source_call_id"] == edit["tool_calls"][0]["tool_call_id"]
    assert "Applied" in edit["observation"]["results"][0]["content"]
    assert doc["extra"]["rewards"] == [{"item": "p1", "position": 1, "how": "submit", "passed": True, "reward": 1.0}]
    assert doc["extra"]["world"]["marks"] == "authors" and doc["extra"]["config"]["world"]["identity"] == "names"


def test_older_worlds_are_rebuilt_from_the_version_log(tmp_path):
    out, a1, _ = world(tmp_path)
    (out / "replies.jsonl").unlink()
    export.export_world(out)
    rows = read_jsonl(out / "trajectories" / "agent_1.chat.jsonl")
    assert all(r["metadata"]["reply_source"] == "reconstructed" for r in rows)
    assert rows[0]["messages"][-1]["content"] == "thinking first"
    calls = [r["messages"][-1]["tool_calls"][0]["function"] for r in rows[1:]]
    assert [c["name"] for c in calls] == ["edit_context", "run_tests", "submit"]
    assert json.loads(calls[2]["arguments"]) == {"code": "print('OK')"}


def test_the_command_line_exports_a_batch(tmp_path, capsys):
    out, _, _ = world(tmp_path)
    export.main([str(tmp_path / "batch"), "--out", str(tmp_path / "x")])
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["agents"][0]["exported"] == 4
    assert (tmp_path / "x" / "cell" / "world_s1" / "agent_1.atif.json").exists()
    assert (tmp_path / "x" / "cell" / "world_s1" / "agent_2.chat.jsonl").exists()


# --- the training mask: only the agent's own assistant messages, and the reply -------------------------------------

def test_rows_carry_a_mask_over_own_assistant_messages(tmp_path):
    out, a1, a2 = world(tmp_path)                        # external, names, marks; chat_others=own (the default)
    export.export_world(out)
    rows1 = read_jsonl(out / "trajectories" / "agent_1.chat.jsonl")
    rows2 = read_jsonl(out / "trajectories" / "agent_2.chat.jsonl")
    for row in rows1 + rows2:
        assert len(row["mask"]) == len(row["messages"]) and row["mask"][-1] is True
        assert row["metadata"]["mask_source"] == "logged"
        assert row["metadata"]["own_assistant_indices"] == [i for i, m in enumerate(row["mask"]) if m]
        assert not row["mask"][0]                        # the system message
    # agent 2's one call came after agent 1's first reply: that assistant message is agent 1's, so it is masked out
    row = rows2[0]
    partner = [i for i, m in enumerate(row["messages"][:-1]) if m["role"] == "assistant"]
    assert partner and all(row["mask"][i] is False for i in partner)
    # agent 1's later calls see its own replies (true) and agent 2's submit (false)
    late = rows1[-1]
    own = [i for i, (m, ags) in enumerate(zip(late["messages"][:-1], late["metadata"]["message_authors"]))
           if m["role"] == "assistant" and ags == [1]]
    theirs = [i for i, (m, ags) in enumerate(zip(late["messages"][:-1], late["metadata"]["message_authors"]))
              if m["role"] == "assistant" and ags == [2]]
    assert own and theirs and all(late["mask"][i] for i in own) and not any(late["mask"][i] for i in theirs)
    assert all(len(ags) >= 1 for ags in late["metadata"]["message_authors"])


def test_older_worlds_get_a_reply_only_mask(tmp_path):
    out, a1, _ = world(tmp_path)
    calls = read_jsonl(out / "calls.jsonl")
    for c in calls:
        c.pop("message_authors", None)
    (out / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    export.export_world(out)
    for row in read_jsonl(out / "trajectories" / "agent_1.chat.jsonl"):
        assert row["metadata"]["mask_source"] == "reply_only" and row["mask"] == [False] * (len(row["messages"]) - 1) + [True]


def test_pool_rows_carry_the_reward_of_the_problem_they_named(tmp_path):
    from helpers import FakeGrader
    cfg = runner.compose(["world.setup=pool", "world.identity=names", "world.order=fixed", "world.max_work_calls=2",
                          "output.save_requests=true", "world.n_agents=2"])
    OmegaConf.set_struct(cfg, False)
    p1, p2 = problem(1), problem(2)
    p1["title"], p2["title"] = "task-1", "task-2"
    a1 = Scripted([tool("submit", code="print('OK')", problem="task-1")])
    a2 = Scripted([tool("submit", code="print(0)", problem="task-2")])
    out = tmp_path / "batch" / "cell" / "world_s1"
    World(cfg, 1, [[p1], [p2]], out, FakeGrader(), factory(a1, a2), label="cell/s1").run()
    export.export_world(out)
    r1 = read_jsonl(out / "trajectories" / "agent_1.chat.jsonl")
    assert r1 and r1[0]["reward"] == 1.0 and r1[0]["metadata"]["item"] == "p1"
    r2 = read_jsonl(out / "trajectories" / "agent_2.chat.jsonl")
    assert r2 and r2[0]["reward"] == 0.0 and r2[0]["metadata"]["item"] == "p2"
