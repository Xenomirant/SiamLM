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
