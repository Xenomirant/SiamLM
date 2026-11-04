"""End to end through the command line entry point, with the mock provider (no network, no key, no cost)."""
import json
import shutil
import subprocess

import pytest

from siamlm import run as runner
from siamlm import sweep


def test_cli_runs_an_open_task_world_with_the_mock(tmp_path, capsys):
    runner.main(["task.name=open", "world.setup=external", "world.identity=names", "world.chat_others=user",
                 "world.alter_notice=true", "world.rounds=4", "ctx.count=turns", "world.seeds=[1,2]", "parallel=2",
                 f"output.directory={tmp_path}", "output.batch=smoke"])
    worlds = sorted((tmp_path / "siamlm" / "smoke").glob("*/world_s*/world.json"))
    assert len(worlds) == 2
    for w in worlds:
        doc = json.loads(w.read_text())
        assert doc["status"] == "complete" and doc["rounds"] == 4 and doc["task"] == "open"
        assert (w.parent / "calls.jsonl").exists() and (w.parent / "context.txt").exists()
    assert "-> " in capsys.readouterr().out
    runner.main(["task.name=open", "world.setup=external", "world.identity=names", "world.chat_others=user",
                 "world.alter_notice=true", "world.rounds=4", "ctx.count=turns", "world.seeds=[1,2]",
                 f"output.directory={tmp_path}", "output.batch=smoke"])
    assert "skipped (complete)" in capsys.readouterr().out          # a finished world is not run again


def sandbox_image():
    image = str(runner.compose().ctx.bash.image)
    return bool(shutil.which("docker")) and subprocess.run(["docker", "image", "inspect", image],
                                                           capture_output=True).returncode == 0


@pytest.mark.skipif(not sandbox_image(), reason="needs Docker and the sandbox image (docker/sandbox.Dockerfile)")
def test_cli_runs_a_state_chain_world_with_bash_and_the_mock(tmp_path):
    runner.main(["task.name=state_chain", "ctx.edit=bash", "world.setup=external", "world.identity=names",
                 "world.marks=authors", "world.max_work_calls=1", "world.after_stream=stay", "world.rounds=14",
                 "world.evict=false", "world.end_over_limit=3", "ctx.count=turns", f"output.directory={tmp_path}",
                 "output.batch=chain"])
    w = next((tmp_path / "siamlm" / "chain").glob("*/world_s1"))
    doc = json.loads((w / "world.json").read_text())
    assert doc["status"] == "complete" and doc["task"] == "state_chain"
    assert len([l for l in (w / "results.jsonl").open()]) == 24          # 2 agents x 12 items, every one closed
    assert json.loads((w / "manifest.json").read_text())["task"] == "state_chain"
    assert any(json.loads(l).get("interface") == "bash" for l in (w / "edits.jsonl").open())


@pytest.mark.parametrize("design", sweep.DESIGNS)
def test_every_sweep_cell_composes(design, tmp_path):
    shares = tmp_path / "shares.json"
    shares.write_text(json.dumps({"h": 9000, "l": 9000, "g": 9000, "q": 9000}))
    a = sweep.parse_args(["--design", design, "--batch", "b", "--shares-file", str(shares)]
                         + (["--rounds", "80"] if design == "s3" else []))
    cells = sweep.study(a)
    assert cells and len({name for name, _ in cells}) == len(cells)
    for name, over in cells:
        cfg = runner.compose(["provider=openrouter", "ctx.edit=bash", *over])
        assert cfg.world.setup == "external" and cfg.ctx.count == "turns" and cfg.world.end_over_limit == 3, name


def test_the_study_preset_composes_and_the_default_is_warned_about(tmp_path, capsys):
    from omegaconf import OmegaConf
    from helpers import FakeGrader, Scripted, factory, problem
    from siamlm.world import World
    cfg = runner.compose(["+design=study"])
    assert (cfg.world.evict, cfg.world.end_over_limit, cfg.ctx.count, cfg.world.setup) == (False, 3, "turns", "external")
    assert runner.compose([]).world.evict is True
    for overrides, warned in ((["world.n_agents=1"], True), (["world.n_agents=1", "world.evict=false"], False)):
        cfg = runner.compose(overrides)
        OmegaConf.set_struct(cfg, False)
        out = tmp_path / ("warned" if warned else "quiet")
        World(cfg, 1, [[problem(1)]], out, FakeGrader(), factory(Scripted()), label="t").run()
        doc = json.loads((out / "world.json").read_text())
        assert bool(doc["warnings"]) is warned
        assert ("world.evict=true" in capsys.readouterr().err) is warned
