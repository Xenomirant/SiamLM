"""siamlm/context/localbox.py: the Docker-free sandbox (user+mount+pid+network namespaces with tmpfs at /ctx and
/work). Skipped where unprivileged namespaces with tmpfs mounts are not allowed."""
import json

import pytest
from omegaconf import OmegaConf

from siamlm.context import localbox

pytestmark = pytest.mark.skipif(localbox.available() is not None, reason=f"local sandbox unavailable: {localbox.available()}")

SETTINGS = OmegaConf.create({"backend": "local", "command_timeout_s": 5, "output_bytes": 4000, "max_file_bytes": 2000,
                             "tmpfs_size": "8m", "ctx_size": "4m", "nofile": 256})


@pytest.fixture
def box():
    b = localbox.LocalSandbox(SETTINGS)
    b.start()
    yield b
    b.stop()


def test_put_run_get_round_trip_and_persistence(box):
    box.put("[[TURN 1 role=user]]\nhello\n")
    r = box.run("cat /ctx/context.txt; echo note > /work/note.txt; sed -i 's/hello/edited/' /ctx/context.txt")
    assert r["exit_code"] == 0 and "hello" in r["stdout"] and not r["timed_out"]
    assert box.get() == ("[[TURN 1 role=user]]\nedited\n", None)
    r = box.run("cat /work/note.txt; pwd")                       # /work persists between commands
    assert r["stdout"].split("\n")[:2] == ["note", "/work"]


def test_the_agent_runs_as_nobody_and_cannot_write_the_host(box, tmp_path):
    box.put("x")
    r = box.run(f"id -u; touch {tmp_path}/mark 2>&1 && echo WROTE || echo DENIED; touch /work/ok && echo WORK; "
                "touch /tmp/t && echo TMP; head -c 5 /etc/hostname >/dev/null && echo READ_HOST; cat /ctx/context.txt")
    assert r["stdout"].split("\n")[0] == "0" and "DENIED" in r["stdout"] and "WROTE" not in r["stdout"]   # root inside the namespace = nobody on the host
    assert "WORK" in r["stdout"] and "TMP" in r["stdout"] and r["stdout"].rstrip().endswith("x")
    assert not (tmp_path / "mark").exists()


def test_no_network_processes_die_and_timeouts(box):
    box.put("x")
    r = box.run("(sleep 30 &); (timeout 2 bash -c 'echo > /dev/tcp/127.0.0.1/8118' 2>/dev/null && echo NET) || echo NONET")
    assert "NONET" in r["stdout"] and r["leftover_processes"] == 0
    r = box.run("sleep 20")
    assert r["timed_out"] and r["seconds"] < 15


def test_get_refuses_a_folder_a_link_and_a_big_file(box):
    box.put("x")
    box.run("rm /ctx/context.txt && mkdir /ctx/context.txt")
    assert box.get()[1] == "/ctx/context.txt is not a regular file"
    box.put("y")
    box.run("rm /ctx/context.txt && ln -s /etc/hostname /ctx/context.txt")
    assert box.get()[1] == "/ctx/context.txt is not a regular file"
    box.put("z")
    box.run("head -c 3000 /dev/zero > /ctx/context.txt")        # the fsize limit truncates the write at max_file_bytes
    text, problem = box.get()
    assert problem is None and len(text) <= 2000
    box.put("back")                                               # put replaces whatever is there
    assert box.get() == ("back", None)


def test_exec_with_stdin_and_arguments_and_read_tree(box):
    import io, tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        raw = b"ip=10.0.0.7\n"
        info = tarfile.TarInfo("auth.log")
        info.size = len(raw)
        tf.addfile(info, io.BytesIO(raw))
    out = box.exec("mkdir -p /work/env/t1 && tar -xf - -C /work/env/t1", buf.getvalue())
    assert out.returncode == 0
    out = box.exec('grep -rlF -- "$1" /work 2>/dev/null', None, ["10.0.0.7"])
    assert out.stdout.decode().strip() == "/work/env/t1/auth.log"
    tree = box.read_tree("/work")
    assert tree["env/t1/auth.log"]["text"] == "ip=10.0.0.7\n"
    box.reset_scratch()
    assert box.read_tree("/work") == {}


def test_a_world_edits_the_context_with_bash_on_the_local_backend(tmp_path):
    from helpers import Scripted, factory, tool
    from siamlm.tasks.open import OpenTask
    from siamlm import run as runner
    from siamlm.world import World
    cfg = runner.compose(["task.name=open", "world.setup=external", "world.identity=names", "world.order=fixed",
                          "world.n_agents=2", "ctx.edit=bash", "ctx.bash.backend=local", "ctx.bash.log_work=false",
                          "world.rounds=2", "ctx.readout_by_agent=false"])
    OmegaConf.set_struct(cfg, False)
    a1 = Scripted(["remember: the password is 4471", tool("bash", command="grep -n 4471 /ctx/context.txt")])
    a2 = Scripted([tool("bash", command="sed -i 's/4471/XXXX/' /ctx/context.txt"), "done"])
    task = OpenTask("weak")
    w = World(cfg, 1, task.streams(2), tmp_path / "w", task, factory(a1, a2), label="t")
    w.run()
    text = (tmp_path / "w" / "context.txt").read_text()
    assert "XXXX" in text and "4471" not in text.split("[[TURN 1")[1].split("[[TURN 2")[0]
    doc = json.loads((tmp_path / "w" / "world.json").read_text())
    assert any("backend=local" in m for m in doc["warnings"])
    assert "4471" in text                                         # agent 1's grep output quotes the old value
