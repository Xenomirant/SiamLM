"""Harbor-format task directories as a SiamLM task family (`task.name=harbor`, `harbor.path=<dir>`).

Harbor (harborframework.com; github.com/laude-institute/harbor, Apache-2.0) defines a task as a directory
(docs/harbor.md has the details and sources):

    task.toml              [metadata], [verifier] timeout_sec / env / user, [agent] timeout_sec / user,
                           [environment] docker_image / workdir / cpus / memory_mb / build_timeout_sec / env
    instruction.md         what the agent is asked to do (leading canary comment lines are stripped, as Harbor does)
    environment/Dockerfile the container the agent works in (or [environment].docker_image)
    tests/test.sh          the verifier: run in the same container after the agent is done; it writes
                           /logs/verifier/reward.txt (one number) or /logs/verifier/reward.json (named numbers)
    solution/solve.sh      a reference solution (not used here)

Here every Harbor task is one item. While an agent works on it, the item has a container of its own, built from the
task's Dockerfile, and that agent's bash commands run in it (Task.sandbox), with SiamLM's context file mounted at
/ctx/context.txt as in any bash world: the agent edits its shared context there and works on the task in the same
shell. `submit` ends the item: tests/ is copied to /tests, /tests/test.sh runs (as Harbor runs it, output in
/logs/verifier/test-stdout.txt), and the reward it writes is the item's reward (results.jsonl `reward`, `rewards`);
the item counts as solved when the reward is at least `harbor.solved_at`. A forced close (step limit) runs the tests
the same way. The container is then removed; its verifier output is kept in <world>/harbor/<item>/.

Differences from Harbor, by design: containers have no network (tests must not download anything; bake test tools
into the image), the agent's commands run one at a time under a time limit and every process they leave is killed
after each command (as in SiamLM's own sandboxes), [agent].timeout_sec is not enforced (world.max_steps_per_problem
bounds an item), and multi-step tasks, separate verifier environments and artifacts are not supported.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import uuid
from pathlib import Path

from siamlm.context.sandbox import LIVE, Sandbox, docker
from siamlm.task import Outcome, Task

_CANARY = re.compile(r"^(<!--.*canary.*-->|#.*canary.*)$", re.I)

# The agent's commands, then every process but the container's own idle ones is killed (in sh: task images need not
# have python3); the count goes to stderr after the marker, as in Sandbox.WRAPPER.
_KILL = ('n=0; for d in /proc/[0-9]*; do p=${d#/proc/}; case " $4 $$ " in *" $p "*) ;; '
         '*) kill -9 "$p" 2>/dev/null && n=$((n+1));; esac; done')
_WRAPPER = ('timeout -k 5 "$1" bash -c "$2"; rc=$?; ' + _KILL + '; printf "\\n__cmd_end__%s\\n" "$n" >&2; exit $rc')


def _toml(path: Path) -> dict:
    try:
        import tomllib
    except ImportError:                                       # Python 3.10
        try:
            import tomli as tomllib
        except ImportError:
            raise ValueError("Harbor tasks need Python 3.11+ or the tomli package (pip install tomli)") from None
    return tomllib.loads(path.read_text(encoding="utf-8"))


def strip_canary(text: str) -> str:
    """Leading canary comment lines (and the blank lines after them) removed, as Harbor does before an agent sees an
    instruction."""
    lines = text.split("\n")
    i = 0
    while i < len(lines) and _CANARY.match(lines[i].strip()):
        i += 1
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    return "\n".join(lines[i:])


def load(path) -> list[dict]:
    """The Harbor tasks under `path`: a task directory (it has task.toml) or a folder of task directories, in name
    order. Each becomes an item {id, title, slug, dir, instruction, config}."""
    root = Path(path)
    dirs = [root] if (root / "task.toml").exists() else sorted(p for p in root.iterdir() if (p / "task.toml").exists())
    if not dirs:
        raise ValueError(f"no Harbor task (a directory with task.toml) at {root}")
    items = []
    for d in dirs:
        config = _toml(d / "task.toml")
        if config.get("steps"):
            raise ValueError(f"{d.name}: multi-step Harbor tasks are not supported")
        if not (d / "instruction.md").exists() or not (d / "tests" / "test.sh").exists():
            raise ValueError(f"{d.name}: a Harbor task needs instruction.md and tests/test.sh")
        name = str((config.get("task") or {}).get("name") or d.name)
        items.append({"id": f"harbor/{name}", "title": name, "slug": re.sub(r"[^a-z0-9]+", "-", d.name.lower()).strip("-"),
                      "dir": str(d.resolve()), "instruction": strip_canary((d / "instruction.md").read_text(encoding="utf-8")).strip(),
                      "config": config})
    return items


def streams(items: list[dict], n_agents: int, per_agent: int, relation: str, seed: int) -> list[list[dict]]:
    """Each agent's items, as copies (each with its own `uid`, so each gets its own container). The tasks are shuffled
    by the seed; `different`: consecutive blocks, `same`: every agent the same block. per_agent 0: all of them (split
    between the agents for `different`)."""
    order = list(items)
    random.Random(seed).shuffle(order)
    if relation not in ("same", "different"):
        raise ValueError(f"world.relation must be same or different (got {relation})")
    per = per_agent or (len(order) if relation == "same" else len(order) // n_agents)
    need = per if relation == "same" else per * n_agents
    if per < 1 or len(order) < need:
        raise ValueError(f"{len(order)} Harbor tasks, {need} needed ({per} per agent)")
    out = []
    for k in range(n_agents):
        block = order[:per] if relation == "same" else order[k * per:(k + 1) * per]
        out.append([dict(it, uid=f"a{k + 1}-{i + 1}-{it['slug']}") for i, it in enumerate(block)])
    return out


def image_for(item: dict, settings: dict) -> tuple[str, str | None, str]:
    """(image, working directory, the image's user) for an item: [environment].docker_image, or built from environment/ (tagged by the
    hash of its files, so an unchanged environment is built once)."""
    env = item["config"].get("environment") or {}
    folder = Path(item["dir"]) / "environment"
    if (folder / "Dockerfile").exists():
        h = hashlib.sha256()
        for f in sorted(p for p in folder.rglob("*") if p.is_file()):
            h.update(str(f.relative_to(folder)).encode() + b"\0" + f.read_bytes() + b"\0")
        image = f"siamlm-harbor-{item['slug']}:{h.hexdigest()[:12]}"
        if docker(["image", "inspect", image], timeout=30).returncode != 0:
            if not settings.get("build", True):
                raise ValueError(f"image {image} is missing and harbor.build=false")
            out = docker(["build", "-t", image, str(folder)], timeout=float(env.get("build_timeout_sec", 600)))
            if out.returncode != 0:
                raise RuntimeError(f"docker build of {folder} failed: {out.stderr.decode(errors='replace')[-600:]}")
    elif env.get("docker_image"):
        image = str(env["docker_image"])
    else:
        raise ValueError(f"{item['dir']}: no environment/Dockerfile and no [environment].docker_image")
    info = docker(["image", "inspect", image, "--format", "{{.Config.WorkingDir}}\t{{.Config.User}}"], timeout=30)
    if info.returncode != 0:
        raise ValueError(f"image {image} not found (pull it, or give the task an environment/Dockerfile)")
    found_dir, _, found_user = info.stdout.decode().strip("\n").partition("\t")
    return image, env.get("workdir") or found_dir.strip() or None, found_user.strip() or "0"


class ItemSandbox(Sandbox):
    """An item's own container (from its task's image): no network, no host mounts, no privilege escalation, limits on
    processes, memory and CPU; the root filesystem is writable (tasks install and change files), /ctx is an in-memory
    folder for the context file. The image's own user and working directory are kept (or [agent].user, which then
    also writes the context file, and [environment].workdir)."""
    WRAPPER = _WRAPPER

    def __init__(self, settings: dict, image: str, workdir: str | None, user: str | None, env: dict):
        super().__init__(settings)
        self.image, self.workdir, self.user, self.env = image, workdir, user, dict(env or {})
        self.name = f"sbx-harbor-{uuid.uuid4().hex[:12]}"

    def start(self) -> None:
        s = self.s
        args = ["run", "-d", "--rm", "--init", "--name", self.name, "--network", "none",
                "--security-opt", "no-new-privileges", "--pids-limit", str(s.get("pids_limit", 256)),
                "--memory", str(s.get("memory", "2g")), "--cpus", str(s.get("cpus", 1)),
                "--tmpfs", f"/ctx:rw,nosuid,nodev,size={s.get('ctx_size', '16m')},mode=1777"]
        if self.user:
            args += ["--user", str(self.user)]
        for k, v in self.env.items():
            args += ["-e", f"{k}={v}"]
        out = docker(args + ["--entrypoint", "sleep", self.image, "infinity"])
        if out.returncode != 0:
            raise RuntimeError(f"could not start the task container: {out.stderr.decode(errors='replace').strip()[:400]}")
        self.id = out.stdout.decode().strip()
        LIVE[self.name] = self
        listed = docker(["exec", self.name, "sh", "-c", 'for d in /proc/[0-9]*; do p=${d#/proc/}; [ "$p" = "$$" ] || printf "%s " "$p"; done'])
        self.keep = listed.stdout.decode().strip() or "1"

    def restart(self) -> None:
        self.stop()
        self.name, self.restarts = f"sbx-harbor-{uuid.uuid4().hex[:12]}", self.restarts + 1
        self.start()

    def exec_options(self) -> list[str]:
        return ["-w", self.workdir] if self.workdir else []

    def sweep(self) -> int:
        out = docker(["exec", self.name, "sh", "-c", _KILL + '; echo "$n"', "sh", "", "", "", self.keep], timeout=60)
        try:
            return int(out.stdout.decode().strip() or 0)
        except ValueError:
            return -1


SUBMIT = {"type": "function", "function": {
    "name": "submit",
    "description": "Submit your current task when you are done with it. Its tests then run in the task's container "
                   "and decide whether it is solved; then your next task begins.",
    "parameters": {"type": "object", "properties": {
        "note": {"type": "string", "description": "Optional: a short note on what you did (the tests do not read it)."}},
        "required": []}}}


class HarborTask(Task):
    name = "harbor"
    try_name, final_name = None, "submit"
    arg_name = "note"
    needs_shell = True
    item_containers = True                       # the world says so in the bash tool (ctx.bash.workspace becomes "item")
    items = "tasks"
    tool_list = "submit"
    work_or = "submit"
    work_and = "submit calls"
    left_words = "Submits"
    runs_word = "submits"
    last_try_note = ""
    image_id = None

    def __init__(self, settings: dict, out: Path):
        self.s = dict(settings or {})
        self.out = Path(out)
        self.boxes: dict[str, ItemSandbox] = {}
        self.images: dict[str, tuple] = {}       # task directory -> (image, workdir, image user), built once

    def image(self, item: dict) -> tuple:
        if item["dir"] not in self.images:
            self.images[item["dir"]] = image_for(item, self.s)
        return self.images[item["dir"]]

    # --- the item's container -------------------------------------------------------------------------------------
    def sandbox(self, item: dict):
        uid = item["uid"]
        if uid not in self.boxes:
            image, workdir, _ = self.image(item)
            cfg = item["config"]
            env = (cfg.get("environment") or {})
            settings = dict(self.s)
            if env.get("memory_mb"):
                settings["memory"] = f"{int(env['memory_mb'])}m"
            if env.get("cpus"):
                settings["cpus"] = env["cpus"]
            box = ItemSandbox(settings, image, workdir, (cfg.get("agent") or {}).get("user"), env.get("env") or {})
            box.start()
            self.boxes[uid] = box
        return self.boxes[uid]

    def verify(self, item: dict) -> dict:
        """Harbor's verifier in the item's container: /tests from tests/, then /tests/test.sh; the reward files."""
        box = self.sandbox(item)
        if not box.alive():
            box.restart()
        v = item["config"].get("verifier") or {}
        timeout = float(v.get("timeout_sec", 600))
        root = ["exec", "-u", "0", box.name, "sh", "-c"]
        docker(root + ["rm -rf /tests /logs/verifier/reward.txt /logs/verifier/reward.json; mkdir -p /logs/verifier "
                       "/logs/agent && chmod -R 777 /logs"])
        docker(["cp", str(Path(item["dir"]) / "tests") + "/.", f"{box.name}:/tests"], timeout=120)
        docker(root + ["chmod -R a+rX /tests && chmod +x /tests/test.sh"])
        opts = (["-w", box.workdir] if box.workdir else []) + ["-u", str(v.get("user") or self.image(item)[2])]
        for k, val in (v.get("env") or {}).items():
            opts += ["-e", f"{k}={val}"]
        try:
            run = docker(["exec", *opts, box.name, "sh", "-c",
                          f"timeout -k 5 {int(timeout)} /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1"],
                         timeout=timeout + 60)
            code = run.returncode
        except subprocess.TimeoutExpired:
            code = -9
        read = lambda f: docker(["exec", "-u", "0", box.name, "sh", "-c", f"cat /logs/verifier/{f} 2>/dev/null"])
        rewards, error = None, None
        raw_json, raw_txt = read("reward.json").stdout.decode(errors="replace"), read("reward.txt").stdout.decode(errors="replace")
        try:
            if raw_json.strip():
                rewards = json.loads(raw_json)
                if not isinstance(rewards, dict) or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                           and x == x and abs(x) != float("inf") for x in rewards.values()):
                    rewards, error = None, "reward.json is not a flat object of finite numbers"
            elif raw_txt.strip():
                rewards = {"reward": float(raw_txt.strip())}
            else:
                error = "no reward file (reward.txt or reward.json) was written"
        except ValueError as e:
            rewards, error = None, f"unreadable reward file: {e}"
        stdout = read("test-stdout.txt").stdout.decode(errors="replace")
        folder = self.out / "harbor" / item["uid"] / "verifier"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "test-stdout.txt").write_text(stdout[-200000:], encoding="utf-8")
        if raw_json.strip():
            (folder / "reward.json").write_text(raw_json, encoding="utf-8")
        if raw_txt.strip():
            (folder / "reward.txt").write_text(raw_txt, encoding="utf-8")
        return {"rewards": rewards, "error": error, "exit_code": code}

    def release(self, item: dict) -> None:
        box = self.boxes.pop(item["uid"], None)
        if box is not None:
            box.stop()

    # --- the Task API ------------------------------------------------------------------------------------------
    def statement(self, item: dict) -> str:
        workdir = self.image(item)[1]
        where = f" (working directory {workdir})" if workdir else ""
        return (f"Task: {item['title']}\n\n{item['instruction']}\n\nYour bash commands for this task run in its own "
                f"container{where}; the container is removed when the task is closed. When you are done, call submit: "
                "the task's tests then run in that container and decide whether it is solved.")

    def meta(self, item: dict) -> dict:
        return {"harbor_dir": Path(item["dir"]).name, "uid": item.get("uid"),
                "difficulty": (item["config"].get("metadata") or {}).get("difficulty")}

    def preamble(self) -> str:
        return ("You work on tasks in a Linux shell. Each task has a container of its own, where your bash commands run "
                "while the task is open; you submit each task with submit when it is done, and its tests decide whether "
                "it is solved.")

    def series(self, n: int) -> str:
        return f"You work through a series of {n} {self.items}, one at a time."

    def record_words(self, view: str) -> str:
        return {"shared": "each agent's tasks, messages, commands and their output, and the submissions.",
                "alone": "your tasks, your messages, your commands and their output, and your submissions.",
                "chat": "messages, commands and their output, and the submissions."}[view]

    def tool_lines(self, setup: str) -> list[str]:
        return ["- submit(note): submit your current task when it is done; its tests run, and your next task begins."]

    def limit_line(self, T: int, pool: bool) -> str:
        return "Each task takes one submit."

    def work_tools(self, pool: bool) -> list[dict]:
        if pool:
            raise ValueError("Harbor tasks need setup assigned or external (not pool)")
        return [SUBMIT]

    def payload(self, arguments: dict) -> str:
        return str((arguments or {}).get("note") or "").strip()

    def shown(self, words: str, name: str, payload: str) -> str:
        return ((words.strip() + "\n\n") if words.strip() else "") + f"[{name}]" + (f" {payload}" if payload else "")

    def attempt(self, item: dict, payload: str) -> Outcome:
        raise RuntimeError("Harbor tasks have no try calls")

    def grade(self, item: dict, payload: str) -> Outcome:
        try:
            r = self.verify(item)
        finally:
            self.release(item)
        rewards = r["rewards"] or {}
        reward = float(rewards.get("reward", next(iter(rewards.values()), 0.0))) if rewards else 0.0
        ok = r["rewards"] is not None and reward >= float(self.s.get("solved_at", 1.0))
        return Outcome("", True, ok, {"reward": reward, "rewards": r["rewards"], "verifier_exit": r["exit_code"],
                                      "verifier_error": r["error"]})

    def verdict(self, success: bool, title: str | None = None) -> str:
        return " Solved: the tests passed." if success else " Not solved: the tests did not pass."

    def forced_notice(self, position: int, how: str) -> str:
        return f"Task {position} was closed after too many turns without a submit; its tests were run."

    def close(self) -> None:
        for uid in list(self.boxes):
            self.boxes.pop(uid).stop()
