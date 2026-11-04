"""A Docker-free sandbox for bash editing (ctx.bash.backend=local): each command runs in a fresh user + mount + pid +
network namespace of this host (`unshare -rmnpf --kill-child`), with tmpfs mounted at the real paths the agents are
told about, /ctx and /work (and /tmp), no network, every process killed when the command ends, a time limit, output
caps, file-size and address-space limits. Between commands the agent's /work and /ctx live in a folder of the host
(`root`), copied in before and out after each command, so the workspace persists for the run as in Docker.

What it does NOT give, compared with the Docker sandbox: the host's filesystem is visible (read-only for the
agent: the namespace is entered as nobody, uid 65534, so outside /work, /ctx, /tmp and the sandbox's own folder it
can write nothing and read only what is world-readable), and there is no /proc inside (the container forbids mounting it), so `ps`
and the like do not work. It is for
local, offline models on a machine without Docker. The world warns at start-up when it is used. Needs: a kernel that
allows unprivileged user namespaces with tmpfs mounts, and the directories /ctx and /work on the host (created if
this user may; they stay empty on the host).
"""
from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from .sandbox import FILE, _drain

LIVE: dict[str, "LocalSandbox"] = {}


def _sq(text: str) -> str:
    """Single-quoted for sh."""
    return "'" + text.replace("'", "'\\''") + "'"
ROOTS = Path(os.environ.get("SIAMLM_LOCALBOX_ROOT", "/var/tmp/siamlm-localbox"))
MOUNTS = ("/ctx", "/work")
_checked: dict = {}


def available() -> str | None:
    """None when the local backend can run here; else why not."""
    if "why" in _checked:
        return _checked["why"]
    why = None
    for d in MOUNTS:
        try:
            Path(d).mkdir(exist_ok=True)
        except OSError as e:
            why = f"{d} cannot be created on the host ({e})"
            break
    if why is None:
        try:
            out = subprocess.run(["unshare", "-rmnpf", "--kill-child", "--propagation", "private", "sh", "-c",
                                  "mount -t tmpfs -o size=1m none /work && mount -t tmpfs -o size=1m none /ctx && echo ok"],
                                 capture_output=True, timeout=30)
            if out.returncode != 0 or b"ok" not in out.stdout:
                why = f"user-namespace tmpfs mounts fail here: {out.stderr.decode(errors='replace').strip()[:200]}"
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            why = f"unshare not usable ({e})"
    _checked["why"] = why
    return why


def check_available() -> None:
    why = available()
    if why:
        raise ValueError(f"ctx.bash.backend=local: {why}")


class LocalSandbox:
    """The Docker Sandbox's interface (start, alive, restart, put, get, run, exec, read_tree, sweep, reset_scratch,
    stop, name, restarts, keep) on this host."""
    workdir = "/work"

    def __init__(self, settings):
        self.s = settings
        self.name, self.root, self.restarts, self.keep, self.id = f"lbx-{uuid.uuid4().hex[:12]}", None, 0, "", None

    def _own(self) -> None:
        """The sandbox's host folder belongs to nobody, the user the namespace is entered as (when this process is
        root; an unprivileged user keeps its own files)."""
        if os.getuid() == 0 and self.root is not None:
            for p in [self.root, *self.root.rglob("*")]:
                try:
                    os.chown(p, self.UID, self.UID, follow_symlinks=False)
                except OSError:
                    pass

    def start(self) -> None:
        ROOTS.mkdir(parents=True, exist_ok=True)
        if os.getuid() == 0:
            ROOTS.chmod(0o1777)
        self.root = Path(tempfile.mkdtemp(prefix=self.name + "-", dir=ROOTS))
        for d in ("ctx", "work", "tmp"):
            (self.root / d).mkdir()
        self._own()
        self.id = str(self.root)
        LIVE[self.name] = self

    def alive(self) -> bool:
        return self.root is not None and self.root.is_dir()

    def restart(self) -> None:
        self.stop()
        self.name, self.restarts = f"lbx-{uuid.uuid4().hex[:12]}", self.restarts + 1
        self.start()

    def put(self, text: str) -> None:
        if not self.alive():
            self.restart()
        target = self.root / "ctx" / "context.txt"
        if target.is_symlink() or target.is_dir():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        target.write_text(text, encoding="utf-8")
        self._own()

    def get(self) -> tuple[str | None, str | None]:
        limit = int(self.s.get("max_file_bytes", 8 * 2**20))
        target = self.root / "ctx" / "context.txt"
        if target.is_symlink() or not target.is_file():
            return None, f"{FILE} is not a regular file"
        if target.stat().st_size > limit:
            return None, f"{FILE} is larger than {limit} bytes"
        return target.read_text(encoding="utf-8", errors="replace"), None

    UID = 65534                                       # nobody: the user the namespace is entered as

    def _inner(self, body: str, drop: bool = True) -> str:
        """The script run inside the namespace: mount tmpfs at /work, /ctx and /tmp, copy the agent's files in, run
        `body` from /work, copy out. The namespace is entered as nobody (uid 65534 on the host), which `unshare -r`
        maps to root inside: it may mount tmpfs there and nothing else; on the host it is nobody, so outside the
        three tmpfs mounts and the sandbox's own folder it can write nothing and read only what is world-readable."""
        s, r = self.s, self.root
        return (f"mount -t tmpfs -o size={s.get('tmpfs_size', '64m')} none /work && "
                f"mount -t tmpfs -o size={s.get('ctx_size', '16m')} none /ctx && "
                f"mount -t tmpfs -o size={s.get('tmpfs_size', '64m')} none /tmp 2>/dev/null; "
                f"cp -a '{r}/work/.' /work/ 2>/dev/null; cp -a '{r}/ctx/.' /ctx/ 2>/dev/null; cp -a '{r}/tmp/.' /tmp/ 2>/dev/null; "
                f"cd /work; "
                f"{body}; rc=$?; "
                f"rm -rf '{r}/work' '{r}/ctx' '{r}/tmp'; cp -a /work '{r}/work' 2>/dev/null; cp -a /ctx '{r}/ctx' 2>/dev/null; "
                f"cp -a /tmp '{r}/tmp' 2>/dev/null; exit $rc")

    def _launch(self, body: str, stdin_data: bytes | None, timeout: float, drop: bool = True):
        limits = ["prlimit", f"--fsize={int(self.s.get('max_file_bytes', 8 * 2**20))}",
                  f"--nofile={int(self.s.get('nofile', 256))}", f"--as={int(self.s.get('local_as_bytes', 4 * 2**30))}"]
        who = ["setpriv", f"--reuid={self.UID}", f"--regid={self.UID}", "--clear-groups"] if os.getuid() == 0 else []
        cmd = [*limits, *who, "unshare", "-rmnpf", "--kill-child", "--propagation", "private", "bash", "-c", self._inner(body, drop)]
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/work", "TMPDIR": "/tmp", "LANG": "C.UTF-8",
               "TERM": "dumb"}
        return subprocess.Popen(cmd, stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    def run(self, command: str) -> dict:
        limit, cap = int(self.s.get("command_timeout_s", 120)), int(self.s.get("output_bytes", 65536))
        if not self.alive():
            self.restart()
        (self.root / "tmp").mkdir(exist_ok=True)
        (self.root / "tmp" / ".cmd.sh").write_text(command, encoding="utf-8")   # reaches the agent as /tmp/.cmd.sh
        self._own()
        started, sink = time.time(), {}
        proc = self._launch(f"timeout -k 5 {limit} bash /tmp/.cmd.sh", None, limit, drop=True)
        readers = [threading.Thread(target=_drain, args=(proc.stdout, cap, sink, "stdout")),
                   threading.Thread(target=_drain, args=(proc.stderr, cap, sink, "stderr"))]
        for t in readers:
            t.start()
        try:
            code = proc.wait(timeout=limit + 30)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = proc.wait()
        for t in readers:
            t.join(timeout=10)
        elapsed = time.time() - started
        return {"exit_code": code, "stdout": sink.get("stdout", b"").decode("utf-8", errors="replace"),
                "stderr": sink.get("stderr", b"").decode("utf-8", errors="replace"), "container_died": not self.alive(),
                "stdout_bytes": sink.get("stdout_bytes", 0), "stderr_bytes": sink.get("stderr_bytes", 0),
                "timed_out": code in (124, 137, -9) and elapsed >= limit - 1, "seconds": round(elapsed, 2),
                "killed_processes": -1, "leftover_processes": 0}      # the pid namespace ends with the command

    def exec(self, script: str, data: bytes | None = None, args: list[str] | None = None) -> subprocess.CompletedProcess:
        """A harness script (sh -c) inside the sandbox with optional stdin and positional arguments, as `docker exec -i
        <box> sh -c script sh args...` would run it."""
        if not self.alive():
            self.restart()
        path = self.root / ".exec.sh"
        path.write_text(script, encoding="utf-8")
        self._own()
        quoted = " ".join(_sq(a) for a in (args or []))
        proc = self._launch(f"sh '{path}' {quoted}", data, 120, drop=False)   # a harness script, not the agent's
        try:
            out, err = proc.communicate(input=data, timeout=180)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
        return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)

    def read_tree(self, path: str, cap: int = 20000) -> dict:
        import hashlib
        if not self.alive():
            return {}
        base = self.root / path.lstrip("/")
        out = {}
        for d, _, fs in os.walk(base):
            for f in fs:
                q = Path(d) / f
                if q.is_symlink() or not q.is_file():
                    continue
                b = q.read_bytes()[:8 * 2**20]
                out[str(q.relative_to(base))] = {"bytes": q.stat().st_size, "sha1": hashlib.sha1(b).hexdigest(),
                                                 "text": b[:cap * 4].decode("utf-8", "replace")[:cap]}
        return out

    def sweep(self) -> int:
        return 0

    def reset_scratch(self) -> None:
        if not self.alive():
            self.restart()
        for d in ("work", "tmp"):
            shutil.rmtree(self.root / d, ignore_errors=True)
            (self.root / d).mkdir(exist_ok=True)
        for p in (self.root / "ctx").iterdir():
            if p.name != "context.txt":
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() and not p.is_symlink() else p.unlink(missing_ok=True)
        self._own()

    def stop(self) -> None:
        if self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)
            self.root = None
        LIVE.pop(self.name, None)


def stop_all() -> None:
    for box in list(LIVE.values()):
        try:
            box.stop()
        except Exception:
            pass


atexit.register(stop_all)
