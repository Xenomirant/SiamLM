#!/bin/bash
# Environment setup, idempotent: run after every container restart (the container here is rebuilt about daily).
#   1. the venv (system torch, hydra, transformers from the uv cache or the mirror);
#   2. Docker when this container can run it, else the Docker-free local sandbox (ctx.bash.backend=local);
#   3. the sandbox images when Docker is available.
# Docker inside this container needs the container to be started with privileges it does not have by default
# (CAP_SYS_ADMIN for overlay/fuse mounts, a writable cgroup root, /dev/fuse, network namespaces with iptables):
# `--privileged`, or `--cap-add SYS_ADMIN --device /dev/fuse` plus a writable cgroup, or a mounted
# /var/run/docker.sock of the host. Without one of those, dockerd, rootless Docker and Podman all fail at their
# first mount, and this script falls back to the local backend and says so.
set -u
cd "$(dirname "$0")/.."
UV=${UV:-/workspace/opt/bin/uv}
LOG=outputs/_logs; mkdir -p $LOG

# 1. venv
if [ ! -x .venv/bin/python ]; then
  $UV venv --system-site-packages --python /usr/bin/python3.12 .venv
  $UV sync --inexact --extra tiktoken --offline || $UV sync --inexact --extra tiktoken
  $UV pip install --offline "transformers==4.57.6" || $UV pip install "transformers==4.57.6"
  $UV pip uninstall numpy >/dev/null 2>&1 || true          # the venv must see the system numpy that system scipy wants
fi
.venv/bin/python -c "import torch, transformers, hydra; print('venv ok: torch', torch.__version__, 'transformers', transformers.__version__)"

# 2. Docker, if this container can run it
can_mount() { d=$(mktemp -d); mount -t tmpfs none "$d" 2>/dev/null && { umount "$d"; rmdir "$d"; return 0; }; rmdir "$d"; return 1; }
if [ -S /var/run/docker.sock ] || command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "docker: available (daemon reachable)"; DOCKER=1
elif can_mount && grep -q cap_sys_admin <(capsh --decode="$(awk '/CapEff/{print $2}' /proc/self/status)" 2>/dev/null); then
  echo "docker: this container can mount; installing the daemon"
  if ! command -v dockerd >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq docker.io >/dev/null 2>&1 || echo "docker: apt install failed"
  fi
  if command -v dockerd >/dev/null 2>&1; then
    (setsid nohup dockerd --storage-driver=vfs --iptables=false --bridge=none > $LOG/dockerd.log 2>&1 &)
    for i in $(seq 1 30); do docker info >/dev/null 2>&1 && break; sleep 1; done
    docker info >/dev/null 2>&1 && { echo "docker: daemon started"; DOCKER=1; } || echo "docker: daemon did not start ($LOG/dockerd.log)"
  fi
else
  echo "docker: not possible here (no CAP_SYS_ADMIN / no mounts): bash editing uses ctx.bash.backend=local"
  mkdir -p /ctx /work 2>/dev/null
  .venv/bin/python -c "from siamlm.context import localbox; w = localbox.available(); print('local sandbox:', 'ok' if w is None else w)"
fi

# 3. the images, when Docker runs
if [ "${DOCKER:-0}" = 1 ]; then
  docker image inspect siamlm-sandbox:1 >/dev/null 2>&1 || docker build -q -t siamlm-sandbox:1 -f docker/sandbox.Dockerfile docker
  docker image inspect siamlm-lcb-grader:1 >/dev/null 2>&1 || docker build -q -t siamlm-lcb-grader:1 -f docker/lcb-grader.Dockerfile .
  docker image inspect siamlm-cep-grader:1 >/dev/null 2>&1 || docker build -q -t siamlm-cep-grader:1 -f docker/cep-grader.Dockerfile .
  echo "docker: images ready"
fi
