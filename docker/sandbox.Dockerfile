# Sandbox for bash editing (ctx.edit=bash) and the state_chain task: python3, bash, coreutils, sed, grep.
# At run time (siamlm/context/sandbox.py): no network, read-only root, all capabilities dropped, no host mounts,
# run as the host user, in-memory /ctx /work /tmp, process / memory / CPU / file-size limits.
# Build (in the repo root):  docker build -t siamlm-sandbox:1 -f docker/sandbox.Dockerfile docker
# The base is pinned by digest (python:3.12-slim-bookworm); runs record the built image's id.
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
RUN mkdir -p /ctx /work && chmod 1777 /ctx /work
WORKDIR /work
CMD ["sleep", "infinity"]
