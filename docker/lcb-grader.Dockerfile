# Grader for LiveCodeBench programs: python 3.12 + numpy (LCB's testing_util imports it), LCB's testing_util (vendored,
# unchanged; siamlm/lcb/) and our driver. Runs with the sandbox settings of siamlm/context/sandbox.py (no network, a
# read-only root, as the host user). Agents never get a shell in it: only the harness runs grading jobs.
# Build (in the repo root):  docker build -t siamlm-lcb-grader:1 -f docker/lcb-grader.Dockerfile .
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
RUN pip install --no-cache-dir numpy==2.3.4
RUN mkdir -p /grader /work && chmod 1777 /work
COPY siamlm/lcb/testing_util.py siamlm/lcb/driver.py /grader/
WORKDIR /work
CMD ["sleep", "infinity"]
