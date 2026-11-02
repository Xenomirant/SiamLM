# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow>=14"]
# ///
"""Download the CodeEval-Pro data SiamLM's codeeval_pro task uses, from the upstream repositories, into a local folder.

    uv run scripts/fetch_codeeval_pro.py [--out data/codeeval_pro]       # (uv installs pyarrow for this script)

Nothing of this data is shipped with SiamLM. CHECK THE TERMS BEFORE USE: the CodeEval-Pro repository has no licence
file (its README and pyproject say MIT for the code; nothing is said about the data). BigCodeBench is Apache-2.0;
HumanEval (the base of HumanEval-Pro) is MIT. You are responsible for whether your use is permitted.

What is written (pinned revisions; sha256 checked where the upstream file is used as it is):
- bigcodebench_lite_pro.json, humaneval_pro.json: CodeEval-Pro (github.com/CodeEval-Pro/CodeEval-Pro, commit
  36f292e, dataset/), unchanged;
- bigcodebench_v0.1.4_tests.json: for the 57 BigCodeBench tasks that are the bases of BigCodeBench-Lite-Pro, the fields
  task_id, complete_prompt, canonical_solution, test, entry_point, libs of BigCodeBench (huggingface.co/datasets/
  bigcode/bigcodebench, revision b74c0d0, split v0.1.4), unchanged, ordered by task number. CodeEval-Pro has no
  tests for the base problems; these are BigCodeBench's own unit tests. Tasks are matched by prompt text, since
  CodeEval-Pro's `id` field is shifted for most records (siamlm/tasks/codeeval_pro.py).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

CEP = "https://raw.githubusercontent.com/CodeEval-Pro/CodeEval-Pro/36f292eafc597b38535fbc8b4aafea8e5c654e3c/dataset/"
BCB = ("https://huggingface.co/datasets/bigcode/bigcodebench/resolve/b74c0d0bf70d2c0bc459be537895cca163007f1a/"
       "data/v0.1.4-00000-of-00001.parquet")
SHA256 = {"bigcodebench_lite_pro.json": "1a9e818a7ecece6346f47673276bfc47ab2bb34efd8cda26e0b819eb13bb6ec3",
          "humaneval_pro.json": "d67ad94366d8182804d8cb59de558d211072493b13a6c23ddda42a2e1367dc7a",
          "bigcodebench_v0.1.4_tests.json": "f823fb73143f6e24c1d3461f6314e0e3cf5e2bc9578d44d768d0a9356d0a5b30"}
FIELDS = ("task_id", "complete_prompt", "canonical_solution", "test", "entry_point", "libs")
NOTICE = ("CodeEval-Pro has no licence file; check its terms (github.com/CodeEval-Pro/CodeEval-Pro) before you use or "
          "share this data.")


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def check(name: str, raw: bytes) -> None:
    got = hashlib.sha256(raw).hexdigest()
    if got != SHA256[name]:
        raise SystemExit(f"{name}: sha256 {got}, expected {SHA256[name]} (the upstream file changed?)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(Path(os.environ.get("SIAMLM_DATA", "data")) / "codeeval_pro"))
    a = ap.parse_args(argv)
    print(NOTICE, file=sys.stderr)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("bigcodebench_lite_pro.json", "humaneval_pro.json"):
        files[name] = get(CEP + name)
        check(name, files[name])
    import pyarrow.parquet as pq
    rows = pq.read_table(io.BytesIO(get(BCB))).to_pylist()
    by_prompt = {r["complete_prompt"].strip(): r for r in rows}
    bases = [by_prompt[r["raw_problem"].strip()] for r in json.loads(files["bigcodebench_lite_pro.json"])]
    bases.sort(key=lambda r: int(r["task_id"].split("/")[1]))
    name = "bigcodebench_v0.1.4_tests.json"
    files[name] = json.dumps([{k: r[k] for k in FIELDS} for r in bases], indent=1).encode()
    check(name, files[name])
    for name, raw in files.items():
        (out / name).write_bytes(raw)
        print(f"-> {out / name} ({len(raw)} bytes)")


if __name__ == "__main__":
    main()
