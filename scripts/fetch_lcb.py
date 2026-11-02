"""Download LiveCodeBench problems (code_generation_lite) from Hugging Face into a local folder.

    python scripts/fetch_lcb.py [--files test6.jsonl] [--out data/lcb]

Nothing of this data is shipped with SiamLM. CHECK THE TERMS BEFORE USE: LiveCodeBench's code is MIT, but the licence
on the dataset card of livecodebench/code_generation_lite is given only as "cc" (which Creative Commons licence is not
stated). You are responsible for whether your use is permitted.

The default is test6.jsonl (the problems added in release_v6, 175 problems, ~134 MB) at a pinned revision, checked by
sha256; SiamLM's default config reads <data.dir>/lcb/test6.jsonl (problems.file). Other release files (test.jsonl ...
test5.jsonl, up to 1.25 GB each) can be fetched with --files; they are checked only against the size Hugging Face
reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = "https://huggingface.co/datasets/livecodebench/code_generation_lite"
REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"      # main as of 5 Jun 2025
SHA256 = {"test6.jsonl": "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"}
NOTICE = ("LiveCodeBench's dataset licence is unclear ('cc' on the dataset card); check its terms "
          f"({REPO}) before you use or share this data.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--files", nargs="+", default=["test6.jsonl"])
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--out", default=str(Path(os.environ.get("SIAMLM_DATA", "data")) / "lcb"))
    a = ap.parse_args(argv)
    print(NOTICE, file=sys.stderr)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/livecodebench/code_generation_lite/tree/{a.revision}",
                                timeout=60) as r:
        sizes = {x["path"]: x.get("size") for x in json.load(r)}
    for name in a.files:
        if name not in sizes:
            raise SystemExit(f"{name} is not in the dataset at revision {a.revision}: {sorted(sizes)}")
        target, part = out / name, out / (name + ".part")
        digest, n = hashlib.sha256(), 0
        with urllib.request.urlopen(f"{REPO}/resolve/{a.revision}/{name}", timeout=600) as r, part.open("wb") as f:
            for chunk in iter(lambda: r.read(1 << 20), b""):
                f.write(chunk)
                digest.update(chunk)
                n += len(chunk)
        if n != sizes[name] or (a.revision == REVISION and name in SHA256 and digest.hexdigest() != SHA256[name]):
            part.unlink()
            raise SystemExit(f"{name}: {n} bytes, sha256 {digest.hexdigest()}: not the expected file")
        part.replace(target)
        print(f"-> {target} ({n} bytes)")


if __name__ == "__main__":
    main()
