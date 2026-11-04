"""provider.cache none | prefix | span: per cell, the mean call latency, prompt length, tokens prefilled and reused,
and the behaviour measures (edits applied, acknowledgement rate) from scripts/presence_metrics.py.

    python scripts/cache_reuse_report.py <batch> [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from siamlm.analysis.events import load, worlds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    batch = Path(args.batch)
    cells = defaultdict(list)
    for w in worlds(batch):
        for c in load(w / "calls.jsonl"):
            u = c.get("usage") or {}
            cells[w.parent.name].append({"latency": c.get("latency_s"), "prompt": u.get("prompt_tokens"), "completion": u.get("completion_tokens"),
                                         "prefilled": u.get("cache_prefilled", u.get("prompt_tokens")), "reused": u.get("cache_reused", 0),
                                         "relocated": u.get("cache_relocated", 0), "mode": u.get("cache_mode", "none"), "world": w.name})
    lines = ["| cell | calls | mode | latency s/call (mean, median) | prompt tokens | prefilled | reused | relocated | prefill share |",
             "|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for cell, rows in sorted(cells.items()):
        lat = [r["latency"] for r in rows if r["latency"] is not None]
        pr = [r["prompt"] or 0 for r in rows]
        pf = [r["prefilled"] or 0 for r in rows]
        re_ = [r["reused"] or 0 for r in rows]
        rl = [r["relocated"] or 0 for r in rows]
        share = sum(pf) / max(1, sum(pr))
        summary[cell] = {"calls": len(rows), "mode": rows[0]["mode"], "latency_mean": round(st.mean(lat), 2), "latency_median": round(st.median(lat), 2),
                         "prompt_mean": round(st.mean(pr)), "prefilled_mean": round(st.mean(pf)), "reused_mean": round(st.mean(re_)),
                         "relocated_mean": round(st.mean(rl)), "prefill_share": round(share, 3)}
        s = summary[cell]
        lines.append(f"| {cell} | {s['calls']} | {s['mode']} | {s['latency_mean']}, {s['latency_median']} | {s['prompt_mean']} | {s['prefilled_mean']} | "
                     f"{s['reused_mean']} | {s['relocated_mean']} | {s['prefill_share']} |")
    out = Path(args.out) if args.out else batch / "cache_report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cache_report.json").write_text(json.dumps(summary, indent=1))
    (out / "cache_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
