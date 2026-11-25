#!/usr/bin/env python3
"""Phase 0 of VLM_ASSOCIATION_RECORD_REPLAY_PLAN.md: build a human-reviewable
gallery from a gold-label capture run (mtmc.gold_dump_dir in sidecar.yaml).

Reads data/gold_labels/summary.jsonl (one row per (cam_id, local_id) tracklet,
written the first time it gets a global_id -- see MTMCManager._maybe_dump_gold
in perception/modules/mtmc/core/__init__.py) and groups rows by global_id, then renders one
static HTML page with a crop thumbnail per member so a human can confirm or
correct each proposed grouping. Nothing here writes gold_mapping.json --that
file is only written after a human has reviewed this page (see the plan).

Usage:
    python3 scripts/build_gold_review.py [data/gold_labels]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

CAM_LABELS = {  # matches config.yaml's mtmc_use_case.camera_labels for readability
    "cam-01": "Cam-41 (D41)",
    "cam-02": "Cam-16 (D16)",
    "cam-03": "Cam-52 (D52)",
    "cam-04": "Cam-05 (CH05)",
    "cam-05": "Cam-23 (CH23)",
}


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/gold_labels")
    summary_path = root / "summary.jsonl"
    if not summary_path.exists():
        print(f"no {summary_path} -- nothing to build (did the run enable gold_dump_dir?)")
        sys.exit(1)

    groups: dict[int, list[dict]] = defaultdict(list)
    with open(summary_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            groups[row["global_id"]].append(row)

    for gid in groups:
        groups[gid].sort(key=lambda r: r["pts"])

    ordered_gids = sorted(groups.keys(), key=lambda g: min(r["pts"] for r in groups[g]))

    n_multi = sum(1 for g in groups.values() if len(g) > 1)
    n_single = len(groups) - n_multi

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Phase 0 gold-label review</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:24px}",
        "h1{font-size:20px} p.sub{color:#999;margin-top:-8px}",
        ".group{border:1px solid #333;border-radius:8px;margin:16px 0;padding:12px;background:#1a1a1a}",
        ".group.single{opacity:0.7}",
        ".gid{font-weight:600;font-size:15px;margin-bottom:8px}",
        ".gid .n{color:#8ab4f8}",
        ".members{display:flex;flex-wrap:wrap;gap:12px}",
        ".m{width:160px;text-align:center;font-size:12px;color:#ccc}",
        ".m img{width:160px;height:120px;object-fit:cover;border-radius:4px;border:1px solid #444;background:#000}",
        ".m .cam{font-weight:600;color:#fff}",
        ".tag{display:inline-block;font-size:11px;padding:1px 6px;border-radius:3px;margin-left:8px}",
        ".tag.multi{background:#1e4620;color:#8fd98f}",
        ".tag.single{background:#402a1e;color:#e0a97a}",
        "textarea{width:100%;min-height:60px;background:#000;color:#0f0;border:1px solid #333;",
        "  font-family:monospace;font-size:12px;margin-top:8px;display:none}",
        "</style></head><body>",
        f"<h1>Phase 0 gold-label review</h1>",
        f"<p class='sub'>{len(groups)} proposed global IDs "
        f"({n_multi} multi-camera, {n_single} single-camera-only) from "
        f"<code>{summary_path}</code>. This is a proposal, not gold -- confirm "
        f"or correct each group below, then tell the agent what to fix before "
        f"data/gold_labels/gold_mapping.json gets written.</p>",
    ]

    for gid in ordered_gids:
        members = groups[gid]
        cams = [m["cam_id"] for m in members]
        cls = "single" if len(members) == 1 else "multi"
        tag = f"<span class='tag {cls}'>{len(members)} camera{'s' if len(members) != 1 else ''}</span>"
        parts.append(f"<div class='group {cls}'>")
        parts.append(f"<div class='gid'>Global ID <span class='n'>{gid}</span>{tag}</div>")
        parts.append("<div class='members'>")
        for m in members:
            label = CAM_LABELS.get(m["cam_id"], m["cam_id"])
            parts.append(
                f"<div class='m'><img src='{m['crop']}' loading='lazy'>"
                f"<div class='cam'>{label}</div>"
                f"<div>local_id={m['local_id']} @ {m['pts']}s</div></div>"
            )
        parts.append("</div></div>")

    parts.append("</body></html>")

    out_path = root / "review.html"
    out_path.write_text("\n".join(parts))
    print(f"wrote {out_path} ({len(groups)} groups: {n_multi} multi-camera, {n_single} single)")
    print(f"open with: xdg-open {out_path.resolve()}  (or just open the file in a browser)")


if __name__ == "__main__":
    main()
