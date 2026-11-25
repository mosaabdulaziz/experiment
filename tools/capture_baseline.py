"""Capture one attendance baseline run as JSON, for before/after comparison.

A restructure (moving folders, extracting a product) must not change what the
pipeline recognises. Comparing needs the same input -- a recorded clip -- and
the same numbers captured the same way, so this writes them to a file instead
of leaving them in a terminal.

    python3 tools/capture_baseline.py cam-21 --since "2026-09-22 21:04:53" \
        --out tests/baselines/attendance-cam21-before.json

Counts only rows whose sighting ENDED after --since, so an earlier run's rows
in the same database are excluded (the table is shared; the run is not).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "products" / "attendance" / "data" / "attendance.db"


def capture(camera: str, since: str) -> dict:
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = conn.execute(
        """SELECT e.name, COUNT(*), MIN(s.first_seen_at), MAX(s.last_seen_at)
             FROM sighting s JOIN employee e ON e.id = s.employee_id
            WHERE s.camera = ? AND s.last_seen_at >= ?
            GROUP BY e.name ORDER BY e.name""",
        (camera, since),
    ).fetchall()
    total = conn.execute(
        "SELECT COUNT(*) FROM sighting WHERE camera = ? AND last_seen_at >= ?",
        (camera, since),
    ).fetchone()[0]
    conn.close()
    return {
        "camera": camera,
        "since": since,
        "total_sightings": total,
        "employees_recognised": len(rows),
        "per_employee": {
            name: {"sightings": n, "first_seen": first, "last_seen": last}
            for name, n, first, last in rows
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("camera")
    ap.add_argument("--since", required=True, help="ISO timestamp the run started")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = capture(args.camera, args.since)
    Path(args.out).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
