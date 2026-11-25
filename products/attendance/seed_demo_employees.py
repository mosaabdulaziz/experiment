#!/usr/bin/env python3
"""Seed the original 13 demo employees (2026-09-02, DB-driven enrollment:
/grilling round 1 Q8 + /mentor design, "keep the justfile for developing").
Runs ONCE, ever, so a fresh checkout still works out of the box without
needing the admin UI first -- everyone added after that (via the admin API)
is left alone.

2026-09-06: "only if the employee table is empty" (the original rule) is
NOT the same thing as "only once" -- an admin who deletes every employee
through the UI also leaves the table empty, and this script can't tell that
apart from a fresh checkout by row count alone. Found live: delete
everyone, `just down`/`just up`, and the demo roster silently comes back,
undoing the deletion (and burying whatever was added before the table was
next observed empty). The guard is now a one-way seed_state marker
(schema.sql) instead of a row count -- once set, this never seeds again no
matter how many employees currently exist, including zero.

Plain host python3 (sqlite3 stdlib only, via storage.py -- no redis, no
InsightFace, no GPU): this only creates rows. Registering them into the
face gallery is a separate step needing InsightFace -- see
reregister_all.py, which runs inside the GPU container the same way this
recipe's docker commands always have.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from storage import connect, create_employee  # noqa: E402

# person-1..13: the original CHIRLA seq_020 roster (2026-08-31/09-01).
# Simulation-placeholder names -- CHIRLA's subjects are anonymized, these
# were picked to demo the tool, not to represent real people.
DEMO_ROSTER = ["Youssef Amin", "Omar Saeed", "Nour Ibrahim", "Karim Fathy", "Rana Adel",
              "Tarek Younes", "Dina Farouk", "Sami Nabil", "Hassan Mansour", "Sherif Hamdy",
              "Mostafa Kamel", "Ziad Selim", "Fadi Nassar"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    args = ap.parse_args()

    conn = connect(args.db)
    if conn.execute("SELECT 1 FROM seed_state WHERE key = 'demo_roster'").fetchone() is not None:
        print("demo roster already seeded once (or the table was emptied on purpose since) -- not re-seeding")
        return

    for i, name in enumerate(DEMO_ROSTER, start=1):
        create_employee(conn, f"emp-{i}", name, f"person-{i}")
    conn.execute("INSERT INTO seed_state (key, seeded_at) VALUES ('demo_roster', datetime('now'))")
    conn.commit()
    print(f"seeded {len(DEMO_ROSTER)} demo employees")


if __name__ == "__main__":
    main()
