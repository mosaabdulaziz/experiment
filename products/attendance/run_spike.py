"""One-off spike runner: backfills every recognition entry seen so far on the
CHIRLA seq_004 streams into Sighting rows via the real ACL, then prints the
computed daily attendance for each enrolled employee. Not the production
entrypoint (that's consumer.ConsumerThread, running continuously) -- this
reads full history via XRANGE for one-off backfills/debugging.
"""
import json
import os
from datetime import date

import redis

from acl import RawRecognitionEntry, SightingCoalescer, wall_clock_for_spike
from cameras import CAM_DURATIONS, CAM_FILENAMES
from storage import connect, compute_daily_attendance

r = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
conn = connect("data/attendance.db")
coalescer = SightingCoalescer(conn)

total_ingested = 0
for cam_id, filename in CAM_FILENAMES.items():
    key = f"iris:face_rec:{cam_id}"
    raw_entries = r.xrange(key, "-", "+")
    for _id, fields in raw_entries:
        pts = float(fields[b"pts"])
        wall_clock = wall_clock_for_spike(filename, pts, CAM_DURATIONS[cam_id])
        for face in json.loads(fields[b"face_rec"]):
            entry = RawRecognitionEntry(
                pts=pts,
                parent_id=face.get("parent_id"),
                name=face["name"],
                confidence=face["confidence"],
                rec_id=face.get("rec_id"),
            )
            coalescer.ingest(cam_id, entry, wall_clock)
            total_ingested += 1

flushed = coalescer.flush_due_buckets(force=True)  # backfill: flush everything unconditionally
print(f"ingested {total_ingested} raw recognition entries -> {len(flushed)} coalesced sightings")
for s in flushed:
    print(f"  {s}")

print()
print("=== daily attendance (spike's synthetic 'day' = 2023-06-08, the recording date) ===")
employees = conn.execute("SELECT id, name FROM employee WHERE active = 1 ORDER BY name").fetchall()
for emp in employees:
    record = compute_daily_attendance(conn, emp["id"], date(2023, 6, 8))
    print(f"{emp['name']}: {record}")
