#!/usr/bin/env python3
"""Are all the recordings of the day there and readable? Runs on the server host (ffprobe needed).

  scale_day_preflight.py [--videos DIR] [--day 2026-09-13]

For each of the 15 cameras: number of files, every file opens, first file starts at the beginning of the day, the
last one ends at its end, and the holes and overlaps the timeline player will fill / cut. Exit status 1 when a camera
has no recordings, an unreadable file, or less than 23.5 h of footage.
"""
import argparse, datetime as dt, importlib.util, os, sys

spec = importlib.util.spec_from_file_location("streamer", os.path.join(os.path.dirname(os.path.abspath(__file__)), "scale13_streamer.py"))
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="/data/IRIS/data/el-mohandes")
    ap.add_argument("--day", default="2026-09-13")
    a = ap.parse_args()
    day0 = dt.datetime.fromisoformat(a.day)
    day1 = day0 + dt.timedelta(days=1)
    bad = 0
    print(f"{'cam':7}{'src':6}{'files':>6}  {'covered':>8}  first start       last end            holes (>3 s)   overlaps")
    for cam, prefix in st.CAMS:
        files = st.recordings(a.videos, prefix)
        entries, broken = [], []
        for t, p in files:
            try:
                entries.append((t, p, tuple(st.probe(p))))
            except Exception as exc:  # unreadable file
                broken.append((os.path.basename(p), str(exc)[:60]))
        if not entries:
            print(f"{cam:7}{prefix:6}  NO RECORDINGS"); bad += 1; continue
        pieces = st.plan_timeline(entries, day0 - dt.timedelta(days=1))
        covered = sum((min(p["end"], day1) - max(p["start"], day0)).total_seconds() for p in pieces
                      if p["kind"] == "file" and p["end"] > day0 and p["start"] < day1) / 3600
        holes = [(p["start"], (p["end"] - p["start"]).total_seconds()) for p in pieces if p["kind"] == "fill"]
        overlaps = [(e[0], (e[0] + dt.timedelta(seconds=e[2][0]) - n[0]).total_seconds())
                    for e, n in zip(entries, entries[1:]) if e[0] + dt.timedelta(seconds=e[2][0]) > n[0] + dt.timedelta(seconds=3)]
        last_end = max(p["end"] for p in pieces)
        ok = covered >= 23.5 and not broken
        bad += not ok
        print(f"{cam:7}{prefix:6}{len(files):6}  {covered:7.2f}h  {entries[0][0]:%d %H:%M:%S}   {last_end:%d %H:%M:%S}   "
              f"{', '.join(f'{h:%H:%M}+{s:.0f}s' for h, s in holes) or '-':14} {', '.join(f'{o:%H:%M}:{s:.0f}s' for o, s in overlaps) or '-'}"
              + (f"   BROKEN: {broken}" if broken else "") + ("" if ok else "   <-- PROBLEM"))
    print("PREFLIGHT " + ("FAILED" if bad else "OK: every camera has its day"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
