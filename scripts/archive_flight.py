#!/usr/bin/env python3
"""Move the flight records of the last run (events.log, stall dumps, native backtraces) into
<dir>/archive-<time>/ so a fresh run starts with an empty folder but the evidence is kept.
Run inside a container as root (the files are root-owned):  archive_flight.py /f"""
import os
import shutil
import sys
import time

d = sys.argv[1] if len(sys.argv) > 1 else "data/flight"
files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))] if os.path.isdir(d) else []
if files:
    dest = os.path.join(d, "archive-" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(dest, exist_ok=True)
    for f in files:
        shutil.move(os.path.join(d, f), os.path.join(dest, f))
    print(f"    last run's flight records ({len(files)} files) archived in {dest}")
