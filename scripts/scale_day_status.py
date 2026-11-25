#!/usr/bin/env python3
"""How the full-day run is doing, in one screen. Runs on the host next to the stack (python3 + docker only).

  scale_day_status.py            footage clock, containers, cameras, heals, platform counts, feed, disk
  scale_day_status.py --brief    one line (for a periodic check)
"""
import argparse, datetime as dt, glob, json, os, shutil, subprocess, time

DEST = os.environ.get("SCALE_DEST", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DAY_STATE = os.path.join(DEST, "data/day-player")
STATE = os.environ.get("SCALE13_STATE") or (_DAY_STATE if os.path.exists(os.path.join(DEST, "data/.day_run")) else "/tmp/scale13-streamer")
PG = "scale-ui-postgres-1"


def sh(cmd, timeout=30):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def footage():
    try:
        c = json.load(open(os.path.join(STATE, "clock.json")))
    except (OSError, ValueError):
        return None
    start = dt.datetime.fromisoformat(c["start"])
    day0 = start.replace(hour=0, minute=0, second=0, microsecond=0)
    now = day0 + dt.timedelta(seconds=(start - day0).total_seconds() + time.time() - c["t0"])
    return now, day0, c


def camera_ages():
    out = sh("docker exec scale-redis-1 sh -c 'for c in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15; do "
             "echo cam-$c $(redis-cli XINFO STREAM iris:meta:cam-$c | grep -a -A1 last-generated-id | tail -1); done'")
    ages = {}
    if not out.strip():
        return None          # Redis is not answering: the cameras' state is unknown, not "all fine"
    for line in out.splitlines():
        p = line.split()
        try:
            ages[p[0]] = time.time() - int(p[1].split("-")[0]) / 1000.0
        except (IndexError, ValueError):
            ages[p[0] if p else "?"] = float("inf")
    return ages


def counts():
    q = ("select 'trucks',count(*) from trucks union all select 'journeys',count(*) from journeys "
         "union all select 'visits',count(*) from visits union all select 'sightings',count(*) from sightings "
         "union all select 'alerts',count(*) from alerts union all select 'raw_events',count(*) from feed_events_raw")
    out = sh(f'docker exec {PG} psql -U iris -d iris -At -F" " -c "{q}"')
    return {l.split()[0]: int(l.split()[1]) for l in out.splitlines() if len(l.split()) == 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", action="store_true")
    a = ap.parse_args()
    f = footage()
    ages = camera_ages()
    redis_down = ages is None
    ages = ages or {}
    dead = sorted(c for c, v in ages.items() if v > 30)
    heal_path = os.path.join(DEST, "data/scale13_heals.log")
    heals = open(heal_path).read().splitlines() if os.path.exists(heal_path) else []
    c = counts()
    up = [l.split()[0] for l in sh("docker ps --format '{{.Names}} {{.Status}}'").splitlines() if l.startswith("scale-")]
    free_gb = shutil.disk_usage(DEST).free / 1e9
    clock = f"{f[0]:%H:%M:%S} ({(f[0] - f[1]).total_seconds() / 864:.1f}% of the day)" if f else "no player clock"
    if a.brief:
        print(f"{time.strftime('%H:%M')} footage {clock}; cameras dead {'UNKNOWN (redis down)' if redis_down else str(len(dead)) + (str(dead) if dead else '')}; heals {len(heals)}; "
              f"trucks {c.get('trucks', '?')} journeys {c.get('journeys', '?')} sightings {c.get('sightings', '?')} alerts {c.get('alerts', '?')}; "
              f"{len(up)} containers up; {free_gb:.0f} GB free")
        return
    print(f"wall {time.strftime('%F %T')}   footage clock {clock}")
    if f:
        begin = dt.datetime.fromtimestamp(f[2]["t0"])
        print(f"footage {f[2]['start']} began at wall {begin:%F %T}" + ("  (not yet)" if time.time() < f[2]["t0"] else ""))
    sup = sh(f"cat {STATE}/supervisor.pid 2>/dev/null").strip()
    pushes = sh("pgrep -fc 'ffmpeg.*raw-cam'").strip() or "0"
    print(f"player supervisor: {'running pid ' + sup if sup and os.path.exists('/proc/' + sup) else 'NOT running'}; ffmpeg pushes: {pushes}")
    print(f"containers up ({len(up)}): {' '.join(sorted(up))}")
    print("cameras publishing: UNKNOWN, Redis is not answering" if redis_down else
          f"cameras publishing: {15 - len(dead)}/15; oldest output age {max(ages.values(), default=0):.0f}s; dead: {dead or 'none'}")
    for cam in sorted(glob.glob(os.path.join(STATE, "cam-*.log"))):
        tail = [l for l in open(cam, errors="replace").read().splitlines() if "hole" in l][-1:]
        if tail:
            print("  ", os.path.basename(cam)[:6], tail[0][:100])
    print(f"heals so far: {len(heals)}" + (f"; last: {heals[-1]}" if heals else ""))
    print("platform:", ", ".join(f"{k} {v}" for k, v in c.items()) or "database not answering")
    print("feed queue (db1 iris:camera-feed):", sh("docker exec scale-redis-1 redis-cli -n 1 llen iris:camera-feed").strip())
    jobs = sh(f"wc -l < {DEST}/data/vlm_arbitration/arbitration_log.jsonl 2>/dev/null").strip()
    crops = sh(f"find {DEST}/data/vehicle_crops -type f 2>/dev/null | wc -l").strip()
    plates = sh(f"ls {DEST}/data/plates 2>/dev/null | wc -l").strip()
    size = sh(f"du -sh {DEST}/data | cut -f1").strip()
    print(f"VLM jobs {jobs or 0}, crop files {crops}, plate files {plates}; disk free {free_gb:.0f} GB; data dir {size}")
    print("last metric row:", sh(f"tail -1 {DEST}/data/scale13_metrics.csv").strip())
    print("stays file:", sh(f"ls -l --time-style=+%T {DEST}/data/tracking_stays.json 2>/dev/null | awk '{{print $5, $6}}'").strip() or "none yet")


if __name__ == "__main__":
    main()
