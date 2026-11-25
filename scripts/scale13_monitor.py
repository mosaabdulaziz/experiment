#!/usr/bin/env python3
"""Once a minute, append one row of health numbers for the scale13 run to a CSV.

Runs on the server host next to the stack (python3 + docker + nvidia-smi only).
  scale13_monitor.py run [--out FILE] [--every 60]     scale13_monitor.py stop
Columns: time, min/avg fps over the 13 cameras (from the sidecar's gauge: stale for a dead camera), cameras with no\ninference output for 30 s (the real liveness number),
GPU util %, decoder util %, GPU memory MiB (shared GPU: includes other people's jobs), sidecar CPU %,
sidecar / iris / UI memory MiB, go2rtc publishers, streamer restarts so far, tracks, transitions,
VLM requests in the last minute.
"""
import argparse, csv, json, os, re, signal, subprocess, sys, time, urllib.request

STATE = "/tmp/scale13-monitor.pid"
SIDECAR, IRIS, UI = "scale-deepstream-sidecar-1", "scale-iris-1", "scale-ui-1"


def sh(cmd, timeout=20):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def get(url, timeout=8):
    try:
        return urllib.request.urlopen(url, timeout=timeout).read().decode()
    except Exception:
        return ""


def stale_cameras(limit=30.0):
    """Cameras whose newest inference metadata is older than `limit` s: dead, not just slow.

    The sidecar's fps gauge keeps its last value after a source dies, so it cannot be trusted for
    liveness (it showed ~9 fps for cameras that had been dead for hours).
    """
    out = sh("docker exec scale-redis-1 sh -c 'for c in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15; do "
             "redis-cli XINFO STREAM iris:meta:cam-$c | grep -a -A1 last-generated-id | tail -1; done'")
    now = time.time()
    ages = []
    for line in out.split():
        try:
            ages.append(now - int(line.split("-")[0]) / 1000.0)
        except ValueError:
            ages.append(float("inf"))
    return sum(1 for a in ages if a > limit) + max(0, 15 - len(ages))


DEST = os.environ.get("SCALE_DEST", "/data/IRIS/TESTING_PLAYGROUND/surveillance-camera-sys-scale")
DC = ("docker compose -p scale -f docker-compose.yml -f docker-compose.scale.yml -f docker-compose.scale-ui.yml "
      "-f docker-compose.scale-monitoring.yml")


def heal(log_path):
    """Bring a lost camera back by restarting the sidecar only.

    A source that has gone silent does not resume inside the running sidecar (reproduced: the sidecar's
    own reconnect and the stall watchdog both fail), so the sidecar is restarted. Its global ids carry on
    from the counter kept in Redis (`iris:mtmc:next_gid`), so the app, Redis streams and the UI trucks keep
    their history and no id is reused. Only containers of the `scale` compose project are touched.
    """
    sh(f"cd {DEST} && {DC} restart deepstream-sidecar", timeout=180)
    with open(log_path, "a") as f:
        f.write(f"{time.strftime('%FT%T')} healed: restarted the sidecar (app, redis and UI trucks kept)\n")


def row(ports):
    fps = [float(v) for v in re.findall(r'iris_gauge\{name="fps_cam-\d+"\} ([0-9.]+)', get(f"http://127.0.0.1:{ports['telemetry']}/metrics"))]
    gpu = (sh("nvidia-smi --query-gpu=utilization.gpu,utilization.decoder,memory.used --format=csv,noheader,nounits").strip().split(",") + ["", "", ""])[:3]
    stats = {}
    for line in sh('docker stats --no-stream --format "{{.Name}} {{.CPUPerc}} {{.MemUsage}}"').splitlines():
        p = line.split()
        if p and p[0] in (SIDECAR, IRIS, UI):
            mem = re.match(r"([0-9.]+)([MG]iB)", p[2])
            stats[p[0]] = (float(p[1].rstrip("%")), float(mem.group(1)) * (1024 if mem.group(2) == "GiB" else 1) if mem else 0)
    try:
        streams = json.loads(get(f"http://127.0.0.1:{ports['go2rtc']}/api/streams") or "{}")
        pubs = sum(1 for k, v in streams.items() if k.startswith("raw-cam") and any(p.get("protocol") for p in (v.get("producers") or [])))
    except Exception:
        pubs = ""
    try:
        t = json.loads(get(f"http://127.0.0.1:{ports['iris']}/api/tracking/stats") or "{}")
    except Exception:
        t = {}
    restarts = sh("grep -c rejoining /tmp/scale13-streamer.log").strip() or "0"
    vlm = sh(f'docker logs --since 61s {SIDECAR} 2>&1 | grep -c "VLM ARBITER submitted"').strip() or "0"
    return [time.strftime("%Y-%m-%dT%H:%M:%S"), f"{min(fps):.1f}" if fps else "", f"{sum(fps)/len(fps):.1f}" if fps else "",
            stale_cameras(), gpu[0].strip(), gpu[1].strip(), gpu[2].strip(),
            stats.get(SIDECAR, ("", ""))[0], int(stats.get(SIDECAR, (0, 0))[1]), int(stats.get(IRIS, (0, 0))[1]), int(stats.get(UI, (0, 0))[1]),
            pubs, restarts, t.get("total_tracks", ""), t.get("total_transitions", ""), vlm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "stop"])
    ap.add_argument("--out", default=os.path.join(DEST, "data/scale13_metrics.csv"))
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--heal-after", type=int, default=0,
                    help="if a camera has produced no output for this many seconds, restart the tracking side (0 = off)")
    ap.add_argument("--heal-cooldown", type=int, default=1800)
    a = ap.parse_args()
    if a.action == "stop":
        try:
            os.kill(int(open(STATE).read()), signal.SIGTERM)
            print("stopped")
        except (OSError, ValueError):
            print("not running")
        return
    open(STATE, "w").write(str(os.getpid()))
    ports = {"telemetry": 39101, "go2rtc": 31984, "iris": 38000}
    head = ["time", "fps_min", "fps_avg", "cams_dead_30s", "gpu_util", "dec_util", "gpu_mem_mib", "sidecar_cpu",
            "sidecar_mem_mib", "iris_mem_mib", "ui_mem_mib", "publishers", "stream_restarts", "tracks", "transitions", "vlm_last_min"]
    new = not os.path.exists(a.out)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(head)
        bad_since, last_heal = None, 0.0
        heal_log = os.path.join(os.path.dirname(a.out), "scale13_heals.log")
        while not stop:
            r = row(ports)
            w.writerow(r)
            f.flush()
            if a.heal_after > 0:
                dead = r[3] if isinstance(r[3], int) else 0
                now = time.time()
                bad_since = (bad_since or now) if dead > 0 else None
                if bad_since and now - bad_since >= a.heal_after and now - last_heal >= a.heal_cooldown:
                    heal(heal_log)
                    last_heal, bad_since = time.time(), None
            for _ in range(a.every):
                if stop:
                    break
                time.sleep(1)


if __name__ == "__main__":
    main()
