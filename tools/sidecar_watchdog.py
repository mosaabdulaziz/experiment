#!/usr/bin/env python3
"""Restart the DeepStream sidecar when a camera's pipeline is stuck (2026-09-13).

Why: the sidecar's RTSP reconnect can wedge -- it keeps "resetting" a source on a
connection that is already closed and never recovers, while the camera is still
reachable through go2rtc (checked live 2026-09-13 with an independent ffmpeg
pull). Only a container restart clears it. Without this, a stall lasted until
someone noticed (once ~1 hour).

Rule, checked every CHECK_SECONDS:
  * the sidecar container has been up at least STARTUP_GRACE_SECONDS, and
  * a camera enabled in sidecar.yaml has published no frame record
    (iris:meta:<cam>) for more than STALE_SECONDS, and
  * that camera still delivers video through go2rtc -- tested by pulling it
    for PROBE_SECONDS with ffmpeg. go2rtc's own byte counters can't be used:
    go2rtc only pulls a camera while a viewer is attached, and the wedged
    sidecar isn't one, so the counter sits still even when the camera is fine.
    If the camera itself is offline a restart can't help, so none is done, and
  * the last restart was at least MIN_RESTART_GAP_SECONDS ago.
Then: `docker restart` the sidecar container. Each decision is logged.

Run from the repo root (justfile starts it with `just up`):
  products/attendance/.venv/bin/python tools/sidecar_watchdog.py [--dry-run] [--once]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import redis

REPO_ROOT = Path(__file__).resolve().parents[1]
# The pipeline's own config loader, so `extends:` overlays resolve exactly as
# the sidecar resolves them.
sys.path.insert(0, str(REPO_ROOT))
from perception.engine.sources import load_sidecar_config  # noqa: E402

CHECK_SECONDS = 10
STALE_SECONDS = 60
STARTUP_GRACE_SECONDS = 90
MIN_RESTART_GAP_SECONDS = 300
PROBE_SECONDS = 3
SERVICE = "deepstream-sidecar"
# The shared base config: it declares no cameras of its own (products do).
BASE_SIDECAR_YAML = "perception/pipeline/sidecar.yaml"


def sidecar_config_path() -> Path:
    """The config the pipeline is actually running: SIDECAR_CONFIG, the same
    variable docker-compose hands the sidecar, falling back to the base file.

    2026-09-24: this was hardcoded to the base file. The restructure moved the
    cameras into each product's overlay, so the base lists none -- the watchdog
    watched zero cameras, never restarted anything, and a wedged pipeline sat
    frozen for an hour on production's first day in the new layout."""
    path = Path(os.environ.get("SIDECAR_CONFIG") or BASE_SIDECAR_YAML)
    return path if path.is_absolute() else REPO_ROOT / path


def log(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def enabled_cameras(path: Path) -> dict[str, str]:
    """{cam id: go2rtc stream name} for every source enabled in sidecar.yaml
    that is pulled through go2rtc (rtsp://go2rtc:8554/<stream>)."""
    cfg = load_sidecar_config(str(path))
    cams = {}
    for s in cfg.get("sources") or []:
        uri = str(s.get("uri", ""))
        if uri.startswith("rtsp://go2rtc:"):
            cams[s["id"]] = uri.rsplit("/", 1)[-1]
    return cams


def container_id() -> str | None:
    out = subprocess.run(["docker", "compose", "ps", "-q", SERVICE],
                         capture_output=True, text=True).stdout.strip()
    return out.splitlines()[0] if out else None


def uptime_seconds(cid: str) -> float | None:
    out = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}} {{.State.StartedAt}}", cid],
                         capture_output=True, text=True).stdout.strip()
    if not out.startswith("true "):
        return None
    started = datetime.fromisoformat(out.split(" ", 1)[1][:26].rstrip("Z") + "+00:00")
    return (datetime.now(timezone.utc) - started).total_seconds()


def meta_age_seconds(r: redis.Redis, cam: str) -> float | None:
    tip = r.xrevrange(f"iris:meta:{cam}", count=1)
    if not tip:
        return None
    return time.time() - int(tip[0][0].split(b"-")[0]) / 1000


def frames_via_go2rtc(stream: str, rtsp_port: str) -> int:
    """Pull the camera through go2rtc for PROBE_SECONDS and count video frames.
    Adds no load on the camera itself: go2rtc shares its one upstream pull."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp",
             "-i", f"rtsp://127.0.0.1:{rtsp_port}/{stream}", "-t", str(PROBE_SECONDS),
             "-an", "-f", "null", "-", "-progress", "pipe:1"],
            capture_output=True, text=True, timeout=PROBE_SECONDS + 15).stdout
    except subprocess.TimeoutExpired:
        return 0
    counts = re.findall(r"^frame=(\d+)", out, re.M)
    return int(counts[-1]) if counts else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="log what would be restarted, restart nothing")
    ap.add_argument("--once", action="store_true", help="run one check, then exit")
    args = ap.parse_args()

    redis_url = os.environ.get("REDIS_URL", f"redis://localhost:{os.environ.get('REDIS_HOST_PORT', '6379')}/0")
    rtsp_port = os.environ.get("GO2RTC_RTSP_PORT", "8554")
    r = redis.Redis.from_url(redis_url)
    last_restart = 0.0
    log(f"watchdog started (stale>{STALE_SECONDS}s, grace {STARTUP_GRACE_SECONDS}s, "
        f"min gap {MIN_RESTART_GAP_SECONDS}s, dry_run={args.dry_run})")
    # Say what is being watched: a watchdog with nothing to watch runs
    # quietly forever, which is how the 2026-09-24 hour-long stall went unseen.
    config = sidecar_config_path()
    cams = enabled_cameras(config)
    shown = config.relative_to(REPO_ROOT) if config.is_relative_to(REPO_ROOT) else config
    log(f"watching {len(cams)} camera(s) from {shown}: "
        + (", ".join(f"{c}->{s}" for c, s in sorted(cams.items())) or "NONE"))
    if not cams:
        log("WARNING: no go2rtc cameras in that config -- nothing will ever be restarted")

    while True:
        try:
            cid = container_id()
            up = uptime_seconds(cid) if cid else None
            if up is not None and up >= STARTUP_GRACE_SECONDS:
                stuck = []
                for cam, stream in enabled_cameras(sidecar_config_path()).items():
                    age = meta_age_seconds(r, cam)
                    age = up if age is None else age
                    if age <= STALE_SECONDS:
                        continue
                    frames = frames_via_go2rtc(stream, rtsp_port)
                    if frames > 0:
                        stuck.append(f"{cam} (no frames in pipeline for {age:.0f}s, "
                                     f"camera OK: {frames} frames in {PROBE_SECONDS}s via go2rtc)")
                    else:
                        log(f"{cam}: no frames for {age:.0f}s and none via go2rtc either "
                            f"(camera/network down) -- not restarting")
                if stuck:
                    since = time.time() - last_restart
                    if since < MIN_RESTART_GAP_SECONDS:
                        log(f"stuck: {'; '.join(stuck)} -- last restart {since:.0f}s ago, waiting")
                    elif args.dry_run:
                        log(f"WOULD restart {SERVICE}: {'; '.join(stuck)}")
                        last_restart = time.time()
                    else:
                        log(f"restarting {SERVICE}: {'; '.join(stuck)}")
                        subprocess.run(["docker", "restart", cid], check=False)
                        last_restart = time.time()
        except Exception as e:  # keep watching through Redis/docker/yaml hiccups
            log(f"check failed: {e!r}")
        if args.once:
            return
        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    main()
