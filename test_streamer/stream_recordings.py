#!/usr/bin/env python3
"""Publish local recorded videos as live RTSP streams through go2rtc.

that points at rtsp://go2rtc:8554/<cam-id> gets its underlying upload file
(see the CAM_FILES mapping below) looped with ffmpeg and published to go2rtc's
RTSP ingest as raw-<cam-id>, so the DeepStream sidecar consumes it over the
same RTSP path a real camera uses -- including Smart Video Record
(recorder: auto -> rtsp:// => svr).

Codecs are published with `-c copy` (originals preserved: High-profile H.264 /
HEVC) because go2rtc transcodes to browser-safe H.264 on demand for WebRTC
viewers (go2rtc.yaml `ffmpeg:` variants) -- the sidecar and SVR recordings keep
full quality. Pass --h264-baseline only if you run a plain push-only server
(no on-demand transcode) and need browsers to play the pushed stream directly.

Usage:
  python3 test_streamer/stream_recordings.py                stream every cam in CAM_FILES
  python3 test_streamer/stream_recordings.py --profile multicam   the 13 MTMC cams (or: vlm)
  python3 test_streamer/stream_recordings.py --ids cam-1,cam-4   only those cams
  python3 test_streamer/stream_recordings.py --host 0.0.0.0 --port 8554

Notes:
- ffmpeg reads the file at native rate (-re) and loops it forever
  (-stream_loop -1), so the RTSP stream behaves like a live camera.
- A dead ffmpeg is restarted automatically after a short backoff.
- Synced groups (GROUPED_STREAMS below, e.g. barber-c/l/r) always restart
  together as a group, from a freshly shared start timestamp, even if only
  one member died -- restarting a single member alone desyncs the group's
  shared clip clock (see build_group_map / group_needs_restart). Send
  SIGUSR1 to this process to force a resync of any running synced group on
  demand (`just barber-streamer-restart`), without restarting cams that
  aren't part of a synced group.
- A group (or a lone cam) that keeps dying right after each restart backs
  off exponentially (2s, 4s, 8s, ... capped at 60s) instead of hammering a
  restart on every ~2s poll tick -- see backoff_seconds / next_group_action.
  A group resets back to no backoff once it's stayed up 60s. While backing
  off, still-healthy siblings in the same group are left running, not
  killed on every tick.
- Ctrl-C / SIGTERM stops all children cleanly.
- go2rtc is started separately:  docker compose up -d go2rtc
- Browser transcodes are LAZY (go2rtc.yaml): they start only while a viewer
  needs them and stop when idle, so nothing runs when nobody is watching.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS_DIR = os.path.join(REPO, "data", "uploads")
# Real (unix) start time per cam-id, updated every (re)start. products/attendance
# reads this to compute "position in file" from real elapsed time instead of
# the sidecar's own pts -- which does NOT reset on an upstream RTSP reconnect
# (the sidecar container keeps running, so its pts just keeps accumulating,
# independently per source depending on each one's own reconnect history --
# found the hard way, see products/attendance/PIPELINE.md). Real elapsed time
# since ffmpeg -re started is authoritative regardless of sidecar-side hiccups.
STREAM_STARTS_FILE = os.path.join(REPO, "data", "stream_starts.json")

# Camera groups whose ffmpeg children must be restarted together: these are
# synchronized recordings of the same scene (see barber-c/l/r in CAM_FILES),
# so restarting one child alone desyncs the group's shared clip clock --
# found live 2026-09-25: a lone ffmpeg restart left barber-r 97.6s behind
# barber-c/l, causing two engine-reset loop wraps per lap instead of one
# (products/barber's consumer.py _handle_loop_wraps assumes "the three clips
# loop together"). A group is only synced when >=2 of its members are part
# of THIS run's --ids; a lone member restarts on its own like any other cam.
GROUPED_STREAMS = (
    ("barber-c", "barber-l", "barber-r"),
    tuple(f"ph-{i}" for i in range(1, 8)),
)


def build_group_map(ids, grouped_streams=GROUPED_STREAMS):
    """cam_id -> the tuple of cam ids (including itself) that must restart
    together. A cam id not part of a known group, or whose group has fewer
    than 2 members in `ids`, maps to its own 1-tuple (restarts alone, same
    as before this existed)."""
    ids_set = set(ids)
    group_map = {}
    for group in grouped_streams:
        present = tuple(cid for cid in group if cid in ids_set)
        if len(present) >= 2:
            for cid in present:
                group_map[cid] = present
    for cid in ids:
        group_map.setdefault(cid, (cid,))
    return group_map


def group_needs_restart(group_ids, procs) -> bool:
    """True if any member of the group hasn't been started yet or has
    exited. The whole group restarts together (not just the dead member) --
    that's what keeps a synced group's shared clip clock correct."""
    return any(cid not in procs or procs[cid].poll() is not None for cid in group_ids)


# Backoff for a group that keeps failing right after each restart (bad
# file, go2rtc down, ...) -- found in review round 1: without this, a
# member that dies immediately makes every ~2s poll tick restart the WHOLE
# group forever, killing its healthy siblings too on every tick. 2s, 4s,
# 8s, ... capped at BACKOFF_CAP_S; a group that stays up
# RESET_AFTER_UPTIME_S resets back to no backoff.
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 60.0
RESET_AFTER_UPTIME_S = 60.0


def backoff_seconds(consecutive_failures: int, base: float = BACKOFF_BASE_S,
                     cap: float = BACKOFF_CAP_S) -> float:
    """How long to wait before the NEXT restart attempt, given how many
    consecutive times in a row the group has already failed (died again
    before staying up RESET_AFTER_UPTIME_S). 0 (or fewer) -> no wait (the
    very first failure always restarts immediately; only a repeat failure
    is backed off): 1 -> base, 2 -> 2*base, 3 -> 4*base, ... capped at
    `cap`."""
    if consecutive_failures <= 0:
        return 0.0
    return min(base * (2 ** (consecutive_failures - 1)), cap)


def next_group_action(state: dict, now: float, needs_restart: bool, forced: bool):
    """Pure per-tick decision for one group's restart/backoff state machine.

    state: {"failures": int, "started_at": float | None,
            "next_allowed_ts": float | None} -- caller holds one of these
    per group across ticks and feeds back the returned state next time.

    Returns (action, new_state) where action is one of:
      "start" -- (re)start the group now: the very first start, an
                 on-demand (forced) resync, or a failure whose backoff has
                 elapsed. `new_state["started_at"] == now` in this case.
      "wait"  -- a member is down but still inside its backoff window;
                 do nothing (don't touch the still-healthy siblings).
      "idle"  -- the group is healthy; do nothing (and, once it's been up
                 RESET_AFTER_UPTIME_S, `failures` resets to 0)."""
    state = dict(state)

    if forced:
        state["failures"] = 0
        state["next_allowed_ts"] = None
        state["started_at"] = now
        return "start", state

    if not needs_restart:
        if state["failures"] and state["started_at"] is not None \
                and now - state["started_at"] >= RESET_AFTER_UPTIME_S:
            state["failures"] = 0
            state["next_allowed_ts"] = None
        return "idle", state

    if state["started_at"] is None:
        # never started before -- always immediate, no backoff.
        state["started_at"] = now
        return "start", state

    if state["next_allowed_ts"] is None:
        # first tick this failure was observed -- schedule its backoff wait
        # (based on how many consecutive failures preceded it).
        state["next_allowed_ts"] = now + backoff_seconds(state["failures"])
    if now < state["next_allowed_ts"]:
        return "wait", state

    state["failures"] += 1
    state["next_allowed_ts"] = None
    state["started_at"] = now
    return "start", state


def merge_group_start(starts: dict, group_ids, ts: float) -> dict:
    """New stream_starts.json content with every id in the group set to the
    SAME timestamp, so products/barber (load_stream_starts / consumer.py's
    _handle_loop_wraps) computes an identical clip position for all of
    them. Pure: returns a new dict, doesn't touch the file."""
    updated = dict(starts)
    for cid in group_ids:
        updated[cid] = ts
    return updated


def _record_start_many(cam_ids, ts: float | None = None) -> None:
    """Write the SAME start timestamp for every id in cam_ids in one atomic
    update (see merge_group_start)."""
    ts = time.time() if ts is None else ts
    try:
        with open(STREAM_STARTS_FILE) as f:
            starts = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        starts = {}
    starts = merge_group_start(starts, cam_ids, ts)
    tmp = STREAM_STARTS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(starts, f)
    os.replace(tmp, STREAM_STARTS_FILE)


def _prune_starts(active_ids) -> None:
    """Drop any cam-id from the starts file that isn't part of THIS
    invocation's --ids. Without this, restarting with a narrower camera set
    (e.g. `just up cam-9` after a run with all 7) leaves stale entries for
    the cameras no longer streaming -- found live: products/attendance kept
    reporting a "current time" for cameras that had actually gone dead,
    because their old entry was still sitting in the file, never cleared on
    a start with a different --ids. Call once at startup, before any cam's
    ffmpeg actually starts."""
    try:
        with open(STREAM_STARTS_FILE) as f:
            starts = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return
    pruned = {cam_id: ts for cam_id, ts in starts.items() if cam_id in active_ids}
    if pruned == starts:
        return
    tmp = STREAM_STARTS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(pruned, f)
    os.replace(tmp, STREAM_STARTS_FILE)

# cam-id -> file under data/uploads/, OR an absolute path to a file anywhere
# on the host (e.g. "/mnt/datasets/clip.mp4") if you don't want to copy it
# into the repo. Keep in sync with the rtsp:// sources in perception/pipeline/sidecar.yaml
# (only ids listed here are published).
# 2026-09-13: cam-1/2/3 are REAL live cameras now (go2rtc pulls them from the
# camera IPs) and cam-6 is go2rtc's copy of cam-3. Publishing a file into
# their raw-cam-N path gave go2rtc two different videos for one camera, which
# froze the sidecar's streams ~26 min in. So they're refused here -- see
# LIVE_CAMS below. Old test files, for reference:
#   cam-1: "Red light camera footage [QwueyJr-El4].mp4"
#   cam-2: "YTDown.com_Shorts_Two-Cars-Go-Straight-Through-a-Red-Traff_Media_I01ORKkV-C4_001_720p.mp4"
#   cam-3: "camera_5_2023-06-08-12-29-08_h264.mp4"
#   cam-6: "/data/IRIS/repo/surveillance-camera-sys/data/uploads/fire/wildfire1.mp4"
LIVE_CAMS = {"cam-1", "cam-2", "cam-3", "cam-6"}

CAM_FILES = {
    "cam-4": "attendance/camera_3_2023-06-08-12-29-08_h264.mp4",
    "cam-5": "attendance/camera_6_2023-06-08-12-29-08_h264.mp4",
    "cam-7": "/data/IRIS/repo/surveillance-camera-sys/data/uploads/fire/part1/bucketfire.mp4",
    "cam-8": "/data/IRIS/repo/surveillance-camera-sys/data/uploads/fire/part2/roomfire.mp4",
    # CHIRLA seq_004, all 7 cameras of one sequence (products/attendance spike,
    # ticket 02 — cam-3/camera_5 above is also part of this same sequence).
    "cam-9":  "attendance/camera_1_2023-06-08-12-29-08_h264.mp4",
    "cam-10": "attendance/camera_2_2023-06-08-12-29-08_h264.mp4",
    "cam-11": "attendance/camera_3_2023-06-08-12-29-08_h264.mp4",
    "cam-12": "attendance/camera_4_2023-06-08-12-29-08_h264.mp4",
    "cam-13": "attendance/camera_6_2023-06-08-12-29-08_h264.mp4",
    "cam-14": "attendance/camera_7_2023-06-08-12-29-08_h264.mp4",
    # CHIRLA seq_020 (2026-09-01): camera_4 is the video the 13-person
    # roster's multi-embedding enrollment was built from; camera_5/6 are the
    # same synchronized scene from other angles.
    "cam-15": "attendance/camera_4_2023-07-26-13-21-40_h264.mp4",
    "cam-16": "attendance/camera_5_2023-07-26-13-21-40_h264.mp4",
    "cam-17": "attendance/camera_6_2023-07-26-13-21-40_h264.mp4",
    # camera_3 = the open-plan room from an elevated angle: ~6 seats that
    # don't stack behind each other -- the desk-monitoring camera (2026-09-01).
    "cam-18": "attendance/camera_3_2023-07-26-13-21-40_h264.mp4",
    # 2026-09-10: real ~34min recording (4K HEVC, 13fps) for checking
    # face/person detection quality against actual camera footage rather
    # than the CHIRLA dataset.
    "cam-19": "output.mkv",
    # 2026-09-10: 3min reception clip, real footage of people moving through
    # the reception area -- for checking face/person bbox quality.
    "cam-20": "face_detection_test001-reception.mp4",
    # 2026-09-10: 12.7min elevator clip -- face-ID accuracy test against
    # emp1/2/4/5/7/8, each with known appearance windows (see prompt.md).
    "cam-21": "face_detection_test001-elevator.mkv",
    # 2026-11-02: 6 s bus-cabin clip (3 people, near/mid/far) for attendance with enrolment taken from the video
    # itself (`just attendance-enroll-video`, tests/configs/attendance-bus.yaml).
    "cam-22": "attendance/Bus/bus-cabin.mp4",
    # 2026-09-14: barber shop test footage (YouTube 9Hwju1yZBAM, 720p H.264,
    # 31:57) for the barber use case -- published as raw-barber-1, read by
    # the barber sidecar (products/barber/sidecar.yaml). Not in git: download
    # it into data/uploads/barber/ (see products/barber/README.md).
    "barber-1": "barber/9Hwju1yZBAM.mp4",
    # 2026-09-27: the full 39-min synced footage (09-03 12:07-12:46) is data/uploads/barber/bekky-39min/;
    # 2026-10-07: the testbed runs the 109-min synced night footage (1080p, 15 fps; same camera views,
    # the sofa moved -- still inside the waiting zone). Keep products/barber/barber.yaml's clip_duration_s in step.
    "barber-c": "/data/IRIS/data/BekkyBarber/train2/synced3/center.mp4",
    "barber-l": "/data/IRIS/data/BekkyBarber/train2/synced3/left.mp4",
    "barber-r": "/data/IRIS/data/BekkyBarber/train2/synced3/right.mp4",
    # Bekky Palm Hills ("branch 2", 2026-10-14): the 2 h synced recording of 2026-10-10 19:00-21:00, 7 cameras
    # (products/barber/barber.palm-hills.yaml). HEVC 720p, 20 fps (Cam2 25 fps); published with -c copy.
    "ph-1": "/data/IRIS/data/BekkyBarber/palm_hills/Cam1_Shampoo-Wash-Area_2026-10-10_19-00-00_to_21-00-00_SYNCED.mp4",
    "ph-2": "/data/IRIS/data/BekkyBarber/palm_hills/Cam2_Barber-Stations-Wide_2026-10-10_19-00-00_to_21-00-00_SYNCED.mp4",
    "ph-3": "/data/IRIS/data/BekkyBarber/palm_hills/Cam3_Reception-Desk-Front_2026-10-10_19-00-00_to_21-00-00_SYNCED.mp4",
    "ph-4": "/data/IRIS/data/BekkyBarber/palm_hills/Cam4_Barber-Stations-Close_2026-10-10_19-00-00_to_21-00-00_SYNCED.mp4",
    "ph-5": "/data/IRIS/data/BekkyBarber/palm_hills/Cam5_Reception-Lounge_2026-10-10_19-00-00_to_21-00-00_SYNCED.mp4",
    "ph-6": "/data/IRIS/data/BekkyBarber/palm_hills/Cam6_Pedicure-Area_2026-10-10_19-00-00_to_21-00-00_SYNCED.mp4",
    "ph-7": "/data/IRIS/data/BekkyBarber/palm_hills/Cam7_Barber-Chairs-Glass-Wall_2026-10-10_19-00-00_to_21-00-00_SYNCED.mp4",
}

# cam-id -> file under data/uploads/ (use-case/client folders: attendance/, el-mohandes/, fire/ ...) or products/el-mohandes/videos/.
CAM_FILES_MULTICAM = {
    "cam-1": "el-mohandes/1-D41-enter.mp4",
    "cam-2": "el-mohandes/2-D16_clip.mp4",
    "cam-3": "el-mohandes/3-D52-enter.mp4",
    "cam-4": "el-mohandes/4-ch05_clip.mp4",
    "cam-5": "el-mohandes/5-CH23-enter.mp4",
    "cam-6":  "el-mohandes/D01-crop.mp4",
    "cam-7":  "el-mohandes/D05-crop.mp4",
    "cam-8":  "el-mohandes/D09-crop.mp4",
    "cam-9":  "el-mohandes/D11-crop.mp4",
    "cam-10": "el-mohandes/D15-crop.mp4",
    "cam-11": "el-mohandes/D18-crop.mp4",
    "cam-12": "el-mohandes/D20-crop.mp4",
    "cam-13": "el-mohandes/D21-crop.mp4",
}

CAM_FILES_VLM = {
    "cam-1": "D05_20260727144246_clip.mp4",
}

# --profile -> cam map. attendance (the default) is CAM_FILES above.
PROFILE_CAM_FILES = {
    "attendance": CAM_FILES,
    "multicam": CAM_FILES_MULTICAM,
    "vlm": CAM_FILES_VLM,
}


def _missing_file_message(cid, path):
    """A clip that is versioned with DVC but not pulled yet says so, and how to get it (DATA.md)."""
    if os.path.isfile(path + ".dvc"):
        rel = os.path.relpath(path + ".dvc", REPO)
        return (f"missing file for {cid}: {path}\n"
                f"  it is versioned with DVC but not pulled: `dvc pull {rel}`, or `just data-pull <group>` "
                f"(groups: data/catalog.yaml, DATA.md)")
    return f"missing file for {cid}: {path}"


def _resolve_path(value):
    """A CAM_FILES value is either a path under data/uploads/ (e.g. attendance/x.mp4) or a bare name in products/el-mohandes/videos/,
    or already an absolute path to a file elsewhere on the host.

    An absolute path that does not exist on THIS machine (CAM_FILES carries the shared server's
    /data/IRIS/... locations) is looked up by its file name under data/uploads/ (any depth) and
    products/el-mohandes/videos/, so the same clip works from a checkout that keeps it under data/uploads/fire/."""
    if os.path.isabs(value):
        if os.path.isfile(value):
            return value
        name = os.path.basename(value)
        for root in (UPLOADS_DIR, os.path.join(REPO, "mtmc", "videos")):
            for dirpath, _dirs, files in os.walk(root):
                if name in files:
                    return os.path.join(dirpath, name)
        return value
    uploads_path = os.path.join(UPLOADS_DIR, value)
    if os.path.isfile(uploads_path):
        return uploads_path
    mtmc_path = os.path.join(REPO, "mtmc", "videos", value)
    if os.path.isfile(mtmc_path):
        return mtmc_path
    # A bare name from before the use-case folders (attendance/, multicam/, ...): find it by name at any depth.
    name = os.path.basename(value)
    for dirpath, _dirs, files in os.walk(UPLOADS_DIR):
        if name in files:
            return os.path.join(dirpath, name)
    return uploads_path


def _detect_codec(path):
    """ffprobe one file -> codec_name or None."""
    try:
        import json as _json
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "json=path=0"],
            stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        return _json.loads(out)["streams"][0]["codec_name"]
    except Exception:
        return None


def _normalize_cam_id(cid: str) -> str:
    """Normalize cam-01 -> cam-1, raw-cam-01 -> cam-1, etc."""
    c = cid.strip()
    if c.startswith("raw-"):
        c = c[4:]
    if c.startswith("cam-") and len(c) > 4 and c[4:].isdigit():
        return f"cam-{int(c[4:])}"
    return c


def parse_files(arg):
    """--files "ph-1=/path/a.mp4,ph-3=/path/b.mp4": other recordings for these ids, over CAM_FILES (2026-10-18:
    a barber site plays the 16 Oct synced day as ph-*, the same ids and zones as the 10 Oct clips)."""
    out = {}
    for item in (arg or "").split(","):
        item = item.strip()
        if not item:
            continue
        cam, sep, path = item.partition("=")
        if not sep or not cam.strip() or not path.strip():
            raise ValueError(f"--files: {item!r} is not id=path")
        out[cam.strip()] = path.strip()
    return out


def _ffmpeg_cmd(cam_id, path, host, port, h264_baseline):
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-re", "-stream_loop", "-1", "-i", path,
    ]
    if h264_baseline:
        # Baseline profile has no B-frames, so every browser can play it over
        # WebRTC. HEVC sources (e.g. exit-1.mp4) MUST be transcoded here. Only
        # needed when the server does NOT transcode on demand (see docstring).
        cmd += ["-c:v", "libx264", "-preset", "ultrafast",
                "-tune", "zerolatency", "-profile:v", "baseline",
                "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c", "copy"]
    # RTSP output flags:
    #  -rtsp_transport tcp : interleaved RTP-over-TCP (no UDP packet loss)
    #  -max_muxing_queue_size : prevent "av_interleaved_write_frame queue"
    #    overflow when the RTSP server is slow to accept packets
    #  -flush_packets 1 : flush muxer buffers immediately for lower latency
    cmd += ["-max_muxing_queue_size", "1024",
            "-flush_packets", "1",
            "-rtsp_transport", "tcp", "-f", "rtsp",
            f"rtsp://{host}:{port}/raw-{cam_id}"]
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROFILE_CAM_FILES), default="attendance",
                    help="stream profile: attendance (CAM_FILES), multicam (13 cams) or "
                         "vlm (single cam-1) [default: attendance]")
    ap.add_argument("--ids", default=None,
                    help="comma-separated cam ids to stream (e.g. cam-1,cam-02)")
    ap.add_argument("--host", default="localhost", help="go2rtc host (default localhost)")
    ap.add_argument("--port", default=8554, type=int, help="go2rtc RTSP port (default 8554)")
    ap.add_argument("--h264-baseline", action="store_true",
                    help="re-encode to H.264 baseline (no B-frames) so browsers can "
                         "play the streams over WebRTC raw previews")
    ap.add_argument("--pid-file", default=None, help="write streamer PID to this file")
    ap.add_argument("--files", default="",
                    help='other recordings for some ids, "id=path,id=path" (over CAM_FILES)')
    args = ap.parse_args()
    try:
        CAM_FILES.update(parse_files(args.files))
    except ValueError as exc:
        sys.exit(str(exc))

    active_cam_files = PROFILE_CAM_FILES[args.profile]

    if args.ids:
        raw_ids = [i.strip() for i in args.ids.split(",") if i.strip()]
        ids = [_normalize_cam_id(i) for i in raw_ids]
    else:
        ids = list(active_cam_files)
    if args.profile == "attendance":
        live = [i for i in ids if i in LIVE_CAMS]
        if live:
            sys.exit(f"{', '.join(live)}: live camera(s) -- the test streamer can't play on them "
                     f"(use `just up live` to run the real cameras without any test video)")
    missing = [i for i in ids if i not in active_cam_files]
    if missing:
        sys.exit(f"unknown cam ids for profile '{args.profile}': {missing}; known: {sorted(active_cam_files)}")

    if args.pid_file:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.pid_file)), exist_ok=True)
            with open(args.pid_file, "w") as f:
                f.write(str(os.getpid()))
        except Exception as e:
            print(f"warning: failed to write pid file {args.pid_file}: {e}", file=sys.stderr)

    # Drop any stale entries from a previous run with a wider --ids (e.g. all
    # 7 cams) before this run's ffmpegs start writing their own. Otherwise
    # products/attendance keeps reporting a "current time" for cameras that
    # aren't actually streaming anymore.
    _prune_starts(ids)

    group_map = build_group_map(ids)
    backoff_state = {}  # group tuple -> {"failures", "started_at", "next_allowed_ts"}

    def _default_backoff_state():
        return {"failures": 0, "started_at": None, "next_allowed_ts": None}

    procs = {}  # cam_id -> Popen
    stopped = {"flag": False}
    resync = {"flag": False}

    def _stop(_sig=None, _frame=None):
        stopped["flag"] = True
        for p in procs.values():
            try:
                p.terminate()
            except Exception:
                pass
        if args.pid_file and os.path.exists(args.pid_file):
            try:
                os.remove(args.pid_file)
            except Exception:
                pass

    def _request_resync(_sig=None, _frame=None):
        resync["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGUSR1, _request_resync)

    print(f"publishing {len(ids)} stream(s) to rtsp://{args.host}:{args.port}/<cam-id>"
          + (" (H.264 baseline transcode)" if args.h264_baseline else ""))
    hevc_warned = False
    for cid in ids:
        path = _resolve_path(active_cam_files[cid])
        if not os.path.isfile(path):
            sys.exit(_missing_file_message(cid, path))
        codec = _detect_codec(path)
        tag = f" [{codec}]" if codec else ""
        print(f"  {cid}: {active_cam_files[cid]}{tag}")
        if codec in ("hevc", "h265") and not args.h264_baseline and not hevc_warned:
            print(f"  ⚠  HEVC source detected — go2rtc relay may produce corrupted "
                  f"frames on some cameras. Consider --h264-baseline or switching "
                  f"to file:// URIs in sidecar.yaml.", file=sys.stderr)
            hevc_warned = True

    while not stopped["flag"]:
        forced = resync["flag"]
        resync["flag"] = False
        now = time.time()
        handled = set()
        for cid in ids:
            if stopped["flag"]:
                break
            group = group_map[cid]
            if group in handled:
                continue
            handled.add(group)

            force_this_group = forced and len(group) > 1
            needs_restart = group_needs_restart(group, procs)
            state = backoff_state.get(group, _default_backoff_state())
            had_pending_backoff = state["next_allowed_ts"] is not None
            action, state = next_group_action(state, now, needs_restart, force_this_group)
            backoff_state[group] = state
            if action == "idle":
                continue
            label = "/".join(group)
            if action == "wait":
                if not had_pending_backoff:
                    # just entered this backoff window this tick -- log once,
                    # not on every ~2s poll while still waiting.
                    wait_s = state["next_allowed_ts"] - now
                    print(f"[{label}] member(s) down, backing off {wait_s:.0f}s before "
                          f"restart (consecutive failure {state['failures'] + 1})", flush=True)
                continue

            # action == "start"
            exited = {m: procs[m].returncode for m in group
                      if m in procs and procs[m].poll() is not None}
            if force_this_group:
                reason = "on-demand resync"
            elif len(group) == 1 and exited:
                (only,) = group
                reason = f"ffmpeg exited rc={exited[only]}, restarting"
            elif len(group) == 1:
                reason = "restarting"
            elif exited:
                rcs = ", ".join(f"{m} rc={rc}" for m, rc in exited.items())
                reason = f"{rcs} exited, resyncing group"
            else:
                reason = "resyncing group"
            # Stop any still-alive members of the group first (a lone dead
            # member already exited on its own) so every member starts fresh
            # together -- restarting only the dead one is exactly what
            # desyncs a synced group's shared clip clock.
            for member in group:
                p = procs.get(member)
                if p is not None and p.poll() is None:
                    try:
                        p.terminate()
                        p.wait(timeout=5)
                    except Exception:
                        try:
                            p.kill()
                            p.wait(timeout=5)
                        except Exception:
                            pass
            print(f"[{label}] {reason}", flush=True)
            ts = now
            for member in group:
                path = _resolve_path(active_cam_files[member])
                procs[member] = subprocess.Popen(
                    _ffmpeg_cmd(member, path, args.host, args.port, args.h264_baseline),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"[{member}] ffmpeg started (pid {procs[member].pid})", flush=True)
            _record_start_many(group, ts)
        time.sleep(2)

    for cid, p in procs.items():
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()
    print("all streamers stopped")


if __name__ == "__main__":
    main()
