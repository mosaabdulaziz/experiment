"""Spike-specific camera registry: cam id -> CHIRLA source filename (used for
wall-clock anchoring, see acl.wall_clock_from_stream_time). Not part of the
domain contract -- purely local wiring for ticket 02's CHIRLA seq_004 run.
"""

import json
from datetime import datetime
from pathlib import Path

CAM_FILENAMES = {
    # 2026-09-12: cam-1/cam-2/cam-3 are real live cameras (see sidecar.yaml --
    # they're wired to rtsp://go2rtc:8554/raw-cam-{1,2,3}, which go2rtc pulls
    # from real camera IPs, not a looping test file). cam-3's id happens to be
    # reused from this dict's old CHIRLA seq_004 slot (it used to ALSO be
    # "camera_5", one of the looping test recordings) -- that leftover
    # entry, and its CAM_DURATIONS twin below, caused a real bug: the
    # consumer's loop-wrap detector treated cam-3's real elapsed streaming
    # time modulo that old fake 284s "duration" as the file looping, and
    # wiped the ENTIRE attendance table every ~4.7 minutes even though
    # cam-3 was never looping at all (found live 2026-09-12: two unprompted
    # full-table wipes, logged as "cam-3 looped back to the start of its
    # recording", ~1m43s apart -- consistent with that 284s cycle). No
    # camera in this dict has a real "source filename" -- the value is only
    # ever used to look up an offline recording file, which real live
    # cameras don't have (server.py's /media endpoint 404s cleanly when the
    # file doesn't exist, which is correct: there's nothing to seek/scrub
    # for a live feed). cam-1/cam-2 are added here for the first time (they
    # weren't in this dict at all before -- meaning the consumer never
    # watched them, so no sighting from either could ever reach the
    # attendance table, regardless of how well recognition worked upstream)
    # so all three real cameras get watched on equal footing. None of the
    # three appear in CAM_DURATIONS below -- consumer.py's loop-wrap check
    # already skips any camera with no duration entry (`if duration is
    # None: continue`), which is exactly correct for a real live stream
    # that never loops.
    "cam-1":  "(no recording -- real live camera)",
    "cam-2":  "(no recording -- real live camera)",
    "cam-3":  "(no recording -- real live camera)",
    "cam-9":  "camera_1_2023-06-08-12-29-08.avi",
    "cam-10": "camera_2_2023-06-08-12-29-08.avi",
    "cam-11": "camera_3_2023-06-08-12-29-08.avi",
    "cam-12": "camera_4_2023-06-08-12-29-08.avi",
    "cam-13": "camera_6_2023-06-08-12-29-08.avi",
    "cam-14": "camera_7_2023-06-08-12-29-08.avi",
    # seq_020 (2026-09-01): enrollment-source sequence, streamed for
    # verification. Anchored to the same simulated RECORDING_START/SPIKE_DAY
    # as the seq_004 cams (its real date is 2023-07-26, but the dashboard's
    # daily view is pinned to SPIKE_DAY -- this is a simulation either way).
    "cam-15": "camera_4_2023-07-26-13-21-40.avi",
    "cam-16": "camera_5_2023-07-26-13-21-40.avi",
    "cam-17": "camera_6_2023-07-26-13-21-40.avi",
    "cam-18": "camera_3_2023-07-26-13-21-40.avi",   # open-plan desks, elevated view
    # 2026-09-10/11: real test clips (not CHIRLA), added so the consumer
    # actually watches them -- it only iterates CAM_FILENAMES (see
    # consumer.py), so a camera absent from this dict never produces a
    # sighting/attendance-table row no matter how well recognition itself
    # works. Filenames have no ".avi" to strip (see camera_recording()'s
    # ".replace(...)"), so they resolve to themselves under UPLOADS_DIR
    # unchanged -- these are the real uploaded filenames, verified to exist.
    "cam-20": "face_detection_test001-reception.mp4",  # 3min reception clip
    "cam-21": "face_detection_test001-elevator.mkv",   # 12.7min elevator clip, face-ID test
    "cam-22": "attendance/Bus/bus-cabin.mp4",          # 6s bus-cabin clip, people enrolled from the video itself
}

# Exact duration of each camera's file (ffprobe'd from the h264 upload, see
# PIPELINE.md). test_streamer loops these forever for the live simulation --
# used to mod real-elapsed-since-stream-start into "position in file".
CAM_DURATIONS = {
    # cam-1/cam-2/cam-3 deliberately absent -- see the comment on CAM_FILENAMES
    # above. They're real live cameras with no fixed duration to loop; giving
    # them one here is exactly the bug that caused the false table-wipes.
    "cam-9":  286.033333,
    "cam-10": 286.033333,
    "cam-11": 286.033333,
    "cam-12": 285.633333,
    "cam-13": 284.833333,
    "cam-14": 285.633333,
    "cam-15": 463.5,       # ffprobe'd from the h264 uploads (seq_020)
    "cam-16": 463.533333,
    "cam-17": 463.533333,
    "cam-18": 463.533333,
    "cam-20": 180.600000,   # ffprobe'd 2026-09-10
    "cam-21": 762.691000,   # ffprobe'd 2026-09-10
}

# 2026-09-12 (issue 1, Phase 1): CAM_RECORDING_START_OVERRIDES removed. It
# held a per-camera override of the old shared RECORDING_START/SPIKE_DAY
# anchor -- a fake "when was this file recorded" date used to compute
# "position inside a looping recording" (real-elapsed-since-stream-start,
# modulo the file's duration, offset onto that anchor). consumer.py and
# server.py no longer do that simulation at all: every sighting and every
# day-query now just uses real current time (see consumer.py's wall_clock
# and server.py's date.today() usages), so there is no anchor left to
# override, per-camera or otherwise. See ADJUSTMENTS_2026-09-12.md issue 1.

# test_streamer writes each camera's real ffmpeg start time here on every
# (re)start -- see stream_recordings.py's _record_start(). This is the
# authoritative clock for "position in file" (acl.wall_clock_from_stream_time);
# the sidecar's own pts is NOT reliable for this (see acl.py's docstring on
# wall_clock_for_spike for why).
# The repo's data/, two levels up since this folder moved to products/attendance/
# (it was one level up as attendance-service/). With the old depth this pointed at
# products/data/, the file was never found, and every replayed camera was skipped.
_STREAM_STARTS_FILE = Path(__file__).resolve().parents[2] / "data" / "stream_starts.json"


def load_stream_starts() -> dict[str, datetime]:
    """cam_id -> real datetime its current ffmpeg stream started. Re-read
    fresh on every call (cheap, small file) so callers always see the latest
    after a reconnect/restart. Missing file or missing cam_id -> just absent
    from the returned dict; callers should treat that as "not streaming yet"."""
    try:
        with open(_STREAM_STARTS_FILE) as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {cam_id: datetime.fromtimestamp(ts) for cam_id, ts in raw.items()}
