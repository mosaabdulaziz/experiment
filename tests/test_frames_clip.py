"""build_frames_clip: JPEG frames -> browser-playable mp4.

The fallback behind build_preview_clip, for streams with no recorded footage to
cut from. Codec matters here in a way it does not for the VLM path: this file is
played by a <video> element, so it must be H.264, not the mp4v that
Analyzer._frames_to_mp4 writes for vLLM's server-side decoder."""

import os
import subprocess

import cv2
import numpy as np

from app.frames_clip import build_frames_clip


def _jpeg(value: int, w: int = 64, h: int = 64) -> bytes:
    img = np.full((h, w, 3), value, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _probe(path: str, entries: str) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", entries,
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        check=True, capture_output=True, text=True,
    )
    return out.stdout.strip()


def _duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def test_builds_an_h264_mp4_at_the_requested_fps(tmp_path):
    frames = [_jpeg(v) for v in range(0, 160, 20)]      # 8 frames
    out = build_frames_clip(frames, fps=4, dest_dir=str(tmp_path))

    assert out is not None and os.path.exists(out)
    # H.264, not mp4v -- a browser <video> element has to play this.
    assert _probe(out, "stream=codec_name") == "h264"
    assert abs(_duration(out) - 2.0) < 0.5, "8 frames at 4fps should be ~2s"


def test_odd_sized_frames_do_not_fail_the_encode(tmp_path):
    # yuv420p requires even dimensions; an odd-sized published frame must not
    # sink the clip.
    frames = [_jpeg(v, w=65, h=33) for v in range(0, 120, 20)]
    out = build_frames_clip(frames, fps=4, dest_dir=str(tmp_path))
    assert out is not None and os.path.exists(out)
    assert _probe(out, "stream=width") == "64"
    assert _probe(out, "stream=height") == "32"


def test_no_frames_returns_none(tmp_path):
    assert build_frames_clip([], fps=4, dest_dir=str(tmp_path)) is None


def test_undecodable_frames_return_none_rather_than_raising(tmp_path):
    assert build_frames_clip([b"not a jpeg", b"also not"], fps=4,
                             dest_dir=str(tmp_path)) is None


def test_basename_scopes_the_output_so_two_cameras_do_not_collide(tmp_path):
    frames = [_jpeg(v) for v in range(0, 80, 20)]
    a = build_frames_clip(frames, fps=4, dest_dir=str(tmp_path), basename="cam0")
    b = build_frames_clip(frames, fps=4, dest_dir=str(tmp_path), basename="cam1")
    assert a != b
    assert os.path.exists(a) and os.path.exists(b)


def test_fps_below_one_is_floored_rather_than_producing_an_invalid_rate(tmp_path):
    frames = [_jpeg(v) for v in range(0, 80, 20)]      # 4 frames
    out = build_frames_clip(frames, fps=0, dest_dir=str(tmp_path))
    assert out is not None
    assert abs(_duration(out) - 4.0) < 0.5, "floored to 1fps, so 4 frames is ~4s"
