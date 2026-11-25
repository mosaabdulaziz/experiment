"""Build a preview mp4 from the sparse JPEG frames a search turn actually showed
the model, for streams with no recorded footage to cut from.

This is the fallback behind build_preview_clip: the local backend has no segment
recorder, the segment index can have no entry covering a FOUND window, and RTSP
sources under smart-record keep no continuous segments at all. In every case the
frames are already in hand -- they are what produced the FOUND, annotated by the
sidecar with boxes and track ids -- so a low-frame-rate clip beats no clip.

ffmpeg, not cv2.VideoWriter: cv2 writes mp4v (MPEG-4 Part 2), which browsers
largely refuse to play, and most opencv-python wheels carry no H.264 encoder.
That is fine for Analyzer._frames_to_mp4, whose output is decoded server-side by
vLLM, but this file goes into a <video controls> element. ffmpeg is already a
hard dependency of app/clip_builder.py.
"""
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def build_frames_clip(frames: list, fps: float, dest_dir: str,
                      basename: str = "frames") -> str | None:
    """Encode JPEG `frames` to a browser-playable H.264 mp4 at `fps`, under
    `dest_dir`. `basename` scopes the output so several cameras can build into the
    same directory. Returns the path, or None when there is nothing to encode or
    ffmpeg fails -- the caller then surfaces no clip, exactly as before."""
    if not frames:
        return None
    seq_dir = os.path.join(dest_dir, f"{basename}_seq")
    os.makedirs(seq_dir, exist_ok=True)
    for i, jpeg in enumerate(frames):
        with open(os.path.join(seq_dir, f"{i:05d}.jpg"), "wb") as fh:
            fh.write(jpeg)

    out_path = os.path.join(dest_dir, f"{basename}_frames.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", f"{max(float(fps), 1.0):g}",
        "-i", os.path.join(seq_dir, "%05d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        # yuv420p needs even dimensions; published frame sizes are whatever the
        # sidecar's capture_width/height produce, so round down rather than fail.
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-movflags", "+faststart",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        stderr = getattr(exc, "stderr", b"") or b""
        logger.warning("frames preview build failed (%d frame(s)): %s | %s",
                       len(frames), exc, stderr[-400:].decode("utf-8", "replace"))
        return None
    return out_path
