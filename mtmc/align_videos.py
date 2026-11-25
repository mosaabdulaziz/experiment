#!/usr/bin/env python3
"""Align a set of camera videos onto one shared timeline.

Every source video is a window of a longer per-camera recording. Each window
starts at a different wall-clock moment, so the same instant appears at a
different pts in every file and cross-camera time checks (MTMC gaps, ordering)
see wrong deltas. This tool trims every file to the *common span* of the
windows and re-encodes them so that:

  * pts starts at exactly 0.000 in every output (same start pts),
  * all outputs are constant frame rate (native fps per file),
  * every output covers the same wall-clock span, so the pipeline sees
    consistent relative timing across cameras.

The per-file offset is the wall-clock start of that file relative to a common
origin (can be negative: the file's content begins BEFORE the origin).

NO OFFSETS? Each video starts at the same origin as every other (offset 0):
all outputs are then cut from each file's own beginning to the length of the
shortest file. If the recordings really did start at (nearly) the same moment
this aligns them to the same timings — the common span is simply
[0 .. min(duration)]. Pass --offsets when you know the true start shifts to
trim each head exactly.

Usage:
    # no offsets: cut every file from its start to the shortest file's length
    .venv/bin/python -m mtmc.align_videos \\
        --dir data/uploads --out products/el-mohandes/videos/aligned

    # with known start offsets (negative = file starts before the origin):
    .venv/bin/python -m mtmc.align_videos \\
        --offsets 1-D41-enter.mp4=0.0,2-D16_clip.mp4=-3.2,3-D52-enter.mp4=1.1 \\
        --dir data/uploads --out products/el-mohandes/videos/aligned

    # normalize only (no trimming, no shifting): full duration, CFR, pts 0
    .venv/bin/python -m mtmc.align_videos --normalize \\
        --dir data/uploads --out products/el-mohandes/videos/aligned
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def probe(path: Path) -> dict:
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,codec_type",
        "-of", "json", str(path),
    ])
    data = json.loads(out)
    fmt = data["format"]
    vs = next(s for s in data["streams"] if s.get("codec_type") == "video")
    num, den = vs["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    return {
        "duration": float(fmt["duration"]),
        "codec": vs.get("codec_name"),
        "w": int(vs["width"]),
        "h": int(vs["height"]),
        "fps": fps,
    }


def ffmpeg_cut(in_path: Path, out_path: Path, start: float, nframes: int,
               fps: float, encoder: str = "x264", crf: int = 18,
               preset: str = "veryfast", max_w: int = 0) -> None:
    """Frame-accurate cut beginning at `start` for exactly `nframes` frames,
    re-encoded to CFR with pts starting at exactly 0.

    -ss before -i + re-encode = frame-accurate start (decodes and drops up to
    the target). -bf 0 (no B-frames) so the first written frame has pts 0 and
    there is no edit-list/reorder residue. -frames:v nframes gives every output
    the same wall-clock length to within one frame.
    """
    vf = f"fps={fps:.6f},setpts=PTS-STARTPTS"
    if max_w:
        vf += f",scale='min({max_w},iw)':-2"
    cmd = ["ffmpeg", "-hide_banner", "-y",
           "-ss", f"{start:.6f}", "-i", str(in_path),
           "-frames:v", str(nframes),
           "-map", "0:v:0", "-an",
           "-vf", vf,
           "-r", f"{fps:.6f}",
           "-avoid_negative_ts", "make_zero",
           "-video_track_timescale", "90000",
           "-movflags", "+faststart"]
    if encoder == "nvenc":
        cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
                "-cq", str(crf), "-b:v", "0", "-bf", "0"]
    else:
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-bf", "0"]
    cmd += ["-pix_fmt", "yuv420p", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {in_path.name}:\n{r.stderr[-2000:]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/uploads", help="directory with source videos")
    ap.add_argument("--files", default=None,
                    help="comma-separated subset of filenames (default: all *.mp4)")
    ap.add_argument("--offsets", default=None,
                    help="comma-separated file=seconds offsets (all files must be listed); "
                         "negative = file starts before the common origin. "
                         "Omit to use offset 0 for every file (cut from each file's "
                         "beginning to the shortest file's length)")
    ap.add_argument("--normalize", action="store_true",
                    help="no content shifting: full duration, CFR, pts starts at 0")
    ap.add_argument("--out", default="products/el-mohandes/videos/aligned",
                    help="output directory")
    ap.add_argument("--encoder", default="x264", choices=["x264", "nvenc"],
                    help="libx264 (CPU, default) or h264_nvenc (GPU, fast for 4K)")
    ap.add_argument("--crf", type=int, default=18,
                    help="x264 crf / nvenc cq (lower = better quality)")
    ap.add_argument("--preset", default="veryfast", help="x264 preset only")
    ap.add_argument("--max-w", type=int, default=0,
                    help="optionally downscale width to <= N px (height auto)")
    args = ap.parse_args()

    src = Path(args.dir)
    names = [n.strip() for n in args.files.split(",")] if args.files else \
        sorted(p.name for p in src.glob("*.mp4"))
    names = [n for n in names if (src / n).exists()]

    if not names:
        sys.exit("no source files found")

    offsets = {}
    if args.offsets:
        for item in args.offsets.split(","):
            tok = item.strip()
            name, _, val = tok.partition("=")
            if "=" not in tok:
                sys.exit(f"bad offset token: {tok!r} (expected file=seconds, e.g. 1-D41-enter.mp4=-3.2)")
            if not name:
                sys.exit(f"bad offset token: {tok!r} (missing filename before '=')")
            if not val:
                sys.exit(f"bad offset token: {tok!r} (missing value after '=')")
            try:
                num = float(val)
            except ValueError:
                sys.exit(f"bad offset for {name!r}: {val!r} is not a number")
            if not math.isfinite(num):
                sys.exit(f"bad offset for {name!r}: {val!r} is not a finite number")
            offsets[name] = num
        if names and not (set(offsets) & set(names)):
            sys.exit(f"offsets reference none of the selected files; "
                     f"expected e.g. {names[0]}=<seconds>,... "
                     f"(remove the trailing '...' placeholder if you pasted the example)")
        missing = [n for n in names if n not in offsets]
        if missing:
            sys.exit(f"missing offsets for: {missing}")
    else:
        print("note: no --offsets given — using offset 0 for every file "
              "(cut from each file's beginning to the shortest file's length)")
    if not args.normalize and not args.offsets:
        print("note: if the recordings did NOT all start at the same moment, pass --offsets "
              "to trim each head by its true shift (see --help)")

    info = {n: probe(src / n) for n in names}

    # Wall-clock window of each file: [offset, offset + duration]
    windows = {n: (offsets.get(n, 0.0),
                   offsets.get(n, 0.0) + info[n]["duration"]) for n in names}
    span_start = max(w[0] for w in windows.values())
    span_end = min(w[1] for w in windows.values())
    span_len = span_end - span_start
    print(f"files: {len(names)}")
    for n in names:
        o = offsets.get(n, 0.0)
        print(f"  {n:<24} dur={info[n]['duration']:7.2f}s fps={info[n]['fps']:6.3f} "
              f"offset={o:+7.2f}s window=[{windows[n][0]:+7.2f}, {windows[n][1]:+7.2f}]")

    if not args.normalize:
        if span_len <= 0:
            sys.exit(f"\nNO common span: windows intersect by {span_len:.2f}s. "
                     f"The offsets/windows don't overlap — trim files or re-check offsets.")
        print(f"\ncommon span: [{span_start:+.2f}, {span_end:+.2f}]  length={span_len:.2f}s")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"span_start": span_start, "span_end": span_end,
              "span_length": span_len, "files": {}}
    for n in names:
        meta = info[n]
        if args.normalize:
            start, nframes = 0.0, int(round(meta["duration"] * meta["fps"]))
        else:
            start = span_start - offsets.get(n, 0.0)
            nframes = int(round(span_len * meta["fps"]))
        out_path = out_dir / n
        print(f"[cut] {n}: start={start:+.2f}s frames={nframes} "
              f"({meta['w']}x{meta['h']} @ {meta['fps']:.3f}fps) -> {out_path.name}")
        ffmpeg_cut(src / n, out_path, start, nframes, meta["fps"],
                   encoder=args.encoder, crf=args.crf,
                   preset=args.preset, max_w=args.max_w)
        report["files"][n] = {"cut_start_s": round(start, 3),
                              "frames": nframes,
                              "fps": round(meta["fps"], 4),
                              "out": str(out_path)}

    report_path = out_dir / "aligned_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {report_path}")
    print("verify:  ffprobe -v error -show_entries format=start_time,duration "
          "-of default=noprint_wrappers=1 <out>/*.mp4   (want start_time=0.000)")


if __name__ == "__main__":
    main()
