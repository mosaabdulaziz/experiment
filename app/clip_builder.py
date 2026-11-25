"""Builds a browser-playable mp4 preview clip from recorded H.264 segments via
ffmpeg stream-copy concat + trim (no re-encode). Segment lookup/fetch goes
through a SegmentStore-shaped object; this module owns only the ffmpeg
orchestration."""
import logging
import os
import subprocess
import uuid

from .chat_search import segments_for_found

logger = logging.getLogger("iris.chat")

# Slack for comparing the cut against the material: below this the request covers
# the whole concatenation and there is nothing to cut.
_CUT_EPSILON = 0.001

# Two segments count as adjacent within this much pts. Consecutive fragments share
# an exact rotation boundary, so anything beyond it is a real hole.
_ADJACENT_EPSILON = 0.5


def _has_decodable_video(path: str) -> bool:
    """True when ffprobe can read frame dimensions off `path` on its own.

    A fragment that opens mid-stream carries slices but no SPS/PPS, so its frame
    size is unknowable and ffmpeg cannot write an mp4 header from it -- feeding one
    to `-f concat` kills the whole clip with "dimensions not set" /
    "Could not write header", exit 234. Note ffprobe still EXITS 0 on such a file
    and merely reports width=0, so the exit status proves nothing here and the
    dimensions have to be read.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-of", "csv=p=0", path],
            check=True, capture_output=True, text=True)
    except Exception:
        return False
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            if int(parts[0]) > 0 and int(parts[1]) > 0:
                return True
    return False


def _usable_run(refs: list, paths: list, window: tuple | None) -> tuple:
    """Drop segments that cannot decode standalone, then return the best gap-free
    run of what survives, as (refs, paths).

    Dropping a segment leaves a real hole, and the window->offset arithmetic in
    `build_clip` assumes the concatenated material is continuous -- so the survivors
    are regrouped into runs and only one run is used. With a window (an alert clip)
    the run holding its midpoint wins, because that is the fired moment the clip
    exists to show; otherwise the longest run does.
    """
    usable = [(r, p) for r, p in zip(refs, paths) if _has_decodable_video(p)]
    if not usable:
        return [], []
    runs = [[usable[0]]]
    for ref, path in usable[1:]:
        if ref.start_pts <= runs[-1][-1][0].end_pts + _ADJACENT_EPSILON:
            runs[-1].append((ref, path))
        else:
            runs.append([(ref, path)])
    chosen = None
    if window is not None:
        mid = (window[0] + window[1]) / 2.0
        chosen = next((run for run in runs
                       if run[0][0].start_pts <= mid <= run[-1][0].end_pts), None)
    if chosen is None:
        chosen = max(runs, key=lambda run: run[-1][0].end_pts - run[0][0].start_pts)
    return [r for r, _ in chosen], [p for _, p in chosen]


def _run_ffmpeg(args: list) -> None:
    try:
        subprocess.run(["ffmpeg", "-y", *args], check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        logger.error("ffmpeg failed: %s", stderr)
        raise


def _probe_duration(path: str):
    """Measured duration of `path` in seconds, or None if ffprobe cannot say.

    What was asked for is not what gets delivered, so the range a clip claims has
    to be read back off the file rather than assumed from the request.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            check=True, capture_output=True, text=True)
        return float(out.stdout.strip())
    except Exception:
        logger.warning("ffprobe could not measure %s — falling back to the requested "
                       "duration for the reported range", path)
        return None


def probe_duration(path: str) -> "float | None":
    """The clip's measured duration in seconds, or None (ffprobe could not say)."""
    return _probe_duration(path)


# The overlay pass is a second encoder generation over already-lossy input, so it uses
# a tighter quantiser than the first pass's 23 to avoid visibly softening detail.
_OVERLAY_CRF = "20"


def probe_dimensions(path: str) -> "tuple | None":
    """(width, height) of `path`'s first video stream, or None if ffprobe cannot say.

    Same ffprobe call _has_decodable_video makes, returning the size rather than a
    verdict. The overlay PNG is rendered at exactly these dimensions so a plain
    `overlay` filter suffices -- scale2ref would avoid the probe but is deprecated in
    current ffmpeg, which would make the build depend on the installed version.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-of", "csv=p=0", path],
            check=True, capture_output=True, text=True)
    except Exception:
        return None
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            width, height = int(parts[0]), int(parts[1])
            if width > 0 and height > 0:
                return width, height
    return None


def apply_overlay(clip_path: str, overlay_png: str, dest_dir: str) -> "str | None":
    """Burn `overlay_png` onto every frame of `clip_path`; return the new path, or None.

    A deliberate SECOND pass rather than a filter inside build_clip: that cut command
    encodes a long tail of keyframe and decodability fixes, and a cosmetic overlay must
    not be able to break it. Returning None on any failure lets the caller keep the clip
    it already built -- an overlay must never cost an alert its clip.

    eof_action=repeat is pinned rather than left to ffmpeg's default: the overlay is a
    single-frame input, and under `endall` the output would stop after one frame.
    """
    out_path = os.path.join(dest_dir, f"overlay_{uuid.uuid4().hex[:8]}.mp4")
    try:
        _run_ffmpeg(["-i", clip_path, "-i", overlay_png,
                     "-filter_complex", "[0:v][1:v]overlay=eof_action=repeat",
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", _OVERLAY_CRF,
                     "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path])
    except Exception:
        logger.warning("overlay pass failed for %s — keeping the un-overlaid clip",
                       clip_path, exc_info=True)
        return None
    return out_path


def apply_overlays(clip_path: str, still_png: "str | None", timed_list: "str | None",
                   dest_dir: str) -> "str | None":
    """Burn a still image (the zone) and/or a timed image sequence (an ffconcat list from app/overlay_track.py) onto
    `clip_path` in one pass, as apply_overlay does; return the new path, or None (the caller keeps its clip). The
    clip's timestamps restart at 0 so the sequence's own clock, which starts at 0, lines up with it."""
    if still_png is None and timed_list is None:
        return None
    out_path = os.path.join(dest_dir, f"overlay_{uuid.uuid4().hex[:8]}.mp4")
    args, chain, last, n = ["-i", clip_path], ["[0:v]setpts=PTS-STARTPTS[v0]"], "v0", 1
    if still_png is not None:
        args += ["-i", still_png]
        chain.append(f"[{last}][{n}:v]overlay=eof_action=repeat[v{n}]")
        last, n = f"v{n}", n + 1
    if timed_list is not None:
        args += ["-f", "concat", "-safe", "0", "-i", timed_list]
        chain.append(f"[{last}][{n}:v]overlay=eof_action=pass[v{n}]")
        last, n = f"v{n}", n + 1
    try:
        _run_ffmpeg(args + ["-filter_complex", ";".join(chain), "-map", f"[{last}]",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", _OVERLAY_CRF,
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path])
    except Exception:
        logger.warning("overlay pass failed for %s — keeping the clip without it", clip_path, exc_info=True)
        return None
    return out_path


def remux_to_mp4(src_path: str, dest_dir: str) -> "str | None":
    """Rewrap `src_path` into an MP4 without re-encoding; return the path, or None.

    Smart Record writes Matroska, not MP4: qtmux refuses ANY caps change once a
    track is configured, and an RTSP source whose SDP parameter sets disagree with
    the ones it sends in-band changes caps ~0.65s after connecting. That refusal
    returned NOT_NEGOTIATED into nvurisrcbin's internal pre-decode tee and killed
    the whole source (see perception/engine/source_recovery.py). matroskamux accepts the
    change, so the container moved and the rewrap happens here instead.

    Stream copy, so this costs a file copy rather than an encode. Browsers will not
    reliably play Matroska, and ClipStore stores every clip as `<camera>.mp4`, so
    the bytes must be in an MP4 before they are handed over.

    Returns None on failure rather than raising: the caller must still resolve its
    events -- as 'unavailable' -- because an alert left 'pending' never clears.
    """
    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, f"svr_{uuid.uuid4().hex[:8]}.mp4")
    try:
        _run_ffmpeg(["-i", src_path, "-c", "copy",
                     "-movflags", "+faststart", out_path])
    except Exception:
        logger.error("remux of %s to mp4 failed", src_path, exc_info=True)
        return None
    return out_path


def build_clip(segment_paths: list, dest_dir: str, base_pts: float, total_span: float,
               trim_enabled: bool, trim_seconds: float, window: tuple | None = None) -> tuple | None:
    """Concatenate segment_paths (stream-copy), then cut.

    When `window` is given as an absolute (start_pts, end_pts) pair it wins: the
    range is mapped into the concatenated timeline and clamped to the material
    actually present, which is how alert clips get centred on the moment they
    fired. Clamping matters when recording began after the requested start, or a
    gap/retention left the tail missing -- the returned range then reports what is
    really in the file rather than what was asked for. Returns None if the window
    and material are disjoint.

    Otherwise the original heuristic applies: if trim_enabled and the material is
    long enough (total_span > 2*trim_seconds), trim `trim_seconds` off each end;
    otherwise keep the whole concatenated clip.

    Returns (output_path, actual_start_pts, actual_end_pts), or None if the window
    does not overlap the material. The returned end comes from probing the built
    file, so it reports the footage actually delivered.

    Any cut is re-encoded rather than stream-copied -- copying can only cut on
    keyframes, which both skews the duration by up to a GOP and strands the clip
    on an undecodable opening frame. Passthrough (no cut) still stream-copies.
    """
    list_path = os.path.join(dest_dir, f"concat_{uuid.uuid4().hex[:8]}.txt")
    with open(list_path, "w") as fh:
        for p in segment_paths:
            safe = os.path.abspath(p).replace("\\", "/")
            fh.write(f"file '{safe}'\n")

    concat_path = os.path.join(dest_dir, f"concat_{uuid.uuid4().hex[:8]}.mp4")
    _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", list_path,
                "-c", "copy", "-movflags", "+faststart", concat_path])

    if window is not None:
        start = min(max(window[0] - base_pts, 0.0), total_span)
        end = max(min(window[1] - base_pts, total_span), 0.0)
        if end - start <= 0:
            return None        # window and material are disjoint
        duration = end - start
    elif trim_enabled and total_span > 2 * trim_seconds:
        start = trim_seconds
        duration = total_span - 2 * trim_seconds
    else:
        start = 0.0
        duration = total_span
    duration = max(duration, 0.1)

    out_path = os.path.join(dest_dir, f"clip_{uuid.uuid4().hex[:8]}.mp4")
    # Stream copy can only cut on keyframes, and the recorder's GOP is ~1s
    # (iframeinterval=30 at 25fps), so a copied cut lands up to a whole GOP away
    # from the request. Worse, ffmpeg's accurate-seek then drops the leading IDR,
    # leaving the clip opening on P/B frames that reference a picture the player
    # never received -- it shows black or freezes until the next keyframe. So
    # re-encode whenever we actually cut. A whole-concat passthrough is not a cut
    # and still copies, which keeps chat-search clips free.
    cutting = start > _CUT_EPSILON or duration < total_span - _CUT_EPSILON
    if cutting:
        trim_args = ["-ss", f"{start:.3f}", "-i", concat_path, "-t", f"{duration:.3f}",
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                     "-pix_fmt", "yuv420p"]
    else:
        trim_args = ["-i", concat_path, "-c", "copy"]
    _run_ffmpeg([*trim_args, "-movflags", "+faststart", out_path])

    # Report what the file holds, not what was requested -- the two diverged
    # silently before, which is why broken clips still logged a healthy range.
    measured = _probe_duration(out_path)
    actual_start = base_pts + start
    actual_end = actual_start + (duration if measured is None else measured)
    return out_path, actual_start, actual_end


def build_preview_clip(segment_store, source_id: str, lo: float, hi: float,
                       padding_seconds: float, trim_enabled: bool, trim_seconds: float,
                       dest_dir: str, window: tuple | None = None):
    """Fetch the segments covering a FOUND window for one camera (padded by
    padding_seconds on each side of [lo, hi], absorbing clock drift between
    the sweep's chunk boundaries and the recorder's rotation schedule) and
    build a preview mp4. When trim_enabled, the concatenated material is
    trimmed to trim_seconds of context on each end (not to [lo, hi] itself --
    see the clip-trim design). Returns (path, actual_start_pts,
    actual_end_pts), or None if no segments cover the range (recording not
    started yet, or the window has aged past retention), or if a window is
    provided but does not overlap the material."""
    refs = segments_for_found(segment_store, source_id, lo, hi, padding_seconds)
    if not refs:
        return None

    # list_range sorts by start_pts, so a nested ref (one wholly inside an earlier
    # one) can appear last without ending last. Two things go wrong if it does:
    # total_span computed from refs[-1] understates the material, and concatenating
    # the nested ref duplicates footage the earlier one already carries, desyncing
    # the -ss arithmetic from the pts timeline. Drop the contained refs, then take
    # the true envelope for whatever partial overlap remains.
    deduped = []
    for ref in refs:
        if deduped and ref.end_pts <= deduped[-1].end_pts:
            continue        # wholly inside the ref we already kept
        deduped.append(ref)
    refs = deduped

    local_paths = [segment_store.fetch(r, dest_dir) for r in refs]

    # One unusable segment used to sink the whole clip: the ffmpeg failure escaped
    # as an exception and the alert resolved 'unavailable', discarding the good
    # footage either side of it. A preview should degrade to a shorter clip instead.
    kept_refs, kept_paths = _usable_run(refs, local_paths, window)
    if not kept_refs:
        logger.warning("[%s] none of the %d segment(s) covering %.3f-%.3f decode "
                       "standalone — no clip", source_id, len(refs), lo, hi)
        return None
    if len(kept_refs) != len(refs):
        logger.warning("[%s] dropped %d of %d segment(s) covering %.3f-%.3f that do not "
                       "decode standalone; building from %.3f-%.3f instead",
                       source_id, len(refs) - len(kept_refs), len(refs), lo, hi,
                       kept_refs[0].start_pts, kept_refs[-1].end_pts)

    base_pts = kept_refs[0].start_pts
    total_span = max(r.end_pts for r in kept_refs) - base_pts
    return build_clip(kept_paths, dest_dir, base_pts, total_span, trim_enabled,
                      trim_seconds, window=window)


def extract_frame(clip_path: str, at_s: float, dest_path: str) -> bool:
    """One frame of `clip_path`, `at_s` seconds in, as an image at `dest_path` (its extension picks the format)."""
    try:
        _run_ffmpeg(["-ss", f"{max(at_s, 0.0):.3f}", "-i", clip_path, "-frames:v", "1", "-q:v", "3", dest_path])
    except Exception:
        logger.warning("no frame at %.3f s of %s", at_s, clip_path, exc_info=True)
        return False
    return os.path.exists(dest_path)
