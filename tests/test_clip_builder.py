import os
import subprocess
import uuid

import cv2
import numpy as np

from app import clip_builder
from app.clip_builder import build_clip, build_preview_clip
from app.segment_store import SegmentRef


def _make_segment(path: str, duration: float):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"testsrc=duration={duration}:size=64x64:rate=10", "-c:v", "libx264",
         "-g", "1", "-pix_fmt", "yuv420p", "-f", "mpegts", path],
        check=True, capture_output=True,
    )


def _make_gop_segment(path: str, duration: float, gop: int = 10, rate: int = 10):
    """A segment with a REAL keyframe interval.

    The helper above uses -g 1, which makes every frame a keyframe and therefore
    hides every keyframe-granularity defect: a stream-copy cut always lands
    exactly where asked and always opens on an I-frame. Production segments carry
    iframeinterval=30 (~1s of GOP), so tests that care about where a cut lands or
    whether the result is decodable must use this instead.
    """
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"testsrc=duration={duration}:size=64x64:rate={rate}", "-c:v", "libx264",
         "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
         "-pix_fmt", "yuv420p", "-f", "mpegts", path],
        check=True, capture_output=True,
    )


def _make_headless_segment(path: str, duration: float):
    """A segment that opens mid-stream, with no SPS/PPS of its own.

    This is what the recorder produced for every fragment after the first when
    h264parse negotiated stream-format=avc: the parameter sets went out-of-band
    into caps codec_data, so the muxer wrote them once at stream start and later
    fragments carried only slices referencing them. Standalone it reads as
    "non-existing PPS 0 referenced" and probes as width=0 -- and note ffprobe
    still EXITS 0 on it, so nothing but the dimensions gives it away.

    Built by dropping whole 188-byte TS packets off the head, which keeps the file
    packet-aligned while losing the PAT/PMT and the leading parameter sets. The
    source is encoded with a SINGLE keyframe on purpose: ffmpeg's mpegts muxer
    re-inserts SPS/PPS ahead of every keyframe, so a multi-GOP segment still
    decodes after its head is removed and would silently not test anything.
    """
    whole = path + ".whole.ts"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"testsrc=duration={duration}:size=64x64:rate=10", "-c:v", "libx264",
         "-g", "1000", "-keyint_min", "1000", "-sc_threshold", "0",
         "-pix_fmt", "yuv420p", "-f", "mpegts", whole],
        check=True, capture_output=True,
    )
    with open(whole, "rb") as fh:
        data = fh.read()
    with open(path, "wb") as fh:
        fh.write(data[188 * 16:])
    os.remove(whole)


def _dimensions(path: str) -> tuple:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", path],
        check=True, capture_output=True, text=True,
    )
    w, h = out.stdout.strip().splitlines()[0].split(",")[:2]
    return int(w), int(h)


def _duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def _first_pict_types(path: str, n: int = 3) -> list:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "frame=pict_type", "-of", "default=noprint_wrappers=1:nokey=1", path],
        check=True, capture_output=True, text=True,
    )
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()][:n]


def test_build_clip_trims_fixed_context_off_each_end_when_enabled(tmp_path):
    seg0, seg1, seg2 = str(tmp_path / "seg0.ts"), str(tmp_path / "seg1.ts"), str(tmp_path / "seg2.ts")
    _make_segment(seg0, 1.0)
    _make_segment(seg1, 1.0)
    _make_segment(seg2, 1.0)

    # 3 one-second segments concatenated = 3.0s total; trim 0.5s off each end.
    out, actual_start, actual_end = build_clip(
        [seg0, seg1, seg2], dest_dir=str(tmp_path), base_pts=0.0,
        total_span=3.0, trim_enabled=True, trim_seconds=0.5)

    assert os.path.exists(out)
    dur = _duration(out)
    assert 1.6 <= dur <= 2.4  # requested 2.0s (3.0 - 2*0.5); keyframe-snapped, not frame-exact
    assert actual_start == 0.5
    assert actual_end == 2.5


def test_build_clip_skips_trim_when_material_too_short(tmp_path):
    seg0 = str(tmp_path / "seg0.ts")
    _make_segment(seg0, 1.0)

    # Only 1.0s of material; trimming 0.5s off each end would leave 0s, so no trim
    # even though trim_enabled=True.
    out, actual_start, actual_end = build_clip(
        [seg0], dest_dir=str(tmp_path), base_pts=10.0,
        total_span=1.0, trim_enabled=True, trim_seconds=0.5)

    assert os.path.exists(out)
    dur = _duration(out)
    assert 0.6 <= dur <= 1.4  # requested the whole 1.0s untrimmed
    assert actual_start == 10.0
    assert actual_end == 11.0


def test_build_clip_no_trim_when_disabled(tmp_path):
    seg0, seg1, seg2 = str(tmp_path / "seg0.ts"), str(tmp_path / "seg1.ts"), str(tmp_path / "seg2.ts")
    _make_segment(seg0, 1.0)
    _make_segment(seg1, 1.0)
    _make_segment(seg2, 1.0)

    # Same 3.0s material as the "enabled" test above, but trim_enabled=False --
    # the whole concatenated clip must come back untrimmed regardless of
    # trim_seconds.
    out, actual_start, actual_end = build_clip(
        [seg0, seg1, seg2], dest_dir=str(tmp_path), base_pts=0.0,
        total_span=3.0, trim_enabled=False, trim_seconds=0.5)

    assert os.path.exists(out)
    dur = _duration(out)
    assert 2.6 <= dur <= 3.4  # requested the whole 3.0s untrimmed
    assert actual_start == 0.0
    assert actual_end == 3.0


def test_build_preview_clip_returns_none_when_no_segments():
    class _EmptyStore:
        def list_range(self, source_id, lo, hi):
            return []

    assert build_preview_clip(_EmptyStore(), "cam0", 0.0, 8.0, 8.0, False, 4.0, "/tmp") is None


def test_build_preview_clip_fetches_and_builds(tmp_path):
    seg0, seg1 = str(tmp_path / "seg0.ts"), str(tmp_path / "seg1.ts")
    _make_segment(seg0, 1.0)
    _make_segment(seg1, 1.0)

    class _Store:
        def list_range(self, source_id, lo, hi):
            return [SegmentRef(0.0, 1.0, "a", 0.0), SegmentRef(1.0, 2.0, "b", 0.0)]

        def fetch(self, ref, dest_dir):
            return seg0 if ref.object_key == "a" else seg1

    result = build_preview_clip(_Store(), "cam0", lo=0.3, hi=1.7, padding_seconds=0.0,
                                trim_enabled=True, trim_seconds=0.0, dest_dir=str(tmp_path))

    assert result is not None
    path, actual_start, actual_end = result
    assert os.path.exists(path)
    # trim_seconds=0.0 -> no actual trimming happens even though trim_enabled=True;
    # full 2.0s concatenated span kept.
    assert actual_start == 0.0
    assert actual_end == 2.0


def test_build_clip_window_cuts_exactly_the_requested_range(tmp_path):
    seg0, seg1, seg2 = str(tmp_path / "w0.ts"), str(tmp_path / "w1.ts"), str(tmp_path / "w2.ts")
    _make_segment(seg0, 1.0)
    _make_segment(seg1, 1.0)
    _make_segment(seg2, 1.0)

    # 3.0s of material starting at pts 100.0; ask for [100.5, 102.0].
    out, actual_start, actual_end = build_clip(
        [seg0, seg1, seg2], dest_dir=str(tmp_path), base_pts=100.0,
        total_span=3.0, trim_enabled=False, trim_seconds=0.5,
        window=(100.5, 102.0))

    assert os.path.exists(out)
    assert 1.2 <= _duration(out) <= 1.8      # requested 1.5s
    assert actual_start == 100.5
    assert actual_end == 102.0


def test_build_clip_window_clamps_when_it_starts_before_the_material(tmp_path):
    seg0 = str(tmp_path / "c0.ts")
    _make_segment(seg0, 1.0)

    # Recording began after the window's start -- the footage simply is not there.
    out, actual_start, actual_end = build_clip(
        [seg0], dest_dir=str(tmp_path), base_pts=100.0,
        total_span=1.0, trim_enabled=False, trim_seconds=0.0,
        window=(99.0, 100.6))

    assert os.path.exists(out)
    assert actual_start == 100.0             # clamped to what exists, not the request
    assert actual_end == 100.6


def test_build_clip_window_clamps_when_it_runs_past_the_material(tmp_path):
    seg0 = str(tmp_path / "d0.ts")
    _make_segment(seg0, 1.0)

    out, actual_start, actual_end = build_clip(
        [seg0], dest_dir=str(tmp_path), base_pts=100.0,
        total_span=1.0, trim_enabled=False, trim_seconds=0.0,
        window=(100.4, 105.0))

    assert os.path.exists(out)
    assert actual_start == 100.4
    assert actual_end == 101.0               # clamped to base_pts + total_span


def test_build_clip_without_window_keeps_the_existing_trim_behaviour(tmp_path):
    seg0, seg1, seg2 = str(tmp_path / "r0.ts"), str(tmp_path / "r1.ts"), str(tmp_path / "r2.ts")
    _make_segment(seg0, 1.0)
    _make_segment(seg1, 1.0)
    _make_segment(seg2, 1.0)

    # Regression guard for chat search, which never passes a window.
    out, actual_start, actual_end = build_clip(
        [seg0, seg1, seg2], dest_dir=str(tmp_path), base_pts=0.0,
        total_span=3.0, trim_enabled=True, trim_seconds=0.5)

    assert os.path.exists(out)
    assert actual_start == 0.5
    assert actual_end == 2.5


def test_build_preview_clip_forwards_the_window(tmp_path):
    seg0, seg1 = str(tmp_path / "p0.ts"), str(tmp_path / "p1.ts")
    _make_segment(seg0, 1.0)
    _make_segment(seg1, 1.0)

    class _Store:
        def list_range(self, source_id, lo, hi):
            return [SegmentRef(50.0, 51.0, "a", 0.0), SegmentRef(51.0, 52.0, "b", 0.0)]

        def fetch(self, ref, dest_dir):
            return seg0 if ref.object_key == "a" else seg1

    result = build_preview_clip(_Store(), "cam0", lo=50.5, hi=51.5, padding_seconds=0.0,
                                trim_enabled=False, trim_seconds=0.0, dest_dir=str(tmp_path),
                                window=(50.5, 51.5))

    assert result is not None
    _path, actual_start, actual_end = result
    assert actual_start == 50.5
    assert actual_end == 51.5


def test_build_clip_window_entirely_after_the_material_returns_none(tmp_path):
    seg0 = str(tmp_path / "n0.ts")
    _make_segment(seg0, 1.0)

    # Material spans [100.0, 101.0]; the window starts after it ends. A 0.1s
    # clip claiming [102.0, 103.0] would be a fabrication, so refuse instead.
    assert build_clip([seg0], dest_dir=str(tmp_path), base_pts=100.0,
                      total_span=1.0, trim_enabled=False, trim_seconds=0.0,
                      window=(102.0, 103.0)) is None


def test_build_clip_window_entirely_before_the_material_returns_none(tmp_path):
    seg0 = str(tmp_path / "n1.ts")
    _make_segment(seg0, 1.0)

    assert build_clip([seg0], dest_dir=str(tmp_path), base_pts=100.0,
                      total_span=1.0, trim_enabled=False, trim_seconds=0.0,
                      window=(97.0, 98.5)) is None


def test_windowed_cut_opens_on_a_decodable_keyframe(tmp_path):
    """Regression: a stream-copied cut left clips starting on P/B frames.

    ffmpeg's accurate-seek drops the leading IDR when copying, so the clip opened
    with predicted frames referencing a picture the player never received — it
    rendered black or froze until the next keyframe, up to a full GOP in. Only
    visible with a realistic GOP, which is why -g 1 fixtures never caught it.
    """
    segs = [str(tmp_path / f"k{i}.ts") for i in range(3)]
    for s in segs:
        _make_gop_segment(s, 3.0, gop=10, rate=10)

    out, _start, _end = build_clip(
        segs, dest_dir=str(tmp_path), base_pts=0.0, total_span=9.0,
        trim_enabled=False, trim_seconds=0.0, window=(2.55, 6.55))

    assert _first_pict_types(out)[0] == "I", (
        "clip must open on a keyframe or players show black until the next one")


def test_windowed_cut_delivers_the_requested_duration(tmp_path):
    """Regression: keyframe-snapped copying skewed durations by up to a GOP.

    A 4s window must yield ~4s of footage regardless of where in a GOP it starts.
    """
    segs = [str(tmp_path / f"d{i}.ts") for i in range(3)]
    for s in segs:
        _make_gop_segment(s, 3.0, gop=10, rate=10)

    for start in (2.35, 2.55, 2.85, 3.15):
        out, actual_start, actual_end = build_clip(
            segs, dest_dir=str(tmp_path), base_pts=0.0, total_span=9.0,
            trim_enabled=False, trim_seconds=0.0, window=(start, start + 4.0))
        assert abs(_duration(out) - 4.0) < 0.25, f"window starting at {start} skewed"
        # The reported range is measured off the file, so it tracks reality.
        assert abs((actual_end - actual_start) - _duration(out)) < 0.05


def test_reported_range_reflects_the_built_file_not_the_request(tmp_path):
    """The old code returned request arithmetic, so a wrong clip still logged a
    healthy range — the reason broken clips went unnoticed."""
    seg = str(tmp_path / "m0.ts")
    _make_gop_segment(seg, 3.0, gop=10, rate=10)

    out, actual_start, actual_end = build_clip(
        [seg], dest_dir=str(tmp_path), base_pts=100.0, total_span=3.0,
        trim_enabled=False, trim_seconds=0.0, window=(100.5, 102.0))

    assert abs((actual_end - actual_start) - _duration(out)) < 0.05
    assert actual_start == 100.5


def test_passthrough_is_not_re_encoded(tmp_path):
    """A whole-concat request is not a cut, so it must stay a stream copy —
    chat-search clips take this path and must not pay for an encode."""
    seg = str(tmp_path / "p_only.ts")
    _make_gop_segment(seg, 2.0, gop=10, rate=10)

    calls = []
    real_run = clip_builder._run_ffmpeg

    def _spy(args):
        calls.append(args)
        return real_run(args)

    clip_builder._run_ffmpeg = _spy
    try:
        build_clip([seg], dest_dir=str(tmp_path), base_pts=0.0, total_span=2.0,
                   trim_enabled=False, trim_seconds=0.0)
    finally:
        clip_builder._run_ffmpeg = real_run

    assert "libx264" not in calls[-1], "passthrough must stream-copy, not re-encode"
    assert "copy" in calls[-1]


def test_build_preview_clip_ignores_a_nested_segment_when_sizing_the_material(tmp_path):
    # list_range sorts by start_pts, so the nested (96,98) ref sorts LAST without
    # ending last. Sizing the material from refs[-1] would give total_span=8.0 and
    # cut the clip to [95,98] -- losing the alert moment at 100 that the window was
    # built around -- and concatenating it would duplicate footage (90,106) already
    # holds.
    long_seg, nested_seg = str(tmp_path / "long.ts"), str(tmp_path / "nested.ts")
    _make_segment(long_seg, 3.0)
    _make_segment(nested_seg, 1.0)

    class _Store:
        def __init__(self):
            self.fetched = []

        def list_range(self, source_id, lo, hi):
            return [SegmentRef(90.0, 106.0, "long", 0.0),
                    SegmentRef(96.0, 98.0, "nested", 0.0)]

        def fetch(self, ref, dest_dir):
            self.fetched.append(ref.object_key)
            return long_seg if ref.object_key == "long" else nested_seg

    store = _Store()
    result = build_preview_clip(store, "cam0", lo=95.0, hi=105.0, padding_seconds=0.0,
                                trim_enabled=False, trim_seconds=0.0,
                                dest_dir=str(tmp_path), window=(95.0, 105.0))

    assert result is not None
    _path, actual_start, actual_end = result
    assert store.fetched == ["long"]          # nested ref never fetched or concatenated
    assert actual_start == 95.0
    assert actual_end == 105.0                # full window, not truncated to 98.0


def _headless_store(tmp_path, refs, headless_keys):
    """A store whose segments are real files, some of them missing parameter sets."""
    paths = {}
    for ref in refs:
        p = str(tmp_path / f"{ref.object_key}.ts")
        if ref.object_key in headless_keys:
            _make_headless_segment(p, ref.end_pts - ref.start_pts)
        else:
            _make_gop_segment(p, ref.end_pts - ref.start_pts, gop=10, rate=10)
        paths[ref.object_key] = p

    class _Store:
        def list_range(self, source_id, lo, hi):
            return list(refs)

        def fetch(self, ref, dest_dir):
            return paths[ref.object_key]

    return _Store()


def test_build_preview_clip_drops_a_leading_segment_without_parameter_sets(tmp_path):
    # The exact production failure: the first segment in the window carried no
    # SPS/PPS, so `-f concat -c copy` could not learn the frame size and died with
    # "dimensions not set" / "Could not write header" -- exit 234. That exception
    # escaped to the alert worker and resolved a perfectly recoverable alert
    # 'unavailable', discarding the good footage sitting either side of it.
    refs = [SegmentRef(90.0, 93.0, "bad", 0.0),
            SegmentRef(93.0, 96.0, "good0", 0.0),
            SegmentRef(96.0, 99.0, "good1", 0.0)]
    store = _headless_store(tmp_path, refs, {"bad"})

    result = build_preview_clip(store, "cam0", lo=91.0, hi=98.0, padding_seconds=0.0,
                                trim_enabled=False, trim_seconds=0.0,
                                dest_dir=str(tmp_path), window=(91.0, 98.0))

    assert result is not None, "one unusable segment must not sink the whole clip"
    path, actual_start, _actual_end = result
    assert actual_start == 93.0, "clip must start where the usable footage starts"
    assert _dimensions(path) == (64, 64), "built clip must actually decode"


def test_build_preview_clip_returns_none_when_no_segment_is_usable(tmp_path):
    # Nothing salvageable: the caller must get a clean None (which resolves the
    # alert 'unavailable' with a logged reason) rather than an ffmpeg traceback.
    refs = [SegmentRef(90.0, 93.0, "bad0", 0.0), SegmentRef(93.0, 96.0, "bad1", 0.0)]
    store = _headless_store(tmp_path, refs, {"bad0", "bad1"})

    assert build_preview_clip(store, "cam0", lo=91.0, hi=95.0, padding_seconds=0.0,
                              trim_enabled=False, trim_seconds=0.0,
                              dest_dir=str(tmp_path), window=(91.0, 95.0)) is None


# ── overlay pass ────────────────────────────────────────────────────────────

def _first_frame_bgr(video_path: str, tmp_dir):
    shot = os.path.join(str(tmp_dir), f"shot_{uuid.uuid4().hex[:6]}.png")
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-frames:v", "1", shot],
                   check=True, capture_output=True)
    return cv2.imread(shot)


def _clip_from_segment(tmp_path, duration=2.0):
    """A plain mp4 to overlay onto, shaped like a finished alert clip."""
    seg = str(tmp_path / f"seg_{uuid.uuid4().hex[:6]}.ts")
    _make_gop_segment(seg, duration)
    clip = str(tmp_path / f"clip_{uuid.uuid4().hex[:6]}.mp4")
    subprocess.run(["ffmpeg", "-y", "-i", seg, "-c", "copy", clip],
                   check=True, capture_output=True)
    return clip


def _opaque_red_png(path, width=64, height=64):
    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[:, :] = (0, 0, 255, 255)        # fully opaque red over the whole frame
    cv2.imwrite(path, img)
    return path


def test_probe_dimensions_reads_the_video_size(tmp_path):
    clip = _clip_from_segment(tmp_path)
    assert clip_builder.probe_dimensions(clip) == (64, 64)


def test_probe_dimensions_returns_none_for_a_missing_file(tmp_path):
    assert clip_builder.probe_dimensions(str(tmp_path / "nope.mp4")) is None


def test_apply_overlay_burns_the_image_in_and_keeps_the_duration(tmp_path):
    clip = _clip_from_segment(tmp_path, duration=2.0)
    png = _opaque_red_png(str(tmp_path / "ov.png"))

    out = clip_builder.apply_overlay(clip, png, str(tmp_path))

    assert out is not None
    assert _dimensions(out) == (64, 64)
    # No cut happens in this pass, so the duration must carry over -- alert_clips
    # relies on that by NOT recomputing the clip's reported range.
    assert abs(_duration(out) - _duration(clip)) <= 0.15
    # The overlay actually reached the pixels. yuv420p round-trips, so allow slack
    # rather than asserting an exact (0, 0, 255).
    b, g, r = _first_frame_bgr(out, tmp_path)[32, 32]
    assert r > 180 and b < 60 and g < 60


def test_apply_overlay_returns_none_when_ffmpeg_fails(tmp_path):
    clip = _clip_from_segment(tmp_path)
    assert clip_builder.apply_overlay(clip, str(tmp_path / "missing.png"),
                                      str(tmp_path)) is None


def test_build_clip_never_uses_a_filter_complex(tmp_path, monkeypatch):
    """Regression guard: the stabilized cut command must stay as it is.

    The zone overlay is a separate pass on purpose -- the cut in build_clip carries a
    long tail of keyframe/decodability fixes and must not gain new inputs or filters.
    """
    seen = []
    real = clip_builder._run_ffmpeg

    def spy(args):
        seen.append(list(args))
        return real(args)

    monkeypatch.setattr(clip_builder, "_run_ffmpeg", spy)
    seg = str(tmp_path / "s.ts")
    _make_gop_segment(seg, 3.0)
    build_clip([seg], str(tmp_path), 0.0, 3.0, False, 0.0, window=(0.5, 2.0))

    assert seen, "expected build_clip to invoke ffmpeg"
    assert all("-filter_complex" not in args for args in seen)
    assert any("libx264" in args for args in seen)      # the cut still re-encodes


def _make_mkv(path: str, duration: float, gop: int = 10, rate: int = 10):
    """Matroska with a realistic keyframe interval — what Smart Record now writes.

    The container had to move off MP4 because qtmux refuses any caps change after
    a track is configured, and RTSP sources whose SDP parameter sets disagree with
    their in-band ones change caps ~0.65s in. matroskamux accepts it.
    """
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"testsrc=duration={duration}:size=64x64:rate={rate}", "-c:v", "libx264",
         "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
         "-pix_fmt", "yuv420p", "-f", "matroska", path],
        check=True, capture_output=True,
    )


class TestRemuxToMp4:
    """Smart Record writes MKV; browsers need MP4. The bytes must be copied, not
    re-encoded — this runs on every alert."""

    def test_produces_a_decodable_mp4(self, tmp_path):
        src = str(tmp_path / "rec.mkv")
        _make_mkv(src, 2.0)

        out = clip_builder.remux_to_mp4(src, str(tmp_path / "out"))

        assert out is not None and out.endswith(".mp4")
        # width=0 is how an undecodable file reports itself; ffprobe exits 0 on
        # those, so the dimensions are the only honest check.
        assert clip_builder.probe_dimensions(out) == (64, 64)

    def test_preserves_the_recording_duration(self, tmp_path):
        src = str(tmp_path / "rec.mkv")
        _make_mkv(src, 2.0)

        out = clip_builder.remux_to_mp4(src, str(tmp_path / "out"))

        assert abs(clip_builder._probe_duration(out) - 2.0) < 0.3

    def test_returns_none_instead_of_raising_when_the_source_is_unusable(self, tmp_path):
        """attach_svr_file must still resolve its events. An exception here would
        leave them 'pending' for the life of the process — the failure mode
        StreamManager's describe-flag guard already exists to prevent."""
        junk = tmp_path / "not-video.mkv"
        junk.write_bytes(b"this is not a matroska file")

        assert clip_builder.remux_to_mp4(str(junk), str(tmp_path / "out")) is None


def _frame_at(video_path, t, tmp_dir):
    shot = os.path.join(str(tmp_dir), f"at_{uuid.uuid4().hex[:6]}.png")
    subprocess.run(["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", video_path, "-frames:v", "1", shot],
                   check=True, capture_output=True)
    return cv2.imread(shot)


def test_apply_overlays_burns_a_timed_sequence_only_while_it_lasts(tmp_path):
    clip = _clip_from_segment(tmp_path, duration=2.0)
    blank = str(tmp_path / "blank.png")
    cv2.imwrite(blank, np.zeros((64, 64, 4), np.uint8))
    red = _opaque_red_png(str(tmp_path / "red.png"))
    timed = tmp_path / "seq.ffconcat"
    timed.write_text(f"ffconcat version 1.0\nfile '{blank}'\nduration 1.0\nfile '{red}'\nduration 0.5\n"
                     f"file '{blank}'\nduration 0.5\nfile '{blank}'\n")

    out = clip_builder.apply_overlays(clip, None, str(timed), str(tmp_path))

    assert out is not None and abs(_duration(out) - _duration(clip)) <= 0.15

    def same(a, b):
        return int(np.abs(a.astype(int) - b.astype(int)).max()) < 40

    assert same(_frame_at(out, 0.5, tmp_path)[32, 32], _frame_at(clip, 0.5, tmp_path)[32, 32])    # before
    b, g, r = _frame_at(out, 1.25, tmp_path)[32, 32]
    assert r > 180 and b < 60 and g < 60                                                          # during
    assert same(_frame_at(out, 1.8, tmp_path)[32, 32], _frame_at(clip, 1.8, tmp_path)[32, 32])    # after


def test_apply_overlays_with_nothing_to_burn_returns_none(tmp_path):
    assert clip_builder.apply_overlays(_clip_from_segment(tmp_path), None, None, str(tmp_path)) is None
