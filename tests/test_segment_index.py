"""SegmentBoundaryTracker: real-PTS fragment boundary bookkeeping.

sidecar.py itself needs gi/pyds (GPU container only), but segment_index is pure
Python and importable off-GPU (same convention as test_zone_dedup.py)."""

import pytest

from perception.engine.segment_index import (SegmentBoundaryTracker, StreamClockOffsets,
                                   has_parameter_sets)


def test_first_rotation_finalises_nothing_and_seeds_origin():
    t = SegmentBoundaryTracker(fallback_seconds=8.0)
    # The record branch links on pad-added, after the pipeline is already running,
    # so fragment 0 starts at real running time 12.4 -- not 0.0.
    assert t.rotate(12.4) is None
    assert t.rotate(20.5) == (12.4, 20.5)


def test_boundaries_chain_on_measured_values_not_a_counter():
    t = SegmentBoundaryTracker(fallback_seconds=8.0)
    t.rotate(0.0)
    # splitmuxsink rotates at the first keyframe at or after max-size-time, so real
    # spans run slightly long. The counter would have said 8.0 / 16.0 / 24.0.
    assert t.rotate(8.3) == (0.0, 8.3)
    assert t.rotate(16.7) == (8.3, 16.7)
    assert t.rotate(25.1) == (16.7, 25.1)


def test_missing_pts_falls_back_to_synthetic_span():
    t = SegmentBoundaryTracker(fallback_seconds=8.0)
    t.rotate(None)                              # origin unknown -> 0.0
    assert t.rotate(None) == (0.0, 8.0)
    assert t.rotate(None) == (8.0, 16.0)


def test_missing_pts_midstream_falls_back_then_resumes_on_real_pts():
    t = SegmentBoundaryTracker(fallback_seconds=8.0)
    t.rotate(4.0)
    assert t.rotate(None) == (4.0, 12.0)        # synthetic
    assert t.rotate(21.0) == (12.0, 21.0)       # measured again


def test_non_monotonic_pts_falls_back_instead_of_emitting_negative_span():
    t = SegmentBoundaryTracker(fallback_seconds=8.0)
    t.rotate(100.0)
    # A looping file source restarts pts near zero; accepting it would emit an
    # end before its own start.
    assert t.rotate(0.5) == (100.0, 108.0)
    assert t.rotate(116.0) == (108.0, 116.0)


def test_bogus_origin_from_a_null_first_sample_is_clamped_to_one_fragment():
    # splitmuxsink can hand the first format-location-full a NULL buffer, so the
    # origin falls back to 0.0 even though recording really began at ~120s. Without
    # the clamp, fragment 0 would be indexed 0.000-128.300 -- ~8s of footage claiming
    # to span over two minutes, which reads as full coverage to segments_cover.
    t = SegmentBoundaryTracker(fallback_seconds=8.0)
    assert t.rotate(None) is None            # origin unmeasured -> 0.0
    start, end = t.rotate(128.3)
    assert abs(start - 120.3) < 0.0001
    assert end == 128.3
    assert t.rotate(136.5) == (128.3, 136.5)  # chains normally afterwards


def test_a_recording_gap_leaves_a_hole_instead_of_one_long_span():
    # The source drops for ~40s; splitmuxsink cannot rotate without data, so the
    # next boundary lands 48s late. The fragment still holds only ~8s of footage.
    t = SegmentBoundaryTracker(fallback_seconds=8.0)
    t.rotate(0.0)
    assert t.rotate(8.2) == (0.0, 8.2)
    assert t.rotate(56.2) == (48.2, 56.2)     # hole from 8.2 to 48.2, not a 48s span
    assert t.rotate(64.4) == (56.2, 64.4)


def test_a_normally_long_fragment_is_not_clamped():
    # Keyframe-snapped rotation routinely overshoots record_seconds a little; that
    # must not trip the guard.
    t = SegmentBoundaryTracker(fallback_seconds=8.0)
    t.rotate(0.0)
    assert t.rotate(11.5) == (0.0, 11.5)      # 1.44x, under the 2.0 factor


def test_max_span_factor_is_configurable():
    t = SegmentBoundaryTracker(fallback_seconds=8.0, max_span_factor=1.5)
    t.rotate(0.0)
    assert t.rotate(13.0) == (5.0, 13.0)      # 1.625x exceeds 1.5 -> clamped


def test_non_positive_fallback_seconds_is_rejected():
    # The clamp computes start = end - fallback_seconds, so a non-positive value
    # would yield a zero-length or inverted span. Fail loudly at construction
    # instead of emitting nonsense spans into the segment index.
    with pytest.raises(ValueError):
        SegmentBoundaryTracker(fallback_seconds=0.0)
    with pytest.raises(ValueError):
        SegmentBoundaryTracker(fallback_seconds=-8.0)


def test_max_span_factor_at_or_below_one_is_rejected():
    # At 1.0 the threshold equals fallback_seconds, so every ordinary fragment --
    # which overshoots, because splitmuxsink rotates at the first keyframe at or
    # after max-size-time -- would be clamped and silently shortened.
    with pytest.raises(ValueError):
        SegmentBoundaryTracker(fallback_seconds=8.0, max_span_factor=1.0)
    with pytest.raises(ValueError):
        SegmentBoundaryTracker(fallback_seconds=8.0, max_span_factor=0.5)


def test_offset_is_the_difference_between_the_two_clock_constants():
    # Reproduces the measured case: probe pts runs 1.033s ahead of recorder pts.
    o = StreamClockOffsets()
    o.observe_probe("cam0", pts=100.0, now=1000.0)          # probe const -900.0
    assert o.offset("cam0", boundary=98.967, now=1000.0) == pytest.approx(1.033)


def test_offset_is_frozen_after_the_first_measurement():
    # The recorder side jitters ~0.012s with keyframe snapping. If that leaked into
    # every span, consecutive spans would stop sharing an exact boundary and
    # segments_cover would read the seam as a gap.
    o = StreamClockOffsets()
    o.observe_probe("cam0", pts=100.0, now=1000.0)
    first = o.offset("cam0", boundary=98.967, now=1000.0)
    o.observe_probe("cam0", pts=140.0, now=1040.0)
    assert o.offset("cam0", boundary=138.9, now=1040.0) == first


def test_offset_is_unmeasurable_before_the_probe_has_been_seen():
    o = StreamClockOffsets()
    assert o.offset("cam0", boundary=98.967, now=1000.0) is None


def test_offset_is_unmeasurable_without_a_boundary_pts():
    o = StreamClockOffsets()
    o.observe_probe("cam0", pts=100.0, now=1000.0)
    assert o.offset("cam0", boundary=None, now=1000.0) is None


def test_offsets_are_independent_per_stream():
    o = StreamClockOffsets()
    o.observe_probe("cam0", pts=100.0, now=1000.0)
    o.observe_probe("cam1", pts=100.0, now=1000.0)
    assert o.offset("cam0", boundary=98.967, now=1000.0) == pytest.approx(1.033)
    assert o.offset("cam1", boundary=97.500, now=1000.0) == pytest.approx(2.5)


def test_needs_probe_stops_once_frozen():
    o = StreamClockOffsets()
    assert o.needs_probe("cam0") is True
    o.observe_probe("cam0", pts=100.0, now=1000.0)
    o.offset("cam0", boundary=98.967, now=1000.0)
    assert o.needs_probe("cam0") is False


def test_frozen_offset_keeps_shifted_spans_exactly_contiguous():
    # The recorder side jitters ~0.012s per rotation. Freezing means every span is
    # shifted by the SAME constant, so start_{n+1} == end_n survives exactly. Recompute
    # per rotation and the seams drift apart by the jitter -- which segments_cover reads
    # as a discontinuity and refuses to concatenate across.
    tracker = SegmentBoundaryTracker(fallback_seconds=8.0)
    offsets = StreamClockOffsets()

    boundaries = [0.0, 8.2, 16.1, 24.4, 32.3]
    jitter = [0.0, 0.011, -0.008, 0.012, -0.010]   # recorder-side noise, wall side steady

    shifted = []
    for i, b in enumerate(boundaries):
        now = 1000.0 + b
        offsets.observe_probe("cam0", pts=b + 1.033, now=now)
        off = offsets.offset("cam0", boundary=b + jitter[i], now=now)
        span = tracker.rotate(b)
        if span is not None:
            shifted.append((span[0] + off, span[1] + off))

    assert len(shifted) == 4
    for (_s0, e0), (s1, _e1) in zip(shifted, shifted[1:]):
        assert e0 == s1          # exact, not approx -- any drift here is a real gap


def test_an_implausible_offset_is_refused_rather_than_frozen():
    # An unset buf_pts (CLOCK_TIME_NONE) reaching observe_probe would otherwise freeze
    # a ~1.8e10 constant, indexing every segment billions of seconds away and killing
    # every clip on the stream until the process restarts.
    o = StreamClockOffsets()
    o.observe_probe("cam0", pts=1.8446744e10, now=1000.0)
    assert o.offset("cam0", boundary=98.967, now=1000.0) is None
    assert o.needs_probe("cam0") is True          # not frozen, so it will retry


def test_a_refused_offset_recovers_on_a_later_good_sample():
    o = StreamClockOffsets()
    o.observe_probe("cam0", pts=1.8446744e10, now=1000.0)
    assert o.offset("cam0", boundary=98.967, now=1000.0) is None
    o.observe_probe("cam0", pts=140.0, now=1040.0)
    assert o.offset("cam0", boundary=138.967, now=1040.0) == pytest.approx(1.033)


def test_the_plausibility_bound_is_configurable():
    o = StreamClockOffsets(max_plausible_offset=0.5)
    o.observe_probe("cam0", pts=100.0, now=1000.0)
    assert o.offset("cam0", boundary=98.967, now=1000.0) is None   # 1.033 > 0.5


# ── has_parameter_sets ────────────────────────────────────────────────────────

def _annexb(*nal_types: int) -> bytes:
    """A byte-stream carrying one NAL of each given type, with realistic padding."""
    out = b"\x47\x40\x00\x10" + b"\xff" * 20      # a TS-ish header, no start codes
    for t in nal_types:
        out += b"\x00\x00\x01" + bytes([t & 0x1F]) + b"\x80\x42\x13" * 4
    return out


def test_fragment_with_an_sps_is_usable(tmp_path):
    p = tmp_path / "frag.ts"
    p.write_bytes(_annexb(7, 8, 5))               # SPS, PPS, IDR
    assert has_parameter_sets(str(p)) is True


def test_fragment_opening_mid_gop_without_an_sps_is_rejected(tmp_path):
    # The production symptom: slices referencing PPS 0 with no SPS anywhere. ffmpeg
    # cannot learn the frame size, so `-f concat` fails the whole clip.
    p = tmp_path / "frag.ts"
    p.write_bytes(_annexb(1, 1, 5))               # non-IDR slices + an IDR, no 7/8
    assert has_parameter_sets(str(p)) is False


def test_an_sps_beyond_the_probe_window_is_not_searched_for(tmp_path):
    # Bounded read: rotation runs on the GLib main loop, so this must not grow with
    # fragment size. An SPS only in the tail reads as absent, which is the safe
    # direction -- it costs one skipped fragment, never a broken clip.
    p = tmp_path / "frag.ts"
    p.write_bytes(b"\xff" * 4096 + _annexb(7))
    assert has_parameter_sets(str(p), probe_bytes=1024) is False
    assert has_parameter_sets(str(p)) is True


def test_a_missing_fragment_is_not_usable(tmp_path):
    assert has_parameter_sets(str(tmp_path / "gone.ts")) is False


def test_a_real_muxed_mpegts_fragment_is_recognised(tmp_path):
    """Guards the gap that shipped a regression: has_parameter_sets was verified
    only against hand-built byte patterns, so nothing proved it could read an
    actual mpegts fragment. When it then flagged most real fragments, the sidecar
    stopped indexing them and EVERY alert lost its clip. A synthetic-only test for
    a parser of real container bytes is not a test.
    """
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    p = tmp_path / "real.ts"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=4:size=320x240:rate=30",
         "-c:v", "libx264", "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
         "-pix_fmt", "yuv420p", "-f", "mpegts", str(p)],
        check=True, capture_output=True,
    )
    assert has_parameter_sets(str(p)) is True


def test_track_drift_updates_when_drift_exceeds_threshold():
    o = StreamClockOffsets(track_drift=True, drift_threshold=0.5)
    o.observe_probe("cam0", pts=100.0, now=1000.0)
    first = o.offset("cam0", boundary=98.967, now=1000.0)
    assert first == pytest.approx(1.033)

    # Minor jitter (0.02s) does not change the offset:
    o.observe_probe("cam0", pts=106.0, now=1006.0)
    assert o.offset("cam0", boundary=104.987, now=1006.0) == first

    # Major drift (>0.5s, e.g. file loop or clock jump of 6.0s):
    o.observe_probe("cam0", pts=306.0, now=1200.0)
    updated = o.offset("cam0", boundary=298.967, now=1200.0)  # candidate = (306-1200) - (299-1200) = 7.033
    assert updated == pytest.approx(7.033)
    assert updated != first



def test_the_probe_side_of_the_clock_correction_uses_the_frames_arrival_at_the_mux():
    """The probe runs after inference, seconds after the frame entered the pipeline; the recorder samples its clock
    right after encoding. Stamping the probe side with the mux's arrival time (attach-sys-ts) keeps that inference
    delay out of the correction -- it made alert-clip overlays trail the footage (2026-10-29)."""
    from perception.engine.segment_index import arrival_wall
    now = 1_000_000.0
    assert arrival_wall(int((now - 2.4) * 1e9), use_mux_time=True, now=now) == pytest.approx(now - 2.4)
    assert arrival_wall(int((now - 2.4) * 1e9), use_mux_time=False, now=now) == now        # a camera's own clock
    assert arrival_wall(0, use_mux_time=True, now=now) == now                               # no stamp
    assert arrival_wall(int((now + 5) * 1e9), use_mux_time=True, now=now) == now            # from the future
    assert arrival_wall(int((now - 600) * 1e9), use_mux_time=True, now=now) == now          # implausibly old


def test_the_recorder_side_of_the_clock_correction_uses_the_latest_encoded_frame_not_the_delayed_boundary():
    """splitmuxsink announces a fragment only once it has the next keyframe -- a whole GOP after the boundary frame
    (6 s at this product's 5 fps and 30-frame GOP) -- so pairing the boundary's pts with the time of that announcement
    put a GOP into the correction (2026-10-29: clips 6 s off). The latest encoded frame's (pts, wall) is used instead."""
    from perception.engine.segment_index import recorder_clock
    now = 1_000_000.0
    assert recorder_clock({"last": (500.0, now - 0.05)}, boundary=494.0, now=now) == (500.0, now - 0.05)
    assert recorder_clock({}, boundary=494.0, now=now) == (494.0, now)                       # nothing sampled yet
    assert recorder_clock({"last": (400.0, now - 30.0)}, boundary=494.0, now=now) == (494.0, now)   # stale sample
