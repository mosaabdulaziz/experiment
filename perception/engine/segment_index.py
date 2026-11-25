"""Fragment-boundary bookkeeping for the sidecar's segment recorder.

Pure Python (no gi/pyds imports) so it stays unit-testable off-GPU, same
convention as zone_dedup.py and zone_config.py.
"""


def has_parameter_sets(path: str, probe_bytes: int = 1 << 20) -> bool:
    """True when `path` opens with an H.264 SPS, i.e. it can decode on its own.

    The first fragment or two of a run are written while the pipeline is still
    settling and can open mid-GOP, carrying slices that reference parameter sets
    the file never contains. ffmpeg cannot learn the frame size from such a file,
    so `-f concat` dies with "dimensions not set" / "Could not write header" --
    and because that is the fragment the FIRST alert of a run reads from, the
    first alert was the one that lost its clip.

    Indexing it anyway would be a lie: `segments_cover` would report the window
    covered and the build would then fail. Skipping it leaves an honest hole that
    the consumers already know how to refuse.

    Scans for an Annex-B start code followed by nal_unit_type 7. Byte-level rather
    than an ffprobe subprocess: this runs on the GLib main loop at every rotation,
    and the DeepStream image is not guaranteed to ship ffprobe.
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read(probe_bytes)
    except OSError:
        return False
    i = data.find(b"\x00\x00\x01")
    while i != -1 and i + 3 < len(data):
        if data[i + 3] & 0x1F == 7:        # nal_unit_type 7 == SPS
            return True
        i = data.find(b"\x00\x00\x01", i + 3)
    return False


class SegmentBoundaryTracker:
    """Turns splitmuxsink rotation signals into real (start_pts, end_pts) spans.

    `format-location-full` fires just before each NEW fragment opens, so the PTS
    carried by that signal's sample is simultaneously the start of the fragment
    about to open and the end of the one just finalised. Feeding those boundaries
    in yields each finalised fragment's true span.

    This replaces a synthetic `segment_index * record_seconds` counter, which was
    wrong two ways: it assumed fragment 0 began at running time 0 (it does not --
    the record branch links on pad-added, after the pipeline is running), and it
    assumed rotation lands exactly on `record_seconds` (splitmuxsink rotates at
    the first keyframe at or after it, so error accumulated).

    When PTS is unavailable (`CLOCK_TIME_NONE`) or non-monotonic (a looping file
    source restarting near zero), it falls back to `start + fallback_seconds` so
    recording keeps producing usable, monotonic spans.
    """

    def __init__(self, fallback_seconds: float, max_span_factor: float = 2.0):
        if fallback_seconds <= 0:
            raise ValueError(
                f"fallback_seconds must be positive, got {fallback_seconds!r} -- a "
                "non-positive segment length would make the span clamp emit "
                "zero-length or inverted spans")
        if max_span_factor <= 1.0:
            raise ValueError(
                f"max_span_factor must exceed 1.0, got {max_span_factor!r} -- at or "
                "below 1.0 every ordinary keyframe-snapped fragment would be clamped")
        self.fallback_seconds = float(fallback_seconds)
        self.max_span_factor = float(max_span_factor)
        self._start = None      # start_pts of the fragment currently being written

    def rotate(self, boundary):
        """Record one rotation. `boundary` is the sample's PTS in seconds, or None.

        Returns (start_pts, end_pts) for the fragment just finalised, or None on
        the first call -- nothing has been finalised yet, so there is only an
        origin to seed.

        A span far longer than a fragment can actually be means either the origin
        was never measured (a NULL first sample seeded 0.0) or recording stalled
        and splitmuxsink could not rotate. Either way the fragment's real footage
        ENDS at this boundary and is about fallback_seconds long, so the span is
        re-anchored to its end. That deliberately leaves a hole before it -- an
        honest gap the consumers already know how to refuse, rather than a
        contiguous span that lies about covering the outage.
        """
        if self._start is None:
            self._start = boundary if boundary is not None else 0.0
            return None
        start = self._start
        end = boundary if boundary is not None else start + self.fallback_seconds
        if end <= start:
            end = start + self.fallback_seconds
        elif end - start > self.max_span_factor * self.fallback_seconds:
            start = end - self.fallback_seconds
        self._start = end
        return (start, end)


ARRIVAL_PLAUSIBLE_S = 60.0     # a mux arrival stamp older than this (or in the future) is not this frame's


def arrival_wall(ntp_ns: int, use_mux_time: bool, now: float) -> float:
    """The wall time to pair with a probed frame's pts for the clock correction: when the frame entered nvstreammux
    (its system timestamp, `ntp_timestamp` with attach-sys-ts), else `now`. The probe runs after inference -- seconds
    after the frame entered the pipeline -- while the recorder samples its clock right after encoding, so pairing the
    probe's pts with `now` put that inference delay into the correction and alert-clip overlays trailed the footage by
    it (2026-10-29). `use_mux_time` is False when the stamp is a camera's own clock (rtcp_camera_clock)."""
    if use_mux_time and ntp_ns > 0:
        arrived = ntp_ns / 1e9
        if 0.0 <= now - arrived <= ARRIVAL_PLAUSIBLE_S:
            return arrived
    return now


RECORDER_SAMPLE_FRESH_S = 5.0   # an encoded-frame clock sample older than this is not the current one


def recorder_clock(sample: dict, boundary: "float | None", now: float):
    """The (pts, wall) pair for the recorder side of the clock correction: the latest encoded frame's, sampled as it
    left the parser (`sample["last"]`), else the fragment boundary's pts with `now`. splitmuxsink announces a fragment
    only once it has the next keyframe, a whole GOP after the boundary frame -- 6 s at 5 fps with a 30-frame GOP -- so
    pairing the boundary with that moment put a GOP into the correction and alert-clip overlays ran that far off
    (2026-10-29)."""
    last = sample.get("last")
    if last is not None and 0.0 <= now - last[1] <= RECORDER_SAMPLE_FRESH_S:
        return last
    return boundary, now


class StreamClockOffsets:
    """Maps the recorder's pre-mux pts onto the probe's post-mux clock, per stream.

    nvstreammux gives the record branch and the analysis probe different pts
    origins -- measured at +1.033s on one live pipeline -- so segments indexed in
    recorder pts do not line up with the alert pts the probe stamps. Both sides'
    `pts - wall_clock` is invariant while the pipeline runs, so their difference is
    the correction, and both are observable in-process.

    The offset is stabilized per stream. Small jitter (< drift_threshold, e.g.
    keyframe-snapping ~0.012s) is kept stable so consecutive spans share exact
    boundaries. When track_drift is enabled, real drift (such as source looping
    delays or clock drift exceeding drift_threshold) updates the offset dynamically
    so indexed segments stay aligned with the live probe across long runs.
    """

    def __init__(self, max_plausible_offset: float = 7200.0, drift_threshold: float = 0.5,
                 track_drift: bool = False):
        self._probe: dict = {}    # sid -> probe-side `pts - wall` constant
        self._frozen: dict = {}   # sid -> current offset
        self.max_plausible_offset = float(max_plausible_offset)
        self.drift_threshold = float(drift_threshold)
        self.track_drift = bool(track_drift)

    def needs_probe(self, sid: str) -> bool:
        """True when sampling is needed. If track_drift is false, stops once frozen."""
        if self.track_drift:
            return True
        return sid not in self._frozen

    def observe_probe(self, sid: str, pts: float, now: float) -> None:
        """Record the probe side's constant. `now` MUST be wall clock (time.time()),
        matching the recorder side -- a monotonic clock would not be comparable."""
        self._probe[sid] = pts - now

    def offset(self, sid: str, boundary: "float | None", now: float):
        """Correction to ADD to recorder pts to express it in the probe's clock.

        Returns the stabilized value once known, measures and updates it when
        both sides are available. Small jitter below drift_threshold retains the
        current offset; drift exceeding the threshold updates it.
        """
        probe_const = self._probe.get(sid)
        if probe_const is None or boundary is None:
            return self._frozen.get(sid)
        candidate = probe_const - (boundary - now)
        if abs(candidate) > self.max_plausible_offset:
            return self._frozen.get(sid)
        current = self._frozen.get(sid)
        if current is None or (self.track_drift and abs(candidate - current) > self.drift_threshold):
            self._frozen[sid] = candidate
        return self._frozen[sid]

