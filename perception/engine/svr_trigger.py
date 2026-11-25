"""Per-source decision: should this alert start a new SVR recording?

DS 9.0's `start-sr` action signal returns void, so a start issued while a session
is already running cannot be detected at the call site. The readable
`smart-rec-status` property is the only way to know, and NVIDIA's own reference
gates on the equivalent flag (deepstream_source_bin.c:642 checks ctx->recordOn
before calling NvDsSRStart). Every emit must therefore be guarded.

That guard doubles as the coalescing window: a recording started with
start-sr(startTime=pad, duration=2*pad) already covers [T-pad, T+pad], so a burst
of zone entries inside it belongs on that one clip. Coalescing is capped at
`coalesce_seconds` after the session start so an alert arriving near the end of a
recording is not attached to a clip that gives it no post-context -- the same
reasoning behind alerts.clip.coalesce_seconds staying below pad_seconds app-side.

Because the guard permits only one session per source at a time, an `sr-done` on a
given nvurisrcbin can only refer to the session started on it. That is what makes
the sessionId out-param unnecessary for correlating a finished file to its alert.

`min_gap_seconds` exists for a different reason -- not correctness, mitigation.
Traced live (gdb, symbol-resolved): DeepStream's SmartRecord element
(libnvdsgst_smartrecord.so) calls g_signal_emit_by_name() into the source's
nvurisrcbin (libnvdsgst_nvurisrcbin.so) from a background "enc_que1:src" thread --
a use-after-free in NVIDIA's closed-source library when that call lands after the
target GObject is already gone, segfaulting inside g_type_check_instance(). An
alert incident that stays open re-arms a new session the instant the previous
one's smart-rec-status flips false (see a module's open-incident recheck
loop), which is exactly the back-to-back cycling that keeps SmartRecord's
background threads maximally busy and gives this race the most opportunities to
strike. This can't be fixed here -- the bug is in NVIDIA's binary -- but a deliberate
gap between one session's close and the next one's start gives that background
thread more room to actually finish before the next session's teardown/build churn
starts. Not a guaranteed fix, only a frequency reduction. 0.0 (the default)
preserves the original instant-rearm behavior byte-for-byte.

Pure Python: importable off-GPU, unit-tested, no gi/pyds.
"""


class SvrTrigger:
    """One instance per source. `observe` is called once per fired alert."""

    def __init__(self, coalesce_seconds: float, min_gap_seconds: float = 0.0):
        self._coalesce = max(float(coalesce_seconds), 0.0)
        self._min_gap = max(float(min_gap_seconds), 0.0)
        self._session_start = None
        self._closed_at = None   # when recording_active last went True -> False

    def closed(self, now: float) -> None:
        """The session ended at `now` (its sr-done, on observe's clock): the gap counts from here. Without it the close
        was only noticed by the next alert, which then always fell inside the gap -- the first alert after any quiet
        spell lost its clip."""
        self._closed_at = now

    def observe(self, now: float, recording_active: bool) -> str:
        """Return "start" (emit start-sr), "coalesce" (attach to the running
        session's clip), "cooldown" (closed too recently -- see min_gap_seconds),
        or "skip" (no clip for this alert).

        `recording_active` is the source's live `smart-rec-status`.
        """
        if recording_active:
            self._closed_at = None   # still open -- not "closed" until it ends
            if self._session_start is None:
                # A session we did not start -- no anchor, so nothing to coalesce onto.
                return "skip"
            if now - self._session_start <= self._coalesce:
                return "coalesce"
            return "skip"
        if self._session_start is None:
            # Never started a session -- nothing to cool down from.
            self._session_start = now
            return "start"
        if self._closed_at is None:
            self._closed_at = now
        if now - self._closed_at < self._min_gap:
            return "cooldown"
        self._session_start = now
        self._closed_at = None
        return "start"


def svr_window_for_age(age: float, pad: float, duration: float, cache: float):
    """`(start_time, forward)` for start-sr covering a moment `age` seconds old, or
    None when that moment has already fallen out of the pre-roll cache.

    Module alerts come from the buffer probe on the very frame that triggered them, so
    their age is 0. A PROACTIVE VLM alert is different: it is only known after the
    batch was sampled and the model answered, so by request time the moment is
    several seconds in the past and a recording starting `pad` back would miss it
    entirely.

    `start_time` therefore grows with age, clamped to what `smart-rec-cache` actually
    holds -- past the clamp the clip keeps the moment but loses some lead-in, which is
    strictly better than starting after it. Once the moment itself is older than the
    cache, no recording can contain it and None says so rather than writing a clip of
    the wrong footage.

    `forward` (the second element) is `duration` MINUS however far back `start_time`
    already reaches, floored at 1s. DeepStream counts its `duration` argument as
    wall-clock recording time from the call, NOT as the total length of the file:
    sr-done was observed firing exactly that many seconds after every start-sr
    regardless of start_time, so the file spans [now - start_time, now + forward] and
    the TOTAL span is start_time + forward, not forward alone. Passing `duration`
    through unchanged (the previous behaviour) therefore grew the file with age --
    even a fresh zone alert (age=0, start_time=pad) ran pad+duration long, e.g. 15s
    for the 5/10 defaults, not the 10s `duration` is documented (config comment) as
    meaning. Shrinking `forward` by `start_time` pins the total at the configured
    `duration` whenever the pre-roll can supply it (age small enough that
    start_time <= duration - 1); a sufficiently stale alert still grows past it
    because there is no way to shrink pre-roll below what is needed to keep the
    alert moment itself in frame, but far less than before.
    """
    age = max(float(age), 0.0)
    if age >= cache:
        return None
    start_time = min(age + pad, max(cache - 1.0, 0.0))
    forward = max(duration - start_time, 1.0)
    return start_time, forward
