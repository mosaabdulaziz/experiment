"""Overcrowding alert dedup: frame-level threshold-crossing with debounced release.

Pure Python (no gi/pyds imports) so it stays unit-testable off-GPU. Unlike
ZoneReentryGrace (per-object, edge-triggered on entry), overcrowding is a frame-level
*state* -- there is no single object to attach the alert to, so the natural dedup
question is "when does this state fire again," not "which object hasn't alerted yet."
"""


class OvercrowdingLatch:
    """Per-stream/per-ROI edge-triggered alert with debounced release and sustain.

    Fires when the in-zone count reaches the threshold and STAYS at or above it
    continuously for ``sustain_seconds``. Won't fire again until the count has stayed
    continuously below the threshold for ``release_seconds`` -- without this, a count
    hovering right at the threshold (5, 6, 5, 6, ...) would fire on every single frame.

    The sustain requirement is the primary false-positive control for fire/smoke detection:
    a single flickering false detection (steam, headlight, sun glare) reaching threshold
    for one frame would raise an alarm without it. Set ``sustain_seconds=0.0`` to restore
    the original overcrowding behavior (fire on first crossing).
    """

    def __init__(self, release_seconds: float = 5.0, sustain_seconds: float = 0.0,
                 dropout_grace_seconds: float = 0.0):
        self.release_seconds = float(release_seconds)
        # How long the count must stay at/above threshold CONTINUOUSLY before firing.
        # 0.0 = fire on the first crossing (overcrowding's original behaviour, kept as
        # the default so existing callers are unaffected). Fire/smoke set this well
        # above zero: a detector flicker on steam or a headlight reaches threshold for
        # one frame, and without a sustain requirement that is an alarm.
        self.sustain_seconds = float(sustain_seconds)
        # How long the condition may be ABSENT without breaking an in-progress sustain
        # run. 0.0 = zero tolerance (the original behaviour, kept as the default so
        # overcrowding is untouched). Detectors flicker -- smoke especially -- so with
        # zero tolerance an 8s window at 10fps demands 80 consecutive hits and may never
        # be met. The sustain timer keeps accruing across a tolerated gap, so this is a
        # duty-cycle allowance, not an extension of the window.
        self.dropout_grace_seconds = float(dropout_grace_seconds)
        self._armed = True
        self._below_since: "float | None" = None
        self._at_since: "float | None" = None   # start of the current at/above run
        self._dropout_since: "float | None" = None   # start of the current tolerated gap

    def observe(self, now: float, count: int, threshold: int) -> bool:
        """Returns True exactly on the frame this ROI transitions into 'overcrowded'
        (or, for fire/smoke, into a confirmed detection state)."""
        if count >= threshold:
            if self._armed:
                self._below_since = None
                self._dropout_since = None
            if self._at_since is None:
                self._at_since = now
            if not self._armed:
                # Already latched open (mid release countdown, or between release
                # and re-arm): a recurrence only cancels the release countdown once
                # IT clears the same sustain bar that opened the incident. Without
                # this, a single stray frame -- a flicker, an ember, a glare --
                # would reset the countdown from zero, and the incident would never
                # resolve as long as noise like that recurs at least once every
                # release_seconds. (sustain_seconds == 0.0 makes this check pass
                # instantly, same as before -- plain overcrowding is unaffected.)
                if now - self._at_since >= self.sustain_seconds:
                    self._below_since = None
                    self._dropout_since = None
                return False
            if now - self._at_since >= self.sustain_seconds:
                self._armed = False
                return True
            return False

        if self._at_since is not None:
            if self._dropout_since is None:
                self._dropout_since = now
            if now - self._dropout_since >= self.dropout_grace_seconds:
                self._at_since = None
                self._dropout_since = None

        if self._below_since is None:
            self._below_since = now
            if not self._armed:
                # The release countdown is starting: forget the ended run's onset
                # right away rather than waiting out dropout grace, so a stray
                # recurrence in that window is measured against ITS OWN start (see
                # the >= threshold branch above), not against a stale onset that
                # may already exceed sustain_seconds on its own.
                self._at_since = None
                self._dropout_since = None
        elif now - self._below_since >= self.release_seconds:
            self._armed = True
            # A re-arm must start a fresh sustain window. Without this, a stale
            # _at_since from a run that never got dropout-cleared (grace >=
            # release -- the gap re-arms before the dropout path clears it) lets
            # the very next above-threshold frame compute elapsed time against
            # that stale start and fire instantly, skipping sustain entirely.
            self._at_since = None
            self._dropout_since = None
        return False

    @property
    def armed(self) -> bool:
        """Whether this latch can still fire (False from the confirming `observe()`
        call until a full release_seconds of continuous absence re-arms it)."""
        return self._armed

    @property
    def run_started_at(self) -> "float | None":
        """`now` of the first frame in the CURRENT continuous at/above-threshold
        run, or None outside one. Read this the instant `observe()` returns True:
        it is the condition's true onset, sustain_seconds (or more, if dropout
        grace tolerated gaps) before the confirming call -- callers wanting an
        incident's real start time want this, not the confirming `now`."""
        return self._at_since

    @property
    def run_ended_at(self) -> "float | None":
        """`now` of the first frame the condition went (continuously) below
        threshold in the run that is re-arming, or None if not currently in a
        release countdown. Meaningful right after an `observe()` call re-arms
        the latch (see `armed`): the condition's true end, release_seconds
        before that re-arming call."""
        return self._below_since
