"""source_recovery: deciding when a bus message means one source is dead.

The strings here are verbatim from a deployment where a Smart Record branch
refused a mid-stream caps change (go2rtc advertised H.265 parameter sets in its
SDP that did not match the ones it sent in-band; qtmux cannot renegotiate a
configured track). The NOT_NEGOTIATED travelled back through nvurisrcbin's
internal pre-decode tee and stopped rtspsrc, and nvurisrcbin's own reconnect
could never recover it -- the camera was lost until the sidecar restarted.
"""

from perception.engine.source_recovery import (MainLoopWatchdog, SourceRestartPolicy, SourceStallWatch,
                                                 dead_source_index)


RECORD_BIN_ERROR = (
    "../plugins/elements/gstqueue.c(1035): gst_queue_handle_sink_event (): "
    "/GstPipeline:pipeline0/GstDsNvUriSrcBin:src_4/GstBin:record_bin6/"
    "GstBin:enc_bin6/GstQueue:enc_que6:\n"
    "streaming stopped, reason not-negotiated (-4)"
)


TRANSIENT_READ_WARNING = (
    "../gst/rtsp/gstrtspsrc.c(5993): gst_rtspsrc_loop_interleaved (): "
    "/GstPipeline:pipeline0/GstDsNvUriSrcBin:src_4/GstRTSPSrc:src:"
)

WATCHDOG_WARNING = (
    "gstdsnvurisrcbin.cpp(1710): watch_source_status (): "
    "/GstPipeline:pipeline0/GstDsNvUriSrcBin:src_4"
)


def test_a_record_branch_failure_names_the_source_it_killed():
    assert dead_source_index(RECORD_BIN_ERROR) == 4


def test_a_transient_read_error_is_not_fatal():
    """Every source posted this at once when go2rtc dropped its connections, and
    rtspsrc's own reconnect recovered all of them. Restarting here would turn a
    self-healing blip into a fleet-wide teardown."""
    assert dead_source_index(TRANSIENT_READ_WARNING) is None


def test_the_reconnect_watchdog_is_not_fatal():
    """nvurisrcbin posts this immediately before resetting a source itself. It is
    a report that recovery is already under way, not a request for ours."""
    assert dead_source_index(WATCHDOG_WARNING) is None


class TestRestartPolicy:
    """One failure posts several fatal messages -- rtspsrc stops immediately, the
    record bin's queue reports it again when the session's duration expires -- and
    the reconnect loop keeps posting after that. All of it describes ONE incident."""

    def test_the_first_report_restarts_the_source(self):
        policy = SourceRestartPolicy(cooldown_seconds=30.0)
        assert policy.observe(now=100.0, index=4) is True

    def test_a_second_report_of_the_same_incident_is_suppressed(self):
        policy = SourceRestartPolicy(cooldown_seconds=30.0)
        policy.observe(now=100.0, index=4)
        assert policy.observe(now=110.0, index=4) is False

    def test_another_source_failing_meanwhile_still_restarts(self):
        policy = SourceRestartPolicy(cooldown_seconds=30.0)
        policy.observe(now=100.0, index=4)
        assert policy.observe(now=100.5, index=7) is True

    def test_a_fresh_failure_after_the_cooldown_restarts_again(self):
        """A camera that breaks again an hour later must not be written off."""
        policy = SourceRestartPolicy(cooldown_seconds=30.0)
        policy.observe(now=100.0, index=4)
        assert policy.observe(now=131.0, index=4) is True


class TestStalledSources:
    """A source whose decoder stops handing frames to the mux while its RTSP
    session stays up (observed 2026-09-24: barber-r, then barber-l and barber-c
    in other runs) posts nothing on the bus -- rtspsrc keeps reading, so
    nvurisrcbin's "No data from source" watchdog stays quiet. Only the absence
    of its frames after the mux can tell."""

    def test_a_source_that_stops_delivering_is_reported(self):
        watch = SourceStallWatch(stall_seconds=10.0)
        watch.note_buffer(0, now=100.0)
        watch.note_buffer(2, now=100.0)
        for t in range(101, 112):
            watch.note_buffer(0, now=float(t))
        assert watch.stalled(now=109.0).restart == []
        assert watch.stalled(now=111.0).restart == [2]

    def test_a_source_that_keeps_delivering_is_not(self):
        watch = SourceStallWatch(stall_seconds=10.0)
        for t in range(100, 130):
            watch.note_buffer(0, now=float(t))
            watch.note_buffer(2, now=float(t) if t < 115 else 114.0)
        assert watch.stalled(now=130.0).restart == [2]

    def test_a_source_that_never_started_is_not_stalled(self):
        """Before its first frame a source is still connecting; nvurisrcbin's
        own reconnect owns that phase."""
        watch = SourceStallWatch(stall_seconds=10.0)
        watch.note_buffer(0, now=100.0)
        watch.note_buffer(2, now=100.0)
        for t in range(101, 500):
            watch.note_buffer(2, now=float(t))
        assert watch.stalled(now=111.0).restart == [0]
        for now in (200.0, 300.0, 400.0, 500.0):
            assert 1 not in watch.stalled(now=now).restart

    def test_a_dead_camera_is_retried_with_backoff(self):
        """A camera that is really gone must not be restarted every period
        forever: each restart also resets nvurisrcbin's own reconnect cycle."""
        watch = SourceStallWatch(stall_seconds=10.0, max_backoff_seconds=40.0)
        watch.note_buffer(0, now=100.0)
        watch.note_buffer(2, now=100.0)
        reports = []
        for t in range(101, 300):
            watch.note_buffer(0, now=float(t))
            if watch.stalled(now=float(t)).restart:
                reports.append(t)
        # silent 10 s, then 20, 40, then capped at 40
        assert reports == [111, 132, 173, 214, 255, 296]

    def test_a_recovered_source_is_healthy_again_and_its_backoff_resets(self):
        watch = SourceStallWatch(stall_seconds=10.0)
        watch.note_buffer(0, now=100.0)
        watch.note_buffer(2, now=100.0)
        for t in range(101, 200):
            watch.note_buffer(0, now=float(t))
        assert watch.stalled(now=111.0).restart == [2]
        assert watch.stalled(now=132.0).restart == [2]     # backed off to 20 s
        watch.note_buffer(2, now=133.0)
        assert watch.stalled(now=140.0).restart == []
        assert watch.stalled(now=144.0).restart == [2]     # 10 s again, not 40

    def test_every_source_silent_at_once_is_a_downstream_stall(self):
        """Nothing after the mux means inference/probe stopped, not the
        cameras: restarting every source would not help."""
        watch = SourceStallWatch(stall_seconds=10.0)
        for i in (0, 1, 2):
            watch.note_buffer(i, now=100.0)
        check = watch.stalled(now=111.0)
        assert check.restart == [] and check.downstream is True
        assert watch.stalled(now=115.0).downstream is False   # re-armed, not every poll

    def test_a_single_source_can_still_be_restarted(self):
        watch = SourceStallWatch(stall_seconds=10.0)
        watch.note_buffer(0, now=100.0)
        check = watch.stalled(now=111.0)
        assert check.restart == [0] and check.downstream is False

    def test_reset_rearms_every_source(self):
        """A paused pipeline delivers nothing on purpose; on resume the old
        timestamps must not count as a stall."""
        watch = SourceStallWatch(stall_seconds=10.0)
        watch.note_buffer(0, now=100.0)
        watch.note_buffer(2, now=100.0)
        watch.reset(now=500.0)
        assert watch.stalled(now=505.0).restart == []
        watch.note_buffer(0, now=511.0)
        assert watch.stalled(now=511.0).restart == [2]

    def test_zero_disables_the_watch(self):
        watch = SourceStallWatch(stall_seconds=0)
        watch.note_buffer(2, now=100.0)
        assert watch.stalled(now=10_000.0).restart == []


class TestStallEscalation:
    """2026-09-26 13:34 and 13:51: the test streams were restarted, barber-c and
    barber-l never delivered again, and the stall watch restarted them every
    1-3 min for good -- no restart helped, the loop was not wedged, and only a
    full sidecar restart brought them back. After K restarts of one source
    with no frame in between, the watch gives up on in-process recovery."""

    @staticmethod
    def _silent_2(watch, until, start=101):
        """Source 0 keeps delivering, source 2 stays silent; the polls' results."""
        out = []
        for t in range(start, until):
            watch.note_buffer(0, now=float(t))
            check = watch.stalled(now=float(t))
            if check.restart or check.hopeless:
                out.append((t, check.restart, check.hopeless))
        return out

    def test_off_by_default(self):
        watch = SourceStallWatch(stall_seconds=10.0, max_backoff_seconds=40.0)
        watch.note_buffer(0, now=100.0)
        watch.note_buffer(2, now=100.0)
        assert all(h == () for _t, _r, h in self._silent_2(watch, 1000))

    def test_the_report_after_k_failed_restarts_gives_up(self):
        watch = SourceStallWatch(stall_seconds=10.0, max_backoff_seconds=40.0,
                                 max_failed_restarts=2)
        watch.note_buffer(0, now=100.0)
        watch.note_buffer(2, now=100.0)
        # restart at 111 and 132 (backoff 20 s); no frame -> give up at 173
        assert self._silent_2(watch, 180) == [
            (111, [2], ()), (132, [2], ()), (173, [], (2,))]

    def test_a_restart_that_brings_frames_back_resets_the_count(self):
        watch = SourceStallWatch(stall_seconds=10.0, max_failed_restarts=2)
        watch.note_buffer(0, now=100.0)
        watch.note_buffer(2, now=100.0)
        assert self._silent_2(watch, 140) == [(111, [2], ()), (132, [2], ())]
        watch.note_buffer(2, now=140.0)           # the second restart worked
        # it stalls again later: two fresh restarts before giving up
        assert self._silent_2(watch, 200, start=141) == [
            (151, [2], ()), (172, [2], ())]

    def test_restarts_older_than_the_window_do_not_count(self):
        watch = SourceStallWatch(stall_seconds=10.0, max_backoff_seconds=40.0,
                                 max_failed_restarts=2,
                                 failed_restart_window_seconds=30.0)
        watch.note_buffer(0, now=100.0)
        watch.note_buffer(2, now=100.0)
        # 111 and 132 fall inside 30 s of each other, but at 173 the one at
        # 111 is 62 s old: only one recent failure, so restart again
        got = self._silent_2(watch, 180)
        assert got[2] == (173, [2], ())

    def test_one_camera_giving_up_does_not_count_for_another(self):
        watch = SourceStallWatch(stall_seconds=10.0, max_failed_restarts=2)
        for i in (0, 1, 2):
            watch.note_buffer(i, now=100.0)
        for t in range(101, 120):
            watch.note_buffer(0, now=float(t))
            watch.note_buffer(1, now=float(t))
        assert watch.stalled(now=111.0).restart == [2]
        watch.note_buffer(1, now=121.0)          # 1 last seen at 121 ...
        for t in range(121, 133):
            watch.note_buffer(0, now=float(t))
        assert watch.stalled(now=132.0).restart == [1, 2]
        got = []
        for t in range(133, 181):
            watch.note_buffer(0, now=float(t))
            check = watch.stalled(now=float(t))
            if check.restart or check.hopeless:
                got.append((t, check.restart, check.hopeless))
        # 2 gives up on its third report; 1 is only on its second
        assert got == [(153, [1], ()), (173, [], (2,))]

    def test_every_source_silent_k_plus_one_times_gives_up_downstream(self):
        """All cameras dead at once (the whole stream group restarted, a
        wedged inference stage): nothing is restarted, so only the repeated
        report can tell it is not clearing."""
        watch = SourceStallWatch(stall_seconds=10.0, max_failed_restarts=2)
        for i in (0, 1, 2):
            watch.note_buffer(i, now=100.0)
        seen = []
        for t in range(101, 150):
            check = watch.stalled(now=float(t))
            if check.downstream or check.hopeless_downstream:
                seen.append((t, check.downstream, check.hopeless_downstream))
        assert seen[:3] == [(111, True, False), (122, True, False), (133, False, True)]

    def test_any_frame_clears_the_downstream_count(self):
        watch = SourceStallWatch(stall_seconds=10.0, max_failed_restarts=2)
        for i in (0, 1, 2):
            watch.note_buffer(i, now=100.0)
        assert watch.stalled(now=111.0).downstream
        assert watch.stalled(now=122.0).downstream
        for i in (0, 1, 2):
            watch.note_buffer(i, now=123.0)
        check = watch.stalled(now=134.0)
        assert check.downstream and not check.hopeless_downstream

    def test_reset_clears_the_count(self):
        """A pause is not a failed restart."""
        watch = SourceStallWatch(stall_seconds=10.0, max_failed_restarts=1)
        watch.note_buffer(0, now=100.0)
        watch.note_buffer(2, now=100.0)
        assert self._silent_2(watch, 112) == [(111, [2], ())]
        watch.reset(now=200.0)
        assert self._silent_2(watch, 212, start=201) == [(211, [2], ())]


class TestMainLoopWatchdog:
    """nvurisrcbin's own reconnect runs on the GLib main loop and can deadlock
    it (2026-09-24 20:41: rtspsrc's pad deactivation waited forever on a
    streaming thread stuck behind a starved decoder). Every recovery path --
    the stall watch, the bus handler -- is a main-loop callback, so only a
    thread outside the loop can notice."""

    def test_a_beating_loop_is_not_wedged(self):
        dog = MainLoopWatchdog(timeout_seconds=60.0, now=0.0)
        for t in range(1, 200):
            dog.beat(now=float(t))
        assert dog.wedged(now=200.0) is False

    def test_a_loop_that_stops_beating_is_wedged(self):
        dog = MainLoopWatchdog(timeout_seconds=60.0, now=0.0)
        dog.beat(now=10.0)
        assert dog.wedged(now=69.0) is False
        assert dog.wedged(now=71.0) is True

    def test_it_counts_from_its_creation_before_the_first_beat(self):
        dog = MainLoopWatchdog(timeout_seconds=60.0, now=100.0)
        assert dog.wedged(now=150.0) is False
        assert dog.wedged(now=161.0) is True

    def test_zero_disables_it(self):
        dog = MainLoopWatchdog(timeout_seconds=0, now=0.0)
        assert dog.wedged(now=1e9) is False

    def test_it_is_polled_often_enough_for_a_short_timeout(self):
        # A wedge never clears by itself (61 of 61 on 2026-09-25/26 ran the
        # full timeout), so the exit should follow the timeout closely: the
        # poll adds at most a tenth of it, 1 to 5 s.
        assert MainLoopWatchdog(timeout_seconds=60.0, now=0.0).poll_seconds == 5.0
        assert MainLoopWatchdog(timeout_seconds=20.0, now=0.0).poll_seconds == 2.0
        assert MainLoopWatchdog(timeout_seconds=5.0, now=0.0).poll_seconds == 1.0
