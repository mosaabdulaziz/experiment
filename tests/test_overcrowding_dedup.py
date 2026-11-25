"""OvercrowdingLatch: frame-level threshold-crossing alert with debounced release.

Pure Python, importable off-GPU (same convention as test_zone_dedup.py)."""

from perception.lib.overcrowding_dedup import OvercrowdingLatch


def test_fires_on_first_threshold_crossing():
    latch = OvercrowdingLatch()
    assert latch.observe(0.0, count=5, threshold=5) is True


def test_does_not_refire_while_still_above_threshold():
    latch = OvercrowdingLatch()
    assert latch.observe(0.0, count=5, threshold=5) is True
    assert latch.observe(1.0, count=6, threshold=5) is False
    assert latch.observe(2.0, count=5, threshold=5) is False


def test_below_threshold_never_fires():
    latch = OvercrowdingLatch()
    assert latch.observe(0.0, count=2, threshold=5) is False
    assert latch.observe(1.0, count=4, threshold=5) is False


def test_brief_dip_below_threshold_shorter_than_release_does_not_rearm():
    latch = OvercrowdingLatch(release_seconds=5.0)
    assert latch.observe(0.0, count=5, threshold=5) is True
    # dips below threshold for only 2s (a detector flicker), then back up
    assert latch.observe(1.0, count=4, threshold=5) is False
    assert latch.observe(2.0, count=5, threshold=5) is False   # not rearmed yet -> no refire


def test_dip_below_threshold_for_release_seconds_then_recross_rearms_and_refires():
    latch = OvercrowdingLatch(release_seconds=5.0)
    assert latch.observe(0.0, count=5, threshold=5) is True
    assert latch.observe(1.0, count=4, threshold=5) is False   # below_since = 1.0
    assert latch.observe(6.0, count=4, threshold=5) is False   # still below at 6.0 (rearmed internally)
    assert latch.observe(7.0, count=5, threshold=5) is True    # re-crosses -> refires


def test_fresh_instance_refires_independently():
    latch = OvercrowdingLatch()
    assert latch.observe(0.0, count=5, threshold=5) is True
    latch = OvercrowdingLatch()
    assert latch.observe(1.0, count=5, threshold=5) is True


def test_threshold_can_vary_per_call():
    latch = OvercrowdingLatch()
    assert latch.observe(0.0, count=3, threshold=5) is False
    assert latch.observe(1.0, count=3, threshold=3) is True    # threshold lowered externally


def test_sustain_zero_fires_immediately_like_before():
    latch = OvercrowdingLatch(sustain_seconds=0.0)
    assert latch.observe(0.0, count=5, threshold=5) is True


def test_transient_spike_does_not_fire():
    # THE test for this feature: one frame above threshold, then back below.
    # Without sustain this fires and the whole false-positive control is absent.
    latch = OvercrowdingLatch(sustain_seconds=3.0)
    assert latch.observe(0.0, count=1, threshold=1) is False
    assert latch.observe(0.5, count=0, threshold=1) is False
    assert latch.observe(1.0, count=0, threshold=1) is False


def test_sustained_condition_fires_once_threshold_elapses():
    latch = OvercrowdingLatch(sustain_seconds=3.0)
    assert latch.observe(0.0, count=1, threshold=1) is False
    assert latch.observe(2.9, count=1, threshold=1) is False
    assert latch.observe(3.0, count=1, threshold=1) is True


def test_sustain_timer_restarts_after_a_break():
    latch = OvercrowdingLatch(sustain_seconds=3.0)
    assert latch.observe(0.0, count=1, threshold=1) is False
    assert latch.observe(2.0, count=0, threshold=1) is False   # break resets the run
    assert latch.observe(4.0, count=1, threshold=1) is False   # new run starts here
    assert latch.observe(6.9, count=1, threshold=1) is False
    assert latch.observe(7.0, count=1, threshold=1) is True


def test_no_refire_while_condition_persists_after_sustained_fire():
    latch = OvercrowdingLatch(sustain_seconds=1.0, release_seconds=5.0)
    assert latch.observe(0.0, count=1, threshold=1) is False
    assert latch.observe(1.0, count=1, threshold=1) is True
    assert latch.observe(2.0, count=1, threshold=1) is False
    assert latch.observe(30.0, count=1, threshold=1) is False


def test_rearms_only_after_release_then_needs_sustain_again():
    latch = OvercrowdingLatch(sustain_seconds=1.0, release_seconds=5.0)
    assert latch.observe(0.0, count=1, threshold=1) is False
    assert latch.observe(1.0, count=1, threshold=1) is True
    assert latch.observe(2.0, count=0, threshold=1) is False
    assert latch.observe(7.0, count=0, threshold=1) is False   # re-armed at 7.0
    assert latch.observe(7.5, count=1, threshold=1) is False   # sustain restarts
    assert latch.observe(8.5, count=1, threshold=1) is True


def test_dropout_grace_zero_matches_todays_zero_tolerance():
    # Default behaviour must be unchanged: one frame below threshold restarts the run.
    latch = OvercrowdingLatch(sustain_seconds=3.0, dropout_grace_seconds=0.0)
    assert latch.observe(0.0, count=1, threshold=1) is False
    assert latch.observe(1.0, count=0, threshold=1) is False   # run broken here
    assert latch.observe(2.0, count=1, threshold=1) is False   # restarted at 2.0
    assert latch.observe(4.9, count=1, threshold=1) is False
    assert latch.observe(5.0, count=1, threshold=1) is True


def test_short_dropout_within_grace_does_not_restart_the_run():
    # THE test for this feature: a flickering detector must still confirm.
    latch = OvercrowdingLatch(sustain_seconds=3.0, dropout_grace_seconds=1.0)
    assert latch.observe(0.0, count=1, threshold=1) is False
    assert latch.observe(1.0, count=0, threshold=1) is False   # gap opens, run preserved
    assert latch.observe(1.5, count=1, threshold=1) is False   # back before grace expires
    assert latch.observe(3.0, count=1, threshold=1) is True    # original run still counts


def test_dropout_longer_than_grace_restarts_the_run():
    latch = OvercrowdingLatch(sustain_seconds=3.0, dropout_grace_seconds=1.0)
    assert latch.observe(0.0, count=1, threshold=1) is False
    assert latch.observe(1.0, count=0, threshold=1) is False
    assert latch.observe(2.5, count=0, threshold=1) is False   # gap exceeded grace
    assert latch.observe(3.0, count=1, threshold=1) is False   # fresh run starts at 3.0
    assert latch.observe(5.9, count=1, threshold=1) is False
    assert latch.observe(6.0, count=1, threshold=1) is True


def test_repeated_short_gaps_still_accumulate_toward_the_window():
    latch = OvercrowdingLatch(sustain_seconds=3.0, dropout_grace_seconds=1.0)
    assert latch.observe(0.0, count=1, threshold=1) is False
    for t in (0.5, 1.2, 1.9, 2.6):
        latch.observe(t, count=0, threshold=1)
        latch.observe(t + 0.2, count=1, threshold=1)
    assert latch.observe(3.0, count=1, threshold=1) is True


def test_run_started_at_is_the_true_onset_not_the_confirming_now():
    latch = OvercrowdingLatch(sustain_seconds=3.0)
    assert latch.run_started_at is None   # nothing observed yet
    latch.observe(0.0, count=1, threshold=1)
    assert latch.run_started_at == 0.0    # run open, not yet confirmed
    assert latch.observe(3.0, count=1, threshold=1) is True
    assert latch.run_started_at == 0.0    # confirmed call -- true onset, not 3.0


def test_run_ended_at_is_the_true_end_at_the_rearming_call():
    latch = OvercrowdingLatch(sustain_seconds=1.0, release_seconds=5.0)
    assert latch.observe(0.0, count=1, threshold=1) is False
    assert latch.observe(1.0, count=1, threshold=1) is True
    assert latch.run_ended_at is None      # still active, no drop yet
    assert latch.armed is False
    assert latch.observe(2.0, count=0, threshold=1) is False   # below_since = 2.0
    assert latch.armed is False            # not yet re-armed (release_seconds not elapsed)
    assert latch.observe(7.0, count=0, threshold=1) is False   # re-arms here
    assert latch.armed is True
    assert latch.run_ended_at == 2.0       # true end, not 7.0


def test_armed_property_tracks_the_same_transitions_observe_reports():
    latch = OvercrowdingLatch(sustain_seconds=0.0, release_seconds=5.0)
    assert latch.armed is True
    assert latch.observe(0.0, count=1, threshold=1) is True
    assert latch.armed is False
    assert latch.observe(1.0, count=0, threshold=1) is False   # below_since = 1.0
    assert latch.armed is False
    assert latch.observe(6.0, count=0, threshold=1) is False   # re-arms (release elapsed)
    assert latch.armed is True


def test_stray_single_frame_recurrence_during_release_does_not_reset_countdown():
    # Traced defect: a single unsustained recurrence during the release countdown
    # used to reset it from zero -- a flicker, ember, or glare after a fire/smoke
    # incident's real end would keep the incident "ongoing" indefinitely as long as
    # noise like that recurred at least once every release_seconds.
    latch = OvercrowdingLatch(sustain_seconds=2.0, release_seconds=5.0)
    assert latch.observe(0.0, count=1, threshold=1) is False
    assert latch.observe(2.0, count=1, threshold=1) is True    # fires; below_since will be 3.0
    assert latch.observe(3.0, count=0, threshold=1) is False   # release countdown starts
    assert latch.observe(4.0, count=1, threshold=1) is False   # stray single frame, unsustained
    assert latch.observe(4.5, count=0, threshold=1) is False   # back below immediately
    assert latch.armed is False
    assert latch.observe(7.9, count=0, threshold=1) is False   # still short of 3.0+5.0=8.0
    assert latch.armed is False
    assert latch.observe(8.0, count=0, threshold=1) is False   # re-arms at the ORIGINAL deadline
    assert latch.armed is True


def test_recurrence_sustained_long_enough_does_reset_the_countdown():
    # The mirror case: a recurrence that itself clears the sustain bar is a real
    # second incident, and legitimately cancels the release countdown in progress.
    latch = OvercrowdingLatch(sustain_seconds=2.0, release_seconds=5.0)
    assert latch.observe(0.0, count=1, threshold=1) is False
    assert latch.observe(2.0, count=1, threshold=1) is True    # fires
    assert latch.observe(3.0, count=0, threshold=1) is False   # release countdown starts (below_since=3.0)
    assert latch.observe(4.0, count=1, threshold=1) is False   # a real second run begins
    assert latch.observe(6.0, count=1, threshold=1) is False   # sustained 2.0s -> legitimately cancels it
    assert latch.observe(7.9, count=1, threshold=1) is False   # original 3.0+5.0=8.0 deadline passes, no effect
    assert latch.armed is False
    assert latch.observe(10.0, count=0, threshold=1) is False  # the real run's true end (below_since=10.0)
    assert latch.observe(14.9, count=0, threshold=1) is False  # short of the NEW deadline, 10.0+5.0=15.0
    assert latch.armed is False
    assert latch.observe(15.0, count=0, threshold=1) is False  # re-arms at the new deadline
    assert latch.armed is True


def test_rearm_with_long_grace_does_not_skip_the_sustain_window():
    # Traced defect: dropout_grace_seconds > release_seconds lets the latch re-arm
    # while _at_since is still stale from the original run, so the very next
    # above-threshold frame fires instantly instead of waiting out sustain again.
    latch = OvercrowdingLatch(release_seconds=5.0, sustain_seconds=3.0,
                               dropout_grace_seconds=20.0)
    assert latch.observe(0.0, count=1, threshold=1) is False
    assert latch.observe(3.0, count=1, threshold=1) is True    # original fire
    assert latch.observe(4.0, count=0, threshold=1) is False   # gap opens
    assert latch.observe(9.0, count=0, threshold=1) is False   # re-armed at 9.0 (gap=5>=release)
    assert latch.observe(10.0, count=1, threshold=1) is False  # must NOT fire instantly
    assert latch.observe(12.9, count=1, threshold=1) is False
    assert latch.observe(13.0, count=1, threshold=1) is True   # fires only after sustain again
