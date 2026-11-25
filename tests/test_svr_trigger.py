"""SvrTrigger: whether an alert should start a new SVR recording.

start-sr returns void in DS 9.0 -- a start issued while a session is already
running cannot be detected at the call site -- so every emit must be gated on the
readable smart-rec-status property. That same gate doubles as the coalescing
window, since an in-flight recording already covers the burst."""

from perception.engine.svr_trigger import SvrTrigger, svr_window_for_age


def test_first_alert_fires():
    t = SvrTrigger(coalesce_seconds=2.5)
    assert t.observe(100.0, recording_active=False) == "start"


def test_alert_during_the_coalesce_window_shares_the_recording():
    t = SvrTrigger(coalesce_seconds=2.5)
    assert t.observe(100.0, recording_active=False) == "start"
    assert t.observe(101.0, recording_active=True) == "coalesce"
    assert t.observe(102.5, recording_active=True) == "coalesce"   # boundary is inclusive


def test_alert_late_in_an_in_flight_recording_is_skipped():
    """Coalescing onto a clip that is about to end would give the alert no
    post-context, so it is refused rather than attached."""
    t = SvrTrigger(coalesce_seconds=2.5)
    t.observe(100.0, recording_active=False)
    assert t.observe(104.0, recording_active=True) == "skip"


def test_a_new_alert_after_the_recording_ends_fires_again():
    t = SvrTrigger(coalesce_seconds=2.5)
    t.observe(100.0, recording_active=False)
    # smart-rec-status has gone false: the session finished.
    assert t.observe(115.0, recording_active=False) == "start"


def test_fire_resets_the_coalesce_window():
    t = SvrTrigger(coalesce_seconds=2.5)
    t.observe(100.0, recording_active=False)
    t.observe(115.0, recording_active=False)          # second fire
    assert t.observe(116.0, recording_active=True) == "coalesce"   # measured from 115, not 100
    assert t.observe(118.0, recording_active=True) == "skip"


def test_recording_active_without_a_prior_fire_does_not_coalesce():
    """A recording we did not start (a leftover session from before a reload, say)
    has no anchor to coalesce against, so alerts are skipped rather than attached
    to a window we know nothing about."""
    t = SvrTrigger(coalesce_seconds=2.5)
    assert t.observe(100.0, recording_active=True) == "skip"


def test_zero_coalesce_seconds_disables_sharing():
    t = SvrTrigger(coalesce_seconds=0.0)
    t.observe(100.0, recording_active=False)
    assert t.observe(100.0, recording_active=True) == "coalesce"   # same instant still shares
    assert t.observe(100.1, recording_active=True) == "skip"


# ── min_gap_seconds (mitigation for the traced SmartRecord/nvurisrcbin
# use-after-free -- see the module docstring) ──────────────────────────────

def test_min_gap_zero_fires_instantly_like_before():
    t = SvrTrigger(coalesce_seconds=2.5, min_gap_seconds=0.0)
    t.observe(100.0, recording_active=False)
    assert t.observe(115.0, recording_active=False) == "start"


def test_min_gap_withholds_fire_until_the_cooldown_elapses():
    t = SvrTrigger(coalesce_seconds=2.5, min_gap_seconds=3.0)
    assert t.observe(100.0, recording_active=False) == "start"   # first session, no prior close
    assert t.observe(101.0, recording_active=True) == "coalesce"
    assert t.observe(115.0, recording_active=False) == "cooldown"   # just closed
    assert t.observe(117.9, recording_active=False) == "cooldown"   # still short of 115+3.0
    assert t.observe(118.0, recording_active=False) == "start"       # cooldown elapsed


def test_min_gap_does_not_delay_the_very_first_ever_session():
    # No prior close to cool down from -- must fire immediately, same as
    # min_gap=0.0, regardless of how large min_gap_seconds is configured.
    t = SvrTrigger(coalesce_seconds=2.5, min_gap_seconds=30.0)
    assert t.observe(0.0, recording_active=False) == "start"


def test_min_gap_only_starts_counting_once_the_session_actually_closes():
    # A session that stays open (recording_active=True) for a while must not have
    # its cooldown clock running against some earlier moment.
    t = SvrTrigger(coalesce_seconds=2.5, min_gap_seconds=3.0)
    t.observe(100.0, recording_active=False)
    assert t.observe(105.0, recording_active=True) == "skip"    # past coalesce, still open
    assert t.observe(110.0, recording_active=True) == "skip"    # still open
    assert t.observe(110.0, recording_active=False) == "cooldown"   # closes NOW, at 110
    assert t.observe(112.9, recording_active=False) == "cooldown"   # short of 110+3.0
    assert t.observe(113.0, recording_active=False) == "start"


def test_negative_coalesce_seconds_is_clamped_not_trusted():
    """A misconfigured negative window must not make every in-flight alert skip in
    a way that looks like a coalescing bug."""
    t = SvrTrigger(coalesce_seconds=-5.0)
    t.observe(100.0, recording_active=False)
    assert t.observe(100.0, recording_active=True) == "coalesce"


# ── start-sr window for a late (proactive VLM) alert ──────────────────────────
# Zone alerts fire on the triggering frame; a proactive alert is only known after
# the VLM answers, so its moment is already seconds in the past by request time.
#
# `duration` (the config knob, documented as the TOTAL clip length) must be honoured
# as start_time + forward, not as `forward` alone -- DeepStream counts the second
# start-sr argument as wall-clock recording time from the call, not total file
# length, so passing `duration` straight through let the file grow with age.

def test_a_zone_alert_gets_exactly_the_configured_total_length():
    start, forward = svr_window_for_age(0.0, pad=5.0, duration=10.0, cache=15.0)
    assert start == 5.0
    assert forward == 5.0, "forward is shrunk by start_time so start+forward == duration(10)"
    assert start + forward == 10.0


def test_a_late_alert_reaches_further_back_and_shrinks_the_forward_portion():
    """Observed on DS 9.0: sr-done fires exactly the `forward` argument's number of
    seconds after start-sr whatever start_time is, so the file is
    [now - start_time, now + forward] and the total span is start_time + forward.
    Reaching back further must shrink forward, or the total keeps growing with age."""
    start, forward = svr_window_for_age(8.0, pad=5.0, duration=10.0, cache=15.0)
    assert start == 13.0, "must reach back past the moment, not just pad before now"
    assert start - 8.0 == 5.0, "the moment keeps its full pad of lead-in"
    assert forward == 1.0, "duration(10) - start_time(13) floors at the 1s minimum"
    assert start + forward == 14.0, "total span grows only because start_time already"\
        " exceeds the configured duration -- there is no pre-roll left to trade away"


def test_the_reach_is_clamped_to_what_the_cache_actually_holds():
    start, forward = svr_window_for_age(12.0, pad=5.0, duration=10.0, cache=15.0)
    assert start == 14.0, "cannot ask for pre-roll the cache never held"
    assert start > 12.0, "but must still reach back past the moment itself"
    assert forward == 1.0, "still floored -- start_time alone already exceeds duration"


def test_a_moment_older_than_the_cache_gets_no_recording():
    assert svr_window_for_age(15.0, pad=5.0, duration=10.0, cache=15.0) is None
    assert svr_window_for_age(40.0, pad=5.0, duration=10.0, cache=15.0) is None


def test_a_negative_age_is_treated_as_now_not_trusted():
    """pts arriving slightly ahead of the probe's last sample must not shorten the
    reach-back into a window that starts after the alert."""
    start, forward = svr_window_for_age(-3.0, pad=5.0, duration=10.0, cache=15.0)
    assert (start, forward) == (5.0, 5.0)


def test_the_cooldown_counts_from_the_session_s_sr_done_not_from_the_next_alert():
    """Told when its session ended (sr-done), the trigger does not refuse the first alert after a quiet spell: before,
    the close was only noticed by that alert, so it always fell inside the gap and its clip was lost (2026-11-09:
    every other proximity alert of the safety demo, a minute apart)."""
    t = SvrTrigger(coalesce_seconds=2.5, min_gap_seconds=1.0)
    assert t.observe(100.0, recording_active=False) == "start"
    t.closed(110.0)                                                   # its sr-done
    assert t.observe(160.0, recording_active=False) == "start"        # a minute later: long past the gap
    t.closed(170.0)
    assert t.observe(170.5, recording_active=False) == "cooldown"     # the gap still holds right after a close
