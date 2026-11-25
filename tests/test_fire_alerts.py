"""Fire/smoke alert state: per-class sustained latches over per-frame detections.

Pure Python, importable off-GPU (same convention as test_overcrowding_dedup.py)."""

import pytest

from perception.modules.fire_alerts.state import FireAlertSettings, FireAlertState, FireIncidentEvent


def _started(cls, count, onset_age):
    return FireIncidentEvent(cls=cls, kind="started", count=count, onset_age=onset_age)


def _resolved(cls, duration):
    return FireIncidentEvent(cls=cls, kind="resolved", duration=duration)


def _settings(**over):
    base = dict(enabled=True, classes=("fire", "smoke"),
                sustain_seconds={"fire": 3.0, "smoke": 8.0},
                release_seconds=30.0, min_confidence=0.4)
    base.update(over)
    return FireAlertSettings(**base)


def test_sustained_fire_fires_once_with_its_count():
    st = FireAlertState(_settings())
    dets = [("fire", 0.9), ("fire", 0.8)]
    assert st.observe(0.0, dets) == []
    assert st.observe(3.0, dets) == [_started("fire", 2, onset_age=3.0)]
    assert st.observe(4.0, dets) == []


def test_transient_fire_never_fires():
    st = FireAlertState(_settings())
    assert st.observe(0.0, [("fire", 0.9)]) == []
    assert st.observe(0.5, []) == []
    assert st.observe(10.0, []) == []


def test_classes_are_independent_and_use_their_own_sustain():
    st = FireAlertState(_settings())
    dets = [("fire", 0.9), ("smoke", 0.9)]
    assert st.observe(0.0, dets) == []
    assert st.observe(3.0, dets) == [_started("fire", 1, onset_age=3.0)]   # smoke's 8s has not elapsed
    assert st.observe(8.0, dets) == [_started("smoke", 1, onset_age=8.0)]


def test_detections_below_min_confidence_do_not_count():
    st = FireAlertState(_settings(min_confidence=0.5))
    weak = [("fire", 0.49)]
    assert st.observe(0.0, weak) == []
    assert st.observe(10.0, weak) == []


def test_unlisted_classes_are_ignored():
    st = FireAlertState(_settings())
    assert st.observe(0.0, [("person", 0.99)]) == []
    assert st.observe(10.0, [("person", 0.99)]) == []


def test_disabled_settings_never_fire():
    st = FireAlertState(_settings(enabled=False))
    dets = [("fire", 0.9)]
    assert st.observe(0.0, dets) == []
    assert st.observe(100.0, dets) == []


def test_missing_sustain_entry_falls_back_to_most_conservative_default():
    s = _settings(classes=("fire", "smoke", "ember"),
                  sustain_seconds={"fire": 3.0, "smoke": 8.0})
    st = FireAlertState(s)
    # ember has no explicit entry -- must not fire before the most conservative
    # known default (smoke's 8.0s) elapses, and must not fire at 0.0s.
    assert st.observe(0.0, [("ember", 0.9)]) == []
    assert st.observe(3.0, [("ember", 0.9)]) == []
    assert st.observe(8.0, [("ember", 0.9)]) == [_started("ember", 1, onset_age=8.0)]


def test_flickering_smoke_still_confirms():
    # A detector that drops a frame here and there must still reach its window --
    # this is the failure mode dropout tolerance exists to prevent.
    st = FireAlertState(_settings(sustain_seconds={"fire": 3.0, "smoke": 8.0}))
    present = [("smoke", 0.9)]
    assert st.observe(0.0, present) == []
    t = 0.5
    while t < 8.0:
        st.observe(t, [])            # a dropped frame, inside the default 1.0s grace
        st.observe(t + 0.2, present)
        t += 1.0
    assert st.observe(8.0, present) == [_started("smoke", 1, onset_age=8.0)]


def test_resolved_event_reports_the_true_visible_duration():
    # True lifetime is onset (0.0) to true disappearance (2.0) = 2.0s -- NOT the
    # 1.0s to sustain-confirm, and NOT stretched by the 5.0s release debounce
    # that follows before "resolved" is even allowed to fire (at 7.0).
    st = FireAlertState(_settings(sustain_seconds={"fire": 1.0, "smoke": 8.0},
                                  release_seconds=5.0))
    present = [("fire", 0.9)]
    assert st.observe(0.0, present) == []
    assert st.observe(1.0, present) == [_started("fire", 1, onset_age=1.0)]
    assert st.observe(2.0, []) == []                       # first frame it's gone
    assert st.observe(6.9, []) == []                       # release not elapsed yet
    assert st.observe(7.0, []) == [_resolved("fire", duration=2.0)]


def test_stray_recurrence_during_release_does_not_delay_resolution():
    # Traced defect: a single stray recurrence during the release window used to
    # reset the countdown from scratch, so a flicker (ember, glare, dissipating
    # smoke wisp) after the real fire ended would keep the incident "ongoing" far
    # longer than release_seconds -- this is the bug-report-level regression, one
    # layer above test_stray_single_frame_recurrence_during_release_does_not_reset_countdown
    # in test_overcrowding_dedup.py.
    st = FireAlertState(_settings(sustain_seconds={"fire": 1.0, "smoke": 8.0},
                                  release_seconds=5.0))
    present = [("fire", 0.9)]
    st.observe(0.0, present)
    assert st.observe(1.0, present) == [_started("fire", 1, onset_age=1.0)]
    assert st.observe(2.0, []) == []             # true end; release countdown starts
    assert st.observe(3.0, present) == []        # stray single-frame recurrence
    assert st.observe(3.5, []) == []             # gone again immediately
    assert st.observe(6.9, []) == []             # original deadline (2.0+5.0=7.0) not yet reached
    assert st.observe(7.0, []) == [_resolved("fire", duration=2.0)]  # resolves right on time


def test_resolved_then_a_fresh_incident_starts_and_resolves_independently():
    st = FireAlertState(_settings(sustain_seconds={"fire": 1.0, "smoke": 8.0},
                                  release_seconds=5.0))
    present = [("fire", 0.9)]
    st.observe(0.0, present)
    st.observe(1.0, present)                                # started
    st.observe(2.0, [])
    assert st.observe(7.0, []) == [_resolved("fire", duration=2.0)]
    # a second, later incident -- onset/duration measured fresh, no leakage
    # from the first one's timestamps.
    assert st.observe(20.0, present) == []
    assert st.observe(21.0, present) == [_started("fire", 1, onset_age=1.0)]
    assert st.observe(22.0, []) == []
    assert st.observe(27.0, []) == [_resolved("fire", duration=2.0)]


def test_classes_resolve_independently():
    st = FireAlertState(_settings(sustain_seconds={"fire": 1.0, "smoke": 1.0},
                                  release_seconds=5.0))
    both = [("fire", 0.9), ("smoke", 0.9)]
    st.observe(0.0, both)
    assert st.observe(1.0, both) == [
        _started("fire", 1, onset_age=1.0), _started("smoke", 1, onset_age=1.0)]
    # fire alone drops out; smoke stays present -- only fire should resolve.
    assert st.observe(2.0, [("smoke", 0.9)]) == []
    assert st.observe(7.0, [("smoke", 0.9)]) == [_resolved("fire", duration=2.0)]


def test_has_open_incident_tracks_the_started_resolved_window():
    st = FireAlertState(_settings(sustain_seconds={"fire": 1.0, "smoke": 8.0},
                                  release_seconds=5.0))
    present = [("fire", 0.9)]
    assert st.has_open_incident is False
    st.observe(0.0, present)
    assert st.has_open_incident is False       # not yet confirmed
    st.observe(1.0, present)                   # started
    assert st.has_open_incident is True
    st.observe(2.0, [])
    assert st.has_open_incident is True        # still open through the release window
    st.observe(7.0, [])                        # resolved
    assert st.has_open_incident is False


def test_has_open_incident_true_while_any_class_is_open():
    st = FireAlertState(_settings(sustain_seconds={"fire": 1.0, "smoke": 1.0},
                                  release_seconds=5.0))
    both = [("fire", 0.9), ("smoke", 0.9)]
    st.observe(0.0, both)
    st.observe(1.0, both)                      # both started
    assert st.has_open_incident is True
    st.observe(2.0, [("smoke", 0.9)])           # fire alone drops out
    st.observe(7.0, [("smoke", 0.9)])           # fire resolves, smoke still present
    assert st.has_open_incident is True         # smoke keeps the incident open
