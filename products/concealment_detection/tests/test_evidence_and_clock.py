"""Unit tests for `evidence.py` and `footage_clock.py` (ACs 4, 5)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest

from concealment.types import CandidateEvent, StageTransition
from evidence import (
    FORBIDDEN_PHRASES,
    build_candidate_record,
    build_description,
    tier_to_severity,
    validate_non_accusatory_text,
)
from footage_clock import display_timestamp, footage_clock


def _sample_candidate(tier: str = "high", target_kind: str = "clothing") -> CandidateEvent:
    return CandidateEvent(
        camera="cam-1",
        track_id=42,
        pts=12.5,
        tier=tier,
        score=0.88,
        zone_id="shelf-a",
        target_kind=target_kind,
        body_zone_name="waist" if target_kind == "clothing" else "personal_bag",
        terminal_stage="RETURNED",
        hold_duration_s=1.4,
        dwell_duration_s=3.2,
        stage_timeline=(
            StageTransition("DWELL", 7.0, "shelf-a"),
            StageTransition("REACH", 10.2, "excursion=0.62"),
            StageTransition("AT_BODY", 11.1, "waist"),
            StageTransition("RETURNED", 12.5, "hold=1.4s"),
        ),
        clip_window=(6.5, 18.5),
        keyframes_pts=(7.0, 10.2, 11.1, 12.5),
    )


def test_tier_to_severity_maps_tiers_to_platform_severities():
    assert tier_to_severity("high") == "critical"
    assert tier_to_severity("medium") == "warning"
    assert tier_to_severity("low") == "info"
    with pytest.raises(ValueError):
        tier_to_severity("critical")
    with pytest.raises(ValueError):
        tier_to_severity("unknown")


def test_build_description_and_non_accusatory_guard():
    """AC-4 & AC-5: Alert description includes camera location, zone, behavior details, and identifies
    the incident as suspected activity requiring human verification, never confirmed theft."""
    cand_cloth = _sample_candidate("high", "clothing")
    desc = build_description(cand_cloth, display_time="2026-11-02T14:00:00+00:00")
    validate_non_accusatory_text(desc)
    assert "cam-1" in desc
    assert "shelf-a" in desc
    assert "clothing" in desc.lower()
    assert "waist" in desc.lower()
    assert "suspected" in desc.lower()
    assert "human verification" in desc.lower()
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in desc.lower()

    cand_bag = _sample_candidate("medium", "personal_bag")
    desc_bag = build_description(cand_bag)
    validate_non_accusatory_text(desc_bag)
    assert "personal bag" in desc_bag.lower()

    # Guard rejects any accusatory wording or missing verification framing
    with pytest.raises(ValueError):
        validate_non_accusatory_text("Confirmed theft on cam-1 at shelf-a.")
    with pytest.raises(ValueError):
        validate_non_accusatory_text("Theft detected by person #42.")
    with pytest.raises(ValueError):
        validate_non_accusatory_text("Person reached shelf-a.")  # missing suspected / human verification


def test_build_candidate_record_contains_all_required_alert_fields():
    cand = _sample_candidate("high", "clothing")
    dev_dt = datetime(2026, 11, 2, 12, 0, 10, tzinfo=timezone.utc)
    rec = build_candidate_record(cand, clock=None, device_dt=dev_dt)
    assert rec["camera"] == "cam-1"
    assert rec["location"] == "cam-1"
    assert rec["track_id"] == 42
    assert rec["pts"] == 12.5
    assert rec["display_time"] == "2026-11-02T12:00:10+00:00"
    assert rec["tier"] == "high"
    assert rec["severity"] == "critical"
    assert rec["category"] == "security"
    assert rec["kind"] == "suspected_concealment"
    assert rec["requires_human_verification"] is True
    assert rec["clip_window"] == [6.5, 18.5]
    validate_non_accusatory_text(rec["description"])


def test_footage_clock_shifts_device_time_to_recorded_footage_time():
    start = datetime(2026, 10, 16, 9, 30, 0)
    # Lap 1 started at unix ts 1000.0 with 60s clip duration -> lap 1 start is ts 1060.0
    lap0_ts = 1000.0
    run = {"cam-1": [lap0_ts, 60.0, 1]}
    clk = footage_clock(start, run)
    assert clk is not None
    assert clk.end == start + timedelta(seconds=60.0)
    # Device time 15s into lap 1 -> footage time should be 09:30:15
    dev_now = datetime.fromtimestamp(lap0_ts + 60.0 + 15.0)
    mapped = clk.footage(dev_now)
    assert mapped == datetime(2026, 10, 16, 9, 30, 15)
    assert clk.device(mapped) == dev_now
    assert display_timestamp(dev_now, clock=clk).startswith("2026-10-16T09:30:15")
    assert footage_clock(None, run) is None


@pytest.mark.parametrize("start", [
    datetime(2026, 10, 16, 9, 30),
    datetime(2026, 10, 16, 9, 30, tzinfo=timezone.utc),
    datetime(2026, 10, 16, 9, 30, tzinfo=timezone(timedelta(hours=3))),
])
def test_clock_maps_aware_and_naive_device_datetimes_from_same_epoch(start):
    # Modern epochs avoid Windows CRT's unsupported early-1970 naive timestamp().
    clk = footage_clock(start, {"cam-1": [1700000000.0, 60.0, 1]})
    expected = start + timedelta(seconds=15)
    for device in (datetime.fromtimestamp(1700000075.0),
                   datetime.fromtimestamp(1700000075.0, timezone.utc),
                   datetime.fromtimestamp(1700000075.0, timezone(timedelta(hours=-5)))):
        assert clk.footage(device) == expected
        assert clk.device(expected).timestamp() == pytest.approx(1700000075.0)
        rec = build_candidate_record(_sample_candidate(), clock=clk, device_dt=device)
        assert rec["display_time"] == expected.isoformat()
        assert expected.isoformat() in rec["description"]


def test_candidate_record_with_default_aware_now_and_naive_footage_clock():
    clk = footage_clock(datetime(2026, 10, 16, 9, 30), {"cam-1": [1000.0, 60.0, 0]})
    rec = build_candidate_record(_sample_candidate(), clock=clk)
    assert datetime.fromisoformat(rec["display_time"]).tzinfo is None
