"""Severity-grouped history rendering.

The model reasons in critical/warning/info, but history used to be rendered with the
derived event_type only — so it could not see what it had already reported at each
level. These pin the grouping, the per-tier retention, and the null-severity fallback
for rows written before the severity column existed.
"""
import asyncio

from app.analyzer import Analyzer, select_history_by_severity
from app.events import EventStore


def _ev(sev, msg, etype=None, start=None):
    return {"severity": sev, "event_type": etype or ("comment" if sev == "info" else "alert"),
            "message": msg, "stream_offset_start": start, "stream_offset_end": start,
            "timestamp": "2026-08-30T12:00:00Z", "camera": None}


def test_select_groups_by_severity():
    out = select_history_by_severity(
        [_ev("critical", "c1"), _ev("warning", "w1"), _ev("info", "i1")], per_severity=5)
    assert [e["message"] for e in out["critical"]] == ["c1"]
    assert [e["message"] for e in out["warning"]] == ["w1"]
    assert [e["message"] for e in out["info"]] == ["i1"]


def test_select_caps_each_tier_independently():
    """A flood of warnings must not crowd critical or info out of the model's view."""
    events = [_ev("warning", f"w{i}") for i in range(10)] + [_ev("info", "i1")]
    out = select_history_by_severity(events, per_severity=3)
    assert len(out["warning"]) == 3
    assert [e["message"] for e in out["info"]] == ["i1"]


def test_select_keeps_the_newest_of_a_tier_and_returns_chronological_order():
    """`events` arrives in EventStore.get_recent's order: oldest-first. When a tier
    has more than `per_severity` entries, the selector must keep the NEWEST ones
    (the model needs to see what it most recently reported) and drop the oldest —
    then hand them back in chronological order (oldest of the kept set first)."""
    events = [_ev("warning", f"w{i}") for i in range(5)]  # oldest-first: w0..w4
    out = select_history_by_severity(events, per_severity=3)
    assert [e["message"] for e in out["warning"]] == ["w2", "w3", "w4"]


def test_select_falls_back_to_event_type_when_severity_is_null():
    """Rows written before the severity column exists carry event_type only."""
    out = select_history_by_severity(
        [{"severity": None, "event_type": "alert", "message": "old alert"},
         {"severity": None, "event_type": "comment", "message": "old comment"}],
        per_severity=5)
    assert [e["message"] for e in out["warning"]] == ["old alert"]
    assert [e["message"] for e in out["info"]] == ["old comment"]


def test_select_drops_an_unrecognisable_severity_rather_than_crashing():
    out = select_history_by_severity([{"severity": "banana", "message": "x"}], per_severity=5)
    assert out["critical"] == [] and out["warning"] == [] and out["info"] == []


def test_format_groups_under_severity_headings():
    text = Analyzer._format_history(
        [_ev("critical", "person down", start=72.0), _ev("warning", "aisle blocked", start=10.0)],
        per_severity=5)
    assert "CRITICAL:" in text
    assert "WARNING:" in text
    assert "person down" in text
    assert "aisle blocked" in text


def test_format_omits_empty_tiers():
    """An empty heading wastes prompt tokens and implies nothing was seen at that level."""
    text = Analyzer._format_history([_ev("warning", "only a warning")], per_severity=5)
    assert "WARNING:" in text
    assert "CRITICAL:" not in text
    assert "INFO:" not in text


def test_format_never_prints_none_for_a_null_severity_row():
    text = Analyzer._format_history(
        [{"severity": None, "event_type": "alert", "message": "legacy row"}], per_severity=5)
    assert "None" not in text
    assert "legacy row" in text


def test_format_reports_no_history_when_empty():
    assert Analyzer._format_history([], per_severity=5) == "No previous observations."


def test_format_without_per_severity_keeps_the_flat_legacy_rendering():
    """This branch exists for Analyzer.chat() and its direct tests; chat() itself
    currently has no callers in production — chat goes through chat_multi()."""
    text = Analyzer._format_history([_ev("warning", "flat one", start=5.0)])
    assert "WARNING:" not in text
    assert "ALERT: flat one" in text


def test_select_survives_end_to_end_through_the_real_event_store(tmp_path):
    """Regression for the C1 bug: `select_history_by_severity` was iterating
    `get_recent`'s oldest-first list forward, so it kept the OLDEST `per_severity`
    entries of a tier and silently dropped the newest — including whatever was
    just reported on the immediately preceding cycle. This drives 8 warning saves
    through a real EventStore (no hand-built ordering assumptions) and pins that
    the newest 5 survive and the oldest 3 are gone."""
    store = EventStore({"events": {"db_path": str(tmp_path / "events.db"), "retention_hours": 48}})

    async def run():
        await store.init()
        for i in range(8):
            await store.save("alert", f"cycle {i}", [], camera="cam0", severity="warning")
        return await store.get_recent(limit=20)

    events = asyncio.run(run())
    assert [e["message"] for e in events] == [f"cycle {i}" for i in range(8)]  # oldest-first

    out = select_history_by_severity(events, per_severity=5)
    kept = [e["message"] for e in out["warning"]]
    assert kept == ["cycle 3", "cycle 4", "cycle 5", "cycle 6", "cycle 7"]  # newest 5, chronological
    for dropped in ("cycle 0", "cycle 1", "cycle 2"):
        assert dropped not in kept
