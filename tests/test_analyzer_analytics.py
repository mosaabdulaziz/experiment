from app.analyzer import Analyzer


def test_format_analytics_empty():
    assert Analyzer._format_analytics(None) == "No events recorded yet."
    empty = {"total_alerts": 0, "total_comments": 0, "alerts_today": 0, "alerts_24h": 0,
             "per_camera": [], "most_recent_alert": None, "top_tags": []}
    assert Analyzer._format_analytics(empty) == "No events recorded yet."


def test_format_analytics_full():
    summary = {
        "total_alerts": 142, "total_comments": 318, "alerts_today": 17, "alerts_24h": 41,
        "per_camera": [{"camera": "cam1", "alerts": 63}, {"camera": "cam0", "alerts": 51}],
        "most_recent_alert": {"camera": "cam1", "timestamp": "2026-06-24T14:32:08Z",
                              "message": "red-light runner, dark sedan"},
        "top_tags": [{"tag": "red-light-violation", "count": 38}, {"tag": "night-time", "count": 22}],
    }
    out = Analyzer._format_analytics(summary)
    assert "142 alerts, 318 comments" in out
    assert "Today: 17" in out and "Last 24h: 41" in out
    assert "cam1=63" in out and "cam0=51" in out
    assert "red-light runner, dark sedan" in out
    assert "red-light-violation (38)" in out
    # Most-recent-alert time is a dated UTC wall-clock stamp (not a footage HH:MM:SS).
    assert "recorded 2026-06-24 14:32:08 UTC" in out
