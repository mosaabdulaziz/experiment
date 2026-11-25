from app.metrics import MetricsRegistry, StreamRow, render_table


def test_record_analyze_avg_is_last_10():
    r = MetricsRegistry()
    for i, d in enumerate([1.0] * 12):  # 12 calls; avg should use only last 10
        r.record_analyze("cam0", d, now=100.0 + i)
    for d in [3.0, 5.0]:                # two more recent, larger durations
        r.record_analyze("cam0", d, now=200.0)
    row = {row.stream: row for row in r.rows(now=200.0)}["cam0"]
    assert row.last_s == 5.0
    # last 10 = eight 1.0s + 3.0 + 5.0 = (8 + 3 + 5)/10 = 1.6
    assert abs(row.avg_s - 1.6) < 1e-9


def test_calls_per_min_is_60s_window():
    r = MetricsRegistry()
    r.record_analyze("cam0", 1.0, now=10.0)    # 70s before 'now' -> excluded
    r.record_analyze("cam0", 1.0, now=50.0)    # within 60s
    r.record_analyze("cam0", 1.0, now=70.0)    # within 60s
    row = r.rows(now=80.0)[0]
    assert row.cpm == 2


def test_note_event_age():
    r = MetricsRegistry()
    r.record_analyze("cam0", 1.0, now=100.0)
    r.note_event("cam0", now=100.0)
    row = r.rows(now=103.0)[0]
    assert row.age_s == 3.0


def test_rows_empty_when_no_data():
    assert MetricsRegistry().rows(now=1.0) == []


def test_render_table_snapshot():
    rows = [
        StreamRow("cam0", last_s=1.8, avg_s=2.1, cpm=24, age_s=3.0),
        StreamRow("cam1", last_s=2.4, avg_s=2.3, cpm=19, age_s=11.0),
    ]
    out = render_table(rows, {"cam0": 2, "cam1": 0}, worker="cam2", sweeps=1, now_str="12:04:31")
    expected = (
        "----- analysis status @12:04:31 -----\n"
        " stream  q  last   avg    calls/m  last-event\n"
        " cam0    2  1.8s   2.1s   24       3s ago\n"
        " cam1    0  2.4s   2.3s   19       11s ago\n"
        " VLM worker: busy(cam2)  chat-sweeps: 1 active"
    )
    assert out == expected


def test_render_table_handles_missing_data_and_idle_worker():
    rows = [StreamRow("local", last_s=None, avg_s=None, cpm=0, age_s=None)]
    out = render_table(rows, {}, worker=None, sweeps=0, now_str="00:00:00")
    assert " local   0  -      -      0        -" in out
    assert "VLM worker: idle  chat-sweeps: 0 active" in out
