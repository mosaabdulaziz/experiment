import threading
import time

from domain import Alert, EngineUpdate
from datetime import datetime
from snapshots import Snapshotter
from storage import apply_update, connect, list_alerts

T = datetime(2026, 10, 5, 12, 0, 0)


def _alert(conn, aid="a1"):
    a = Alert(aid, "excessive_wait", "shop", "w1", None, T, subject_cam="barber-c", subject_box=(0.5, 0.4, 0.1, 0.3))
    apply_update(conn, EngineUpdate(alerts=[a]), T)


def _wait_for(pred, s=3.0):
    end = time.monotonic() + s
    while time.monotonic() < end and not pred():
        time.sleep(0.02)
    return pred()


def test_a_snapshot_is_saved_and_linked_to_its_alert(tmp_path):
    db = str(tmp_path / "b.db")
    conn = connect(db)
    _alert(conn)
    calls = []
    snap = Snapshotter("http://go2rtc:1984", tmp_path / "crops", db, fetch=lambda url: calls.append(url) or b"\xff\xd8jpeg")
    snap.request("a1", "barber-c")
    assert _wait_for(lambda: list_alerts(conn)[0]["has_snapshot"])
    assert calls == ["http://go2rtc:1984/api/frame.jpeg?src=barber-c"]
    (r,) = list_alerts(conn)
    assert r["subject_box"] == [0.5, 0.4, 0.1, 0.3]
    assert (tmp_path / "crops" / "alerts" / "a1.jpg").read_bytes() == b"\xff\xd8jpeg"
    snap.stop()


def test_an_unreachable_camera_server_leaves_no_picture_and_warns(tmp_path, caplog):
    db = str(tmp_path / "b.db")
    conn = connect(db)
    _alert(conn)

    def boom(url):
        raise OSError("connection refused")
    snap = Snapshotter("http://go2rtc:1984", tmp_path / "crops", db, fetch=boom)
    snap.request("a1", "barber-c")
    assert _wait_for(lambda: any("snapshot" in r.message for r in caplog.records))
    assert list_alerts(conn)[0]["has_snapshot"] is False
    snap.stop()


def test_a_hanging_fetch_never_blocks_request(tmp_path):
    db = str(tmp_path / "b.db")
    connect(db)
    gate = threading.Event()
    snap = Snapshotter("http://go2rtc:1984", tmp_path / "crops", db, fetch=lambda url: gate.wait(5) and b"x")
    t0 = time.monotonic()
    for i in range(5):
        snap.request(f"a{i}", "barber-c")
    assert time.monotonic() - t0 < 0.1
    gate.set()
    snap.stop()
