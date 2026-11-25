"""The haircut_photo table (2026-10-22, spec haircut photos section 3)."""
from datetime import datetime, timedelta

from domain import EngineUpdate, ServiceSession, Visit
from storage import (apply_update, clear_all, connect, drop_haircut_photos, haircut_photos, purge_haircut_photos,
                     referenced_crops, set_haircut_photo)

T = datetime(2026, 10, 22, 12, 0)


def _haircut(conn, sid="s1", at=T):
    apply_update(conn, EngineUpdate(visits={sid: Visit(sid, sid, at, False, confirmed=True, service_started_at=at)},
                                    sessions={sid: ServiceSession(sid, sid, "R1", at)}), at)


def test_a_photo_is_kept_per_haircut_and_kind_and_an_end_photo_is_replaced(tmp_path):
    conn = connect(str(tmp_path / "b.db"))
    set_haircut_photo(conn, "s1", "start", "haircuts/s1_start.jpg", T)
    set_haircut_photo(conn, "s1", "end", "haircuts/s1_end.jpg", T + timedelta(minutes=2))
    set_haircut_photo(conn, "s1", "end", "haircuts/s1_end.jpg", T + timedelta(minutes=4))
    assert haircut_photos(conn, ["s1", "s2"]) == {"s1": {
        "start": {"path": "haircuts/s1_start.jpg", "taken_at": "2026-10-22T12:00:00"},
        "end": {"path": "haircuts/s1_end.jpg", "taken_at": "2026-10-22T12:04:00"}}}
    assert haircut_photos(conn, []) == {}


def test_a_dropped_haircut_loses_its_photos(tmp_path):
    conn = connect(str(tmp_path / "b.db"))
    set_haircut_photo(conn, "h1", "start", "haircuts/h1_start.jpg", T)
    assert drop_haircut_photos(conn, "h1") == ["haircuts/h1_start.jpg"]
    assert haircut_photos(conn, ["h1"]) == {} and drop_haircut_photos(conn, "h1") == []


def test_photos_are_referenced_crops_and_a_lap_clear_removes_them(tmp_path):
    conn = connect(str(tmp_path / "b.db"))
    _haircut(conn)
    set_haircut_photo(conn, "s1", "start", "haircuts/s1_start.jpg", T)
    assert "haircuts/s1_start.jpg" in referenced_crops(conn)
    assert "haircuts/s1_start.jpg" in clear_all(conn)
    assert haircut_photos(conn, ["s1"]) == {}


def test_the_purge_takes_old_photos_and_stray_rows_of_no_haircut(tmp_path):
    """Review focus: a request queued at a lap's reset lands after clear_all -- a row of no haircut. The nightly pass
    removes those older than an hour; a start photo younger than that may still become a haircut."""
    conn = connect(str(tmp_path / "b.db"))
    now = T + timedelta(days=400)
    _haircut(conn, "old", T)
    set_haircut_photo(conn, "old", "start", "haircuts/old_start.jpg", T)                     # past retention
    _haircut(conn, "kept", now - timedelta(hours=5))
    set_haircut_photo(conn, "kept", "start", "haircuts/kept_start.jpg", now - timedelta(hours=5))
    set_haircut_photo(conn, "stray", "start", "haircuts/stray_start.jpg", now - timedelta(hours=2))  # no haircut row
    set_haircut_photo(conn, "young", "start", "haircuts/young_start.jpg", now - timedelta(minutes=10))
    assert sorted(purge_haircut_photos(conn, now, 365.0)) == ["haircuts/old_start.jpg", "haircuts/stray_start.jpg"]
    assert set(haircut_photos(conn, ["old", "kept", "stray", "young"])) == {"kept", "young"}


def test_the_orphan_sweep_keeps_a_pedicures_photos(tmp_path):
    """2026-11-10: a pedicure's photos are under its pedicure_session id -- not orphans of service_session."""
    from domain import PedicureRecord
    from storage import set_haircut_photo
    conn = connect(tmp_path / "b.db")
    t = datetime(2026, 10, 16, 14, 16)
    apply_update(conn, EngineUpdate(pedicures={"p1": PedicureRecord("p1", "P1", t)}), t)
    set_haircut_photo(conn, "p1", "start", "haircuts/p1-start.jpg", t)
    assert purge_haircut_photos(conn, t + timedelta(hours=3), days=30) == []
