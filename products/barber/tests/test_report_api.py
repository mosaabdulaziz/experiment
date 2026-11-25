from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from policy import DEFAULT_POLICY
from report_pdf import chrome_path
from shopday import shop_day
from test_server import client


def today():
    return shop_day(datetime.now(), int(DEFAULT_POLICY["day_start_hour"])).isoformat()


def test_html_report_with_its_file_name(tmp_path):
    c = client(tmp_path)
    r = c.get(f"/api/barber/report?kind=day&start={today()}&lang=en&format=html")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "_analysis_report.html" in r.headers["content-disposition"]
    assert 'id="summary"' in r.text


def test_bad_parameters_and_future_periods_are_400(tmp_path):
    c = client(tmp_path)
    assert c.get("/api/barber/report?kind=year&start=2026-10-11&lang=en&format=html").status_code == 400
    assert c.get("/api/barber/report?kind=day&start=bad&lang=en&format=html").status_code == 400
    assert c.get("/api/barber/report?kind=day&start=2026-10-11&lang=fr&format=html").status_code == 400
    assert c.get("/api/barber/report?kind=day&start=2026-10-11&lang=en&format=xls").status_code == 400
    assert c.get("/api/barber/report?kind=day&start=2999-01-01&lang=en&format=html").status_code == 400


@pytest.mark.skipif(chrome_path() is None, reason="no Chrome on this machine")
def test_pdf_and_word_download(tmp_path):
    c = client(tmp_path)
    pdf = c.get(f"/api/barber/report?kind=week&start={today()}&lang=ar&format=pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    assert "attachment" in pdf.headers["content-disposition"]
    docx = c.get(f"/api/barber/report?kind=month&start={today()}&lang=en&format=docx")
    assert docx.status_code == 200 and docx.content[:2] == b"PK"


def test_a_remote_client_gets_no_photos(tmp_path):
    c = client(tmp_path)
    remote = TestClient(c.app, client=("203.0.113.9", 50000))
    r = remote.get(f"/api/barber/report?kind=day&start={today()}&lang=en&format=html")
    assert r.status_code == 200 and "data:image/jpeg" not in r.text


def test_a_non_ascii_or_quoted_name_makes_a_safe_header(tmp_path):
    from config import CameraConfig, ServiceConfig
    from server import create_app
    from zones import FrameGeometry
    cfg = ServiceConfig(redis_url="redis://localhost:1/0", frame=FrameGeometry(1000, 1000),
                        cameras=(CameraConfig("barber-c"),), data_dir=tmp_path, stream_starts_file=tmp_path / "s.json",
                        crops_dir=tmp_path / "crops", place='Bekky"Barber', branch="مدينتي", branch_slug="مدينتي")
    (tmp_path / "crops").mkdir()
    c = TestClient(create_app(cfg, start_consumer=False), client=("127.0.0.1", 50000))
    r = c.get(f"/api/barber/report?kind=day&start={today()}&lang=ar&format=html")
    assert r.status_code == 200
    cd = r.headers["content-disposition"]
    assert cd.isascii() and 'filename="' in cd and "filename*=UTF-8''" in cd and '"Bekky"' not in cd


def test_the_html_can_be_downloaded_as_a_file(tmp_path):
    c = client(tmp_path)
    r = c.get(f"/api/barber/report?kind=day&start={today()}&lang=en&format=html&download=1")
    assert r.status_code == 200 and r.headers["content-disposition"].startswith("attachment;")
    assert "_analysis_report.html" in r.headers["content-disposition"]


def test_a_haircut_under_an_old_zone_id_shows_its_chairs_name(tmp_path):
    """A chair moved to another camera (pair L8, owner ph-1) keeps its old copy ph2-L8: rows recorded before the
    move carry chair_id ph2-L8 and must still read "L8"."""
    import json as _json
    from datetime import datetime as _dt, timedelta as _td
    from domain import EngineUpdate, ServiceSession, Visit
    from storage import apply_update, connect
    sq = [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]
    (tmp_path / "zones.json").write_text(_json.dumps({
        "barber-c": [{"id": "c1-L8", "name": "L8", "type": "chair", "points": sq, "pair": "L8", "owner": True},
                     {"id": "c1-L8-seat", "name": "L8", "type": "seat", "points": sq, "seat_of": "c1-L8"}],
        "barber-l": [{"id": "ph2-L8", "name": "L8", "type": "chair", "points": sq, "pair": "L8"}]}))
    c = client(tmp_path)
    t = _dt.now() - _td(minutes=30)
    apply_update(connect(str(tmp_path / "barber-mc.db")), EngineUpdate(
        visits={"v1": Visit("v1", "p1", t, False, confirmed=True, service_started_at=t)},
        sessions={"s1": ServiceSession("s1", "v1", "ph2-L8", t, ended_at=t + _td(minutes=20), service_s=1200.0, open=False)}), t)
    html = c.get(f"/api/barber/report?kind=day&start={today()}&lang=en&format=html").text
    assert ">L8<" in html and "ph2-L8" not in html


def test_a_run_on_recordings_reports_the_footage_day_at_the_footage_times(tmp_path):
    """2026-10-18 (user): a run playing 16 Oct 09:30-19:00 gives that day's report in those times, wherever the
    device's clock is; the live snapshot names the footage day for the Report panel."""
    from datetime import timedelta
    from config import CameraConfig, ServiceConfig
    from saved_reports import SavedReports
    from server import create_app
    from zones import FrameGeometry
    cfg = ServiceConfig(redis_url="redis://localhost:1/0", frame=FrameGeometry(1000, 1000),
                        cameras=(CameraConfig("ph-1", clip_duration_s=34200, clear_on_loop=True),),
                        data_dir=tmp_path, stream_starts_file=tmp_path / "starts.json", crops_dir=tmp_path / "crops",
                        footage_start=datetime(2026, 10, 16, 9, 29, 58),
                        saved_reports=SavedReports(("en",), ("html",)))
    (tmp_path / "crops").mkdir()
    app = create_app(cfg, start_consumer=False)
    assert not app.state.saver.live                                   # saved at the lap's end, not by the clock
    started = datetime.now() - timedelta(minutes=90)                  # 90 min into the recording
    app.state.consumer._run = {"ph-1": [started.timestamp(), 34200.0, 0]}
    with TestClient(app, client=("127.0.0.1", 50000)) as c:
        assert c.get("/api/barber/live").json()["footage"]["day"] == "2026-10-16"
        r = c.get("/api/barber/report?kind=day&start=2026-10-16&lang=en&format=html")
        assert r.status_code == 200 and "16oct_" in r.headers["content-disposition"]
        assert "Covers Fri 16 Oct 09:29 – 10:59 (the recording&#x27;s time)" in r.text
        app.state.consumer.on_lap_end()                               # the clips played to their end
        [lap] = (tmp_path / "reports" / "day" / "2026-10-16").iterdir()
        assert lap.name.startswith("lap-")
        app.state.saver.run_once()
        assert sorted(f.name for f in lap.iterdir()) == ["16oct_BekkyBarber_branch_analysis_report_en.html", "report.json"]
