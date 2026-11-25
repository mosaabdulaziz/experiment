"""The service's saved reports (2026-10-25; spec 2026-10-25-barber-saved-reports-design.md): saved on the download
endpoints' code, listed and served on the shop network only."""
import json
from datetime import date, datetime, timedelta

from fastapi.testclient import TestClient

from domain import EngineUpdate, Visit
from saved_reports import SavedReports, write_manifest
from shopday import day_start
from storage import apply_update, connect
from test_server import client


def test_the_service_saves_a_finished_day_on_the_download_code(tmp_path):
    c = client(tmp_path)
    saver = c.app.state.saver
    assert saver.live and saver.cfg is None and saver.run_once() == []           # no saved_reports: off
    saver.cfg = SavedReports(("ar",), ("html",))
    saver.clock = lambda: datetime(2026, 10, 21, 12, 0)
    t = day_start(date(2026, 10, 20), 5) + timedelta(hours=8)
    apply_update(connect(str(tmp_path / "barber-mc.db")),
                 EngineUpdate(visits={"v1": Visit("v1", "p1", t, False, confirmed=True)}), t)
    [folder] = saver.run_once()
    assert folder == tmp_path / "reports" / "day" / "2026-10-20"
    [page] = folder.glob("*_analysis_report_ar.html")
    assert 'dir="rtl"' in page.read_text() and 'id="summary"' in page.read_text()
    assert json.loads((folder / "report.json").read_text())["files"] == \
        [{"report": "analysis", "lang": "ar", "format": "html", "name": page.name}]


def _one_saved_day(tmp_path):
    folder = tmp_path / "reports" / "day" / "2026-10-24"
    folder.mkdir(parents=True)
    (folder / "24oct_x_analysis_report_en.html").write_text("<p>saved</p>")
    (folder / "24oct_x_analysis_report_en.pdf").write_bytes(b"%PDF saved")
    write_manifest(folder, {"kind": "day", "first": "2026-10-24", "last": "2026-10-24", "start": "2026-10-24T05:00:00",
                            "end": "2026-10-25T05:00:00", "lap_ended": None, "label": None,
                            "saved_at": "2026-10-25T05:00:41",
                            "files": [{"report": "analysis", "lang": "en", "format": "html",
                                       "name": "24oct_x_analysis_report_en.html"},
                                      {"report": "analysis", "lang": "en", "format": "pdf",
                                       "name": "24oct_x_analysis_report_en.pdf"}]})


def test_the_list_answers_everyone_and_says_who_may_open_the_files(tmp_path):
    c = client(tmp_path)
    _one_saved_day(tmp_path)
    r = c.get("/api/barber/reports/saved?kind=day").json()
    assert (r["enabled"], r["allowed"]) == (False, True) and [i["id"] for i in r["items"]] == ["day/2026-10-24"]
    c.app.state.saver.cfg = SavedReports(("en",), ("html",))
    assert c.get("/api/barber/reports/saved?kind=day").json()["enabled"] is True
    assert c.get("/api/barber/reports/saved?kind=week").json()["items"] == []
    assert c.get("/api/barber/reports/saved?kind=year").status_code == 400
    remote = TestClient(c.app, client=("8.8.8.8", 50000))             # 203.0.113.x counts as private in Python 3.11
    r = remote.get("/api/barber/reports/saved?kind=day").json()
    assert r["allowed"] is False and len(r["items"]) == 1


def test_a_saved_file_opens_inline_or_downloads_on_the_shop_network(tmp_path):
    c = client(tmp_path)
    _one_saved_day(tmp_path)
    url = "/api/barber/reports/saved/file?id=day/2026-10-24&name="
    h = c.get(url + "24oct_x_analysis_report_en.html")
    assert h.status_code == 200 and h.text == "<p>saved</p>" and h.headers["content-disposition"].startswith("inline;")
    assert c.get(url + "24oct_x_analysis_report_en.html&download=1").headers["content-disposition"].startswith("attachment;")
    p = c.get(url + "24oct_x_analysis_report_en.pdf")
    assert p.content == b"%PDF saved" and p.headers["content-type"] == "application/pdf"
    assert p.headers["content-disposition"].startswith("attachment;")


def test_off_the_shop_network_a_saved_file_is_403_even_with_a_forged_header(tmp_path):
    c = client(tmp_path)
    _one_saved_day(tmp_path)
    remote = TestClient(c.app, client=("8.8.8.8", 50000))
    for headers in ({}, {"X-Forwarded-For": "127.0.0.1"}):
        r = remote.get("/api/barber/reports/saved/file?id=day/2026-10-24&name=24oct_x_analysis_report_en.pdf",
                       headers=headers)
        assert r.status_code == 403 and r.json()["detail"] == "saved reports open on the shop network only"


def test_a_file_not_in_its_manifest_or_outside_the_reports_is_404(tmp_path):
    c = client(tmp_path)
    _one_saved_day(tmp_path)
    (tmp_path / "reports" / "day" / "2026-10-24" / "24oct_x_analysis_report_en.pdf").unlink()    # moved by hand
    for qs in ("id=day/2026-10-24&name=report.json", "id=day/2026-10-24&name=24oct_x_analysis_report_en.pdf",
               "id=../..&name=barber-mc.db", "id=day/2026-10-25&name=x.pdf"):
        assert c.get(f"/api/barber/reports/saved/file?{qs}").status_code == 404, qs


def test_a_leftover_chrome_temp_folder_is_removed_at_startup(tmp_path):
    left = tmp_path / ".tmp" / "barber-report-abc"
    left.mkdir(parents=True)
    (left / "x").write_text("y")
    other = tmp_path / ".tmp" / "keep-me"
    other.mkdir()
    with client(tmp_path):
        assert not left.exists() and other.exists()


def test_a_clip_site_without_a_footage_start_saves_nothing(tmp_path):
    from config import CameraConfig, ServiceConfig
    from server import create_app
    from zones import FrameGeometry
    cfg = ServiceConfig(redis_url="redis://localhost:1/0", frame=FrameGeometry(1000, 1000),
                        cameras=(CameraConfig("ph-1", clip_duration_s=7200, clear_on_loop=True),),
                        data_dir=tmp_path, stream_starts_file=tmp_path / "starts.json", crops_dir=tmp_path / "crops",
                        saved_reports=SavedReports(("en",), ("html",)))
    (tmp_path / "crops").mkdir()
    app = create_app(cfg, start_consumer=False)
    assert not app.state.saver.live and app.state.consumer.on_lap_end is None
