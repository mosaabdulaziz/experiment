"""GET /api/barber/report/photos (2026-10-22, spec haircut photos section 4)."""
import io
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from domain import EngineUpdate, ServiceSession, Visit
from report_pdf import chrome_path
from storage import apply_update, connect, set_haircut_photo
from test_report_api import today
from test_server import client


def _jpeg():
    out = io.BytesIO()
    Image.new("RGB", (64, 48), (90, 90, 90)).save(out, "JPEG")
    return out.getvalue()


def test_a_days_photo_report_with_its_file_name_and_photos(tmp_path):
    c = client(tmp_path)
    t = datetime.now() - timedelta(minutes=1)
    conn = connect(str(tmp_path / "barber-mc.db"))
    apply_update(conn, EngineUpdate(visits={"v1": Visit("v1", "p1", t, False, confirmed=True, service_started_at=t)},
                                    sessions={"s1": ServiceSession("s1", "v1", "R1", t)}), t)
    (tmp_path / "crops" / "haircuts").mkdir()
    (tmp_path / "crops" / "haircuts" / "s1_start.jpg").write_bytes(_jpeg())
    set_haircut_photo(conn, "s1", "start", "haircuts/s1_start.jpg", t)
    r = c.get(f"/api/barber/report/photos?start={today()}&lang=en&format=html")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert r.headers["content-disposition"].startswith("inline;") and "_haircut_photos.html" in r.headers["content-disposition"]
    assert r.text.count("data:image/jpeg;base64,") == 1 and "No picture" in r.text
    d = c.get(f"/api/barber/report/photos?start={today()}&lang=ar&format=html&download=1")
    assert d.headers["content-disposition"].startswith("attachment;")


def test_off_the_shop_network_there_are_no_photos(tmp_path):
    c = client(tmp_path)
    remote = TestClient(c.app, client=("8.8.8.8", 50000))            # 203.0.113.x counts as private in Python 3.11
    assert remote.get(f"/api/barber/report/photos?start={today()}&lang=en&format=html").status_code == 403
    assert remote.get("/api/barber/report/photos/allowed").json() == {"allowed": False}
    assert c.get("/api/barber/report/photos/allowed").json() == {"allowed": True}


def test_bad_parameters_and_future_days_are_400(tmp_path):
    c = client(tmp_path)
    for qs in ("start=bad&lang=en&format=html", f"start={today()}&lang=fr&format=html",
               f"start={today()}&lang=en&format=docx", "start=2999-01-01&lang=en&format=html"):
        assert c.get(f"/api/barber/report/photos?{qs}").status_code == 400, qs


@pytest.mark.skipif(chrome_path() is None, reason="no Chrome on this machine")
def test_the_pdf_downloads(tmp_path):
    r = client(tmp_path).get(f"/api/barber/report/photos?start={today()}&lang=ar&format=pdf")
    assert r.status_code == 200 and r.content.startswith(b"%PDF") and "attachment" in r.headers["content-disposition"]


def test_the_photo_report_carries_the_shops_logo_and_the_logo_route_still_serves_it(tmp_path):
    """Final review: the reports' logo helper shared its name with the /api/barber/logo route inside create_app."""
    from config import CameraConfig, ServiceConfig
    from server import create_app
    from zones import FrameGeometry
    logo = tmp_path / "logo.png"
    Image.new("RGB", (8, 8), (10, 120, 80)).save(logo, "PNG")
    cfg = ServiceConfig(redis_url="redis://localhost:1/0", frame=FrameGeometry(1000, 1000),
                        cameras=(CameraConfig("barber-c"),), data_dir=tmp_path, stream_starts_file=tmp_path / "s.json",
                        crops_dir=tmp_path / "crops", logo=logo)
    (tmp_path / "crops").mkdir()
    c = TestClient(create_app(cfg, start_consumer=False), client=("127.0.0.1", 50000))
    html = c.get(f"/api/barber/report/photos?start={today()}&lang=en&format=html").text
    assert 'class="shopLogo"' in html and "data:image/png;base64," in html
    assert c.get("/api/barber/logo").content == logo.read_bytes()
