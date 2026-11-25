"""The haircut photo worker (2026-10-22, spec haircut photos section 2): fetch, crop, save, link."""
import io
import logging
import threading
from datetime import datetime, timedelta

from PIL import Image

from domain import PhotoRequest
from haircut_photos import HaircutPhotos, cropped
from storage import connect, haircut_photos
from test_snapshots import _wait_for

T = datetime(2026, 10, 22, 12, 0)


def jpeg(w=1920, h=1080, left=(200, 0, 0), right=(0, 0, 200)):
    im = Image.new("RGB", (w, h), left)
    im.paste(right, (w // 2, 0, w, h))
    out = io.BytesIO()
    im.save(out, "JPEG")
    return out.getvalue()


def req(kind="start", at=T, hid="h1", crop=(0.25, 0.0, 0.75, 0.5)):
    return PhotoRequest(hid, "R1", "ph-2", kind, at, crop)


def test_the_crop_is_taken_from_the_frames_own_size_and_scaled_down():
    """Review focus: crop fractions apply to the JPEG's own size, whatever the mux's."""
    im = Image.open(io.BytesIO(cropped(jpeg(1280, 720), (0.5, 0.0, 1.0, 1.0), 640)))
    assert im.size == (640, 720) and im.getpixel((320, 360))[2] > 150           # the right (blue) half
    assert Image.open(io.BytesIO(cropped(jpeg(), (0.25, 0.0, 0.75, 0.5), 640))).size == (640, 360)
    assert Image.open(io.BytesIO(cropped(jpeg(), (0.0, 0.0, 0.2, 0.2), 640))).size == (384, 216)   # never enlarged


def test_a_photo_is_saved_and_linked_and_an_end_photo_replaced(tmp_path):
    db = str(tmp_path / "b.db")
    conn = connect(db)
    urls = []
    w = HaircutPhotos("http://go2rtc:1984", tmp_path / "crops", db, fetch=lambda u: urls.append(u) or jpeg())
    w.request(req("start"), 640)
    w.request(req("end", T + timedelta(minutes=2)), 640)
    w.request(req("end", T + timedelta(minutes=4)), 640)
    assert _wait_for(lambda: haircut_photos(conn, ["h1"]).get("h1", {}).get("end", {}).get("taken_at")
                     == "2026-10-22T12:04:00")
    assert urls[0] == "http://go2rtc:1984/api/frame.jpeg?src=ph-2"
    assert haircut_photos(conn, ["h1"])["h1"]["start"]["path"] == "haircuts/h1_start.jpg"
    assert sorted(p.name for p in (tmp_path / "crops" / "haircuts").iterdir()) == ["h1_end.jpg", "h1_start.jpg"]
    w.stop()


def test_a_failed_fetch_leaves_no_photo_and_warns_once_a_minute_per_camera(tmp_path, caplog):
    db = str(tmp_path / "b.db")
    conn = connect(db)

    def down(url):
        raise OSError("connection refused")
    w = HaircutPhotos("http://go2rtc:1984", tmp_path / "crops", db, fetch=down)
    with caplog.at_level(logging.WARNING, logger="barber.haircut_photos"):
        w.request(req("start"), 640)
        w.request(req("end"), 640)
        w.stop()
    assert haircut_photos(conn, ["h1"]) == {} and not (tmp_path / "crops" / "haircuts").exists()
    assert sum("no start photo" in r.message or "no end photo" in r.message for r in caplog.records) == 1


def test_a_drop_after_its_requests_removes_the_files_and_rows(tmp_path):
    db = str(tmp_path / "b.db")
    conn = connect(db)
    w = HaircutPhotos("http://go2rtc:1984", tmp_path / "crops", db, fetch=lambda u: jpeg())
    w.request(req("start"), 640)
    w.drop("h1")
    w.stop()
    assert haircut_photos(conn, ["h1"]) == {} and not (tmp_path / "crops" / "haircuts" / "h1_start.jpg").exists()


def test_a_full_queue_drops_the_request_and_never_blocks(tmp_path, caplog):
    db = str(tmp_path / "b.db")
    gate, small = threading.Event(), jpeg(64, 48)              # small: the 200 queued photos go fast after the gate
    w = HaircutPhotos("http://go2rtc:1984", tmp_path / "crops", db, fetch=lambda u: gate.wait(5) and small)
    with caplog.at_level(logging.WARNING, logger="barber.haircut_photos"):
        for i in range(205):
            w.request(req(hid=f"h{i}"), 640)
    assert any("queue full" in r.message for r in caplog.records)
    gate.set()
    w.stop()
