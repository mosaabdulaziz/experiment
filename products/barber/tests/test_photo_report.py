"""The haircut photo report (2026-10-22, spec haircut photos section 4)."""
from datetime import date, datetime, timedelta

from photo_report import build_photo_report, photo_file_name, render_photo_html
from storage import set_haircut_photo
from test_report_data import P, SITE, _shop

NAMES = {"ph2-L7": "L7", "L8": "L8"}
JPEG = b"\xff\xd8\xff\xe0fakejpeg"


def _model(tmp_path, clock=None, day=date(2026, 10, 11)):
    conn, T = _shop(tmp_path)                    # s0, s2 on L7 (done), s1 on L8 (done), s9 on L8 (running)
    set_haircut_photo(conn, "s0", "start", "haircuts/s0_start.jpg", T + timedelta(seconds=20))
    set_haircut_photo(conn, "s0", "end", "haircuts/s0_end.jpg", T + timedelta(minutes=29))
    set_haircut_photo(conn, "s1", "start", "haircuts/missing.jpg", T + timedelta(minutes=10))
    set_haircut_photo(conn, "s9", "end", "haircuts/s9_end.jpg", T + timedelta(hours=3, minutes=5))
    return build_photo_report(conn, day, T + timedelta(hours=3, minutes=10), P, SITE, NAMES, clock=clock), T


def test_the_model_gives_each_haircut_its_photos_and_counts_those_with_both(tmp_path):
    m, _ = _model(tmp_path)
    rows = {r["id"]: r for c in m["chairs"] for r in c["rows"]}
    assert set(rows) == {"s0", "s1", "s2", "s9"} and m["photos"] == {"haircuts": 4, "both": 1}
    assert rows["s0"]["photos"]["start"] == {"path": "haircuts/s0_start.jpg", "taken_at": "2026-10-11T12:00:20"}
    assert rows["s2"]["photos"] == {"start": None, "end": None}
    assert rows["s9"]["state"] == "running" and rows["s9"]["photos"]["start"] is None


def test_a_run_on_recordings_gives_the_photos_footage_times(tmp_path):
    from footage_clock import FootageClock
    foot = datetime(2026, 10, 16, 9, 30)                                   # = device 11 Oct 12:00
    clock = FootageClock(foot, foot - datetime(2026, 10, 11, 12, 0), 34200.0)
    m, _ = _model(tmp_path, clock=clock, day=date(2026, 10, 16))
    s0 = next(r for c in m["chairs"] for r in c["rows"] if r["id"] == "s0")
    assert s0["photos"]["start"]["taken_at"] == "2026-10-16T09:30:20" and s0["start"] == "2026-10-16T09:30:00"


def test_the_page_shows_a_card_per_haircut_with_two_captioned_photos(tmp_path):
    m, _ = _model(tmp_path)
    html = render_photo_html(m, "en", lambda rel: None if rel == "haircuts/missing.jpg" else JPEG)
    assert html.count('class="cut"') == 4 and html.count("data:image/jpeg;base64,") == 3     # s0 x2, s9's end
    assert html.count("No picture") == 5                                       # s1's missing file and the rest
    assert "Start 12:00" in html and "End 12:29" in html and "End 15:05 · so far" in html
    assert "1 of 4 haircuts with both photos" in html and "Haircut photos" in html
    pos = [html.index(f"<h2>{c}</h2>") for c in ("L7", "L8")]
    assert pos == sorted(pos)


def test_a_missing_file_is_no_picture(tmp_path):
    """Review focus: a row whose file is gone shows the placeholder, never fails."""
    m, _ = _model(tmp_path)
    html = render_photo_html(m, "en", lambda rel: None)
    assert "data:image/jpeg" not in html and html.count("No picture") == 8


def test_arabic_is_right_to_left_and_the_printed_page_is_numbered(tmp_path):
    m, _ = _model(tmp_path)
    html = render_photo_html(m, "ar", lambda rel: JPEG, printing=True)
    assert 'dir="rtl"' in html and "صور القصّات" in html and "counter(pages)" in html


def test_a_day_without_haircuts_says_so(tmp_path):
    """Review focus: an empty day renders, with the analysis report's no-data line."""
    m, _ = _model(tmp_path, day=date(2026, 10, 10))
    assert m["photos"] == {"haircuts": 0, "both": 0}
    assert "No data in this period" in render_photo_html(m, "en", lambda rel: JPEG)


def test_the_file_name_is_the_analysis_reports_with_haircut_photos():
    s = datetime(2026, 10, 16, 9, 0)
    assert photo_file_name(s, s + timedelta(days=1), "BekkyBarber", "palmhills", "pdf") == \
        "16oct_BekkyBarber_palmhills_haircut_photos.pdf"


def _pages(pdf: bytes) -> int:
    import re
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf))


def test_three_haircut_cards_fit_on_a_printed_page(tmp_path):
    """2026-10-22 (layout check): chair close-ups are portrait; at half the page's width one card filled a page."""
    import io
    import pytest
    from PIL import Image
    from report_pdf import chrome_path, html_to_pdf
    if chrome_path() is None:
        pytest.skip("no Chrome on this machine")
    m, _ = _model(tmp_path)
    c = m["chairs"][0]
    c["rows"] = [dict(c["rows"][0], id=f"x{i}") for i in range(6)]
    m["chairs"] = [c]
    buf = io.BytesIO()
    Image.new("RGB", (360, 540), (120, 90, 60)).save(buf, "JPEG")
    portrait = buf.getvalue()
    pdf = html_to_pdf(render_photo_html(m, "en", lambda rel: portrait, printing=True), timeout=120)
    assert _pages(pdf) <= 3


# ── pedicure chairs (2026-11-10, spec 2026-11-10-barber-pedicure-chairs-design.md) ──────────────────────────────
def test_pedicures_get_their_own_photo_cards(tmp_path):
    from photo_report import render_photo_html
    m, _ = _model(tmp_path)
    m["pedicure"] = {"count": 1, "total_s": 2460.0, "chairs": [{"chair": "P1", "total_s": 2460.0, "rows": [
        {"id": "p1", "start": "2026-10-16T14:16:00", "end": "2026-10-16T14:57:00", "state": "done", "duration_s": 2460.0,
         "excessive": False, "photos": {"start": None, "end": None}}]}]}
    html = render_photo_html(m, "en", lambda _: None)
    assert "Pedicure" in html and html.count('class="cut"') == 1 + sum(len(c["rows"]) for c in m["chairs"])


def test_the_photo_model_reads_pedicure_photos(tmp_path):
    from datetime import date, datetime, timedelta
    from domain import EngineUpdate, PedicureRecord
    from photo_report import build_photo_report
    from policy import DEFAULT_POLICY
    from storage import apply_update, connect, set_haircut_photo
    conn = connect(tmp_path / "b.db")
    t = datetime(2026, 10, 16, 14, 16)
    apply_update(conn, EngineUpdate(pedicures={"p1": PedicureRecord("p1", "ph6-P1", t, t + timedelta(minutes=41), 2460.0,
                                                                     False)}), t)
    set_haircut_photo(conn, "p1", "start", "haircuts/p1-start.jpg", t)
    m = build_photo_report(conn, date(2026, 10, 16), t + timedelta(hours=2), {**DEFAULT_POLICY, "day_start_hour": 9.0},
                           {"place": "BekkyBarber", "branch": "Palm Hills", "branch_slug": "palmhills"}, {"ph6-P1": "P1"})
    (r,) = m["pedicure"]["chairs"][0]["rows"]
    assert r["photos"]["start"]["path"] == "haircuts/p1-start.jpg" and r["photos"]["end"] is None


def test_a_low_occupancy_haircut_has_a_highlighted_card_with_a_note(tmp_path):
    m, _ = _model(tmp_path)                                     # 2026-11-14 (user)
    plain = render_photo_html(m, "en", lambda rel: JPEG)
    assert '<div class="cut low">' not in plain and 'class="lowNote"' not in plain
    m["chairs"][0]["rows"][0]["low_occupancy"] = True
    html = render_photo_html(m, "en", lambda rel: JPEG)
    assert html.count('<div class="cut low">') == 1
    assert '<span class="lowNote">Low occupancy: possibly not a haircut</span>' in html
    assert "إشغال منخفض" in render_photo_html(m, "ar", lambda rel: JPEG)
