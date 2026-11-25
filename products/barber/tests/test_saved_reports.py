"""Saved reports (2026-10-25; spec 2026-10-25-barber-saved-reports-design.md): settings, periods, names."""
import json
import time
from dataclasses import replace
from datetime import date, datetime
from datetime import time as clock_time

import pytest

from saved_reports import (MANIFEST, PENDING, PRINT_DIR, Hooks, Pages, Period, ReportSaver, SavedReports, day_of,
                           due_periods, lap_folder, list_saved, month_of, out_name, parse_saved_reports, saved_file,
                           week_of, write_manifest)


def test_the_site_block_is_read_and_checked():
    assert parse_saved_reports(None) is None
    assert parse_saved_reports({"langs": ["en", "ar"], "formats": ["pdf", "html"]}) == \
        SavedReports(("en", "ar"), ("pdf", "html"), 5)
    assert parse_saved_reports({"langs": ["en"], "week_start": "mon"}).week_start == 0
    for bad in ({"langs": ["fr"]}, {"formats": ["xls"]}, {"week_start": "someday"}):
        with pytest.raises(ValueError):
            parse_saved_reports(bad)


def test_weeks_run_saturday_to_friday_and_months_by_the_calendar():
    assert week_of(date(2026, 10, 23), 5) == Period("week", date(2026, 10, 17), date(2026, 10, 23))   # a Friday
    assert week_of(date(2026, 10, 17), 5) == Period("week", date(2026, 10, 17), date(2026, 10, 23))   # its Saturday
    assert week_of(date(2026, 10, 25), 0).first == date(2026, 10, 19)                                  # Monday weeks
    assert month_of(date(2026, 2, 14)) == Period("month", date(2026, 2, 1), date(2026, 2, 28))
    assert [p.folder for p in (day_of(date(2026, 10, 24)), week_of(date(2026, 10, 24), 5), month_of(date(2026, 10, 24)))] \
        == ["day/2026-10-24", "week/2026-10-24", "month/2026-10"]


def test_a_day_is_due_once_the_next_shop_day_has_started_and_seven_days_back():
    before = due_periods(datetime(2026, 10, 25, 4, 59), 5, 5)          # still the 24th's shop day
    after = due_periods(datetime(2026, 10, 25, 5, 0), 5, 5)
    assert day_of(date(2026, 10, 24)) not in before and day_of(date(2026, 10, 24)) in after
    assert [p.first for p in after if p.kind == "day"] == [date(2026, 10, d) for d in range(24, 17, -1)]


def test_the_week_is_due_after_its_friday_and_the_month_after_its_last_day():
    assert week_of(date(2026, 10, 17), 5) in due_periods(datetime(2026, 10, 24, 5, 0), 5, 5)       # Sat 17 - Fri 23
    friday_night = due_periods(datetime(2026, 10, 24, 4, 0), 5, 5)                                 # Fri 23 still running
    assert [p for p in friday_night if p.kind == "week"] == [week_of(date(2026, 10, 10), 5)]
    assert [p for p in due_periods(datetime(2026, 11, 1, 5, 0), 5, 5) if p.kind == "month"] == [month_of(date(2026, 10, 1))]
    assert [p for p in due_periods(datetime(2026, 11, 1, 4, 0), 5, 5) if p.kind == "month"] == [month_of(date(2026, 9, 1))]


def test_file_and_lap_folder_names():
    assert out_name("16oct_BekkyBarber_palmhills_haircut_photos.pdf", "ar") == \
        "16oct_BekkyBarber_palmhills_haircut_photos_ar.pdf"
    assert lap_folder(date(2026, 10, 16), datetime(2026, 10, 22, 13, 13, 18)) == "day/2026-10-16/lap-2026-10-22_1313"


NOW = datetime(2026, 10, 25, 5, 0)


def _saver(tmp_path, now, live=True, photos=True, visit_days=(date(2026, 10, 24),), fail_build=(), fail_print=False,
           formats=("pdf", "html"), langs=("en", "ar")):
    built, printed = [], []

    def build(period, report, lang, docx):
        built.append((period.folder, report, lang))
        if (report, lang) in fail_build:
            raise RuntimeError("render error")
        return Pages(f"<p>{report} {lang} {period.first}</p>", f"<p>print {report} {lang}</p>", b"PK" if docx else None)

    def print_pdf(html):
        if fail_print:
            raise RuntimeError("no Chrome")
        printed.append(html)
        return b"%PDF " + html.encode()

    hooks = Hooks(build=build,
                  name=lambda p, r, f: f"{p.first:%d%b}_{'haircut_photos' if r == 'photos' else 'analysis_report'}.{f}".lower(),
                  has_visits=lambda s, e: any(s <= datetime.combine(d, clock_time(12)) < e for d in visit_days),
                  photos=lambda: photos, print_pdf=print_pdf, day_start_hour=lambda: 5)
    s = ReportSaver(SavedReports(langs, formats, 5), tmp_path / "reports", hooks, live=live, clock=lambda: now[0])
    return s, built, printed


def test_a_finished_day_is_saved_once_with_its_photos_and_the_manifest_last(tmp_path):
    now = [datetime(2026, 10, 25, 4, 59)]
    s, built, _ = _saver(tmp_path, now)
    assert s.run_once() == [] and built == []                                   # the 24th is still running
    now[0] = NOW
    [folder] = s.run_once()
    assert folder == tmp_path / "reports" / "day" / "2026-10-24"
    assert sorted(p.name for p in folder.iterdir()) == [
        "24oct_analysis_report_ar.html", "24oct_analysis_report_ar.pdf", "24oct_analysis_report_en.html",
        "24oct_analysis_report_en.pdf", "24oct_haircut_photos_ar.html", "24oct_haircut_photos_ar.pdf",
        "24oct_haircut_photos_en.html", "24oct_haircut_photos_en.pdf", MANIFEST]
    assert (folder / "24oct_haircut_photos_en.pdf").read_bytes() == b"%PDF <p>print photos en</p>"
    m = json.loads((folder / MANIFEST).read_text())
    assert (m["kind"], m["first"], m["last"], m["lap_ended"], m["saved_at"]) == \
        ("day", "2026-10-24", "2026-10-24", None, "2026-10-25T05:00:00")
    assert (m["start"], m["end"]) == ("2026-10-24T05:00:00", "2026-10-25T05:00:00") and len(m["files"]) == 8
    assert s.run_once() == [] and len(built) == 4                                # never twice


def test_a_period_without_visits_gets_no_folder(tmp_path):
    s, built, _ = _saver(tmp_path, [NOW], visit_days=())
    assert s.run_once() == [] and built == [] and not (tmp_path / "reports").exists()


def test_weeks_and_months_are_saved_with_the_analysis_report_only(tmp_path):
    s, _, _ = _saver(tmp_path, [datetime(2026, 11, 1, 5, 0)], visit_days=(date(2026, 10, 28),))
    assert sorted(f.relative_to(tmp_path / "reports").as_posix() for f in s.run_once()) == \
        ["day/2026-10-28", "month/2026-10", "week/2026-10-24"]
    for kind in ("week", "month"):
        [item] = list_saved(tmp_path / "reports", kind)
        assert {f["report"] for f in item["files"]} == {"analysis"}


def test_a_service_that_was_down_catches_up_seven_days(tmp_path):
    s, _, _ = _saver(tmp_path, [NOW], visit_days=(date(2026, 10, 17), date(2026, 10, 18), date(2026, 10, 22)))
    days = sorted(f.name for f in s.run_once() if f.parent.name == "day")
    assert days == ["2026-10-18", "2026-10-22"]                                  # the 17th is 8 days back


def test_a_pdf_that_fails_keeps_the_html_unlisted_and_retries_only_the_print(tmp_path):
    s, built, _ = _saver(tmp_path, [NOW], fail_print=True)
    assert s.run_once() == [] and list_saved(tmp_path / "reports", "day") == []
    folder = tmp_path / "reports" / "day" / "2026-10-24"
    assert (folder / "24oct_analysis_report_en.html").is_file() and (folder / PRINT_DIR / PENDING).is_file()
    assert s.run_once() == [] and len(built) == 4                                # the HTML is never rebuilt
    s2, built2, printed2 = _saver(tmp_path, [NOW])
    assert s2.run_once() == [folder] and built2 == [] and len(printed2) == 4
    assert not (folder / PRINT_DIR).exists() and len(list_saved(tmp_path / "reports", "day")) == 1


def test_a_failed_build_leaves_no_pending_and_the_retry_builds_only_what_is_missing(tmp_path):
    s, _, _ = _saver(tmp_path, [NOW], fail_build={("photos", "ar")})
    assert s.run_once() == []
    assert not (tmp_path / "reports" / "day" / "2026-10-24" / PRINT_DIR / PENDING).exists()
    s2, built2, _ = _saver(tmp_path, [NOW])
    assert len(s2.run_once()) == 1 and built2 == [("day/2026-10-24", "photos", "ar")]


def test_photos_off_or_word_only_gives_no_photo_report(tmp_path):
    s, _, _ = _saver(tmp_path, [NOW], photos=False)
    s.run_once()
    s2, _, _ = _saver(tmp_path / "w", [NOW], formats=("docx",), langs=("en",))
    s2.run_once()
    assert not list((tmp_path / "reports" / "day" / "2026-10-24").glob("*haircut_photos*"))
    assert sorted(p.name for p in (tmp_path / "w" / "reports" / "day" / "2026-10-24").iterdir()) == \
        ["24oct_analysis_report_en.docx", MANIFEST]


def test_a_recording_lap_builds_its_html_now_and_prints_on_the_next_pass(tmp_path):
    s, built, printed = _saver(tmp_path, [NOW], live=False, visit_days=())
    folder = s.save_lap(date(2026, 10, 16), datetime(2026, 10, 22, 13, 13, 18))
    assert folder == tmp_path / "reports" / "day" / "2026-10-16" / "lap-2026-10-22_1313"
    assert (folder / "16oct_haircut_photos_ar.html").is_file() and printed == [] and not (folder / MANIFEST).exists()
    assert s.run_once() == [folder] and len(printed) == 4                       # no clock periods on a recording run
    [item] = list_saved(tmp_path / "reports", "day")
    assert item["id"] == "day/2026-10-16/lap-2026-10-22_1313" and item["lap_ended"] == "2026-10-22T13:13:18"


def test_a_second_lap_in_the_same_minute_gets_its_own_folder(tmp_path):
    s, _, _ = _saver(tmp_path, [NOW], live=False)
    a = s.save_lap(date(2026, 10, 16), datetime(2026, 10, 22, 13, 13, 18))
    b = s.save_lap(date(2026, 10, 16), datetime(2026, 10, 22, 13, 13, 50))
    assert a.name == "lap-2026-10-22_1313" and b.name == "lap-2026-10-22_1313-2"
    assert len(s.run_once()) == 2


def test_a_restart_finishes_a_lap_from_its_print_pages_without_the_rows(tmp_path):
    s, _, _ = _saver(tmp_path, [NOW], live=False, fail_print=True)
    folder = s.save_lap(date(2026, 10, 16), datetime(2026, 10, 22, 13, 13, 18))
    s.run_once()
    s2, built2, printed2 = _saver(tmp_path, [NOW], live=False, fail_build={("analysis", "en"), ("analysis", "ar"),
                                                                           ("photos", "en"), ("photos", "ar")})
    assert s2.run_once() == [folder] and built2 == [] and len(printed2) == 4


def test_listing_is_newest_first_and_skips_folders_without_a_good_manifest(tmp_path):
    root = tmp_path / "reports"
    for fid, first, lap in (("day/2026-10-16/lap-2026-10-22_1313", "2026-10-16", "2026-10-22T13:13:18"),
                            ("day/2026-10-16/lap-2026-10-23_0343", "2026-10-16", "2026-10-23T03:43:18"),
                            ("day/2026-10-24", "2026-10-24", None)):
        (root / fid).mkdir(parents=True)
        write_manifest(root / fid, {"kind": "day", "first": first, "last": first, "lap_ended": lap, "files": []})
    (root / "day" / "2026-10-20").mkdir()                                       # copied by hand, no manifest
    (root / "day" / "2026-10-20" / "x.pdf").write_bytes(b"%PDF")
    (root / "day" / "2026-10-21").mkdir()
    (root / "day" / "2026-10-21" / MANIFEST).write_text("{not json")
    assert [i["id"] for i in list_saved(root, "day")] == [
        "day/2026-10-24", "day/2026-10-16/lap-2026-10-23_0343", "day/2026-10-16/lap-2026-10-22_1313"]
    assert list_saved(root, "week") == []


def test_only_a_file_its_manifest_lists_is_served_and_a_deleted_one_is_none(tmp_path):
    root = tmp_path / "reports"
    folder = root / "day" / "2026-10-24"
    folder.mkdir(parents=True)
    for n in ("a_en.pdf", "b_en.html"):
        (folder / n).write_bytes(b"x")
    write_manifest(folder, {"kind": "day", "files": [{"name": "a_en.pdf"}, {"name": "b_en.html"}]})
    assert saved_file(root, "day/2026-10-24", "a_en.pdf") == folder / "a_en.pdf"
    (folder / "b_en.html").unlink()                                             # moved away by hand
    for fid, name in (("day/2026-10-24", "b_en.html"), ("day/2026-10-24", MANIFEST), ("day/2026-10-24", "../x"),
                      ("../..", "a_en.pdf"), ("day/2026-10-24/../../..", "a_en.pdf"), ("/etc", "passwd"), ("", "a_en.pdf")):
        assert saved_file(root, fid, name) is None, (fid, name)


def test_the_thread_prints_a_lap_soon_after_it_is_staged(tmp_path):
    s, _, _ = _saver(tmp_path, [NOW], live=False)
    s.every_s = 30                                                              # the wake, not the minute, runs it
    s.start()
    try:
        folder = s.save_lap(date(2026, 10, 16), datetime(2026, 10, 22, 13, 13, 18))
        for _ in range(100):
            if (folder / MANIFEST).exists():
                break
            time.sleep(0.05)
        assert (folder / MANIFEST).exists()
    finally:
        s.stop()


def test_a_lap_with_one_failed_build_is_retried_once_then_saved_without_it(tmp_path):
    s, built, _ = _saver(tmp_path, [NOW], live=False, fail_build={("photos", "ar")})
    folder = s.save_lap(date(2026, 10, 16), datetime(2026, 10, 22, 13, 13, 18))
    assert built.count(("day/2026-10-16", "photos", "ar")) == 2                  # the first try and the retry
    assert s.run_once() == [folder]
    m = json.loads((folder / MANIFEST).read_text())
    assert len(m["files"]) == 6 and not [f for f in m["files"] if (f["report"], f["lang"]) == ("photos", "ar")]
    assert [i["id"] for i in list_saved(tmp_path / "reports", "day")] == ["day/2026-10-16/lap-2026-10-22_1313"]


def test_one_bad_pending_file_does_not_stop_the_others(tmp_path):
    s, _, _ = _saver(tmp_path, [NOW], live=False)
    bad = tmp_path / "reports" / "day" / "2026-10-20" / PRINT_DIR
    bad.mkdir(parents=True)
    (bad / PENDING).write_text("{not json")
    folder = s.save_lap(date(2026, 10, 16), datetime(2026, 10, 22, 13, 13, 18))
    assert s.run_once() == [folder]


def test_a_period_whose_check_raises_does_not_stop_the_day(tmp_path):
    s, _, _ = _saver(tmp_path, [NOW], visit_days=(date(2026, 10, 24),))
    has_visits = s.hooks.has_visits

    def flaky(start, end):
        if start.date() == date(2026, 10, 17):                                  # the week's first day
            raise RuntimeError("database is locked")
        return has_visits(start, end)

    s.hooks = replace(s.hooks, has_visits=flaky)
    folders = s.run_once()
    assert [f.relative_to(tmp_path / "reports").as_posix() for f in folders] == ["day/2026-10-24"]


def test_a_write_error_in_a_lap_still_leaves_the_folder_to_finish(tmp_path, monkeypatch):
    """2026-10-25 final review: one failed write counts its pair as failed; the partial stage still saves the rest."""
    import saved_reports
    real, calls = saved_reports.write_atomic, [0]

    def flaky(path, data):
        calls[0] += 1
        if calls[0] == 2:
            raise OSError(28, "No space left on device")
        real(path, data)

    monkeypatch.setattr(saved_reports, "write_atomic", flaky)
    s, _, printed = _saver(tmp_path, [NOW], live=False, photos=False)
    folder = s.save_lap(date(2026, 10, 16), datetime(2026, 10, 22, 13, 13, 18))
    assert (folder / PRINT_DIR / PENDING).is_file()
    assert s.run_once() == [folder]
    assert (folder / MANIFEST).is_file() and printed


def test_a_failed_atomic_write_leaves_no_tmp_file(tmp_path, monkeypatch):
    from pathlib import Path
    from saved_reports import write_atomic

    def boom(self, target):
        raise OSError("full")

    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(OSError):
        write_atomic(tmp_path / "a.html", b"x")
    assert list(tmp_path.iterdir()) == []


def test_saved_file_refuses_odd_ids_and_names(tmp_path):
    folder = tmp_path / "day" / "2026-10-24"
    folder.mkdir(parents=True)
    (folder / "a.html").write_text("x")
    (tmp_path / "report.json").write_text("{}")
    write_manifest(folder, {"files": [{"name": "a.html"}, {"name": "../report.json"}]})
    assert saved_file(tmp_path, "day/2026-10-24", "a.html") == (folder / "a.html").resolve()
    assert saved_file(tmp_path, "day/2026-10-24\x00", "a.html") is None
    assert saved_file(tmp_path, "day/2026-10-24", "../report.json") is None
    assert saved_file(tmp_path, "x/2026-10-24", "a.html") is None


def test_a_manifest_the_page_could_not_read_is_not_listed(tmp_path):
    for d in range(21, 25):
        (tmp_path / "day" / f"2026-10-{d}").mkdir(parents=True)
    good = {"kind": "day", "first": "2026-10-24", "last": "2026-10-24", "lap_ended": None,
            "files": [{"report": "analysis", "lang": "en", "format": "html", "name": "a.html"}]}
    write_manifest(tmp_path / "day" / "2026-10-24", good)
    write_manifest(tmp_path / "day" / "2026-10-23", {**good, "first": "2026-10-23", "files": None})
    write_manifest(tmp_path / "day" / "2026-10-22", {**good, "first": None})
    write_manifest(tmp_path / "day" / "2026-10-21", {**good, "first": "2026-10-21", "files": [{"name": "a.html"}]})
    assert [i["id"] for i in list_saved(tmp_path, "day")] == ["day/2026-10-24"]
