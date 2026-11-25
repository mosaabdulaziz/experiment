"""The one-off move of the old flat reports/ into the saved-reports layout (2026-10-25; spec section 4)."""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev"))

from migrate_reports import main, plan  # noqa: E402
from saved_reports import list_saved  # noqa: E402


def _file(path: Path, at: datetime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    os.utime(path, (at.timestamp(), at.timestamp()))


def _tree(tmp_path):
    root = tmp_path / "reports"
    t18 = datetime(2026, 10, 18, 19, 8)
    _file(root / "16oct_BekkyBarber_palmhills_analysis_report_en.pdf", t18)
    _file(root / "16oct_BekkyBarber_palmhills_analysis_report_ar.html", t18)
    _file(root / "17oct_BekkyBarber_palmhills_analysis_report_en.pdf", t18)
    _file(root / "2026-10-19_lap_09-30-19-00_staff-r3" / "16oct_BekkyBarber_palmhills_analysis_report_en.pdf",
          datetime(2026, 10, 19, 8, 6))
    _file(root / "final" / "16oct_BekkyBarber_palmhills_haircut_photos_ar.pdf", datetime(2026, 10, 19, 11, 49))
    return root


def test_the_plan_puts_day_files_in_day_folders_and_old_folders_in_lap_folders(tmp_path):
    root = _tree(tmp_path)
    moves = plan(root, laps=False, labels={"final": "final"})
    assert [(m.target, len(m.files), m.label) for m in moves] == [
        ("day/2026-10-16", 2, None), ("day/2026-10-17", 1, None),
        ("day/2026-10-16/lap-2026-10-19_0806", 1, "2026-10-19_lap_09-30-19-00_staff-r3"),
        ("day/2026-10-16/lap-2026-10-19_1149", 1, "final")]
    assert [m.target for m in plan(root, laps=True, labels={})][:2] == \
        ["day/2026-10-16/lap-2026-10-18_1908", "day/2026-10-17/lap-2026-10-18_1908"]


def test_a_dry_run_moves_nothing_and_apply_moves_and_writes_the_manifests(tmp_path, capsys):
    root = _tree(tmp_path)
    assert main([str(tmp_path)]) == 0 and "dry run" in capsys.readouterr().out
    assert (root / "16oct_BekkyBarber_palmhills_analysis_report_en.pdf").is_file()
    assert main([str(tmp_path), "--apply", "--label", "final=final"]) == 0
    assert sorted(p.name for p in root.iterdir()) == ["day"]                    # the old folders are gone
    m = json.loads((root / "day" / "2026-10-16" / "report.json").read_text())
    assert (m["kind"], m["first"], m["saved_at"], m["lap_ended"], m["label"]) == \
        ("day", "2026-10-16", "2026-10-18T19:08:00", None, None)
    assert {(f["report"], f["lang"], f["format"]) for f in m["files"]} == {("analysis", "en", "pdf"), ("analysis", "ar", "html")}
    lap = json.loads((root / "day" / "2026-10-16" / "lap-2026-10-19_1149" / "report.json").read_text())
    assert (lap["lap_ended"], lap["label"], lap["files"][0]["report"]) == ("2026-10-19T11:49:00", "final", "photos")
    assert len(list_saved(root, "day")) == 4
    assert main([str(tmp_path)]) == 0 and "nothing to move" in capsys.readouterr().out      # a second run
    assert main([str(tmp_path / "no-site")]) == 0                                           # no reports/ at all


def test_a_target_that_exists_or_a_name_without_a_day_moves_nothing(tmp_path):
    root = _tree(tmp_path)
    (root / "day" / "2026-10-17").mkdir(parents=True)
    assert main([str(tmp_path), "--apply"]) == 2
    assert (root / "16oct_BekkyBarber_palmhills_analysis_report_en.pdf").is_file()
    (root / "day" / "2026-10-17").rmdir()
    (root / "day").rmdir()
    _file(root / "11oct_17oct_BekkyBarber_palmhills_analysis_report_en.pdf", datetime(2026, 10, 18, 19, 8))   # a week
    assert main([str(tmp_path), "--apply"]) == 2
    assert (root / "16oct_BekkyBarber_palmhills_analysis_report_en.pdf").is_file()


def test_the_day_start_comes_from_the_sites_policy(tmp_path, capsys):
    """2026-10-25 final review: Palm Hills' shop day starts at 9, not 5."""
    root = _tree(tmp_path)
    (tmp_path / "policy.json").write_text('{"day_start_hour": 9.0}')
    assert main([str(tmp_path), "--apply"]) == 0
    assert "day_start_hour: 9" in capsys.readouterr().out
    m = json.loads((root / "day" / "2026-10-16" / "report.json").read_text())
    assert m["start"] == "2026-10-16T09:00:00"


def test_without_a_policy_or_with_a_flag_the_day_start_is_5_or_the_flag(tmp_path, capsys):
    root = _tree(tmp_path)
    assert main([str(tmp_path)]) == 0 and "day_start_hour: 5" in capsys.readouterr().out
    assert main([str(tmp_path), "--apply"]) == 0
    assert json.loads((root / "day" / "2026-10-16" / "report.json").read_text())["start"] == "2026-10-16T05:00:00"
    (tmp_path / "policy.json").write_text('{"day_start_hour": 9.0}')
    assert main([str(tmp_path), "--day-start-hour", "7"]) == 0 and "day_start_hour: 7" in capsys.readouterr().out
