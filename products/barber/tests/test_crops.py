import logging
import os
import time
from pathlib import Path

from crops import delete, sweep_orphans


def test_delete_stays_inside_the_root(tmp_path):
    (tmp_path / "faces").mkdir()
    (tmp_path / "faces" / "a.jpg").write_bytes(b"x")
    outside = tmp_path.parent / "keep.jpg"
    outside.write_bytes(b"x")
    assert delete(tmp_path, "faces/a.jpg") and not (tmp_path / "faces" / "a.jpg").exists()
    assert not delete(tmp_path, "../keep.jpg") and outside.exists()
    assert not delete(tmp_path, "faces/missing.jpg")


def test_delete_logs_and_returns_false_on_a_permission_error(tmp_path, monkeypatch, caplog):
    (tmp_path / "faces").mkdir()
    f = tmp_path / "faces" / "a.jpg"
    f.write_bytes(b"x")

    def boom(self, missing_ok=False):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "unlink", boom)
    with caplog.at_level(logging.WARNING):
        assert delete(tmp_path, "faces/a.jpg") is False
    assert f.exists()
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_sweep_removes_old_unreferenced_files_only(tmp_path):
    d = tmp_path / "faces" / "2026-09-25"
    d.mkdir(parents=True)
    for name in ("old.jpg", "kept.jpg", "new.jpg"):
        (d / name).write_bytes(b"x")
    old = time.time() - 3 * 86400
    os.utime(d / "old.jpg", (old, old))
    os.utime(d / "kept.jpg", (old, old))
    n = sweep_orphans(tmp_path, {"faces/2026-09-25/kept.jpg"}, time.time(), 86400)
    assert n == 1 and not (d / "old.jpg").exists() and (d / "kept.jpg").exists() and (d / "new.jpg").exists()


def test_sweep_survives_a_file_disappearing_mid_walk(tmp_path, monkeypatch):
    d = tmp_path / "faces" / "2026-09-25"
    d.mkdir(parents=True)
    gone = d / "gone.jpg"
    gone.write_bytes(b"x")
    kept = d / "kept.jpg"
    kept.write_bytes(b"x")
    old = time.time() - 3 * 86400
    os.utime(gone, (old, old))
    os.utime(kept, (old, old))

    orig_stat = Path.stat

    def flaky_stat(self, *a, **kw):
        if self.name == "gone.jpg":
            raise FileNotFoundError()
        return orig_stat(self, *a, **kw)

    monkeypatch.setattr(Path, "stat", flaky_stat)
    n = sweep_orphans(tmp_path, set(), time.time(), 86400)
    assert n == 1 and not kept.exists()


def test_sweep_removes_empty_day_dirs_but_never_the_roots(tmp_path):
    d = tmp_path / "faces" / "2026-09-25"
    d.mkdir(parents=True)
    (tmp_path / "bodies").mkdir()
    old_file = d / "old.jpg"
    old_file.write_bytes(b"x")
    old = time.time() - 3 * 86400
    os.utime(old_file, (old, old))
    sweep_orphans(tmp_path, set(), time.time(), 86400)
    assert not d.exists()
    assert (tmp_path / "faces").is_dir()
    assert (tmp_path / "bodies").is_dir()
    assert tmp_path.is_dir()


def test_sweep_normalises_paths_before_comparing(tmp_path):
    d = tmp_path / "faces" / "2026-09-25"
    d.mkdir(parents=True)
    f = d / "x.jpg"
    f.write_bytes(b"x")
    old = time.time() - 3 * 86400
    os.utime(f, (old, old))
    assert sweep_orphans(tmp_path, {"./faces/2026-09-25/x.jpg"}, time.time(), 86400) == 0 and f.exists()
    assert sweep_orphans(tmp_path, {"faces//2026-09-25/x.jpg"}, time.time(), 86400) == 0 and f.exists()


def test_sweep_skips_a_file_it_cannot_delete(tmp_path, monkeypatch):
    for name in ("a.jpg", "b.jpg"):
        (tmp_path / "faces" / "d").mkdir(parents=True, exist_ok=True)
        (tmp_path / "faces" / "d" / name).write_bytes(b"x")
    real = Path.unlink

    def unlink(self, *a, **k):
        if self.name == "a.jpg":
            raise PermissionError("root-owned")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", unlink)
    assert sweep_orphans(tmp_path, set(), time.time() + 10, 0) == 1
    assert (tmp_path / "faces" / "d" / "a.jpg").exists()
    assert not (tmp_path / "faces" / "d" / "b.jpg").exists()
