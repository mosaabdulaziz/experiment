"""Crop files of the customer register (project C), under one root
(data/barber/crops). Paths are always relative to the root; nothing outside
it is ever deleted."""
from __future__ import annotations

import logging
import os
import posixpath
from pathlib import Path

logger = logging.getLogger("barber.crops")


def inside(root: Path, rel: str) -> Path | None:
    root = root.resolve()
    p = (root / rel).resolve()
    return p if p.is_relative_to(root) and p != root else None


def delete(root: Path, rel: str) -> bool:
    p = inside(root, rel)
    if p is None or not p.is_file():
        return False
    try:
        p.unlink(missing_ok=True)
    except OSError:
        logger.warning("could not delete crop file %s (rel %s) -- the nightly sweep will retry", p, rel,
                       exc_info=True)
        return False
    return True


def sweep_orphans(root: Path, referenced: set[str], now: float, older_than_s: float) -> int:
    """Delete crop files no register row references and older than
    older_than_s (candidates of visits still running are younger).
    Both the walked paths and `referenced` are normalised before comparing,
    so a stray "./faces/x.jpg" or "faces//x.jpg" in the DB never causes a
    referenced file to be treated as an orphan. Empty day directories left
    behind are removed too, but never crops_root itself or the top-level
    faces/bodies directories."""
    root = root.resolve()
    keep = {posixpath.normpath(r) for r in referenced}
    n = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            p = Path(dirpath) / name
            rel = posixpath.normpath(p.relative_to(root).as_posix())
            if rel in keep:
                continue
            try:
                if now - p.stat().st_mtime > older_than_s:
                    p.unlink()
                    n += 1
            except FileNotFoundError:
                continue
            except OSError:
                logger.warning("could not sweep crop file %s", p, exc_info=True)
    _remove_empty_day_dirs(root)
    return n


def _remove_empty_day_dirs(root: Path) -> None:
    for top in ("faces", "bodies"):
        base = root / top
        if not base.is_dir():
            continue
        for day_dir in base.iterdir():
            if day_dir.is_dir():
                try:
                    day_dir.rmdir()  # only succeeds when empty
                except OSError:
                    pass
