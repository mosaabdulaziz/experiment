"""Every path git tracks must be creatable on Windows (and macOS).

A ':' in a file name (the CHIRLA clips were stored as `camera_4_2023-07-26-13:21:40_h264.mp4`) made `git pull` fail on
Windows with "error: invalid path" for the whole checkout. This also covers the data paths inside the DVC pointers
(`outs: - path:`), since `dvc pull`/`dvc checkout` has to create those files too.
"""
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WINDOWS_BAD_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')
RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?$", re.IGNORECASE)


def _problems(path: str) -> list[str]:
    out = []
    for part in path.split("/"):
        if WINDOWS_BAD_CHARS.search(part):
            out.append(f"{path!r}: {part!r} has a character Windows cannot store (one of < > : \" | ? * or a control)")
        elif part.endswith((" ", ".")) and part not in (".", ".."):
            out.append(f"{path!r}: {part!r} ends with a space or a dot")
        elif RESERVED.match(part):
            out.append(f"{path!r}: {part!r} is a reserved device name on Windows")
    return out


# never tracked, and big: not worth walking when there is no git to ask
_NOT_TRACKED_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def _tracked() -> list[str]:
    """The tracked paths: `git ls-files`, or, where git is not installed (the CI image), every file of the checkout.
    A fresh CI checkout holds only tracked files, so a bad name is found either way."""
    try:
        r = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True)
        return [p for p in r.stdout.split("\0") if p]
    except (FileNotFoundError, subprocess.CalledProcessError):
        found = []
        for dirpath, dirnames, filenames in os.walk(REPO):
            dirnames[:] = [d for d in dirnames if d not in _NOT_TRACKED_DIRS]
            rel = Path(dirpath).relative_to(REPO)
            found += [(rel / name).as_posix() for name in filenames]
        return found


def test_no_tracked_path_is_invalid_on_windows():
    bad = [p for path in _tracked() for p in _problems(path)]
    assert not bad, "\n" + "\n".join(bad) + "\nRename them (e.g. 13:21:40 -> 13-21-40) and fix what refers to them."


def test_no_dvc_data_path_is_invalid_on_windows():
    bad = []
    for path in _tracked():
        if not path.endswith(".dvc"):
            continue
        folder = path.rsplit("/", 1)[0] + "/" if "/" in path else ""
        for m in re.finditer(r"^\s*-?\s*path:\s*(.+?)\s*$", (REPO / path).read_text(), re.MULTILINE):
            bad += _problems(folder + m.group(1).strip("'\""))
    assert not bad, "\n" + "\n".join(bad)


def test_the_chirla_timestamp_parses_with_dashes_and_with_the_old_colons():
    sys.path.insert(0, str(REPO / "products" / "attendance"))
    from acl import wall_clock_for_spike
    expect = datetime(2023, 7, 26, 13, 21, 40)
    assert wall_clock_for_spike("camera_4_2023-07-26-13-21-40.avi", 0.0, 100.0) == expect
    assert wall_clock_for_spike("camera_4_2023-07-26-13:21:40.avi", 5.0, 100.0).second == 45
