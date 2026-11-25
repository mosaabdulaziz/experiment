"""Lineage facts for a run: git, DVC data revisions, lockfile, environment, host.

Pure functions, no MLflow import: everything here returns plain dicts of string tags, so it is testable without
MLflow installed and never fails a run (a fact that cannot be read becomes a tag saying so, plus a warning).
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import platform
import re
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("mtmc.tracking")

# Packages whose versions decide what a training/export run produces (Part A of the prerequisites: the Ultralytics
# version the deployed vehicle model was trained with is recorded nowhere). Read from installed metadata, so nothing
# heavy (torch) is imported just to tag a run.
ENV_PACKAGES = ("torch", "torchvision", "ultralytics", "onnx", "onnxslim", "onnxruntime", "onnxruntime-gpu", "mlflow")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def repo_root(start: Path | None = None) -> Path:
    """The git work tree containing `start` (default: this package), or the package's repo by layout."""
    start = Path(start or Path(__file__).resolve().parent)
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=start, capture_output=True, text=True,
                             timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return Path(__file__).resolve().parents[2]


@dataclass
class GitInfo:
    tags: dict[str, str]
    dirty_files: list[str] = field(default_factory=list)


def git_info(root: Path) -> GitInfo:
    """`git.commit`, `git.branch`, `git.dirty`, plus the modified tracked files when dirty.

    Dirty means a tracked file differs from HEAD (staged or not). Untracked files do not count: run outputs, pulled DVC
    data and local scratch would otherwise make every run dirty.
    """
    def git(*args: str) -> str | None:
        try:
            p = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return p.stdout if p.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        logger.warning("tracking: %s is not a git work tree; git.commit=unknown", root)
        return GitInfo({"git.commit": "unknown", "git.branch": "unknown", "git.dirty": "unknown"})
    branch = (git("rev-parse", "--abbrev-ref", "HEAD") or "unknown").strip()  # "HEAD" when detached
    status = git("status", "--porcelain", "--untracked-files=no") or ""
    dirty = [line[3:] for line in status.splitlines() if line.strip()]
    return GitInfo({"git.commit": commit.strip(), "git.branch": branch, "git.dirty": "true" if dirty else "false"},
                   dirty)


_MD5 = re.compile(r"^\s*-?\s*md5:\s*['\"]?([0-9a-f]{32}(?:\.dir)?)['\"]?\s*$")
_PATH = re.compile(r"^\s*-?\s*path:\s*['\"]?(.+?)['\"]?\s*$")


def parse_dvc_pointer(pointer: Path) -> dict[str, str]:
    """The first `outs` entry of a .dvc pointer: {"md5": ..., "path": ...}.

    Parsed line by line, without the dvc package or a YAML parser: the pointers are machine-written and flat
    (`outs:` then `- md5: ...`, `size: ...`, `path: ...`). A directory output's md5 ends in `.dir`.
    """
    md5 = path = None
    in_outs = False
    for line in pointer.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[0].isspace() and not line.startswith("-"):   # a top-level key
            if in_outs and md5:
                break                                            # past the first output
            in_outs = line.split(":", 1)[0].strip() == "outs"
            continue
        if not in_outs:
            continue
        if line.lstrip().startswith("-") and md5 is not None:    # the second output starts
            break
        if md5 is None and (m := _MD5.match(line)):
            md5 = m.group(1)
        elif path is None and (m := _PATH.match(line)):
            path = m.group(1)
    if md5 is None:
        raise ValueError(f"{pointer}: no outs[0].md5 in this .dvc pointer")
    return {"md5": md5, "path": path or ""}


def find_dvc_pointer(path: Path) -> Path | None:
    """The .dvc pointer that versions `path`: the path itself, `<path>.dvc`, or `<ancestor>.dvc` (a file inside a
    tracked directory, e.g. a dataset's data.yaml)."""
    path = Path(path)
    if path.suffix == ".dvc" and path.is_file():
        return path
    for p in (path, *path.parents):
        if p.name and (candidate := p.with_name(p.name + ".dvc")).is_file():
            return candidate
    return None


def data_tags(data: dict[str, str | Path] | list[str | Path] | None, root: Path) -> dict[str, str]:
    """`data.<name>.dvc_md5` and `data.<name>.dvc_path` per dataset (D6).

    `data` maps a name to a dataset path (the .dvc pointer, the tracked file/dir, or a file inside a tracked dir); a
    list uses each path's stem (without `.dvc`) as the name. A path that no pointer versions is tagged
    `dvc_md5=unversioned` with `data.<name>.sha256` of the file when it is one, and a warning: the run still starts.
    """
    if not data:
        return {}
    items = data.items() if isinstance(data, dict) else ((None, p) for p in data)
    tags: dict[str, str] = {}
    for name, raw in items:
        p = Path(raw)
        if not p.is_absolute():
            p = root / p
        pointer = find_dvc_pointer(p)
        if name is None:
            name = pointer.name[: -len(".dvc")] if pointer else p.stem
        key = f"data.{name}"
        tags[f"{key}.path"] = _rel(p, root)
        if pointer is None:
            logger.warning("tracking: %s is not versioned by a .dvc pointer; tagging %s.dvc_md5=unversioned", p, key)
            tags[f"{key}.dvc_md5"] = "unversioned"
            if p.is_file():
                tags[f"{key}.sha256"] = sha256_file(p)
            continue
        try:
            out = parse_dvc_pointer(pointer)
        except (OSError, ValueError) as exc:
            logger.warning("tracking: cannot read %s (%s); tagging %s.dvc_md5=unreadable", pointer, exc, key)
            tags[f"{key}.dvc_md5"] = "unreadable"
            tags[f"{key}.dvc_path"] = _rel(pointer, root)
            continue
        tags[f"{key}.dvc_md5"] = out["md5"]
        tags[f"{key}.dvc_path"] = _rel(pointer, root)
    return tags


def lockfile_tags(lockfile: str | Path | None, root: Path) -> dict[str, str]:
    """`env.lockfile` and `env.lockfile_sha256` (D7). A lockfile that does not exist in this checkout is tagged
    `env.lockfile=missing` with the expected path, and warned about; it never fails the run, but the registry refuses
    to register a source=trained run without a real lockfile."""
    if lockfile is None:
        return {"env.lockfile": "none"}
    p = Path(lockfile)
    if not p.is_absolute():
        p = root / p
    if not p.is_file():
        logger.warning("tracking: lockfile %s does not exist (an old checkout?); tagging env.lockfile=missing and "
                       "logging the installed packages instead (the registry refuses a trained run without one)", p)
        return {"env.lockfile": "missing", "env.lockfile_expected": _rel(p, root)}
    return {"env.lockfile": _rel(p, root), "env.lockfile_sha256": sha256_file(p)}


def env_tags() -> dict[str, str]:
    tags = {"env.python": platform.python_version(), "host": socket.gethostname()}
    for pkg in ENV_PACKAGES:
        try:
            tags[f"env.{pkg}"] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    return tags


def installed_packages() -> str:
    """`name==version` for every installed distribution: the record when there is no lockfile to hash (D7)."""
    seen = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            seen[name.lower()] = f"{name}=={dist.version}"
    return "\n".join(seen[k] for k in sorted(seen)) + "\n"


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(p)
