"""Verify the model files a deployment loads against its committed model manifest, before anything loads them.

Standard library only (tests/test_model_manifest.py checks the imports): the engine imports this module, and the
product services, which run in their own venvs without the repo on sys.path, load this file by path:

    spec = importlib.util.spec_from_file_location("model_manifest", REPO_DIR / "perception/engine/model_manifest.py")

A manifest (JSON, one per deployment, generated from the registry's aliases by `python -m mtmc.tracking.manifest`)
lists every model file the deployment may load, as the registered version it is:

    {"schema": 1, "product": ..., "site": ..., "generated_at": ..., "generated_from_commit": ..., "tracking_uri": ...,
     "models": [{"path", "registered_model", "version", "alias", "run_id", "sha256", "size_bytes", "loaded_by"}]}

`path` is repo-relative (or `~/...` for files outside the repo). At startup the caller says which files it is about to
load (`ModelRef`s: plain files, nvinfer configs, tracker configs, model-pack directories); every one of them must be
listed, present and byte-identical (sha256) to the listed version, or startup is refused with one message naming every
problem and how to fix it. Listed files this configuration does not load (a fallback model, a disabled feature) are
not checked.

No manifest (the `model_manifest:` setting unset, or its file missing) only warns, unless MODEL_MANIFEST_REQUIRED is
set (1/true/yes): then it refuses too. A manifest that exists is always enforced.
"""
from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

logger = logging.getLogger("iris.manifest")

SCHEMA = 1
REQUIRED_ENV = "MODEL_MANIFEST_REQUIRED"
CONTAINER_ROOT = "/workspace/iris"      # how the configs name the repo root (docker-compose mounts ./ there)
ENTRY_FIELDS = ("path", "registered_model", "version", "alias", "run_id", "sha256", "size_bytes")
_CHUNK = 1 << 20

# nvinfer config keys that name a file the GIE reads; tracker config keys likewise. TensorRT engines are not here on
# purpose: they are built per device from the ONNX and never shipped (spike D1).
NVINFER_FILE_KEYS = ("onnx-file", "labelfile-path")
_TRACKER_ONNX = re.compile(r"^\s*onnxFile:\s*[\"']?([^\"'#\s]+)", re.M)
_TRACKER_REID_TYPE = re.compile(r"^\s*reidType:\s*(\d+)", re.M)


class ManifestError(RuntimeError):
    """Startup must be refused. The message names every problem found and what to run."""


@dataclass(frozen=True)
class ModelRef:
    """Something a deployment is about to load. kind: file | nvinfer (a GIE config: its onnx-file and labelfile-path)
    | tracker (an NvDCF config: its ReID onnxFile when reidType != 0; tracker-all: whenever named) | pack (a directory
    whose *.onnx all load)."""
    path: str
    via: str                                   # the setting that names it, for messages
    kind: str = "file"


@dataclass(frozen=True)
class Entry:
    path: str
    registered_model: str
    version: str
    alias: str
    run_id: str
    sha256: str
    size_bytes: int
    loaded_by: str = ""

    def describe(self) -> str:
        return f"{self.registered_model} v{self.version} ({self.alias})"


@dataclass
class Manifest:
    source: Path
    product: str
    site: str | None
    generated_at: str
    tracking_uri: str
    entries: dict[str, Entry]


@dataclass
class Verified:
    """What startup verification found. `sha256` maps every checked file's resolved path to its digest, so later
    steps (the TensorRT engine marker) need not hash it again."""
    manifest: Manifest | None
    checked: list[str] = field(default_factory=list)
    sha256: dict[str, str] = field(default_factory=dict)
    warning: str | None = None


# -- paths --------------------------------------------------------------------------------------------------------
def manifest_key(path: str | os.PathLike, root: str | os.PathLike, base: str | os.PathLike | None = None) -> str:
    """The manifest's spelling of a path: repo-relative POSIX, `~/...` for the home directory, else absolute.

    Configs name files as /workspace/iris/... (the container mount), relative to the repo root, or (inside an nvinfer
    config) relative to that config's directory (`base`)."""
    p = str(path)
    if p.startswith("~"):
        return Path(p).as_posix()
    if not os.path.isabs(p):
        p = os.path.join(str(base) if base else str(root), p)
    p = os.path.normpath(p)
    for prefix in (os.path.normpath(str(root)), CONTAINER_ROOT):
        if p == prefix or p.startswith(prefix.rstrip("/") + "/"):
            return Path(os.path.relpath(p, prefix)).as_posix()
    home = os.path.expanduser("~")
    if p.startswith(home.rstrip("/") + "/"):
        return "~/" + Path(os.path.relpath(p, home)).as_posix()
    return Path(p).as_posix()


def resolve(key: str, root: str | os.PathLike) -> Path:
    """Where a manifest path is on this machine."""
    if key.startswith("~"):
        return Path(os.path.expanduser(key))
    return Path(key) if os.path.isabs(key) else Path(root) / key


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


_DIGESTS: dict[tuple[str, int, int], str] = {}


def file_sha256(path: str | os.PathLike) -> str:
    """sha256_file, remembered for this process by (real path, size, mtime): the startup check hashes each model once,
    and later steps (the TensorRT engine marker, engine_path.py) get the same digest without reading the file again.
    A file rewritten in between has a new size or mtime, so it is hashed afresh."""
    real = os.path.realpath(path)
    st = os.stat(real)
    key = (real, st.st_size, st.st_mtime_ns)
    if key not in _DIGESTS:
        _DIGESTS[key] = sha256_file(real)
    return _DIGESTS[key]


# -- reading ------------------------------------------------------------------------------------------------------
def load_manifest(path: str | os.PathLike) -> Manifest:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"model manifest {path}: unreadable ({exc}). Refusing to start.") from exc
    if raw.get("schema") != SCHEMA:
        raise ManifestError(f"model manifest {path}: schema {raw.get('schema')!r}, this verifier reads schema "
                            f"{SCHEMA}. Refusing to start.")
    entries: dict[str, Entry] = {}
    for i, m in enumerate(raw.get("models") or []):
        missing = [k for k in ENTRY_FIELDS if m.get(k) in (None, "")]
        if missing:
            raise ManifestError(f"model manifest {path}: models[{i}] lacks {missing}. Refusing to start.")
        key = Path(str(m["path"])).as_posix()
        if key in entries:
            raise ManifestError(f"model manifest {path}: {key} is listed twice. Refusing to start.")
        entries[key] = Entry(key, str(m["registered_model"]), str(m["version"]), str(m["alias"]), str(m["run_id"]),
                             str(m["sha256"]).lower(), int(m["size_bytes"]), str(m.get("loaded_by", "")))
    return Manifest(path, str(raw.get("product", "")), raw.get("site"), str(raw.get("generated_at", "")),
                    str(raw.get("tracking_uri", "")), entries)


def read_nvinfer_files(config: str | os.PathLike, root: str | os.PathLike) -> list[tuple[str, str]]:
    """(manifest key, key name) of the files an nvinfer config makes the GIE read. Relative values are relative to
    the config's own directory, as nvinfer resolves them."""
    real = resolve(manifest_key(config, root), root)
    out = []
    with open(real, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            k, v = (s.strip() for s in line.split("=", 1))
            if k in NVINFER_FILE_KEYS and v:
                out.append((manifest_key(v, root, base=real.parent), k))
    return out


def read_tracker_files(config: str | os.PathLike, root: str | os.PathLike,
                       even_if_off: bool = False) -> list[tuple[str, str]]:
    """The ReID model an NvDCF tracker config loads: its onnxFile, only when reidType is not 0 (0 = no ReID net),
    or whenever it is named, with `even_if_off` (a manifest covers it so it can be switched on)."""
    real = resolve(manifest_key(config, root), root)
    text = real.read_text(encoding="utf-8")
    reid = _TRACKER_REID_TYPE.search(text)
    onnx = _TRACKER_ONNX.search(text)
    if not onnx or (not even_if_off and (not reid or int(reid.group(1)) == 0)):
        return []
    return [(manifest_key(onnx.group(1), root, base=real.parent), "ReID.onnxFile")]


def expand(refs: Iterable[ModelRef], root: str | os.PathLike) -> tuple[dict[str, str], list[str]]:
    """({manifest key: what loads it}, problems): the files `refs` load. A config that cannot be read is a problem."""
    files: dict[str, str] = {}
    problems: list[str] = []
    for ref in refs:
        try:
            if ref.kind == "file":
                files.setdefault(manifest_key(ref.path, root), ref.via)
            elif ref.kind == "nvinfer":
                for key, name in read_nvinfer_files(ref.path, root):
                    files.setdefault(key, f"{ref.via} -> {name}")
            elif ref.kind in ("tracker", "tracker-all"):
                for key, name in read_tracker_files(ref.path, root, even_if_off=ref.kind == "tracker-all"):
                    files.setdefault(key, f"{ref.via} -> {name}")
            elif ref.kind == "pack":
                d = resolve(manifest_key(ref.path, root), root)
                found = sorted(glob.glob(str(d / "*.onnx")))
                for f in found:
                    files.setdefault(manifest_key(f, root), ref.via)
                if not found:
                    files.setdefault(manifest_key(ref.path, root).rstrip("/") + "/*.onnx", ref.via)
            else:
                problems.append(f"{ref.via}: unknown model reference kind {ref.kind!r}")
        except OSError as exc:
            problems.append(f"{ref.via} names {ref.path}, which cannot be read ({exc.strerror or exc})")
    return files, problems


# -- verifying ----------------------------------------------------------------------------------------------------
def _pointer_hint(key: str, root) -> str:
    pointer = resolve(key, root).with_name(resolve(key, root).name + ".dvc")
    return f"`dvc pull {key}.dvc`" if pointer.is_file() else "(it has no DVC pointer: restore it from its release)"


def required(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(REQUIRED_ENV, "")).strip().lower() in ("1", "true", "yes", "on")


def verify(manifest_path: str | os.PathLike | None, refs: Iterable[ModelRef], root: str | os.PathLike,
           env: Mapping[str, str] | None = None, *, what: str = "this deployment") -> Verified:
    """Refuse (ManifestError) unless every file `refs` load is listed in the manifest, present, and the listed sha256.

    `manifest_path` is the deployment's `model_manifest:` setting, relative to `root` unless absolute; None/empty or a
    missing file only warns, unless MODEL_MANIFEST_REQUIRED is set."""
    root = Path(root)
    path = resolve(manifest_key(manifest_path, root), root) if manifest_path else None
    if path is None or not path.is_file():
        why = "model_manifest is not set" if path is None else f"{manifest_key(manifest_path, root)} does not exist"
        if required(env):
            raise ManifestError(f"no model manifest for {what} ({why}) and {REQUIRED_ENV} is set. Generate it with "
                                "`python -m mtmc.tracking.manifest generate` and commit it. Refusing to start.")
        warning = (f"no model manifest for {what} ({why}): model files are NOT verified. "
                   f"Set {REQUIRED_ENV}=1 to refuse instead.")
        logger.warning(warning)
        return Verified(None, warning=warning)

    manifest = load_manifest(path)
    shown = manifest_key(path, root)
    files, problems = expand(refs, root)
    for key in [k for k in files if k.endswith("/*.onnx")]:      # a pack directory that is not there at all
        prefix = key[:-len("*.onnx")]
        listed = [k for k in manifest.entries if k.startswith(prefix)]
        if listed:                                                # report its listed files as missing
            via = files.pop(key)
            for k in listed:
                files.setdefault(k, via)
    result = Verified(manifest)
    for key, via in files.items():
        entry = manifest.entries.get(key)
        if entry is None:
            problems.append(
                f"{via} loads {key}, which this manifest does not list. Every model a deployment loads must be a "
                "registered version: register and promote it, then regenerate the manifest "
                "(`python -m mtmc.tracking.manifest generate`)")
            continue
        real = resolve(key, root)
        if not real.is_file():
            problems.append(f"lists {key} ({entry.describe()}), which does not exist. Run {_pointer_hint(key, root)}")
            continue
        size = real.stat().st_size
        digest = file_sha256(real)
        result.checked.append(key)
        result.sha256[str(real.resolve())] = digest
        if digest != entry.sha256:
            problems.append(
                f"{key} is not {entry.describe()}: sha256 expected {entry.sha256[:12]}... ({entry.size_bytes} bytes), "
                f"on disk {digest[:12]}... ({size} bytes). Run `dvc checkout {key}.dvc` (or `dvc pull` it); if the "
                "manifest is what changed, its .dvc pointer must change in the same commit")
    if problems:
        raise ManifestError(f"model manifest {shown}:\n  - " + "\n  - ".join(problems) + "\nRefusing to start.")
    logger.info("model manifest %s: %d model file(s) verified (%s)", shown, len(result.checked),
                ", ".join(f"{manifest.entries[k].registered_model} v{manifest.entries[k].version}"
                          for k in result.checked))
    return result
