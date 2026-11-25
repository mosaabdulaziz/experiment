"""The model files the fall detection service loads itself, and the startup check of them against its model manifest.

The sidecar's models (the shared PGIE, pose) are declared by its modules and checked by the engine; this service loads
the classifier's two checkpoints. fall_detection.yaml names the same manifest as the product's sidecar yaml
(`model_manifest:`), and boot checks the checkpoints against it before the vendor's own SHA-256 registry check, which
stays.

The verifier is perception/engine/model_manifest.py (stdlib only), loaded by path: this service runs in its own venv
without the repo on sys.path. Standard library only here (pyyaml when reading config.yaml for the default streams),
and no torch: the checkpoint names are read from the vendored tsstg_model.py's STREAM_FILE_MAP with `ast`, not by
importing it, so the manifest generator and the CI completeness test can use this file too.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Iterable, Mapping

SERVICE_DIR = Path(__file__).resolve().parent
REPO_DIR = SERVICE_DIR.parents[1]
VARIANT = "mmaction2"                       # boot.VARIANT
VERIFIER = "perception/engine/model_manifest.py"


def stream_file_map(variant: str = VARIANT) -> dict[str, str]:
    """stream -> checkpoint file name, from the vendored copy's own STREAM_FILE_MAP (read, not imported)."""
    tree = ast.parse((SERVICE_DIR / "fall_detection" / "tsstg_model.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "STREAM_FILE_MAP" for t in node.targets):
            return dict(ast.literal_eval(node.value)[variant])
    raise LookupError("STREAM_FILE_MAP not found in fall_detection/tsstg_model.py")


def default_streams() -> tuple[str, ...]:
    """config.yaml's detector.streams, which boot.DEFAULT_STREAMS reads through the vendored fall_monitor.CFG."""
    import yaml                                                     # noqa: PLC0415 (in the service's venv)
    cfg = yaml.safe_load((SERVICE_DIR / "config.yaml").read_text(encoding="utf-8")) or {}
    return tuple(cfg["detector"]["streams"])


def service_model_files(classifier: Mapping) -> list[tuple[str, str]]:
    """(path, the setting that names it) for the checkpoints the configured streams load, as boot resolves them.
    `classifier` is fall_detection.yaml's classifier block (weights_dir relative to this folder)."""
    weights_dir = Path(str(classifier.get("weights_dir") or "weights"))
    weights_dir = weights_dir if weights_dir.is_absolute() else SERVICE_DIR / weights_dir
    names = stream_file_map()
    return [(str(weights_dir / names[s]), f"fall service classifier.streams[{s}]")
            for s in tuple(classifier.get("streams") or default_streams())]


def deployment_model_files(config_path: str | Path) -> list[tuple[str, str]]:
    """service_model_files for fall_detection.yaml (or another config with its shape). For the manifest generator and
    the CI completeness test."""
    import yaml                                                     # noqa: PLC0415
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    return service_model_files(raw.get("classifier") or {})


def load_verifier(repo_dir: Path = REPO_DIR):
    name = "iris_model_manifest"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(repo_dir) / VERIFIER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_checkpoints(paths: Iterable[Path], model_manifest: str | None, env: Mapping[str, str] | None = None):
    """Refuse (the verifier's ManifestError) unless every checkpoint is the registered version the manifest lists; no
    manifest only warns (unless MODEL_MANIFEST_REQUIRED)."""
    mm = load_verifier()
    refs = [mm.ModelRef(str(p), f"fall service checkpoint {Path(p).name}") for p in paths]
    return mm.verify(model_manifest, refs, REPO_DIR, env, what="the fall detection service")
