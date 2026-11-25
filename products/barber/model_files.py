"""The model files the barber service loads itself, and the startup check of them against the site's model manifest.

The sidecar's models (segmentation, faces, staff classifier, re-ID) are declared by its modules and checked by the
engine; this service loads one more, the kids car detector. Its site yaml names the same manifest as the site's
sidecar yaml (`model_manifest:`), and at startup the file must be the registered version that manifest lists.

The verifier is perception/engine/model_manifest.py (stdlib only), loaded by path: this service runs in its own venv
without the repo on sys.path. Standard library only here too, so the manifest generator and the CI completeness test
can import this file the same way.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Mapping

KIDS_CAR_DEFAULT = "perception/models/kids_car/kids_car.onnx"    # repo-relative, as config.py resolves it
VERIFIER = "perception/engine/model_manifest.py"


def service_model_files(raw: Mapping) -> list[tuple[str, str]]:
    """(repo-relative path, the setting that names it) for every model file this service loads with `raw` (its
    merged site yaml)."""
    return [(str(raw.get("kids_car_model") or KIDS_CAR_DEFAULT), "barber service kids_car_model")]


def deployment_model_files(config_path: str | Path) -> list[tuple[str, str]]:
    """service_model_files for a site yaml, `extends:` resolved (top-level keys: all this needs). For the manifest
    generator and the CI completeness test (pyyaml, which they have)."""
    import yaml                                                     # noqa: PLC0415

    def read(path: Path) -> dict:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        base = raw.pop("extends", None)
        return {**read(path.parent / base), **raw} if base else raw
    return service_model_files(read(Path(config_path)))


def load_verifier(repo_dir: Path):
    name = "iris_model_manifest"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(repo_dir) / VERIFIER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_service_models(raw: Mapping, repo_dir: Path, env: Mapping[str, str] | None = None):
    """Refuse (the verifier's ManifestError) unless the kids car model is the registered version the site's manifest
    lists; no manifest only warns (unless MODEL_MANIFEST_REQUIRED)."""
    mm = load_verifier(repo_dir)
    refs = [mm.ModelRef(path, via) for path, via in service_model_files(raw)]
    return mm.verify(raw.get("model_manifest"), refs, repo_dir, env, what="the barber service")
