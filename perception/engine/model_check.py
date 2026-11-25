"""The sidecar's startup model check: every model file this replica is about to load, against its model manifest.

Runs in main() right after the role overrides (so a replica whose role turns the PGIE off is not asked for the PGIE's
files) and before anything is built: no Gst element, no module object, no model exists yet. The files come from the
engine's own GIE and tracker settings and from each module's `model_files(settings)`; the comparison is
model_manifest.verify (stdlib only, shared with the product services). Pure Python, importable off-GPU.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

from . import model_manifest
from .model_manifest import ModelRef
from .modules import declared_model_files

REPO_ROOT = Path(__file__).resolve().parents[2]          # perception/engine/model_check.py -> the repo root


def engine_model_refs(cfg: dict, every_variant: bool = False) -> list:
    """What the engine itself loads (the PGIE when enabled, the tracker's ReID net when on), then every module's.
    `every_variant`: also what a module's switches could load instead (see Module.model_files)."""
    refs = []
    if cfg.get("pgie_enabled", True) and cfg.get("pgie_config"):
        refs.append(ModelRef(str(cfg["pgie_config"]), "pgie_config", "nvinfer"))
    if cfg.get("tracker_config"):
        refs.append(ModelRef(str(cfg["tracker_config"]), "tracker_config", "tracker-all" if every_variant else "tracker"))
    return refs + declared_model_files(cfg.get("modules"), every_variant=every_variant)


def verify_models(cfg: dict, what: str = "this sidecar", root: Path = REPO_ROOT,
                  env: Optional[Mapping[str, str]] = None) -> model_manifest.Verified:
    """model_manifest.verify for this config: raises ManifestError (refuse) or ModuleConfigError (a bad module
    setting), warns when the config names no manifest (unless MODEL_MANIFEST_REQUIRED)."""
    return model_manifest.verify(cfg.get("model_manifest"), engine_model_refs(cfg), root, env, what=what)
