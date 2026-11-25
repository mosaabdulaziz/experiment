"""Every model/tracker config a pipeline settings file names exists in the repo.

A missing tracker config does not fail: nvtracker logs a warning and runs on its
built-in defaults, so a path left behind by a folder move goes unnoticed
(2026-10-06: the shared file still named sidecar/config_tracker_nvdcf.yml after
the move to perception/pipeline/). Paths are checked as the container sees them:
/workspace/iris is the repo root, and a relative path is resolved from it."""
from pathlib import Path

import pytest

from perception.engine.sources import load_sidecar_config

REPO = Path(__file__).resolve().parents[1]
FILES = [REPO / "perception" / "pipeline" / "sidecar.yaml", *sorted(REPO.glob("products/*/sidecar.yaml"))]
KEYS = ("tracker_config", "pgie_config", "sgie_face_config", "pgie_fire_config")


def _on_host(p: str) -> Path:
    return REPO / p.removeprefix("/workspace/iris/") if p.startswith("/workspace/iris/") else REPO / p


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_named_configs_exist(path):
    cfg = load_sidecar_config(str(path))
    missing = {k: cfg[k] for k in KEYS if cfg.get(k) and not _on_host(cfg[k]).is_file()}
    assert not missing, f"{path.relative_to(REPO)} names files that are not in the repo: {missing}"
