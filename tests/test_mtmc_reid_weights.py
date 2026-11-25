"""The vehicle Re-ID weights load from the repo's DVC file, and a missing file refuses startup (no torch hub download).

Until 2026-11 the loader looked one directory too deep (perception/modules/mtmc/models/...), never found the file and
silently downloaded the same weights from torch hub. The file is byte-identical to that download (sha256 11ad3fa6...,
the prefix in its name; md5 and size equal to its .dvc pointer's).
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("numpy")

from perception.modules.mtmc.core import _reid  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_weights_path_is_the_dvc_tracked_file():
    assert _reid.REID_WEIGHTS == ROOT / "mtmc" / "models" / "reid" / "resnet50-11ad3fa6.pth"
    pointer = ROOT / "mtmc" / "models" / "reid" / "resnet50-11ad3fa6.pth.dvc"
    text = pointer.read_text()
    assert "md5: 012571d812f34f8442473d8b827077b5" in text and "size: 102540417" in text   # = torch hub's file


def test_missing_weights_refuse_with_the_dvc_command(monkeypatch, tmp_path):
    monkeypatch.setattr(_reid, "REID_WEIGHTS", tmp_path / "absent.pth")
    engine = _reid.ReIDEngine(cfg=None)
    with pytest.raises(_reid.ReIDWeightsMissing, match=r"dvc pull mtmc/models/reid/resnet50-11ad3fa6\.pth\.dvc"):
        engine.ensure_loaded()
    with pytest.raises(_reid.ReIDWeightsMissing):                  # not retried into a silent "unavailable"
        engine.ensure_loaded()


def test_manager_does_not_swallow_missing_weights(monkeypatch, tmp_path):
    from perception.modules.mtmc.core import MTMCConfig, MTMCManager
    monkeypatch.setattr(_reid, "REID_WEIGHTS", tmp_path / "absent.pth")
    with pytest.raises(_reid.ReIDWeightsMissing):
        MTMCManager(MTMCConfig(enabled=True, reid_enabled=True))


def test_no_torch_hub_fallback_in_the_loader():
    src = Path(_reid.__file__).read_text()
    assert "ResNet50_Weights" not in src and "weights=None" in src
