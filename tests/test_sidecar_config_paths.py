"""Every file the pipeline's settings point at must exist.

DeepStream does not fail on a missing tracker config: it logs "File doesn't exist.
Will go ahead with default values" and runs untuned. After `sidecar/` moved to
`perception/pipeline/`, the shared file kept the old path for weeks and the only test
that looked (barber's) checked barber's own override. So walk every settings file
the repo ships, resolved through `extends:`, and check each path it names.
"""
from pathlib import Path

import pytest
import yaml

from perception.engine.sources import load_sidecar_config

REPO = Path(__file__).resolve().parents[1]
CONTAINER_ROOT = "/workspace/iris/"

# Named in the settings but not files of the repo: the library lives in the image, the
# analytics file is written at run time, and the CLIP vision model is a download, too big
# for git, loaded only by the zero-shot staff fallback (appearance_staff_model: clip). The model manifest is generated
# from the central MLflow server's aliases and committed later (until then a missing one only warns at startup); its
# own tests are tests/test_model_manifest_completeness.py.
NOT_REPO_FILES = {"tracker_lib", "analytics_config_path", "appearance_clip_model", "model_manifest"}
FILE_SUFFIXES = (".txt", ".yml", ".yaml", ".onnx", ".json")
NVINFER_FILES = {"onnx-file", "labelfile-path"}   # inside an nvinfer config; engines are built, the parser lib is in the image


def settings_files():
    shared = REPO / "perception" / "pipeline" / "sidecar.yaml"
    products = sorted((REPO / "products").glob("*/sidecar.yaml"))
    clips = sorted((REPO / "tests" / "configs").glob("*.yaml"))
    presets = sorted((REPO / "configs" / "presets").glob("*/sidecar.yaml"))
    return [shared, *products, *clips, *presets]


def named_files(cfg, prefix=""):
    for key, value in cfg.items():
        if isinstance(value, dict):
            yield from named_files(value, f"{prefix}{key}.")
        elif isinstance(value, str) and value.endswith(FILE_SUFFIXES) and key not in NOT_REPO_FILES:
            yield f"{prefix}{key}", value


def resolve(value):
    return REPO / (value[len(CONTAINER_ROOT):] if value.startswith(CONTAINER_ROOT) else value)


def is_there(value):
    """The file, or -- for a weight versioned with DVC (DATA.md) -- its pointer: CI does not pull data, and a wrong
    path still fails because it has no pointer either."""
    path = resolve(value)
    return path.is_file() or path.with_name(path.name + ".dvc").is_file()


@pytest.mark.parametrize("path", settings_files(), ids=lambda p: str(p.relative_to(REPO)))
def test_every_file_a_settings_file_names_exists(path):
    cfg = load_sidecar_config(str(path))
    missing = [f"{key}: {value}" for key, value in named_files(cfg)
               if not is_there(value)
               # a face/plate engine is built on first start and cached beside its model
               and not value.endswith(".engine")]
    # ...and what the nvinfer configs it names point at: the model, and the class list (labels.txt was once lost from
    # git by an ignore rule while the config still named it)
    for key, value in named_files(cfg):
        conf = resolve(value)
        if value.endswith(".txt") and conf.is_file():
            for line in conf.read_text().splitlines():
                k, _, v = line.partition("=")
                if k.strip() in NVINFER_FILES and v.strip():
                    target = resolve(v.strip()) if v.strip().startswith(CONTAINER_ROOT) else conf.parent / v.strip()
                    if not (target.is_file() or target.with_name(target.name + ".dvc").is_file()):
                        missing.append(f"{value} -> {k.strip()}={v.strip()}")
    assert not missing, f"{path.relative_to(REPO)} names files that do not exist:\n  " + "\n  ".join(missing)


def test_the_app_container_can_read_the_clip_overlays():
    """RUNNING.md tells you to run clips with SIDECAR_CONFIG=tests/configs/<file>. The app
    reads that same file (camera roles, fire incidents), so it must be mounted there."""
    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text())
    mounts = " ".join(compose["services"]["iris"]["volumes"])
    assert "/app/tests/configs" in mounts
