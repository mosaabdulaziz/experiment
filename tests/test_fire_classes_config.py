"""sidecar.yaml's relevant_classes must list every class the fire model emits.

The probe drops any label not in relevant_classes (sidecar.py's `label in
ctx.g.relevant`), and the comparison is case-sensitive -- so a case or spelling
drift between the active labels file and sidecar.yaml makes fire detections
disappear with no error anywhere. Neither file can be imported off-GPU; comparing
them is the part that can be tested, and it is the part that actually drifts.

The labels file is resolved from the fire module's own `config` setting ->
labelfile-path=, not hardcoded to labels.txt: perception/models/DeepStream-Yolo-Fire/ has held more
than one model+labels pair since the fire_smoke_box swap (see its README's
"escape hatch" section), and hardcoding the old filename here would silently stop
checking the file that is actually loaded the moment sidecar.yaml points
elsewhere."""

import pathlib

import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SIDECAR_YAML = _ROOT / "perception" / "pipeline" / "sidecar.yaml"


def _active_gie_config():
    # The module's `config` setting is an in-container path (/workspace/iris/...); the repo root here IS the
    # /workspace/iris mount, so drop that prefix and resolve the rest against _ROOT. The product that runs fire
    # may override it; the module's default is what the shipped products use.
    from perception.modules.fire_alerts import Settings
    rel = Settings().config.removeprefix("/workspace/iris/")
    return _ROOT / rel


def _labels_path():
    gie_config = _active_gie_config()
    for line in gie_config.read_text().splitlines():
        line = line.strip()
        if line.startswith("labelfile-path="):
            return gie_config.parent / line.split("=", 1)[1].strip()
    raise AssertionError(f"{gie_config} has no labelfile-path=")


def _fire_labels():
    return [ln.strip() for ln in _labels_path().read_text().splitlines() if ln.strip()]


def _relevant_classes():
    cfg = yaml.safe_load(_SIDECAR_YAML.read_text())
    return set(cfg.get("relevant_classes") or [])


def test_labels_file_has_exactly_two_classes():
    assert len(_fire_labels()) == 2


def test_every_fire_label_is_in_relevant_classes():
    missing = [c for c in _fire_labels() if c not in _relevant_classes()]
    assert not missing, (
        f"{missing} are emitted by the fire model but absent from "
        "sidecar.yaml's relevant_classes -- the probe would drop them silently")


def test_hazard_classes_match_the_labels_file():
    from perception.modules.fire_alerts import HAZARD_CLASSES
    assert set(HAZARD_CLASSES) == set(_fire_labels()), (
        "the fire module would not hazard-colour a class the model actually emits")
