"""Every model a deployment's configs can load is covered: by its manifest when one is committed, by the import catalog
always (so it can be registered and a manifest generated). CI, no MLflow, no model files.

"Can load" is every role of the sidecar yaml and every setting switch (Module.model_files with every_variant: barber's
staff_model classifier/clip, a tracker ReID net that is off), plus the service's own models
(products/<product>/model_files.py), exactly what mtmc.tracking.manifest generates from.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("yaml")
pytest.importorskip("numpy")
pytest.importorskip("cv2")

from mtmc.tracking import import_models as im  # noqa: E402
from mtmc.tracking import manifest as gen  # noqa: E402
from perception.engine import model_manifest as mm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENTS = gen.load_deployments()
EXAMPLE = ROOT / "mtmc" / "tracking" / "examples" / "models.manifest.barber-bekky.EXAMPLE.json"
CATALOG_FILES = {Path(f).as_posix() for e in im.load_catalog(im.CATALOG) for f in e.files}


def _uncovered(files: dict[str, str], listed: set[str]) -> list[str]:
    """Loadable files not in `listed`. A model pack not on this machine (`<dir>/*.onnx`) needs at least one listed
    file under its directory, and then every one the catalog knows there."""
    out = []
    for key, via in sorted(files.items()):
        if key.endswith("/*.onnx"):
            prefix = key[:-len("*.onnx")]
            expected = {k for k in CATALOG_FILES if k.startswith(prefix)}
            if not expected or not expected <= listed:
                out.append(f"{key} ({via}): {sorted(expected - listed) or 'no file of the pack listed'}")
        elif key not in listed:
            out.append(f"{key} ({via})")
    return out


def test_deployments_cover_every_product_and_site():
    assert {(d.product, d.site) for d in DEPLOYMENTS} == {
        ("attendance", None), ("barber", "bekky"), ("barber", "palm-hills"), ("concealment_detection", None), ("fall_detection", None),
        ("multicam", None), ("proximity", None)}


@pytest.mark.parametrize("dep", DEPLOYMENTS, ids=lambda d: d.label)
def test_the_deployments_yamls_name_its_manifest(dep):
    assert gen.named_manifests(dep, ROOT) == {path: dep.manifest for path in gen.named_manifests(dep, ROOT)}
    assert dep.manifest == f"products/{dep.product}/models.manifest" + (f".{dep.site}" if dep.site else "") + ".json"


def test_every_product_sidecar_yaml_belongs_to_a_deployment():
    """Site variants (palm-hills-live, -16oct) inherit their site's manifest through `extends:`."""
    from perception.engine.sources import load_sidecar_config
    manifests = {d.manifest for d in DEPLOYMENTS}
    for y in sorted((ROOT / "products").glob("*/sidecar*.yaml")):
        assert load_sidecar_config(str(y)).get("model_manifest") in manifests, y.relative_to(ROOT)


@pytest.mark.parametrize("dep", DEPLOYMENTS, ids=lambda d: d.label)
def test_every_loadable_model_is_in_the_import_catalog(dep):
    files, problems = gen.deployment_files(dep, ROOT)
    assert problems == []
    assert _uncovered(files, CATALOG_FILES) == [], "register these (add them to mtmc/tracking/imported_models.yaml)"


@pytest.mark.parametrize("dep", DEPLOYMENTS, ids=lambda d: d.label)
def test_a_committed_manifest_lists_every_loadable_model(dep):
    path = ROOT / dep.manifest
    if not path.is_file():
        # a deployment whose models are not all registered yet has no manifest (none since buffalo_l, 2026-11-05)
        pytest.skip(f"{dep.manifest} not generated yet (a model it loads is not registered: see the baseline)")
    files, _ = gen.deployment_files(dep, ROOT)
    listed = set(mm.load_manifest(path).entries)
    assert _uncovered(files, listed) == [], f"regenerate {dep.manifest}"


def test_the_staff_switch_and_the_tracker_reid_are_covered_conditionally():
    dep = next(d for d in DEPLOYMENTS if d.site == "bekky")
    files, _ = gen.deployment_files(dep, ROOT)
    assert files["perception/models/staff/staff_dinov2_s.onnx"] == "modules.appearance.staff_classifier"
    assert files["perception/models/staff/vision_model.onnx"] == "modules.appearance.clip_model"   # staff_model: clip
    assert files["models/tracker/resnet50_market1501_aicity156.onnx"] == "tracker_config -> ReID.onnxFile"
    assert files["perception/models/kids_car/kids_car.onnx"] == "barber service kids_car_model"


def test_the_example_manifest_is_complete_for_bekky_and_clearly_not_real():
    raw = json.loads(EXAMPLE.read_text())
    assert raw["_EXAMPLE_ONLY"].startswith("NOT A REAL MANIFEST")
    assert EXAMPLE.relative_to(ROOT).as_posix() not in {d.manifest for d in DEPLOYMENTS}
    dep = next(d for d in DEPLOYMENTS if d.site == "bekky")
    files, _ = gen.deployment_files(dep, ROOT)
    assert _uncovered(files, set(mm.load_manifest(EXAMPLE).entries)) == []      # the check catches a gap...
    assert _uncovered(files, set(mm.load_manifest(EXAMPLE).entries) - {"perception/models/staff/vision_model.onnx"}) \
        == ["perception/models/staff/vision_model.onnx (modules.appearance.clip_model)"]   # ...like a missing fallback
