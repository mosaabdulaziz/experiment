"""mtmc.tracking.import_models: catalog, record parsers, idempotency, --force, --dry-run, skips.

Runs go to a temp sqlite store (no server). Needs pyyaml for the catalog and mlflow for the import tests; each test
skips without what it needs.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from mtmc.tracking import import_models as im  # noqa: E402

D2_CAPABILITIES = {
    "vehicle-detector", "plate-detector", "plate-ocr", "vehicle-reid", "staff-classifier", "staff-clip",
    "person-reid", "tracker-reid", "person-segmenter", "kids-car-detector", "object-detector", "fire-detector",
    "face-detector", "face-recognizer", "pose-estimator", "fall-classifier", "depth-estimator",
}

RESULTS_CSV = (
    "                  epoch,         time,  train/box_loss,  metrics/mAP50-95(B),  lr/pg0\n"
    "                      1,       68.759,         0.93281,              0.84806,  0.0016\n"
    "                      2,      126.459,         0.62784,              0.82779,  0.0033\n"
    "                      3,      180.100,         0.51000,              0.86000,  0.0040\n"
)
REPORT_MD = """# v5_arabic_mobile: training performance through epoch 3

GPUs: Tesla T4, Tesla T4 | batch per GPU: 8 | samples: train=2720, validation=340, test=164

| Epoch | Steps | Train loss | LR (last) | Val acc | Best acc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 240 | 86.4272 | 0.00005000 | 0.0000 | 0.0000 |
| 2 | 240 | 48.2251 | 0.00005000 | — | 0.0000 |
| 3 | 240 | 32.0896 | 0.00004900 | 0.1059 | 0.1059 |
"""


def _git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def repo(tmp_path) -> Path:
    r = tmp_path / "repo"
    (r / "models" / "vehicle").mkdir(parents=True)
    (r / "runs" / "v3").mkdir(parents=True)
    (r / "staff").mkdir()
    (r / "ocr").mkdir()
    (r / "models" / "vehicle" / "best.onnx").write_bytes(b"vehicle-onnx")
    (r / "models" / "vehicle" / "labels.txt").write_text("vehicle\n")
    # 120 args: more than MLflow takes in one batch (100), as the real args.yaml (114 keys)
    args = {"model": "yolo11s.pt", "epochs": 150, "batch": 48, "lr0": 0.005, "data": "x/data.yaml", "freeze": None,
            **{f"extra_{i}": i for i in range(115)}}
    (r / "runs" / "v3" / "args.yaml").write_text(yaml.safe_dump(args, sort_keys=False))
    (r / "runs" / "v3" / "results.csv").write_text(RESULTS_CSV)
    staff = b"staff-onnx"
    (r / "staff" / "staff.onnx").write_bytes(staff)
    (r / "staff" / "staff.json").write_text(json.dumps({
        "input_hw": [252, 126], "temperature": 0.8359, "threshold": 0.79, "timm_id": "vit_small_patch14_dinov2",
        "val_points": [{"threshold": 0.5, "staff_recall": 0.91, "customers_as_staff": 0.03}],
        "palm_test": {"auroc": 0.99, "at_threshold": {"recall": 0.9, "fpr": 0.02}},
        "source_run": "staff_palm/B_s1", "sha256": _sha(staff)}))
    (r / "ocr" / "plate_rec.onnx").write_bytes(b"ocr-onnx")
    (r / "ocr" / "effective.yml").write_text(yaml.safe_dump(
        {"Global": {"epoch_num": 60, "seed": 42}, "Optimizer": {"name": "Adam", "lr": {"learning_rate": 5e-5}}}))
    (r / "ocr" / "training-report.md").write_text(REPORT_MD)
    (r / "datasets").mkdir()
    (r / "datasets" / "vehicle_detection_1cls.dvc").write_text(
        "outs:\n- md5: ad5ed68445f31c338d2c5fdf8c8f4283.dir\n  size: 1\n  nfiles: 1\n  hash: md5\n"
        "  path: vehicle_detection_1cls\n")
    _git(r, "init", "-q", "-b", "main")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "fixture")
    return r


CATALOG = {
    "entries": [
        {"id": "vehicle-detector/v3", "capability": "vehicle-detector", "source": "imported",
         "files": ["models/vehicle/best.onnx", "models/vehicle/labels.txt"],
         "records": [{"kind": "ultralytics_args", "file": "runs/v3/args.yaml"},
                     {"kind": "ultralytics_results", "file": "runs/v3/results.csv"}],
         "data": {"vehicle_detection_1cls": "datasets/vehicle_detection_1cls.dvc"},
         "provenance": {"origin": "in-house run v3", "gaps": ["ultralytics version"]},
         "evidence": ["DATA.md"]},
        {"id": "staff-classifier/r3", "capability": "staff-classifier", "source": "imported",
         "files": ["staff/staff.onnx", "staff/staff.json"],
         "records": [{"kind": "staff_json", "file": "staff/staff.json"}],
         "provenance": {"origin": "in-house trainer, not in repo"}},
        {"id": "plate-ocr/v5", "capability": "plate-ocr", "source": "kaggle", "files": ["ocr/plate_rec.onnx"],
         "records": [{"kind": "paddle_config", "file": "ocr/effective.yml"},
                     {"kind": "ocr_training_report", "file": "ocr/training-report.md"}],
         "provenance": {"origin": "PaddleOCR on Kaggle", "notebook": "nb/PP-OCRv5.ipynb"}},
        {"id": "pose-estimator/missing", "capability": "pose-estimator", "source": "imported",
         "files": ["models/pose/yolo26x-pose.onnx"], "provenance": "unknown"},
    ]
}


@pytest.fixture
def catalog(tmp_path) -> Path:
    p = tmp_path / "catalog.yaml"
    p.write_text(yaml.safe_dump(CATALOG, sort_keys=False))
    return p


@pytest.fixture
def mlflow_env(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("IRIS_MLFLOW_OFFLINE_DIR", str(tmp_path / "offline"))
    for k in ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME", "MLFLOW_EXPERIMENT_ID", "MLFLOW_RUN_ID"):
        monkeypatch.delenv(k, raising=False)
    yield mlflow
    while mlflow.active_run():
        mlflow.end_run()


@pytest.fixture
def store(mlflow_store) -> str:
    return mlflow_store                 # tests/conftest.py: this test's own copy of a migrated store


def _run(catalog, repo, store, *extra) -> int:
    return im.main(["--catalog", str(catalog), "--root", str(repo), "--tracking-uri", store, *extra])


def _statuses(out: str) -> dict[str, str]:
    """entry id -> status, from the result table (lines `<status> <entry id> <run id> <detail>`)."""
    rows = (line.split() for line in out.splitlines())
    return {r[1]: r[0] for r in rows if len(r) >= 3 and "/" in r[1] and not r[0].endswith(",")}


def _imported(store):
    from mlflow.tracking import MlflowClient
    c = MlflowClient(tracking_uri=store)
    exp = c.get_experiment_by_name(im.EXPERIMENT)
    return c, (c.search_runs([exp.experiment_id]) if exp else [])


# -- catalog ------------------------------------------------------------------------------------------------------
def test_catalog_parses_and_validates(catalog, tmp_path):
    entries = im.load_catalog(catalog)
    assert [e.id for e in entries] == ["vehicle-detector/v3", "staff-classifier/r3", "plate-ocr/v5",
                                       "pose-estimator/missing"]
    assert [e.provenance_status for e in entries] == ["partial", "known", "known", "unknown"]
    for bad, msg in (({"source": "github"}, "source must be"), ({"provenance": "maybe"}, "provenance must be"),
                     ({"records": [{"kind": "nope", "file": "x"}]}, "needs kind"), ({"colour": "red"}, "unknown keys"),
                     ({"files": []}, "missing")):
        e = {**CATALOG["entries"][0], **bad}
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump({"entries": [e]}))
        with pytest.raises(im.CatalogError, match=msg):
            im.load_catalog(p)
    p = tmp_path / "dup.yaml"
    p.write_text(yaml.safe_dump({"entries": [CATALOG["entries"][0]] * 2}))
    with pytest.raises(im.CatalogError, match="duplicate"):
        im.load_catalog(p)


def test_the_committed_catalog_covers_d2():
    entries = im.load_catalog(im.CATALOG)
    assert {e.capability for e in entries} == D2_CAPABILITIES
    assert {e.capability for e in entries if e.source == "kaggle"} == {"plate-ocr"}
    for e in entries:                                  # every claim is either sourced or marked unknown
        assert e.provenance == "unknown" or e.evidence, e.id


# -- parsers ------------------------------------------------------------------------------------------------------
def test_parse_ultralytics_args_and_results(repo):
    params, steps = im.parse_ultralytics_args(repo / "runs" / "v3" / "args.yaml")
    assert params["epochs"] == "150" and params["batch"] == "48" and params["freeze"] == "None"
    assert len(params) == 121 and steps == {}
    params, steps = im.parse_ultralytics_results(repo / "runs" / "v3" / "results.csv")
    assert params == {} and sorted(steps) == [0, 1, 2]                 # epoch 1..3 -> step 0..2 (Ultralytics' MLflow)
    assert steps[0] == {"time": 68.759, "train/box_loss": 0.93281, "metrics/mAP50-95B": 0.84806, "lr/pg0": 0.0016}


def test_parse_ocr_report_and_paddle_config(repo):
    params, steps = im.parse_ocr_training_report(repo / "ocr" / "training-report.md")
    assert params == {"report.gpus": "Tesla T4, Tesla T4", "report.batch_per_gpu": "8",
                      "report.samples": "train=2720, validation=340, test=164"}
    assert sorted(steps) == [0, 1, 2]                                  # Epoch 1..3 -> step 0..2
    assert steps[0] == {"steps": 240.0, "train_loss": 86.4272, "lr_last": 5e-05, "val_acc": 0.0, "best_acc": 0.0}
    assert "val_acc" not in steps[1]                   # `—`: no validation that epoch (epoch 2)
    params, _ = im.parse_paddle_config(repo / "ocr" / "effective.yml")
    assert params == {"Global.epoch_num": "60", "Global.seed": "42", "Optimizer.name": "Adam",
                      "Optimizer.lr.learning_rate": "5e-05"}


def test_parse_staff_json(repo):
    params, steps = im.parse_staff_json(repo / "staff" / "staff.json")
    assert params["temperature"] == "0.8359" and params["input_hw"] == "[252, 126]"
    assert params["source_run"] == "staff_palm/B_s1"
    assert steps == {None: {"val/staff_recall_at_0.5": 0.91, "val/customers_as_staff_at_0.5": 0.03,
                            "palm_test/auroc": 0.99, "palm_test/at_threshold/recall": 0.9,
                            "palm_test/at_threshold/fpr": 0.02}}


# -- import -------------------------------------------------------------------------------------------------------
def test_import_creates_runs_sidecars_and_skips_missing(mlflow_env, catalog, repo, store, capsys):
    assert _run(catalog, repo, store) == 0
    out = capsys.readouterr().out
    assert "skipped       pose-estimator/missing" in out and "not present locally" in out
    c, runs = _imported(store)
    by_entry = {r.data.tags["import.catalog_entry"]: r for r in runs}
    assert set(by_entry) == {"vehicle-detector/v3", "staff-classifier/r3", "plate-ocr/v5"}

    v = by_entry["vehicle-detector/v3"]
    assert v.info.status == "FINISHED" and v.info.run_name == "vehicle-detector/v3"
    t = v.data.tags
    assert t["source"] == "imported" and t["origin"] == "in-house run v3" and t["provenance.status"] == "partial"
    assert t["provenance.gaps"] == "ultralytics version" and t["provenance.evidence"] == "DATA.md"
    assert t["model.sha256"] == _sha(b"vehicle-onnx")
    assert t["import.key"] == f"vehicle-detector:{_sha(b'vehicle-onnx')}"
    assert t["git.commit"] == _git(repo, "rev-parse", "HEAD")            # lineage of the import itself
    assert t["data.vehicle_detection_1cls.dvc_md5"] == "ad5ed68445f31c338d2c5fdf8c8f4283.dir"
    assert v.data.params["batch"] == "48" and len(v.data.params) == 121  # > 100: batched
    hist = c.get_metric_history(v.info.run_id, "metrics/mAP50-95B")
    assert [(m.step, m.value) for m in hist] == [(0, 0.84806), (1, 0.82779), (2, 0.86)]
    assert t["import.step_base"] == "0"
    assert sorted(a.path for a in c.list_artifacts(v.info.run_id, "records")) == [
        "records/args.yaml", "records/results.csv"]
    assert sorted(a.path for a in c.list_artifacts(v.info.run_id, "model")) == [
        "model/best.onnx", "model/labels.txt"]

    side = json.loads((repo / "models" / "vehicle" / "best.onnx.mlflow.json").read_text())
    assert side["run_id"] == v.info.run_id and side["sha256"] == _sha(b"vehicle-onnx")
    assert side["experiment"] == im.EXPERIMENT and side["registered_model"] is None
    assert (repo / "models" / "vehicle" / "labels.txt.mlflow.json").is_file()

    o = by_entry["plate-ocr/v5"]
    assert o.data.tags["source"] == "kaggle" and o.data.tags["notebook"] == "nb/PP-OCRv5.ipynb"
    assert o.data.params["Global.epoch_num"] == "60" and o.data.params["report.batch_per_gpu"] == "8"
    assert [m.step for m in c.get_metric_history(o.info.run_id, "train_loss")] == [0, 1, 2]
    assert o.data.tags["import.step_base"] == "0"
    s = by_entry["staff-classifier/r3"]
    assert s.data.metrics["palm_test/auroc"] == 0.99 and s.data.params["threshold"] == "0.79"


def test_second_run_creates_nothing(mlflow_env, catalog, repo, store, capsys):
    assert _run(catalog, repo, store) == 0
    _, first = _imported(store)
    capsys.readouterr()
    assert _run(catalog, repo, store) == 0
    out = capsys.readouterr().out
    _, second = _imported(store)
    assert len(second) == len(first) == 3
    assert _statuses(out) == {"vehicle-detector/v3": "exists", "staff-classifier/r3": "exists",
                              "plate-ocr/v5": "exists", "pose-estimator/missing": "skipped"}


def test_force_reimports_and_marks_the_old_run(mlflow_env, catalog, repo, store):
    assert _run(catalog, repo, store, "--only", "plate-ocr/v5") == 0
    _, (old,) = _imported(store)
    assert _run(catalog, repo, store, "--only", "plate-ocr/v5", "--force") == 0
    c, runs = _imported(store)
    assert len(runs) == 2
    new = next(r for r in runs if r.info.run_id != old.info.run_id)
    assert new.data.tags["import.forced"] == "true"
    assert c.get_run(old.info.run_id).data.tags["import.superseded_by"] == new.info.run_id
    side = json.loads((repo / "ocr" / "plate_rec.onnx.mlflow.json").read_text())
    assert side["run_id"] == new.info.run_id                            # the sidecar follows the newest import


@pytest.mark.mlflow_store_absent
def test_dry_run_touches_nothing(mlflow_env, catalog, repo, store, tmp_path, capsys):
    db = Path(store.removeprefix("sqlite:///"))
    assert _run(catalog, repo, store, "--dry-run") == 0
    out = capsys.readouterr().out
    assert _statuses(out) == {"vehicle-detector/v3": "would-import", "staff-classifier/r3": "would-import",
                              "plate-ocr/v5": "would-import", "pose-estimator/missing": "skipped"}
    assert not db.exists() and not (tmp_path / "offline").exists()
    assert not list(repo.rglob("*.mlflow.json"))
    assert _git(repo, "status", "--porcelain") == ""


def test_pinned_sha256_mismatch_is_skipped(mlflow_env, catalog, repo, store, capsys):
    cat = yaml.safe_load(catalog.read_text())
    cat["entries"][2]["expected_sha256"] = {"ocr/plate_rec.onnx": "0" * 64}
    (repo / "staff" / "staff.onnx").write_bytes(b"not the model the json describes")
    catalog.write_text(yaml.safe_dump(cat))
    assert _run(catalog, repo, store) == 0
    out = capsys.readouterr().out
    assert "does not match the pinned" in out and "does not match the one its staff/staff.json declares" in out
    _, runs = _imported(store)
    assert {r.data.tags["import.catalog_entry"] for r in runs} == {"vehicle-detector/v3"}


def test_hidden_sidecars_reported(catalog, repo):
    (repo / ".gitignore").write_text("models/vehicle/**\n")
    hidden = im.hidden_sidecars(im.load_catalog(catalog), repo)
    assert hidden == ["models/vehicle/best.onnx.mlflow.json", "models/vehicle/labels.txt.mlflow.json"]
