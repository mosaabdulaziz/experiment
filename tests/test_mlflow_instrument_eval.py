"""Evaluations logged into the evaluated model's run (sub-task 6 of the MLflow story; spike D5, D9, Q9).

Temp sqlite store, fixture data, no torch, no models: the scripts' heavy parts (Ultralytics, cv2, onnxruntime, the
AICity22 eval) are replaced by fakes, everything about tracking is real. The fake Ultralytics is the one of
tests/test_mlflow_instrument_training.py.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest

mlflow = pytest.importorskip("mlflow")

from mtmc import tracking                                   # noqa: E402
from mtmc.tracking import eval as tev                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_mlflow_instrument_training as T                 # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MD5_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.dir"
MD5_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.dir"


# -- fixtures -----------------------------------------------------------------------------------------------------
@pytest.fixture
def env(tmp_path, monkeypatch, mlflow_store):
    monkeypatch.setenv("IRIS_MLFLOW_OFFLINE_DIR", str(tmp_path / "offline"))
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    for k in tracking._ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    store = mlflow_store                # tests/conftest.py: this test's own copy of a migrated store
    monkeypatch.setenv("MLFLOW_TRACKING_URI", store)
    monkeypatch.chdir(tmp_path)
    yield types.SimpleNamespace(store=store, db=tmp_path / "store" / "mlflow.db", offline=tmp_path / "offline",
                                root=tmp_path)
    while mlflow.active_run():
        mlflow.end_run()


def _pointer(path: Path, md5: str) -> Path:
    p = path.with_name(path.name + ".dvc")
    p.write_text(f"outs:\n- md5: {md5}\n  size: 1\n  hash: md5\n  path: {path.name}\n")
    return p


def _dataset(root: Path, name="golden", md5=MD5_A) -> Path:
    d = root / "data" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "data.yaml").write_text("nc: 1\n")
    if md5:
        _pointer(d, md5)
    return d


def _model_run(env, name="model.pt", status="FINISHED", experiment="vehicle-detector") -> tuple[Path, str]:
    """A finished run that logged `name` with log_model_file (sidecar written)."""
    m = env.root / name
    m.write_bytes(b"weights:" + name.encode())
    with pytest.raises(RuntimeError) if status == "FAILED" else _null():
        with tracking.tracked_run(experiment, "train", tracking_uri=env.store) as run:
            tracking.log_model_file(m, "weights", run=run)
            if status == "FAILED":
                raise RuntimeError("training failed")
    return m, run.run_id


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _args(**kw):
    base = dict(eval_set="golden", run_id=None, force=False, no_track=False)
    return argparse.Namespace(**{**base, **kw})


def _client(env):
    from mlflow.tracking import MlflowClient
    return MlflowClient(tracking_uri=env.store)


def _evaluate(env, model, data, metrics=None, artifacts=(), **kw):
    target = tev.prepare_evaluation(_args(**kw), model, data, "tests.fake_eval", root=env.root)
    target.log(metrics or {"recall": 0.9, "precision": 0.8}, artifacts=artifacts)
    return target


# -- into the model's run -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("status", ["FINISHED", "FAILED"])
def test_metrics_tags_and_report_land_in_the_models_run_and_status_is_kept(env, status):
    model, rid = _model_run(env, status=status)
    data = _dataset(env.root)
    before = _client(env).get_run(rid).info
    report = env.root / "report.json"
    report.write_text("{}")
    target = _evaluate(env, model, data, artifacts=[report])
    assert target.logged_run_id == rid
    r = _client(env).get_run(rid)
    assert (r.info.status, r.info.end_time) == (before.status, before.end_time) == (status, before.end_time)
    assert r.data.metrics["eval/golden/recall"] == pytest.approx(0.9)
    assert r.data.metrics["eval/golden/precision"] == pytest.approx(0.8)
    t = r.data.tags
    assert t["eval.golden.dvc_md5"] == MD5_A and t["eval.golden.versioned"] == "true"
    assert t["eval.golden.path"] == "data/golden"
    assert len(t["eval.golden.git_commit"]) == 40 or t["eval.golden.git_commit"] == "unknown"
    assert t["eval.golden.evaluated_at"].endswith("+00:00") and t["eval.golden.script"] == "tests.fake_eval"
    arts = [a.path for a in _client(env).list_artifacts(rid, "eval/golden")]
    assert arts == ["eval/golden/report.json"]
    assert len(_client(env).search_runs([r.info.experiment_id])) == 1           # no new run


def test_run_id_override_without_a_sidecar(env):
    _, rid = _model_run(env)
    bare = env.root / "exported.onnx"
    bare.write_bytes(b"no sidecar")
    _evaluate(env, bare, _dataset(env.root), run_id=rid)
    assert "eval/golden/recall" in _client(env).get_run(rid).data.metrics


def test_no_sidecar_refuses_with_the_way_to_get_a_run(env):
    bare = env.root / "third_party.onnx"
    bare.write_bytes(b"x")
    with pytest.raises(tev.EvalRefused, match="has no third_party.onnx.mlflow.json.*import_models.*--run-id"):
        tev.prepare_evaluation(_args(), bare, _dataset(env.root), "t", root=env.root)


def test_sha256_mismatch_refuses(env):
    model, _ = _model_run(env)
    model.write_bytes(b"retrained, not the logged bytes")
    with pytest.raises(tev.EvalRefused, match="does not match its model.pt.mlflow.json"):
        tev.prepare_evaluation(_args(), model, _dataset(env.root), "t", root=env.root)


def test_run_missing_from_the_store_refuses(env):
    model, _ = _model_run(env)
    with pytest.raises(tev.EvalRefused, match="is not in"):
        tev.prepare_evaluation(_args(run_id="0" * 32), model, _dataset(env.root), "t", root=env.root)


def test_bad_set_name_refuses(env):
    model, _ = _model_run(env)
    with pytest.raises(tev.EvalRefused, match="letters, digits"):
        tev.prepare_evaluation(_args(eval_set="plate/test:v1"), model, _dataset(env.root), "t", root=env.root)


# -- data versions ------------------------------------------------------------------------------------------------
def test_same_version_again_is_fine(env):
    model, rid = _model_run(env)
    data = _dataset(env.root)
    _evaluate(env, model, data, {"recall": 0.9})
    _evaluate(env, model, data, {"recall": 0.95})
    hist = _client(env).get_metric_history(rid, "eval/golden/recall")
    assert sorted(h.value for h in hist) == pytest.approx([0.9, 0.95])
    assert _client(env).get_run(rid).data.metrics["eval/golden/recall"] == pytest.approx(0.95)


def test_other_version_on_the_same_set_refuses_and_force_records_the_old_one(env):
    model, rid = _model_run(env)
    data = _dataset(env.root)
    _evaluate(env, model, data, {"recall": 0.9})
    _pointer(data, MD5_B)                                                    # the set changed
    with pytest.raises(tev.EvalRefused, match=f"data version {MD5_A}.*this data is {MD5_B}.*--force"):
        tev.prepare_evaluation(_args(), model, data, "t", root=env.root)
    assert _client(env).get_run(rid).data.tags["eval.golden.dvc_md5"] == MD5_A   # nothing written
    _evaluate(env, model, data, {"recall": 0.7}, force=True)
    t = _client(env).get_run(rid).data.tags
    assert t["eval.golden.dvc_md5"] == MD5_B and t["eval.golden.previous_dvc_md5"] == MD5_A
    # another name for the other version is the non-destructive way
    _pointer(data, MD5_A)
    _evaluate(env, model, data, {"recall": 0.9}, eval_set="golden-v1")
    assert _client(env).get_run(rid).data.tags["eval.golden-v1.dvc_md5"] == MD5_A


def test_unversioned_data_is_tagged_and_warned(env, caplog):
    model, rid = _model_run(env)
    gt = env.root / "ground_truth.txt"
    gt.write_text("1 2 3\n")
    with caplog.at_level(logging.WARNING, logger="mtmc.tracking"):
        _evaluate(env, model, gt)
    assert "UNVERSIONED" in caplog.text
    t = _client(env).get_run(rid).data.tags
    assert t["eval.golden.dvc_md5"] == "unversioned" and t["eval.golden.versioned"] == "false"
    assert t["eval.golden.content_sha256"] == __import__("hashlib").sha256(b"1 2 3\n").hexdigest()


def test_a_set_of_several_pointers_has_one_version_and_lists_its_members(env):
    model, rid = _model_run(env)
    clips = []
    for i, md5 in enumerate(["1" * 32, "2" * 32]):
        c = env.root / f"clip{i}.mp4"
        c.write_bytes(b"v")
        _pointer(c, md5)
        clips.append(c)
    s1 = tev.resolve_eval_set("clips", clips, env.root)
    assert s1.versioned and len(s1.dvc_md5) == 32 and len(s1.pointers) == 2
    _evaluate(env, model, clips, eval_set="clips")
    listed = _client(env).download_artifacts(rid, "eval/clips/dvc_pointers.txt", str(env.root))
    assert Path(listed).read_text() == f"clip0.mp4.dvc {'1' * 32}\nclip1.mp4.dvc {'2' * 32}\n"
    _pointer(clips[1], "3" * 32)
    assert tev.resolve_eval_set("clips", clips, env.root).dvc_md5 != s1.dvc_md5     # one member changed
    assert tev.resolve_eval_set("clips", clips[:1], env.root).dvc_md5 != s1.dvc_md5  # a subset is another version


# -- offline, no-track --------------------------------------------------------------------------------------------
def test_server_down_records_offline_and_the_evaluation_completes(env, monkeypatch):
    model, rid = _model_run(env)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"http://127.0.0.1:{T._closed_port()}")
    target = _evaluate(env, model, _dataset(env.root))
    assert target.mode == "offline" and target.logged_run_id and target.logged_run_id != rid
    from mlflow.tracking import MlflowClient
    off = MlflowClient(tracking_uri="sqlite:///" + str((env.offline / "mlflow.db").resolve()))
    (pending,) = off.search_runs([off.get_experiment_by_name("eval-pending").experiment_id])
    assert pending.data.tags["eval.target_run_id"] == rid and pending.info.status == "FINISHED"
    assert pending.data.metrics["eval/golden/recall"] == pytest.approx(0.9)
    assert "eval/golden/recall" not in _client(env).get_run(rid).data.metrics


@pytest.mark.mlflow_store_absent
def test_no_track_records_nothing_and_needs_no_run(env):
    bare = env.root / "x.onnx"
    bare.write_bytes(b"x")
    target = tev.prepare_evaluation(_args(no_track=True), bare, env.root / "nowhere", "t", root=env.root)
    assert not target.active and target.log({"recall": 1.0}) is None
    assert not env.db.exists() and not env.offline.exists()


# -- the scripts --------------------------------------------------------------------------------------------------
@pytest.fixture
def fake_cv2(monkeypatch):
    """cv2 for the scripts' imports when it is not installed: imread returns a small crop for a file that exists."""
    if importlib.util.find_spec("cv2") is not None:
        return sys.modules.setdefault("cv2", importlib.import_module("cv2"))
    cv2 = types.ModuleType("cv2")
    cv2.COLOR_BGR2RGB = cv2.COLOR_RGB2BGR = cv2.IMWRITE_JPEG_QUALITY = cv2.INTER_LINEAR = cv2.INTER_CUBIC = 0
    cv2.imread = lambda p: np.full((20, 60, 3), 128, np.uint8) if Path(p).is_file() else None
    cv2.imwrite = lambda p, img, *a: Path(p).write_bytes(b"jpg") or True
    cv2.cvtColor = lambda img, code: img
    cv2.resize = lambda img, size, **k: np.zeros((size[1], size[0], 3), np.uint8)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    return cv2


def test_vehicle_test_only_logs_into_the_weights_run(env, monkeypatch, fake_cv2):
    T._install_fake_ultralytics(monkeypatch, env.root, "simulated")
    data = _dataset(env.root, "vehicle_detection_1cls")
    model, rid = _model_run(env, "best.pt")
    v = importlib.import_module("mtmc.finetune_vehicle_detector")
    v.main(["--test-only", "--model", str(model), "--data", str(data / "data.yaml"), "--device", "cpu"])
    r = _client(env).get_run(rid)
    assert r.data.metrics["eval/vehicle_detection_1cls-val/mAP50-95"] == pytest.approx(0.61)
    assert r.data.metrics["eval/vehicle_detection_1cls-val/mAP50-95/vehicle"] == pytest.approx(0.61)
    assert r.data.tags["eval.vehicle_detection_1cls-val.dvc_md5"] == MD5_A
    assert r.data.tags["eval.vehicle_detection_1cls-val.split"] == "val"
    assert r.info.status == "FINISHED"


def test_plate_test_only_refuses_without_a_run_before_evaluating(env, monkeypatch, fake_cv2):
    T._install_fake_ultralytics(monkeypatch, env.root, "simulated")
    data = _dataset(env.root, "plate_detection")
    base = env.root / "plate_base.pt"
    base.write_bytes(b"base")
    w = env.root / "untracked.pt"
    w.write_bytes(b"no sidecar")
    p = importlib.import_module("mtmc.finetune_plate_detector")
    with pytest.raises(tev.EvalRefused):
        p.main(["--test-only", "--resume", str(w), "--model", str(base), "--data", str(data / "data.yaml")])
    assert not [e for e in T.FakeYOLO.log if e["event"] == "val"], "refused before the (slow) evaluation"
    _, rid = _model_run(env, "untracked.pt")
    p.main(["--test-only", "--resume", str(w), "--model", str(base), "--data", str(data / "data.yaml")])
    assert "eval/plate_detection-test/recall" in _client(env).get_run(rid).data.metrics


def test_audit_logs_recall_into_the_vehicle_models_run(env, monkeypatch, fake_cv2):
    T._install_fake_ultralytics(monkeypatch, env.root, "simulated")
    monkeypatch.delitem(sys.modules, "mtmc.audit_vehicle_detector", raising=False)
    monkeypatch.delitem(sys.modules, "mtmc.yolo_detector", raising=False)
    a = importlib.import_module("mtmc.audit_vehicle_detector")
    videos = env.root / "videos"
    videos.mkdir()
    for i, name in enumerate(a.MULTICAM_CLIPS[:3]):
        (videos / name).write_bytes(b"mp4")
        _pointer(videos / name, f"{i}" * 32)
    monkeypatch.setattr(a, "VIDEO_DIR", videos)
    vehicle, rid = _model_run(env, "best.onnx")
    plate = env.root / "plate.pt"
    plate.write_bytes(b"plate")

    def fake_clip(clip, *a_, **k):
        bucket = {"w<32px": {"plates": 4, "hits": 2, "recall": 0.5, "containment": 0.75},
                  "w>=128": {"plates": 6, "hits": 6, "recall": 1.0, "containment": 1.0}}
        r = {"plates": 10, "hits": 8, "recall": 0.8, "containment": 0.9, "iou_recall": 0.1, "mean_best_iou": 0.07,
             "by_bucket": bucket}
        return {"clip": clip.name, "frames_sampled": 5, "vehicle_boxes": 12, "vehicle_boxes_without_plate": 3,
                "vehicle_class_counts": {"vehicle": 12},
                "plate_anchor": {a.KEY_AS_WIRED: r, a.KEY_LOOSE: {**r, "hits": 9, "recall": 0.9}}}

    monkeypatch.setattr(a, "audit_clip", fake_clip)
    monkeypatch.setattr(a, "resolve_vehicle_class_ids", lambda m: {0: "vehicle"})
    out = env.root / "audit" / "report.json"
    argv = ["--vehicle-model", str(vehicle), "--plate-model", str(plate), "--limit-clips", "3", "--out", str(out)]
    with pytest.raises(SystemExit, match="no nvinfer config in the repo loads .*--conf"):
        a.main(argv)                                        # a model no config loads: its threshold must be given
    assert not out.exists() and "eval/el-mohandes-13clips/clips" not in _client(env).get_run(rid).data.metrics
    a.main(argv + ["--conf", "0.5"])
    r = _client(env).get_run(rid)
    m, t = r.data.metrics, r.data.tags
    assert m["eval/el-mohandes-13clips/recall_conf0.5"] == pytest.approx(0.8)
    assert m["eval/el-mohandes-13clips/recall_conf0.1"] == pytest.approx(0.9)
    assert m["eval/el-mohandes-13clips/recall_conf0.5/wlt32px"] == pytest.approx(0.5)
    assert m["eval/el-mohandes-13clips/recall_conf0.5/wge128"] == pytest.approx(1.0)
    assert not [k for k in m if "conf0.25" in k]
    assert t["eval.el-mohandes-13clips.conf_as_wired"] == "0.5"
    assert t["eval.el-mohandes-13clips.conf_as_wired_source"] == "--conf"
    assert t["eval.el-mohandes-13clips.anchor_conf"] == "0.25"
    assert json.loads(out.read_text())["config"]["conf_as_wired"] == 0.5
    assert m["eval/el-mohandes-13clips/clips"] == 3
    assert t["eval.el-mohandes-13clips.dvc_pointers"] == "3"
    assert t["eval.el-mohandes-13clips.model.anchor.sha256"] == __import__("hashlib").sha256(b"plate").hexdigest()
    arts = {x.path for x in _client(env).list_artifacts(rid, "eval/el-mohandes-13clips")}
    assert arts == {"eval/el-mohandes-13clips/report.json", "eval/el-mohandes-13clips/dvc_pointers.txt"}


def _jpeg(path: Path) -> None:
    """A real (tiny) JPEG: readable by real cv2 as well as by the fake one."""
    from PIL import Image
    Image.new("RGB", (60, 20), (128, 128, 128)).save(path, "JPEG")


def _ocr_module(monkeypatch, fake_cv2):
    if importlib.util.find_spec("onnxruntime") is None:
        monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    monkeypatch.delitem(sys.modules, "mtmc.eval_ocr_quality", raising=False)
    return importlib.import_module("mtmc.eval_ocr_quality")


def test_ocr_eval_needs_a_model_and_logs_quality_into_its_run(env, monkeypatch, fake_cv2):
    o = _ocr_module(monkeypatch, fake_cv2)
    crops = env.root / "Good-plates-OCR"
    for cam, n in (("cam1", 3), ("cam 2", 1)):
        (crops / cam).mkdir(parents=True)
        for i in range(n):
            _jpeg(crops / cam / f"{i}.jpg")
    model, rid = _model_run(env, "plate_rec.onnx", experiment="plate-ocr")
    dict_path = env.root / "plate_dict.txt"
    dict_path.write_text("1\n2\n")
    reads = iter([("123", 0.9), ("", 0.0), ("45", 0.7), ("6", 0.5)])
    monkeypatch.setattr(o, "load_plate_reader", lambda m, d, enhance: types.SimpleNamespace(read=lambda img: next(reads)))
    o.main(["--model", str(model), "--dict", str(dict_path), "--crops-dir", str(crops), "--out-dir",
            str(env.root / "ocr_out"), "--no-annotate"])
    r = _client(env).get_run(rid)
    m, t = r.data.metrics, r.data.tags
    assert m["eval/plate-ocr-golden/crops"] == 4 and m["eval/plate-ocr-golden/with_text_rate"] == pytest.approx(0.75)
    assert m["eval/plate-ocr-golden/mean_conf"] == pytest.approx((0.9 + 0.7 + 0.5) / 3)
    assert m["eval/plate-ocr-golden/camera/cam_2/with_text_rate"] == pytest.approx(1.0)
    assert t["eval.plate-ocr-golden.dvc_md5"] == "unversioned"                # the golden crops are not in DVC yet
    assert t["eval.plate-ocr-golden.metric_kind"] == "unlabelled-quality"
    assert t["eval.plate-ocr-golden.dict_sha256"] == __import__("hashlib").sha256(b"1\n2\n").hexdigest()
    assert t["eval.plate-ocr-golden.reader"].startswith("perception/modules/mtmc/plate_reader.py@")
    arts = {x.path for x in _client(env).list_artifacts(rid, "eval/plate-ocr-golden")}
    # no CSV, no crops; dvc_pointers.txt names the unversioned crops dir
    assert arts == {"eval/plate-ocr-golden/ocr_summary.json", "eval/plate-ocr-golden/dvc_pointers.txt"}
    assert (env.root / "ocr_out" / "ocr_results.csv").is_file()               # the per-crop CSV stays local


PLATE_TEXTS = ("أ ب ج ١٢٣", "س ص ٤٥٦٧", "م ن ٨٩")                       # what a plate reader returns


def test_ocr_artifacts_carry_no_plate_text(env, monkeypatch, fake_cv2, tmp_path):
    """Plate readings are personal data from client footage: none may reach the artifact store, as a string or as a
    plate-text column. Checks every artifact of the run, whatever its name."""
    o = _ocr_module(monkeypatch, fake_cv2)
    crops = env.root / "crops"
    (crops / "cam1").mkdir(parents=True)
    for i in range(len(PLATE_TEXTS)):
        _jpeg(crops / "cam1" / f"{i}.jpg")
    model, rid = _model_run(env, "plate_rec.onnx", experiment="plate-ocr")
    d = env.root / "plate_dict.txt"
    d.write_text("x\n")
    reads = iter([(t, 0.9) for t in PLATE_TEXTS])
    monkeypatch.setattr(o, "load_plate_reader", lambda m, dd, enhance: types.SimpleNamespace(read=lambda img: next(reads)))
    o.main(["--model", str(model), "--dict", str(d), "--crops-dir", str(crops), "--out-dir", str(env.root / "out"),
            "--no-annotate"])
    local = (env.root / "out" / "ocr_results.csv").read_text(encoding="utf-8")
    assert all(t in local for t in PLATE_TEXTS), "the fixture must really produce plate text"
    dst = tmp_path / "downloaded"
    dst.mkdir()
    files = [p for p in Path(_client(env).download_artifacts(rid, "", str(dst))).rglob("*") if p.is_file()]
    assert files, "the run must have artifacts to check"
    for f in files:
        body = f.read_bytes().decode("utf-8", errors="replace")
        assert not [t for t in PLATE_TEXTS if t in body], f"plate text in artifact {f.relative_to(dst)}"
        if f.suffix == ".csv":
            header = {h.strip().lower() for h in body.splitlines()[0].split(",")}
            assert not header & {"text", "plate", "plate_text", "model_v3_text", "model_v5_text"}, \
                f"plate-text column in artifact {f.relative_to(dst)}"


def test_ocr_eval_missing_crops_names_the_expected_path(env, monkeypatch, fake_cv2):
    o = _ocr_module(monkeypatch, fake_cv2)
    with pytest.raises(SystemExit, match="Good-plates-OCR.*dvc add"):
        o.main(["--crops-dir", str(env.root / "absent"), "--no-track"])


def test_ocr_eval_uses_the_sidecars_own_plate_reader(env, monkeypatch, fake_cv2):
    """load_plate_reader loads perception/modules/mtmc/plate_reader.py by path (not the heavy package)."""
    o = _ocr_module(monkeypatch, fake_cv2)
    ort = types.ModuleType("onnxruntime")
    ort.SessionOptions = lambda: types.SimpleNamespace(intra_op_num_threads=0)
    ort.InferenceSession = lambda path, sess_options=None, providers=None: types.SimpleNamespace(path=path)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    d = env.root / "dict.txt"
    d.write_text("A\nB\n")
    before = set(sys.modules)
    reader = o.load_plate_reader(env.root / "m.onnx", d, enhance=True)
    assert type(reader).__module__ == "iris_plate_reader"
    # loading it imported nothing of the engine or of the mtmc module package (whatever earlier tests imported)
    new = set(sys.modules) - before
    assert not [m for m in new if m.startswith(("perception.engine", "perception.modules.mtmc"))], sorted(new)
    probs = np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.1, 0.8]], np.float32)
    assert reader.decode(probs) == ("AB", pytest.approx((0.8 + 0.8) / 2))


def test_pipeline_eval_is_its_own_run_tagged_with_every_model(env, monkeypatch, fake_cv2):
    reid, reid_run = _model_run(env, "reid.pth", experiment="imported-models")
    plate = env.root / "plate.pt"
    plate.write_bytes(b"plate")                                              # no sidecar: sha256 only
    manifest = env.root / "models.manifest.json"
    manifest.write_text(json.dumps({"schema": 1, "models": [
        {"path": "products/el-mohandes/models/finetuned/best.pt", "registered_model": "plate-detector",
         "version": "3", "run_id": "r" * 32, "sha256": __import__("hashlib").sha256(b"plate").hexdigest()}]}))
    gt = env.root / "ground_truth_train.txt"
    gt.write_text("1 1 1 0 0 10 10 -1 -1\n")
    report = env.root / "report.json"
    report.write_text("{}")
    run = tev.log_pipeline_eval(eval_set="aicity22-S01", gt_path=gt, metrics={"idf1": 81.2, "idp": 85.0, "idr": 78.0},
                                params={"scene": "S01", "sim": 0.55}, models={"reid": reid, "plate_detector": plate},
                                manifest=manifest, artifacts=[report], run_name="S01-plate", script="mtmc.run_pipeline",
                                root=env.root)
    r = _client(env).get_run(run.run_id)
    m, t = r.data.metrics, r.data.tags
    assert _client(env).get_experiment(r.info.experiment_id).name == "mtmc-pipeline-eval"
    assert m["eval/aicity22-S01/idf1"] == pytest.approx(81.2)
    assert t["eval.aicity22-S01.dvc_md5"] == "unversioned" and t["eval.aicity22-S01.content_sha256"]
    assert t["model.reid.run_id"] == reid_run and len(t["model.reid.sha256"]) == 64
    assert t["model.plate_detector.registered_model"] == "plate-detector" and t["model.plate_detector.version"] == "3"
    assert t["model.plate_detector.in_manifest"] == "true" and t["model.reid.in_manifest"] == "false"
    assert r.data.params["sim"] == "0.55"
    assert {x.path for x in _client(env).list_artifacts(run.run_id, "eval/aicity22-S01")} == \
        {"eval/aicity22-S01/report.json"}
    assert reid_run not in {x.info.run_id for x in _client(env).search_runs([r.info.experiment_id])}


@pytest.mark.mlflow_store_absent
def test_run_pipeline_track_run_and_no_track(env, monkeypatch, fake_cv2):
    """run_pipeline's own wiring: models used by role, and --no-track (imports stubbed: no torch here)."""
    monkeypatch.setitem(sys.modules, "torch", sys.modules.get("torch") or types.ModuleType("torch"))
    T._install_fake_ultralytics(monkeypatch, env.root, "simulated")
    for mod in ("mtmc.run_pipeline", "mtmc.reid", "mtmc.yolo_detector"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    rp = importlib.import_module("mtmc.run_pipeline")
    gt_dir = env.root / "eval"
    gt_dir.mkdir()
    (gt_dir / "ground_truth_train.txt").write_text("gt\n")
    monkeypatch.setattr(rp, "EVAL_DIR", gt_dir)
    args = rp.build_parser().parse_args(["--det-source", "plate", "--plate-model", "p.pt"])
    assert set(rp.models_used(args)) == {"reid", "plate_detector"}
    assert set(rp.models_used(rp.build_parser().parse_args(["--skip-reid"]))) == set()
    report = {"scene": "S01", "cams": ["c001"], "det": "det.txt", "metrics": {"idf1": 50.0}, "tracklets": 3,
              "global_ids": 2, "prediction": "p.txt", "sim": 0.55}
    rpath = env.root / "report.json"
    rpath.write_text(json.dumps(report))
    rp.track_run(rp.build_parser().parse_args(["--no-track"]), report, rpath)
    assert not env.db.exists()
    rp.track_run(args, report, rpath)
    exp = _client(env).get_experiment_by_name("mtmc-pipeline-eval")
    (r,) = _client(env).search_runs([exp.experiment_id])
    assert r.info.run_name == "S01-plate" and r.data.metrics["eval/aicity22-S01/idf1"] == 50.0
    assert r.data.tags["model.plate_detector.sha256"] == "missing" and r.data.tags["detections"] == "plate"


# -- the audit's as-wired threshold -------------------------------------------------------------------------------
@pytest.fixture
def audit(monkeypatch, fake_cv2, tmp_path):
    T._install_fake_ultralytics(monkeypatch, tmp_path, "simulated")
    monkeypatch.delitem(sys.modules, "mtmc.audit_vehicle_detector", raising=False)
    monkeypatch.delitem(sys.modules, "mtmc.yolo_detector", raising=False)
    return importlib.import_module("mtmc.audit_vehicle_detector")


def test_audit_threshold_comes_from_the_real_deployed_configs(audit):
    """The repo's own configs: /workspace/iris/<path> and config-relative onnx-file entries both resolve."""
    assert audit.as_wired_conf(REPO / "products/el-mohandes/models/vehicle_1280_1cls/best.onnx") == \
        (0.5, "perception/models/DeepStream-Yolo/config_infer_primary_vehicle_1cls.txt")
    assert audit.as_wired_conf(audit._REPO / audit.VEHICLE_MODEL)[0] == 0.5            # the default model
    assert audit.as_wired_conf(REPO / "perception/models/DeepStream-Yolo/yolo11s.onnx") == \
        (0.25, "perception/models/DeepStream-Yolo/config_infer_primary_yolo11.txt")    # the legacy model
    assert audit.as_wired_conf("anything.onnx", conf=0.3) == (0.3, "--conf")


def _config(path: Path, onnx: str, threshold: float | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    attrs = f"\n[class-attrs-all]\nnms-iou-threshold=0.45\npre-cluster-threshold={threshold}  # comment\n" \
        if threshold is not None else ""
    path.write_text(f"[property]\ngpu-id=0\nonnx-file={onnx}\n# pre-cluster-threshold=0.9 in a comment\n{attrs}")
    return path


def test_audit_threshold_lookup_rules(audit, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    model = repo / "products/x/models/det/best.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"onnx")
    monkeypatch.setattr(audit, "_REPO", repo)
    with pytest.raises(SystemExit, match="no nvinfer config.*--conf"):
        audit.as_wired_conf(model)
    a = _config(repo / "perception/models/X/config_infer_a.txt", "/workspace/iris/products/x/models/det/best.onnx", 0.4)
    assert audit.as_wired_conf(model) == (0.4, "perception/models/X/config_infer_a.txt")
    _config(repo / "products/x/config_infer_b.txt", "models/det/best.onnx", 0.4)     # relative, same threshold
    assert audit.as_wired_conf(model)[0] == 0.4
    _config(repo / "products/x/config_infer_c.txt", "models/det/best.onnx", 0.6)     # disagrees
    with pytest.raises(SystemExit, match="disagree.*0.4.*0.6|disagree.*0.6.*0.4"):
        audit.as_wired_conf(model)
    assert audit.as_wired_conf(model, config=a) == (0.4, str(a))                     # --nvinfer-config settles it
    no_threshold = _config(tmp_path / "config_infer_none.txt", "best.onnx", None)
    with pytest.raises(SystemExit, match="no \\[class-attrs-all\\] pre-cluster-threshold"):
        audit.as_wired_conf(model, config=no_threshold)


def test_audit_report_keys_follow_the_threshold_and_anchors_do_not(audit):
    audit.set_thresholds(0.5)
    assert (audit.KEY_AS_WIRED, audit.KEY_LOOSE) == ("at_conf_0.5", "at_conf_0.1")
    assert audit.ANCHOR_CONF == 0.25                       # which plates count as vehicles never moves with it
    with pytest.raises(SystemExit, match="equals the low-threshold reference"):
        audit.set_thresholds(0.10)
