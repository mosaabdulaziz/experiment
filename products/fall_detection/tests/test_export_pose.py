"""The pose model's ONNX export: the three tensors the parser reads, and the numeric proof.

The export is model-agnostic: the checkpoint's own head says how many classes and joints it has,
so any YOLO pose checkpoint goes through the same code and produces the same three tensors. The
heavy parts (torch, ultralytics, onnxruntime) and the checkpoint itself are prerequisites: the
tests that need them skip cleanly where they are absent, and the export runs on CPU, so no GPU is
involved.
"""
import importlib
import types
from pathlib import Path

import pytest

import export_pose

PRODUCT = Path(__file__).resolve().parents[1]
REPO = PRODUCT.parents[1]
MODEL_DIR = REPO / "perception" / "models" / "DeepStream-Yolo-Pose"


def _require(module: str):
    try:
        return importlib.import_module(module)
    except ImportError:
        pytest.skip(f"{module} is not installed here: the export's prerequisites are missing")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """The exported graph of the real checkpoint, and the session that runs it."""
    for module in ("torch", "ultralytics", "onnxruntime"):
        _require(module)
    weights = export_pose.DEFAULT_WEIGHTS
    if not weights.is_file():
        pytest.skip(f"the pose checkpoint {weights} is not on this machine")
    out = tmp_path_factory.mktemp("onnx") / f"{export_pose.model_stem(weights)}.onnx"
    export_pose.export(weights, out)
    import onnxruntime as ort
    return weights, out, ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])


# ── the naming: the checkpoint's stem is the model's identity everywhere ──────────────────────────────────

def test_the_model_name_and_the_onnx_path_follow_the_checkpoint(tmp_path):
    assert export_pose.model_stem(Path("weights/yolo11n-pose.pt")) == "yolo11n-pose"
    assert export_pose.model_stem(Path("weights/yolo26x-pose.pt")) == "yolo26x-pose"
    out = export_pose.output_path(Path("weights/yolo26x-pose.pt"), model_dir=tmp_path)
    assert out == tmp_path / "yolo26x-pose.onnx"


# ── the three tensors the parser reads ────────────────────────────────────────────────────────────────────

def test_the_export_writes_the_three_tensors_the_parser_reads(exported):
    _, _, session = exported
    assert [o.name for o in session.get_outputs()] == ["boxes", "scores", "keypoints"]
    (boxes, scores, keypoints) = session.get_outputs()
    assert boxes.shape[-1] == 4                     # [N, 4]   x1, y1, x2, y2
    assert scores.shape[-1] == boxes.shape[1]       # [N]      the confidence
    assert keypoints.shape[-2:] == [17, 3]          # [N,17,3] x, y, conf
    assert keypoints.shape[1] == boxes.shape[1]     # the same detections
    assert [i.name for i in session.get_inputs()] == ["images"]
    assert session.get_inputs()[0].shape[1:] == [3, 640, 640]


def test_the_joint_count_follows_the_checkpoint_not_the_script(exported):
    """The model-agnostic claim, in one assertion: the joints come from the checkpoint's head."""
    weights, _, session = exported
    _require("ultralytics")
    from ultralytics import YOLO
    head = YOLO(str(weights)).model.model[-1]
    assert list(head.kpt_shape) == list(session.get_outputs()[2].shape[-2:])
    assert int(head.nc) == 1                        # the pose model's one class: a person


# ── the file that SHIPS, not the procedure that makes it ────────────────────────────────────────────────

def test_the_tracked_onnx_is_the_graph_the_parser_reads():
    """The proof above re-exports the graph; the deployment loads the TRACKED one. That file must
    carry the three names the parser looks for and the shapes the parser reads -- and it must
    carry them in ONE file, because this repository ignores the *.onnx.data sidecar that torch's
    exporter writes by default: a split export would ship a graph with no weights."""
    _require("onnx")
    import onnx
    from onnx import TensorProto

    onnx_path = MODEL_DIR / "yolo11n-pose.onnx"
    if not onnx_path.is_file():
        pytest.skip(f"the tracked pose ONNX {onnx_path} is not in this tree")

    model = onnx.load(str(onnx_path))

    def shape(value):
        return [d.dim_param or d.dim_value for d in value.type.tensor_type.shape.dim]

    outputs = {o.name: shape(o) for o in model.graph.output}
    assert set(outputs) == {"boxes", "scores", "keypoints"}, outputs
    assert [i.name for i in model.graph.input] == ["images"]
    # the batch axis is there (nvinfer batches the people's crops), and the detection count and
    # the joints come after it
    assert outputs["boxes"][1:] == [8400, 4]
    assert outputs["scores"][1:] == [8400]
    assert outputs["keypoints"][1:] == [8400, 17, 3]

    # ONE file: no tensor's data lives beside the graph
    external = [t.name for t in model.graph.initializer if t.data_location == TensorProto.EXTERNAL]
    assert external == [], external
    assert not onnx_path.with_suffix(onnx_path.suffix + ".data").exists()


# ── the proof: the export equals the model it came from ─────────────────────────────────────────────────

def test_the_tracked_onnx_agrees_with_the_model_it_came_from():
    """The tracked graph is the file nvinfer loads, so it is the one the proof must judge.

    Every other test here re-exports a fresh copy and proves THAT. A stale, truncated or
    wrong-but-well-shaped tracked file would pass them all -- and the deployment loads the tracked
    one, not a fresh export."""
    _require("onnxruntime")
    weights = export_pose.DEFAULT_WEIGHTS
    if not weights.is_file():
        pytest.skip(f"the pose checkpoint {weights} is not on this machine")
    onnx_path = MODEL_DIR / "yolo11n-pose.onnx"
    if not onnx_path.is_file():
        pytest.skip(f"the tracked pose ONNX {onnx_path} is not in this tree")
    image = export_pose.default_check_image()
    if image is None:
        pytest.skip("ultralytics ships no test image here to check the export against")

    report = export_pose.check(weights, onnx_path, image)
    assert report["ok"], report
    assert report["box_error_px"] <= export_pose.PIXEL_TOLERANCE
    assert report["keypoint_error_px"] <= export_pose.PIXEL_TOLERANCE


def test_a_checkpoint_with_more_than_one_class_is_refused():
    """The parser reads ONE class and ONE confidence per detection: the person. A checkpoint with
    more classes would have its class scores flattened into one number here, and the parser would
    report every class as class 0 -- a graph that cannot mean what it says."""
    head = types.SimpleNamespace(nc=2, nk=51, kpt_shape=(17, 3), end2end=False, xyxy=False)
    model = types.SimpleNamespace(model=[None, head])
    with pytest.raises(ValueError, match="ONE class"):
        export_pose._wrapper(model)


def test_the_exported_graph_agrees_with_the_model_it_came_from(exported):
    weights, onnx, _ = exported
    image = export_pose.default_check_image()
    if image is None:
        pytest.skip("ultralytics ships no test image here to check the export against")
    report = export_pose.check(weights, onnx, image)
    assert report["ok"], report
    assert report["box_error_px"] <= export_pose.PIXEL_TOLERANCE
    assert report["keypoint_error_px"] <= export_pose.PIXEL_TOLERANCE


# ── the proof compares like with like ────────────────────────────────────────────────────────────────────

def test_the_proof_makes_the_parsers_own_choices():
    """The proof compares the export against predict() with the parser's threshold and NMS, or it
    compares two different sets of detections and calls the difference an export error."""
    config = (MODEL_DIR / "config_sgie_pose.txt").read_text()
    assert f"pre-cluster-threshold={export_pose.PROOF_CONF}" in config
    parser = (MODEL_DIR / "nvdsinfer_custom_impl_Yolo_pose" / "nvdsparsepose_Yolo.cpp").read_text()
    assert f"#define NMS_THRESH {export_pose.PROOF_IOU}" in parser


def test_the_proof_reads_the_outputs_by_name_the_way_the_parser_finds_them(exported, tmp_path):
    """The three names are the contract between the export and the parser, so the proof must read
    them by NAME. A graph whose outputs are named differently is refused, not read by position --
    a positional read would compare one tensor against another and call it an export error, or
    worse, pass."""
    _require("onnx")
    import onnx

    weights, onnx_path, _ = exported
    renames = {"boxes": "output_boxes", "scores": "output_scores", "keypoints": "output_keypoints"}
    model = onnx.load(str(onnx_path))
    for node in model.graph.node:
        for index, out in enumerate(node.output):
            if out in renames:
                node.output[index] = renames[out]
    for output in model.graph.output:
        if output.name in renames:
            output.name = renames[output.name]
    renamed = tmp_path / "renamed.onnx"
    onnx.save(model, str(renamed))

    report = export_pose.check(weights, renamed, export_pose.default_check_image())
    assert not report["ok"], report
    assert "boxes" in report["reason"]


# ── the gate: a wrong export must not pass ───────────────────────────────────────────────────────────────

def _agreeing_sides():
    """The two sides of the comparison when the export is RIGHT: two people, 17 joints each."""
    import numpy as np

    boxes = np.array([[10.0, 20.0, 110.0, 220.0], [300.0, 40.0, 380.0, 240.0]])
    scores = np.array([0.90, 0.80])
    keypoints = np.array([[[10.0 + k, 20.0 + k, 0.80] for k in range(17)],
                          [[300.0 + k, 40.0 + k, 0.70] for k in range(17)]])
    predict = (boxes, scores, keypoints[:, :, 0:2], keypoints[:, :, 2])
    return (boxes, scores, keypoints), predict


def test_the_proof_passes_when_the_two_sides_agree():
    onnx_side, predict_side = _agreeing_sides()
    report = export_pose.compare(onnx_side, predict_side)
    assert report["ok"], report
    assert report["detections_compared"] == 2
    assert report["detections_unmatched"] == 0


def test_a_wrong_export_does_not_pass_the_proof():
    """The gate is the point of the proof, so it has to reject each way an export can be wrong:
    the boxes, the joints, the joints' confidences, the detection scores, and a detection that is
    on one side only."""
    import numpy as np

    onnx_side, predict_side = _agreeing_sides()

    moved_boxes = (onnx_side[0] + 10.0, onnx_side[1], onnx_side[2])
    assert not export_pose.compare(moved_boxes, predict_side)["ok"]

    moved_joints = (onnx_side[0], onnx_side[1], onnx_side[2] + 5.0)
    assert not export_pose.compare(moved_joints, predict_side)["ok"]

    wrong_joint_conf = onnx_side[2].copy()
    wrong_joint_conf[:, :, 2] = 0.1
    assert not export_pose.compare((onnx_side[0], onnx_side[1], wrong_joint_conf), predict_side)["ok"]

    wrong_scores = np.array([0.5, 0.5])
    assert not export_pose.compare((onnx_side[0], wrong_scores, onnx_side[2]), predict_side)["ok"]

    # a detection the export has and predict() does not: the two sides hold different objects
    assert not export_pose.compare(onnx_side, (predict_side[0][:1], predict_side[1][:1],
                                                predict_side[2][:1], predict_side[3][:1]))["ok"]


def test_a_graph_of_the_wrong_shape_is_refused_with_a_verdict_not_a_crash():
    """A wrong export can be wrong in its SHAPE as well as in its numbers, and reading a tensor
    of the wrong shape would raise an IndexError in the middle of the comparison. The proof
    answers with a verdict instead: a broken export is a failed proof, not a crash."""
    onnx_side, predict_side = _agreeing_sides()

    # the boxes without their last number: [N,3] instead of [N,4]
    report = export_pose.compare((onnx_side[0][:, 0:3], onnx_side[1], onnx_side[2]), predict_side)
    assert not report["ok"]
    assert "box" in report["reason"]

    # the keypoints flattened: [N,51] instead of [N,17,3]
    report = export_pose.compare((onnx_side[0], onnx_side[1], onnx_side[2].reshape(2, -1)), predict_side)
    assert not report["ok"]
    assert "keypoint" in report["reason"]

    # the two sides disagree about how many joints a person has
    report = export_pose.compare((onnx_side[0], onnx_side[1], onnx_side[2][:, 0:5, :]), predict_side)
    assert not report["ok"]
    assert "joint" in report["reason"]

    # the exported side holds more boxes than scores
    report = export_pose.compare((onnx_side[0], onnx_side[1][0:1], onnx_side[2]), predict_side)
    assert not report["ok"]
    assert "detections" in report["reason"]


def test_a_graph_that_writes_another_shape_is_refused_before_a_number_is_read():
    """The export writes the three tensors with the batch axis in front, because nvinfer batches
    the people's crops into one inference. A graph that wrote another shape would be read
    wrongly, so the proof checks the shape before it reads a single number."""
    import numpy as np

    good = {"boxes": np.zeros((1, 10, 4)), "scores": np.zeros((1, 10)),
            "keypoints": np.zeros((1, 10, 17, 3))}
    assert export_pose.onnx_shape_problem(good) is None

    # no batch axis
    problem = export_pose.onnx_shape_problem({"boxes": np.zeros((10, 4)), "scores": np.zeros((10,)),
                                               "keypoints": np.zeros((10, 17, 3))})
    assert problem and "boxes" in problem

    # the boxes hold five numbers each
    problem = export_pose.onnx_shape_problem({"boxes": np.zeros((1, 10, 5)), "scores": np.zeros((1, 10)),
                                               "keypoints": np.zeros((1, 10, 17, 3))})
    assert problem and "4" in problem


def test_a_joint_the_network_did_not_see_is_counted_not_judged():
    """A joint the network never saw carries an arbitrary position: the two runtimes do not agree
    on it, so its position is excused -- but its CONFIDENCE is not, and the count of excused joints
    is reported rather than hidden."""
    import numpy as np

    onnx_side, predict_side = _agreeing_sides()

    # Both sides say the network did not see the joint (its confidence is 0.05), but they place it
    # differently: this is what two runtimes do with a joint that carries no information.
    onnx_kpts = onnx_side[2].copy()
    onnx_kpts[0, 3, 0] += 50.0
    onnx_kpts[0, 3, 2] = 0.05
    predict_kpt_conf = predict_side[3].copy()
    predict_kpt_conf[0, 3] = 0.05

    report = export_pose.compare((onnx_side[0], onnx_side[1], onnx_kpts),
                                 (predict_side[0], predict_side[1], predict_side[2], predict_kpt_conf))
    assert report["ok"], report
    assert report["joints_unseen"] == 1
    assert report["unseen_joint_error_px"] > 2.0

    # The same moved joint, SEEN by the network (its confidence above the threshold), is judged.
    seen = onnx_side[2].copy()
    seen[0, 3, 0] += 50.0
    assert not export_pose.compare((onnx_side[0], onnx_side[1], seen), predict_side)["ok"]

    # A joint the network did not see still has to carry the SAME confidence on both sides: the
    # confidence is a sigmoid, and it is judged whether the joint was seen or not.
    wrong_conf = onnx_side[2].copy()
    wrong_conf[0, 3, 2] = 0.05
    assert not export_pose.compare((onnx_side[0], onnx_side[1], wrong_conf), predict_side)["ok"]
