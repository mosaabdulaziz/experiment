"""The pose model's ONNX export: the three tensors the parser reads.

The stock Ultralytics pose export writes one interleaved tensor of raw head channels, so the
parser would have to know the model's channel layout to read it -- and the layout changes with
the model. This export writes the three tensors the parser reads, each one self-describing
(design §3):

    boxes      [batch, N, 4]        x1, y1, x2, y2 in network-input pixels
    scores     [batch, N]           the detection confidence
    keypoints  [batch, N, 17, 3]    x, y, conf, in network-input pixels

The threshold and the NMS stay in the parser (the face parser's arrangement): the graph answers
"what did the network see", the parser answers "what is a detection".

The export is **model-agnostic**: the checkpoint's own head says how many classes and how many
joints it has, so yolo11n-pose and yolo26x-pose go through the same code and produce the same
three tensors. The batch is the only dynamic axis (nvinfer batches the people it crops into one
inference, `batch-size` in the config file); the spatial size is fixed, because a secondary GIE
runs on a crop scaled to fit the network's input.

**The proof that the export is right is numeric**: the exported graph's detections must equal
`model.predict()`'s, to within a couple of pixels (design §11). `check()` does that on a real
image: the image is resized to the network's input size first, so the letterbox is the identity
and both sides speak in the same pixels -- no transform to argue about. Every detection is
compared, and so is every confidence; only the joints the network never saw are excused, because
their positions are arbitrary in both runtimes (see `check`).

Usage:
    python export_pose.py --weights weights/yolo11n-pose.pt
    python export_pose.py --weights weights/yolo11n-pose.pt --check <image>

The heavy imports (torch, ultralytics, onnxruntime) are inside the functions, so importing this
module needs none of them: a measurement is gated by its prerequisites, not by an import.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

PRODUCT = Path(__file__).resolve().parent
REPO = PRODUCT.parents[1]
MODEL_DIR = REPO / "perception" / "models" / "DeepStream-Yolo-Pose"
DEFAULT_WEIGHTS = PRODUCT / "weights" / "yolo11n-pose.pt"
DEFAULT_IMGSZ = 640
# The shipped face model's ONNX is opset 18 and the card is a 5090 (TensorRT 10): both take it.
OPSET = 18
BOXES, SCORES, KEYPOINTS = "boxes", "scores", "keypoints"
# The proof's tolerance: the design asks for "a couple of pixels" (design §11).
PIXEL_TOLERANCE = 2.0
# The face parser's threshold and NMS IoU. The proof has to make the same choices the parser will,
# or it compares two different sets of detections.
PROOF_CONF = 0.25
PROOF_IOU = 0.45
# The proof's tolerance for the confidences: they are sigmoid outputs of the same graph read by
# two runtimes (ONNX Runtime and PyTorch), so they agree to floating-point noise.
PROOF_CONF_TOLERANCE = 1e-3
# A joint the network did not see carries an arbitrary position -- the head's raw output for it is
# noise, and the two runtimes do not agree on noise. So the position is compared only for the
# joints the network SAW, and the rest are reported without a verdict. The threshold is the
# DETECTION floor -- the parser's own -- and NOT the module's joint floor (0.30): the proof judges
# the positions of every joint at or above the detection floor, so it judges a SUPERSET of the
# joints the module counts as seen. It cannot pass an export the module would mis-read.
JOINT_JUDGED_THRESHOLD = PROOF_CONF
# Two detections are the same object when their boxes overlap this much.
MATCH_IOU = 0.5


def model_stem(weights: Path) -> str:
    """The model's name without its extension: yolo11n-pose.pt -> yolo11n-pose.

    The stem is the model's identity everywhere else -- the ONNX file's name, the engine's
    derived name, the module's `model` setting -- so it is derived in one place."""
    return Path(weights).stem


def output_path(weights: Path, model_dir: Path = MODEL_DIR) -> Path:
    """Where the export writes: the model's own directory, under the model's own name."""
    return Path(model_dir) / f"{model_stem(weights)}.onnx"


def _wrapper(model):
    """The model wrapped so its forward returns the three tensors.

    The head already decodes: `_get_decode_boxes` gives the boxes in the network's input pixels
    and `kpts_decode` gives the joints in the network's input pixels with a sigmoided
    confidence. So the wrapper only splits the head's one tensor into three -- it does not
    re-derive anything, which is what keeps the export model-agnostic.

    One thing is converted, not split: the head's boxes come out as xywh (the head's
    `dist2bbox` returns xywh unless the head is end2end), while the parser's contract is xyxy,
    like the face parser's. The conversion is the head's own convention, read off the head --
    so a checkpoint whose head answers xyxy is not converted twice."""
    import torch
    from torch import nn

    head = model.model[-1]
    head.export = True          # ultralytics' own exporter sets this: the head's export path
    head.dynamic = False        # the spatial size is fixed; only the batch moves
    if getattr(head, "end2end", False):
        raise ValueError("the head is end2end: its output is already post-processed, "
                         "so this export's split does not apply")
    classes = int(head.nc)
    if classes != 1:
        # The parser reads ONE class and ONE confidence per detection: the person. A checkpoint
        # with more classes would have its class scores flattened into one number here, and the
        # parser would report every class as class 0 -- a graph that cannot mean what it says.
        raise ValueError(f"the pose model must have ONE class (the person it detects); this "
                         f"checkpoint's head has {classes}. The parser reads one class and one "
                         f"confidence per detection.")
    joints = int(head.nk)       # joints * dims (17 * 3)
    kpt_shape = tuple(int(v) for v in head.kpt_shape)
    xywh = not getattr(head, "xyxy", False)

    class Decoded(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, images):
            y = self.model(images)
            if isinstance(y, (list, tuple)):
                y = y[0]                       # the training path returns (decoded, raw)
            batch = y.shape[0]
            boxes = y[:, 0:4].transpose(1, 2)                       # [B, N, 4]   xywh, input px
            if xywh:
                cx, cy, w, h = boxes.unbind(-1)
                boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
            scores = y[:, 4:4 + classes].transpose(1, 2).amax(dim=2)  # [B, N]     the confidence
            kpts = y[:, 4 + classes:4 + classes + joints].transpose(1, 2)
            kpts = kpts.reshape(batch, -1, *kpt_shape)              # [B, N, 17, 3]
            return boxes, scores, kpts

    return Decoded()


def export(weights: Path, out: Path, imgsz: int = DEFAULT_IMGSZ, opset: int = OPSET) -> Path:
    """Export the checkpoint to the three-tensor ONNX. Returns the file it wrote."""
    import torch
    from ultralytics import YOLO

    weights = Path(weights)
    out = Path(out)
    if not weights.is_file():
        raise FileNotFoundError(f"the checkpoint {weights} is not on this machine")
    out.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights)).model.eval()
    wrapper = _wrapper(model)
    wrapper.eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz)

    from torch.export import Dim
    with warnings.catch_warnings():
        # The head recomputes its anchors and strides inside the traced forward, and torch's
        # exporter warns about tensor attributes assigned during the export. They are the
        # head's own constants -- the class defines the names, so they cannot be registered as
        # buffers -- and the exporter restores the module's attributes afterwards.
        warnings.filterwarnings("ignore", message=".*assigned during export.*")
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out),
            input_names=["images"],
            output_names=[BOXES, SCORES, KEYPOINTS],
            dynamic_shapes={"images": {0: Dim("batch")}},
            opset_version=opset,
            dynamo=True,
            # ONE file. Torch writes the weights beside the graph as <name>.onnx.data by default,
            # and this repo ignores that sidecar (*.onnx.data in .gitignore) while tracking the
            # models themselves -- so a split export would leave a model whose weights are
            # missing. Every model in perception/models/ is a single self-contained .onnx.
            external_data=False,
        )
    _drop_stale_sidecar(out)
    return out


def _drop_stale_sidecar(out: Path) -> None:
    """Remove the weight sidecar an earlier split export left beside the graph.

    The graph is now self-contained, so the sidecar is dead weight; leaving it in the model's
    directory would make a reader believe the ONNX still needs it."""
    sidecar = out.with_suffix(out.suffix + ".data")
    if sidecar.is_file():
        sidecar.unlink()


def _letterbox_free_image(path: Path, imgsz: int):
    """The image as the network sees it, resized to the input size.

    The image is resized to the network's own input size, so ultralytics' letterbox is the
    identity: the proof compares the two sides in the same pixels, with no transform to argue
    about. Returns the RGB float tensor and the BGR image."""
    import cv2
    import numpy as np

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"the image {path} could not be read")
    image = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    tensor = image[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor), image


def _nms(boxes, scores, conf: float, iou: float):
    """The detections the parser will keep: the threshold and the NMS the parser does.

    The proof has to make the parser's choices, or it compares two different sets of
    detections and the difference is the NMS, not the export."""
    import numpy as np

    keep = [i for i, s in enumerate(scores) if s >= conf]
    keep.sort(key=lambda i: -scores[i])
    kept = []
    for i in keep:
        if all(_iou(boxes[i], boxes[j]) <= iou for j in kept):
            kept.append(i)
    return kept, np.asarray(boxes, dtype=float), np.asarray(scores, dtype=float)


def _iou(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _match(onnx_boxes, predict_boxes):
    """Pair the two sides' detections: each ONNX detection with the predict detection it is.

    Both sides now make the same choices (the same threshold, the same NMS IoU), so the two sets
    hold the same objects. The pairing is by overlap rather than by position in the list, because
    two runtimes may order equal scores differently. Returns the pairs and the number of
    detections that could not be paired."""
    pairs = []
    unmatched = 0
    used = set()
    for i, box in enumerate(onnx_boxes):
        best, best_iou = -1, 0.0
        for j, other in enumerate(predict_boxes):
            if j in used:
                continue
            overlap = _iou(box, other)
            if overlap > best_iou:
                best, best_iou = j, overlap
        if best < 0 or best_iou < MATCH_IOU:
            unmatched += 1
            continue
        used.add(best)
        pairs.append((i, best))
    return pairs, unmatched


def onnx_shape_problem(outputs: dict) -> str | None:
    """Why the exported graph's three outputs are not the shapes the parser reads, or None.

    The parser reads one crop's boxes `[N,4]`, scores `[N]` and keypoints `[N,17,3]`. The export
    writes them with the batch axis in front -- `[B,N,4]`, `[B,N]`, `[B,N,17,3]` -- because nvinfer
    batches the people's crops into one inference. A graph that wrote another shape would be read
    wrongly, so the proof refuses it with a verdict instead of reading a number out of the wrong
    place."""
    import numpy as np

    boxes = np.asarray(outputs[BOXES])
    scores = np.asarray(outputs[SCORES])
    keypoints = np.asarray(outputs[KEYPOINTS])

    if boxes.ndim != 3:
        return (f"the exported boxes are {boxes.ndim}-dimensional ({boxes.shape}), "
                f"expected [batch, detections, 4]")
    if scores.ndim != 2:
        return (f"the exported scores are {scores.ndim}-dimensional ({scores.shape}), "
                f"expected [batch, detections]")
    if keypoints.ndim != 4:
        return (f"the exported keypoints are {keypoints.ndim}-dimensional ({keypoints.shape}), "
                f"expected [batch, detections, joints, 3]")
    if boxes.shape[2] != 4:
        return f"the exported boxes hold {boxes.shape[2]} numbers per box, expected 4 (x1, y1, x2, y2)"
    if keypoints.shape[3] != 3:
        return f"the exported keypoints hold {keypoints.shape[3]} numbers per joint, expected 3 (x, y, conf)"
    if boxes.shape[:2] != keypoints.shape[:2]:
        return (f"the exported boxes ({boxes.shape[:2]}) and keypoints ({keypoints.shape[:2]}) "
                f"do not hold the same detections")
    if scores.shape != boxes.shape[:2]:
        return (f"the exported scores ({scores.shape}) do not hold the exported boxes' "
                f"detections ({boxes.shape[:2]})")
    return None


def compare_shape_problem(onnx_side, predict_side) -> str | None:
    """Why the two sides' shapes cannot be compared, or None.

    A wrong export can be wrong in its SHAPE as well as in its numbers, and reading a tensor of
    the wrong shape would raise an IndexError in the middle of the comparison. The proof answers
    with a verdict instead, so a broken export is a failed proof and not a crash."""
    import numpy as np

    boxes, scores, keypoints = (np.asarray(value) for value in onnx_side)
    p_boxes, p_scores, p_kpts, p_kpt_conf = (np.asarray(value) for value in predict_side)

    for value, axes, name in ((boxes, 2, "the exported boxes"),
                              (scores, 1, "the exported scores"),
                              (keypoints, 3, "the exported keypoints"),
                              (p_boxes, 2, "predict()'s boxes"),
                              (p_scores, 1, "predict()'s scores"),
                              (p_kpts, 3, "predict()'s joints"),
                              (p_kpt_conf, 2, "predict()'s joints' confidence")):
        if value.ndim != axes:
            return f"{name} are {value.ndim}-dimensional ({value.shape}), expected {axes} dimensions"

    if boxes.shape[1] != 4:
        return f"the exported boxes hold {boxes.shape[1]} numbers per box, expected 4 (x1, y1, x2, y2)"
    if p_boxes.shape[1] != 4:
        return f"predict()'s boxes hold {p_boxes.shape[1]} numbers per box, expected 4 (x1, y1, x2, y2)"
    if keypoints.shape[2] != 3:
        return f"the exported keypoints hold {keypoints.shape[2]} numbers per joint, expected 3 (x, y, conf)"
    if p_kpts.shape[2] != 2:
        return f"predict()'s joints hold {p_kpts.shape[2]} numbers per joint, expected 2 (x, y)"
    if not len(boxes) == len(scores) == len(keypoints):
        return (f"the exported side holds {len(boxes)} boxes, {len(scores)} scores and "
                f"{len(keypoints)} keypoint sets -- they must be the same detections")
    if not len(p_boxes) == len(p_scores) == len(p_kpts) == len(p_kpt_conf):
        return (f"predict()'s side holds {len(p_boxes)} boxes, {len(p_scores)} scores, "
                f"{len(p_kpts)} joint sets and {len(p_kpt_conf)} joint confidences -- they must "
                f"be the same detections")
    if not keypoints.shape[1] == p_kpts.shape[1] == p_kpt_conf.shape[1]:
        return (f"the exported side holds {keypoints.shape[1]} joints per detection and predict()'s "
                f"side holds {p_kpts.shape[1]} -- they must be the same joints")
    return None


def compare(onnx_side, predict_side, tolerance_px: float = PIXEL_TOLERANCE,
            conf_tolerance: float = PROOF_CONF_TOLERANCE) -> dict:
    """The two sides' detections, compared. Returns the report, and the verdict.

    `onnx_side` is (boxes, scores, keypoints) as the exported graph reports them -- the boxes
    `[N,4]`, the scores `[N]`, the keypoints `[N,17,3]` of ONE crop (the graph's own outputs carry
    the batch axis; `check()` hands over one crop's) -- and `predict_side` is (boxes, scores,
    kpts, kpt_conf) as `model.predict()` reports them: the boxes `[N,4]`, the scores `[N]`, the
    joints' xy `[N,17,2]` and the joints' confidence `[N,17]`.

    This is the whole gate, separated from the two runtimes that feed it, so it can be handed
    two sides that DISAGREE: a wrong export has to fail here, and that is only demonstrable if
    the comparison can be driven directly.

    Every detection is compared -- not only the best one -- and every quantity is gated: the
    boxes, the joints, the joints' confidences and the detection scores. A proof that compared
    one detection and ignored the confidences would pass an export that sliced the wrong
    confidence channel, which is exactly what the module's observation rule reads.

    The joints' POSITIONS are gated for the joints the network saw (confidence at or above the
    parser's threshold). A joint it did not see carries an arbitrary position -- the head's raw
    output for it is noise, and two runtimes do not agree on noise -- so those are counted and
    reported, not judged. Their CONFIDENCES are gated for every joint, seen or not: a sigmoid is
    a sigmoid, and it has to match."""
    import numpy as np

    problem = compare_shape_problem(onnx_side, predict_side)
    if problem:
        return {"ok": False, "reason": problem}

    onnx_boxes, onnx_scores, onnx_keypoints = onnx_side
    predict_boxes, predict_scores, predict_kpts, predict_kpt_conf = predict_side

    kept, box_array, score_array = _nms(onnx_boxes, onnx_scores, PROOF_CONF, PROOF_IOU)
    if not kept:
        return {"ok": False, "reason": "the exported graph kept nothing",
                "onnx_detections": int(len(onnx_scores))}

    pairs, unmatched = _match(box_array[kept], predict_boxes)
    if not pairs:
        return {"ok": False, "reason": "no detection of the exported graph matched one of "
                                       "predict()'s",
                "detections_onnx": len(kept), "detections_predict": int(len(predict_boxes))}

    box_error = kpt_error = kpt_conf_error = score_error = 0.0
    unseen_error = 0.0
    seen_joints = unseen_joints = 0
    for i, j in pairs:
        box_error = max(box_error, float(np.abs(box_array[kept[i]] - predict_boxes[j]).max()))
        score_error = max(score_error, abs(float(score_array[kept[i]]) - float(predict_scores[j])))
        onnx_kpts = onnx_keypoints[kept[i]]
        error = np.abs(onnx_kpts[:, 0:2] - predict_kpts[j]).max(axis=1)
        # The joints the network saw are compared; the ones it did not are counted and their
        # (arbitrary) positions are reported, not judged.
        seen = onnx_kpts[:, 2] >= JOINT_JUDGED_THRESHOLD
        seen_joints += int(seen.sum())
        unseen_joints += int((~seen).sum())
        if seen.any():
            kpt_error = max(kpt_error, float(error[seen].max()))
        if (~seen).any():
            unseen_error = max(unseen_error, float(error[~seen].max()))
        kpt_conf_error = max(kpt_conf_error,
                             float(np.abs(onnx_kpts[:, 2] - predict_kpt_conf[j]).max()))

    ok = (box_error <= tolerance_px
          and kpt_error <= tolerance_px
          and kpt_conf_error <= conf_tolerance
          and score_error <= conf_tolerance
          and unmatched == 0
          and len(kept) == len(predict_boxes))
    return {"ok": ok,
            "detections_onnx": len(kept), "detections_predict": int(len(predict_boxes)),
            "detections_compared": len(pairs), "detections_unmatched": unmatched,
            "box_error_px": round(box_error, 4), "keypoint_error_px": round(kpt_error, 4),
            "joints_seen": seen_joints, "joints_unseen": unseen_joints,
            "unseen_joint_error_px": round(unseen_error, 4),
            "keypoint_conf_error": round(kpt_conf_error, 6),
            "score_error": round(score_error, 6), "tolerance_px": tolerance_px,
            "confidence_tolerance": conf_tolerance}


def check(weights: Path, onnx_path: Path, image: Path, imgsz: int = DEFAULT_IMGSZ) -> dict:
    """The numeric proof: the exported graph's detections against `model.predict()`'s.

    Both sides see the same image at the network's own input size and make the same choices (the
    same threshold, the same NMS IoU), so both speak in the same pixels and hold the same objects.
    The comparison itself is `compare()`, which holds the verdict."""
    import onnxruntime as ort
    from ultralytics import YOLO

    weights, onnx_path, image = Path(weights), Path(onnx_path), Path(image)
    tensor, resized = _letterbox_free_image(image, imgsz)

    # A graph that cannot even be LOADED is a wrong export too, and it answers with a verdict like
    # every other way of being wrong: a failed proof, not a crash.
    try:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        outputs = session.run(None, {"images": tensor})
    except Exception as exc:  # noqa: BLE001 -- the reason is in the report
        return {"ok": False, "reason": f"the exported graph {onnx_path} could not be run: {exc}"}
    # The outputs are read BY NAME, the way the parser finds them: the three names are the
    # contract between this export and the parser, so a proof that read them by position would
    # be checking a different pairing than the one that ships.
    by_name = {o.name: value for o, value in zip(session.get_outputs(), outputs)}
    missing = [name for name in (BOXES, SCORES, KEYPOINTS) if name not in by_name]
    if missing:
        return {"ok": False, "reason": f"the exported graph has no {', '.join(missing)} output",
                "outputs": [o.name for o in session.get_outputs()]}
    problem = onnx_shape_problem(by_name)
    if problem:
        return {"ok": False, "reason": problem,
                "outputs": {name: list(value.shape) for name, value in by_name.items()}}
    boxes, scores, keypoints = by_name[BOXES][0], by_name[SCORES][0], by_name[KEYPOINTS][0]

    model = YOLO(str(weights))
    # The *resized* image goes in, not the original: predict() letterboxes what it is given and
    # scales its answer back to that image, so passing the original would answer in the original's
    # pixels while the graph answers in the network's. The resized image is already square, so
    # ultralytics' letterbox is the identity and both sides speak the network's pixels.
    # The IoU is the parser's own: predict() would otherwise use ultralytics' default (0.7) and
    # the two sides would hold different numbers of detections for NMS reasons, not export ones.
    result = model.predict(source=resized, conf=PROOF_CONF, iou=PROOF_IOU, imgsz=imgsz,
                           verbose=False)[0]
    if len(result.boxes) == 0:
        return {"ok": False, "reason": f"the model found nothing in {image}",
                "onnx_detections": int(len(scores))}

    predict_boxes = result.boxes.xyxy.cpu().numpy()
    predict_scores = result.boxes.conf.cpu().numpy()
    predict_kpts = result.keypoints.xy.cpu().numpy()          # [N, 17, 2] -- the joints' x, y
    predict_kpt_conf = result.keypoints.conf.cpu().numpy()   # [N, 17]    -- the joints' confidence

    report = compare((boxes, scores, keypoints),
                      (predict_boxes, predict_scores, predict_kpts, predict_kpt_conf))
    report["image"] = str(image)
    report["onnx"] = str(onnx_path)
    return report


def default_check_image() -> Path | None:
    """Ultralytics' own test image, when the package ships one. None when it does not."""
    try:
        import ultralytics
    except Exception:  # noqa: BLE001 -- the proof's fallback image is optional
        return None
    candidate = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    return candidate if candidate.is_file() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                        help="the pose checkpoint to export (default: %(default)s)")
    parser.add_argument("--out", default=None,
                        help="the ONNX to write (default: the model's directory, under its own name)")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--opset", type=int, default=OPSET)
    parser.add_argument("--check", nargs="?", const="", default=None,
                        help="an image to check the export against (default: ultralytics' own)")
    args = parser.parse_args(argv)

    weights = Path(args.weights)
    out = Path(args.out) if args.out else output_path(weights)

    image = args.check
    if image == "":
        fallback = default_check_image()
        if fallback is None:
            print("no image to check against -- pass one: --check <image>")
            return 1
        image = str(fallback)

    # The graph is PROVED before it reaches the model's directory. The directory is where nvinfer
    # loads it from, so a failing proof must not leave a broken graph sitting there.
    staging = out.with_name(out.stem + ".proving.onnx")
    try:
        export(weights, staging, imgsz=args.imgsz, opset=args.opset)
        if image:
            report = check(weights, staging, Path(image), imgsz=args.imgsz)
            for key, value in report.items():
                print(f"{key}: {value}")
            if not report.get("ok"):
                print(f"the proof failed -- {out} was NOT written")
                return 1
        os.replace(staging, out)
    finally:
        if staging.exists():
            staging.unlink()
    print(f"exported {weights.name} -> {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
