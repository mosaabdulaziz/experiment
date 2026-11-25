"""Reusable face-detector & ArcFace ONNX adapters for Phase 1 benchmarking.

Keeps production perception modules untouched while evaluating commercial face
detectors (Qualcomm face_det_lite, MediaPipe Face Detection, and box+landmark
combinations) and AuraFace (glintr100.onnx ONLY) against the current
yolov8n-face + buffalo_l (det_10g / w600k_r50) baselines.

All pure-NumPy geometry, decoding, NMS, and Umeyama alignment helpers live at
top level with no ``cv2`` / ``onnxruntime`` import requirement so unit tests can
verify the math anywhere. Runtime ONNX sessions import ``cv2`` and
``onnxruntime`` lazily when instantiated.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
from urllib.request import urlretrieve

import numpy as np

logger = logging.getLogger("iris.bench_face")

# Canonical 112x112 ArcFace alignment template (exact match to insightface.utils.face_align.arcface_dst).
ARCFACE_DST_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# MediaPipe 468-point FaceMesh vertex indices -> 5 ArcFace points (viewer's left/right).
FACEMESH_LEFT_EYE_PAIR = (33, 133)
FACEMESH_RIGHT_EYE_PAIR = (362, 263)
FACEMESH_NOSE_TIP = 1
FACEMESH_MOUTH_LEFT = 61
FACEMESH_MOUTH_RIGHT = 291

# Commercial model URLs (AuraFace explicitly pulls glintr100.onnx ONLY — never InsightFace files).
AURAFACE_GLINTR100_URL = "https://huggingface.co/fal/AuraFace-v1/resolve/main/glintr100.onnx"
FACE_DET_LITE_ASSETS_URL = (
    "https://huggingface.co/qualcomm/Lightweight-Face-Detection/resolve/main/release_assets.json"
)
MEDIAPIPE_FACE_ASSETS_URL = (
    "https://huggingface.co/qualcomm/MediaPipe-Face-Detection/resolve/main/release_assets.json"
)


@dataclass
class DetectedFace:
    """Single detected face in original image or person-crop pixel coordinates."""

    bbox: tuple[float, float, float, float]       # (x1, y1, x2, y2)
    score: float
    landmarks: Optional[np.ndarray] = None        # (5, 2) float32: [l_eye, r_eye, nose, l_mouth, r_mouth]

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def short_side(self) -> float:
        return min(self.width, self.height)


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def letterbox_params(src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[float, float, float]:
    """Scale and symmetric padding matching nvinfer maintain-aspect-ratio=1, symmetric-padding=1."""
    scale = min(dst_w / max(1, src_w), dst_h / max(1, src_h))
    pad_x = (dst_w - src_w * scale) / 2.0
    pad_y = (dst_h - src_h * scale) / 2.0
    return scale, pad_x, pad_y


def unmap_coords(
    pts: np.ndarray, scale: float, pad_x: float, pad_y: float, max_w: float, max_h: float
) -> np.ndarray:
    """Map (..., 2) points from letterboxed network pixels back to source image pixels."""
    out = np.empty_like(pts, dtype=np.float32)
    out[..., 0] = np.clip((pts[..., 0] - pad_x) / scale, 0.0, max_w)
    out[..., 1] = np.clip((pts[..., 1] - pad_y) / scale, 0.0, max_h)
    return out


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45) -> list[int]:
    """Pure-NumPy greedy NMS on (N, 4) xyxy boxes."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(-scores, kind="mergesort")
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0.0, inter / union, 0.0)
        order = rest[iou <= iou_thresh]
    return keep


def box_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def landmarks_plausible(landmarks: Optional[np.ndarray], bbox: Sequence[float]) -> bool:
    """Match FaceRecognizer._embed_aligned_face plausibility checks."""
    if landmarks is None or landmarks.shape != (5, 2):
        return False
    left, top, right, bottom = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    w, h = right - left, bottom - top
    if w <= 0.0 or h <= 0.0:
        return False
    mx, my = 0.5 * w, 0.5 * h
    inside = (
        (landmarks[:, 0] >= left - mx)
        & (landmarks[:, 0] <= right + mx)
        & (landmarks[:, 1] >= top - my)
        & (landmarks[:, 1] <= bottom + my)
    )
    if not bool(inside.all()):
        return False
    return bool(max(landmarks[0, 1], landmarks[1, 1]) < min(landmarks[3, 1], landmarks[4, 1]))


def face_yaw_from_5pt(landmarks: np.ndarray) -> float:
    """Match CustomerFaceCollector's face_yaw: |nose_x - eye_mid_x| / eye_dist."""
    p = np.asarray(landmarks, dtype=np.float32)
    eye_mid = (p[0] + p[1]) / 2.0
    dist = float(np.linalg.norm(p[1] - p[0]))
    if dist <= 1e-6:
        return float("inf")
    return abs(float(p[2, 0] - eye_mid[0])) / dist


def estimate_arcface_affine(src_pts: np.ndarray, image_size: int = 112) -> np.ndarray:
    """Compute the 2x3 similarity transform matrix mapping 5 src_pts -> ArcFace 112x112 template.

    Uses Umeyama's least-squares similarity transform (rotation + uniform scale + translation)
    without needing scikit-image, matching insightface.utils.face_align.estimate_norm.
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = ARCFACE_DST_112.astype(np.float64) * (float(image_size) / 112.0)
    if src.shape != (5, 2):
        raise ValueError(f"Expected (5, 2) landmarks, got {src.shape}")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    cov = (dst_demean.T @ src_demean) / 5.0
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2, dtype=np.float64)
    if np.linalg.det(cov) < 0:
        d[-1] = -1.0
    rot = u @ np.diag(d) @ vt
    var_src = float(np.sum(src_demean ** 2) / 5.0)
    scale = float(np.sum(s * d) / var_src) if var_src > 1e-12 else 1.0
    trans = dst_mean - scale * (rot @ src_mean)

    mat = np.zeros((2, 3), dtype=np.float32)
    mat[:, :2] = (scale * rot).astype(np.float32)
    mat[:, 2] = trans.astype(np.float32)
    return mat


def norm_crop_112(img_bgr: np.ndarray, landmark_5pt: np.ndarray, image_size: int = 112) -> np.ndarray:
    """Warp an image into a 112x112 upright ArcFace crop using 5 landmarks."""
    import cv2

    mat = estimate_arcface_affine(landmark_5pt, image_size=image_size)
    return cv2.warpAffine(img_bgr, mat, (image_size, image_size), borderValue=0.0)


# ── Pure-NumPy model output decoders ─────────────────────────────────────────

def decode_yolov8_face(
    output: np.ndarray,
    conf_thresh: float = 0.18,
    iou_thresh: float = 0.45,
) -> list[tuple[np.ndarray, float, Optional[np.ndarray]]]:
    """Decode YOLOv8-face ONNX outputs across common export layouts.

    Supports:
      - [N, 15|20]: DeepStream-Yolo-Face (x1,y1,x2,y2,conf, 5*(x,y[,c]))
      - [N, 21]:    Ultralytics end2end=True pose/face (x1,y1,x2,y2,conf,cls, 5*(x,y,vis))
      - [5, N] / [N, 5]: box-only YOLOv8-face (cx,cy,w,h,conf) -> landmarks=None
    """
    arr = np.asarray(output, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected yolov8n-face output shape: {arr.shape}")

    # Transpose channel-first Ultralytics raw head [C, anchors] (e.g., [5, 8400] or [20, 8400]).
    if arr.shape[0] in (5, 15, 20) and arr.shape[1] not in (5, 15, 20, 21):
        arr = arr.T
        cx, cy, bw, bh = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        xyxy = np.stack([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], axis=1)
        arr = np.concatenate([xyxy, arr[:, 4:]], axis=1)

    channels = arr.shape[1]
    if channels < 5:
        raise ValueError(f"Unexpected yolov8n-face channel count: {arr.shape}")

    scores = arr[:, 4]
    keep_mask = scores >= conf_thresh
    if not np.any(keep_mask):
        return []
    arr = arr[keep_mask]
    scores = scores[keep_mask]

    boxes = arr[:, :4]
    valid = (boxes[:, 2] - boxes[:, 0] >= 1.0) & (boxes[:, 3] - boxes[:, 1] >= 1.0)
    if not np.any(valid):
        return []
    arr, boxes, scores = arr[valid], boxes[valid], scores[valid]

    kpts: Optional[np.ndarray] = None
    if channels == 21:
        # [x1, y1, x2, y2, conf, cls, 5*(x, y, vis)]
        kpts = np.zeros((len(arr), 5, 2), dtype=np.float32)
        for p in range(5):
            kpts[:, p, 0] = arr[:, 6 + p * 3]
            kpts[:, p, 1] = arr[:, 7 + p * 3]
    elif channels >= 15:
        kpts = np.zeros((len(arr), 5, 2), dtype=np.float32)
        if channels - 5 == 10:
            for p in range(5):
                kpts[:, p, 0] = arr[:, 5 + p * 2]
                kpts[:, p, 1] = arr[:, 6 + p * 2]
        else:
            for p in range(5):
                kpts[:, p, 0] = arr[:, 5 + p * 3]
                kpts[:, p, 1] = arr[:, 6 + p * 3]

    kept = nms_xyxy(boxes, scores, iou_thresh=iou_thresh)
    return [(boxes[i].copy(), float(scores[i]), None if kpts is None else kpts[i].copy()) for i in kept]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _maxpool2d_3x3(hm: np.ndarray) -> np.ndarray:
    """Pure-NumPy 3x3 stride-1 padding-1 maxpool for (H, W) array."""
    padded = np.pad(hm, ((1, 1), (1, 1)), mode="constant", constant_values=-np.inf)
    views = [
        padded[dy : dy + hm.shape[0], dx : dx + hm.shape[1]]
        for dy in range(3)
        for dx in range(3)
    ]
    return np.maximum.reduce(views)


def decode_face_det_lite(
    hm: np.ndarray,
    box: np.ndarray,
    landmark: np.ndarray,
    conf_thresh: float = 0.55,
    nms_iou: float = 0.45,
    stride: int = 8,
    hm_is_logits: bool = True,
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Decode Qualcomm face_det_lite maps (hm [1,1,H,W], box [1,4,H,W], landmark [1,10,H,W])."""
    hm_2d = np.asarray(hm, dtype=np.float32).squeeze()
    box_3d = np.asarray(box, dtype=np.float32)
    lmk_3d = np.asarray(landmark, dtype=np.float32)
    if box_3d.ndim == 4:
        box_3d = box_3d[0]
    if lmk_3d.ndim == 4:
        lmk_3d = lmk_3d[0]

    prob = _sigmoid(hm_2d) if hm_is_logits else hm_2d
    peaks = (prob == _maxpool2d_3x3(prob)) & (prob >= conf_thresh)
    ys, xs = np.where(peaks)
    if len(xs) == 0:
        return []

    scores = prob[ys, xs]
    order = np.argsort(-scores, kind="mergesort")[:2000]
    xs, ys, scores = xs[order], ys[order], scores[order]

    boxes = np.empty((len(xs), 4), dtype=np.float32)
    kpts = np.empty((len(xs), 5, 2), dtype=np.float32)
    for i, (cx, cy) in enumerate(zip(xs, ys)):
        l, t, r, b = box_3d[:, cy, cx]
        boxes[i] = np.array([cx - l, cy - t, cx + r, cy + b], dtype=np.float32) * stride
        x5y5 = lmk_3d[:, cy, cx]
        kpts[i, :, 0] = (x5y5[:5] + float(cx)) * stride
        kpts[i, :, 1] = (x5y5[5:10] + float(cy)) * stride

    valid = (boxes[:, 2] - boxes[:, 0] >= 1.0) & (boxes[:, 3] - boxes[:, 1] >= 1.0)
    if not np.any(valid):
        return []
    boxes, scores, kpts = boxes[valid], scores[valid], kpts[valid]

    kept = nms_xyxy(boxes, scores, iou_thresh=nms_iou) if nms_iou > 0 else list(range(len(boxes)))
    return [(boxes[i].copy(), float(scores[i]), kpts[i].copy()) for i in kept]


def generate_blazeface_back_anchors() -> np.ndarray:
    """Generate the exact (896, 4) anchor table for MediaPipe BlazeFace 256x256 (anchorsback.npy)."""
    anchors: list[list[float]] = []
    for grid, count in ((16, 2), (8, 6)):
        step = 1.0 / grid
        for gy in range(grid):
            cy = (gy + 0.5) * step
            for gx in range(grid):
                cx = (gx + 0.5) * step
                for _ in range(count):
                    anchors.append([cx, cy, 1.0, 1.0])
    return np.asarray(anchors, dtype=np.float32)


def decode_mediapipe_face(
    box_coords: np.ndarray,
    box_scores: np.ndarray,
    anchors: Optional[np.ndarray] = None,
    img_size: tuple[int, int] = (256, 256),
    conf_thresh: float = 0.5,
    iou_thresh: float = 0.3,
    score_clip: float = 100.0,
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Decode MediaPipe BlazeFace detector raw outputs -> (bbox_xyxy, score, 6x2_kpts)."""
    coords = np.asarray(box_coords, dtype=np.float32)
    scores = np.asarray(box_scores, dtype=np.float32)
    if coords.ndim == 3:
        coords = coords[0]
    if scores.ndim == 3:
        scores = scores[0, :, 0]
    elif scores.ndim == 2:
        scores = scores.reshape(-1)

    if anchors is None:
        anchors = generate_blazeface_back_anchors()
    if len(coords) != len(anchors):
        raise ValueError(f"Expected {len(anchors)} anchors, got {len(coords)} predictions")

    probs = _sigmoid(np.clip(scores, -score_clip, score_clip))
    mask = probs >= conf_thresh
    if not np.any(mask):
        return []

    coords = coords[mask]
    probs = probs[mask]
    anc = anchors[mask]

    h_size, w_size = float(img_size[0]), float(img_size[1])
    k = coords.shape[1] // 2
    pts = coords.reshape(len(coords), k, 2).copy()
    offset = anc[:, 0:2][:, None, :] * np.array([[[w_size, h_size]]], dtype=np.float32)
    scale = anc[:, 2:4][:, None, :]
    pts = pts * scale
    # Index 1 is (w, h), so offset is not added to index 1.
    pts[:, 0, :] += offset[:, 0, :]
    if k > 2:
        pts[:, 2:, :] += offset

    cx, cy = pts[:, 0, 0], pts[:, 0, 1]
    w, h = pts[:, 1, 0], pts[:, 1, 1]
    boxes = np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1)
    kpts = pts[:, 2:8, :] if k >= 8 else pts[:, 2:, :]

    kept = nms_xyxy(boxes, probs, iou_thresh=iou_thresh)
    return [(boxes[i].copy(), float(probs[i]), kpts[i].copy()) for i in kept]


def extract_5pt_from_facemesh(landmarks_468: np.ndarray) -> np.ndarray:
    """Extract 5 canonical ArcFace landmarks from MediaPipe's 468-point FaceMesh output."""
    pts = np.asarray(landmarks_468, dtype=np.float32)
    if pts.ndim == 3:
        pts = pts[0]
    if pts.shape[0] < 363 or pts.shape[1] < 2:
        raise ValueError(f"Expected FaceMesh landmarks (>=363, >=2), got {pts.shape}")
    left_eye = 0.5 * (pts[FACEMESH_LEFT_EYE_PAIR[0], :2] + pts[FACEMESH_LEFT_EYE_PAIR[1], :2])
    right_eye = 0.5 * (pts[FACEMESH_RIGHT_EYE_PAIR[0], :2] + pts[FACEMESH_RIGHT_EYE_PAIR[1], :2])
    nose = pts[FACEMESH_NOSE_TIP, :2]
    mouth_left = pts[FACEMESH_MOUTH_LEFT, :2]
    mouth_right = pts[FACEMESH_MOUTH_RIGHT, :2]
    out = np.stack([left_eye, right_eye, nose, mouth_left, mouth_right], axis=0).astype(np.float32)
    # Ensure viewer left/right ordering regardless of model mirror conventions.
    if out[0, 0] > out[1, 0]:
        out[[0, 1]] = out[[1, 0]]
    if out[3, 0] > out[4, 0]:
        out[[3, 4]] = out[[4, 3]]
    return out


def decode_tao_detectnet_v2(
    cov: np.ndarray,
    bbox: np.ndarray,
    class_idx: int = 2,
    conf_thresh: float = 0.2,
    nms_iou: float = 0.45,
    stride: float = 16.0,
    box_norm: float = 35.0,
    offset: float = 0.5,
) -> list[tuple[np.ndarray, float]]:
    """Decode NVIDIA TAO DetectNet_v2 gridbox outputs (PeopleNet) -> list of (xyxy, score) in net pixels."""
    cov_arr = np.asarray(cov, dtype=np.float32)
    bbox_arr = np.asarray(bbox, dtype=np.float32)
    if cov_arr.ndim == 4:
        cov_arr = cov_arr[0]
    if bbox_arr.ndim == 4:
        bbox_arr = bbox_arr[0]
    if class_idx >= cov_arr.shape[0] or (class_idx + 1) * 4 > bbox_arr.shape[0]:
        return []

    cmap = cov_arr[class_idx]
    ys, xs = np.where(cmap >= conf_thresh)
    if len(xs) == 0:
        return []
    scores = cmap[ys, xs]
    b4 = bbox_arr[class_idx * 4 : (class_idx + 1) * 4, ys, xs]  # (4, M) -> dx1, dy1, dx2, dy2
    cx = xs.astype(np.float32) * stride + offset
    cy = ys.astype(np.float32) * stride + offset
    boxes = np.stack(
        [
            cx - b4[0] * box_norm,
            cy - b4[1] * box_norm,
            cx + b4[2] * box_norm,
            cy + b4[3] * box_norm,
        ],
        axis=1,
    ).astype(np.float32)
    valid = (boxes[:, 2] - boxes[:, 0] >= 1.0) & (boxes[:, 3] - boxes[:, 1] >= 1.0)
    if not np.any(valid):
        return []
    boxes, scores = boxes[valid], scores[valid]
    kept = nms_xyxy(boxes, scores, iou_thresh=nms_iou) if nms_iou > 0 else list(range(len(boxes)))
    return [(boxes[i].copy(), float(scores[i])) for i in kept]


def decode_yunet(
    outputs_by_name: dict[str, np.ndarray],
    strides: Sequence[int] = (8, 16, 32),
    input_wh: tuple[int, int] = (640, 640),
    conf_thresh: float = 0.35,
    nms_iou: float = 0.30,
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Decode OpenCV Zoo YuNet (cls/obj/bbox/kps across strides 8/16/32) -> (xyxy, score, 5x2_kpts)."""
    in_w, in_h = input_wh
    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_kpts: list[np.ndarray] = []

    for s in strides:
        cls = np.asarray(outputs_by_name[f"cls_{s}"], dtype=np.float32).reshape(-1)
        obj = np.asarray(outputs_by_name[f"obj_{s}"], dtype=np.float32).reshape(-1)
        bbox = np.asarray(outputs_by_name[f"bbox_{s}"], dtype=np.float32).reshape(-1, 4)
        kps = np.asarray(outputs_by_name[f"kps_{s}"], dtype=np.float32).reshape(-1, 5, 2)

        scores = np.sqrt(np.clip(cls, 0.0, 1.0) * np.clip(obj, 0.0, 1.0))
        mask = scores >= conf_thresh
        if not np.any(mask):
            continue

        grid_w = in_w // s
        idxs = np.where(mask)[0]
        gx = (idxs % grid_w).astype(np.float32)
        gy = (idxs // grid_w).astype(np.float32)

        bb = bbox[idxs]
        sc = scores[idxs]
        kp = kps[idxs].copy()

        cx = (gx + bb[:, 0]) * float(s)
        cy = (gy + bb[:, 1]) * float(s)
        bw = np.exp(np.clip(bb[:, 2], -10.0, 10.0)) * float(s)
        bh = np.exp(np.clip(bb[:, 3], -10.0, 10.0)) * float(s)
        xyxy = np.stack([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], axis=1).astype(np.float32)

        kp[:, :, 0] = (gx[:, None] + kp[:, :, 0]) * float(s)
        kp[:, :, 1] = (gy[:, None] + kp[:, :, 1]) * float(s)
        all_boxes.append(xyxy)
        all_scores.append(sc)
        all_kpts.append(kp)

    if not all_boxes:
        return []
    boxes = np.concatenate(all_boxes, axis=0)
    scores_cat = np.concatenate(all_scores, axis=0)
    kpts_cat = np.concatenate(all_kpts, axis=0)
    kept = nms_xyxy(boxes, scores_cat, iou_thresh=nms_iou) if nms_iou > 0 else list(range(len(boxes)))
    return [(boxes[i].copy(), float(scores_cat[i]), kpts_cat[i].copy()) for i in kept]


def decode_rfdetr_face(
    boxes_cxcywh: np.ndarray,
    logits: np.ndarray,
    img_wh: tuple[int, int] = (384, 384),
    face_col: int = 0,
    conf_thresh: float = 0.30,
    nms_iou: float = 0.45,
) -> list[tuple[np.ndarray, float]]:
    """Decode RF-DETR ONNX normalized cxcywh boxes + sigmoid logits -> (xyxy, score) in img_wh coords."""
    b_arr = np.asarray(boxes_cxcywh, dtype=np.float32)
    l_arr = np.asarray(logits, dtype=np.float32)
    if b_arr.ndim == 3:
        b_arr = b_arr[0]
    if l_arr.ndim == 3:
        l_arr = l_arr[0]

    scores = _sigmoid(l_arr[:, face_col])
    mask = scores >= conf_thresh
    if not np.any(mask):
        return []
    b_sel = b_arr[mask]
    s_sel = scores[mask]
    w_img, h_img = float(img_wh[0]), float(img_wh[1])
    cx, cy, bw, bh = b_sel[:, 0], b_sel[:, 1], b_sel[:, 2], b_sel[:, 3]
    boxes = np.stack(
        [
            np.clip((cx - bw / 2.0) * w_img, 0.0, w_img),
            np.clip((cy - bh / 2.0) * h_img, 0.0, h_img),
            np.clip((cx + bw / 2.0) * w_img, 0.0, w_img),
            np.clip((cy + bh / 2.0) * h_img, 0.0, h_img),
        ],
        axis=1,
    ).astype(np.float32)
    valid = (boxes[:, 2] - boxes[:, 0] >= 1.0) & (boxes[:, 3] - boxes[:, 1] >= 1.0)
    if not np.any(valid):
        return []
    boxes, s_sel = boxes[valid], s_sel[valid]
    kept = nms_xyxy(boxes, s_sel, iou_thresh=nms_iou) if nms_iou > 0 else list(range(len(boxes)))
    return [(boxes[i].copy(), float(s_sel[i])) for i in kept]


# ── Runtime ONNX Helpers & Adapters ──────────────────────────────────────────

def _make_ort_session(onnx_path: str | Path, gpu: bool = True):
    import onnxruntime as ort

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    return sess


def _download_file(url: str, dst: Path) -> Optional[Path]:
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    try:
        logger.info("Fetching %s -> %s", url, dst)
        urlretrieve(url, str(dst))
        if dst.is_file() and dst.stat().st_size > 0:
            return dst
    except Exception as exc:  # noqa: BLE001
        logger.warning("Download failed for %s: %s", url, exc)
        if dst.exists():
            dst.unlink()
    return None


def _fetch_qualcomm_onnx_bundle(assets_url: str, dest_dir: Path) -> list[Path]:
    """Resolve release_assets.json -> precisions.float.universal_assets.onnx.download_url and extract .onnx files."""
    import json
    import zipfile
    from urllib.request import urlopen

    existing = sorted(dest_dir.rglob("*.onnx"))
    if existing:
        return existing
    try:
        meta = json.loads(urlopen(assets_url, timeout=15).read().decode("utf-8"))
        dl_url = meta["precisions"]["float"]["universal_assets"]["onnx"]["download_url"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not resolve %s: %s", assets_url, exc)
        return []

    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "bundle.zip"
    if _download_file(dl_url, zip_path) is None:
        return []
    extracted: list[Path] = []
    root = dest_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir() or not member.filename.endswith((".onnx", ".data")):
                continue
            target = (root / Path(member.filename).name).resolve()
            if target.parent != root:
                continue
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            if target.suffix == ".onnx":
                extracted.append(target)
    zip_path.unlink(missing_ok=True)
    return sorted(extracted)


def fetch_commercial_face_models(cache_dir: str | Path) -> dict[str, Path]:
    """Download ONLY verified commercially usable model files into cache_dir.

    Strictly downloads ``glintr100.onnx`` from ``fal/AuraFace-v1`` (never the 4
    bundled InsightFace files), plus ``face_det_lite`` and ``MediaPipe-Face-Detection``
    ONNX bundles via Qualcomm's ``release_assets.json`` if missing.
    """
    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}

    aura = _download_file(AURAFACE_GLINTR100_URL, out_dir / "aurafaces_glintr100.onnx")
    if aura is not None:
        resolved["aurafaces_glintr100"] = aura

    fdl_files = _fetch_qualcomm_onnx_bundle(FACE_DET_LITE_ASSETS_URL, out_dir / "face_det_lite")
    if fdl_files:
        resolved["face_det_lite"] = fdl_files[0]

    mp_files = _fetch_qualcomm_onnx_bundle(MEDIAPIPE_FACE_ASSETS_URL, out_dir / "mediapipe_face")
    for p in mp_files:
        low = p.name.lower()
        if "landmark" in low:
            resolved["mediapipe_landmark"] = p
        elif "detect" in low or "face" in low:
            resolved.setdefault("mediapipe_detector", p)

    return resolved


class BaseFaceDetector:
    name: str = "base"

    def detect(self, img_bgr: np.ndarray, conf_thresh: Optional[float] = None) -> list[DetectedFace]:
        raise NotImplementedError

    def benchmark_batch(self, batch_size: int = 1, iters: int = 50, warmup: int = 10) -> float:
        raise NotImplementedError


def _time_ort_session(sess, input_shape: tuple[int, ...], iters: int, warmup: int, dtype=np.float32) -> float:
    import time

    inp = sess.get_inputs()[0]
    batch_req = input_shape[0]
    inp_batch = inp.shape[0] if inp.shape else None
    if isinstance(inp_batch, int) and inp_batch != batch_req:
        raise ValueError(f"static batch={inp_batch} (cannot run batch={batch_req})")
    shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in input_shape]
    dummy = np.zeros(shape, dtype=dtype)
    for _ in range(warmup):
        sess.run(None, {inp.name: dummy})
    t0 = time.perf_counter()
    for _ in range(iters):
        sess.run(None, {inp.name: dummy})
    return ((time.perf_counter() - t0) * 1000.0) / max(1, iters)


class YoloV8FaceDetector(BaseFaceDetector):
    """Baseline SGIE face detector (yolov8n-face.onnx)."""

    name = "yolov8n-face"

    def __init__(self, onnx_path: str | Path, gpu: bool = True, default_conf: float = 0.18, input_wh=(640, 640)):
        self.sess = _make_ort_session(onnx_path, gpu=gpu)
        self.input_name = self.sess.get_inputs()[0].name
        self.default_conf = default_conf
        self.input_wh = input_wh

    def detect(self, img_bgr: np.ndarray, conf_thresh: Optional[float] = None) -> list[DetectedFace]:
        import cv2

        thresh = self.default_conf if conf_thresh is None else conf_thresh
        h, w = img_bgr.shape[:2]
        dst_w, dst_h = self.input_wh
        scale, pad_x, pad_y = letterbox_params(w, h, dst_w, dst_h)
        resized = cv2.resize(img_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
        x0, y0 = int(round(pad_x)), int(round(pad_y))
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]

        raw = self.sess.run(None, {self.input_name: blob})[0]
        decoded = decode_yolov8_face(raw, conf_thresh=thresh, iou_thresh=0.45)
        out: list[DetectedFace] = []
        for box, score, kpts in decoded:
            ubox = unmap_coords(box.reshape(2, 2), scale, pad_x, pad_y, w, h).reshape(4)
            ukpts = unmap_coords(kpts, scale, pad_x, pad_y, w, h) if kpts is not None else None
            out.append(DetectedFace(bbox=tuple(float(v) for v in ubox), score=score, landmarks=ukpts))
        return out

    def benchmark_batch(self, batch_size: int = 1, iters: int = 50, warmup: int = 10) -> float:
        return _time_ort_session(self.sess, (batch_size, 3, self.input_wh[1], self.input_wh[0]), iters, warmup)


class QualcommFaceDetLiteDetector(BaseFaceDetector):
    """Rank 1 candidate: qualcomm/Lightweight-Face-Detection (BSD-3-Clause, no WIDER FACE)."""

    name = "qualcomm_face_det_lite"

    def __init__(self, onnx_path: str | Path, gpu: bool = True, default_conf: float = 0.35):
        self.sess = _make_ort_session(onnx_path, gpu=gpu)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.channels = int(shape[1]) if len(shape) == 4 and isinstance(shape[1], int) else 1
        self.dst_h = int(shape[2]) if len(shape) == 4 and isinstance(shape[2], int) else 480
        self.dst_w = int(shape[3]) if len(shape) == 4 and isinstance(shape[3], int) else 640
        self.default_conf = default_conf

    def detect(self, img_bgr: np.ndarray, conf_thresh: Optional[float] = None) -> list[DetectedFace]:
        import cv2

        thresh = self.default_conf if conf_thresh is None else conf_thresh
        h, w = img_bgr.shape[:2]
        scale, pad_x, pad_y = letterbox_params(w, h, self.dst_w, self.dst_h)
        resized = cv2.resize(img_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.dst_h, self.dst_w, 3), dtype=np.uint8)
        x0, y0 = int(round(pad_x)), int(round(pad_y))
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized

        if self.channels == 1:
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            blob = gray[None, None, :, :]
        else:
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = np.transpose(rgb, (2, 0, 1))[None, ...]

        outputs = self.sess.run(None, {self.input_name: blob})
        if len(outputs) == 1 and outputs[0].ndim == 3 and outputs[0].shape[-1] == 15:
            decoded = decode_yolov8_face(outputs[0], conf_thresh=thresh, iou_thresh=0.45)
        else:
            hm, box, landmark = _match_face_det_lite_outputs(outputs)
            decoded = decode_face_det_lite(hm, box, landmark, conf_thresh=thresh, nms_iou=0.45, stride=8)
        out: list[DetectedFace] = []
        for b_xyxy, score, kpts in decoded:
            ubox = unmap_coords(b_xyxy.reshape(2, 2), scale, pad_x, pad_y, w, h).reshape(4)
            ukpts = unmap_coords(kpts, scale, pad_x, pad_y, w, h) if kpts is not None else None
            out.append(DetectedFace(bbox=tuple(float(v) for v in ubox), score=score, landmarks=ukpts))
        return out

    def benchmark_batch(self, batch_size: int = 1, iters: int = 50, warmup: int = 10) -> float:
        return _time_ort_session(self.sess, (batch_size, self.channels, self.dst_h, self.dst_w), iters, warmup)


def _match_face_det_lite_outputs(outputs: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify hm (1 ch), box (4 ch), landmark (10 ch) regardless of ONNX output ordering."""
    if len(outputs) < 3:
        raise ValueError(f"face_det_lite requires 3 output tensors, got {len(outputs)}")
    by_ch = {arr.shape[1]: arr for arr in outputs if arr.ndim == 4}
    if {1, 4, 10}.issubset(by_ch):
        return by_ch[1], by_ch[4], by_ch[10]
    return outputs[0], outputs[1], outputs[2]


class MediaPipeLandmarkRefiner:
    """Runs MediaPipe's 192x192 FaceLandmark ONNX on a cropped face box to yield 5 ArcFace points."""

    def __init__(self, landmark_onnx: str | Path, gpu: bool = True):
        self.sess = _make_ort_session(landmark_onnx, gpu=gpu)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.dst_h = int(shape[2]) if len(shape) == 4 and isinstance(shape[2], int) else 192
        self.dst_w = int(shape[3]) if len(shape) == 4 and isinstance(shape[3], int) else 192

    def predict_5pt(self, img_bgr: np.ndarray, bbox_xyxy: Sequence[float], expand: float = 0.25) -> Optional[np.ndarray]:
        import cv2

        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        bw, bh = x2 - x1, y2 - y1
        if bw < 2 or bh < 2:
            return None
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = max(bw, bh) * (1.0 + expand)
        rx1 = max(0, int(math.floor(cx - side / 2.0)))
        ry1 = max(0, int(math.floor(cy - side / 2.0)))
        rx2 = min(w, int(math.ceil(cx + side / 2.0)))
        ry2 = min(h, int(math.ceil(cy + side / 2.0)))
        crop = img_bgr[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            return None

        ch, cw = crop.shape[:2]
        resized = cv2.resize(crop, (self.dst_w, self.dst_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]
        outputs = self.sess.run(None, {self.input_name: blob})
        lm_tensor = max(outputs, key=lambda a: a.size)
        pts = lm_tensor.reshape(-1, 3) if lm_tensor.size % 3 == 0 else lm_tensor.reshape(-1, 2)
        k5 = extract_5pt_from_facemesh(pts)
        # Handle normalized [0, 1] vs network-pixel [0, dst_w] outputs.
        if float(np.max(np.abs(pts[:, :2]))) <= 1.5:
            k5[:, 0] *= cw
            k5[:, 1] *= ch
        else:
            k5[:, 0] *= cw / float(self.dst_w)
            k5[:, 1] *= ch / float(self.dst_h)
        k5[:, 0] += rx1
        k5[:, 1] += ry1
        return k5


class MediaPipeFaceDetector(BaseFaceDetector):
    """Rank 2a candidate: qualcomm/MediaPipe-Face-Detection (Apache-2.0)."""

    name = "qualcomm_mediapipe_face"

    def __init__(
        self,
        detector_onnx: str | Path,
        landmark_onnx: Optional[str | Path] = None,
        gpu: bool = True,
        default_conf: float = 0.5,
    ):
        self.sess = _make_ort_session(detector_onnx, gpu=gpu)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.dst_h = int(shape[2]) if len(shape) == 4 and isinstance(shape[2], int) else 256
        self.dst_w = int(shape[3]) if len(shape) == 4 and isinstance(shape[3], int) else 256
        self.default_conf = default_conf
        self.anchors = generate_blazeface_back_anchors()
        self.refiner = MediaPipeLandmarkRefiner(landmark_onnx, gpu=gpu) if landmark_onnx else None

    def detect(self, img_bgr: np.ndarray, conf_thresh: Optional[float] = None) -> list[DetectedFace]:
        import cv2

        thresh = self.default_conf if conf_thresh is None else conf_thresh
        h, w = img_bgr.shape[:2]
        scale, pad_x, pad_y = letterbox_params(w, h, self.dst_w, self.dst_h)
        resized = cv2.resize(img_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.dst_h, self.dst_w, 3), dtype=np.uint8)
        x0, y0 = int(round(pad_x)), int(round(pad_y))
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]

        outputs = self.sess.run(None, {self.input_name: blob})
        if len(outputs) == 4:
            coords = np.concatenate([outputs[0], outputs[1]], axis=1)
            scores = np.concatenate([outputs[2], outputs[3]], axis=1)
            decoded = decode_mediapipe_face(
                coords, scores, self.anchors, img_size=(self.dst_h, self.dst_w), conf_thresh=thresh
            )
        elif len(outputs) == 2:
            decoded = decode_mediapipe_face(
                outputs[0], outputs[1], self.anchors, img_size=(self.dst_h, self.dst_w), conf_thresh=thresh
            )
        else:
            raise ValueError(f"Unexpected MediaPipe FaceDetector output count: {len(outputs)}")

        out: list[DetectedFace] = []
        for box, score, kpts6 in decoded:
            ubox = unmap_coords(box.reshape(2, 2), scale, pad_x, pad_y, w, h).reshape(4)
            box_tuple = tuple(float(v) for v in ubox)
            k5 = self.refiner.predict_5pt(img_bgr, box_tuple) if self.refiner else None
            if k5 is None and kpts6.shape[0] >= 4:
                # Fallback from BlazeFace 6-pt (r_eye, l_eye, nose, mouth_center, r_ear, l_ear).
                uk6 = unmap_coords(kpts6, scale, pad_x, pad_y, w, h)
                bw = ubox[2] - ubox[0]
                m_cx, m_cy = uk6[3, 0], uk6[3, 1]
                k5 = np.stack(
                    [
                        uk6[0],
                        uk6[1],
                        uk6[2],
                        np.array([m_cx - 0.18 * bw, m_cy], dtype=np.float32),
                        np.array([m_cx + 0.18 * bw, m_cy], dtype=np.float32),
                    ],
                    axis=0,
                )
                if k5[0, 0] > k5[1, 0]:
                    k5[[0, 1]] = k5[[1, 0]]
            out.append(DetectedFace(bbox=box_tuple, score=score, landmarks=k5))
        return out

    def benchmark_batch(self, batch_size: int = 1, iters: int = 50, warmup: int = 10) -> float:
        return _time_ort_session(self.sess, (batch_size, 3, self.dst_h, self.dst_w), iters, warmup)


class ScrfdDet10gDetector(BaseFaceDetector):
    """Baseline enrolment detector (InsightFace buffalo_l det_10g.onnx via FaceAnalysis)."""

    name = "buffalo_l_det_10g"

    def __init__(self, pack_root: Optional[str | Path] = None, gpu: bool = True, default_conf: float = 0.4):
        from insightface.app import FaceAnalysis

        if pack_root is None:
            from perception import insightface_pack

            pack_root = insightface_pack.root("buffalo_l")
        else:
            pack_root = str(pack_root)
        self.app = FaceAnalysis(name="buffalo_l", root=pack_root, allowed_modules=["detection"])
        self.app.prepare(ctx_id=0 if gpu else -1, det_thresh=default_conf, det_size=(640, 640))
        self.default_conf = default_conf

    def detect(self, img_bgr: np.ndarray, conf_thresh: Optional[float] = None) -> list[DetectedFace]:
        thresh = self.default_conf if conf_thresh is None else conf_thresh
        faces = self.app.get(img_bgr)
        out: list[DetectedFace] = []
        for f in faces:
            if float(f.det_score) < thresh:
                continue
            kps = np.asarray(f.kps, dtype=np.float32) if getattr(f, "kps", None) is not None else None
            out.append(DetectedFace(bbox=tuple(float(v) for v in f.bbox), score=float(f.det_score), landmarks=kps))
        return out

    def benchmark_batch(self, batch_size: int = 1, iters: int = 50, warmup: int = 10) -> float:
        det = self.app.models["detection"]
        return _time_ort_session(det.session, (batch_size, 3, 640, 640), iters, warmup)


class BoxPlusMediaPipeLandmarkDetector(BaseFaceDetector):
    """Wraps a box-only DeepStream-Yolo-compatible ONNX detector ([B,N,6]) + MediaPipe 5-pt crop landmarks."""

    def __init__(
        self,
        name: str,
        box_onnx: str | Path,
        landmark_onnx: str | Path,
        gpu: bool = True,
        default_conf: float = 0.25,
        input_wh: tuple[int, int] = (640, 640),
    ):
        self.name = name
        self.sess = _make_ort_session(box_onnx, gpu=gpu)
        self.input_name = self.sess.get_inputs()[0].name
        self.input_wh = input_wh
        self.default_conf = default_conf
        self.refiner = MediaPipeLandmarkRefiner(landmark_onnx, gpu=gpu)

    def detect(self, img_bgr: np.ndarray, conf_thresh: Optional[float] = None) -> list[DetectedFace]:
        import cv2

        thresh = self.default_conf if conf_thresh is None else conf_thresh
        h, w = img_bgr.shape[:2]
        dst_w, dst_h = self.input_wh
        scale, pad_x, pad_y = letterbox_params(w, h, dst_w, dst_h)
        resized = cv2.resize(img_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
        x0, y0 = int(round(pad_x)), int(round(pad_y))
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]

        raw = self.sess.run(None, {self.input_name: blob})[0]
        arr = raw[0] if raw.ndim == 3 else raw
        mask = arr[:, 4] >= thresh
        if not np.any(mask):
            return []
        arr = arr[mask]
        kept = nms_xyxy(arr[:, :4], arr[:, 4], iou_thresh=0.45)
        out: list[DetectedFace] = []
        for idx in kept:
            ubox = unmap_coords(arr[idx, :4].reshape(2, 2), scale, pad_x, pad_y, w, h).reshape(4)
            box_tuple = tuple(float(v) for v in ubox)
            k5 = self.refiner.predict_5pt(img_bgr, box_tuple)
            out.append(DetectedFace(bbox=box_tuple, score=float(arr[idx, 4]), landmarks=k5))
        return out

    def benchmark_batch(self, batch_size: int = 1, iters: int = 50, warmup: int = 10) -> float:
        return _time_ort_session(self.sess, (batch_size, 3, self.input_wh[1], self.input_wh[0]), iters, warmup)


class TaoPeopleNetFaceDetector(BaseFaceDetector):
    """NVIDIA TAO PeopleNet v2.3.4 (DetectNet_v2 ResNet34, NVIDIA Open Model License, class 2 = face)."""

    name = "tao_peoplenet_v2_3_4"

    def __init__(
        self,
        onnx_path: str | Path,
        landmark_onnx: Optional[str | Path] = None,
        gpu: bool = True,
        default_conf: float = 0.20,
        class_idx: int = 2,
    ):
        self.sess = _make_ort_session(onnx_path, gpu=gpu)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.dst_h = int(shape[2]) if len(shape) == 4 and isinstance(shape[2], int) else 544
        self.dst_w = int(shape[3]) if len(shape) == 4 and isinstance(shape[3], int) else 960
        self.default_conf = default_conf
        self.class_idx = class_idx
        self.refiner = MediaPipeLandmarkRefiner(landmark_onnx, gpu=gpu) if landmark_onnx else None

    def detect(self, img_bgr: np.ndarray, conf_thresh: Optional[float] = None) -> list[DetectedFace]:
        import cv2

        thresh = self.default_conf if conf_thresh is None else conf_thresh
        h, w = img_bgr.shape[:2]
        resized = cv2.resize(img_bgr, (self.dst_w, self.dst_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]
        outs = {o.name: v for o, v in zip(self.sess.get_outputs(), self.sess.run(None, {self.input_name: blob}))}
        cov = outs.get("output_cov/Sigmoid:0", next(v for v in outs.values() if v.shape[1] <= 4))
        bbox = outs.get("output_bbox/BiasAdd:0", next(v for v in outs.values() if v.shape[1] >= 12))

        decoded = decode_tao_detectnet_v2(cov, bbox, class_idx=self.class_idx, conf_thresh=thresh, nms_iou=0.45)
        sx, sy = w / float(self.dst_w), h / float(self.dst_h)
        results: list[DetectedFace] = []
        for box_net, score in decoded:
            box_src = (
                float(np.clip(box_net[0] * sx, 0.0, w)),
                float(np.clip(box_net[1] * sy, 0.0, h)),
                float(np.clip(box_net[2] * sx, 0.0, w)),
                float(np.clip(box_net[3] * sy, 0.0, h)),
            )
            k5 = self.refiner.predict_5pt(img_bgr, box_src) if self.refiner else None
            results.append(DetectedFace(bbox=box_src, score=score, landmarks=k5))
        return results

    def benchmark_batch(self, batch_size: int = 1, iters: int = 50, warmup: int = 10) -> float:
        return _time_ort_session(self.sess, (batch_size, 3, self.dst_h, self.dst_w), iters, warmup)


class YuNetFaceDetector(BaseFaceDetector):
    """OpenCV Zoo YuNet (MIT code / WIDER FACE Flag 9 reference, 5-pt landmarks)."""

    name = "opencv_yunet_flag9"

    def __init__(self, onnx_path: str | Path, gpu: bool = True, default_conf: float = 0.35):
        self.sess = _make_ort_session(onnx_path, gpu=gpu)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.dst_h = int(shape[2]) if len(shape) == 4 and isinstance(shape[2], int) else 640
        self.dst_w = int(shape[3]) if len(shape) == 4 and isinstance(shape[3], int) else 640
        self.default_conf = default_conf

    def detect(self, img_bgr: np.ndarray, conf_thresh: Optional[float] = None) -> list[DetectedFace]:
        import cv2

        thresh = self.default_conf if conf_thresh is None else conf_thresh
        h, w = img_bgr.shape[:2]
        scale, pad_x, pad_y = letterbox_params(w, h, self.dst_w, self.dst_h)
        resized = cv2.resize(img_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.dst_h, self.dst_w, 3), dtype=np.uint8)
        x0, y0 = int(round(pad_x)), int(round(pad_y))
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        blob = np.transpose(canvas.astype(np.float32), (2, 0, 1))[None, ...]  # YuNet uses BGR 0..255

        outs = {o.name: v for o, v in zip(self.sess.get_outputs(), self.sess.run(None, {self.input_name: blob}))}
        decoded = decode_yunet(outs, input_wh=(self.dst_w, self.dst_h), conf_thresh=thresh, nms_iou=0.30)
        results: list[DetectedFace] = []
        for box_net, score, kpts_net in decoded:
            ubox = unmap_coords(box_net.reshape(2, 2), scale, pad_x, pad_y, w, h).reshape(4)
            ukpts = unmap_coords(kpts_net, scale, pad_x, pad_y, w, h)
            results.append(DetectedFace(bbox=tuple(float(v) for v in ubox), score=score, landmarks=ukpts))
        return results

    def benchmark_batch(self, batch_size: int = 1, iters: int = 50, warmup: int = 10) -> float:
        return _time_ort_session(self.sess, (batch_size, 3, self.dst_h, self.dst_w), iters, warmup)


class RfDetrFaceDetector(BaseFaceDetector):
    """RF-DETR ONNX face detector + optional MediaPipe 5-pt landmark refiner."""

    name = "rfdetr_nano_wider_ref"

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        onnx_path: str | Path,
        landmark_onnx: Optional[str | Path] = None,
        gpu: bool = True,
        default_conf: float = 0.30,
        face_col: int = 0,
    ):
        self.sess = _make_ort_session(onnx_path, gpu=gpu)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.dst_h = int(shape[2]) if len(shape) == 4 and isinstance(shape[2], int) else 384
        self.dst_w = int(shape[3]) if len(shape) == 4 and isinstance(shape[3], int) else 384
        self.default_conf = default_conf
        self.face_col = face_col
        self.refiner = MediaPipeLandmarkRefiner(landmark_onnx, gpu=gpu) if landmark_onnx else None

    def detect(self, img_bgr: np.ndarray, conf_thresh: Optional[float] = None) -> list[DetectedFace]:
        import cv2

        thresh = self.default_conf if conf_thresh is None else conf_thresh
        h, w = img_bgr.shape[:2]
        resized = cv2.resize(img_bgr, (self.dst_w, self.dst_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = (rgb - self._MEAN) / self._STD
        blob = np.transpose(norm, (2, 0, 1))[None, ...]

        outs = self.sess.run(None, {self.input_name: blob})
        boxes = next(v for v in outs if v.shape[-1] == 4)
        logits = next(v for v in outs if v.shape[-1] != 4)
        decoded = decode_rfdetr_face(
            boxes, logits, img_wh=(w, h), face_col=self.face_col, conf_thresh=thresh, nms_iou=0.45
        )
        results: list[DetectedFace] = []
        for box_src, score in decoded:
            box_tuple = tuple(float(v) for v in box_src)
            k5 = self.refiner.predict_5pt(img_bgr, box_tuple) if self.refiner else None
            results.append(DetectedFace(bbox=box_tuple, score=score, landmarks=k5))
        return results

    def benchmark_batch(self, batch_size: int = 1, iters: int = 50, warmup: int = 10) -> float:
        return _time_ort_session(self.sess, (batch_size, 3, self.dst_h, self.dst_w), iters, warmup)


class ArcFaceOnnxEmbedder:
    """Direct ONNX Runtime embedder for AuraFace (glintr100.onnx) or buffalo_l (w600k_r50.onnx)."""

    def __init__(self, name: str, onnx_path: str | Path, gpu: bool = True):
        self.name = name
        self.sess = _make_ort_session(onnx_path, gpu=gpu)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name

    def embed_aligned(self, crop_112_bgr: np.ndarray) -> np.ndarray:
        import cv2

        if crop_112_bgr.shape[:2] != (112, 112):
            crop_112_bgr = cv2.resize(crop_112_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop_112_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        norm = (rgb - 127.5) / 127.5
        blob = np.transpose(norm, (2, 0, 1))[None, ...]
        feat = self.sess.run(None, {self.input_name: blob})[0][0]
        return l2_normalize(feat)

    def embed_face(self, img_bgr: np.ndarray, face: DetectedFace) -> Optional[np.ndarray]:
        import cv2

        if landmarks_plausible(face.landmarks, face.bbox):
            aligned = norm_crop_112(img_bgr, face.landmarks)
            return self.embed_aligned(aligned)
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in face.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img_bgr[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if ch != cw:
            side = max(ch, cw)
            top = (side - ch) // 2
            bottom = side - ch - top
            left = (side - cw) // 2
            right = side - cw - left
            crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_REPLICATE)
        return self.embed_aligned(crop)
