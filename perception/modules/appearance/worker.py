"""Body appearance for the barber use case: a re-identification embedding
and a zero-shot "is this staff?" probability per sampled person crop.

Models (all ONNX, run with onnxruntime-gpu):
  * Re-ID, one ReidSpec per candidate (256x128 RGB, per-model mean/std on
    0..255, optional letterbox to 1:2):
      - TAO ReIdentificationNet ResNet50 (models/tracker/resnet50_market1501_aicity156.onnx,
        `just barber-reid-model`): DeepStream's own preprocessing, ImageNet
        mean with a single std of 0.226*255; 256-d.
      - TAO ReIdentificationNet Swin-T (models/reid/..., `just barber-reid-model`):
        mean/std 127.5; 256-d.
      - CLIP-ReID ViT-B/16 MSMT17 (models/reid/clipreid_vitb16_msmt17.onnx,
        products/barber/dev/appearance/export_clipreid.py): mean/std 127.5; 1280-d.
  * CLIP ViT-B/32 image tower (perception/models/staff/vision_model.onnx,
    `just barber-staff-model`): 224x224 RGB, CLIP mean/std on 0..1, 512-d
    image_embeds. The text side is precomputed once into prompts.json
    (products/barber/dev/appearance/make_prompts.py), so the sidecar needs
    no tokenizer.
  * Staff classifier (perception/models/staff/staff_dinov2_s.onnx + .json, project A):
    DINOv2 ViT-S/14 fine-tuned on labelled crops of this shop; 252x126 RGB
    stretched, ImageNet mean/std; outputs P(staff) directly. Selected by
    appearance_staff_model: classifier (CLIP stays as the fallback).
Staff are recognized by uniform, not face: the shop's staff wear masks.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from perception.lib.imaging import MIN_CROP_PX, crop, decode_embedding, encode_embedding, l2n, save_jpeg  # noqa: F401

logger = logging.getLogger("sidecar.appearance")
APPEARANCE_STREAM_PREFIX = "iris:appearance:"

REID_W, REID_H = 128, 256
REID_MEAN = np.array([123.675, 116.28, 103.53], np.float32)   # RGB
REID_STD = np.float32(57.63)                                  # 0.226 * 255, as DeepStream's config
CLIP_SIZE = 224
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
CLIP_LOGIT_SCALE = 100.0
CLASSIFIER_HW = (252, 126)   # height, width: the staff classifier's input (perception/models/staff/staff_dinov2_s.json)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # RGB, on 0..1
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
MODELS = "/workspace/iris/models"


@dataclass(frozen=True)
class ReidSpec:
    """A re-ID model and its input normalisation (mean/std on 0..255 RGB).
    letterbox pads the crop with black to 1:2 (w:h) instead of stretching it."""
    name: str
    path: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    letterbox: bool = False
    output: str | None = None


TAO_RESNET50 = ReidSpec("tao_resnet50", f"{MODELS}/tracker/resnet50_market1501_aicity156.onnx",
                        tuple(float(m) for m in REID_MEAN), (57.63, 57.63, 57.63), output="fc_pred")
TAO_SWIN_T = ReidSpec("tao_swin_t", f"{MODELS}/reid/swin_tiny_market1501_aicity156_featuredim256.onnx",
                      (127.5, 127.5, 127.5), (127.5, 127.5, 127.5), output="fc_pred")
CLIPREID_VITB16 = ReidSpec("clipreid_vitb16", f"{MODELS}/reid/clipreid_vitb16_msmt17.onnx",
                           (127.5, 127.5, 127.5), (127.5, 127.5, 127.5), output="embedding")


def _fit_reid(c: np.ndarray, letterbox: bool) -> np.ndarray:
    if not letterbox:
        return cv2.resize(c, (REID_W, REID_H), interpolation=cv2.INTER_LINEAR)
    h, w = c.shape[:2]
    r = min(REID_W / w, REID_H / h)
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    out = np.zeros((REID_H, REID_W, 3), np.uint8)
    y0, x0 = (REID_H - nh) // 2, (REID_W - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out


def reid_batch(crops: list[np.ndarray], spec: ReidSpec | None = None) -> np.ndarray:
    """[N, 3, 256, 128] float32 input for `spec` (default: the TAO ResNet50)."""
    spec = spec or TAO_RESNET50
    mean, std = np.array(spec.mean, np.float32), np.array(spec.std, np.float32)
    out = []
    for c in crops:
        rgb = _fit_reid(c, spec.letterbox)[:, :, ::-1].astype(np.float32)
        out.append(((rgb - mean) / std).transpose(2, 0, 1))
    return np.ascontiguousarray(np.stack(out), dtype=np.float32)


def clip_batch(crops: list[np.ndarray]) -> np.ndarray:
    """Pad each crop to a square (black) before resizing: CLIP's usual centre
    crop would cut a standing person's head and feet off."""
    out = []
    for c in crops:
        h, w = c.shape[:2]
        s = max(h, w)
        sq = np.zeros((s, s, 3), np.uint8)
        y0, x0 = (s - h) // 2, (s - w) // 2
        sq[y0:y0 + h, x0:x0 + w] = c
        rgb = cv2.resize(sq, (CLIP_SIZE, CLIP_SIZE), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        rgb = rgb.astype(np.float32) / 255.0
        out.append(((rgb - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1))
    return np.ascontiguousarray(np.stack(out), dtype=np.float32)


def classifier_batch(crops: list[np.ndarray]) -> np.ndarray:
    """The staff classifier's input, exactly as it was trained
    (products/barber/dev/staff/train.py, letterbox off): the crop stretched to
    CLASSIFIER_HW with INTER_AREA -- no padding -- RGB, ImageNet-normalised."""
    h, w = CLASSIFIER_HW
    out = []
    for c in crops:
        rgb = cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255.0
        out.append(((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))
    return np.ascontiguousarray(np.stack(out), dtype=np.float32)


def class_probability(image_embeds: np.ndarray, prompts: dict, positive: str, negative: str) -> np.ndarray:
    """Softmax over two classes of CLIP's scaled cosine; a class's score is
    the mean cosine over its prompts. Returns P(positive), shape [N]."""
    img = l2n(image_embeds)
    scores = np.stack([(img @ l2n(np.asarray(prompts[k], np.float32)).T).mean(axis=1)
                       for k in (positive, negative)], axis=1) * CLIP_LOGIT_SCALE
    scores -= scores.max(axis=1, keepdims=True)
    e = np.exp(scores)
    return e[:, 0] / e.sum(axis=1)


def staff_probability(image_embeds: np.ndarray, prompts: dict) -> np.ndarray:
    return class_probability(image_embeds, prompts, "staff", "customer")


REID_SPECS = {s.name: s for s in (TAO_RESNET50, TAO_SWIN_T, CLIPREID_VITB16)}


def spec_by_name(name: str) -> ReidSpec:
    try:
        return REID_SPECS[name]
    except KeyError:
        raise ValueError(f"unknown re-ID spec {name!r}; valid names are {sorted(REID_SPECS)}") from None


@dataclass
class _Job:
    source_id: str
    track_ids: list[int]
    crops: list[np.ndarray]


class AppearanceWorker(threading.Thread):
    """Samples each person track at most once per `interval_s`, embeds the
    crops off the streaming thread, publishes one entry per frame to
    iris:appearance:<source_id>. submit() copies the crops, so the caller's
    frame can be reused at once; a full queue drops the job (counted), never
    blocks the pipeline."""

    def __init__(self, redis_client, reid, staff, interval_s: float = 1.0,
                 min_height_px: int = 64, maxlen: int = 512, queue_size: int = 8,
                 crops_dir: str | None = None, body_interval_s: float = 5.0):
        super().__init__(daemon=True, name="appearance")
        self.r, self.reid, self.staff = redis_client, reid, staff   # staff: crops -> P(staff), shape [n]
        self.interval_s, self.min_height_px, self.maxlen = interval_s, min_height_px, maxlen
        self.crops_dir, self.body_interval_s = crops_dir, body_interval_s
        self.queue: queue.Queue[_Job] = queue.Queue(maxsize=queue_size)
        self.dropped = 0
        self._last: dict[tuple[str, int], float] = {}
        self._last_body: dict[tuple[str, int], float] = {}

    def _due(self, source_id: str, h: float, tid: int, now: float) -> bool:
        return h >= self.min_height_px and now - self._last.get((source_id, tid), -1e9) >= self.interval_s

    def wants(self, source_id: str, boxes, now: float) -> bool:
        """Whether submit() would sample any of `boxes` now. The caller asks
        before copying the frame off the GPU: that copy runs on the streaming
        thread, and doing it for every frame with people -- when at most one
        per track per interval is used -- slowed the whole pipeline. False
        while the queue is full too, since submit() would drop the job."""
        if self.queue.full():
            return False
        return any(self._due(source_id, h, tid, now) for _l, _t, _w, h, tid in boxes)

    def submit(self, source_id: str, frame_bgr: np.ndarray, boxes, now: float) -> int:
        ids, crops = [], []
        for left, top, w, h, tid in boxes:
            if not self._due(source_id, h, tid, now):
                continue
            c = crop(frame_bgr, (left, top, w, h))
            if c is None:
                continue
            ids.append(int(tid))
            crops.append(c.copy())
        if not ids:
            return 0
        try:
            self.queue.put_nowait(_Job(source_id, ids, crops))
        except queue.Full:
            self.dropped += 1
            return 0
        for tid in ids:
            self._last[(source_id, tid)] = now
        if len(self._last) > 4096:   # forget tracks not sampled for a minute
            self._last = {k: t for k, t in self._last.items() if now - t < 60.0}
        return len(ids)

    def process(self, job: _Job, now: float | None = None) -> None:
        now = time.time() if now is None else now
        reid = l2n(self.reid(job.crops))
        staff = np.asarray(self.staff(job.crops), np.float32).reshape(-1)
        items = []
        for i, tid in enumerate(job.track_ids):
            item = {"track_id": tid, "reid": encode_embedding(reid[i]), "staff_p": round(float(staff[i]), 4)}
            key = (job.source_id, tid)
            if self.crops_dir and now - self._last_body.get(key, -1e9) >= self.body_interval_s:
                try:
                    item["crop_path"] = save_jpeg(self.crops_dir, "bodies", job.crops[i], now)
                    item["crop_quality"] = round(min(1.0, job.crops[i].shape[0] / 480.0), 3)
                    self._last_body[key] = now
                except OSError as exc:
                    logger.warning("[%s] body crop not written: %s", job.source_id, exc)
            items.append(item)
        if len(self._last_body) > 4096:
            self._last_body = {k: t for k, t in self._last_body.items() if now - t < 60.0}
        self.r.xadd(f"{APPEARANCE_STREAM_PREFIX}{job.source_id}", {"items": json.dumps(items)},
                    maxlen=self.maxlen, approximate=True)

    def run(self) -> None:
        while True:
            job = self.queue.get()
            try:
                self.process(job)
            except Exception:
                logger.exception("[%s] appearance batch failed (skipped)", job.source_id)


def onnx_embedder(path: str, preprocess, output: str | None = None):
    """A callable crops -> embeddings over one ONNX model (CUDA when present)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    idx = outs.index(output) if output in outs else 0
    return lambda crops: sess.run(None, {name: preprocess(crops)})[idx]


STAFF_MODELS = ("classifier", "clip")


def build_staff_scorer(staff_model: str, classifier_path: str, clip_model: str, prompts_path: str):
    """(crops -> P(staff) [n], description) for the chosen staff scorer: "classifier" = the trained staff/customer
    ONNX (sigmoid and temperature are inside the model), "clip" = the zero-shot CLIP image tower + prompts.json (the
    fallback). Anything else, or a missing classifier file, raises: never fall back to CLIP silently."""
    if staff_model not in STAFF_MODELS:
        raise ValueError(f"unknown staff_model {staff_model!r}; valid values are {list(STAFF_MODELS)}")
    if staff_model == "classifier":
        if not os.path.isfile(classifier_path):
            raise FileNotFoundError(f"staff classifier not found: {classifier_path}")
        run = onnx_embedder(classifier_path, classifier_batch, output="staff_p")
        return (lambda crops: np.asarray(run(crops), np.float32).reshape(-1)), f"classifier {classifier_path}"
    with open(prompts_path) as f:
        prompts = json.load(f)
    embed = onnx_embedder(clip_model, clip_batch, output="image_embeds")
    return (lambda crops: staff_probability(embed(crops), prompts)), f"clip {clip_model}"
