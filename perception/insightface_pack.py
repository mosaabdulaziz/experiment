"""Where the InsightFace model pack lives: in the repo, versioned with DVC, never downloaded at run time.

insightface's FaceAnalysis(name, root=...) loads every *.onnx of `<root>/models/<name>/`, and downloads the pack from
GitHub when that folder does not exist (insightface 1.0.1 and 2.0, utils/storage.py `download`). Its default root is
~/.insightface. Every runtime here passes `root=insightface_pack.root(name)` instead:

    from perception import insightface_pack
    app = FaceAnalysis(name="auraface", root=insightface_pack.root("auraface"), allowed_modules=[...])

root() first checks that the pack's files are all present, so insightface always finds its folder (no download) and
never loads a partial pack. A missing file raises InsightFacePackMissing, whose message is the `dvc pull` that fixes
it; the sidecar modules turn it into a startup refusal, the app into a 503, the CLIs into an exit (cli_root()).

The root is found from this file, not the working directory, so it is right in every layout: the repo
(/workspace/iris in the sidecar, which mounts the repo), the app container (/app, which mounts ./perception at
/app/perception) and the enrollment containers (./perception at /workspace/iris/perception).

Standard library only: the app and the CLIs import it without the engine's dependencies.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

ROOT = Path(__file__).resolve().parent / "models" / "insightface"     # insightface's `root`
ROOT_REL = "perception/models/insightface"

DEFAULT_PACK = "auraface"

# The files of each pack this repo versions.
PACKS = {
    "auraface": ("glintr100.onnx",),
    "buffalo_l": ("1k3d68.onnx", "2d106det.onnx", "det_10g.onnx", "genderage.onnx", "w600k_r50.onnx"),
}


class InsightFacePackMissing(FileNotFoundError):
    """The pack (or part of it) is not on disk; the message says which `dvc pull` brings it."""


class _FaceDetLiteDetectionModel:
    """Drop-in InsightFace detection model backed by face_det_lite.onnx (lazy ONNX session)."""

    taskname = "detection"
    DETECTOR_ONNX = Path(__file__).resolve().parent / "models" / "DeepStream-Yolo-Face" / "face_det_lite.onnx"
    DETECTOR_REL = "perception/models/DeepStream-Yolo-Face/face_det_lite.onnx"
    INPUT_W = 640
    INPUT_H = 480
    NMS_IOU = 0.45

    def __init__(self, onnx_path: Path | None = None) -> None:
        self.onnx_path = Path(onnx_path) if onnx_path is not None else self.DETECTOR_ONNX
        self.session: Any = None
        self.input_name: str = "images"
        self.ctx_id: int = 0
        self.det_thresh: float = 0.35
        self.input_size: tuple[int, int] = (self.INPUT_W, self.INPUT_H)

    def prepare(
        self,
        ctx_id: int = 0,
        det_thresh: float = 0.5,
        input_size: tuple[int, int] = (640, 640),
        **kwargs: Any,
    ) -> None:
        self.ctx_id = int(ctx_id)
        # Enrollment callers pass SCRFD's legacy 0.4/0.5 defaults; map those to face_det_lite's
        # calibrated 0.35 operating point while honoring explicit stricter (>0.5) or looser (<0.35) values.
        thresh = float(det_thresh)
        self.det_thresh = 0.35 if 0.35 < thresh <= 0.5 else thresh
        if input_size is not None:
            self.input_size = (int(input_size[0]), int(input_size[1]))

    def _ensure_session(self) -> Any:
        if self.session is not None:
            return self.session
        if not self.onnx_path.is_file():
            raise InsightFacePackMissing(
                f"the enrollment face detector is missing ({self.DETECTOR_REL}); it is versioned with DVC and "
                f"never downloaded at run time: run `dvc pull {self.DETECTOR_REL}.dvc` from the repo root "
                f"(or `just data-pull`)"
            )
        import onnxruntime as ort

        avail = set(ort.get_available_providers())
        if self.ctx_id >= 0 and "CUDAExecutionProvider" in avail:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        return self.session

    def detect(
        self,
        img: np.ndarray,
        max_num: int = 0,
        metric: str = "default",
    ) -> tuple[np.ndarray, np.ndarray]:
        import cv2
        import numpy as np

        if img is None or img.size == 0 or img.shape[0] <= 0 or img.shape[1] <= 0:
            return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 5, 2), dtype=np.float32)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.ndim != 3 or img.shape[2] != 3:
            return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 5, 2), dtype=np.float32)

        sess = self._ensure_session()
        h, w = img.shape[:2]
        dst_w, dst_h = self.INPUT_W, self.INPUT_H
        scale = min(dst_w / max(1, w), dst_h / max(1, h))
        pad_x = (dst_w - w * scale) / 2.0
        pad_y = (dst_h - h * scale) / 2.0

        resized = cv2.resize(
            img,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_LINEAR,
        )
        canvas = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
        x0, y0 = int(round(pad_x)), int(round(pad_y))
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])

        raw = np.asarray(sess.run(None, {self.input_name: blob})[0], dtype=np.float32)
        if raw.ndim == 3:
            raw = raw[0]

        scores = raw[:, 4]
        keep_mask = scores >= self.det_thresh
        if not np.any(keep_mask):
            return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 5, 2), dtype=np.float32)
        raw = raw[keep_mask]
        scores = scores[keep_mask]

        boxes = raw[:, :4]
        valid = (boxes[:, 2] - boxes[:, 0] >= 1.0) & (boxes[:, 3] - boxes[:, 1] >= 1.0)
        if not np.any(valid):
            return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 5, 2), dtype=np.float32)
        raw, boxes, scores = raw[valid], boxes[valid], scores[valid]

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
            order = rest[iou <= self.NMS_IOU]

        raw = raw[keep]
        bboxes = np.empty((len(keep), 5), dtype=np.float32)
        bboxes[:, 0] = np.clip((raw[:, 0] - pad_x) / scale, 0.0, float(w))
        bboxes[:, 1] = np.clip((raw[:, 1] - pad_y) / scale, 0.0, float(h))
        bboxes[:, 2] = np.clip((raw[:, 2] - pad_x) / scale, 0.0, float(w))
        bboxes[:, 3] = np.clip((raw[:, 3] - pad_y) / scale, 0.0, float(h))
        bboxes[:, 4] = raw[:, 4]

        kpss = np.empty((len(keep), 5, 2), dtype=np.float32)
        for p in range(5):
            kpss[:, p, 0] = np.clip((raw[:, 5 + p * 2] - pad_x) / scale, 0.0, float(w))
            kpss[:, p, 1] = np.clip((raw[:, 6 + p * 2] - pad_y) / scale, 0.0, float(h))

        if max_num > 0 and bboxes.shape[0] > max_num:
            b_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            img_center = h // 2, w // 2
            offsets = np.vstack(
                [
                    (bboxes[:, 0] + bboxes[:, 2]) / 2.0 - img_center[1],
                    (bboxes[:, 1] + bboxes[:, 3]) / 2.0 - img_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            values = b_area if metric == "max" else (b_area - offset_dist_squared * 2.0)
            bindex = np.argsort(values)[::-1][:max_num]
            bboxes = bboxes[bindex, :]
            kpss = kpss[bindex, :]

        return bboxes, kpss


def _install_auraface_shim() -> None:
    """Allow real insightface.app.FaceAnalysis to initialize the recognition-only auraface pack."""
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        return

    if not getattr(FaceAnalysis, "__module__", "").startswith("insightface."):
        return
    if getattr(FaceAnalysis, "_auraface_shim_installed", False):
        return

    orig_init = FaceAnalysis.__init__

    def _auraface_init(
        self: Any,
        name: str = DEFAULT_PACK,
        root: str = "~/.insightface",
        allowed_modules: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if name == "auraface":
            from insightface.model_zoo import model_zoo

            self.models = {}
            self.addons = {}
            self.liveness_mode = kwargs.pop("liveness_mode", "normal")
            self.liveness_threshold = kwargs.pop("liveness_threshold", 0.8)
            self.model_package = None
            self.model_dir = str(Path(root).expanduser() / "models" / name)
            onnx_files = sorted(Path(self.model_dir).glob("*.onnx"))
            for onnx_file in onnx_files:
                model = model_zoo.get_model(str(onnx_file), **kwargs)
                if model is None:
                    continue
                if allowed_modules is not None and model.taskname not in allowed_modules:
                    del model
                    continue
                if model.taskname not in self.models:
                    self.models[model.taskname] = model
                else:
                    del model
            if "detection" not in self.models and (allowed_modules is None or "detection" in allowed_modules):
                self.models["detection"] = _FaceDetLiteDetectionModel()
            self.det_model = self.models.get("detection")
            return
        orig_init(self, name=name, root=root, allowed_modules=allowed_modules, **kwargs)

    FaceAnalysis.__init__ = _auraface_init
    FaceAnalysis._auraface_shim_installed = True


_install_auraface_shim()


def pack_dir(name: str = DEFAULT_PACK) -> Path:
    return ROOT / "models" / name


def pack_rel(name: str = DEFAULT_PACK) -> str:
    """The pack's folder relative to the repo root (what model manifests list)."""
    return f"{ROOT_REL}/models/{name}"


def missing(name: str = DEFAULT_PACK) -> list[str]:
    """The pack's files that are not on disk (repo-relative). An unknown pack name is missing as a whole."""
    files = PACKS.get(name)
    if files is None:
        return [pack_rel(name) + "/"]
    return [f"{pack_rel(name)}/{f}" for f in files if not (pack_dir(name) / f).is_file()]


def root(name: str = DEFAULT_PACK) -> str:
    """insightface's `root` for pack `name`, once every file of the pack is present; InsightFacePackMissing otherwise."""
    gone = missing(name)
    if gone:
        if name not in PACKS:
            raise InsightFacePackMissing(
                f"InsightFace pack {name!r} is not versioned in this repo (known: {', '.join(PACKS)}); add it under "
                f"{ROOT_REL}/models/{name}/ with `dvc add`, one pointer per file")
        pulls = " ".join(f"{g}.dvc" for g in gone)
        raise InsightFacePackMissing(
            f"the InsightFace {name} pack is missing {len(gone)} file(s) ({', '.join(Path(g).name for g in gone)} in "
            f"{pack_rel(name)}); it is versioned with DVC and never downloaded at run time: run `dvc pull {pulls}` "
            f"from the repo root (or `just data-pull`)")
    if name == "auraface":
        _install_auraface_shim()
    return str(ROOT)


def cli_root(name: str = DEFAULT_PACK) -> str:
    """root() for a command-line tool: a missing pack exits with the message instead of a traceback."""
    try:
        return root(name)
    except InsightFacePackMissing as exc:
        raise SystemExit(f"error: {exc}") from None
