"""customer_faces: quality-gated face pictures with embeddings, for a register of visiting customers.

For each face face_detect reports, a worker thread of this module applies the quality gates (a shorter side of at
least `min_pixels` capture pixels, a yaw from the five landmarks within `max_yaw`, the Laplacian sharpness of the
aligned 112x112 face at least `min_sharpness`), embeds the aligned face with ArcFace, saves the padded face crop under
`crops_dir` and publishes it on `iris:customer_face:<cam>`. At most one face per tracked person every `interval_s`.

Staff are not filtered here: the consuming service drops the faces of tracks that do not belong to a customer's visit.
Crop files are world-writable because the sidecar runs as root and that service, a host user, deletes the ones it does
not keep.

It needs the landmarks, so face_detect must run with `landmarks: true`. It loads its own ArcFace model: the embedding
must not wait behind (or take time from) the recognition thread.
"""
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from perception.engine.frame import FrameOutput, FrameView
from perception.engine.model_manifest import ModelRef
from perception import insightface_pack
from perception.engine.modules import Module, ModuleConfigError, ModuleStartupError

from .collector import CustomerFaceCollector

logger = logging.getLogger("iris.sidecar")


@dataclass
class Settings:
    crops_dir: str = "/workspace/iris/data/barber/crops"
    min_pixels: int = 35
    min_sharpness: float = 60.0
    max_yaw: float = 0.35
    interval_s: float = 1.0
    model: str = "auraface"                    # the ArcFace pack the embedding comes from
    ctx_id: int = 0
    queue_size: int = 8                         # frames waiting for the worker; the oldest is dropped when full

    def __post_init__(self) -> None:
        if self.queue_size < 1:
            raise ModuleConfigError("modules.customer_faces: queue_size must be >= 1")
        if self.max_yaw <= 0:
            raise ModuleConfigError("modules.customer_faces: max_yaw must be positive")


class CustomerFaces(Module):
    name = "customer_faces"
    camera_keyed_streams = True       # its worker writes iris:<stream>:<camera>, not role-scoped
    requires = ("face_detect",)
    Settings = Settings

    @classmethod
    def model_files(cls, settings, every_variant: bool = False) -> list:
        # insightface's FaceAnalysis opens every *.onnx of the pack (app/face_analysis.py:46-60), not only the two
        # allowed_modules keeps, so the whole directory is what loads: the repo's DVC copy (perception/insightface_pack)
        return [ModelRef(insightface_pack.pack_rel(settings.model), "model", "pack")]

    def __init__(self, settings: Settings, services) -> None:
        super().__init__(settings, services)
        self.collector = None
        self._queue: queue.Queue = queue.Queue(maxsize=settings.queue_size)
        self._thread = None
        self._running = False

    def on_built(self, pipeline) -> None:
        s = self.settings
        try:                                    # never downloaded at run time: a missing pack refuses startup
            pack_root = insightface_pack.root(s.model)
        except insightface_pack.InsightFacePackMissing as exc:
            raise ModuleStartupError(f"modules.customer_faces could not start: {exc}") from exc
        # the consuming service (a host user) saves its alert snapshots under the crops root this container owns
        # (2026-10-05: it couldn't create the folder itself -- Permission denied, no alert ever got a picture)
        try:
            alerts_dir = os.path.join(s.crops_dir, "alerts")
            os.makedirs(alerts_dir, exist_ok=True)
            os.chmod(alerts_dir, 0o777)
        except OSError as exc:
            logger.warning("could not prepare %s for the register service's alert snapshots: %s", s.crops_dir, exc)
        from insightface.app import FaceAnalysis
        from insightface.utils import face_align
        # the detection pack has to stay loaded (this insightface version asserts it) but is never run: face_detect
        # already located every face
        app = FaceAnalysis(name=s.model, root=pack_root, allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=s.ctx_id, det_size=(640, 640))
        model = app.models["recognition"]

        def embed(aligned: np.ndarray):
            feats = model.get_feat(aligned)
            return None if feats is None or len(feats) == 0 else np.asarray(feats[0], np.float32)

        self.collector = CustomerFaceCollector(
            self.services.redis, s.crops_dir,
            align=lambda frame, lmk: face_align.norm_crop(frame, landmark=lmk, image_size=112), embed=embed,
            min_pixels=s.min_pixels, min_sharpness=s.min_sharpness, max_yaw=s.max_yaw, interval_s=s.interval_s)
        logger.info("Customer face capture on (min %d px, crops under %s)", s.min_pixels, s.crops_dir)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="customer-faces", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while self._running:
            try:
                stream_id, frame, faces = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            now = time.time()
            for face in faces:
                try:
                    self.collector.offer(stream_id, frame, face, now)
                except Exception:  # noqa: BLE001
                    logger.exception("customer face capture failed (skipped)")

    # hot path ---------------------------------------------------------------------------------------------
    @staticmethod
    def _faces(frame: FrameView) -> list:
        return [o for o in frame.objects if o.kind == "face"]

    def wants_pixels(self, frame: FrameView) -> bool:
        return frame.need_meta and self.collector is not None and bool(self._faces(frame))

    def analyze(self, frame: FrameView, out: FrameOutput) -> None:
        faces = self._faces(frame)
        if self.collector is None or not (frame.need_meta and faces and frame.bgr is not None):
            return
        sx, sy = frame.capture_scale()
        face_dicts = [{"left": int(f.left * sx), "top": int(f.top * sy), "width": int(f.width * sx),
                       "height": int(f.height * sy), "parent_id": f.parent_id, "confidence": f.conf,
                       "landmarks": f.attrs.get("landmarks")} for f in faces]
        stream_id = frame.stream_id

        def hand_over(clean):
            try:
                self._queue.put_nowait((stream_id, clean, face_dicts))
            except queue.Full:
                logger.debug("customer face queue full, dropping a frame")
        out.after_publish(hand_over)


MODULE = CustomerFaces
