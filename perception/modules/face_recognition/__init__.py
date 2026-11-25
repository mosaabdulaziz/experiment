"""face_recognition: name the people whose faces face_detect found, against a gallery of registered faces.

Faces come from the frame's objects (kind "face", with their owner in `parent_id` and, when face_detect is asked for
them, landmarks). Each meta frame that has faces hands the frame's clean pixels to a recognition thread of its own,
which embeds each face with ArcFace, matches it against the gallery in Redis (`gallery_key`, a hash the app fills when
an employee is registered) and votes over a short window per tracked person so a name does not flicker. Results are
published on `iris:face_rec:<cam>`; the engine's preview writes a recognised name over the person's (or face's) box.

Name stability (anti-flicker) is tuned by vote_ttl, min_votes, majority, lock_seconds, reconfirm_seconds and
max_jump (a locked name whose track teleports farther than this is a different person). `margin` is the open-set rule:
the best identity must beat the runner-up by this much, else "unknown".

`capture` turns on CCTV face capture: faces recognised as nobody are saved, novel ones only, so the gallery can be
filled from the camera itself (it consumes the face detector's output, so a detector that idles while the gallery is
empty must be told capture is on). `enhancement` upscales small faces before embedding.

Recognition runs on the GPU when `ctx_id` >= 0 and says so loudly when it did not get there (a mismatched cuDNN falls
back to the CPU, 16x slower).
"""
import logging
from dataclasses import dataclass, field, fields
from typing import Optional, Union

from perception.engine.frame import FrameOutput, FrameView
from perception.engine.model_manifest import ModelRef
from perception import insightface_pack
from perception.engine.modules import Module, ModuleConfigError, ModuleStartupError

logger = logging.getLogger("iris.sidecar")


@dataclass
class Capture:
    min_pixels: int = 24                        # a smaller face is not saved
    min_det_confidence: float = 0.25            # nor one the detector was less sure of
    novelty_threshold: float = 0.25             # cosine distance: skip a face too like one already captured
    dir: str = "/workspace/iris/data/face_captures"


@dataclass
class Enhancement:
    backend: str = "preprocess"                 # preprocess (OpenCV only) | gfpgan | none
    min_face_size: int = 48                     # faces smaller than this (shorter side, px) get upscaled
    target_size: int = 256                      # upscale target
    weight: float = 0.5                         # restoration weight (gfpgan)


def _nested(cls, value, name):
    if value is None or isinstance(value, cls):
        return value
    if not isinstance(value, dict):
        raise ModuleConfigError(f"modules.face_recognition.{name}: expected a mapping, got {type(value).__name__}")
    unknown = sorted(set(value) - {f.name for f in fields(cls)})
    if unknown:
        raise ModuleConfigError(f"modules.face_recognition.{name}: unknown setting(s) {unknown}; "
                                f"known: {sorted(f.name for f in fields(cls))}")
    return cls(**value)


@dataclass
class Settings:
    model: str = "auraface"
    det_size: list = field(default_factory=lambda: [640, 640])
    det_thresh: float = 0.25
    threshold: float = 0.28                     # cosine similarity (0-1, lower = more permissive; calibrated for auraface)
    margin: float = 0.05
    ctx_id: int = 0                             # GPU device id; -1 = CPU
    gallery_key: str = "iris:face:gallery"
    vote_ttl: float = 4.0
    min_votes: int = 2
    majority: float = 0.6
    lock_seconds: float = 5.0
    reconfirm_seconds: float = 3.0
    max_jump: float = 0.35
    align: bool = True                          # straighten faces with their landmarks (face_detect: landmarks)
    align_debug_samples: int = 20
    align_debug_dir: str = "/workspace/iris/data/aligned_debug"
    capture: Optional[Union[Capture, dict]] = None
    enhancement: Optional[Union[Enhancement, dict]] = None

    def __post_init__(self) -> None:
        self.capture = _nested(Capture, self.capture, "capture")
        self.enhancement = _nested(Enhancement, self.enhancement, "enhancement")


class FaceRecognition(Module):
    name = "face_recognition"
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
        self.recognizer = None

    def on_built(self, pipeline) -> None:
        """Built once the pipeline's elements exist (models load after the detectors are in place, as they always
        did). A recogniser that cannot start is logged and the module stays inert: detection, drawing and everything
        else still run. Except a missing model pack: that refuses startup (it is never downloaded at run time)."""
        try:
            insightface_pack.root(self.settings.model)
        except insightface_pack.InsightFacePackMissing as exc:
            raise ModuleStartupError(f"modules.face_recognition could not start: {exc}") from exc
        try:
            from .recognizer import FaceCaptureStore, FaceRecognizer
        except ImportError:
            logger.warning("face_recognition is configured but insightface is not installed")
            return
        s, svc = self.settings, self.services
        try:
            enhancer = None
            if s.enhancement is not None:
                try:
                    from .enhancer import FaceEnhancer
                    enhancer = FaceEnhancer({
                        "face_enhancement_enabled": True, "face_enhancement_backend": s.enhancement.backend,
                        "face_enhancement_min_face_size": s.enhancement.min_face_size,
                        "face_enhancement_target_size": s.enhancement.target_size,
                        "face_enhancement_weight": s.enhancement.weight})
                except Exception as exc:  # noqa: BLE001
                    logger.warning("FaceEnhancer init failed (non-fatal, continuing without enhancement): %s", exc)
            self.recognizer = FaceRecognizer(
                redis_client=svc.redis, model_name=s.model, det_size=tuple(s.det_size), det_thresh=s.det_thresh,
                rec_thresh=s.threshold, rec_margin=s.margin, ctx_id=s.ctx_id, gallery_key=s.gallery_key,
                enhancer=enhancer, track_vote_ttl=s.vote_ttl, track_min_votes=s.min_votes,
                track_majority=s.majority, track_lock_seconds=s.lock_seconds, track_max_jump=s.max_jump,
                track_reconfirm_seconds=s.reconfirm_seconds, align_enabled=s.align,
                align_debug_samples=s.align_debug_samples, align_debug_dir=s.align_debug_dir)
            if s.capture is not None:
                self.recognizer._init_capture(
                    capture_store=FaceCaptureStore(svc.redis, s.capture.dir), min_pixels=s.capture.min_pixels,
                    min_det_confidence=s.capture.min_det_confidence, novelty_threshold=s.capture.novelty_threshold)
            logger.info("Face recognition initialized (model=%s, ctx_id=%d, capture=%s)", s.model, s.ctx_id,
                        s.capture is not None)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialize face recognition: %s", exc)
            self.recognizer = None

    def start(self) -> None:
        if self.recognizer is not None:
            self.recognizer.start()
            logger.info("Face recognizer thread started")

    def stop(self) -> None:
        if self.recognizer is not None:
            self.recognizer.stop()

    def gauges(self) -> dict:
        if self.recognizer is None:
            return {}
        return {"recognition_pending": lambda: self.recognizer._queue.qsize()}

    # hot path ---------------------------------------------------------------------------------------------
    @staticmethod
    def _faces(frame: FrameView) -> list:
        return [o for o in frame.objects if o.kind == "face"]

    def wants_pixels(self, frame: FrameView) -> bool:
        return frame.need_meta and self.recognizer is not None and bool(self._faces(frame))

    def analyze(self, frame: FrameView, out: FrameOutput) -> None:
        rec = self.recognizer
        if rec is None:
            return
        sx, sy = frame.capture_scale()
        if frame.need_meta:
            # Tell the recognizer which person tracks are still being followed, so a confirmed name stays on a body
            # while it is tracked -- face turned away or not. Capture coords, to match the face boxes it works in.
            alive = [(o.left * sx, o.top * sy, o.width * sx, o.height * sy, o.id) for o in frame.objects
                     if o.kind == "object" and o.label == "person" and o.raw is not None]
            if alive:
                rec.note_alive_tracks(frame.stream_id, alive)
        # A recognised name is drawn instead of the label, once per name: on the face box when there is one, else on
        # the person's own (faces are drawn first), so the name stays on a body while it is tracked.
        names = rec.snapshot_names(frame.stream_id)
        if names:
            written = set()
            for o in sorted((o for o in frame.objects if o.draw), key=lambda o: o.layer):
                name = names.get(o.id) if o.id != -1 else None
                if name:
                    # a name already written on another box leaves this one without any label
                    out.annotate(o, text="" if name in written else name)
                    written.add(name)
        faces = self._faces(frame)
        if frame.need_meta and faces and frame.bgr is not None:
            face_dicts = [{"left": int(f.left * sx), "top": int(f.top * sy), "width": int(f.width * sx),
                           "height": int(f.height * sy), "parent_id": f.parent_id, "confidence": f.conf,
                           "landmarks": f.attrs.get("landmarks")} for f in faces]
            stream_id, pts = frame.stream_id, frame.pts
            out.after_publish(lambda clean: rec.enqueue(clean, face_dicts, stream_id, pts))


MODULE = FaceRecognition
