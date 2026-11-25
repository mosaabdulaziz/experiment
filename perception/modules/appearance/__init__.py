"""appearance: a re-identification embedding and a staff probability for each sampled person, off the hot path.

Staff are told by uniform, not face (a shop's staff wear masks), customers are kept apart by body re-ID. For every
person track the module samples one crop at most every `interval_s`, and a worker thread embeds the crops with a
re-ID model (`reid_spec`) and scores each one with a staff scorer (`staff_model`: "classifier", the trained
staff/customer ONNX, or "clip", the zero-shot fallback), then publishes one entry per sampled frame on
`iris:appearance:<cam>`: [{track_id, reid (base64 float16), staff_p, crop_path?, crop_quality?}, ...].

With `crops_dir` set the worker also saves a body crop of a track every `body_interval_s` (picture files for the
register's customer cards), world-writable because the sidecar runs as root and the consuming service is a host user.

The frame is copied off the GPU only when a person is due (the copy runs on the streaming thread, and doing it for
every frame with people slowed the whole pipeline), so the module asks for pixels itself. A full worker queue drops
the sample, never blocks the pipeline.

The people are noted in `transform`, like person_boxes: list the module before one that filters detections to sample
everyone the tracker reports.
"""
import logging
import os
from dataclasses import dataclass
from typing import Optional

from perception.engine.frame import FrameView
from perception.engine.model_manifest import ModelRef
from perception.engine.modules import Module, ModuleConfigError

from .worker import STAFF_MODELS, AppearanceWorker, build_staff_scorer, onnx_embedder, reid_batch, spec_by_name

logger = logging.getLogger("iris.sidecar")


@dataclass
class Settings:
    reid_spec: str = "tao_swin_t"
    staff_model: str = "clip"                   # classifier | clip
    staff_classifier: str = "/workspace/iris/perception/models/staff/staff_dinov2_s.onnx"
    clip_model: str = "/workspace/iris/perception/models/staff/vision_model.onnx"
    prompts: str = "/workspace/iris/perception/models/staff/prompts.json"
    interval_s: float = 1.0                     # a track is sampled at most this often
    min_height_px: int = 64                     # a shorter person (capture pixels) is too small to embed
    crops_dir: str = ""                         # save body crops under here; empty: none
    body_interval_s: float = 5.0

    def __post_init__(self) -> None:
        if self.staff_model not in STAFF_MODELS:
            raise ModuleConfigError(f"modules.appearance: staff_model {self.staff_model!r} is not one of "
                                    f"{list(STAFF_MODELS)}")
        try:
            spec_by_name(self.reid_spec)
        except ValueError as exc:
            raise ModuleConfigError(f"modules.appearance: {exc}") from None
        if self.interval_s <= 0 or self.body_interval_s <= 0:
            raise ModuleConfigError("modules.appearance: interval_s and body_interval_s must be positive")


class Appearance(Module):
    name = "appearance"
    camera_keyed_streams = True       # its worker writes iris:<stream>:<camera>, not role-scoped
    Settings = Settings

    @classmethod
    def model_files(cls, settings, every_variant: bool = False) -> list:
        refs = [ModelRef(spec_by_name(settings.reid_spec).path, f"reid_spec={settings.reid_spec}")]
        if settings.staff_model == "classifier" or every_variant:
            refs.append(ModelRef(settings.staff_classifier, "staff_classifier"))
        if settings.staff_model == "clip" or every_variant:   # the image tower and prompts (the text tower is offline)
            refs += [ModelRef(settings.clip_model, "clip_model"), ModelRef(settings.prompts, "prompts")]
        return refs

    def __init__(self, settings: Settings, services) -> None:
        super().__init__(settings, services)
        self.worker: Optional[AppearanceWorker] = None

    def on_built(self, pipeline) -> None:
        s = self.settings
        if s.crops_dir:
            # the consuming service (a host user) saves its alert snapshots under the crops root this container owns
            # (2026-10-05: it couldn't create the folder itself -- Permission denied, no alert ever got a picture)
            try:
                alerts_dir = os.path.join(s.crops_dir, "alerts")
                os.makedirs(alerts_dir, exist_ok=True)
                os.chmod(alerts_dir, 0o777)
            except OSError as exc:
                logger.warning("could not prepare %s for the register service's alert snapshots: %s", s.crops_dir, exc)
        staff, staff_desc = build_staff_scorer(s.staff_model, s.staff_classifier, s.clip_model, s.prompts)
        spec = spec_by_name(s.reid_spec)
        self.worker = AppearanceWorker(
            self.services.redis, onnx_embedder(spec.path, lambda crops: reid_batch(crops, spec), output=spec.output),
            staff, interval_s=s.interval_s, min_height_px=s.min_height_px, crops_dir=s.crops_dir or None,
            body_interval_s=s.body_interval_s)
        logger.info("Appearance worker built (reid=%s, staff model: %s, interval_s=%s)", spec.name, staff_desc,
                    s.interval_s)

    def start(self) -> None:
        if self.worker is not None:
            self.worker.start()

    # hot path ---------------------------------------------------------------------------------------------
    def transform(self, frame: FrameView) -> None:
        if not frame.need_meta:
            return
        sx, sy = frame.capture_scale()
        boxes = [(o.left * sx, o.top * sy, o.width * sx, o.height * sy, o.id) for o in frame.objects
                 if o.kind == "object" and o.label == "person" and o.draw]
        if boxes:
            frame.scratch[self.name] = {"boxes": boxes, "due": False}

    def wants_pixels(self, frame: FrameView) -> bool:
        note = frame.scratch.get(self.name)
        if self.worker is None or note is None:
            return False
        # Sampled at most once per track per interval, so only a frame with a track due is worth the copy.
        note["due"] = self.worker.wants(frame.stream_id, note["boxes"], frame.now)
        return note["due"]

    def on_pixels(self, frame: FrameView, bgr) -> None:
        note = frame.scratch.get(self.name)
        if self.worker is not None and note is not None and note["due"]:
            self.worker.submit(frame.stream_id, bgr, note["boxes"], frame.now)


MODULE = Appearance
