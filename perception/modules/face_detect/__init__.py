"""face_detect: find faces on the people the primary detector tracks, and decide whose face each one is.

A secondary detector (a YOLO face model) runs on every person's crop. This module adds it to the pipeline and turns
its raw output into one box per face:

- the owner: which person box a face belongs to when person boxes overlap (`owner_mode`: "legacy" only fills in faces
  the detector left unowned, "tracked" re-decides every face by position and continuity, see owner.py);
- the size filter: a face whose shorter side is under `min_pixels` capture pixels is dropped before it is drawn or
  recognised;
- five facial landmarks per face, in capture pixels, when `landmarks` is on (recognition straightens faces with them);
- the `faces` meta field: [[left, top, width, height, confidence, owner id], ...] in mux coordinates.

The faces are objects of kind "face", drawn first (layer 0) on their owner's track id, with the owner in `parent_id`;
face_recognition and customer_faces read them from the frame's objects.

Two ways to spend less GPU time on faces nobody reads:

- `sample_only`: run the face detector only on the frames whose metadata is published, and `regions_key`: only on
  people inside the door areas a service publishes to that Redis key (see regions.py). Needs the preview off and no
  process_fps throttle; otherwise the gate cannot see which frames matter and refuses to start.
- `idle_when_unused`: switch the detector off while the face gallery is empty (and nothing is capturing faces to fill
  it); see idle.py.

List it before `box_stabilizer` (it reads the people the detector reported) and before a module that filters
detections.
"""
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

from perception.engine.engine_path import gie_engine_path
from perception.engine.gie_gate import GieGate
from perception.engine.frame import FrameOutput, FrameView, ObjectView
from perception.engine.model_manifest import ModelRef
from perception.engine.modules import GieSpec, Module, ModuleConfigError, ModuleStartupError
from perception.engine.nvds import instance_mask

from .idle import SGIE_DISABLED_CLASS, SgieGalleryMonitor
from .owner import FaceOwnership, assign_legacy, crop_box
from .regions import FaceRegions

logger = logging.getLogger("iris.sidecar")

FACE_LABEL = "face"
ELEMENT = "sgie_face"


def face_landmarks_raw(mask):
    """The 5 facial landmarks (left eye, right eye, nose tip, left and right mouth corners) of one detection, exactly
    as the face network produced them: (15 floats (x, y, conf) x 5, net_w, net_h), or None.

    nvdsparseface_Yolo.cpp packs them into the object's instance-mask field (`mask`, an nvds.InstanceMask), in the
    face network's INPUT coords (the person crop). Used to straighten the face the way registration does
    (2026-09-12)."""
    if mask is None or mask.values.size < 15:
        return None
    return [float(v) for v in mask.values[:15]], float(mask.width), float(mask.height)


def landmarks_to_capture(raw, person_box, sx, sy):
    """Map face_landmarks_raw's network-input coords to CAPTURE pixels, or None. The face network ran on the person's
    box, scaled to fit net_w x net_h keeping its aspect ratio and centred with equal padding on both sides
    (config_sgie_face.txt: maintain-aspect-ratio=1, symmetric-padding=1): undo the padding and scale, offset by the
    box (mux coords), then scale mux -> capture. The face's own `parent` pointer arrives unset on this pipeline
    (checked live 2026-09-12), so person_box is the one the containment match assigned the face to."""
    if raw is None or person_box is None:
        return None
    pts, net_w, net_h = raw
    pl, pt, pw, ph = person_box
    if pw <= 0 or ph <= 0:
        return None
    scale = min(net_w / pw, net_h / ph)
    pad_x = (net_w - pw * scale) / 2.0
    pad_y = (net_h - ph * scale) / 2.0
    return [[(pl + (pts[i * 3] - pad_x) / scale) * sx,
             (pt + (pts[i * 3 + 1] - pad_y) / scale) * sy] for i in range(5)]


def _config_int(path: str, key: str, default: int) -> int:
    section = ""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                if line.startswith("[") and line.endswith("]"):
                    section = line[1:-1].strip().lower()
                    continue
                if section == "property" and "=" in line:
                    k, _, v = line.partition("=")
                    if k.strip().lower() == key:
                        return int(v.strip())
    except (OSError, ValueError):
        pass
    return default


@dataclass
class Settings:
    config: str = "/workspace/iris/perception/models/DeepStream-Yolo-Face/config_sgie_face.txt"
    # nvinfer serializes the engine beside the ONNX under a derived name; the deserialize probe must point at that
    # name or the engine is rebuilt on EVERY start. Empty: derived from the face SGIE config.
    engine: str = ""
    confidence_threshold: float = 0.18
    min_pixels: int = 24
    owner_mode: str = "legacy"                  # legacy | tracked
    owner_fallback: str = "highest-top"         # tracked only: highest-top | smallest
    landmarks: bool = False
    sample_only: bool = False
    regions_key: str = ""
    # {"gallery_key": <redis hash>, "capture": <bool>, "poll_seconds": 10.0}: idle while the gallery is empty, unless
    # something captures faces (a capturing recogniser fills the gallery from the detector, so it cannot idle).
    idle_when_unused: Optional[dict] = None

    def __post_init__(self) -> None:
        self.owner_mode = str(self.owner_mode).lower()
        self.owner_fallback = str(self.owner_fallback).lower()
        if self.owner_mode not in ("legacy", "tracked"):
            raise ModuleConfigError(f"modules.face_detect: owner_mode must be 'legacy' or 'tracked', got {self.owner_mode!r}")
        if self.owner_fallback not in ("highest-top", "smallest"):
            raise ModuleConfigError(
                f"modules.face_detect: owner_fallback must be 'highest-top' or 'smallest', got {self.owner_fallback!r}")
        if self.regions_key and not self.sample_only:
            raise ModuleConfigError("modules.face_detect: regions_key needs sample_only: the door areas are applied "
                                    "by the sampling gate, so without it faces would come from everywhere")
        if self.idle_when_unused is not None:
            unknown = set(self.idle_when_unused) - {"gallery_key", "capture", "poll_seconds"}
            if unknown or not self.idle_when_unused.get("gallery_key"):
                raise ModuleConfigError("modules.face_detect: idle_when_unused takes gallery_key (required), capture "
                                        f"and poll_seconds; got {self.idle_when_unused!r}")


class FaceDetect(Module):
    name = "face_detect"
    Settings = Settings

    @classmethod
    def model_files(cls, settings, every_variant: bool = False) -> list:
        return [ModelRef(settings.config, "config", "nvinfer")]

    def __init__(self, settings: Settings, services) -> None:
        super().__init__(settings, services)
        self.owner = (FaceOwnership(fallback=settings.owner_fallback) if settings.owner_mode == "tracked" else None)
        self.gate: Optional[GieGate] = None
        self.regions: Optional[FaceRegions] = None
        self._monitor: Optional[SgieGalleryMonitor] = None

    # build ------------------------------------------------------------------------------------------------
    def gies(self) -> list:
        s = self.settings
        if s.engine:
            engine = s.engine
        elif os.path.isfile(s.config):
            engine = gie_engine_path(
                s.config,
                _config_int(s.config, "batch-size", 16),
                _config_int(s.config, "gpu-id", 0),
            )
        else:
            engine = f"{os.path.dirname(s.config)}/face_det_lite.onnx_b16_gpu0_fp32.engine"
        return [GieSpec(name=ELEMENT, config_path=s.config, role="secondary", engine_path=engine,
                        batch_from_mux=False)]

    def on_built(self, pipeline) -> None:
        s, svc = self.settings, self.services
        el = pipeline.get(ELEMENT)
        if s.sample_only:
            self._install_gate(el)
        if s.regions_key and self.gate is None:
            # regions are applied by the gate's probe: without the gate, faces would come from everywhere
            raise ModuleStartupError("modules.face_detect: regions_key is set but the sampling gate could not be "
                                     "installed (see the warning above)")
        if s.idle_when_unused is not None:
            self._install_idle(el)

    def _install_gate(self, el) -> None:
        s, svc = self.settings, self.services
        self.gate = GieGate.install(el, ELEMENT, svc.engine, svc.mux_size, svc.index_to_id,
                                    object_allowed=self._in_regions if s.regions_key else None)
        if self.gate is not None and s.regions_key:
            # own client: the fetch runs on the GStreamer streaming thread, so it must never block -- short socket
            # timeouts and zero retries (the shared client has none); FaceRegions fails closed on errors
            import redis
            from redis.backoff import NoBackoff
            from redis.retry import Retry
            client = redis.Redis.from_url(svc.config["redis_url"], socket_timeout=0.2,
                                          socket_connect_timeout=0.2, retry_on_timeout=False,
                                          retry=Retry(NoBackoff(), 0))
            key = s.regions_key
            self.regions = FaceRegions(lambda: client.get(key))
            logger.info("%s only inside the face regions of %s", ELEMENT, key)

    def _in_regions(self, camera, left, top, width, height) -> bool:
        return self.regions is not None and self.regions.allows(camera, left, top, width, height)

    def _install_idle(self, el) -> None:
        """Empty gallery + capture off -> point infer-on-class-ids at a class that never appears: a secondary GIE runs
        the object path, whose gate that is, so it queues no inference (zero GPU) while buffers keep flowing. The
        first registered face (or capture on) restores the configured class list. See idle.py."""
        cfg = self.settings.idle_when_unused
        classes = el.get_property("infer-on-class-ids") or ""
        redis_client, key = self.services.redis, cfg["gallery_key"]

        def set_enabled(on):
            value = classes if on else SGIE_DISABLED_CLASS
            logger.info("%s infer-on-class-ids -> %r (%s)", ELEMENT, value, "enabled" if on else "disabled")
            el.set_property("infer-on-class-ids", value)

        self._monitor = SgieGalleryMonitor(
            hlen=lambda: redis_client.hlen(key), set_enabled=set_enabled,
            face_capture_enabled=bool(cfg.get("capture", False)), poll_seconds=float(cfg.get("poll_seconds", 10.0)))
        self._monitor.poll()                    # apply the boot state immediately

    def start(self) -> None:
        if self._monitor is not None:
            self.services.timer(float(self.settings.idle_when_unused.get("poll_seconds", 10.0)), self._monitor.poll)

    def gauges(self) -> dict:
        if self._monitor is None:
            return {}
        return {"sgie_mode": lambda: 1 if self._monitor.current_enabled else 0}

    # hot path ---------------------------------------------------------------------------------------------
    def extract(self, frame: FrameView):
        """With `landmarks` on, the face network's raw landmarks of each face box, as attrs["landmarks_raw"]."""
        if not (self.settings.landmarks and frame.analyze):
            return None

        def read(o: ObjectView) -> None:
            if o.label == FACE_LABEL:
                o.attrs["landmarks_raw"] = face_landmarks_raw(instance_mask(o.raw))
        return read

    def sample(self, frame: FrameView) -> Optional[bool]:
        """With the gate on, its decision (made before the face detector ran) is the sampling clock: a frame it
        sampled has faces, one it didn't has none."""
        return None if self.gate is None else self.gate.take(frame.pad_index, frame.buf_pts)

    def transform(self, frame: FrameView) -> None:
        s = self.settings
        claimed = [o for o in frame.objects if o.label == FACE_LABEL and o.conf >= s.confidence_threshold]
        if not claimed:
            return
        persons = [(o.left, o.top, o.width, o.height, o.id) for o in frame.objects
                   if o.kind == "object" and o.label == "person" and o.draw]
        alive = [(o.left, o.top, o.width, o.height, o.id) for o in frame.objects
                 if o.kind == "object" and o.label == "person" and o.raw is not None]
        faces, raw_landmarks = [], {}
        for o in claimed:
            o.draw = o.publish = False                 # redrawn below, on its owner's id
            faces.append((o.left, o.top, o.width, o.height, round(o.conf, 3), o.parent_id))
            raw = o.attrs.get("landmarks_raw")
            if raw is not None:
                raw_landmarks[(o.left, o.top, o.width, o.height)] = raw
        # Which person each face belongs to: this product's own rule -- "tracked" re-decides every face by position,
        # "legacy" only fills in faces the detector left unowned.
        faces = (self.owner.assign(frame.stream_id, faces, persons, frame.now) if self.owner is not None
                 else assign_legacy(faces, persons))
        # Face size filter, applied BEFORE rendering + recognition: a face whose smaller side (in mux coords, scaled to
        # capture) is below min_pixels is low quality and never reaches the stream or the recognizer.
        sx, sy = frame.capture_scale()
        if s.min_pixels > 0:
            kept = [f for f in faces if min(f[2] * sx, f[3] * sy) >= s.min_pixels]
            if len(kept) < len(faces):
                logger.debug("[%s] face filter: %d→%d (min_pixels=%d)", frame.stream_id, len(faces), len(kept),
                             s.min_pixels)
            faces = kept
        person_rect = {oid: (l, t, w, h) for l, t, w, h, oid in alive}

        def landmark_box(face, raw):
            # The raw points are in the coords of the box the face network RAN on, which need not be its owner (a
            # barber's face found in the box of the customer she leans over): crop_box finds that box.
            if raw is None:
                return None
            return (crop_box(face, alive, lambda p: landmarks_to_capture(raw, p[:4], 1.0, 1.0))
                    or person_rect.get(face[5]))

        for face in faces:
            fl, ft, fw, fh, fc, pid = face
            raw = raw_landmarks.get((fl, ft, fw, fh))
            frame.objects.append(ObjectView(
                id=pid or -1, label=FACE_LABEL, conf=fc, tracker_conf=fc, left=fl, top=ft, width=fw, height=fh,
                parent_id=pid, kind="face", publish=False, layer=0,
                attrs={"landmarks": landmarks_to_capture(raw, landmark_box(face, raw), sx, sy)}))
        frame.scratch[self.name] = faces

    def analyze(self, frame: FrameView, out: FrameOutput) -> None:
        # A face box carries no "face 0.80 #7" caption by default; face_recognition writes the person's name on it.
        for o in frame.objects:
            if o.label == FACE_LABEL and o.draw:
                out.annotate(o, text="", priority=-1)
        faces = frame.scratch.get(self.name)
        if faces:
            out.fields["faces"] = faces


MODULE = FaceDetect
