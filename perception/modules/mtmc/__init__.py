"""mtmc: one identity per vehicle across every camera, and a reading of each vehicle's best licence plate.

A vehicle tracker per camera (an IoU tracker over the detections the engine publishes) feeds a cross-camera
association: Re-ID embeddings of crops, the camera layout (which camera may follow which, with what travel time), an
optional vision-language model that confirms or refuses each proposed merge, and the direction a vehicle is facing
(entering or leaving). Each vehicle gets a stable `global id`; the engine publishes it in iris:meta as `global_ids`
({track id: global id}) and draws it under the box, and the module announces assignments, exits and directions on
`iris:mtmc:assigned`, `iris:mtmc:exited` and `iris:mtmc:direction`.

Plates are the module's too, because the best plate per vehicle is kept per GLOBAL id (every camera adds to it). The
`plates` module finds plates on each vehicle's crop and leaves them in the frame's objects (kind "plate"); this
module hands them to the tracker, collects the best plate it saw of each vehicle in each camera, and announces it on
`iris:mtmc:plates`: the crop saved under `plates.dir`, or, with an OCR model, queued for reading and announced with the
digits when read. The digits are written onto the drawn plate box.

It publishes every box it can follow, shadow tracks included (a tracker that sees a detection come and go would
reassign ids), and the detections of iris:meta carry the box (7 fields) so a consumer can follow a vehicle.

    modules:
      mtmc:
        config: {...}              # MTMCConfig, see core/__init__.py
        vlm_arbiter: {enabled: true, ...}
        vehicle_crops: {local_dir: ...}
        plates: {dir: ..., ocr_model: ..., ocr_dict: ...}

List it last among the modules that change detections: it follows the vehicles the modules above it left.
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from perception.engine.frame import FrameOutput, FrameView
from perception.engine.model_manifest import ModelRef
from perception.engine.modules import Module, ModuleConfigError, ModuleStartupError

from .core import MTMCConfig, MTMCManager
from .core._reid import REID_WEIGHTS
from .palette import gid_color_rgb

logger = logging.getLogger("iris.sidecar")

_VLM_KEYS = {"enabled", "host", "model", "model_env", "num_ctx", "temperature", "keep_alive", "shrink_steps",
             "use_format", "min_confidence", "max_wait_s", "worker_threads", "log_dir", "num_predict",
             "request_timeout_s", "provider", "base_url", "base_url_env", "api_key_env"}
_VEHICLE_CROP_KEYS = {"enabled", "local_dir", "min_update_interval_s", "stay_gap_s"}
_PLATE_KEYS = {"dir", "ocr_model", "ocr_dict", "ocr_enhance", "save_crops", "min_width_px", "min_height_px"}
NEXT_GID_KEY = "iris:mtmc:next_gid"


def _only(name: str, value, known: set) -> None:
    unknown = sorted(set(value) - known)
    if unknown:
        raise ModuleConfigError(f"modules.mtmc.{name}: unknown setting(s) {unknown}; known: {sorted(known)}")


@dataclass
class Settings:
    config: dict = field(default_factory=dict)                  # MTMCConfig fields
    vlm_arbiter: Optional[dict] = None
    vehicle_crops: dict = field(default_factory=dict)           # best-shot thumbnails; on unless {"enabled": false}
    plates: Optional[dict] = None

    def __post_init__(self) -> None:
        fields_ = set(MTMCConfig.__dataclass_fields__)
        unknown = sorted(set(self.config) - fields_)
        if unknown:
            raise ModuleConfigError(f"modules.mtmc.config: unknown setting(s) {unknown}")
        for name, known in (("vlm_arbiter", _VLM_KEYS), ("vehicle_crops", _VEHICLE_CROP_KEYS), ("plates", _PLATE_KEYS)):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, dict):
                    raise ModuleConfigError(f"modules.mtmc.{name}: expected a mapping, got {type(value).__name__}")
                _only(name, value, known)
        if self.plates is not None and self.plates.get("ocr_model") and not self.plates.get("ocr_dict"):
            raise ModuleConfigError("modules.mtmc.plates: ocr_model needs ocr_dict (the reader's character list)")


class Mtmc(Module):
    name = "mtmc"
    Settings = Settings

    @classmethod
    def model_files(cls, settings, every_variant: bool = False) -> list:
        refs = []
        if settings.config.get("reid_enabled", MTMCConfig.__dataclass_fields__["reid_enabled"].default):
            refs.append(ModelRef(str(REID_WEIGHTS), "config.reid_enabled"))
        plates = settings.plates or {}
        if plates.get("ocr_model"):
            refs += [ModelRef(plates["ocr_model"], "plates.ocr_model"), ModelRef(plates["ocr_dict"], "plates.ocr_dict")]
        return refs
    after = ("plates",)                 # it hands on the plates that module leaves in the frame's objects

    def __init__(self, settings: Settings, services) -> None:
        super().__init__(settings, services)
        self.manager: Optional[MTMCManager] = None
        self.vlm = None
        self.plate_book = None
        self.plate_digits: dict = {}                            # (camera, track id) -> " 5356"
        self._plate_notes_logged: set = set()
        self._failures, self._last_failure_log = 0, -1e9

    # build ------------------------------------------------------------------------------------------------
    def on_built(self, pipeline) -> None:
        s = self.settings
        try:
            cfg = MTMCConfig(**{**s.config, "enabled": True})        # listing the module is what enables it
            self.vlm = self._build_vlm()
            self.manager = MTMCManager(cfg, vlm_arbiter=self.vlm)
        except Exception as exc:  # noqa: BLE001
            raise ModuleStartupError(f"modules.mtmc could not start: {exc}") from exc
        self._persist_global_ids(cfg)
        self._build_gallery(cfg)
        self._build_plate_book()
        logger.info("MTMC initialized (reid_enabled=%s, sim_threshold=%s, topology=%s)", cfg.reid_enabled,
                    cfg.reid_sim_threshold, cfg.topology)

    def _build_vlm(self):
        va = self.settings.vlm_arbiter or {}
        if not va.get("enabled", False):
            return None
        try:
            from .vlm_arbiter import VLMArbiter
            arbiter = VLMArbiter(
                enabled=True, host=va.get("host", "http://127.0.0.1:11434"),
                model=os.environ.get(va.get("model_env", ""), "") or va.get("model", "qwen3-vl:4b"),
                num_ctx=int(va.get("num_ctx", 8192)), temperature=float(va.get("temperature", 0.0)),
                keep_alive=float(va.get("keep_alive", -1)),
                shrink_steps=va.get("shrink_steps", [1024, 896, 768, 640]),
                use_format=bool(va.get("use_format", False)), min_confidence=float(va.get("min_confidence", 0.5)),
                max_wait_s=float(va.get("max_wait_s", 12.0)), worker_threads=int(va.get("worker_threads", 2)),
                log_dir=va.get("log_dir") or None, num_predict=int(va.get("num_predict", 400)),
                request_timeout_s=float(va.get("request_timeout_s", 30.0)), provider=va.get("provider", "ollama"),
                base_url=os.environ.get(va.get("base_url_env", ""), "") or va.get("base_url", ""),
                api_key_env=va.get("api_key_env", "QWEN_API_KEY"))
            provider = va.get("provider", "ollama")
            if provider == "ollama":
                where, model = va.get("host"), va.get("model", "qwen3-vl:4b")
            else:  # hosted, OpenAI-compatible: the URL and model come from the environment
                where = os.environ.get(va.get("base_url_env", ""), "") or va.get("base_url", "")
                model = os.environ.get(va.get("model_env", ""), "") or va.get("model", "")
            logger.info("VLM arbiter initialized (provider=%s, host=%s, model=%s)", provider, where, model)
            return arbiter
        except Exception as exc:  # noqa: BLE001
            logger.warning("VLM arbiter init failed (non-fatal, continuing without): %s", exc)
            return None

    def _persist_global_ids(self, cfg) -> None:
        """Global ids survive a sidecar restart: the next free id lives in Redis, so a restarted sidecar cannot reuse
        ids the app and the UI already attribute to other vehicles."""
        try:
            import redis
            client = redis.Redis.from_url(cfg.redis_url)
            nxt = client.get(NEXT_GID_KEY)
            if nxt:
                self.manager.association.start_gid_at(int(nxt))
                logger.info("MTMC global ids continue from %d", int(nxt))
            self.manager.association.on_new_gid = lambda n, _r=client: _r.set(NEXT_GID_KEY, n)
        except Exception as exc:  # noqa: BLE001
            logger.warning("global id persistence unavailable (ids restart at 0): %s", exc)

    def _build_gallery(self, cfg) -> None:
        vc = self.settings.vehicle_crops
        if not vc.get("enabled", True):
            return
        try:
            import redis
            from .vehicle_gallery import VehicleGallery
            self.manager.vehicle_gallery = VehicleGallery(
                enabled=True, storage_backend="local",
                local_dir=vc.get("local_dir", "/workspace/iris/data/vehicle_crops"),
                redis_client=redis.Redis.from_url(cfg.redis_url),
                min_update_interval_s=float(vc.get("min_update_interval_s", 0.0)),
                stay_gap_s=float(vc.get("stay_gap_s", 90.0)))
            logger.info("Vehicle thumbnail gallery enabled")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vehicle gallery init failed (non-fatal, no thumbnails): %s", exc)

    def _build_plate_book(self) -> None:
        p = self.settings.plates or {}
        if not p.get("ocr_model"):
            return
        try:
            from .plate_reader import PlateBook, PlateReader
            self.plate_book = PlateBook(
                PlateReader(p["ocr_model"], p["ocr_dict"], bool(p.get("ocr_enhance", True))),
                p.get("dir", "data/plates"), on_read=self._on_plate_read,
                min_size=(float(p.get("min_width_px", 28)), float(p.get("min_height_px", 12))),
                save_crops=bool(p.get("save_crops", False)))
            logger.info("plate OCR enabled: %s", p["ocr_model"])
        except Exception as exc:  # noqa: BLE001
            logger.error("plate OCR unavailable (%s): plates are detected but not read", exc)

    def stop(self) -> None:
        if self.manager is not None:
            try:
                self.manager.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("mtmc close failed: %s", exc)

    def gauges(self) -> dict:
        """Association health (the VLM's backlog, latency, errors, open vehicle tracks) and the work queues."""
        m = self.manager
        if m is None:
            return {}
        gauges = {f"mtmc_{k}": (lambda k=k: m.health().get(k, 0.0)) for k in m.health()}
        if self.vlm is not None:
            gauges.update({f"vlm_{k}": (lambda k=k: self.vlm.health().get(k, 0.0)) for k in self.vlm.health()})
        gauges["q_embed_depth"] = lambda: m._embed_queue.qsize()
        gallery = getattr(m, "vehicle_gallery", None)
        if gallery is not None:
            gauges["q_gallery_depth"] = lambda: gallery._queue.qsize()
        return gauges

    # plates -----------------------------------------------------------------------------------------------
    def _plate_dir(self) -> str:
        return (self.settings.plates or {}).get("dir", "data/plates")

    def _on_plate_read(self, gid, row) -> None:
        """A plate was read: announce it on iris:mtmc:plates with its text (the picture with every camera's crop and
        reading beside it is <plate dir>/<global id>.png)."""
        digits = "".join(c for c in row["text"] if c.isdigit())
        if digits:
            self.plate_digits[(row["cam"], row["local_id"])] = " " + digits
        self.services.emit("iris:mtmc:plates", {
            "global_id": gid, "cam_id": row["cam"], "local_id": row["local_id"], "conf": row["det_conf"],
            "text": row["text"], "text_conf": row["ocr_conf"], "file": os.path.join(self._plate_dir(), f"{gid}.png")},
            maxlen=2000)

    def _publish_plate(self, ev) -> None:
        """A camera's best plate of a vehicle. With OCR on it is queued for reading and announced (with its text)
        when read; without, the crop is saved as <plate dir>/<global id>.jpg and announced on iris:mtmc:plates
        (global id, camera, confidence, box, file)."""
        import json
        crop = ev.get("crop")
        if self.plate_book is not None:
            if crop is not None and crop.size:
                self.plate_book.submit(ev["global_id"], ev["cam_id"], crop, ev["conf"], ev["local_id"])
            elif (ev["cam_id"], ev["global_id"], "nocrop") not in self._plate_notes_logged:
                self._plate_notes_logged.add((ev["cam_id"], ev["global_id"], "nocrop"))
                logger.info("[plate] global %s %s #%s: a plate was found but there was no frame copy to cut it from",
                            ev["global_id"], ev["cam_id"], ev["local_id"])
            return
        path = ""
        if crop is not None and crop.size:
            try:
                import cv2
                os.makedirs(self._plate_dir(), exist_ok=True)
                path = os.path.join(self._plate_dir(), f"{ev['global_id']}.jpg")
                cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            except Exception as exc:  # noqa: BLE001
                logger.warning("plate crop save failed: %s", exc)
                path = ""
        self.services.emit("iris:mtmc:plates", {
            "global_id": ev["global_id"], "cam_id": ev["cam_id"], "local_id": ev["local_id"], "conf": ev["conf"],
            "box": json.dumps(ev["box"]), "file": path}, maxlen=2000)

    # hot path ---------------------------------------------------------------------------------------------
    def transform(self, frame: FrameView) -> None:
        # the tracker needs every box it can follow, shadow tracks (tracker-predicted, detector-missed) included:
        # with only the frames the detector confirmed, ids flicker in the cross-camera association
        frame.dets_with_boxes = True
        for o in frame.objects:
            if o.kind == "object":
                o.publish = o.draw

    def wants_pixels(self, frame: FrameView) -> bool:
        return frame.need_meta and self.manager is not None and any(o.publish for o in frame.objects)

    def analyze(self, frame: FrameView, out: FrameOutput) -> None:
        m = self.manager
        if m is None:
            return
        emit, objects = self.services.emit, frame.objects
        gids: dict = {}
        try:
            m.poll_gate(self.services.redis)
            gids = m.process_frame(
                frame.stream_id, frame.frame_num, frame.pts,
                [[o.label, round(o.shown_conf, 3), o.id] for o in objects if o.publish],
                frame_bgr=frame.bgr,
                boxes=[(o.left, o.top, o.width, o.height, o.label, o.shown_conf, o.id)
                       for o in objects if o.draw and o.kind in ("object", "face")],
                plates=[(o.left, o.top, o.width, o.height, o.conf, o.parent_id) for o in objects if o.kind == "plate"])
            for ev in m.pop_plate_events():
                self._publish_plate(ev)
            for ev in m.pop_assignment_events():
                emit("iris:mtmc:assigned", {k: str(v) for k, v in ev.items()}, maxlen=2000)
            for ev in m.pop_exit_events():
                emit("iris:mtmc:exited", {k: str(v) for k, v in ev.items()}, maxlen=2000)
            for ev in m.pop_direction_events():
                emit("iris:mtmc:direction", {k: str(v) for k, v in ev.items()}, maxlen=2000)
                logger.info("[event] direction: gid %s on %s is %s (%s facing the camera, conf %s)",
                            ev["global_id"], ev["cam_id"], ev["direction"], ev["orientation"], ev["confidence"])
        except Exception as exc:  # noqa: BLE001 -- a tracker failure costs this frame its identities, not the frame
            self._failures += 1
            if frame.now - self._last_failure_log >= 10.0:       # said loudly, but not on every frame
                self._last_failure_log = frame.now
                logger.warning("[%s] mtmc failed on a frame (%d so far): %s", frame.stream_id, self._failures, exc,
                               exc_info=True)
        if gids:
            out.fields["global_ids"] = gids
        for o in objects:
            if not o.draw or o.label == "face":
                continue
            gid = gids.get(o.id)
            if gid is not None:
                o.attrs["global_id"] = gid          # for modules listed after this one
                # the persistent cross-camera identity, written under the box in its own colour
                out.annotate(o, extra=(f"gid:{gid}", gid_color_rgb(gid), 0.75, 2))
            if o.kind == "plate_shown":
                digits = self.plate_digits.get((frame.stream_id, o.parent_id), "")
                if digits:       # the full reading is on the vehicle's plate picture (cv2 cannot draw Arabic letters)
                    out.annotate(o, text=f"plate{digits} {o.conf:.2f}")


MODULE = Mtmc
