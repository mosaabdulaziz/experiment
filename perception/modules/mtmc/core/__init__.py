"""Real-time multi-camera multi-object tracking (MTMC) for the DeepStream sidecar.

Called from the sidecar's probe callback on every analysis frame.  Maintains
per-camera vehicle tracklets (via internal IoU tracker), computes Re-ID
embeddings from CPU frame copies, and runs cross-camera vehicle association to
assign persistent global IDs across cameras.

Usage from the sidecar probe::

    from mtmc import MTMCManager, MTMCConfig

    mtmc = MTMCManager(MTMCConfig.from_yaml("products/multicam/sidecar.yaml"))
    # per frame:
    global_ids = mtmc.process_frame(source_id, frame_num, pts, dets,
                                     frame_bgr=frame_bgr, boxes=boxes)
    # global_ids = {ds_object_id: global_id, ...}
"""
from __future__ import annotations

import collections
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._reid import ReIDEngine, ReIDWeightsMissing
from ._tracklets import CameraTracklets, TrackletRecord, _VEHICLE_CLASSES
from ._associate import (
    GlobalVehicleAssociation,
    GlobalVehicleTrack,
    associate_cameras,
)

logger = logging.getLogger("iris.mtmc")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MTMCConfig:
    """MTMC parameters (loaded from sidecar.yaml's ``mtmc:`` section)."""

    enabled: bool = False

    # ---- per-camera tracking ----
    iou_threshold: float = 0.35       # frame-to-frame association IoU
    max_unmatched_frames: int = 15    # tracklet survives this many empty frames
    min_tracklet_frames: int = 5      # minimum frames before a tracklet is embeddable
    # A vehicle crop smaller than this is not a picture of the vehicle: it is not used for Re-ID, the VLM, the
    # thumbnail gallery or the plate's whole-vehicle shot, and a vehicle seen only that small gets no global id
    # yet. Both limits apply: the shorter side in pixels, and the area in pixels.
    min_crop_side_px: int = 90
    min_crop_area_px: int = 12000

    # ---- Re-ID ----
    reid_checkpoint: str = ""         # (unused) legacy fast-reid path, kept for back-compat
    reid_config: str = ""             # (unused) legacy fast-reid config, kept for back-compat
    reid_enabled: bool = True         # load and use the Re-ID model
    reid_embed_stride: int = 3        # crop sampling stride within a tracklet (1 = use all)
    reid_max_crops: int = 16          # cap crops per tracklet per embed pass
    reid_batch_size: int = 32
    reid_min_crops: int = 6           # embed once a tracklet has this many crops (or when it expires)

    # ---- object classes to track ----
    tracked_classes: tuple = ("license_plate", "plate", "car", "truck", "bus", "motorcycle", "bicycle")

    # ---- cross-camera association ----
    reid_sim_threshold: float = 0.55  # cosine similarity to merge (match threshold)
    reid_gate_threshold: float = 0.15 # below this = completely different vehicle, reject (near-0 FN)
    time_tolerance_s: float = 2.0     # max gap for synchronous overlap
    min_disappearance_s: float = 20.0 # 20-second disappearance rule threshold
    max_inactive_ttl_s: float = 300.0 # TTL for inactive global vehicle tracks
    associate_interval_s: float = 1.0 # run association every N seconds
    min_embed_frames: int = 3         # tracklet must have this many frames before merge
    topology: Any = None              # camera transition topology (chain, dict, or pair list)
    camera_groups: Any = None         # {name: [cams]}: cameras with overlapping views (see _topology.py)
    group_links: Any = None           # [[group, group, min_seconds], ...]: neighbouring groups, both ways
    camera_gaps: Any = None           # per-camera-pair timing gaps (dict or pair mapping)
    vlm_max_candidates: int = 3       # with a VLM: max time-eligible tracks sent to it per tracklet
    vlm_views: int = 3                # with a VLM: crops per vehicle sent together, spread over time on camera
    vlm_history_views: int = 1        # with a VLM: crops sent from EACH earlier camera the candidate passed through
    vlm_collect_max_s: float = 3.0    # with a VLM: max seconds to wait for vlm_views crops of a new vehicle before asking
    overlap_tolerance_s: float = 5.0   # ACTIVE tracks seen on two cameras within this many seconds may merge (overlapping views); 0 = never
    same_camera_regain_s: float = 0.0  # a vehicle its own camera lost this recently may be re-found there under a new tracker id (the VLM confirms)
    gate_wait_s: float = 10.0         # a new vehicle on a gate camera waits this long (video seconds) for the line to say which way it goes
    origin_cameras: Any = None        # only vehicles first seen on these cameras are tracked (None = any camera)
    # A vehicle seen only on entrance cameras, then out of sight of every camera for this many seconds, and
    # then matched at an entrance camera again: its earlier visit is complete (closed at its last sighting)
    # and the return starts a new one. 0 = off.
    entrance_absence_exit_s: float = 0.0
    # Cameras where the VLM tells entering from leaving by which end of the vehicle faces the camera:
    # {camera: "front_is_exit" | "front_is_enter"}. E.g. D16 faces the cab of vehicles that are leaving, so
    # {"cam-02": "front_is_exit"}. A vehicle judged "leaving" is matched to one inside and closed, like a
    # gate-line exit; "entering" is recorded (nothing is skipped: it may be a tracker split of a parked
    # truck). Needs the VLM; None/{} = off.
    direction_cameras: Any = None
    direction_views: int = 3          # crops of the vehicle sent for the direction question
    direction_min_confidence: float = 0.7
    direction_wait_s: float = 25.0    # a new vehicle on such a camera waits this long (video s) for the verdict
    direction_retry_s: float = 0.0    # >0: an unclear/unsure answer is asked again with the newest crops after this many seconds, until direction_wait_s
    direction_require: bool = False   # True: on those cameras only a vehicle decided as ENTERING starts a track

    # ---- Redis ----
    redis_url: str = "redis://redis:6379/0"

    # ---- Gold-label capture (one-off debug dump, off by default) ----
    # Phase 0 of VLM_ASSOCIATION_RECORD_REPLAY_PLAN.md: when set, the first time a
    # (cam_id, local_id) tracklet is assigned a global_id, save a representative crop
    # + append a summary row under this directory. Lets a human build/verify a
    # trusted gold mapping from a real deterministic run without grepping logs for
    # images that were never saved. Leave "" for normal operation (no overhead).
    gold_dump_dir: str = ""

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MTMCConfig":
        """Load from sidecar.yaml (extracts the ``mtmc:`` section)."""
        try:
            import yaml
            from pathlib import Path
            raw = yaml.safe_load(Path(yaml_path).read_text()) or {}
            mtmc_raw = raw.get("mtmc", {})
            kwargs = {}
            for k, v in mtmc_raw.items():
                if k in cls.__dataclass_fields__:
                    kwargs[k] = v
            return cls(**kwargs)
        except Exception as exc:
            logger.warning("mtmc config load failed (%s), using defaults: %s",
                           yaml_path, exc)
            return cls()


# ---------------------------------------------------------------------------
# MTMC Manager — the main entry point
# ---------------------------------------------------------------------------

def _boxes_overlap(a: tuple, b: tuple, min_frac: float = 0.5) -> bool:
    """True when the intersection covers `min_frac` of the smaller (x, y, w, h) box."""
    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return False
    return iw * ih >= min_frac * min(a[2] * a[3], b[2] * b[3])


SIDE_MODE = "side_right_to_left_enters"    # direction_cameras mode of a side-view camera (D41); see perception/modules/mtmc/vlm_arbiter.py


def _side_motion(boxes) -> str | None:
    """Which way a vehicle moved across a side-view camera: right to left = "entering", left to right = "exiting",
    "still" when it hardly moved, None when there are too few boxes to tell yet. boxes are (x, y, w, h)."""
    if not boxes or len(boxes) < 8:
        return None
    cx = [b[0] + b[2] / 2.0 for b in boxes]
    mean_w = sum(b[2] for b in boxes) / len(boxes)
    dx = cx[-1] - cx[0]
    if abs(dx) < 0.5 * mean_w:
        return "still"
    return "entering" if dx < 0 else "exiting"


class MTMCManager:
    """Real-time multi-camera vehicle tracker.

    Called once per analysis frame from the sidecar probe. Maintains
    per-camera vehicle tracklets, computes vehicle Re-ID embeddings,
    and runs cross-camera vehicle identity association.

    The output ``global_ids`` dict maps DeepStream ``object_id`` → ``global_id``,
    so the frontend can overlay the persistent cross-camera vehicle identity.
    """

    def __init__(self, cfg: MTMCConfig, vlm_arbiter=None):
        self.cfg = cfg
        self.vlm_arbiter = vlm_arbiter  # Optional[VLMArbiter]: cross-camera association validation
        self._cameras: dict[str, CameraTracklets] = {}
        self.association = GlobalVehicleAssociation(
            topology=cfg.topology,
            min_disappearance_s=cfg.min_disappearance_s,
            max_inactive_ttl_s=cfg.max_inactive_ttl_s,
            reid_sim_threshold=cfg.reid_sim_threshold,
            reid_gate_threshold=cfg.reid_gate_threshold,
            camera_gaps=cfg.camera_gaps,
            vlm_arbiter=vlm_arbiter,
            vlm_max_candidates=cfg.vlm_max_candidates,
            vlm_views=cfg.vlm_views,
            origin_cameras=cfg.origin_cameras,
            vlm_history_views=cfg.vlm_history_views,
            gate_wait_s=cfg.gate_wait_s,
            same_camera_regain_s=cfg.same_camera_regain_s,
            overlap_tolerance_s=cfg.overlap_tolerance_s,
            camera_groups=cfg.camera_groups,
            group_links=cfg.group_links,
        )
        # Optional best-shot thumbnail gallery (perception/modules/mtmc/vehicle_gallery.py); set by
        # sidecar.py after construction. None = no thumbnails.
        self.vehicle_gallery = None
        # Every first global-id assignment of a tracklet. The per-frame Redis entries can
        # miss it (a slow VLM verdict lands after the vehicle left; a very brief tracklet
        # is never published), so sidecar.py drains these and announces them on a separate
        # Redis stream that the tracking service consumes. See pop_assignment_events().
        self._assignment_events: collections.deque = collections.deque()
        # Vehicles that left through the gate (their track was just closed). Drained by
        # sidecar.py onto a Redis stream, like the assignment events.
        self._exit_events: collections.deque = collections.deque()
        self._exits_total = 0
        self._absence_exit_s = float(getattr(cfg, "entrance_absence_exit_s", 0.0) or 0.0)
        # direction (entering / leaving) by the VLM, per camera; see MTMCConfig.direction_cameras
        self._direction_map: dict[str, str] = {}
        self._dir_state: dict[tuple[str, int], dict] = {}       # (cam, local id) -> {"job", "t0"}
        self._dir_labels: dict[tuple[str, int], dict] = {}      # decided, waiting for a global id to be published
        self._direction_events: collections.deque = collections.deque()
        self._direction_total = collections.Counter()
        if getattr(cfg, "direction_cameras", None) and vlm_arbiter is not None and getattr(vlm_arbiter, "enabled", False) \
                and hasattr(vlm_arbiter, "submit_direction"):
            self._direction_map = {str(c): str(m) for c, m in dict(cfg.direction_cameras).items()}
            self.association.set_direction_cameras(self._direction_map, cfg.direction_wait_s,
                                                   require=bool(getattr(cfg, "direction_require", False)))
        # Best plate seen so far per (global id, camera), and the plate events not yet published.
        self.gid_plates: dict[tuple[int, str], dict] = {}
        self._plate_events: collections.deque = collections.deque()
        self._gate_last_id: str = f"{int(time.time() * 1000)}-0"
        self._gate_next_poll: float = 0.0
        self._gate_lock = threading.Lock()
        self._reid: ReIDEngine | None = None
        if cfg.reid_enabled:
            try:
                self._reid = ReIDEngine(cfg)
                if self._reid.ensure_loaded():
                    logger.info("[mtmc] Re-ID engine initialized successfully")
                else:
                    logger.warning("[mtmc] Re-ID engine created but model load returned false")
            except ReIDWeightsMissing:
                raise                                   # a missing file refuses startup (on_built -> ModuleStartupError)
            except Exception as exc:
                logger.warning("[mtmc] Re-ID engine init failed: %s", exc)
                self._reid = None

        # Async embedding: tracklets are embedded on a background worker thread
        # so the ResNet50 forward never blocks the pipeline's streaming thread.
        self._embed_queue: "queue.Queue" = queue.Queue()
        self._embed_worker: threading.Thread | None = None
        self._embed_stop = threading.Event()
        if self._reid is not None:
            self._embed_worker = threading.Thread(
                target=self._embed_loop, name="mtmc-embed", daemon=True,
            )
            self._embed_worker.start()

        self._last_associate_s: float = 0.0
        self._local_id_counter: int = 0
        self._total_frames: int = 0
        self._total_assoc: int = 0

        self._gold_dump_dir: Path | None = None
        self._gold_dumped_keys: set[tuple[str, int]] = set()
        if cfg.gold_dump_dir:
            self._gold_dump_dir = Path(cfg.gold_dump_dir)
            (self._gold_dump_dir / "crops").mkdir(parents=True, exist_ok=True)
            logger.info("[mtmc] gold-label dump enabled -> %s", self._gold_dump_dir)

    def close(self) -> None:
        """Stop the embedding worker thread (drains queued embeds)."""
        if self.vehicle_gallery is not None:
            self.vehicle_gallery.close()
        self._embed_stop.set()
        if self._embed_worker is not None:
            self._embed_queue.put(None)
            try:
                self._embed_worker.join(timeout=5.0)
            except Exception:
                pass
            self._embed_worker = None
        if self.vlm_arbiter is not None:
            self.vlm_arbiter.close()

    def process_frame(
        self,
        source_id: str,
        frame_num: int,
        pts: float,
        dets: list[list],
        frame_bgr: np.ndarray | None = None,
        boxes: list[tuple] | None = None,
        plates: list[tuple] | None = None,
    ) -> dict[int, int]:
        """Process one analysis frame and return {ds_object_id: global_id}.

        Parameters
        ----------
        source_id : camera identifier (e.g. "cam-1")
        frame_num : sequential frame number from DeepStream
        pts : presentation timestamp in seconds
        dets : list of [label, confidence, object_id] from the probe
        frame_bgr : CPU frame copy (BGR, for Re-ID crops); None if unavailable
        boxes : list of (left, top, width, height, label, conf, object_id)
        plates : list of (left, top, width, height, conf, parent_object_id) from the plate
            detector; each is attached to the vehicle it was found on

        Returns
        -------
        dict mapping **DeepStream** object_id -> global_id.
        Only vehicle objects assigned a global ID are included.
        """
        if not self.cfg.enabled:
            return {}

        self._total_frames += 1

        # Ensure camera state exists
        if source_id not in self._cameras:
            self._cameras[source_id] = CameraTracklets(source_id, self.cfg)

        cam = self._cameras[source_id]

        # 1. Update per-camera IoU tracker
        ds_oid_map = cam.update(frame_num, pts, dets, frame_bgr, boxes)
        if plates:
            cam.add_plates(plates, pts, frame_bgr)

        # 2. Mark expired/aged-out tracklets from this camera as INACTIVE
        expired = cam.pop_expired_tracklets()
        for ex_tr in expired:
            self.association.mark_tracklet_inactive(source_id, ex_tr.object_id, ex_tr.last_pts)

        # 3. Queue tracklets for embedding (async — worker thread computes them)
        self._submit_embeddings()

        # 4. Associate active tracklets with global vehicle tracks
        for tr in cam.all_tracklets():
            prev_gid = tr.global_id
            if prev_gid < 0 and tr.best_shot is None and tr.small_seen:
                continue          # never seen big enough to tell from another vehicle: nothing to match on yet
            # Give a new vehicle a moment to accumulate vlm_views crops over time before the
            # VLM judges it -- unless it already left, or has been visible long enough.
            views = tr.get_view_crops()
            views_ready = (len(views) >= self.cfg.vlm_views or tr.last_frame != frame_num
                           or pts - tr.start_pts >= self.cfg.vlm_collect_max_s)
            overlapping = frozenset()
            if prev_gid < 0 and tr.boxes:
                overlapping = frozenset(
                    o.global_id for o in cam.all_tracklets()
                    if o is not tr and o.global_id >= 0 and o.boxes and _boxes_overlap(tr.boxes[-1], o.boxes[-1])
                )
            if self._direction_map and source_id in self._direction_map:
                self._direction_step(source_id, tr, views, views_ready, pts)
            next_gid_before = self.association._next_gid
            gid = self.association.get_or_create_global_id(
                cam_id=source_id,
                local_id=tr.object_id,
                pts=pts,
                embedding=tr.embedding,
                crops=tr.get_crops(),
                views=views,
                views_ready=views_ready,
                first_pts=tr.start_pts,
                overlapping_gids=overlapping,
            )
            tr.global_id = gid
            if gid >= 0 and self.association.gate_hint(source_id, tr.object_id) == "exit":
                # This tracklet left through the gate and is now known to be this vehicle
                # (possibly only now, once the VLM has answered): the vehicle has left.
                self._close_exited(gid, source_id, tr.object_id)
            if gid >= 0 and (source_id, tr.object_id) in self._dir_labels:
                self._publish_direction(source_id, tr.object_id, gid)
            if gid >= 0 and prev_gid < 0 and gid < next_gid_before:
                self._close_if_reappeared(gid, source_id, tr.object_id)
            if gid >= 0 and prev_gid < 0:
                self._assignment_events.append({
                    "cam_id": source_id, "local_id": tr.object_id, "global_id": gid,
                    "pts_first": round(tr.start_pts, 3), "pts_last": round(tr.last_pts, 3),
                })
            if self.vehicle_gallery is not None and gid >= 0 and frame_bgr is not None:
                self._submit_gallery(source_id, tr, gid, frame_num, pts, frame_bgr)
            if self._gold_dump_dir is not None:
                self._maybe_dump_gold(source_id, tr, gid, pts, frame_bgr)

        # 4b. A vehicle's best plate belongs to its global id (every camera adds to it)
        for tr in cam.all_tracklets():
            if tr.global_id >= 0 and tr.plate is not None and not tr.plate.get("published"):
                tr.plate["published"] = True
                self._publish_plate(source_id, tr)

        # 5. Prune expired inactive global vehicle tracks
        self.association.prune_expired_tracks(pts)

        # 6. Fallback when Re-ID model is unavailable
        if self._reid is None or not self._reid.ensure_loaded():
            self._assign_local_ids(source_id)

        # 7. Return latest global IDs mapped back to DeepStream object_ids
        ds_gids = self._build_ds_global_ids(source_id, ds_oid_map)
        return ds_gids

    def _close_if_reappeared(self, gid: int, cam_id: str, local_id: int) -> None:
        """No gate line on an entrance camera: a vehicle that was inside and is matched there
        again, after being seen on a camera that is not an entrance one, is leaving -- close its visit."""
        a = self.association
        origins = a._origin_cameras
        if (origins is None or cam_id not in origins or cam_id in a._gate_cams
                or a.gate_hint(cam_id, local_id) is not None):
            return
        gtr = a.global_tracks.get(gid)
        if gtr is None or any(al for c, al in gtr.local_track_aliases.items() if c != cam_id):
            return   # still in view on another camera (overlap), not leaving
        if not any(c not in origins for c in gtr.camera_history):
            # Seen only at the entrance so far. A long absence from every camera and then a return to the
            # entrance means its earlier visit is over: close it at its last sighting, keep the vehicle.
            gap = gtr.last_gap_s
            if self._absence_exit_s > 0 and gap >= self._absence_exit_s:
                at = time.time() - gap
                logger.info("[mtmc] Global ID %d back at %s after %.0f s out of sight (entrance only): its earlier "
                            "visit is complete", gid, cam_id, gap)
                self._emit_exit(gid, cam_id, local_id, at=at, reopens=True)
            return   # otherwise it has not been inside yet
        logger.info("[mtmc] reappeared at entrance camera %s without a gate line: Global ID %d leaves", cam_id, gid)
        self._close_exited(gid, cam_id, local_id)

    def health(self) -> dict[str, float]:
        """Numbers for the monitoring gauges."""
        a = self.association
        tracks = list(a.global_tracks.values())
        return {
            "tracks_open": float(sum(1 for t in tracks if not t.exited)),
            "tracks_active": float(sum(1 for t in tracks if not t.exited and t.state == "ACTIVE")),
            "tracks_exited": float(sum(1 for t in tracks if t.exited)),
            "pending_verdicts": float(len(a._pending)),
            "untracked_total": float(len(a._untracked)),
            "exits_total": float(self._exits_total),
            "next_gid": float(a._next_gid),
            "alias_conflicts_total": float(a.alias_conflicts),
            "direction_exiting_total": float(self._direction_total["exiting"]),
            "direction_entering_total": float(self._direction_total["entering"]),
            "direction_undecided_total": float(self._direction_total["undecided"]),
            "direction_not_vehicle_total": float(self._direction_total["not_vehicle"]),
        }

    def _direction_step(self, cam_id: str, tr, views, views_ready: bool, pts: float) -> None:
        """Ask the VLM which end of this vehicle faces the camera and act on the answer.

        front on a camera where that means leaving -> an exit is reported exactly like a gate-line exit
        (the vehicle is matched to one inside and closed); the other end is recorded as entering. A vehicle
        whose answer does not come, or is unclear, is judged as before after direction_wait_s."""
        key = (cam_id, tr.object_id)
        a = self.association
        if key in a._direction_settled or a.gate_hint(cam_id, tr.object_id) is not None:
            a.settle_direction(cam_id, tr.object_id)
            self._dir_state.pop(key, None)
            return
        st = self._dir_state.get(key)
        wait = float(self.cfg.direction_wait_s)
        retry = float(getattr(self.cfg, "direction_retry_s", 0.0) or 0.0)
        n = max(1, int(self.cfg.direction_views))

        def ask(newest: bool) -> int:
            picked = list(views)[-n:] if newest else a._pick_views(list(views), n)
            return self.vlm_arbiter.submit_direction(picked, {
                "cam_id": cam_id, "local_id": tr.object_id, "pts": pts, "mode": self._direction_map[cam_id],
                "captions_a": [f"the tracked vehicle on {cam_id}, view {i} of {len(picked)}, oldest first"
                               for i in range(1, len(picked) + 1)]})

        def give_up(why: str) -> None:
            logger.info("[mtmc] DIRECTION: %s for %s local_id=%d; judged without it", why, cam_id, tr.object_id)
            self._dir_state.pop(key, None)
            a.settle_direction(cam_id, tr.object_id)

        if st is None:
            if not views_ready or not views:
                return
            self._dir_state[key] = {"job": ask(False), "first": pts, "asked": 1, "retry_at": None}
            return
        if st["job"] is None:                      # an unclear answer: ask again with what the camera sees now
            if pts - st["first"] >= wait:
                give_up(f"still not decided after {pts - st['first']:.0f}s")
            elif pts >= st["retry_at"] and views:
                st["job"], st["asked"] = ask(True), st["asked"] + 1
            return
        res = self.vlm_arbiter.take_result(st["job"])
        if res is None:
            if pts - st["first"] >= wait:
                give_up(f"no verdict in {pts - st['first']:.0f}s")
            return
        if self._apply_direction(cam_id, tr.object_id, res, pts, _side_motion(tr.boxes)):
            self._dir_state.pop(key, None)
            a.settle_direction(cam_id, tr.object_id)
        elif retry > 0 and pts - st["first"] + retry < wait:
            st["job"], st["retry_at"] = None, pts + retry
        else:
            give_up("not decided")

    def _apply_direction(self, cam_id: str, local_id: int, res: dict, pts: float, motion: str | None = None) -> bool:
        """True when the answer decided entering or leaving. On a side-view camera (SIDE_MODE) the VLM says which
        way the front points (left/toward = entering, right/away = exiting) and the vehicle's own movement across
        the picture is a second vote: when both are clear and disagree, nothing is decided (asked again)."""
        o, conf = res.get("decision"), res.get("confidence")
        sure = conf is None or conf >= self.cfg.direction_min_confidence
        if o == "none" and sure:
            # not a vehicle at all: final. No retries, no exit or entering, and it never starts or joins a track.
            logger.info("[mtmc] DIRECTION %s local_id=%d: no vehicle in the picture (conf=%s) -> not tracked",
                        cam_id, local_id, conf)
            self._direction_total["not_vehicle"] += 1
            self.association.note_untracked(cam_id, local_id)
            return True
        if self._direction_map.get(cam_id) == SIDE_MODE:
            vlm_dir = {"left": "entering", "toward": "entering", "right": "exiting", "away": "exiting"}.get(o)
            head_on = o in ("toward", "away")       # no left/right to compare with the movement
            too_early = motion is None and not head_on   # the vehicle has not moved far enough to see its way yet
            if (vlm_dir is None or not sure or too_early
                    or (motion in ("entering", "exiting") and not head_on and motion != vlm_dir)):
                logger.info("[mtmc] DIRECTION %s local_id=%d: %s (conf=%s), movement %s -> not decided",
                            cam_id, local_id, o, conf, motion)
                self._direction_total["undecided"] += 1
                return False
            logger.info("[mtmc] DIRECTION %s local_id=%d: front points %s, movement %s -> %s (conf=%s)",
                        cam_id, local_id, o, motion, vlm_dir, conf)
            return self._direction_decided(cam_id, local_id, vlm_dir, o, conf)
        if o not in ("front", "rear") or not sure:
            logger.info("[mtmc] DIRECTION %s local_id=%d: %s (conf=%s) -> not decided", cam_id, local_id, o, conf)
            self._direction_total["undecided"] += 1
            return False
        front_exits = self._direction_map.get(cam_id, "front_is_exit") == "front_is_exit"
        direction = "exiting" if (o == "front") == front_exits else "entering"
        logger.info("[mtmc] DIRECTION %s local_id=%d: %s facing the camera -> %s (conf=%s)",
                    cam_id, local_id, o, direction, conf)
        return self._direction_decided(cam_id, local_id, direction, o, conf)

    def _direction_decided(self, cam_id: str, local_id: int, direction: str, orientation: str, conf) -> bool:
        self._direction_total[direction] += 1
        self._dir_labels[(cam_id, local_id)] = {"direction": direction, "orientation": orientation,
                                                "confidence": conf, "at": time.time()}
        if direction == "exiting":
            self.apply_gate_event(cam_id, local_id, "exit", time.time())
        else:
            self.association.note_direction_entering(cam_id, local_id)
        return True

    def _publish_direction(self, cam_id: str, local_id: int, gid: int) -> None:
        lab = self._dir_labels.pop((cam_id, local_id), None)
        if lab is not None:
            self._direction_events.append({"global_id": gid, "cam_id": cam_id, "local_id": local_id,
                                           "direction": lab["direction"], "orientation": lab["orientation"],
                                           "confidence": lab["confidence"], "at": round(lab["at"], 3)})

    def pop_direction_events(self) -> list[dict]:
        out = []
        while self._direction_events:
            out.append(self._direction_events.popleft())
        return out

    def _emit_exit(self, gid: int, cam_id: str, local_id: int, at: float | None = None,
                   reopens: bool = False) -> None:
        self._exits_total += 1
        ev = {"global_id": gid, "cam_id": cam_id, "local_id": local_id,
              "at": at if at is not None else (self.association.gate_time(cam_id, local_id) or time.time())}
        if reopens:
            ev["reopens"] = 1        # the visit is complete but the vehicle is back: a sighting after it starts a new one
        self._exit_events.append(ev)

    def _close_exited(self, gid: int, cam_id: str, local_id: int) -> None:
        if self.association.close_track(gid):
            self._emit_exit(gid, cam_id, local_id)

    def apply_gate_event(self, cam_id: str, local_id: int, direction: str, at: float = 0.0) -> None:
        """A tracklet crossed the gate line: 'enter' (a new vehicle) or 'exit' (leaving)."""
        gid = self.association.note_gate(cam_id, local_id, direction, at)
        if gid is not None:
            self._emit_exit(gid, cam_id, local_id)   # its track existed and was just closed

    def poll_gate(self, r) -> None:
        """Read new gate crossings the tracking service published (iris:mtmc:gate)."""
        now = time.monotonic()
        if now < self._gate_next_poll or not self._gate_lock.acquire(blocking=False):
            return
        try:
            self._gate_next_poll = now + 0.3
            try:
                # cameras that have a gate line: there a new vehicle waits for the line to say
                # which way it is going before it is judged
                self.association.set_gate_cameras(
                    {(c.decode() if isinstance(c, bytes) else str(c)) for c in r.smembers("iris:mtmc:gate:cams")})
            except Exception:
                pass
            try:
                res = r.xread({"iris:mtmc:gate": self._gate_last_id}, count=100)
            except Exception:
                return
            for _name, messages in res or []:
                for msg_id, fields in messages:
                    self._gate_last_id = msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)
                    try:
                        d = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
                             for k, v in fields.items()}
                        self.apply_gate_event(d["cam_id"], int(d["local_id"]), d["direction"],
                                              float(d.get("at") or 0.0))
                    except Exception as exc:
                        logger.debug("[mtmc] bad gate event %s: %s", fields, exc)
        finally:
            self._gate_lock.release()

    def _publish_plate(self, cam_id: str, tr) -> None:
        """Queue a plate event when `tr`'s best plate beats the best one known for its global id
        ON THIS CAMERA. Every camera keeps its own best plate: they see the vehicle from different
        distances and angles, so which one reads best is decided later from what was read."""
        gid, pl = tr.global_id, tr.plate
        known = self.gid_plates.get((gid, cam_id))
        if known is not None and pl["score"] <= known["score"]:
            return
        self.gid_plates[(gid, cam_id)] = {"score": pl["score"], "conf": pl["conf"]}
        self._plate_events.append({"global_id": gid, "cam_id": cam_id, "local_id": tr.object_id,
                                   "conf": round(pl["conf"], 3), "crop": pl["crop"],
                                   "box": [round(v, 1) for v in pl["box"]]})

    def pop_plate_events(self) -> list[dict]:
        """Drain {global_id, cam_id, local_id, conf, box, crop} for plates that are the best yet
        seen on a vehicle (crop is a BGR array, or None when no frame copy was available)."""
        out = []
        while self._plate_events:
            out.append(self._plate_events.popleft())
        return out

    def pop_exit_events(self) -> list[dict]:
        out = []
        while self._exit_events:
            out.append(self._exit_events.popleft())
        return out

    def pop_assignment_events(self) -> list[dict]:
        """Drain (cam_id, local_id, global_id, pts_first, pts_last) for tracklets that were
        just given their global id."""
        out = []
        while self._assignment_events:
            out.append(self._assignment_events.popleft())
        return out

    def _submit_gallery(self, cam_id, tr, gid, frame_num, pts, frame_bgr) -> None:
        """Offer the tracklet's highest-confidence shot to the thumbnail gallery; the gallery
        keeps it only if it beats the one it already holds for this camera and vehicle."""
        if tr.best_shot is None or not tr.boxes:
            return
        conf, crop = tr.best_shot
        score = conf
        # A shot with the plate showing (the frame its best plate was detected in) is the picture of the truck,
        # whatever the detector confidence of the others: +1 puts every such shot above every one without.
        shown = (tr.plate or {}).get("vehicle_crop")
        if shown is not None and shown.size:
            crop, score = shown, 1.0 + float(tr.plate["conf"])
        self.vehicle_gallery.maybe_submit(cam_id, gid, crop, score, conf, tr.boxes[-1], pts)

    def _maybe_dump_gold(
        self, cam_id: str, tr: "TrackletRecord", gid: int, pts: float,
        frame_bgr: np.ndarray | None,
    ) -> None:
        """Phase 0 one-off gold-label capture (see gold_dump_dir docstring above).

        Fires exactly once per (cam_id, local_id) -- the first frame it gets a
        global_id -- since that assignment is locked for the tracklet's lifetime
        (see the ``existing_gtr`` fast-path in ``get_or_create_global_id``).

        Saves a generously-padded crop around the tracklet's last known box (not
        the tight reid crop) -- for a plate-only detector the tight box is only
        ~20x20px, unreadable for a human reviewer. This dump image is for human
        eyes only; it never feeds the reid/merge decision.
        """
        if gid < 0:
            return
        key = (cam_id, tr.object_id)
        if key in self._gold_dumped_keys:
            return
        crops = tr.get_crops()
        if not crops:
            return  # tracklet has no observation yet -- retry next frame
        import cv2
        crop_name = f"{cam_id}_{tr.object_id}.jpg"
        dump_img = self._padded_review_crop(tr, frame_bgr)
        if dump_img is None:
            dump_img = crops[len(crops) // 2]  # fallback: the tight reid crop
        cv2.imwrite(str(self._gold_dump_dir / "crops" / crop_name), dump_img)
        with open(self._gold_dump_dir / "summary.jsonl", "a") as f:
            f.write(json.dumps({
                "cam_id": cam_id, "local_id": tr.object_id, "global_id": gid,
                "pts": round(pts, 2), "crop": f"crops/{crop_name}",
            }) + "\n")
        self._gold_dumped_keys.add(key)

    @staticmethod
    def _padded_review_crop(
        tr: "TrackletRecord", frame_bgr: np.ndarray | None,
        pad_factor: float = 3.0, min_pad_px: float = 180.0,
    ) -> np.ndarray | None:
        """Crop a padded region around ``tr``'s last box from the current frame.

        Padding is the larger of pad_factor (3x the box's own width/height) and
        a fixed min_pad_px per side -- a plate-only box is ~20x20px, so a purely
        multiplicative pad would still crop to a postage stamp. min_pad_px=180
        (each side) targets a full-vehicle-sized window regardless of how tiny
        the detected box is, clamped to the frame bounds.
        """
        if frame_bgr is None:
            return None
        box = tr.last_box_xyxy
        if box is None:
            return None
        x0, y0, x1, y1 = box
        w, h = (x1 - x0), (y1 - y0)
        if w <= 0 or h <= 0:
            return None
        pad_x = max(w * (pad_factor - 1) / 2, min_pad_px)
        pad_y = max(h * (pad_factor - 1) / 2, min_pad_px)
        H, W = frame_bgr.shape[:2]
        px0 = max(0, int(x0 - pad_x))
        py0 = max(0, int(y0 - pad_y))
        px1 = min(W, int(x1 + pad_x))
        py1 = min(H, int(y1 + pad_y))
        if px1 <= px0 or py1 <= py0:
            return None
        return frame_bgr[py0:py1, px0:px1].copy()

    def get_global_ids(self, source_id: str) -> dict[int, int]:
        """Get latest global ID mapping for a camera (no frame processing)."""
        if source_id not in self._cameras:
            return {}
        return {tr.object_id: tr.global_id
                for tr in self._cameras[source_id].all_tracklets()
                if tr.global_id >= 0}

    def get_all_global_ids(self) -> dict[str, dict[int, int]]:
        """Get latest global ID mappings for all cameras."""
        return {cid: self.get_global_ids(cid) for cid in self._cameras}

    def stats(self) -> dict[str, Any]:
        """Return current MTMC state for logging/telemetry."""
        cam_stats = {}
        n_cross = 0
        for cam_id, cam in self._cameras.items():
            embedded = cam.embedded_count()
            cam_gids = sum(1 for tr in cam.all_tracklets() if tr.global_id >= 0)
            n_cross += cam_gids
            cam_stats[cam_id] = {
                "tracklets": cam.active_count(),
                "total_embedded": embedded,
                "with_global_id": cam_gids,
            }
        return {
            "cameras": len(self._cameras),
            "frames": self._total_frames,
            "global_tracks": len(self.association.global_tracks),
            "per_camera": cam_stats,
        }

    def _submit_embeddings(self) -> None:
        """Queue tracklets needing an embedding for the async worker.

        Cheap — no model compute here.  Each tracklet is embedded exactly once:
        once it has accumulated ``reid_min_crops`` crops it is queued (guarded
        by ``_embed_submitted`` so it isn't re-queued every frame), and the
        worker sets ``embedding`` once.  No per-frame re-embedding.
        """
        if self._reid is None or not self._reid.ensure_loaded():
            return
        for cam_id, cam in self._cameras.items():
            for tr in cam.all_tracklets():
                if tr.embedding is not None:
                    continue
                if tr._embed_submitted:
                    continue
                if not tr.embeddable:
                    continue
                if len(tr.get_crops()) < self.cfg.reid_min_crops:
                    continue
                tr._embed_submitted = True
                self._embed_queue.put(tr)
                logger.debug("[mtmc] queued embed %s tid=%d crops=%d",
                             cam_id, tr.object_id, len(tr.get_crops()))

    def _embed_loop(self) -> None:
        """Background worker: embed queued tracklets, set their embedding."""
        while True:
            try:
                tr = self._embed_queue.get()
            except Exception:
                continue
            if tr is None or self._embed_stop.is_set():
                self._embed_queue.task_done()
                return
            try:
                n_crops = len(tr.get_crops())
                emb = self._reid.embed_tracklet(tr) if self._reid is not None else None
                if emb is not None:
                    logger.debug("[mtmc] async embedded %s tid=%d crops=%d",
                                 tr.cam_id, tr.object_id, n_crops)
                else:
                    logger.debug("[mtmc] async embed failed %s tid=%d crops=%d",
                                 tr.cam_id, tr.object_id, n_crops)
            except Exception as exc:
                logger.warning("[mtmc] async embed failed %s tid=%d: %s",
                               getattr(tr, "cam_id", "?"), getattr(tr, "object_id", "?"), exc)
            finally:
                tr._embed_submitted = False
                self._embed_queue.task_done()

    def _assign_local_ids(self, source_id: str) -> None:
        """Assign per-camera local IDs to tracklets without a global_id.

        When Re-ID is unavailable, fallback assigns stable numeric IDs.
        """
        if source_id not in self._cameras:
            return
        cam = self._cameras[source_id]
        for tr in cam.all_tracklets():
            if tr.global_id < 0:
                self._local_id_counter += 1
                tr.global_id = self._local_id_counter

    def _build_ds_global_ids(
        self,
        source_id: str,
        ds_oid_map: dict[int, int],
    ) -> dict[int, int]:
        """Map internal global_ids back to DeepStream object_ids."""
        if source_id not in self._cameras or not ds_oid_map:
            return {}

        cam = self._cameras[source_id]
        inv_map: dict[int, int] = {v: k for k, v in ds_oid_map.items()}

        result: dict[int, int] = {}
        for tr in cam.all_tracklets():
            if tr.global_id >= 0:
                ds_oid = inv_map.get(tr.object_id)
                if ds_oid is not None:
                    result[ds_oid] = tr.global_id
        return result
