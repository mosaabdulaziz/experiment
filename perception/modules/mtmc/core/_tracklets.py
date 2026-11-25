"""Per-camera tracklet state with internal IoU tracking.

DeepStream's NvDCF tracker can produce unstable object_ids on distant cameras
(e.g. overhead traffic footage).  This module runs its OWN lightweight IoU
tracker on the detection boxes, producing stable internal track_ids.  A
DeepStream object_id → track_id mapping is maintained so that global_ids can
be published back in the format the frontend expects.

Design mirrors the local ``mtmc/tracker.py`` IoU tracker but operates
incrementally (one frame at a time) for real-time use.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("iris.mtmc")

_VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "bicycle", "license_plate", "plate", "license plate"})
# MTMC operates on vehicle and license plate targets.
_DEFAULT_TRACKED_CLASSES = _VEHICLE_CLASSES# IoU helpers removed — CameraTracklets now uses DS object_ids directly.


# ---------------------------------------------------------------------------
# Tracklet record (same interface as before for compatibility with _reid.py)
# ---------------------------------------------------------------------------

@dataclass
class TrackletRecord:
    """State for one tracked object within a single camera.

    ``object_id`` is the INTERNAL IoU-tracker id, NOT DeepStream's NvDCF id.
    The mapping from DS object_id → internal id is maintained by
    ``CameraTracklets``.
    """

    cam_id: str
    object_id: int  # internal IoU-tracker id
    start_frame: int = 0
    last_frame: int = 0
    start_pts: float = 0.0
    last_pts: float = 0.0

    # Sliding window of recent observations (bounded by max_window)
    frames: list[int] = field(default_factory=list)
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)  # (x, y, w, h)
    crops: list[np.ndarray] = field(default_factory=list)  # BGR crops for Re-ID

    # Re-ID state
    embedding: np.ndarray | None = None
    embedding_frame: int = 0       # frame at which embedding was last computed
    embedding_count: int = 0       # number of embed passes completed
    _embed_submitted: bool = False # True while queued for / in progress on the async embed worker

    # Cross-camera assignment
    global_id: int = -1            # -1 = unassigned

    # Config (injected at construction)
    _max_window: int = 30
    _max_crops: int = 16
    _min_embed_frames: int = 3

    # Time-spread "views" for the VLM: the largest crop seen in each
    # ``_view_bucket_s``-second bucket of the tracklet's life (bucket -> (area, crop)).
    # ``crops`` above is only the recent sliding window (a second or two); views
    # cover the whole pass so the VLM sees the vehicle at different moments.
    _view_bucket_s: float = 1.0
    _max_views: int = 12
    _views: dict = field(default_factory=dict)

    # The shot of this vehicle with the highest detector confidence so far: (conf, crop). It is what
    # the thumbnail gallery offers; nothing else (size, shape) takes part in choosing it.
    best_shot: tuple | None = None
    # Sightings whose crop was too small to stand for the vehicle (they gave no best shot / Re-ID crop / view).
    small_seen: int = 0

    # Best licence plate seen on this vehicle: {"conf", "score", "box", "crop", "pts", "seen"}
    # where score = conf * area (a big sharp plate beats a small confident one) and box is in
    # frame coords. None until the plate detector has found one inside this vehicle's box.
    plate: dict | None = None

    def note_plate(self, box: tuple[float, float, float, float], conf: float, pts: float,
                   crop: np.ndarray | None, vehicle_crop: np.ndarray | None = None) -> bool:
        """Record a plate seen on this vehicle; True when it is the best one so far.
        vehicle_crop is the whole vehicle in the same frame: the picture of it with its plate showing."""
        seen = (self.plate or {}).get("seen", 0) + 1
        score = conf * box[2] * box[3]
        if self.plate is not None and score <= self.plate["score"]:
            self.plate["seen"] = seen
            return False
        self.plate = {"conf": conf, "score": score, "box": box, "crop": crop, "pts": pts, "seen": seen,
                      "vehicle_crop": vehicle_crop}
        return True

    def add_observation(
        self,
        frame_num: int,
        pts: float,
        box: tuple[float, float, float, float],
        crop: np.ndarray | None = None,
        conf: float | None = None,
    ) -> None:
        """Record a new observation for this tracklet."""
        if crop is not None and conf is not None and (self.best_shot is None or conf > self.best_shot[0]):
            self.best_shot = (float(conf), crop.copy())
        self.last_frame = frame_num
        self.last_pts = pts
        self.frames.append(frame_num)
        self.boxes.append(box)
        if crop is not None:
            b = int(pts // self._view_bucket_s)
            area = crop.shape[0] * crop.shape[1]
            if b not in self._views or area > self._views[b][0]:
                self._views[b] = (area, crop)
                if len(self._views) > self._max_views:
                    # Drop the smallest crop, but never the first/last bucket (keeps the time span).
                    keys = sorted(self._views)
                    victim = min(keys[1:-1], key=lambda k: self._views[k][0])
                    del self._views[victim]
            self.crops.append(crop)
            if len(self.crops) > self._max_crops:
                self.crops = self.crops[-self._max_crops:]
        # Bound the observation window
        if len(self.frames) > self._max_window:
            excess = len(self.frames) - self._max_window
            self.frames = self.frames[excess:]
            self.boxes = self.boxes[excess:]
            if len(self.crops) > self._max_crops:
                self.crops = self.crops[-self._max_crops:]

    @property
    def duration_s(self) -> float:
        return self.last_pts - self.start_pts

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def has_embedding(self) -> bool:
        return self.embedding is not None

    @property
    def embeddable(self) -> bool:
        """True if this tracklet has enough frames to compute an embedding."""
        return self.num_frames >= self._min_embed_frames

    @property
    def last_box_xyxy(self) -> np.ndarray | None:
        """Last box in xyxy format, or None if no observations."""
        if not self.boxes:
            return None
        x, y, w, h = self.boxes[-1]
        return np.array([x, y, x + w, y + h], dtype=float)

    def get_view_crops(self) -> list[np.ndarray]:
        """Best crop per time bucket over the tracklet's life, oldest first."""
        return [self._views[k][1] for k in sorted(self._views)]

    def get_crops(self) -> list[np.ndarray]:
        """Return accumulated crops for Re-ID embedding."""
        return list(self.crops)


# ---------------------------------------------------------------------------
# CameraTracklets — IoU-based per-camera tracker
# ---------------------------------------------------------------------------

class CameraTracklets:
    """Manages all active tracklets for one camera.

    Uses DeepStream's object_ids directly as the tracklet identity.
    No internal IoU tracker — DeepStream's NvDCF/SORT already provides
    stable per-camera tracking.  This avoids the "double tracking" problem
    where an internal IoU matcher fights DeepStream's tracker and creates
    spurious new tracklets for small objects like license plates.
    """

    def __init__(self, cam_id: str, cfg):
        self.cam_id = cam_id
        self._cfg = cfg
        self._tracked = frozenset(getattr(cfg, 'tracked_classes', _DEFAULT_TRACKED_CLASSES))
        self._tracklets: dict[int, TrackletRecord] = {}  # ds_object_id → tracklet
        self._expired_tracklets: list[TrackletRecord] = []
        self._last_frame: int = 0
        # Per-tracklet last-seen frame for aging
        self._last_seen: dict[int, int] = {}

    def update(
        self,
        frame_num: int,
        pts: float,
        dets: list[list],
        frame_bgr: np.ndarray | None = None,
        boxes: list[tuple] | None = None,
    ) -> dict[int, int]:
        """Update tracklets with this frame's detections.

        Uses DeepStream object_ids directly — no internal IoU matching.
        """
        self._last_frame = frame_num

        # Collect this frame's vehicle detections keyed by ds_object_id
        frame_oids: dict[int, tuple] = {}  # ds_oid → (left, top, w, h, label, conf)
        if boxes:
            for left, top, w, h, label, conf, ds_oid in boxes:
                if label not in self._tracked:
                    continue
                frame_oids[int(ds_oid)] = (left, top, w, h, label, conf)

        seen_oids: set[int] = set()

        # ── 1. Update existing tracklets ──────────────────────────────────
        for ds_oid, (left, top, w, h, label, conf) in frame_oids.items():
            if ds_oid in self._tracklets:
                # Existing tracklet — append observation
                tr = self._tracklets[ds_oid]
                crop = self._representative(self._extract_crop(frame_bgr, left, top, w, h), tr)
                tr.add_observation(frame_num, pts, (left, top, w, h), crop, conf)
                self._last_seen[ds_oid] = frame_num
            else:
                # New DS oid — create tracklet keyed by ds_oid
                raw_crop = self._extract_crop(frame_bgr, left, top, w, h)
                crop = self._representative(raw_crop)
                tr = TrackletRecord(
                    cam_id=self.cam_id,
                    object_id=ds_oid,
                    start_frame=frame_num,
                    last_frame=frame_num,
                    start_pts=pts,
                    last_pts=pts,
                    _max_window=self._cfg.max_unmatched_frames * 2,
                    _max_crops=self._cfg.reid_max_crops,
                    _min_embed_frames=self._cfg.min_tracklet_frames,
                )
                if raw_crop is not None and crop is None:
                    tr.small_seen += 1
                tr.add_observation(frame_num, pts, (left, top, w, h), crop, conf)
                self._tracklets[ds_oid] = tr
                self._last_seen[ds_oid] = frame_num
            seen_oids.add(ds_oid)

        # ── 2. Age tracklets not seen this frame ──────────────────────────
        disappeared = []
        for ds_oid in list(self._tracklets.keys()):
            if ds_oid not in seen_oids:
                frames_missed = frame_num - self._last_seen.get(ds_oid, frame_num)
                if frames_missed > self._cfg.max_unmatched_frames:
                    expired_tr = self._tracklets.pop(ds_oid, None)
                    self._last_seen.pop(ds_oid, None)
                    if expired_tr is not None:
                        self._expired_tracklets.append(expired_tr)

        # ds_oid_map is identity: ds_oid == internal_id
        return {oid: oid for oid in seen_oids}

    def add_plates(self, plates, pts: float, frame_bgr=None) -> list[tuple[TrackletRecord, bool]]:
        """Attach this frame's plate detections to their vehicles.

        plates: (left, top, width, height, conf, parent_object_id) in frame coords. The detector
        ran on the vehicle's crop, so parent_object_id is the vehicle it belongs to; when that
        id is missing or no longer a live tracklet, the smallest vehicle box containing the
        plate's centre is used. A plate that fits no vehicle is dropped.
        Returns [(tracklet, is_new_best)] for the plates that were attached.
        """
        out = []
        for left, top, w, h, conf, parent in plates:
            tr = self._tracklets.get(int(parent)) if parent is not None and parent >= 0 else None
            if tr is None:
                cx, cy = left + w / 2.0, top + h / 2.0
                best_area = float("inf")
                for cand in self._tracklets.values():
                    if not cand.boxes or cand.last_pts != pts:
                        continue
                    bx, by, bw, bh = cand.boxes[-1]
                    if bx <= cx <= bx + bw and by <= cy <= by + bh and bw * bh < best_area:
                        best_area, tr = bw * bh, cand
            if tr is None:
                continue
            crop = self._extract_crop(frame_bgr, left, top, w, h)
            vehicle = self._representative(self._extract_crop(frame_bgr, *tr.boxes[-1])) if tr.boxes else None
            out.append((tr, tr.note_plate((left, top, w, h), float(conf), pts, crop, vehicle)))
        return out

    def _representative(self, crop, tr=None):
        """The crop when it is big enough to stand for the vehicle (see MTMCConfig.min_crop_side_px), else None
        (and, given the tracklet, counted as a too-small sighting of it)."""
        if crop is None:
            return None
        h, w = crop.shape[:2]
        if min(h, w) < self._cfg.min_crop_side_px or h * w < self._cfg.min_crop_area_px:
            if tr is not None:
                tr.small_seen += 1
            return None
        return crop

    @staticmethod
    def _extract_crop(frame_bgr, left, top, w, h):
        if frame_bgr is None:
            return None
        x0, y0 = int(max(left, 0)), int(max(top, 0))
        x1, y1 = int(min(left + w, frame_bgr.shape[1])), int(min(top + h, frame_bgr.shape[0]))
        if x1 > x0 and y1 > y0:
            return frame_bgr[y0:y1, x0:x1].copy()
        return None

    def ds_oid_map(self) -> dict[int, int]:
        """Return the latest {ds_object_id: internal_track_id} mapping.
        Now identity: ds_oid == internal_id."""
        return {oid: oid for oid in self._tracklets}

    def embeddable_tracklets(self) -> list[TrackletRecord]:
        """Return tracklets that have enough frames for Re-ID embedding."""
        return [tr for tr in self._tracklets.values() if tr.embeddable]

    def all_tracklets(self) -> list[TrackletRecord]:
        """Return all active tracklets."""
        return list(self._tracklets.values())

    def active_count(self) -> int:
        return len(self._tracklets)

    def embedded_count(self) -> int:
        return sum(1 for tr in self._tracklets.values() if tr.has_embedding)

    def pop_expired_tracklets(self) -> list[TrackletRecord]:
        """Return and clear tracklets that aged out during update."""
        expired = self._expired_tracklets
        self._expired_tracklets = []
        return expired

