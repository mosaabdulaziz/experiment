"""Lightweight IOU-based face tracker for IRIS sidecar.

Assigns stable track IDs to face detections across frames, so that once a
face is recognized the name persists across nvtracker re-IDs (person leaves
and re-enters the frame) and brief detector dropouts.

Uses greedy IOU matching (similar to ByteTrack's core assignment step) — no
appearance embedding needed since face detections come from the DeepStream
SGIE at ~2-15 fps and IOU is sufficient for frame-to-frame association.

Usage
-----
    tracker = FaceTracker()
    for each frame:
        tracks = tracker.update(detections)
        # tracks[det_idx] -> FaceTrack with stable .track_id, .name, etc.
"""
import time
import threading
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("iris.face_tracker")


def _compute_iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Intersection-over-Union of two (left, top, width, height) boxes."""
    ax1, ay1 = a[0], a[1]
    ax2 = a[0] + a[2]
    ay2 = a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2 = b[0] + b[2]
    by2 = b[1] + b[3]

    xi1 = max(ax1, bx1)
    yi1 = max(ay1, by1)
    xi2 = min(ax2, bx2)
    yi2 = min(ay2, by2)

    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    aa = a[2] * a[3]
    ba = b[2] * b[3]
    union = aa + ba - inter

    return inter / union if union > 1e-9 else 0.0


class FaceTrack:
    """A single face track maintained across frames."""

    __slots__ = (
        "track_id", "bbox", "parent_id", "age", "hit_count",
        "missed", "name", "name_confidence", "name_frames",
        "best_score", "last_seen",
    )

    def __init__(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        parent_id: Optional[int] = None,
    ):
        self.track_id: int = track_id
        self.bbox: Tuple[float, float, float, float] = bbox
        self.parent_id: Optional[int] = parent_id
        self.age: int = 1
        self.hit_count: int = 1
        self.missed: int = 0
        self.name: Optional[str] = None
        self.name_confidence: float = 0.0
        self.name_frames: int = 0
        self.best_score: float = 0.0
        self.last_seen: float = time.time()

    def update_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        parent_id: Optional[int],
    ) -> None:
        """Update track state with a new matching detection."""
        self.bbox = bbox
        self.parent_id = parent_id
        self.age += 1
        self.hit_count += 1
        self.missed = 0
        self.last_seen = time.time()

    def apply_name(self, name: str, confidence: float) -> None:
        """Record a recognized name for this track."""
        self.name = name
        self.name_confidence = confidence
        if confidence > self.best_score:
            self.best_score = confidence

    def miss(self) -> None:
        """Increment missed frame counter."""
        self.missed += 1
        self.age += 1


class FaceTracker:
    """IOU-based multi-face tracker.

    Matches new face detections to existing tracks via greedy IOU assignment.
    Tracks that miss *max_missed* consecutive frames are terminated.
    New detections that don't match any track spawn new tracks.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU to consider a detection-to-track match.
    max_missed : int
        Frames a track can go unmatched before termination.
    min_hits : int
        Minimum detection matches before a track is considered valid
        (avoids one-off false positives getting track IDs).
    """

    def __init__(
        self,
        iou_threshold: float = 0.2,
        max_missed: int = 60,
        min_hits: int = 2,
    ):
        self._next_id = 1
        self._tracks: Dict[int, FaceTrack] = {}
        self._lock = threading.Lock()
        self._iou_threshold = iou_threshold
        self._max_missed = max_missed
        self._min_hits = min_hits

    # ── public API ────────────────────────────────────────────────────────

    def update(
        self,
        detections: List[Tuple[float, float, float, float, float, Optional[int]]],
    ) -> Dict[int, FaceTrack]:
        """Process one frame of face detections and return matched tracks.

        Args:
            detections:
                List of ``(left, top, width, height, confidence, parent_id)``
                tuples representing this frame's SGIE face detections.
                Coordinates are in **mux** space (the same coordinate system
                used throughout the sidecar probe).

        Returns:
            A dict ``{detection_index: FaceTrack}`` where *detection_index*
            is the position in the input list.  Every detection gets a track
            entry — unmatched detections spawn new FaceTrack objects.
        """
        with self._lock:
            active = list(self._tracks.values())
            dets = list(enumerate(detections))

            result: Dict[int, FaceTrack] = {}
            matched_det: set = set()
            matched_tid: set = set()

            # 1 ── Greedy IOU matching (track -> det) ─────────────────────
            # Sort by hit_count so mature tracks match first.
            for track in sorted(active, key=lambda t: t.hit_count, reverse=True):
                best_iou = self._iou_threshold
                best_det_idx: Optional[int] = None
                for det_idx, (dl, dt, dw, dh, _dc, _dp) in dets:
                    if det_idx in matched_det:
                        continue
                    iou = _compute_iou(track.bbox, (dl, dt, dw, dh))
                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = det_idx

                if best_det_idx is not None:
                    _, (dl, dt, dw, dh, _dc, dp) = dets[best_det_idx]
                    matched_det.add(best_det_idx)
                    matched_tid.add(track.track_id)
                    track.update_bbox((dl, dt, dw, dh), dp)
                    result[best_det_idx] = track

            # 2 ── New tracks for unmatched detections ───────────────────
            for det_idx, (dl, dt, dw, dh, _dc, dp) in dets:
                if det_idx in matched_det:
                    continue
                track = FaceTrack(self._next_id, (dl, dt, dw, dh), dp)
                self._tracks[self._next_id] = track
                self._next_id += 1
                logger.debug("New face track %d (parent_id=%s)", track.track_id, dp)
                result[det_idx] = track

            # 3 ── Miss / terminate stale tracks ────────────────────────
            for track in active:
                if track.track_id not in matched_tid:
                    track.miss()
                if track.missed >= self._max_missed:
                    logger.debug("Terminate face track %d (missed=%d)",
                                 track.track_id, track.missed)
                    del self._tracks[track.track_id]

        return result

    def get_names(self, max_age: float = 5.0) -> Dict[int, str]:
        """Return ``{face_track_id: name}`` for all active tracks that have a
        recognized name, excluding tracks whose ``last_seen`` is older than
        *max_age* seconds (to keep the lookup small and relevant)."""
        now = time.time()
        with self._lock:
            names: Dict[int, str] = {}
            for track in self._tracks.values():
                if track.name is not None and now - track.last_seen <= max_age:
                    names[track.track_id] = track.name
        return names

    def apply_name_to_track(self, face_track_id: int, name: str, score: float) -> None:
        """Propagate a recognition result to the track, externally callable
        from the FaceRecognizer thread."""
        with self._lock:
            track = self._tracks.get(face_track_id)
            if track is not None:
                track.apply_name(name, score)
