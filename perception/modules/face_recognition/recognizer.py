"""Face recognition module for IRIS DeepStream sidecar.

Uses InsightFace (ArcFace) for face embedding extraction and matching.
Runs in the GPU-enabled sidecar container.
"""
import os
import sys
import math
import time
import json
import logging
import threading
import queue
from collections import deque
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import cv2
import redis

from insightface.app import FaceAnalysis

from perception import insightface_pack
from insightface.utils import face_align

from perception.engine import telemetry  # benchmarking-mode per-stage timing (no-op when disabled)

logger = logging.getLogger("iris.face_recognizer")

# Redis key for face gallery
FACE_GALLERY_KEY = "iris:face:gallery"
# Redis key for face recognition results (published alongside meta)
FACE_REC_KEY_PREFIX = "iris:face_rec:"
# Redis key for CCTV face captures
FACE_CAPTURES_KEY = "iris:face:captures"
# Redis prefix for individual capture image blobs
FACE_CAPTURE_IMG_PREFIX = "iris:face:capture_img:"


class FaceGallery:
    """Manages the known face gallery in Redis."""
    
    def __init__(self, redis_client: redis.Redis, gallery_key: str = FACE_GALLERY_KEY):
        self.r = redis_client
        self.gallery_key = gallery_key
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    @staticmethod
    def _normalize_entry(entry: dict) -> dict:
        """Backward compat: ensure the entry has an ``embeddings`` list.
        Old entries stored a single ``embedding`` — wrap it.
        """
        if "embeddings" not in entry:
            emb = entry.get("embedding")
            entry["embeddings"] = [emb] if emb is not None else []
        return entry

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """Load all known faces from Redis into memory."""
        try:
            data = self.r.hgetall(self.gallery_key)
            with self._lock:
                self._cache = {}
                for k, v in data.items():
                    k = k.decode() if isinstance(k, bytes) else k
                    v = v.decode() if isinstance(v, bytes) else v
                    self._cache[k] = self._normalize_entry(json.loads(v))
            logger.info("Loaded %d faces from gallery (total %d embeddings)",
                        len(self._cache),
                        sum(len(e["embeddings"]) for e in self._cache.values()))
            return self._cache
        except Exception as e:
            logger.error("Failed to load face gallery: %s", e)
            return {}
    
    def get_embeddings(self) -> Tuple[List[str], List[str], np.ndarray]:
        """Get all embeddings as a matrix for fast matching.
        
        Each person may have multiple embeddings (from different angles /
        lighting conditions).  All of them are flattened into the output
        matrix so the recognizer can match against any of them.
        
        Returns:
            (face_ids, person_names, embeddings_matrix)
            face_ids[i] maps back to the gallery key, person_names[i] is
            the display name, and embeddings_matrix is (N, 512).
        """
        with self._lock:
            if not self._cache:
                return [], [], np.empty((0, 512), dtype=np.float32)
            face_ids: List[str] = []
            names: List[str] = []
            all_embs: List[np.ndarray] = []
            for fid, entry in self._cache.items():
                embs = entry.get("embeddings", [])
                if not embs:
                    continue
                name = entry["name"]
                for emb in embs:
                    face_ids.append(fid)
                    names.append(name)
                    all_embs.append(np.array(emb, dtype=np.float32))
            if not all_embs:
                return [], [], np.empty((0, 512), dtype=np.float32)
            return face_ids, names, np.stack(all_embs, axis=0)
    
    def add(self, face_id: str, name: str, embedding: np.ndarray, photo_path: str = "") -> bool:
        """Add a face to the gallery."""
        try:
            entry = {
                "name": name,
                "embeddings": [embedding.tolist()],
                "photo_path": photo_path,
                "created_at": time.time()
            }
            self.r.hset(self.gallery_key, face_id, json.dumps(entry))
            with self._lock:
                self._cache[face_id] = entry
            logger.info("Added face to gallery: %s (%s)", face_id, name)
            return True
        except Exception as e:
            logger.error("Failed to add face to gallery: %s", e)
            return False
    
    def append_embedding(self, face_id: str, embedding: np.ndarray) -> bool:
        """Append an additional embedding to an existing gallery entry.
        Used when assigning a CCTV capture to a known person — the
        capture's embedding is appended so the recognizer can match
        against it in future frames.
        """
        try:
            raw = self.r.hget(self.gallery_key, face_id)
            if raw is None:
                return False
            entry = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            entry = self._normalize_entry(entry)
            entry["embeddings"].append(embedding.tolist())
            self.r.hset(self.gallery_key, face_id, json.dumps(entry))
            with self._lock:
                self._cache[face_id] = entry
            logger.info("Appended embedding to face %s (now %d embeddings)",
                        face_id, len(entry["embeddings"]))
            return True
        except Exception as e:
            logger.error("Failed to append embedding to face %s: %s", face_id, e)
            return False
    
    def remove(self, face_id: str) -> bool:
        """Remove a face from the gallery."""
        try:
            self.r.hdel(self.gallery_key, face_id)
            with self._lock:
                self._cache.pop(face_id, None)
            logger.info("Removed face from gallery: %s", face_id)
            return True
        except Exception as e:
            logger.error("Failed to remove face from gallery: %s", e)
            return False
    
    def list_all(self) -> List[Dict[str, Any]]:
        """List all faces in the gallery."""
        with self._lock:
            return [
                {"id": k, "name": v["name"], "photo_path": v.get("photo_path", ""), "created_at": v.get("created_at", 0)}
                for k, v in self._cache.items()
            ]


class FaceCaptureStore:
    """Manages CCTV face captures — unknown faces from live footage that an
    operator can later assign to known persons via the web UI.

    Captures are stored in Redis hash ``iris:face:captures`` (keyed by capture
    UUID) with metadata + embedding.  Thumbnail images are also saved to a
    shared disk volume so the API container can serve them to the frontend.
    """

    def __init__(self, redis_client: redis.Redis, captures_dir: str):
        self.r = redis_client
        self.captures_dir = captures_dir
        os.makedirs(captures_dir, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────

    def add(self, capture_id: str, embedding: np.ndarray,
            source_id: str, track_id: int, pts: float,
            image_bytes: bytes) -> bool:
        """Store a new face capture."""
        try:
            entry = {
                "embedding": embedding.tolist(),
                "source_id": source_id,
                "track_id": track_id,
                "pts": round(pts, 3),
                "captured_at": time.time(),
                "assigned_to": None,
            }
            self.r.hset(FACE_CAPTURES_KEY, capture_id, json.dumps(entry))
            # Save thumbnail to disk for serving via the API
            img_path = os.path.join(self.captures_dir, f"{capture_id}.jpg")
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            # Also store the image bytes in Redis for fast retrieval by the API
            self.r.setex(f"{FACE_CAPTURE_IMG_PREFIX}{capture_id}", 86400, image_bytes)
            logger.info("Face capture saved: %s (source=%s track=%d)",
                        capture_id, source_id, track_id)
            return True
        except Exception as e:
            logger.error("Failed to save face capture %s: %s", capture_id, e)
            return False

    def list_all(self) -> List[Dict[str, Any]]:
        """Return all captures with metadata (no embeddings for listing)."""
        try:
            data = self.r.hgetall(FACE_CAPTURES_KEY)
            out = []
            for k, v in data.items():
                k = k.decode() if isinstance(k, bytes) else k
                v = v.decode() if isinstance(v, bytes) else v
                entry = json.loads(v)
                out.append({
                    "id": k,
                    "source_id": entry["source_id"],
                    "track_id": entry["track_id"],
                    "pts": entry["pts"],
                    "captured_at": entry["captured_at"],
                    "assigned_to": entry.get("assigned_to"),
                })
            return sorted(out, key=lambda x: x["captured_at"], reverse=True)
        except Exception as e:
            logger.error("Failed to list face captures: %s", e)
            return []

    def remove(self, capture_id: str) -> bool:
        """Delete a capture and its thumbnail."""
        try:
            self.r.hdel(FACE_CAPTURES_KEY, capture_id)
            self.r.delete(f"{FACE_CAPTURE_IMG_PREFIX}{capture_id}")
            img_path = os.path.join(self.captures_dir, f"{capture_id}.jpg")
            if os.path.exists(img_path):
                os.remove(img_path)
            logger.info("Face capture removed: %s", capture_id)
            return True
        except Exception as e:
            logger.error("Failed to remove face capture %s: %s", capture_id, e)
            return False

    def assign(self, capture_id: str, person_name: str) -> bool:
        """Mark a capture as assigned to a known person (BUT keep it in the
        captures list until the operator explicitly confirms the assignment
        goes through to the gallery).  The operator assigns a name via the UI;
        this just records the intent.  The API layer promotes it to the gallery."""
        try:
            raw = self.r.hget(FACE_CAPTURES_KEY, capture_id)
            if raw is None:
                return False
            entry = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            entry["assigned_to"] = person_name
            self.r.hset(FACE_CAPTURES_KEY, capture_id, json.dumps(entry))
            logger.info("Face capture %s assigned to '%s'", capture_id, person_name)
            return True
        except Exception as e:
            logger.error("Failed to assign face capture %s: %s", capture_id, e)
            return False

    def get_image_bytes(self, capture_id: str) -> Optional[bytes]:
        """Get the thumbnail JPEG bytes for a capture (from Redis cache, fallback disk)."""
        try:
            cached = self.r.get(f"{FACE_CAPTURE_IMG_PREFIX}{capture_id}")
            if cached is not None:
                return cached if isinstance(cached, bytes) else cached.encode()
        except Exception:
            pass
        # Fallback: read from disk
        img_path = os.path.join(self.captures_dir, f"{capture_id}.jpg")
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                return f.read()
        return None

    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Return {capture_id: embedding_array} for all captures, used for
        cross-track novelty checking.

        Captures without a real embedding are skipped.  Manual "add image"
        uploads from the web UI store an empty list, and an empty vector would
        collapse the novelty matrix to shape (N, 0) — crashing the recognition
        loop's dot product below.
        """
        try:
            data = self.r.hgetall(FACE_CAPTURES_KEY)
            out: Dict[str, np.ndarray] = {}
            for k, v in data.items():
                k = k.decode() if isinstance(k, bytes) else k
                v = v.decode() if isinstance(v, bytes) else v
                entry = json.loads(v)
                emb = entry.get("embedding")
                if not emb:
                    continue
                out[k] = np.array(emb, dtype=np.float32)
            return out
        except Exception as e:
            logger.error("Failed to load capture embeddings: %s", e)
            return {}


def verify_gpu_inference(app) -> tuple:
    """(ok, detail): did the recognition model really run on the GPU?

    A session built with the CUDA provider can still end up on the CPU without any error that reaches
    the caller: ONNX Runtime retries on the CPU when the cuDNN conv engines fail to load, and only logs
    it. The recognizer then works, ~16x slower, and nothing says so. So run one real inference and look
    at which provider the session ended up with."""
    rec = (getattr(app, "models", None) or {}).get("recognition")
    if rec is None:
        return True, "no recognition model"
    try:
        w, h = rec.input_size
        rec.session.run(None, {rec.input_name: np.zeros((1, 3, h, w), np.float32)})
    except Exception as exc:
        return False, f"warm-up inference failed: {str(exc).replace(chr(10), ' ')[:140]}"
    providers = list(rec.session.get_providers())
    if not providers or providers[0] != "CUDAExecutionProvider":
        return False, f"after a warm-up inference the session runs on {providers}"
    return True, "CUDAExecutionProvider"


class FaceRecognizer:
    """InsightFace-based face recognition engine."""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        model_name: str = "auraface",
        det_size: Tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        rec_thresh: float = 0.5,
        rec_margin: float = 0.0,  # open-set margin: best score must beat the best
                                  # OTHER identity's score by at least this much,
                                  # or the face stays "unknown" (0 = disabled).
                                  # Guards against one person being everyone's
                                  # highest-similarity match at a permissive
                                  # rec_thresh.
        ctx_id: int = 0,  # GPU device ID
        gallery_key: str = FACE_GALLERY_KEY,  # overridable so a separate deployment
                                               # (e.g. the attendance service) can point
                                               # at its own gallery without touching
                                               # this codebase's default behavior
        gallery_reload_interval: float = 30.0,  # seconds
        enhancer: Optional[Any] = None,  # FaceEnhancer instance (preprocess/gfpgan/none)
        # Per-track name stability (anti-flicker) tuning:
        track_vote_ttl: float = 4.0,      # keep votes newer than this many seconds
        track_min_votes: int = 1,         # votes needed before a label is adopted (1 = lock on first sighting)
        track_majority: float = 0.6,      # share of the window a name must hold to win
        track_lock_seconds: float = 3.0,  # label released after this long with no sighting
        track_max_jump: float = 0.35,     # fraction of frame diagonal: a locked label's track
                                          # reappearing farther than this = recycled track ID -> release lock
        track_reconfirm_seconds: float = 6.0,  # issue 7C: a name whose VISIBLE face hasn't matched
                                               # it for this long is dropped until re-earned
        align_enabled: bool = False,      # straighten faces with their 5 landmarks before
                                          # embedding, exactly as registration does (2026-09-12)
        align_debug_samples: int = 0,     # save this many raw + aligned face pairs, then stop
        align_debug_dir: str = "",
    ):
        self.r = redis_client
        self.rec_thresh = rec_thresh
        self.rec_margin = rec_margin
        self.gallery = FaceGallery(redis_client, gallery_key=gallery_key)
        self._gallery_reload_interval = gallery_reload_interval
        self._last_gallery_reload = 0.0
        self._enhancer = enhancer
        self._track_vote_ttl = track_vote_ttl
        self._track_min_votes = track_min_votes
        self._track_majority = track_majority
        self._track_lock_seconds = track_lock_seconds
        self._track_max_jump = track_max_jump
        self._track_reconfirm_seconds = track_reconfirm_seconds
        # Issue 7A: recent box-center samples for EVERY tracked person, fed by
        # note_alive_tracks: {source_id: {track_id: deque of (ts, cx, cy, h)}}.
        self._motion: Dict[str, Dict[int, deque]] = {}
        # Face straightening (2026-09-12) -- see _embed_aligned_face.
        self._align_enabled = align_enabled
        self._align_debug_left = max(0, int(align_debug_samples)) if align_debug_dir else 0
        self._align_debug_dir = align_debug_dir
        self._align_stats: Dict[str, int] = {"aligned": 0, "no_landmarks": 0,
                                             "implausible": 0, "failed": 0}
        self._align_stats_logged_at = time.time()
        
        # Initialize InsightFace. The detection module has to stay LOADED (this
        # insightface version hard-asserts 'detection' in FaceAnalysis.models) but
        # it is never RUN at inference: the sidecar's sgie_face (yolov8n-face)
        # already located every face, so each crop here IS a detection. Embeddings
        # come straight from the ArcFace recognition model (_embed_crop) instead
        # of app.get(), which would re-run SCRFD's 640x640 detection pass per crop
        # just to reach the same embedding. det_thresh/det_size are retained for
        # config compatibility and only feed the (idle) detector's prepare().
        logger.info("Initializing InsightFace with model=%s, ctx_id=%d", model_name, ctx_id)
        # the repo's DVC copy, never a download (raises InsightFacePackMissing with the `dvc pull` when absent)
        self.app = FaceAnalysis(name=model_name, root=insightface_pack.root(model_name),
                                allowed_modules=['detection', 'recognition'])
        self.app.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=det_size)
        logger.info("InsightFace initialized successfully (recognition-only at inference)")
        if ctx_id >= 0:
            on_gpu, detail = verify_gpu_inference(self.app)
            if on_gpu:
                logger.info("InsightFace recognition is on the GPU (%s)", detail)
            else:
                logger.error(
                    "InsightFace was asked for the GPU (face_recognition_ctx_id=%d) and is NOT on it: %s. "
                    "Recognition will be ~16x slower. Usual cause: a second cuDNN in the image (torch's "
                    "nvidia-cudnn pip package) that the loader cannot complete -- see the mtmc stage of "
                    "perception/pipeline/Dockerfile.", ctx_id, detail)
        
        # Load gallery
        self.gallery.load_all()
        self._last_gallery_reload = time.time()
        
        # Recognition queue and thread
        self._queue: queue.Queue = queue.Queue(maxsize=100)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Latest recognized name per (source_id -> person track id). Written by the
        # recognition thread, read by the sidecar probe to label boxes on the live
        # JPEG. {source_id: {parent_id: (name, score, wall_clock_ts)}}.
        self._names_lock = threading.Lock()
        self._names: Dict[str, Dict[int, Tuple[str, float, float]]] = {}
        # Per-track temporal vote state: {source_id: {parent_id: state}} where state
        # is {"name": locked name or None, "score": last conf, "votes": deque of
        # (name, ts), "last_seen": ts, "pos": last sighted face-center (capture px)}.
        # Used to smooth labels over a short window AND to catch recycled track IDs
        # (a locked label whose track teleports away is a different person).
        self._track_state: Dict[str, Dict[int, Dict[str, Any]]] = {}

        # CCTV face capture config (optional, off by default)
        self._capture_store: Optional[FaceCaptureStore] = None
        self._capture_enabled = False
        self._capture_min_pixels = 40
        self._capture_min_det_confidence = 0.5
        self._capture_novelty_threshold = 0.25
        # Per-source set of already-captured (source_id, track_id) tuples for dedup.
        self._captured_tracks: Dict[str, set] = {}
        self._captured_tracks_lock = threading.Lock()
        # Cache of existing capture embeddings for cross-track novelty checking,
        # reloaded periodically.
        self._capture_embeddings: Dict[str, np.ndarray] = {}
        self._last_capture_emb_load = 0.0
        self._capture_emb_reload_interval = 30.0

    def snapshot_names(self, source_id: str, max_age: float = 5.0) -> Dict[int, str]:
        """Return {person_track_id: name} for faces recognized on this source within
        the last `max_age` seconds. Used by the probe to overlay names on the video."""
        now = time.time()
        with self._names_lock:
            src = self._names.get(source_id, {})
            return {pid: name for pid, (name, _score, ts) in src.items()
                    if now - ts <= max_age}
    
    def _last_alive(self, st: Dict[str, Any]) -> float:
        """Latest moment this track's body was known present: its last
        confirming face match, or the tracker still reporting the body
        (note_alive_tracks). Issue 5 (2026-09-12): a confirmed name lives as
        long as this stays recent, not a fixed time since the last match."""
        return max(st.get("last_seen", 0.0), st.get("alive_at", 0.0))

    def _lock_live(self, st: Dict[str, Any], now: float) -> bool:
        """True while st holds a name whose body was present within
        track_lock_seconds -- the grace after the tracker loses the body."""
        return bool(st.get("name")) and now - self._last_alive(st) <= self._track_lock_seconds

    # Two tracker reports of the same body further apart than this = the
    # tracker lost it in between (it reports every tracked body every frame,
    # shadow-tracked ones included), so a position jump across that gap is
    # checked by the max_jump guard again.
    ALIVE_CONTINUITY_SECONDS = 1.0

    def note_alive_tracks(self, source_id: str,
                          person_boxes: List[Tuple[float, float, float, float, int]]) -> None:
        """Called by the sidecar probe on every meta frame with the person
        tracks the TRACKER still reports on this source (detected or
        shadow-tracked), boxes in capture pixels: (left, top, w, h, track_id).

        Issue 5 (2026-09-12): keeps a confirmed name on a body for as long as
        the tracker follows that body, even with the face turned away -- the
        live label used to drop it track_lock_seconds after the last face
        match. A track id that reappears after being gone longer than
        track_lock_seconds is treated as a NEW person (DeepStream recycles
        ids only after a track is lost), so its old name is released instead
        of carried over."""
        now = time.time()
        with self._names_lock:
            tracks = self._track_state.setdefault(source_id, {})
            src = self._names.setdefault(source_id, {})
            motion = self._motion.setdefault(source_id, {})
            for left, top, w, h, oid in person_boxes:
                # Issue 7A: motion history for EVERY tracked person -- a
                # crossing involves two bodies, named or not.
                motion.setdefault(oid, deque(maxlen=64)).append(
                    (now, left + w / 2.0, top + h / 2.0, h))
                st = tracks.get(oid)
                if st is None:
                    continue
                if now - st.get("alive_at", 0.0) > self.ALIVE_CONTINUITY_SECONDS:
                    st["continuity_broken"] = True
                if st.get("name") and now - self._last_alive(st) > self._track_lock_seconds:
                    # Body was gone past the grace: this id now belongs to
                    # whoever reappeared -- they must earn a name themselves.
                    st["name"] = None
                    st["score"] = 0.0
                    st["votes"].clear()
                    st["pos"] = None
                    src.pop(oid, None)
                st["alive_at"] = now
                if st.get("name"):
                    src[oid] = (st["name"], round(st["score"], 3), now)
            # Forget motion for bodies the tracker stopped reporting.
            for oid in [o for o, hist in motion.items()
                        if now - hist[-1][0] > self.MOTION_FORGET_SECONDS]:
                del motion[oid]
            self._collision_guard(source_id, person_boxes, tracks, src, now)

    # ── Crossing (collision) guard, issue 7A (2026-09-12) ─────────────────
    # Same thresholds as products/attendance's own guard
    # (acl.PersonPresenceTracker), so the live label and the table agree on
    # what counts as a crossing.
    # Two person boxes overlapping by at least this share of the SMALLER box
    # count as a crossing (intersection over the smaller box, not IoU: a near
    # person hiding a far one covers most of the far box while IoU stays
    # small -- exactly when a tracker swaps ids).
    COLLIDE_OVERLAP = 0.35
    # "Moving" = the box center travelled more than MOTION_FRACTION of the box
    # height (at least MOTION_MIN_PX) over the last MOTION_WINDOW seconds.
    MOTION_WINDOW = 3.5
    MOTION_MIN_SPAN = 0.5    # need at least this much history to judge
    MOTION_FRACTION = 0.2
    MOTION_MIN_PX = 12.0
    MOTION_FORGET_SECONDS = 10.0
    # A face processed within this long after a crossing may have been
    # captured DURING it (recognition runs off a queue), so it gets no vote.
    COLLISION_VOTE_HOLDOFF_SECONDS = 0.5
    # Re-confirmation (issue 7C): a name is only dropped after at least this
    # many visible-but-not-matching face sightings (not one blurry glimpse)...
    RECONFIRM_MIN_FACES = 3
    # ...and a face gap longer than this (face turned away) starts a fresh
    # re-check window, so time spent facing away never counts against it.
    RECONFIRM_FACE_GAP_SECONDS = 2.0

    @staticmethod
    def _overlap_ratio(a: Tuple[float, float, float, float],
                       b: Tuple[float, float, float, float]) -> float:
        """Intersection area over the SMALLER box's area (0 when disjoint)."""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
        smaller = min(aw * ah, bw * bh)
        return ix * iy / smaller if smaller > 0 else 0.0

    def _is_moving(self, source_id: str, track_id: int, now: float) -> bool:
        """True if the track's box center moved more than MOTION_FRACTION of
        its height (min MOTION_MIN_PX) over the last MOTION_WINDOW seconds.
        Compares the MEDIAN position of the older half of the window with the
        newer half, so detector box jitter doesn't read as walking. Too little
        history = not moving (no swap risk worth pausing for yet)."""
        hist = self._motion.get(source_id, {}).get(track_id)
        if not hist:
            return False
        recent = [s for s in hist if now - s[0] <= self.MOTION_WINDOW]
        if len(recent) < 4 or recent[-1][0] - recent[0][0] < self.MOTION_MIN_SPAN:
            return False
        half = len(recent) // 2
        older, newer = recent[:half], recent[half:]
        med = lambda vals: sorted(vals)[len(vals) // 2]
        dx = med([s[1] for s in newer]) - med([s[1] for s in older])
        dy = med([s[2] for s in newer]) - med([s[2] for s in older])
        h = med([s[3] for s in recent])
        return math.hypot(dx, dy) > max(self.MOTION_MIN_PX, self.MOTION_FRACTION * h)

    def _collision_guard(self, source_id: str,
                         person_boxes: List[Tuple[float, float, float, float, int]],
                         tracks: Dict[int, Dict[str, Any]],
                         src: Dict[int, Tuple[str, float, float]], now: float) -> None:
        """When two MOVING people's boxes overlap, the tracker may swap their
        ids -- so any name on either is paused and its votes cleared: each
        must re-earn a name from fresh face matches after the crossing (the
        same name comes straight back if the face agrees). A walker passing
        someone standing still pauses nobody -- trackers essentially never
        swap a moving body with a still one. Caller holds self._names_lock."""
        boxes = {oid: (l, t, w, h) for l, t, w, h, oid in person_boxes}
        ids = list(boxes)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if self._overlap_ratio(boxes[a], boxes[b]) < self.COLLIDE_OVERLAP:
                    continue
                if not (self._is_moving(source_id, a, now) and self._is_moving(source_id, b, now)):
                    continue
                for oid, other in ((a, b), (b, a)):
                    st = tracks.get(oid)
                    if st is None:
                        continue
                    st["collision_at"] = now
                    st["votes"].clear()
                    if st.get("name"):
                        logger.info("Face track %s on %s: collision with moving track %s -- name '%s' "
                                    "paused until its face re-confirms", oid, source_id, other, st["name"])
                        st["name"] = None
                        st["score"] = 0.0
                        st["unconf_since"] = None
                        st["unconf_count"] = 0
                        src.pop(oid, None)

    def start(self):
        """Start the background recognition thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self._thread.start()
        logger.info("Face recognition thread started")
    
    def stop(self):
        """Stop the background recognition thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Face recognition thread stopped")
    
    # Rate-limit the _embed_crop failure log so a persistent get_feat problem
    # (GPU issue, degenerate crop) is visible in the logs instead of silently
    # degrading every face to "unknown" with only debug-level noise.
    _embed_fail_t0 = 0.0
    _embed_fail_lock = threading.Lock()

    def _log_embed_failure(self, exc: Exception):
        now = time.monotonic()
        with self._embed_fail_lock:
            if now - self._embed_fail_t0 >= 30.0:
                self._embed_fail_t0 = now
                logger.warning("recognition get_feat failing (faces will report "
                               "unknown while this persists): %s", exc)

    @telemetry.timeit("recognizer.embed")
    def _embed_crop(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """(512,) L2-normalized embedding of a face crop via the ArcFace model
        directly, without InsightFace's bundled detector.

        The crop passed in was already located by sgie_face — running `app.get()`
        on it would re-run SCRFD (640x640 upscale + a second aligned recrop) purely
        to obtain the same embedding: redundant GPU inference. `get_feat` does the
        resize + mean/std normalization internally; the crop is center-padded to a
        square first so ArcFace's resize is a uniform scale instead of a stretch
        (ArcFace is trained on square, aligned faces).

        Returns None on any failure so callers can fall back / skip.
        """
        try:
            h, w = crop_bgr.shape[:2]
            if h <= 0 or w <= 0:
                return None
            if h != w:
                side = max(h, w)
                top = (side - h) // 2
                bottom = side - h - top
                left = (side - w) // 2
                right = side - w - left
                crop_bgr = cv2.copyMakeBorder(
                    crop_bgr, top, bottom, left, right, cv2.BORDER_REPLICATE)
            feats = self.app.models["recognition"].get_feat(crop_bgr)
            if feats is None or len(feats) == 0:
                return None
            emb = np.asarray(feats[0], dtype=np.float32)
            norm = np.linalg.norm(emb)
            return emb / (norm + 1e-6) if norm > 0 else emb
        except Exception as exc:
            self._log_embed_failure(exc)
            return None

    def _embed_aligned_face(self, frame: np.ndarray, face: Dict[str, Any],
                            parent_id: Any, raw_crop: np.ndarray) -> Optional[np.ndarray]:
        """Straighten (align) a face with its 5 landmarks exactly the way
        registration does -- insightface's norm_crop to an upright 112x112,
        which is what app.get() does internally for the registry photos --
        then embed it. Returns None (the caller falls back to the unaligned
        crop, _embed_crop) when the landmarks are missing or implausible.

        Why (2026-09-12): live faces were embedded UNALIGNED and compared
        against ALIGNED registry embeddings. Measured on emp2's own elevator
        crops against his photos: 0.35-0.42 aligned vs 0.075-0.22 unaligned."""
        lmk = face.get("landmarks")
        if not lmk or len(lmk) != 5:
            self._count_align("no_landmarks")
            return None
        pts = np.asarray(lmk, dtype=np.float32)
        left, top = float(face["left"]), float(face["top"])
        w, h = float(face["width"]), float(face["height"])
        # Plausibility: every point inside the face box grown by half its size
        # on each side (a mis-mapped point lands far away), eyes above mouth.
        mx, my = 0.5 * w, 0.5 * h
        inside = ((pts[:, 0] >= left - mx) & (pts[:, 0] <= left + w + mx)
                  & (pts[:, 1] >= top - my) & (pts[:, 1] <= top + h + my))
        if not inside.all() or max(pts[0, 1], pts[1, 1]) >= min(pts[3, 1], pts[4, 1]):
            self._count_align("implausible")
            return None
        try:
            aligned = face_align.norm_crop(frame, landmark=pts, image_size=112)
            feats = self.app.models["recognition"].get_feat(aligned)
            if feats is None or len(feats) == 0:
                self._count_align("failed")
                return None
            emb = np.asarray(feats[0], dtype=np.float32)
            norm = float(np.linalg.norm(emb))
            if norm <= 0:
                self._count_align("failed")
                return None
        except Exception as exc:
            logger.debug("Face alignment failed for parent_id=%s: %s", parent_id, exc)
            self._count_align("failed")
            return None
        self._count_align("aligned")
        if self._align_debug_left > 0:
            self._save_align_debug(raw_crop, aligned, parent_id)
        return emb / norm

    def _count_align(self, outcome: str) -> None:
        """Tally alignment outcomes; one INFO line a minute shows how many faces
        were straightened vs fell back to the unaligned crop."""
        self._align_stats[outcome] += 1
        now = time.time()
        if now - self._align_stats_logged_at >= 60.0:
            s = self._align_stats
            logger.info("Face alignment (last %.0fs): straightened=%d, fell back: no_landmarks=%d "
                        "implausible=%d failed=%d", now - self._align_stats_logged_at,
                        s["aligned"], s["no_landmarks"], s["implausible"], s["failed"])
            for k in s:
                s[k] = 0
            self._align_stats_logged_at = now

    def _save_align_debug(self, raw_crop: np.ndarray, aligned: np.ndarray, parent_id: Any) -> None:
        """Save a raw crop + its straightened version (both enlarged to 224 px
        with nearest-neighbour, so no detail is invented) for a visual check."""
        try:
            os.makedirs(self._align_debug_dir, exist_ok=True)
            big = lambda img: cv2.resize(img, None, fx=224.0 / max(img.shape[:2]),
                                         fy=224.0 / max(img.shape[:2]),
                                         interpolation=cv2.INTER_NEAREST)
            stem = os.path.join(self._align_debug_dir, f"{self._align_debug_left:03d}_track{parent_id}")
            cv2.imwrite(stem + "_raw.jpg", big(raw_crop))
            cv2.imwrite(stem + "_aligned.jpg", big(aligned))
            self._align_debug_left -= 1
            if self._align_debug_left == 0:
                logger.info("Face alignment debug: samples saved to %s", self._align_debug_dir)
        except Exception as exc:
            logger.debug("Face alignment debug save failed: %s", exc)
            self._align_debug_left = 0

    def enqueue(self, frame: np.ndarray, faces: List[Dict[str, Any]], source_id: str, pts: float):
        """Enqueue face crops for recognition.
        
        Args:
            frame: Full frame in BGR format (capture resolution)
            faces: List of face dicts with keys: left, top, width, height, parent_id, confidence
            source_id: Camera source ID
            pts: Presentation timestamp
        """
        if not self._running:
            return
        try:
            self._queue.put_nowait((frame, faces, source_id, pts))
        except queue.Full:
            logger.warning("Face recognition queue full, dropping frame")
    
    def _init_capture(self, capture_store: 'FaceCaptureStore',
                      min_pixels: int, min_det_confidence: float,
                      novelty_threshold: float):
        """Enable CCTV face capture with the given config."""
        self._capture_store = capture_store
        self._capture_enabled = True
        self._capture_min_pixels = min_pixels
        self._capture_min_det_confidence = min_det_confidence
        self._capture_novelty_threshold = novelty_threshold
        logger.info("Face capture enabled: min_pixels=%d det_conf=%.2f novelty=%.2f",
                    min_pixels, min_det_confidence, novelty_threshold)

    def _recognition_loop(self):
        """Background thread: process face crops and match against gallery."""
        while self._running:
            # Periodic gallery reload (picks up new registrations from the API)
            now = time.time()
            if now - self._last_gallery_reload > self._gallery_reload_interval:
                self.gallery.load_all()
                self._last_gallery_reload = now
            
            # Periodic capture-embedding reload for cross-track novelty check
            if (self._capture_enabled and
                now - self._last_capture_emb_load > self._capture_emb_reload_interval):
                self._capture_embeddings = self._capture_store.get_all_embeddings() \
                    if self._capture_store else {}
                self._last_capture_emb_load = now

            try:
                frame, faces, source_id, pts = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if not faces:
                continue

            # Process one queued frame (all faces from one meta sample).  Wrapped
            # in try/except so a single bad frame (e.g. a malformed capture
            # embedding in Redis) can NEVER kill the recognition thread — the
            # loop must survive to keep draining the queue and publishing results.
            try:
                self._process_frame(frame, faces, source_id, pts)
            except Exception as exc:
                logger.error("Face recognition frame processing failed (continuing): %s", exc)

    @telemetry.timeit("recognizer")
    def _process_frame(self, frame: np.ndarray, faces: List[Dict[str, Any]],
                       source_id: str, pts: float):
        """Match one frame's faces against the gallery, stabilize names, publish
        results, and optionally capture unknown faces.

        Exceptions raised here are caught by _recognition_loop so the recognition
        thread can never die silently (previously an unhandled error killed
        recognition for the rest of the session).
        """
        # Get current gallery embeddings
        face_ids, person_names, gallery_embs = self.gallery.get_embeddings()
        if len(face_ids) == 0:
            # No gallery, publish unknown for all
            self._publish_results(source_id, pts, [
                {"parent_id": f["parent_id"], "name": "unknown", "confidence": 0.0, "rec_id": None}
                for f in faces
            ])
            return

        # Process each face crop and compute candidate scores
        raw_candidates = []
        face_crops: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}  # face_idx -> (crop_bgr, embedding)
        for i, face in enumerate(faces):
            left, top, w, h = face["left"], face["top"], face["width"], face["height"]
            parent_id = face["parent_id"]

            h_frame, w_frame = frame.shape[:2]
            px = int(w * 0.3)
            py = int(h * 0.3)
            orig_left, orig_top = left, top
            crop_left = max(0, orig_left - px)
            crop_top = max(0, orig_top - py)
            crop_right = min(orig_left + w + px, w_frame)
            crop_bottom = min(orig_top + h + py, h_frame)

            if crop_right <= crop_left or crop_bottom <= crop_top:
                continue

            raw_crop = frame[crop_top:crop_bottom, crop_left:crop_right]

            # ── Face enhancement (pre-recognition) ─────────────────────────
            # Enhance the crop BEFORE computing its embedding so the saved
            # capture embedding + thumbnail both benefit from the improved
            # image (Lanczos upscale + CLAHE + unsharp for the lightweight
            # preprocess backend). Falls back to the raw crop if the
            # enhanced crop fails face detection (or on any error).
            # Straighten first (2026-09-12) -- see _embed_aligned_face. When it
            # works, the raw crop only serves the capture thumbnail, and no
            # enhancement is applied, matching how registry photos were embedded.
            aligned_emb = (self._embed_aligned_face(frame, face, parent_id, raw_crop)
                           if self._align_enabled else None)
            face_crop = raw_crop
            if aligned_emb is None and self._enhancer is not None:
                try:
                    face_crop = self._enhancer.enhance(raw_crop)
                except Exception as exc:
                    logger.debug("Face enhancement failed for parent_id=%s: %s", parent_id, exc)

            try:
                # Recognition-only: embed the SGIE crop straight through ArcFace
                # (no bundled re-detection — see _embed_crop) -- unless it was
                # already straightened above.
                embedding_norm = aligned_emb if aligned_emb is not None else self._embed_crop(face_crop)
                if embedding_norm is None and face_crop is not raw_crop:
                    logger.debug("Enhanced crop embedding failed for parent_id=%s; retrying raw", parent_id)
                    face_crop = raw_crop
                    embedding_norm = self._embed_crop(face_crop)
                if embedding_norm is None:
                    logger.debug("Embedding failed for parent_id=%s; skipping", parent_id)
                    continue

                gallery_norm = gallery_embs / (np.linalg.norm(gallery_embs, axis=1, keepdims=True) + 1e-6)

                similarities = np.dot(gallery_norm, embedding_norm)
                best_idx = int(np.argmax(similarities))
                best_score = float(similarities[best_idx])

                # Open-set margin: how far the best identity is ahead of the
                # best OTHER identity. Entries sharing the winner's face_id are
                # the same person (append_embedding allows several embeddings
                # per identity), so they don't count as a competitor.
                rival_mask = np.array([fid != face_ids[best_idx] for fid in face_ids])
                rival_score = float(np.max(similarities[rival_mask])) if rival_mask.any() else -1.0

                raw_candidates.append({
                    "face_idx": i,
                    "parent_id": parent_id,
                    "gallery_idx": best_idx,
                    "rec_id": face_ids[best_idx],
                    "name": person_names[best_idx],
                    "score": best_score,
                    "margin": best_score - rival_score,
                    "embedding": embedding_norm,  # keep for capture logic
                })
                face_crops[i] = (face_crop, embedding_norm)
            except Exception as e:
                logger.error("Face recognition error for parent_id=%s: %s", parent_id, e)

        # Sort candidate matches by similarity score descending
        raw_candidates.sort(key=lambda x: x["score"], reverse=True)

        # Global label uniqueness, raw side (2026-09-01): a name locked on a
        # DIFFERENT live track (any source) with a higher locked score is
        # unavailable to this frame's raw matches -- an unnamed face can only
        # take an untaken name. Without this the same person could be named
        # twice in one frame: once by a lock, once by a raw match.
        now_ts = time.time()
        locked_elsewhere: Dict[str, Tuple[Any, float]] = {}
        with self._names_lock:
            for s_id, s_tracks in self._track_state.items():
                for t_pid, t_st in s_tracks.items():
                    if self._lock_live(t_st, now_ts):
                        prev = locked_elsewhere.get(t_st["name"])
                        if prev is None or t_st["score"] > prev[1]:
                            locked_elsewhere[t_st["name"]] = ((s_id, t_pid), t_st["score"])

        assigned_face_indices = set()
        assigned_gallery_indices = set()
        assigned_parent_ids = set()
        final_matches = {}  # face_idx -> dict

        for cand in raw_candidates:
            f_idx = cand["face_idx"]
            g_idx = cand["gallery_idx"]
            p_id = cand["parent_id"]
            score = cand["score"]

            held = locked_elsewhere.get(cand["name"])
            if held is not None and held[0] != (source_id, p_id):
                if held[1] >= score:
                    logger.info("Face rec TAKEN: parent_id=%s candidate_name='%s' score=%.3f -- name locked on "
                                "track %s/%s (score %.3f), this face stays unnamed",
                                p_id, cand["name"], score, held[0][0], held[0][1], held[1])
                    continue
                # This raw match outbids the lock: release the weaker holder NOW
                # so the frame never carries the same name twice (the vote lock
                # on this track catches up over the next votes).
                if score >= self.rec_thresh:
                    with self._names_lock:
                        h_st = self._track_state.get(held[0][0], {}).get(held[0][1])
                        if h_st is not None and h_st.get("name") == cand["name"]:
                            h_st["released_name"] = cand["name"]
                            h_st["released_at"] = now_ts
                            h_st["name"] = None
                            h_st["score"] = 0.0
                            h_st["votes"].clear()
                            self._names.get(held[0][0], {}).pop(held[0][1], None)
                    locked_elsewhere.pop(cand["name"], None)
                    logger.info("Face rec OUTBID: parent_id=%s takes '%s' (score %.3f) from track %s/%s (lock %.3f)",
                                p_id, cand["name"], score, held[0][0], held[0][1], held[1])
            if score >= self.rec_thresh and cand["margin"] < self.rec_margin:
                # Ambiguous: the runner-up identity is too close behind. Leave
                # the face "unknown" this frame rather than risk locking the
                # wrong name onto the track (prompt.md 2026-08-31, point 4:
                # a misidentified person STAYS misidentified because whoever
                # scores highest wins outright with no ambiguity check).
                logger.info("Face rec AMBIGUOUS: parent_id=%s best='%s' score=%.3f margin=%.3f (<%.2f)",
                            p_id, cand["name"], score, cand["margin"], self.rec_margin)
                continue
            if score >= self.rec_thresh:
                # Enforce 1-to-1 identity uniqueness per frame:
                # - A gallery identity (g_idx) can only be assigned ONCE per frame
                # - A person track ID (p_id) can only be assigned ONCE per frame
                if (f_idx not in assigned_face_indices and
                    g_idx not in assigned_gallery_indices and
                    (p_id is None or p_id not in assigned_parent_ids)):

                    assigned_face_indices.add(f_idx)
                    assigned_gallery_indices.add(g_idx)
                    if p_id is not None:
                        assigned_parent_ids.add(p_id)

                    final_matches[f_idx] = {
                        "parent_id": p_id,
                        "name": cand["name"],
                        "confidence": round(score, 3),
                        "rec_id": cand["rec_id"]
                    }
                    logger.info("Face rec MATCH: parent_id=%s name='%s' score=%.3f (thresh=%.2f)",
                                p_id, cand["name"], score, self.rec_thresh)
                else:
                    logger.info("Face rec SUPPRESSED duplicate match: parent_id=%s candidate_name='%s' score=%.3f (already assigned in frame)",
                                p_id, cand["name"], score)

        # Construct final per-face results list preserving order of input faces
        results = []
        for i, face in enumerate(faces):
            pid = face["parent_id"]
            if i in final_matches:
                results.append(final_matches[i])
            else:
                score = 0.0
                for cand in raw_candidates:
                    if cand["face_idx"] == i:
                        score = round(cand["score"], 3)
                        break
                results.append({"parent_id": pid, "name": "unknown", "confidence": score, "rec_id": None})

        # ── Per-track name stability (anti-flicker) ────────────────────────
        # Smooth each track's label over a short vote window so single-frame
        # argmax wobbles (e.g. Mahmoud <-> Youssef) can't flip the label on
        # one frame. Also refreshes the overlay cache (self._names) used by
        # the sidecar probe to draw names on the live JPEG.
        self._stabilize_names(source_id, results, faces, frame.shape)

        # Publish results
        self._publish_results(source_id, pts, results)

        # ── CCTV face capture (only for unknown/below-threshold faces) ──────
        if self._capture_enabled and self._capture_store is not None and face_crops:
            self._capture_faces(source_id, pts, faces, results, face_crops, raw_candidates)
    
    @telemetry.timeit("recognizer.stabilize")
    def _stabilize_names(self, source_id: str, results: List[Dict[str, Any]],
                         faces: List[Dict[str, Any]], frame_shape: Tuple[int, int]):
        """Temporal smoothing of per-track recognition labels (anti-flicker)
        plus a spatial guard against recycled track IDs (anti-teleport).

        Recognition runs at the meta cadence (~2 fps).  For a clean single frame
        the top gallery match can alternate between two identities that both
        clear ``rec_thresh`` (e.g. Mahmoud Tayee <-> Youssef Noser), which made
        the live overlay label flip every frame.  Here each (source, parent_id)
        track keeps a short vote history; the label shown is the majority vote
        over that window, and a competing identity must dominate the window
        (>= ``track_majority`` share, >= ``track_min_votes`` votes) before it can
        displace the current one.  Labels age out ``track_lock_seconds`` after the
        track's last sighting.

        DeepStream reuses object IDs once a track is lost (a few frames later it
        can hand the same ``parent_id`` to a *different* person elsewhere in the
        frame).  Without a spatial check the stale lock teleports onto that new
        person.  So each track also remembers its last sighted face-center; if a
        locked track reappears farther than ``track_max_jump`` (fraction of the
        frame diagonal) from where its label was last seen, the ID was recycled
        — the lock is released and the new person builds their own votes.

        Mutates ``results`` in place and refreshes the ``self._names`` overlay
        cache under its lock.  ``faces`` is the same list the results were built
        from (1:1 aligned by index); bboxes are in capture resolution.
        """
        now = time.time()
        fh, fw = frame_shape[:2] if frame_shape else (1, 1)
        diag = math.hypot(fw, fh) or 1.0
        with self._names_lock:
            src = self._names.setdefault(source_id, {})
            tracks = self._track_state.setdefault(source_id, {})
            for res, face in zip(results, faces):
                pid = res["parent_id"]
                # This frame's own match score, before the vote lock replaces it
                # with the window's best (below). products/barber secures a
                # barber on repeated strong matches, and published entries are
                # one per frame but share the locked score -- without this it
                # counts one lucky frame as several (2026-09-20).
                res["frame_confidence"] = res["confidence"]
                if pid is None:
                    continue
                # Face-center in capture pixels (bboxes are capture-res, same as frame)
                fcx = face.get("left", 0) + face.get("width", 0) / 2.0
                fcy = face.get("top", 0) + face.get("height", 0) / 2.0
                st = tracks.setdefault(
                    pid, {"name": None, "score": 0.0, "votes": deque(),
                          "last_seen": 0.0, "pos": None,
                          "released_name": None, "released_at": 0.0})
                # ── Body lost past the grace (issue 5, 2026-09-12) ─────────
                # A name lives as long as the tracker follows the body (see
                # note_alive_tracks). A track whose body was gone longer than
                # track_lock_seconds is now whoever this is -- it must earn a
                # name again. Checked BEFORE alive_at is refreshed below (a
                # face arriving means the body is present now).
                if st["name"] and now - self._last_alive(st) > self._track_lock_seconds:
                    st["name"] = None
                    st["score"] = 0.0
                    st["votes"].clear()
                    st["pos"] = None
                    src.pop(pid, None)
                # Did the tracker lose this body at any point since its last
                # face? note_alive_tracks flags gaps; a long gap right now
                # counts too (e.g. no tracker reports arriving at all).
                tracking_gap = (st.get("continuity_broken", False)
                                or now - st.get("alive_at", 0.0) > self.ALIVE_CONTINUITY_SECONDS)
                st["alive_at"] = now
                st["continuity_broken"] = False
                # ── Spatial guard (recycled track ID) ──────────────────────
                # Only across a real tracking gap (issue 5): a body the tracker
                # followed continuously since its last face is the same person
                # even if they walked far with their face away -- DeepStream
                # only recycles an id after losing the track.
                prev_pos = st.get("pos")
                if st["name"] and prev_pos is not None and tracking_gap:
                    jump = math.hypot(fcx - prev_pos[0], fcy - prev_pos[1]) / diag
                    if jump > self._track_max_jump:
                        logger.info(
                            "Face track %s teleported %.2fx diag (>%.2f) while locked as '%s' — "
                            "recycled track ID, releasing label",
                            pid, jump, self._track_max_jump, st["name"])
                        # Remember which label we just ejected so it can't be
                        # re-adopted on the very next frame (the raw match for
                        # the *new* person may still clear rec_thresh against
                        # that same gallery entry — that's what caused this bug).
                        st["released_name"] = st["name"]
                        st["released_at"] = now
                        st["name"] = None
                        st["score"] = 0.0
                        st["votes"].clear()
                st["pos"] = (fcx, fcy)

                cand = res["name"]
                if cand != "unknown" and res.get("confidence", 0.0) >= self.rec_thresh:
                    # Votes carry (name, ts, score) so the adopted name's own score
                    # is reported (not the last-sighted candidate's score).
                    # Issue 7A: a face processed just after a crossing may
                    # have been captured DURING it -- it gets no vote.
                    if now - st.get("collision_at", 0.0) >= self.COLLISION_VOTE_HOLDOFF_SECONDS:
                        st["votes"].append((cand, now, res.get("confidence", 0.0)))
                    st["last_seen"] = now
                # Drop votes older than the window
                while st["votes"] and now - st["votes"][0][1] > self._track_vote_ttl:
                    st["votes"].popleft()
                # Majority vote over the window
                if len(st["votes"]) >= self._track_min_votes:
                    counts: Dict[str, int] = {}
                    scores: Dict[str, float] = {}
                    for n, _ts, sc in st["votes"]:
                        counts[n] = counts.get(n, 0) + 1
                        scores[n] = max(scores.get(n, 0.0), sc)
                    top_name = max(counts, key=counts.get)
                    # Don't immediately re-adopt the label we just ejected for a
                    # teleport: the new person's raw match can still point at the
                    # same gallery entry above threshold.  Give the track a fresh
                    # window (vote_ttl) to build its own identity first.
                    if (top_name == st.get("released_name")
                            and now - st.get("released_at", 0.0) < self._track_vote_ttl):
                        top_name = None
                    # A cooldown that has passed, or a *different* label winning,
                    # means the ejected name no longer has any claim on this track.
                    if (st.get("released_name")
                            and now - st.get("released_at", 0.0) >= self._track_vote_ttl):
                        st["released_name"] = None
                    if top_name and counts[top_name] / len(st["votes"]) >= self._track_majority:
                        if top_name != st.get("released_name"):
                            st["released_name"] = None
                        if st["name"] != top_name:
                            st["unconf_since"] = None  # new name: fresh re-check window
                            st["unconf_count"] = 0
                        st["name"] = top_name
                        st["score"] = scores[top_name]
                # (Issue 5: the old "release lock_seconds after the last face
                # match" rule is gone -- a name is released only when the
                # tracker loses the body, at the top of this loop and in
                # note_alive_tracks.)
                # ── Re-confirmation (issue 7C, 2026-09-12) ─────────────────
                # While a name is held and its face is VISIBLE, the face must
                # keep agreeing: after track_reconfirm_seconds of visible-but-
                # not-matching face (at least RECONFIRM_MIN_FACES sightings),
                # the name is dropped and must be re-earned -- catches an id
                # swap that slipped past the crossing guard. Stretches with the
                # face turned away don't count: a face gap longer than
                # RECONFIRM_FACE_GAP_SECONDS starts a new window.
                if st["name"]:
                    if cand == st["name"] and res.get("confidence", 0.0) >= self.rec_thresh:
                        st["unconf_since"] = None
                        st["unconf_count"] = 0
                    else:
                        if (st.get("unconf_since") is None
                                or now - st.get("last_face_at", 0.0) > self.RECONFIRM_FACE_GAP_SECONDS):
                            st["unconf_since"] = now
                            st["unconf_count"] = 0
                        st["unconf_count"] = st.get("unconf_count", 0) + 1
                        if (now - st["unconf_since"] >= self._track_reconfirm_seconds
                                and st["unconf_count"] >= self.RECONFIRM_MIN_FACES):
                            logger.info("Face track %s on %s: name '%s' not re-confirmed by its visible face "
                                        "for %.1fs (%d sightings) -- dropped until re-earned",
                                        pid, source_id, st["name"], now - st["unconf_since"],
                                        st["unconf_count"])
                            st["name"] = None
                            st["score"] = 0.0
                            st["votes"].clear()
                            st["unconf_since"] = None
                            st["unconf_count"] = 0
                st["last_face_at"] = now
                if st["name"]:
                    if res["name"] != st["name"]:
                        res["rec_id"] = None  # stabilized label differs from raw frame match
                    res["name"] = st["name"]
                    res["confidence"] = round(st["score"], 3)
                    src[pid] = (st["name"], res["confidence"], now)
                else:
                    src.pop(pid, None)
                # Issue 4 (2026-09-12): the attendance table logs only names
                # this vote lock confirmed -- exactly what the live label shows.
                res["confirmed"] = bool(st["name"])
            # ── Global label uniqueness (2026-09-01) ──────────────────────
            # One name may be locked on ONE track across every source: if two
            # live tracks hold the same label, the higher locked score keeps
            # it and the other is released (and can't re-adopt it for a vote
            # window). Mirrors products/attendance's identity-uniqueness rule
            # so the live overlay never shows the same person twice.
            holders: Dict[str, list] = {}
            for s_id, s_tracks in self._track_state.items():
                for t_pid, t_st in s_tracks.items():
                    # Issue 5: a name held by a body the tracker still follows
                    # (face turned away) still counts as held.
                    if self._lock_live(t_st, now):
                        holders.setdefault(t_st["name"], []).append((s_id, t_pid, t_st))
            for name, held in holders.items():
                if len(held) < 2:
                    continue
                held.sort(key=lambda h: (h[2]["score"], h[2]["last_seen"]), reverse=True)
                for s_id, t_pid, t_st in held[1:]:
                    logger.info("label '%s' held by %d tracks -- keeping %s/%s (score %.2f), releasing %s/%s (score %.2f)",
                                name, len(held), held[0][0], held[0][1], held[0][2]["score"], s_id, t_pid, t_st["score"])
                    t_st["released_name"] = name
                    t_st["released_at"] = now
                    t_st["name"] = None
                    t_st["score"] = 0.0
                    t_st["votes"].clear()
                    self._names.get(s_id, {}).pop(t_pid, None)
                    if s_id == source_id:
                        for res in results:
                            if res.get("parent_id") == t_pid and res.get("name") == name:
                                res["name"] = "unknown"
                                res["rec_id"] = None
                                res["confirmed"] = False
            # Bounded memory: drop tracks whose body hasn't been seen (face or
            # tracker) for well beyond the window/lock -- issue 5: NOT the last
            # face match, or a tracked-but-faceless body would lose its name.
            stale_before = now - max(self._track_vote_ttl, self._track_lock_seconds) - 10.0
            dead = [pid for pid, st in tracks.items() if self._last_alive(st) < stale_before]
            for pid in dead:
                del tracks[pid]

    @telemetry.timeit("recognizer.publish")
    def _publish_results(self, source_id: str, pts: float, results: List[Dict[str, Any]]):
        """Publish recognition results to Redis."""
        try:
            key = f"{FACE_REC_KEY_PREFIX}{source_id}"
            fields = {
                "pts": f"{pts:.3f}",
                "face_rec": json.dumps(results)
            }
            self.r.xadd(key, fields, maxlen=64, approximate=True)
        except Exception as e:
            logger.error("Failed to publish face recognition results: %s", e)

    def _capture_faces(self, source_id: str, pts: float,
                       input_faces: List[Dict[str, Any]],
                       results: List[Dict[str, Any]],
                       face_crops: Dict[int, Tuple[np.ndarray, np.ndarray]],
                       raw_candidates: List[Dict[str, Any]]):
        """Evaluate unknown faces from the recognition round and save any that
        meet the capture quality thresholds as new gallery candidates.

        Deduplication strategy:
        1. Per-track dedup: only one capture per (source_id, parent_id) per session.
        2. Cross-track novelty: skip if the embedding is too similar to an
           existing capture (cosine distance < novelty_threshold).
        """
        import uuid

        # Build a quick lookup of candidate by face_idx for embeddings
        cand_by_idx: Dict[int, Dict[str, Any]] = {}
        for cand in raw_candidates:
            cand_by_idx[cand["face_idx"]] = cand

        for i, (face, result) in enumerate(zip(input_faces, results)):
            # Only capture unknown / below-threshold faces
            if result["name"] != "unknown":
                continue

            pid = face["parent_id"]

            # 1 ── Quality gates ────────────────────────────────────────────
            fw, fh = face.get("width", 0), face.get("height", 0)
            det_conf = face.get("confidence", 0)
            if min(fw, fh) < self._capture_min_pixels:
                continue
            if det_conf < self._capture_min_det_confidence:
                continue

            # 2 ── Per-track dedup ──────────────────────────────────────────
            with self._captured_tracks_lock:
                already = self._captured_tracks.setdefault(source_id, set())
                if pid in already:
                    continue
                already.add(pid)

            # 3 ── Cross-track novelty check ───────────────────────────────
            cand = cand_by_idx.get(i)
            if cand is None or "embedding" not in cand:
                continue
            emb = cand["embedding"]  # already normalized

            novel = True
            if self._capture_embeddings:
                existing = np.array(list(self._capture_embeddings.values()), dtype=np.float32)
                # Guard against malformed/empty embeddings (e.g. manual photo
                # uploads stored with embedding=[]): a (N, 0) or object-dtype
                # matrix would raise inside the dot product below.
                if (len(existing) > 0 and existing.ndim == 2
                        and existing.shape[1] == emb.shape[0]):
                    existing_norm = existing / (np.linalg.norm(existing, axis=1, keepdims=True) + 1e-6)
                    sims = np.dot(existing_norm, emb)
                    max_sim = float(np.max(sims))
                    if max_sim > (1.0 - self._capture_novelty_threshold):
                        logger.debug("Capture SKIP (too similar to existing, max_sim=%.3f, track=%s)",
                                     max_sim, pid)
                        novel = False

            if not novel:
                continue

            # 4 ── Save capture ─────────────────────────────────────────────
            crop_bgr = face_crops.get(i, (None, None))[0]
            if crop_bgr is None:
                continue

            capture_id = str(uuid.uuid4())[:8]
            ok, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue

            self._capture_store.add(
                capture_id=capture_id,
                embedding=emb,
                source_id=source_id,
                track_id=pid,
                pts=pts,
                image_bytes=buf.tobytes(),
            )
            logger.info("Capture saved: %s (source=%s track=%s det_conf=%.2f face_size=%dx%d)",
                        capture_id, source_id, pid, det_conf, fw, fh)


def create_face_recognizer(redis_url: str, config: Dict[str, Any]) -> Optional[FaceRecognizer]:
    """Factory function to create FaceRecognizer from config.
    
    Returns None if face recognition is disabled in config.
    """
    if not config.get("face_recognition_enabled", False):
        logger.info("Face recognition disabled in config")
        return None
    
    try:
        # Handle both Redis URL string and Redis client object
        if isinstance(redis_url, str):
            r = redis.Redis.from_url(redis_url)
        else:
            r = redis_url  # Already a Redis client
        recognizer = FaceRecognizer(
            redis_client=r,
            model_name=config.get("face_recognition_model", "auraface"),
            det_size=tuple(config.get("face_recognition_det_size", [640, 640])),
            det_thresh=config.get("face_recognition_det_thresh", 0.5),
            rec_thresh=config.get("face_recognition_threshold", 0.45),
            rec_margin=float(config.get("face_recognition_margin", 0.0)),
            ctx_id=config.get("face_recognition_ctx_id", 0),
            gallery_key=str(config.get("face_gallery_key", FACE_GALLERY_KEY)),
            track_vote_ttl=float(config.get("face_recognition_vote_ttl", 4.0)),
            track_min_votes=int(config.get("face_recognition_min_votes", 1)),
            track_majority=float(config.get("face_recognition_majority", 0.6)),
            track_lock_seconds=float(config.get("face_recognition_lock_seconds", 3.0)),
            track_max_jump=float(config.get("face_recognition_max_jump", 0.35)),
        )
        # ── CCTV face capture initialisation ──────────────────────────────
        if config.get("face_capture_enabled", False):
            captures_dir = str(config.get("face_captures_dir", "/workspace/iris/data/face_captures"))
            capture_store = FaceCaptureStore(r, captures_dir)
            recognizer._init_capture(
                capture_store=capture_store,
                min_pixels=int(config.get("face_capture_min_pixels", 40)),
                min_det_confidence=float(config.get("face_capture_min_det_confidence", 0.5)),
                novelty_threshold=float(config.get("face_capture_novelty_threshold", 0.25)),
            )
        return recognizer
    except Exception as e:
        logger.error("Failed to create FaceRecognizer: %s", e)
        return None