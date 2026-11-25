"""Multi-camera vehicle tracking service.

Reads the global_ids that the DeepStream sidecar's built-in MTMC module already
computes and publishes in iris:meta:<cam-id> Redis streams. Instead of
duplicating cross-camera matching, this service aggregates those global_ids
into a unified view of vehicle journeys across cameras.

The sidecar publishes per-frame entries with:
  - dets: [[label, conf, object_id, left, top, width, height], ...]
  - global_ids: {object_id: global_id}  (MTMC global IDs)
  - pts: presentation timestamp in seconds

This service:
  1. Reads those entries from all cameras
  2. Tracks which global_ids appear on which cameras, and when
  3. Computes transitions (camera-to-camera handoffs) and travel times
  4. Serves the aggregated data to the frontend tracker page via REST + WebSocket
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from app.line_counter import LineCounter
from app.restricted_zones import RestrictedZoneMonitor
from app.truck_feed import TruckFeedBridge
from app.zones import mux_resolution
from app.zones import mux_resolution

logger = logging.getLogger("iris.tracking")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class VehicleObservation:
    """One frame's observation of a vehicle on a camera."""
    cam_id: str
    global_id: int
    object_id: int
    label: str
    confidence: float
    pts: float
    box: tuple[float, float, float, float]  # (left, top, width, height)


@dataclass
class GlobalTrack:
    """A vehicle tracked across one or more cameras."""
    global_id: int
    cameras_seen: list[str] = field(default_factory=list)
    entry_times: dict[str, float] = field(default_factory=dict)   # cam_id -> first pts
    exit_times: dict[str, float] = field(default_factory=dict)    # cam_id -> last pts
    observation_counts: dict[str, int] = field(default_factory=dict)  # cam_id -> count
    label: str = ""
    transitions: list[dict[str, Any]] = field(default_factory=list)
    is_active: bool = True
    first_seen_pts: float = 0.0
    last_seen_pts: float = 0.0
    # Left the premises: crossed the gate line outwards and was matched to this vehicle.
    exited: bool = False
    exited_wall: float = 0.0

    def total_cameras(self) -> int:
        return len(self.cameras_seen)

    def total_duration_s(self) -> float:
        if not self.entry_times:
            return 0.0
        earliest = min(self.entry_times.values())
        latest = max(self.exit_times.values()) if self.exit_times else self.last_seen_pts
        return latest - earliest


@dataclass
class Transition:
    """A cross-camera handoff event."""
    global_id: int
    from_cam: str
    to_cam: str
    travel_time_s: float
    timestamp_s: float
    label: str = ""


# ---------------------------------------------------------------------------
# Camera data aggregator
# ---------------------------------------------------------------------------
class CameraAggregator:
    """Maintains per-camera vehicle state from sidecar Redis entries."""

    # Vehicles not seen for this many seconds are considered "gone" from a camera
    STALE_THRESHOLD_S = 5.0
    # A vehicle out of sight of a camera for longer than this and then seen again starts a new
    # EPISODE there (a new visit for the platform); the old one is kept in v["episodes"].
    EPISODE_GAP_S = 90.0

    def __init__(self, cameras: list[str]):
        self.cameras = cameras
        # Per-camera: {global_id: {first_pts, last_pts, count, label, object_ids, wall_last}}
        self._camera_vehicles: dict[str, dict[int, dict]] = {
            c: {} for c in cameras
        }
        # Per-camera: last seen pts per global_id (for detecting camera exits)
        self._last_seen: dict[str, dict[int, float]] = {c: {} for c in cameras}
        self._pts_monotonic: dict[str, list[float]] = {c: [] for c in cameras}
        # Per camera: latest (pts, wall clock) pair seen, to place a pts on the wall clock.
        self._ref: dict[str, tuple[float, float]] = {}
        self._started = time.time()

    def process_entry(self, cam_id: str, dets: list, global_ids: dict, pts: float,
                      wall: float | None = None) -> None:
        """Process one Redis stream entry for a camera. `wall` is when the entry was
        published (its stream id); replayed history then keeps its real time instead of
        being stamped with the moment of the replay."""
        if cam_id not in self._camera_vehicles:
            return

        vehicles = self._camera_vehicles[cam_id]
        last = self._last_seen[cam_id]

        now = time.time() if wall is None else wall
        self._ref[cam_id] = (pts, now)
        # Record pts for monotonicity tracking
        self._pts_monotonic[cam_id].append(pts)
        if len(self._pts_monotonic[cam_id]) > 1000:
            self._pts_monotonic[cam_id] = self._pts_monotonic[cam_id][-500:]

        seen_gids: set[int] = set()

        for det in dets:
            if len(det) < 7:
                continue
            label, conf, obj_id = det[0], det[1], int(det[2])
            left, top, width, height = det[3], det[4], det[5], det[6]

            # Look up global_id from the sidecar's MTMC output
            gid = global_ids.get(str(obj_id))
            if gid is None:
                gid = global_ids.get(obj_id)
            if gid is None:
                continue

            gid = int(gid)
            seen_gids.add(gid)

            if gid not in vehicles:
                vehicles[gid] = {
                    "first_pts": pts,
                    "last_pts": pts,
                    "wall_first": now,
                    "wall_last": now,
                    "count": 1,
                    "label": label,
                    "object_ids": {obj_id},
                    "last_box": (left, top, width, height),
                }
            else:
                v = vehicles[gid]
                if now - v["wall_last"] > self.EPISODE_GAP_S:
                    self._new_episode(v, now, pts)
                v["last_pts"] = pts
                v["wall_last"] = max(v["wall_last"], now)   # replayed history is older than a restored stay
                v["count"] += 1
                v["object_ids"].add(obj_id)
                v["last_box"] = (left, top, width, height)

            last[gid] = pts

    @staticmethod
    def _new_episode(v: dict, first_wall: float, first_pts: float) -> None:
        """Close the vehicle's current stay on this camera and start a new one at ``first_wall``."""
        v.setdefault("episodes", []).append((v["wall_first"], v["wall_last"]))
        v["wall_first"] = first_wall
        v["first_pts"] = first_pts

    def export_state(self) -> dict:
        """Every vehicle's stays per camera, as JSON-able data (see import_state)."""
        cams: dict[str, dict[str, dict]] = {}
        for cam, per in self._camera_vehicles.items():
            cams[cam] = {
                str(gid): {
                    "first_pts": v["first_pts"], "last_pts": v["last_pts"], "wall_first": v["wall_first"],
                    "wall_last": v["wall_last"], "count": v["count"], "label": v["label"],
                    "object_ids": sorted(v["object_ids"]), "episodes": [list(e) for e in v.get("episodes", [])],
                }
                for gid, v in list(per.items())
            }
        return {"saved_at": time.time(), "cameras": cams}

    def import_state(self, state: dict) -> int:
        """Restore stays saved by export_state. They count as already reported to the platform (not
        `pending`), and replayed stream history only extends them. Returns the number of vehicles loaded."""
        n = 0
        for cam, per in (state.get("cameras") or {}).items():
            if cam not in self._camera_vehicles:
                continue
            for gid, v in per.items():
                self._camera_vehicles[cam][int(gid)] = {
                    "first_pts": v["first_pts"], "last_pts": v["last_pts"], "wall_first": v["wall_first"],
                    "wall_last": v["wall_last"], "count": v["count"], "label": v.get("label", "vehicle"),
                    "object_ids": set(v.get("object_ids", [])), "last_box": None,
                    "episodes": [tuple(e) for e in v.get("episodes", [])],
                }
                n += 1
        return n

    def episodes_of(self, gid: int) -> list[dict]:
        """Every stay of a vehicle on every camera: [{cam, first, last}] (wall clock), oldest first."""
        out = []
        for cam, per in self._camera_vehicles.items():
            v = per.get(gid)
            if not v:
                continue
            for first, last in [*v.get("episodes", []), (v["wall_first"], v["wall_last"])]:
                out.append({"cam": cam, "first": first, "last": last})
        return sorted(out, key=lambda e: e["first"])

    def note_assigned(self, cam_id: str, gid: int, pts_first: float, pts_last: float) -> None:
        """Register a sighting the per-frame entries never carried (global id assigned after
        the vehicle left, or a very brief tracklet), at the wall-clock time it really happened.
        New entries are flagged `pending` so the truck-feed bridge reports them once even
        though they are already older than its visibility window."""
        if cam_id not in self._camera_vehicles or gid < 0:
            return
        ref = self._ref.get(cam_id)
        if ref is None:
            return  # no frame seen yet for this camera: its video time can't be placed
        now = time.time()
        ref_pts, ref_wall = ref
        wall_last = min(now, ref_wall - (ref_pts - pts_last))
        wall_first = min(wall_last, ref_wall - (ref_pts - pts_first))
        v = self._camera_vehicles[cam_id].get(gid)
        if v is None:
            self._camera_vehicles[cam_id][gid] = {
                "first_pts": pts_first, "last_pts": pts_last, "wall_first": wall_first,
                "wall_last": wall_last, "count": 0, "label": "vehicle",
                "object_ids": set(), "last_box": None,
                # Only a sighting confirmed during this run is news for the truck feed;
                # older ones (replayed after a restart) were reported by the previous run.
                "pending": wall_last >= self._started - 30.0,
            }
        elif wall_first - v["wall_last"] > self.EPISODE_GAP_S:
            # a sighting confirmed late that lies after a long absence: a new stay, not a longer one
            self._new_episode(v, wall_first, pts_first)
            v["wall_last"] = wall_last
            if wall_last >= self._started - 30.0:
                v["pending"] = True
        elif wall_last < v["wall_first"] - self.EPISODE_GAP_S:
            # confirmed late and older than the current stay by more than the gap: an earlier stay
            v.setdefault("episodes", []).append((wall_first, wall_last))
            v["episodes"].sort()
        else:
            v["wall_first"] = min(v["wall_first"], wall_first)
            v["wall_last"] = max(v["wall_last"], wall_last)
            if wall_last >= self._started - 30.0:
                v["pending"] = True
        self._last_seen[cam_id][gid] = max(self._last_seen[cam_id].get(gid, pts_last), pts_last)

    def get_vehicle_states(self) -> dict[str, dict[int, dict]]:
        """Return per-camera vehicle states."""
        return self._camera_vehicles

    def get_last_seen(self) -> dict[str, dict[int, float]]:
        return self._last_seen

    def get_cameras(self) -> list[str]:
        return self.cameras


# ---------------------------------------------------------------------------
# TrackingService — main orchestrator
# ---------------------------------------------------------------------------
class TrackingService:
    """Background service that reads sidecar global_ids and serves tracking state.

    Started as an asyncio task in the FastAPI lifespan. Reads detection + global_id
    entries from Redis streams published by the DeepStream sidecar's MTMC module,
    and exposes aggregated vehicle journey data for the frontend tracker page.
    """

    def __init__(self, config: dict, manager: Any):
        self._config = config
        self._manager = manager  # ConnectionManager for WebSocket broadcast

        mtmc_cfg = config.get("mtmc", {})
        self._cameras = mtmc_cfg.get("cameras", ["cam-1", "cam-2", "cam-3"])
        self._camera_names = mtmc_cfg.get("camera_names", {
            "cam-1": "D41", "cam-2": "D52", "cam-3": "CH23",
        })
        self._redis_url = config.get("perception", {}).get("redis_url", "redis://localhost:6379/0")
        self._poll_interval = float(mtmc_cfg.get("poll_interval_s", 1.0))
        self._truck_feed = TruckFeedBridge(config.get("truck_feed") or {}, self._redis_url,
                                           mtmc_cfg.get("camera_groups"))
        # Restricted areas: a truck that stays in one past its threshold raises an alert (to the platform)
        rz = config.get("restricted_zones") or {}
        _mw, _mh = mux_resolution(config)
        self._restricted = RestrictedZoneMonitor(
            rz.get("file_path", "data/restricted_zones.json"), _mw, _mh,
            grace_s=float(rz.get("grace_s", 5.0)), cooldown_s=float(rz.get("cooldown_s", 600.0)))
        self._restricted_alerts: list[dict] = []
        # Entrance/exit counting: lines are drawn per camera in the UI and persisted here.
        # line_count.enabled: false = the gate counting lines are ignored (no gate events, no gate cameras)
        self._lines_enabled = bool((config.get("line_count") or {}).get("enabled", True))
        self.line_counter = LineCounter(
            (config.get("line_count") or {}).get("file_path", "data/count_lines.json"),
            lambda: self._r,
            frame_size=mux_resolution(config),
        )
        # Topology from sidecar config (also loaded in sidecar mtmc)
        self._topology = self._load_topology(mtmc_cfg)
        _cg = mtmc_cfg.get("camera_groups")
        self._camera_groups = (_cg if isinstance(_cg, dict) else {str(i): g for i, g in enumerate(_cg or [])})
        logger.info("[tracking] Topology loaded: %s", self._topology)

        # State
        self._r = None  # Redis client (initialized in run())
        self._last_ids: dict[str, str] = {c: "0" for c in self._cameras}
        self._last_assigned_id = "0"
        self._last_exited_id = "0"
        self._gate_cams_at = 0.0
        self._stays_saved_at = time.time()
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="tracking")
        self._last_direction_id = "0-0"
        # global id -> [{cam, direction, orientation, confidence, at}]: what the VLM decided at direction cameras
        self._directions: dict[int, list[dict]] = {}
        self._beat = time.time()       # last turn of the poll loop (monitoring)
        self._poll_errors = 0
        self._aggregator: CameraAggregator | None = None
        self._global_tracks: dict[int, GlobalTrack] = {}
        self._all_transitions: list[Transition] = []
        self._running = False
        self._start_time: float = 0.0
        self._entry_count = 0
        self._last_broadcast: float = 0.0
        # Dedup set of already-processed Redis stream entries, keyed by
        # (cam_id, msg_id). Redis stream IDs (<ms>-<seq>) are only unique WITHIN
        # one stream — different cameras' entries get identical IDs when their
        # publishes land in the same millisecond. Keying on the bare msg_id alone
        # silently dropped the later-polled camera's colliding entries, so tracks
        # showed paths skipping cameras (e.g. cam-01 -> cam-03 with no cam-02
        # even though the topology required the hop through cam-02).
        self._seen_entries: set[tuple[str, str]] = set()

    @staticmethod
    def _load_topology(mtmc_cfg: dict) -> dict[str, set[str]] | None:
        """Parse topology from config (dict, list of pairs, or ordered chain)."""
        groups = mtmc_cfg.get("camera_groups")
        if groups:
            # Same expansion as perception/modules/mtmc/core/_topology.py: cameras of a group connect to each other, and
            # neighbouring groups (group_links) connect both ways.
            named = groups if isinstance(groups, dict) else {str(i): g for i, g in enumerate(groups)}
            res: dict[str, set[str]] = {c: set() for g in named.values() for c in g}
            for g in named.values():
                for a in g:
                    res[a] |= {b for b in g if b != a}
            for g1, g2, _secs in mtmc_cfg.get("group_links") or []:
                for a in named[str(g1)]:
                    for b in named[str(g2)]:
                        res[a].add(b)
                        res[b].add(a)
            return res
        topo = mtmc_cfg.get("topology")
        if topo is None:
            return None
        if isinstance(topo, dict) and len(topo) == 1 and next(iter(topo)) in ("any", "mesh"):
            cams = [str(c) for c in next(iter(topo.values()))]
            return {a: {b for b in cams if b != a} for a in cams}
        if isinstance(topo, dict):
            return {k: set(v) for k, v in topo.items()}
        if isinstance(topo, list):
            if topo and isinstance(topo[0], (list, tuple)):
                res: dict[str, set[str]] = {}
                for src, dst in topo:
                    res.setdefault(str(src), set()).add(str(dst))
                return res
            chain = [str(x) for x in topo]
            res = {}
            for i in range(len(chain) - 1):
                res.setdefault(chain[i], set()).add(chain[i + 1])
            return res
        return None

    def _is_valid_transition(self, source_cam: str, target_cam: str) -> bool:
        """Check if target_cam is a valid next camera after source_cam in the topology."""
        if source_cam == target_cam:
            return True
        if self._topology is None:
            return True  # no topology -> allow all
        allowed = self._topology.get(source_cam, set())
        return target_cam in allowed

    # -- State accessors (called from API endpoints) -------------------------

    def get_tracks(self, limit: int = 200) -> list[dict]:
        """Return global tracks as JSON-serializable dicts, newest first."""
        tracks = sorted(
            self._global_tracks.values(),
            key=lambda t: t.last_seen_pts,
            reverse=True,
        )[:limit]
        crops = self._crop_urls()
        return [_track_to_dict(t, self._aggregator, crops.get(t.global_id), self._directions.get(t.global_id))
                for t in tracks]

    def _crop_urls(self) -> dict[int, dict[str, str]]:
        """{global_id: {cam_id: image_url}} from the sidecar's best-shot crop index."""
        out: dict[int, dict[str, str]] = {}
        newest: dict[tuple[int, str], float] = {}
        if self._r is None:
            return out
        try:
            raw = self._r.hgetall("iris:vehicle:crops")
        except Exception:
            return out
        for v in raw.values():
            try:
                e = json.loads(v)
                gid, cam = int(e["global_id"]), e["cam_id"]
                if cam in out.get(gid, {}) and float(e.get("updated_at", 0)) < newest.get((gid, cam), 0.0):
                    continue            # a camera can hold several stays' crops: this list shows the newest
                newest[(gid, cam)] = float(e.get("updated_at", 0))
                out.setdefault(gid, {})[cam] = (
                    f"/api/vehicles/crops/{cam}/{gid}/image?stay={int(e.get('stay', 0))}&v={int(e.get('updated_at', 0))}")
            except Exception:
                continue
        return out

    def get_sightings(self, gid: int) -> list[dict]:
        """Where and when (wall clock, UTC) each camera saw this vehicle, one entry per stay:
        [{cam, first, last, crop}] where ``crop`` is the image URL of that stay's best shot."""
        if self._aggregator is None:
            return []
        crops = self._crop_entries(gid)
        out = []
        for e in self._aggregator.episodes_of(gid):
            out.append({"cam": e["cam"], "first": _iso_utc(e["first"]), "last": _iso_utc(e["last"]),
                        "crop": pick_crop_url(crops.get(e["cam"], []), gid, e["cam"], e["first"], e["last"])})
        return out

    def _crop_entries(self, gid: int) -> dict[str, list[dict]]:
        """{cam: [index entries]} of this vehicle's crops (one entry per stay)."""
        out: dict[str, list[dict]] = {}
        if self._r is None:
            return out
        try:
            raw = self._r.hgetall("iris:vehicle:crops")
        except Exception:
            return out
        for v in raw.values():
            try:
                e = json.loads(v)
                if int(e["global_id"]) == gid:
                    out.setdefault(e["cam_id"], []).append(e)
            except Exception:
                continue
        return out

    def get_transitions(self, limit: int = 100) -> list[dict]:
        """Return recent transitions, newest first."""
        trans = sorted(self._all_transitions, key=lambda t: t.timestamp_s, reverse=True)[:limit]
        return [_transition_to_dict(t) for t in trans]

    def health_samples(self) -> list[tuple[str, dict, float]]:
        """(metric, labels, value) for the Prometheus endpoint."""
        tf = self._truck_feed
        out: list[tuple[str, dict, float]] = [
            ("iris_tracking_loop_age_seconds", {}, time.time() - self._beat),
            ("iris_tracking_poll_errors_total", {}, self._poll_errors),
            ("iris_tracking_tracks", {}, len(self._global_tracks)),
            ("iris_tracking_transitions_total", {}, len(self._all_transitions)),
            ("iris_truck_feed_enabled", {}, int(tf.enabled)),
            ("iris_truck_feed_last_push_age_seconds", {},
             time.time() - tf.stats["last_push_ok"] if tf.stats["last_push_ok"] else 1e9),
        ]
        for k, v in tf.stats.items():
            if k != "last_push_ok":
                out.append((f"iris_truck_feed_{k}_total", {}, v))
        return out

    def get_stats(self) -> dict:
        """Return aggregate tracking statistics."""
        gt = self._global_tracks
        now = time.time()
        stale = CameraAggregator.STALE_THRESHOLD_S
        # "active" = seen on any camera within the stale threshold
        active = sum(
            1 for t in gt.values()
            if now - t.last_seen_pts <= stale or (
                # Also check wall-clock if available from aggregator
                self._aggregator is not None and any(
                    now - v.get("wall_last", 0) <= stale
                    for cam in t.cameras_seen
                    for v in [self._aggregator.get_vehicle_states().get(cam, {}).get(t.global_id, {})]
                    if v
                )
            )
        )
        multi = sum(1 for t in gt.values() if t.total_cameras() >= 2)
        travel_times = [tr.travel_time_s for tr in self._all_transitions]
        return {
            "total_tracks": len(gt),
            "active_tracks": active,
            "total_transitions": len(self._all_transitions),
            "avg_travel_time_s": round(sum(travel_times) / len(travel_times), 2) if travel_times else 0,
            "avg_similarity": 0,
            "multi_camera_tracks": multi,
            "cameras_online": len(self._cameras),
            "uptime_s": round(time.time() - self._start_time, 1) if self._start_time else 0,
        }

    def get_camera_summary(self) -> list[dict]:
        """Return per-camera summary."""
        summary = []
        if self._aggregator is None:
            for cam in self._cameras:
                summary.append({
                    "id": cam,
                    "name": self._camera_names.get(cam, cam),
                    "active_vehicles": 0,
                    "finished_vehicles": 0,
                })
            return summary

        veh = self._aggregator.get_vehicle_states()
        now = time.time()
        stale = CameraAggregator.STALE_THRESHOLD_S
        for cam in self._cameras:
            cam_vehicles = veh.get(cam, {})
            # Only count vehicles seen within the stale threshold
            active = sum(
                1 for v in cam_vehicles.values()
                if now - v.get("wall_last", 0) <= stale
            )
            summary.append({
                "id": cam,
                "name": self._camera_names.get(cam, cam),
                "active_vehicles": active,
                "finished_vehicles": 0,
            })
        return summary

    def get_config(self) -> dict:
        """Return the tracking configuration."""
        return {
            "cameras": self._cameras,
            "camera_names": self._camera_names,
            "camera_groups": [list(g) for g in (self._camera_groups or {}).values()],
            "min_travel_s": 0,  # no longer needed — sidecar does matching
            "sim_threshold": 0,
            "running": self._running,
        }

    @property
    def is_running(self) -> bool:
        return self._running

    # -- Background task -----------------------------------------------------

    def _in_pool(self, fn, *args):
        """Run a blocking call on this service's OWN threads. The default executor is shared with the
        stream manager, which parks one thread per camera in a blocking Redis read (15 of its 20 workers),
        so a call queued there waited 1-2 s: the loop turned once every 5-8 s and the truck feed (and so
        every truck's 'visible' state, 5 s window) lagged behind."""
        return asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)

    async def run(self):
        """Main background loop. Called from FastAPI lifespan."""
        self._running = True
        self._start_time = time.time()

        # Initialize Redis connection
        try:
            import redis as redis_mod
            self._r = redis_mod.Redis.from_url(self._redis_url, decode_responses=False)
            self._r.ping()
            logger.info("TrackingService connected to Redis: %s", self._redis_url)
        except Exception as exc:
            logger.error("TrackingService cannot connect to Redis at %s: %s", self._redis_url, exc)
            self._running = False
            return

        self._aggregator = CameraAggregator(self._cameras)
        self._load_stays()
        if os.environ.get("TRACKING_REPLAY", "").strip().lower() in ("0", "false", "no", "off"):
            self._skip_history()

        logger.info("TrackingService started: cameras=%s redis=%s", self._cameras, self._redis_url)

        # Diagnostic: check if the sidecar is publishing global_ids
        for cam in self._cameras:
            key = f"iris:meta:{cam}"
            try:
                entries = self._r.xrevrange(key, count=3)
                if not entries:
                    logger.warning("[tracking] %s: no entries in %s yet", cam, key)
                else:
                    for msg_id, fields in entries[-1:]:  # just the oldest of the 3
                        dets_raw = fields.get(b"dets", b"[]")
                        gids_raw = fields.get(b"global_ids", b"{}")
                        if isinstance(dets_raw, bytes): dets_raw = dets_raw.decode()
                        if isinstance(gids_raw, bytes): gids_raw = gids_raw.decode()
                        dets = json.loads(dets_raw)
                        gids = json.loads(gids_raw)
                        logger.info(
                            "[tracking] %s sample entry: %d dets, %d global_ids=%s",
                            cam, len(dets), len(gids), list(gids.items())[:5],
                        )
            except Exception as exc:
                logger.warning("[tracking] %s diagnostic failed: %s", cam, exc)


        _log_every = 0
        while self._running:
            self._beat = time.time()
            # Poll Redis for new entries (blocking — run in executor)
            try:
                new_entries, skipped_by_cam = await self._in_pool(self._poll_redis)
                if new_entries:
                    n_with_gids = sum(1 for _, _, d in new_entries if d["global_ids"])
                    if _log_every % 10 == 0 or n_with_gids > 0:
                        skipped_total = sum(skipped_by_cam.values())
                        logger.info(
                            "[tracking] poll: %d entries (%d with global_ids, %d skipped as dups: %s), "
                            "tracks=%d, transitions=%d",
                            len(new_entries), n_with_gids, skipped_total,
                            skipped_by_cam or "none",
                            len(self._global_tracks), len(self._all_transitions),
                        )
                    self._process_entries(new_entries)
                assigned = await self._in_pool(self._poll_assigned)
                if assigned:
                    self._apply_assigned(assigned)
                if time.time() - self._gate_cams_at >= 2.0:
                    self._gate_cams_at = time.time()
                    await self._in_pool(self._publish_gate_cameras)
                await self._in_pool(self._poll_direction)
                exited = await self._in_pool(self._poll_exited)
                if exited:
                    self._apply_exited(exited)
                _log_every += 1
                self._restricted.expire(time.time())
                for alert in self._restricted_alerts:
                    if not alert["update"]:
                        who = f"gid={alert['gid']}" if alert["gid"] is not None else "untracked elsewhere"
                        logger.warning("[restricted] %s (#%d on %s) stayed %.0f s in '%s' (threshold %.0f s)",
                                       who, alert["obj_id"], alert["cam_id"], alert["dwell_s"],
                                       alert["zone_name"], alert["threshold_s"])
                    self._truck_feed.note_alert(alert)
                self._restricted_alerts.clear()
                if self._truck_feed.due():
                    await self._in_pool(self._truck_feed.push, self._aggregator.get_vehicle_states())
                if time.time() - self._stays_saved_at >= self.STAYS_SAVE_S:
                    self._stays_saved_at = time.time()
                    await self._in_pool(self._save_stays)
            except Exception as exc:
                self._poll_errors += 1
                logger.warning("TrackingService poll error: %s", exc)

            # Periodic broadcast (every 2s)
            now = time.time()
            if now - self._last_broadcast >= 2.0:
                self._last_broadcast = now
                await self._broadcast_tracking_update()

            await asyncio.sleep(self._poll_interval)

    async def stop(self):
        """Stop the tracking service."""
        self._running = False
        self._save_stays()
        self._pool.shutdown(wait=False)
        if self._r:
            try:
                self._r.close()
            except Exception:
                pass
        logger.info("TrackingService stopped")

    # -- Internal helpers ----------------------------------------------------

    STAYS_SAVE_S = 60.0
    STAYS_PATH = "data/tracking_stays.json"

    def _save_stays(self) -> None:
        """Keep each vehicle's stays on disk: they live only in memory otherwise, and the Redis streams they
        are rebuilt from hold the last 2000 entries per camera, so a restart would lose most of a day."""
        if self._aggregator is None:
            return
        try:
            tmp = self.STAYS_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._aggregator.export_state(), f)
            os.replace(tmp, self.STAYS_PATH)
        except Exception as exc:
            logger.warning("[tracking] cannot save the vehicle stays: %s", exc)

    def _skip_history(self) -> None:
        """Start reading every Redis stream at its newest entry instead of replaying it from the start. An
        imported run (browse mode) gets its vehicles' stays from the saved stays file only: replaying the streams
        would add them a second time, at the times they were published."""
        for key, attr in ((None, "_last_ids"), ("iris:mtmc:assigned", "_last_assigned_id"),
                          ("iris:mtmc:exited", "_last_exited_id"), ("iris:mtmc:direction", "_last_direction_id")):
            try:
                if key is None:
                    for cam in self._cameras:
                        newest = self._r.xrevrange(f"iris:meta:{cam}", count=1)
                        if newest:
                            mid = newest[0][0]
                            self._last_ids[cam] = mid.decode() if isinstance(mid, bytes) else mid
                else:
                    newest = self._r.xrevrange(key, count=1)
                    if newest:
                        mid = newest[0][0]
                        setattr(self, attr, mid.decode() if isinstance(mid, bytes) else mid)
            except Exception as exc:
                logger.warning("[tracking] cannot skip the history of %s: %s", key or "the camera streams", exc)
        logger.info("[tracking] TRACKING_REPLAY=0: Redis history is not replayed")

    def _load_stays(self) -> None:
        try:
            with open(self.STAYS_PATH) as f:
                n = self._aggregator.import_state(json.load(f))
            logger.info("[tracking] restored the stays of %d vehicle-camera pairs from %s", n, self.STAYS_PATH)
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("[tracking] cannot restore the vehicle stays: %s", exc)

    def _poll_redis(self) -> tuple[list[tuple[str, str, dict]], dict[str, int]]:
        """Read new messages from all camera streams.

        Returns (results, skipped_by_cam): the parsed entries and, per camera,
        how many entries were skipped as already-seen (diagnostic for dropped-
        entry debugging — see the warning below).
        """
        results = []
        skipped_by_cam: dict[str, int] = {}
        for cam in self._cameras:
            key = f"iris:meta:{cam}"
            try:
                entries = self._r.xread({key: self._last_ids[cam]}, count=50)
            except Exception:
                continue
            if not entries:
                continue
            skipped = 0
            for _name, messages in entries:
                for msg_id, fields in messages:
                    msg_id_str = msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)
                    # Redis stream IDs (<ms>-<seq>) are only unique WITHIN one
                    # stream: different cameras' entries get identical IDs when
                    # their publishes land in the same millisecond. Dedup must
                    # therefore key on (cam, msg_id) — keying on msg_id alone
                    # dropped the later-polled camera's colliding entries, and
                    # whole cameras vanished from track paths.
                    dedup_key = (cam, msg_id_str)
                    if dedup_key in self._seen_entries:
                        self._last_ids[cam] = msg_id
                        skipped += 1
                        continue
                    self._last_ids[cam] = msg_id
                    self._seen_entries.add(dedup_key)

                    # Parse fields
                    dets_raw = fields.get(b"dets", fields.get("dets", b"[]"))
                    if isinstance(dets_raw, bytes):
                        dets_raw = dets_raw.decode()
                    dets = json.loads(dets_raw) if dets_raw else []

                    pts_raw = fields.get(b"pts", fields.get("pts", b"0"))
                    if isinstance(pts_raw, bytes):
                        pts_raw = pts_raw.decode()
                    pts = float(pts_raw) if pts_raw else 0.0

                    gids_raw = fields.get(b"global_ids", fields.get("global_ids", b"{}"))
                    if isinstance(gids_raw, bytes):
                        gids_raw = gids_raw.decode()
                    global_ids = json.loads(gids_raw) if gids_raw else {}

                    # Include entries that have detections OR global_ids.
                    # global_ids may exist when dets is empty (e.g. filter zones
                    # dropped detections but MTMC assigned IDs before the filter
                    # loaded).  Always include so _process_entries can consume them.
                    if dets or global_ids:
                        try:
                            wall = int(msg_id_str.split("-")[0]) / 1000.0
                        except ValueError:
                            wall = None
                        results.append((cam, msg_id_str, {
                            "dets": dets,
                            "global_ids": global_ids,
                            "pts": pts,
                            "wall": wall,
                        }))
                    elif self._entry_count == 0:
                        # Log a few empty entries at startup for diagnostics
                        logger.debug("[tracking] %s: entry with %d dets, %d gids", cam, len(dets), len(global_ids))

                    # Trim seen_entries to prevent unbounded growth
                    if len(self._seen_entries) > 10000:
                        # Keep only the most recent 5000
                        self._seen_entries = set(list(self._seen_entries)[-5000:])

            if skipped:
                skipped_by_cam[cam] = skipped_by_cam.get(cam, 0) + skipped
                # With (cam, msg_id) keying a repeat should never happen (stream
                # IDs are unique within a stream); if it does, an entry was
                # re-read — e.g. _last_ids fell behind, or the publisher
                # double-XADDed. Loud so it can't silently corrupt tracks again.
                logger.warning(
                    "[tracking] %s: %d entries re-read and skipped as already-seen "
                    "(cam, msg_id) duplicates — check stream trimming or the "
                    "publisher for double-publishing",
                    cam, skipped,
                )

        return results, skipped_by_cam

    def _record_camera_appearance(self, gt: GlobalTrack, cam_id: str, pts: float) -> str | None:
        """Record that a track was seen on a new camera, warning on topology skips.

        Appends cam_id to the track's seen-camera list and returns the previous
        camera (the one right before cam_id in the seen order) so callers can
        compute travel times. If the hop violates the configured topology, an
        intermediate camera sighting was lost between the sidecar's publish and
        this reader (dropped/trimmed Redis entry, or the pre-fix cross-stream
        dedup collision) — the sidecar only merges global IDs along
        topology-valid hops, so a jump like cam-01 -> cam-03 implies cam-02 was
        seen but its entries never reached this service.
        """
        gt.cameras_seen.append(cam_id)
        gt.entry_times[cam_id] = pts
        gt.exit_times[cam_id] = pts
        gt.observation_counts[cam_id] = 1

        prev_cam = gt.cameras_seen[-2] if len(gt.cameras_seen) >= 2 else None
        if prev_cam and not self._is_valid_transition(prev_cam, cam_id):
            allowed = sorted(self._topology.get(prev_cam, [])) if self._topology else "all"
            logger.warning(
                "[tracking] gid=%d appeared on %s after %s without a valid topology "
                "transition (allowed next from %s: %s) — an intermediate camera "
                "sighting was likely missed",
                gt.global_id, cam_id, prev_cam, prev_cam, allowed,
            )
        return prev_cam

    def _record_transition(self, gt: GlobalTrack, gid: int, prev_cam: str | None,
                           cam_id: str, pts: float, label: str) -> None:
        """Record a prev_cam -> cam_id hop (if topology-valid) with its travel time."""
        if not (prev_cam and self._is_valid_transition(prev_cam, cam_id)):
            return
        # prev_exit: last time the vehicle was seen on prev_cam; if no exit yet,
        # it is still there, so use the last observation.
        prev_exit = gt.exit_times.get(prev_cam)
        if prev_exit is None:
            prev_exit = gt.last_seen_pts
        travel = abs(pts - prev_exit)
        self._all_transitions.append(Transition(
            global_id=gid, from_cam=prev_cam, to_cam=cam_id,
            travel_time_s=round(travel, 2), timestamp_s=pts, label=label,
        ))
        if prev_cam not in gt.exit_times or gt.exit_times[prev_cam] < prev_exit:
            gt.exit_times[prev_cam] = prev_exit
        gt.transitions.append({
            "from_cam": prev_cam, "to_cam": cam_id, "similarity": 0,
            "travel_time_s": round(travel, 2), "timestamp_s": round(pts, 2),
        })
        logger.info("TRANSITION: gid=%d %s -> %s (travel=%.1fs, pts=%.2f)",
                    gid, prev_cam, cam_id, travel, pts)

    def _poll_assigned(self) -> list[dict]:
        """New global-id assignment events from the sidecar (iris:mtmc:assigned)."""
        try:
            entries = self._r.xread({"iris:mtmc:assigned": self._last_assigned_id}, count=100)
        except Exception:
            return []
        out = []
        for _name, messages in entries or []:
            for msg_id, fields in messages:
                self._last_assigned_id = msg_id
                try:
                    d = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
                         for k, v in fields.items()}
                    out.append({"cam_id": d["cam_id"], "global_id": int(d["global_id"]),
                                "pts_first": float(d["pts_first"]), "pts_last": float(d["pts_last"])})
                except Exception:
                    continue
        return out

    def _publish_gate_cameras(self) -> None:
        """Tell the matcher which cameras have a gate line (iris:mtmc:gate:cams)."""
        try:
            cams = self.line_counter.cameras() if self._lines_enabled else []
            pipe = self._r.pipeline()
            pipe.delete("iris:mtmc:gate:cams")
            if cams:
                pipe.sadd("iris:mtmc:gate:cams", *cams)
            pipe.execute()
        except Exception as exc:
            logger.warning("[tracking] could not publish gate cameras: %s", exc)

    def _publish_gate_event(self, ev: dict) -> None:
        """Tell the sidecar's matcher which way a tracklet crossed the gate line: an entering
        one is a new vehicle; a leaving one must be matched to a vehicle inside and closed."""
        try:
            self._r.xadd("iris:mtmc:gate", {
                "cam_id": ev["cam_id"], "local_id": str(ev["object_id"]),
                "direction": ev["direction"], "at": str(ev["at"]),
            }, maxlen=2000, approximate=True)
        except Exception as exc:
            logger.warning("[tracking] could not publish gate event: %s", exc)

    def _poll_direction(self) -> None:
        """Entering / leaving verdicts of the VLM (iris:mtmc:direction), kept per vehicle."""
        try:
            entries = self._r.xread({"iris:mtmc:direction": self._last_direction_id}, count=100)
        except Exception:
            return
        for _name, messages in entries or []:
            for msg_id, fields in messages:
                self._last_direction_id = msg_id
                try:
                    d = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
                         for k, v in fields.items()}
                    lst = self._directions.setdefault(int(d["global_id"]), [])
                    lst.append({"cam": d["cam_id"], "direction": d["direction"], "orientation": d.get("orientation"),
                                "confidence": None if d.get("confidence") in (None, "None") else float(d["confidence"]),
                                "at": float(d.get("at") or 0.0)})
                    del lst[:-20]
                except Exception:
                    continue

    def _poll_exited(self) -> list[dict]:
        """Vehicles the sidecar matched to a gate exit (iris:mtmc:exited)."""
        try:
            entries = self._r.xread({"iris:mtmc:exited": self._last_exited_id}, count=100)
        except Exception:
            return []
        out = []
        for _name, messages in entries or []:
            for msg_id, fields in messages:
                self._last_exited_id = msg_id
                try:
                    d = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
                         for k, v in fields.items()}
                    mid = msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)
                    out.append({"global_id": int(d["global_id"]), "cam_id": d["cam_id"],
                                "at": float(d.get("at") or 0.0), "reopens": d.get("reopens") in ("1", "True", "true"),
                                # when the sidecar published it (the stream id's ms): fresh vs replayed history
                                "emitted": int(mid.split("-")[0]) / 1000.0})
                except Exception:
                    continue
        return out

    def _apply_exited(self, events: list[dict]) -> None:
        for ev in events:
            gid = ev["global_id"]
            gt = self._global_tracks.get(gid)
            if gt is None:
                gt = self._global_tracks[gid] = GlobalTrack(
                    global_id=gid, cameras_seen=[ev["cam_id"]], label="vehicle")
            if ev.get("reopens"):
                # Its visit is complete but the vehicle is back (a long absence at the entrance): report the
                # exit at its last sighting and let the next sighting start a new visit.
                logger.info("VISIT COMPLETE: gid=%d was out of sight until it came back to %s", gid, ev["cam_id"])
                # Its time ("at") is deliberately in the past (the last sighting before the absence), so
                # freshness is judged by when the event was published, not by "at".
                if ev.get("emitted", ev["at"]) >= self._start_time - 30.0:
                    self._truck_feed.note_exit(gid, ev["cam_id"], ev["at"], reopens=True)
                continue
            if gt.exited:
                continue
            gt.exited = True
            gt.exited_wall = ev["at"]
            gt.is_active = False
            logger.info("EXITED: gid=%d left the premises via %s", gid, ev["cam_id"])
            # Events from before this run were reported to the platform by the run that saw them.
            if ev["at"] >= self._start_time - 30.0:
                self._truck_feed.note_exit(gid, ev["cam_id"], ev["at"])
            else:
                # history read back after a restart: reported by an earlier run, or still due (see note_exit)
                self._truck_feed.note_exit(gid, ev["cam_id"], ev["at"], replay=True)

    def _apply_assigned(self, events: list[dict]) -> None:
        """Register each (camera, global id) the sidecar assigned, so a sighting the per-frame
        entries missed (late verdict, very brief tracklet) still lands on the track."""
        for ev in events:
            cam_id, gid = ev["cam_id"], ev["global_id"]
            first, last = ev["pts_first"], ev["pts_last"]
            if self._aggregator is not None:
                self._aggregator.note_assigned(cam_id, gid, first, last)
            gt = self._global_tracks.get(gid)
            if gt is None:
                self._global_tracks[gid] = GlobalTrack(
                    global_id=gid, cameras_seen=[cam_id], entry_times={cam_id: first},
                    exit_times={cam_id: last}, observation_counts={cam_id: 1}, label="vehicle",
                    first_seen_pts=first, last_seen_pts=last)
                logger.info("NEW TRACK (assignment event): gid=%d on %s (pts=%.2f-%.2f)", gid, cam_id, first, last)
            elif cam_id in gt.cameras_seen:
                gt.exit_times[cam_id] = max(gt.exit_times.get(cam_id, last), last)
            else:
                prev_cam = self._record_camera_appearance(gt, cam_id, first)
                gt.exit_times[cam_id] = last
                self._record_transition(gt, gid, prev_cam, cam_id, first, gt.label)
                gt.last_seen_pts = max(gt.last_seen_pts, last)
                logger.info("ASSIGNED: gid=%d now on %s (pts=%.2f-%.2f)", gid, cam_id, first, last)

    def _process_entries(self, entries: list[tuple[str, str, dict]]) -> None:
        """Process new Redis entries and update global tracks."""
        for cam_id, _msg_id, data in entries:
            dets = data["dets"]
            global_ids = data["global_ids"]
            pts = data["pts"]

            if global_ids:
                logger.info("[tracking] %s pts=%.1f: gids=%s dets=%d", cam_id, pts, global_ids, len(dets))

            # Entrance/exit counting. Frames re-read after a restart are history that the
            # previous run already counted.
            wall = data.get("wall")
            if self._lines_enabled and (wall is None or wall >= self._start_time - 2.0):
                for ev in self.line_counter.process(cam_id, dets, global_ids, wall):
                    self._publish_gate_event(ev)

            # Restricted areas look at the camera's own detections, so a vehicle seen by this camera alone is
            # covered; frames re-read after a restart are history.
            if wall is None or wall >= self._start_time - 2.0:
                self._restricted_alerts.extend(
                    self._restricted.observe(cam_id, dets, global_ids, wall if wall is not None else time.time()))

            # Feed to aggregator
            self._aggregator.process_entry(cam_id, dets, global_ids, pts, data.get("wall"))

            # Update global tracks
            seen_gids: set[int] = set()
            for det in dets:
                if len(det) < 7:
                    continue
                obj_id = int(det[2])
                label = det[0]

                gid = global_ids.get(str(obj_id))
                if gid is None:
                    gid = global_ids.get(obj_id)
                if gid is None:
                    continue
                gid = int(gid)
                seen_gids.add(gid)

                if gid not in self._global_tracks:
                    self._global_tracks[gid] = GlobalTrack(
                        global_id=gid,
                        cameras_seen=[cam_id],
                        entry_times={cam_id: pts},
                        exit_times={cam_id: pts},
                        observation_counts={cam_id: 1},
                        label=label,
                        first_seen_pts=pts,
                        last_seen_pts=pts,
                    )
                    logger.info("NEW TRACK: gid=%d on %s (label=%s, pts=%.2f)", gid, cam_id, label, pts)
                else:
                    gt = self._global_tracks[gid]
                    gt.label = label  # update label in case it changed
                    gt.is_active = True

                    if cam_id not in gt.cameras_seen:
                        # Vehicle appeared on a new camera — this is a transition!
                        # (also warns if the hop violates topology = lost sighting)
                        prev_cam = self._record_camera_appearance(gt, cam_id, pts)

                        self._record_transition(gt, gid, prev_cam, cam_id, pts, label)
                    else:
                        gt.exit_times[cam_id] = pts
                        gt.observation_counts[cam_id] = gt.observation_counts.get(cam_id, 0) + 1

                    gt.last_seen_pts = pts

            # When dets is empty but global_ids exist (e.g. filter zones dropped
            # all detections but MTMC assigned IDs), update tracks directly from
            # global_ids so they aren't silently lost.
            if not dets and global_ids:
                for _oid_str, gid_raw in global_ids.items():
                    gid = int(gid_raw)
                    if gid not in self._global_tracks:
                        self._global_tracks[gid] = GlobalTrack(
                            global_id=gid,
                            cameras_seen=[cam_id],
                            entry_times={cam_id: pts},
                            exit_times={cam_id: pts},
                            observation_counts={cam_id: 1},
                            label="unknown",
                            first_seen_pts=pts,
                            last_seen_pts=pts,
                        )
                        logger.info("NEW TRACK (no-dets): gid=%d on %s (pts=%.2f)", gid, cam_id, pts)
                    else:
                        gt = self._global_tracks[gid]
                        gt.last_seen_pts = pts
                        gt.is_active = True
                        if cam_id not in gt.cameras_seen:
                            self._record_camera_appearance(gt, cam_id, pts)
                        else:
                            gt.exit_times[cam_id] = pts
                            gt.observation_counts[cam_id] = gt.observation_counts.get(cam_id, 0) + 1

            self._entry_count += 1

    async def _broadcast_tracking_update(self) -> None:
        """Broadcast full tracking state update."""
        try:
            await self._manager.broadcast({
                "type": "tracking_update",
                "tracks": self.get_tracks(limit=100),
                "stats": self.get_stats(),
                "cameras": self.get_camera_summary(),
            })
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
CROP_MATCH_SLACK_S = 60.0


def pick_crop_url(entries: list[dict], gid: int, cam: str, first: float, last: float) -> str | None:
    """The crop of the stay [first, last] among a camera's crop entries: the one whose best shot was taken
    inside the stay (plus a minute of slack for a verdict that came late), else the closest in time."""
    if not entries:
        return None

    def distance(e: dict) -> float:
        t = float(e.get("updated_at", 0.0))
        return 0.0 if first - CROP_MATCH_SLACK_S <= t <= last + CROP_MATCH_SLACK_S else min(abs(t - first), abs(t - last))

    best = min(entries, key=lambda e: (distance(e), -float(e.get("updated_at", 0.0))))
    return f"/api/vehicles/crops/{cam}/{gid}/image?stay={int(best.get('stay', 0))}&v={int(best.get('updated_at', 0))}"


def _iso_utc(epoch_s: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(epoch_s)) + f".{int(epoch_s % 1 * 1000):03d}Z"


def _track_to_dict(t: GlobalTrack, aggregator: CameraAggregator | None = None,
                   crops: dict[str, str] | None = None, directions: list[dict] | None = None) -> dict:
    # Determine if currently active: seen within stale threshold on any camera
    is_active = False
    now = time.time()
    if aggregator is not None:
        veh = aggregator.get_vehicle_states()
        for cam in t.cameras_seen:
            v = veh.get(cam, {}).get(t.global_id)
            if v and now - v.get("wall_last", 0) <= CameraAggregator.STALE_THRESHOLD_S:
                is_active = True
                break
    return {
        "global_id": t.global_id,
        "cameras_seen": t.cameras_seen,
        "total_cameras": t.total_cameras(),
        "local_track_ids": {},
        "entry_times": {k: round(v, 2) for k, v in t.entry_times.items()},
        "exit_times": {k: round(v, 2) for k, v in t.exit_times.items()},
        "label": t.label,
        "transitions": t.transitions,
        "is_active": is_active and not t.exited,
        "exited": t.exited,
        "first_seen_s": round(t.first_seen_pts, 2),
        "last_seen_s": round(t.last_seen_pts, 2),
        "total_duration_s": round(t.total_duration_s(), 2),
        "crops": crops or {},
        "directions": directions or [],
    }


def _transition_to_dict(tr: Transition) -> dict:
    return {
        "global_id": tr.global_id,
        "from_cam": tr.from_cam,
        "to_cam": tr.to_cam,
        "similarity": 0,
        "travel_time_s": tr.travel_time_s,
        "timestamp_s": round(tr.timestamp_s, 2),
    }
