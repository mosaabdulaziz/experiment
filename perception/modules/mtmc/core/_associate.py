"""Cross-camera vehicle association: maintains persistent global vehicle identities across cameras.

Operating strictly on vehicle targets (car, truck, bus, motorcycle, bicycle),
this module maintains global vehicle track state (active and inactive/lost),
accumulates historical appearance crops and embeddings across cameras, and
performs multi-camera vehicle association using camera topology and 20-second
disappearance rules.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ._topology import expand_camera_groups

if TYPE_CHECKING:
    from ._tracklets import TrackletRecord

logger = logging.getLogger("iris.mtmc")

_WAITING = object()  # sentinel: VLM arbitration job submitted, not yet resolved


@dataclass
class GlobalVehicleTrack:
    """Persistent state for a vehicle tracked across one or more cameras.

    Accumulates historical crops, embeddings, and observation timing from all
    cameras the vehicle has visited (e.g. Camera 1 -> Camera 2 -> Camera 3).
    """

    global_id: int
    last_camera_id: str
    last_seen_pts: float
    disappeared_pts: float | None = None
    state: str = "ACTIVE"  # "ACTIVE" | "INACTIVE"
    last_gap_s: float = 0.0  # seconds out of sight of every camera before the latest observation
    # The vehicle left the premises (crossed the gate line in the exit direction). A closed
    # track is never a candidate again: a vehicle coming back in is a new visit, a new track.
    exited: bool = False
    camera_history: list[str] = field(default_factory=list)
    local_track_ids: dict[str, int] = field(default_factory=dict)  # cam_id -> latest local_track_id
    # Every tracker id that is this vehicle on a camera: its camera's tracker can split it in two
    # (people moving on a pickup), and both ids must keep resolving to the one global id.
    local_track_aliases: dict[str, set] = field(default_factory=dict)  # cam_id -> {local_track_id}
    # Per-camera embeddings and crops
    embeddings: dict[str, list[np.ndarray]] = field(default_factory=dict)
    mean_embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    crops: dict[str, list[np.ndarray]] = field(default_factory=dict)
    # Per-camera time-spread views of the vehicle (replaced, not accumulated) -- what the VLM sees.
    views: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def add_camera_observation(
        self,
        cam_id: str,
        local_id: int,
        pts: float,
        embedding: np.ndarray | None = None,
        crops: list[np.ndarray] | None = None,
        views: list[np.ndarray] | None = None,
    ) -> None:
        """Update global track with an observation from a camera."""
        # How long this vehicle was out of sight of EVERY camera before this observation (read right
        # after the observation that first assigns a returning tracklet; see MTMCManager).
        self.last_gap_s = max(0.0, pts - self.last_seen_pts)
        self.last_camera_id = cam_id
        self.last_seen_pts = pts
        self.disappeared_pts = None
        self.state = "ACTIVE"
        if not self.camera_history or self.camera_history[-1] != cam_id:
            self.camera_history.append(cam_id)
        self.local_track_ids[cam_id] = local_id
        self.local_track_aliases.setdefault(cam_id, set()).add(local_id)

        if embedding is not None:
            if cam_id not in self.embeddings:
                self.embeddings[cam_id] = []
            self.embeddings[cam_id].append(embedding)
            # Recompute mean embedding for this camera
            embs = np.vstack(self.embeddings[cam_id])
            mean = embs.mean(axis=0)
            norm = np.linalg.norm(mean)
            self.mean_embeddings[cam_id] = mean / norm if norm > 0 else mean

        if views:
            self.views[cam_id] = list(views)
        if crops:
            if cam_id not in self.crops:
                self.crops[cam_id] = []
            self.crops[cam_id].extend(crops)
            if len(self.crops[cam_id]) > 16:
                self.crops[cam_id] = self.crops[cam_id][-16:]

    def mark_inactive(self, pts: float) -> None:
        """Transition global track to INACTIVE state when it disappears from camera."""
        self.state = "INACTIVE"
        if self.disappeared_pts is None:
            self.disappeared_pts = pts

    def combined_embedding(self) -> np.ndarray | None:
        """Mean of all mean embeddings across all historical camera observations."""
        if not self.mean_embeddings:
            return None
        all_means = list(self.mean_embeddings.values())
        mean = np.mean(all_means, axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 0 else None

    def max_similarity_against(self, new_emb: np.ndarray) -> float:
        """Compute max cosine similarity against all historical camera embeddings."""
        if new_emb is None:
            return 0.0
        best_sim = -1.0
        # Compare against per-camera mean embeddings
        for cam_id, m_emb in self.mean_embeddings.items():
            sim = float(np.dot(m_emb, new_emb))
            if sim > best_sim:
                best_sim = sim
        # Compare against combined mean embedding across cameras
        comb = self.combined_embedding()
        if comb is not None:
            sim = float(np.dot(comb, new_emb))
            if sim > best_sim:
                best_sim = sim
        return max(best_sim, 0.0)


class GlobalVehicleAssociation:
    """Association engine for multi-camera vehicle tracking.

    Maintains active and inactive GlobalVehicleTracks across the camera network,
    evaluating camera topology, 20-second disappearance timing gates, and
    multi-camera appearance similarity.
    """

    def __init__(
        self,
        topology: dict[str, list[str]] | list[tuple[str, str]] | list[str] | None = None,
        min_disappearance_s: float = 20.0,
        max_inactive_ttl_s: float = 300.0,
        reid_sim_threshold: float = 0.55,
        reid_gate_threshold: float = 0.15,
        camera_gaps: Any = None,
        vlm_arbiter: Any = None,
        vlm_max_candidates: int = 3,
        vlm_views: int = 3,
        origin_cameras: list[str] | None = None,
        vlm_history_views: int = 1,
        gate_wait_s: float = 10.0,
        same_camera_regain_s: float = 0.0,
        overlap_tolerance_s: float = 5.0,
        camera_groups: Any = None,
        group_links: Any = None,
    ):
        self.min_disappearance_s = min_disappearance_s
        # <= 0: never forget a vehicle that is out of sight; only its exit (or a restart) ends it
        self.max_inactive_ttl_s = max_inactive_ttl_s if max_inactive_ttl_s > 0 else float("inf")
        self.reid_sim_threshold = reid_sim_threshold
        self.reid_gate_threshold = reid_gate_threshold
        # Gate must not exceed the match threshold (gate <= match), otherwise
        # the gate would silently become the merge threshold.
        if self.reid_gate_threshold > self.reid_sim_threshold:
            self.reid_gate_threshold = self.reid_sim_threshold
        self._topology_map = self._parse_topology(topology)
        self._camera_gaps_map = self._parse_camera_gaps(camera_gaps)
        # Cameras that see the same area, and the links between neighbouring groups (see _topology.py).
        # When given they define the topology, the minimum travel time of every allowed pair (explicit
        # camera_gaps still win) and which pairs may overlap (overlap_tolerance_s applies to those only).
        self._overlap_pairs: set[tuple[str, str]] | None = None
        if camera_groups:
            topo, gaps, self._overlap_pairs = expand_camera_groups(camera_groups, group_links)
            self._topology_map = topo
            self._camera_gaps_map = {**gaps, **self._camera_gaps_map}
        logger.info("[mtmc] Topology map: %s", self._topology_map)
        self.global_tracks: dict[int, GlobalVehicleTrack] = {}  # global_id -> GlobalVehicleTrack
        self._next_gid: int = 0
        # Called with the next free id each time one is handed out, so a restarted sidecar can carry on
        # numbering where it stopped instead of reusing ids the app and the UI already hold.
        self.on_new_gid = None
        # VLM arbiter (perception/modules/mtmc/vlm_arbiter.py) validates every proposed
        # cross-camera association -- topology-only, high-confidence re-id, or
        # ambiguous -- before it is allowed to merge. Optional -- None/disabled
        # means associations merge exactly as before this feature existed.
        self._vlm = vlm_arbiter
        # With a VLM enabled, topology + timing only decide WHICH tracks are
        # eligible; the VLM decides every merge. At most this many eligible
        # candidates (best re-id similarity first, then closest in time) are
        # sent to the VLM per tracklet, to bound API calls in busy scenes.
        self._vlm_max_candidates = max(1, int(vlm_max_candidates))
        # Crops per vehicle sent to the VLM, spread over the vehicle's time on camera.
        self._vlm_views = max(1, int(vlm_views))
        # Crops sent from EACH earlier camera the candidate passed through (the camera it
        # was last on contributes vlm_views).
        self._vlm_history_views = max(0, int(vlm_history_views))
        # Only vehicles that first appear on one of these cameras are tracked. A tracklet
        # on any other camera is tracked only if it is matched to an existing track; an
        # unmatched one gets no global id (never creates a track). None = every camera
        # may start a track.
        self._origin_cameras = set(map(str, origin_cameras)) if origin_cameras else None
        # (cam_id, local_id) decided "not tracked" -- the decision is final, like a match.
        self._untracked: set[tuple[str, int]] = set()
        # (cam_id, local_id) -> (direction, wall time): which way the tracklet crossed the gate
        # line ("enter" | "exit"), reported by the tracking service. An entering vehicle is a
        # new one (no need to match it against the vehicles inside); an exiting one must be
        # matched with a vehicle inside, and that vehicle's track is then closed.
        self._gate_hint: dict[tuple[str, int], tuple[str, float]] = {}
        # Cameras that have a gate line, and how long a new vehicle there waits for the line to
        # tell which way it is going (pts seconds since it was first seen) before being judged
        # without that information.
        self._gate_cams: set[str] = set()
        self._gate_wait_s = float(gate_wait_s)
        # Cameras whose vehicles are classified entering / leaving by the VLM (which end faces the
        # camera). A new vehicle there waits up to _direction_wait_s for that verdict before it is judged;
        # the manager settles it (verdict, timeout or no VLM) with settle_direction().
        self.alias_conflicts = 0       # times one tracker id was found under two global ids (must stay 0)
        self._direction_cams: set[str] = set()
        self._direction_wait_s = 25.0
        self._direction_settled: set[tuple[str, int]] = set()
        # direction_require: on a direction camera a vehicle starts a track ONLY when the VLM decided it is entering;
        # leaving, unclear or unanswered vehicles never start one (they may still join a track they match).
        self._direction_require = False
        self._dir_entering: set[tuple[str, int]] = set()
        self._same_camera_regain_s = float(same_camera_regain_s)
        # An ACTIVE track seen on another camera within this many seconds (either side) may be the
        # same vehicle in overlapping fields of view. 0 disables it: cameras that do not overlap can
        # never see one vehicle at the same instant.
        self._overlap_tolerance_s = float(overlap_tolerance_s)
        # (cam_id, local_id) -> {"jobs": [...], "results": {job_id: verdict}, "submitted_pts": float}
        self._pending: dict[tuple[str, int], dict] = {}
        # Global ids each unresolved sighting has already been compared with, so that a vehicle
        # which appeared while its question was out (the other gate camera's sighting of the same
        # car) is asked about before a new global id is created.
        self._judged: dict[tuple[str, int], set[int]] = {}

    def _parse_camera_gaps(self, camera_gaps: Any) -> dict[tuple[str, str], float]:
        """Parse camera gaps into dict mapping (src_cam, dst_cam) -> min_disappearance_s."""
        if not camera_gaps:
            return {}
        res: dict[tuple[str, str], float] = {}
        if isinstance(camera_gaps, dict):
            for k, v in camera_gaps.items():
                if isinstance(v, dict):
                    # Nested dict: {"cam-1": {"cam-2": 20.0}}
                    for dst, gap in v.items():
                        res[(str(k), str(dst))] = float(gap)
                elif isinstance(k, tuple) and len(k) == 2:
                    # Tuple key: {("cam-1", "cam-2"): 20.0}
                    res[(str(k[0]), str(k[1]))] = float(v)
                elif isinstance(k, str):
                    # String key: "cam-1->cam-2": 20.0 or "cam-1:cam-2": 20.0
                    delim = "->" if "->" in k else (":" if ":" in k else None)
                    if delim:
                        src, dst = k.split(delim, 1)
                        res[(src.strip(), dst.strip())] = float(v)
        return res

    def get_min_disappearance_s(self, source_cam: str, target_cam: str) -> float:
        """Return per-camera-pair min_disappearance_s, or fallback default."""
        # Cameras connect both ways: a pair listed in one direction applies to the other too.
        gap = self._camera_gaps_map.get((source_cam, target_cam))
        if gap is None:
            gap = self._camera_gaps_map.get((target_cam, source_cam))
        if gap is not None:
            return gap
        return self.min_disappearance_s

    def _parse_topology(
        self, topology: dict[str, list[str]] | list[tuple[str, str]] | list[str] | None
    ) -> dict[str, set[str]] | None:
        """Parse topology format into a dict mapping cam_id -> set of allowed next cam_ids."""
        if topology is None:
            return None
        if isinstance(topology, str):
            # "any": every camera may follow every other, in both directions; only the
            # per-pair minimum travel times (camera_gaps) then filter candidates.
            return None
        if isinstance(topology, dict) and len(topology) == 1 and next(iter(topology)) in ("any", "mesh"):
            # {any: [cam-1, cam-2, ...]}: those cameras are connected both ways, each to every
            # other; any camera not listed takes no part (a site may stream more cameras than
            # it tracks across).
            cams = [str(c) for c in next(iter(topology.values()))]
            return {a: {b for b in cams if b != a} for a in cams}
        if isinstance(topology, dict):
            return {k: set(v) for k, v in topology.items()}
        if isinstance(topology, list):
            res: dict[str, set[str]] = {}
            if len(topology) > 0 and isinstance(topology[0], (list, tuple)):
                for src, dst in topology:
                    res.setdefault(str(src), set()).add(str(dst))
                return res
            chain = [str(x) for x in topology]
            for i in range(len(chain) - 1):
                res.setdefault(chain[i], set()).add(chain[i + 1])
            return res
        return None

    def overlap_tolerance_for(self, source_cam: str, target_cam: str) -> float:
        """How many seconds apart an ACTIVE track on `source_cam` and a new sighting on `target_cam` may
        be and still be one vehicle in overlapping views. With camera groups only cameras of the same
        group overlap; otherwise the one global setting applies to every pair."""
        if self._overlap_pairs is None:
            return self._overlap_tolerance_s
        return self._overlap_tolerance_s if (source_cam, target_cam) in self._overlap_pairs else 0.0

    def _exit_overlap_ok(self, gtr: "GlobalVehicleTrack", cam_id: str, pts: float) -> bool:
        """A vehicle that has left is still its overlapping neighbour's vehicle for a moment: the cameras of
        one group see the same lane, and the one that saw it cross the gate may report it a second before
        the other camera's tracker has even finished identifying it. That is the only way back to an exited
        vehicle; anything else seen after the exit is a new visit."""
        if gtr.last_camera_id == cam_id or self._overlap_pairs is None:
            return False
        tol = self.overlap_tolerance_for(gtr.last_camera_id, cam_id)
        return tol > 0 and abs(pts - gtr.last_seen_pts) <= tol

    def is_valid_transition(self, source_cam: str, target_cam: str) -> bool:
        """True if target_cam is a valid successor to source_cam in the topology.

        For the dominant-camera use case (cam-1 is first in chain):
        cam-1 never searches for matches (no candidates from other cameras).
        cam-2 only matches from cam-1.  cam-3 only from cam-2.  etc.
        """
        if source_cam == target_cam:
            return True
        if self._topology_map is None:
            return True  # no topology specified -> all camera transitions allowed
        allowed = self._topology_map.get(source_cam, set())
        return target_cam in allowed

    def start_gid_at(self, n: int) -> None:
        """Never hand out an id below `n` (the next free id remembered from before a restart)."""
        self._next_gid = max(self._next_gid, int(n))

    # -- gate line (entrance/exit) ---------------------------------------------
    def note_gate(self, cam_id: str, local_id: int, direction: str, at: float = 0.0) -> int | None:
        """Record which way a tracklet crossed the gate line. For an exit of a tracklet that
        already has a track, closes that track and returns its global id."""
        if direction not in ("enter", "exit"):
            return None
        self._gate_hint[(cam_id, local_id)] = (direction, at)
        if len(self._gate_hint) > 5000:
            for k in list(self._gate_hint)[:2500]:
                del self._gate_hint[k]
        if direction == "enter":
            self._pending.pop((cam_id, local_id), None)   # a new vehicle: no VLM verdict needed
            return None
        for gtr in self.global_tracks.values():
            if local_id in gtr.local_track_aliases.get(cam_id, ()):
                return gtr.global_id if self.close_track(gtr.global_id) else None
        return None

    def set_direction_cameras(self, cams, wait_s: float = 25.0, require: bool = False) -> None:
        self._direction_cams = set(map(str, cams))
        self._direction_wait_s = float(wait_s)
        self._direction_require = bool(require)

    def note_untracked(self, cam_id: str, local_id: int) -> None:
        """This tracklet is not a vehicle (the VLM said so): it never starts or joins a track."""
        self._untracked.add((cam_id, local_id))
        self._pending.pop((cam_id, local_id), None)

    def note_direction_entering(self, cam_id: str, local_id: int) -> None:
        """The VLM decided this tracklet is entering: it may start a track."""
        self._dir_entering.add((cam_id, local_id))
        if len(self._dir_entering) > 5000:
            for k in list(self._dir_entering)[:2500]:
                self._dir_entering.discard(k)

    def settle_direction(self, cam_id: str, local_id: int) -> None:
        """The tracklet's direction verdict is in (or will never come): stop making it wait."""
        self._direction_settled.add((cam_id, local_id))
        if len(self._direction_settled) > 5000:
            for k in list(self._direction_settled)[:2500]:
                self._direction_settled.discard(k)

    def set_gate_cameras(self, cams: set[str]) -> None:
        self._gate_cams = set(cams)

    def gate_hint(self, cam_id: str, local_id: int) -> str | None:
        h = self._gate_hint.get((cam_id, local_id))
        return h[0] if h else None

    def gate_time(self, cam_id: str, local_id: int) -> float:
        h = self._gate_hint.get((cam_id, local_id))
        return h[1] if h else 0.0

    def close_track(self, gid: int) -> bool:
        """Mark a vehicle as having left the premises. True if it was still open."""
        gtr = self.global_tracks.get(gid)
        if gtr is None or gtr.exited:
            return False
        gtr.exited = True
        logger.info("[mtmc] EXITED: Global ID %d left the premises (last camera %s)", gid, gtr.last_camera_id)
        return True

    def get_or_create_global_id(
        self,
        cam_id: str,
        local_id: int,
        pts: float,
        embedding: np.ndarray | None = None,
        crops: list[np.ndarray] | None = None,
        views: list[np.ndarray] | None = None,
        views_ready: bool = True,
        first_pts: float | None = None,
        overlapping_gids: frozenset = frozenset(),
    ) -> int:
        """Assign or retrieve global_id for a vehicle tracklet.

        If local_id is already assigned to an ACTIVE global track in cam_id:
          - If the global track has already merged across cameras, returns its global_id.
          - If the global track is single-camera AND an embedding is provided, attempts cross-camera match.
        Otherwise, searches INACTIVE global vehicle track candidates matching topology,
        temporal disappearance rule (>= min_disappearance_s), and appearance similarity score.
        """
        if (cam_id, local_id) in self._untracked:
            return -1

        existing_gtr: GlobalVehicleTrack | None = None
        for gtr in self.global_tracks.values():
            # Check local_track_ids only — last_camera_id may have changed
            # after a cross-camera match (e.g. cam-2 matched to cam-1's track,
            # setting last_camera_id=cam-2; cam-1 must still find it).
            if local_id in gtr.local_track_aliases.get(cam_id, ()):
                if existing_gtr is not None:
                    # One tracklet (and every crop taken from it) belongs to exactly ONE global id. Two ids
                    # holding the same tracker id would split its crops and sightings: count it, shout, and
                    # keep the first so the vehicle is at least consistent from here on.
                    self.alias_conflicts += 1
                    logger.error("[mtmc] INVARIANT BROKEN: %s local_id=%d is held by Global IDs %d and %d",
                                 cam_id, local_id, existing_gtr.global_id, gtr.global_id)
                    gtr.local_track_aliases[cam_id].discard(local_id)
                    continue
                existing_gtr = gtr

        if existing_gtr is not None:
            # LOCKED: once assigned, the global_id never changes.
            existing_gtr.add_camera_observation(cam_id, local_id, pts, embedding, crops, views)
            return existing_gtr.global_id

        # NEW tracklet (no existing global track) — search for eligible candidates
        # from PREVIOUS cameras only (respecting topology + timing gate).
        #
        # Phase 1: ACTIVE tracks from valid previous cameras (overlap matching).
        #   If cam-1 is actively tracking a vehicle and cam-2 sees one simultaneously,
        #   they should share the same global ID rather than creating duplicates.
        # Phase 2: INACTIVE tracks (post-disappearance matching).
        best_candidate: GlobalVehicleTrack | None = None
        best_sim = -1.0
        best_gid = float("inf")
        best_elapsed = float("inf")
        # VLM-primary mode: topology (which camera may follow which) and the
        # timing window only decide eligibility; no re-id threshold and no
        # "closest in time wins" rule -- the VLM judges every eligible track.
        vlm_primary = self._vlm is not None and self._vlm.enabled
        time_eligible: list[tuple[GlobalVehicleTrack, float | None, float]] = []

        evaluated_count = 0
        topo_fail_count = 0
        timing_fail_count = 0
        gate_fail_count = 0
        sim_fail_count = 0

        hint = self.gate_hint(cam_id, local_id)
        if (hint is None and cam_id in self._gate_cams and first_pts is not None
                and pts - first_pts < self._gate_wait_s):
            # A gate camera: is this vehicle coming in or going out? The line will say within a
            # moment, and the answer decides whether it is a new vehicle or one to match.
            return -1
        if (hint is None and cam_id in self._direction_cams and (cam_id, local_id) not in self._direction_settled
                and first_pts is not None and pts - first_pts < self._direction_wait_s):
            return -1      # the VLM is still telling whether this vehicle is entering or leaving
        entering_new = (
            hint == "enter"
            and (self._origin_cameras is None or cam_id in self._origin_cameras)
        )
        for gtr in self.global_tracks.values():
            # Cameras that see this vehicle right now. One live on another camera but not on this one
            # (it stayed in the overlap area and left this camera's view) is still that vehicle if it
            # shows up here again -- never a new one.
            own_live = bool(gtr.local_track_aliases.get(cam_id))
            other_live = [c for c, a in gtr.local_track_aliases.items() if a and c != cam_id]
            seen_elsewhere = bool(other_live) and not own_live
            if entering_new and not seen_elsewhere:
                continue
            if gtr.exited and not self._exit_overlap_ok(gtr, cam_id, pts):
                continue   # already left: a vehicle seen again is a new visit
            src_cam = other_live[0] if seen_elsewhere else gtr.last_camera_id
            regained = (
                self._same_camera_regain_s > 0 and gtr.last_camera_id == cam_id and hint != "exit"
                and (0.0 < pts - gtr.last_seen_pts <= self._same_camera_regain_s
                     # ...or its box sits on top of this vehicle's right now: the tracker split one
                     # vehicle in two for a moment (people moving about on a pickup do it)
                     or gtr.global_id in overlapping_gids)
            )
            if regained or seen_elsewhere:
                pass   # the camera's own tracker lost this vehicle and picked it up again under a new id
            elif gtr.last_camera_id == cam_id and (gtr.state == "ACTIVE" or hint != "exit"):
                # Another vehicle in view right now on this same camera -- or, without the
                # gate saying it is leaving, a new vehicle there. Only a vehicle crossing the
                # gate line outwards may be a vehicle that came in through this same camera.
                continue

            evaluated_count += 1
            valid_topo = self.is_valid_transition(src_cam, cam_id)
            if not valid_topo:
                topo_fail_count += 1
                logger.debug(
                    "[mtmc] Candidate topology gate fail %s local_id=%d vs Global ID %d (from %s): "
                    "transition '%s' -> '%s' not allowed in topology %s",
                    cam_id, local_id, gtr.global_id, src_cam,
                    src_cam, cam_id, self._topology_map,
                )
                continue

            # Determine candidate timing & validity
            disappeared_time = gtr.disappeared_pts if gtr.disappeared_pts is not None else gtr.last_seen_pts
            elapsed = pts - disappeared_time
            min_disapp = self.get_min_disappearance_s(src_cam, cam_id)

            if gtr.state == "ACTIVE":
                overlap_elapsed = pts - gtr.last_seen_pts
                OVERLAP_TOLERANCE_S = self.overlap_tolerance_for(src_cam, cam_id)
                if -OVERLAP_TOLERANCE_S <= overlap_elapsed <= OVERLAP_TOLERANCE_S:
                    valid_time = True
                elif overlap_elapsed > OVERLAP_TOLERANCE_S:
                    # Stale active track — mark inactive at last_seen_pts and check min_disapp
                    gtr.mark_inactive(gtr.last_seen_pts)
                    disappeared_time = gtr.last_seen_pts
                    elapsed = pts - disappeared_time
                    valid_time = (elapsed >= min_disapp and elapsed <= self.max_inactive_ttl_s)
                else:
                    valid_time = False
            else:
                valid_time = (elapsed >= min_disapp and elapsed <= self.max_inactive_ttl_s)

            if not valid_time:
                timing_fail_count += 1
                logger.debug(
                    "[mtmc] Candidate timing gate fail %s local_id=%d vs Global ID %d (from %s, state=%s): "
                    "elapsed=%.1fs (min_disapp=%.1fs, max_ttl=%.1fs, pass=False)",
                    cam_id, local_id, gtr.global_id, src_cam, gtr.state,
                    elapsed, min_disapp, self.max_inactive_ttl_s,
                )
                continue

            sim = gtr.max_similarity_against(embedding) if embedding is not None else None
            if vlm_primary:
                time_eligible.append((gtr, sim, elapsed))
                continue

            if embedding is not None:
                if sim < self.reid_gate_threshold:
                    # Coarse gate: completely different vehicle (no VLM to
                    # double-check in this branch).
                    gate_fail_count += 1
                    logger.info(
                        "[mtmc] Candidate GATE reject %s local_id=%d vs Global ID %d (from %s): "
                        "sim=%.3f (gate=%.2f, reject), elapsed=%.1fs",
                        cam_id, local_id, gtr.global_id, gtr.last_camera_id,
                        sim, self.reid_gate_threshold, elapsed,
                    )
                elif sim >= self.reid_sim_threshold:
                    if sim > best_sim:
                        best_sim = sim
                        best_candidate = gtr
                        best_gid = gtr.global_id
                        best_elapsed = elapsed
                    elif sim == best_sim and gtr.global_id < best_gid:
                        best_gid = gtr.global_id
                        best_candidate = gtr
                        best_elapsed = elapsed
                else:
                    # Ambiguous (gate <= sim < match): don't gamble on a merge.
                    sim_fail_count += 1
                    logger.info(
                        "[mtmc] Candidate ambiguous %s local_id=%d vs Global ID %d (from %s): "
                        "sim=%.3f (match thr=%.2f, no merge), elapsed=%.1fs",
                        cam_id, local_id, gtr.global_id, gtr.last_camera_id,
                        sim, self.reid_sim_threshold, elapsed,
                    )
            else:
                # Topology & timing gate: select candidate closest in time (smallest elapsed time),
                # breaking ties with lowest global_id.
                if elapsed < best_elapsed:
                    best_elapsed = elapsed
                    best_gid = gtr.global_id
                    best_candidate = gtr
                elif elapsed == best_elapsed and gtr.global_id < best_gid:
                    best_gid = gtr.global_id
                    best_candidate = gtr

        if vlm_primary and not time_eligible and (cam_id, local_id) in self._pending:
            # Questions are already out for this vehicle. A candidate dropping out of the eligible set
            # meanwhile (it just left through the overlapping gate camera) does not cancel them: their
            # answers still decide, otherwise a "same" arrives for a vehicle that has been made a new id.
            vlm_result = self._resolve_vlm_arbitration(cam_id, local_id, pts, embedding, crops or [], [], views)
            if vlm_result is _WAITING:
                return -1
            self._judged.pop((cam_id, local_id), None)
            if vlm_result is not None:
                return vlm_result
        elif vlm_primary:
            if time_eligible:
                if crops and not views_ready and (cam_id, local_id) not in self._pending:
                    return -1  # still collecting views of the new vehicle; ask once they're in
                if crops:
                    # Best re-id similarity first (ranking only, never a veto),
                    # then closest in time, then lowest global id.
                    # A vehicle in view on a neighbouring camera right now (ACTIVE) goes first: it is the most
                    # plausible one by far and its stored Re-ID score says nothing about it.
                    ordered = sorted(
                        time_eligible,
                        key=lambda t: (t[0].state != "ACTIVE", -(t[1] if t[1] is not None else -1.0),
                                       t[2], t[0].global_id),
                    )
                    ranked = ordered[: self._vlm_max_candidates]
                    if len(ordered) > len(ranked) and (cam_id, local_id) not in self._pending:
                        logger.info("[mtmc] %s local_id=%d: %d eligible, asking about Global IDs %s; NOT asked "
                                    "(cap %d): %s", cam_id, local_id, len(ordered),
                                    [t[0].global_id for t in ranked], self._vlm_max_candidates,
                                    [t[0].global_id for t in ordered[len(ranked):]])
                    vlm_result = self._resolve_vlm_arbitration(
                        cam_id, local_id, pts, embedding, crops, ranked, views)
                    if vlm_result is _WAITING:
                        return -1  # verdicts not in yet -- no gid this frame, try again next frame
                    if vlm_result is not None:
                        self._judged.pop((cam_id, local_id), None)
                        return vlm_result  # VLM said "same" -> merged into that candidate's gid
                    # Nobody judged so far is this vehicle. A vehicle that only appeared while the
                    # question was out (e.g. the same car, seen by the overlapping gate camera and
                    # given its id in the meantime) has never been compared with it: ask about
                    # those too before deciding it is new.
                    judged = self._judged.get((cam_id, local_id), set())
                    fresh = [t for t in ranked if t[0].global_id not in judged]
                    if fresh:
                        again = self._resolve_vlm_arbitration(
                            cam_id, local_id, pts, embedding, crops, fresh, views)
                        if again is _WAITING:
                            return -1
                        if again is not None:
                            self._judged.pop((cam_id, local_id), None)
                            return again
                    self._judged.pop((cam_id, local_id), None)
                    # Every verdict was "different"/"uncertain", or timed out --
                    # never fall back to trusting topology/re-id alone.
                else:
                    logger.info(
                        "[mtmc] VLM ARBITER skipped (no crop yet) %s local_id=%d "
                        "(%d eligible candidates) -> no merge this frame",
                        cam_id, local_id, len(time_eligible),
                    )
        elif best_candidate is not None:
            # No VLM configured: pre-VLM behavior for a confident match. The
            # ambiguous zone never auto-merges -- that was always the
            # conservative default.
            logger.info(
                "[mtmc] MATCH SUCCESS: %s local_id=%d -> Global ID %d (from %s, sim=%.3f, mode=%s)",
                cam_id, local_id, best_candidate.global_id, best_candidate.last_camera_id,
                best_sim if embedding is not None else 1.0,
                "reid" if embedding is not None else "topology",
            )
            best_candidate.add_camera_observation(cam_id, local_id, pts, embedding, crops, views)
            return best_candidate.global_id

        if existing_gtr is not None:
            existing_gtr.add_camera_observation(cam_id, local_id, pts, embedding, crops, views)
            return existing_gtr.global_id

        # No confident merge (or the VLM withheld approval) -> create new GlobalVehicleTrack
        if evaluated_count > 0 and topo_fail_count == evaluated_count:
            cand_summary = [
                (gid2, gtr2.last_camera_id, gtr2.state,
                 round(gtr2.last_seen_pts, 2),
                 round(gtr2.disappeared_pts, 2) if gtr2.disappeared_pts is not None else None,
                 self.is_valid_transition(gtr2.last_camera_id, cam_id))
                for gid2, gtr2 in sorted(self.global_tracks.items(), key=lambda kv: kv[0])
            ]
            logger.warning(
                "[mtmc] ALL topo-fail for %s local_id=%d pts=%.2f — candidate dump: %s",
                cam_id, local_id, pts, cand_summary,
            )
        if hint == "exit":
            # Leaving through the gate but matching no vehicle inside: one we never saw
            # come in. It must not start a track.
            self._untracked.add((cam_id, local_id))
            logger.info("[mtmc] NOT TRACKED: %s local_id=%d is leaving but matched no vehicle inside",
                        cam_id, local_id)
            return -1
        if (self._direction_require and cam_id in self._direction_cams
                and (cam_id, local_id) not in self._dir_entering):
            # A track starts when a vehicle comes in. This one was not decided as entering (unclear, no answer in
            # time): we do not know it is coming in, so it must not start one.
            self._untracked.add((cam_id, local_id))
            logger.info("[mtmc] NOT TRACKED: %s local_id=%d matched no vehicle inside and its direction was not "
                        "decided as entering", cam_id, local_id)
            return -1
        if self._origin_cameras is not None and cam_id not in self._origin_cameras:
            self._untracked.add((cam_id, local_id))
            logger.info(
                "[mtmc] NOT TRACKED: %s local_id=%d matched no existing track and %s is not an "
                "origin camera (%s)", cam_id, local_id, cam_id, sorted(self._origin_cameras))
            return -1
        gid = self._next_gid
        self._next_gid += 1
        if self.on_new_gid is not None:
            try:
                self.on_new_gid(self._next_gid)
            except Exception:
                logger.warning("[mtmc] could not persist the next global id", exc_info=True)
        gtr = GlobalVehicleTrack(
            global_id=gid,
            last_camera_id=cam_id,
            last_seen_pts=pts,
        )
        gtr.add_camera_observation(cam_id, local_id, pts, embedding, crops, views)
        self.global_tracks[gid] = gtr
        logger.info(
            "[mtmc] NEW GLOBAL TRACK: %s local_id=%d -> Global ID %d "
            "(evaluated %d candidates: topo_fail=%d, timing_fail=%d, gate_fail=%d, sim_fail=%d)",
            cam_id, local_id, gid, evaluated_count, topo_fail_count, timing_fail_count,
            gate_fail_count, sim_fail_count,
        )
        return gid

    def _resolve_vlm_arbitration(
        self,
        cam_id: str,
        local_id: int,
        pts: float,
        embedding: np.ndarray | None,
        crops: list[np.ndarray],
        candidates: list[tuple[GlobalVehicleTrack, float | None, float]],
        views: list[np.ndarray] | None = None,
    ):
        """Drive the (cam_id, local_id) VLM arbitration state machine.

        ``candidates`` are (track, re-id sim or None, elapsed_s) tuples that
        already passed the topology + timing filters. One VLM job is submitted
        per candidate; once all verdicts are in (or ``max_wait_s`` passes, in
        which case the verdicts that did arrive are used) the "same" verdict
        with the highest confidence wins.

        Returns an int global_id if the VLM confirmed a match, ``_WAITING`` if
        verdicts are still outstanding (caller returns -1 this frame and tries
        again next frame), or None if there is no confirmed match (caller
        creates a new track).
        """
        key = (cam_id, local_id)
        pend = self._pending.get(key)

        if pend is None:
            probe = self._pick_views(views or crops, self._vlm_views)
            jobs = []
            for cand, sim, elapsed in candidates:
                cand_imgs, cand_caps = self._candidate_views(cand)
                if not cand_imgs:
                    continue  # candidate has no crop on file -- nothing to compare
                job_id = self._vlm.submit(probe, cand_imgs, context={
                    "captions_a": [f"the NEW sighting A on {cam_id}"
                                   + (f", view {i} of {len(probe)}, oldest first" if len(probe) > 1 else "")
                                   for i in range(1, len(probe) + 1)],
                    "captions_b": cand_caps,
                    "cam_id": cam_id, "local_id": local_id,
                    "candidate_global_id": cand.global_id,
                    "candidate_last_camera": cand.last_camera_id,
                    "mode": "vlm_primary", "sim": round(sim, 4) if sim is not None else None,
                    "elapsed_s": round(elapsed, 2), "n_candidates": len(candidates),
                    "pts": pts,
                })
                jobs.append({"job_id": job_id, "candidate": cand, "sim": sim, "elapsed": elapsed})
            if not jobs:
                return None
            self._pending[key] = {"jobs": jobs, "results": {}, "submitted_pts": pts}
            self._judged.setdefault(key, set()).update(j["candidate"].global_id for j in jobs)
            logger.info(
                "[mtmc] VLM ARBITER submitted %s local_id=%d vs Global IDs %s (pending validation)",
                cam_id, local_id, [j["candidate"].global_id for j in jobs],
            )
            return _WAITING

        # Collect whatever has arrived. Always validate against the candidates
        # actually photographed in the submitted jobs -- a later frame's
        # candidate search could differ, but a verdict only speaks to its pair.
        for job in pend["jobs"]:
            if job["job_id"] not in pend["results"]:
                res = self._vlm.take_result(job["job_id"])
                if res is not None:
                    pend["results"][job["job_id"]] = res

        if len(pend["results"]) < len(pend["jobs"]):
            waited = pts - pend["submitted_pts"]
            if waited < self._vlm.max_wait_s:
                return _WAITING
            logger.info(
                "[mtmc] VLM ARBITER timeout (%.1fs, %d/%d verdicts) %s local_id=%d",
                waited, len(pend["results"]), len(pend["jobs"]), cam_id, local_id,
            )

        del self._pending[key]
        winners = []
        for job in pend["jobs"]:
            result = pend["results"].get(job["job_id"])
            if result is None:
                continue
            decision = result.get("decision")
            confidence = result.get("confidence")
            logger.info(
                "[mtmc] VLM ARBITER verdict=%s conf=%s %s local_id=%d vs Global ID %d",
                decision, confidence, cam_id, local_id, job["candidate"].global_id,
            )
            if (decision == "same"
                    and (confidence is None or confidence >= self._vlm.min_confidence)
                    and job["candidate"].global_id in self.global_tracks
                    and (not job["candidate"].exited
                         or self._exit_overlap_ok(job["candidate"], cam_id, pts))):
                conf = confidence if confidence is not None else self._vlm.min_confidence
                winners.append((-conf, job["elapsed"], job["candidate"].global_id, job["candidate"]))

        if not winners:
            return None
        winners.sort(key=lambda w: w[:3])
        best = winners[0][3]
        if len(winners) > 1:
            logger.warning(
                "[mtmc] VLM ARBITER: %d candidates said 'same' for %s local_id=%d (gids %s) -> picking Global ID %d",
                len(winners), cam_id, local_id, [w[2] for w in winners], best.global_id,
            )
        logger.info(
            "[mtmc] VLM ARBITER MATCH: %s local_id=%d -> Global ID %d (conf=%.2f)",
            cam_id, local_id, best.global_id, -winners[0][0],
        )
        best.add_camera_observation(cam_id, local_id, pts, embedding, crops, views)
        return best.global_id

    @staticmethod
    def _pick_views(crops: list[np.ndarray], n: int) -> list[np.ndarray]:
        """Up to ``n`` crops spread over the (time-ordered) list: the largest of each n-th slice."""
        if len(crops) <= n:
            return list(crops)
        bounds = [round(i * len(crops) / n) for i in range(n + 1)]
        return [max(crops[a:b], key=lambda c: c.shape[0] * c.shape[1])
                for a, b in zip(bounds, bounds[1:])]

    def _candidate_views(self, cand: GlobalVehicleTrack) -> tuple[list[np.ndarray], list[str]]:
        """Images of the candidate, at most ``vlm_views`` in all, spread over the cameras it passed through.

        One image per camera is the point: views from different cameras show the vehicle from different
        angles, three crops from the last camera mostly repeat one another. A candidate seen on fewer
        cameras than ``vlm_views`` gets the spare images from its most recent cameras (time-spread);
        one seen on more keeps its first camera (the entrance, a clean view) and the most recent ones.
        Returns (images, captions), oldest camera first."""
        n = self._vlm_views
        cams = [c for c in dict.fromkeys(cand.camera_history) if cand.views.get(c) or cand.crops.get(c)]
        if not cams:
            cams = [c for c in list(cand.views) + list(cand.crops) if cand.views.get(c) or cand.crops.get(c)]
            cams = list(dict.fromkeys(cams))
        if len(cams) > n:
            cams = ([cams[0]] + cams[-(n - 1):]) if n > 1 else cams[-1:]
        share = {c: 1 for c in cams}
        spare = n - len(cams)
        for c in reversed(cams):      # spare images go to the most recent cameras first
            if spare <= 0:
                break
            have = len(cand.views.get(c) or cand.crops.get(c) or [])
            extra = min(spare, max(0, have - 1))
            share[c] += extra
            spare -= extra
        imgs: list[np.ndarray] = []
        caps: list[str] = []
        for cam in cams:
            src = cand.views.get(cam) or cand.crops.get(cam)
            picked = self._pick_views(src, share[cam])
            where = "most recent camera" if cam == cand.last_camera_id else "earlier camera"
            for i, im in enumerate(picked, 1):
                imgs.append(im)
                caps.append(f"the known vehicle B on {cam} ({where})"
                            + (f", view {i} of {len(picked)}" if len(picked) > 1 else ""))
        return imgs, caps

    @classmethod
    def _representative_views(cls, candidate: GlobalVehicleTrack, n: int) -> list[np.ndarray] | None:
        """Time-spread views for candidate from its last-seen camera (falls back to any camera)."""
        for source in (candidate.views, candidate.crops):
            lst = source.get(candidate.last_camera_id)
            if lst:
                return cls._pick_views(lst, n)
        for source in (candidate.views, candidate.crops):
            for lst in source.values():
                if lst:
                    return cls._pick_views(lst, n)
        return None

    @staticmethod
    def _best_crop(crops: list[np.ndarray]) -> np.ndarray:
        """Largest crop (most pixels) -- the clearest view the tracklet has had."""
        return max(crops, key=lambda c: c.shape[0] * c.shape[1])

    @classmethod
    def _representative_crop(cls, candidate: GlobalVehicleTrack) -> np.ndarray | None:
        """Largest crop on file for candidate, preferring its last-seen camera."""
        crops_by_cam = candidate.crops.get(candidate.last_camera_id)
        if crops_by_cam:
            return cls._best_crop(crops_by_cam)
        for crop_list in candidate.crops.values():
            if crop_list:
                return cls._best_crop(crop_list)
        return None

    def mark_tracklet_inactive(self, cam_id: str, local_id: int, pts: float) -> None:
        """Mark global track for (cam_id, local_id) as INACTIVE when tracklet disappears.

        Uses local_track_ids to find the track (not last_camera_id) because
        after a cross-camera overlap match, last_camera_id may have changed to
        the other camera.  e.g. cam-1 creates track, cam-2 matches via overlap,
        last_camera_id becomes cam-2; when cam-1's tracklet expires we still
        need to find and mark it INACTIVE.
        """
        # A pending VLM job for a tracklet that just aged out would otherwise
        # never be consumed (see VLMArbiter's own stale-result sweep for the
        # arbiter-side half of this cleanup). If the tracklet's own DS-level
        # visibility window (max_unmatched_frames, ~20s) is shorter than the
        # VLM's real round-trip under host contention (observed: commonly
        # 40-100s+), the verdict -- even a correct "same" -- arrives after
        # there's no live tracklet left to apply it to or re-detect on a
        # later frame, so it's discarded here with no merge ever happening.
        # Logged explicitly so this shows up in docker logs instead of just
        # silently vanishing (the only other trace is arbitration_log.jsonl
        # resolving with no corresponding MATCH/verdict line ever printed).
        orphaned = self._pending.pop((cam_id, local_id), None)
        self._judged.pop((cam_id, local_id), None)
        if orphaned is not None:
            logger.warning(
                "[mtmc] VLM ARBITER job orphaned: %s local_id=%d vs Global IDs %s "
                "-- tracklet aged out before the verdict arrived (submitted at pts=%.2f, "
                "now pts=%.2f, %.1fs ago)",
                cam_id, local_id, [j["candidate"].global_id for j in orphaned["jobs"]],
                orphaned["submitted_pts"], pts, pts - orphaned["submitted_pts"],
            )
        for gtr in self.global_tracks.values():
            if local_id in gtr.local_track_aliases.get(cam_id, ()):
                gtr.local_track_aliases[cam_id].discard(local_id)
                if any(gtr.local_track_aliases.values()):
                    # still in view of a camera (this one's other tracker id, or an overlapping camera)
                    logger.info("[mtmc] %s local_id=%d (Global ID %d) left view; still seen elsewhere",
                                cam_id, local_id, gtr.global_id)
                    break
                gtr.mark_inactive(pts)
                logger.info(
                    "[mtmc] INACTIVE TRACK: %s local_id=%d (Global ID %d) disappeared at pts=%.2f",
                    cam_id, local_id, gtr.global_id, pts,
                )
                break

    def prune_expired_tracks(self, current_pts: float) -> None:
        """Purge inactive global tracks older than max_inactive_ttl_s."""
        to_del = []
        for gid, gtr in self.global_tracks.items():
            if gtr.state == "INACTIVE":
                dis_time = gtr.disappeared_pts if gtr.disappeared_pts is not None else gtr.last_seen_pts
                if current_pts - dis_time > self.max_inactive_ttl_s:
                    to_del.append(gid)
        for gid in to_del:
            self.global_tracks.pop(gid, None)


# ---------------------------------------------------------------------------
# Legacy Union-Find association (kept for compatibility)
# ---------------------------------------------------------------------------

class _UnionFind:
    """Lightweight union-find for tracklet merging."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _time_compatible(a: TrackletRecord, b: TrackletRecord, tol: float) -> bool:
    """True if two tracklets could be the same vehicle timeline-wise."""
    overlap = min(a.last_pts, b.last_pts) - max(a.start_pts, b.start_pts)
    if overlap >= 0:
        return True
    return -overlap <= tol


def _camera_conflict(uf: _UnionFind, n: int, tracklets: list, root: int, cam: str) -> bool:
    """True if the component at root already holds a tracklet from cam."""
    return any(
        uf.find(i) == root and tracklets[i].cam_id == cam for i in range(n)
    )


def associate_cameras(
    tracklets: list[TrackletRecord],
    sim_threshold: float = 0.55,
    time_tolerance_s: float = 2.0,
    global_id_counter: int = 0,
) -> list[tuple[str, int, int]]:
    """Merge tracklets from different cameras into global identities.

    Parameters
    ----------
    tracklets : all embeddable tracklets with embeddings from all cameras
    sim_threshold : minimum cosine similarity to merge
    time_tolerance_s : max gap between tracklets to allow merge
    global_id_counter : next available global_id

    Returns
    -------
    List of (cam_id, object_id, global_id) for all tracklets.
    """
    n = len(tracklets)
    if n == 0:
        return []

    uf = _UnionFind(n)

    # Collect candidate pairs across cameras
    candidates: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tracklets[i], tracklets[j]
            if a.cam_id == b.cam_id:
                continue
            if not _time_compatible(a, b, time_tolerance_s):
                continue
            if a.embedding is None or b.embedding is None:
                continue
            sim = float(np.dot(a.embedding, b.embedding))
            if sim >= sim_threshold:
                candidates.append((sim, i, j))

    # Merge by descending similarity (greedy)
    candidates.sort(key=lambda t: t[0], reverse=True)
    for sim, i, j in candidates:
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        if _camera_conflict(uf, n, tracklets, ri, tracklets[j].cam_id):
            continue
        if _camera_conflict(uf, n, tracklets, rj, tracklets[i].cam_id):
            continue
        uf.union(i, j)

    # Group by root and assign global IDs
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    result: list[tuple[str, int, int]] = []
    gid = global_id_counter
    for members in groups.values():
        existing_gid = -1
        for i in members:
            if tracklets[i].global_id >= 0:
                existing_gid = tracklets[i].global_id
                break
        if existing_gid >= 0:
            group_gid = existing_gid
        else:
            group_gid = gid
            gid += 1
        for i in members:
            result.append((tracklets[i].cam_id, tracklets[i].object_id, group_gid))

    n_merged = sum(1 for members in groups.values() if len(members) > 1)
    if n_merged:
        logger.debug("[mtmc] association: %d tracklets → %d groups (%d merged)",
                     n, len(groups), n_merged)

    return result
