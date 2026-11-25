"""Sequential stateful multi-camera vehicle tracker.

Implements the camera-topology-aware matching pipeline:
  Camera 1 → Camera 2 → Camera 3
  with minimum travel-time constraints and appearance-based matching.

Key design:
  - Vehicles are processed sequentially through the camera chain.
  - A vehicle from Camera N can only match in Camera N+1 after
    min_travel_time_s has elapsed since its appearance in Camera N.
  - Once a vehicle matches in the next camera, it is "consumed" —
    it won't match additional vehicles in that camera.
  - Each global track retains per-camera appearance templates.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .feature_extractor import AppearanceExtractor
from .yolo_detector import Tracklet, VehicleDetector


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """Configuration for one camera in the chain."""
    cam_id: str
    video_path: str
    fps: float | None = None           # auto-detected from video if None
    time_offset_s: float = 0.0         # wall-clock offset (0 = first camera)


@dataclass
class GlobalTrack:
    """A vehicle tracked across one or more cameras."""
    global_id: int
    camera_ids: list[str] = field(default_factory=list)
    local_ids: dict[str, int] = field(default_factory=dict)   # cam_id → local track_id
    # Per-camera appearance templates
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)  # cam_id → (D,) L2-norm
    # Per-camera timing
    start_times: dict[str, float] = field(default_factory=dict)  # cam_id → start_s
    end_times: dict[str, float] = field(default_factory=dict)    # cam_id → end_s
    # Per-camera bounding boxes (mean)
    mean_boxes: dict[str, np.ndarray] = field(default_factory=dict)  # cam_id → [x1,y1,x2,y2]
    # Track state
    is_complete: bool = False   # True after matching through the full chain

    @property
    def last_camera(self) -> str:
        return self.camera_ids[-1] if self.camera_ids else ""

    @property
    def num_cameras(self) -> int:
        return len(self.camera_ids)

    def combined_embedding(self) -> np.ndarray | None:
        """Mean of all per-camera embeddings."""
        if not self.embeddings:
            return None
        embs = list(self.embeddings.values())
        mean = np.mean(embs, axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 0 else None


@dataclass
class MatchDecision:
    """Record of a cross-camera match attempt (for debug output)."""
    source_cam: str
    source_local_id: int
    source_start_s: float
    target_cam: str
    target_local_id: int
    target_start_s: float
    time_diff_s: float           # target_start - source_end
    min_travel_time_s: float
    time_qualified: bool
    similarity: float
    sim_threshold: float
    matched: bool
    reason: str                  # human-readable explanation


# ---------------------------------------------------------------------------
# Sequential Tracker
# ---------------------------------------------------------------------------

class SequentialTracker:
    """Stateful tracker that processes cameras in chain order."""

    def __init__(
        self,
        camera_chain: list[CameraConfig],
        detector: VehicleDetector,
        extractor: AppearanceExtractor,
        min_travel_time_s: float = 20.0,
        sim_threshold: float = 0.60,
        max_templates_per_vehicle: int = 5,
        frame_skip: int = 1,
        output_dir: str = "mtmc/output/sequential",
    ):
        self.camera_chain = camera_chain
        self.detector = detector
        self.extractor = extractor
        self.min_travel_time_s = min_travel_time_s
        self.sim_threshold = sim_threshold
        self.max_templates = max_templates_per_vehicle
        self.frame_skip = frame_skip
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.global_tracks: list[GlobalTrack] = []
        self.unmatched_tracks: list[GlobalTrack] = []  # tracks waiting for next camera
        self.all_decisions: list[MatchDecision] = []
        self._next_gid = 0
        self._per_camera_tracklets: dict[str, list[Tracklet]] = {}

    def _new_global_id(self) -> int:
        gid = self._next_gid
        self._next_gid += 1
        return gid

    def _create_global_tracks_from_camera(
        self, cam_id: str, tracklets: list[Tracklet], embeddings: dict[int, np.ndarray],
        fps: float, time_offset_s: float,
    ) -> list[GlobalTrack]:
        """Initialize a new set of global tracks from a camera's tracklets."""
        tracks = []
        for t in tracklets:
            emb = embeddings.get(t.track_id)
            if emb is None:
                continue

            start_s = time_offset_s + t.start_frame / fps
            end_s = time_offset_s + t.end_frame / fps
            mean_box = t.mean_box()

            gt = GlobalTrack(
                global_id=self._new_global_id(),
                camera_ids=[cam_id],
                local_ids={cam_id: t.track_id},
                embeddings={cam_id: emb},
                start_times={cam_id: start_s},
                end_times={cam_id: end_s},
                mean_boxes={cam_id: mean_box},
            )
            tracks.append(gt)

        return tracks

    def _match_camera(
        self, source_tracks: list[GlobalTrack], target_tracklets: list[Tracklet],
        target_embeddings: dict[int, np.ndarray], target_cam: str, target_fps: float,
        target_offset: float,
    ) -> tuple[list[GlobalTrack], list[MatchDecision]]:
        """Match source global tracks against target camera's tracklets.

        Returns:
            updated_tracks: source tracks that were extended + new unmatched target tracks
            decisions: all match decisions (for debug)
        """
        decisions: list[MatchDecision] = []
        matched_source: set[int] = set()   # indices into source_tracks
        matched_target: set[int] = set()   # indices into target_tracklets

        # For each target tracklet, find best matching source track
        target_best: dict[int, tuple[float, int]] = {}  # target_idx → (sim, source_idx)

        for ti, tt in enumerate(target_tracklets):
            t_emb = target_embeddings.get(tt.track_id)
            if t_emb is None:
                continue

            t_start_s = target_offset + tt.start_frame / target_fps
            t_end_s = target_offset + tt.end_frame / target_fps

            best_sim = -1.0
            best_si = -1

            for si, st in enumerate(source_tracks):
                if si in matched_source:
                    continue
                if target_cam in st.camera_ids:
                    continue  # already seen this camera

                # Get source end time (when it left the previous camera)
                prev_cam = st.last_camera
                src_end_s = st.end_times[prev_cam]
                src_emb = st.embeddings[prev_cam]

                # Travel time check
                time_diff = t_start_s - src_end_s
                time_qualified = time_diff >= self.min_travel_time_s

                if not time_qualified:
                    decisions.append(MatchDecision(
                        source_cam=prev_cam,
                        source_local_id=st.local_ids[prev_cam],
                        source_start_s=st.start_times[prev_cam],
                        target_cam=target_cam,
                        target_local_id=tt.track_id,
                        target_start_s=t_start_s,
                        time_diff_s=time_diff,
                        min_travel_time_s=self.min_travel_time_s,
                        time_qualified=False,
                        similarity=0.0,
                        sim_threshold=self.sim_threshold,
                        matched=False,
                        reason=f"TIME GATE: {time_diff:.1f}s < {self.min_travel_time_s:.1f}s minimum",
                    ))
                    continue

                # Compute similarity (also try combined embedding if multi-camera)
                combined = st.combined_embedding()
                if combined is not None:
                    sim_combined = float(np.dot(combined, t_emb))
                else:
                    sim_combined = 0.0

                sim_prev = float(np.dot(src_emb, t_emb))
                sim = max(sim_prev, sim_combined)

                if sim > best_sim:
                    best_sim = sim
                    best_si = si

            if best_si >= 0 and best_sim >= self.sim_threshold:
                target_best[ti] = (best_sim, best_si)

        # Resolve conflicts: if multiple targets want the same source, keep best
        source_to_target: dict[int, tuple[float, int]] = {}
        for ti, (sim, si) in target_best.items():
            if si not in source_to_target or sim > source_to_target[si][0]:
                source_to_target[si] = (sim, ti)

        # Apply matches
        for si, (sim, ti) in source_to_target.items():
            st = source_tracks[si]
            tt = target_tracklets[ti]
            t_emb = target_embeddings[tt.track_id]
            t_start_s = target_offset + tt.start_frame / target_fps
            t_end_s = target_offset + tt.end_frame / target_fps

            prev_cam = st.last_camera

            decisions.append(MatchDecision(
                source_cam=prev_cam,
                source_local_id=st.local_ids[prev_cam],
                source_start_s=st.start_times[prev_cam],
                target_cam=target_cam,
                target_local_id=tt.track_id,
                target_start_s=t_start_s,
                time_diff_s=t_start_s - st.end_times[prev_cam],
                min_travel_time_s=self.min_travel_time_s,
                time_qualified=True,
                similarity=sim,
                sim_threshold=self.sim_threshold,
                matched=True,
                reason=f"MATCH: sim={sim:.3f} >= {self.sim_threshold:.2f}, "
                       f"travel={t_start_s - st.end_times[prev_cam]:.1f}s",
            ))

            # Extend the global track
            st.camera_ids.append(target_cam)
            st.local_ids[target_cam] = tt.track_id
            st.start_times[target_cam] = t_start_s
            st.end_times[target_cam] = t_end_s
            st.mean_boxes[target_cam] = tt.mean_box()

            # Add template (cap at max_templates)
            if len(st.embeddings) < self.max_templates:
                st.embeddings[target_cam] = t_emb

            matched_source.add(si)
            matched_target.add(ti)

        # Build result: matched source tracks + new unmatched target tracks
        updated = [source_tracks[si] for si in matched_source]

        # Unmatched target tracks become new global tracks (start of a new chain)
        for ti in range(len(target_tracklets)):
            if ti not in matched_target:
                tt = target_tracklets[ti]
                t_emb = target_embeddings.get(tt.track_id)
                if t_emb is None:
                    continue
                t_start_s = target_offset + tt.start_frame / target_fps
                t_end_s = target_offset + tt.end_frame / target_fps

                new_gt = GlobalTrack(
                    global_id=self._new_global_id(),
                    camera_ids=[target_cam],
                    local_ids={target_cam: tt.track_id},
                    embeddings={target_cam: t_emb},
                    start_times={target_cam: t_start_s},
                    end_times={target_cam: t_end_s},
                    mean_boxes={target_cam: tt.mean_box()},
                )
                updated.append(new_gt)

                decisions.append(MatchDecision(
                    source_cam="",
                    source_local_id=-1,
                    source_start_s=0,
                    target_cam=target_cam,
                    target_local_id=tt.track_id,
                    target_start_s=t_start_s,
                    time_diff_s=0,
                    min_travel_time_s=self.min_travel_time_s,
                    time_qualified=False,
                    similarity=0,
                    sim_threshold=self.sim_threshold,
                    matched=False,
                    reason="NEW: no source candidate (first camera or no match)",
                ))

        return updated, decisions

    def run(self) -> list[GlobalTrack]:
        """Execute the full sequential pipeline."""
        print(f"\n{'='*70}")
        print(f"SEQUENTIAL MULTI-CAMERA TRACKING")
        print(f"  Camera chain: {' → '.join(c.cam_id for c in self.camera_chain)}")
        print(f"  Min travel time: {self.min_travel_time_s}s")
        print(f"  Sim threshold: {self.sim_threshold:.2f}")
        print(f"{'='*70}\n")

        all_tracklets: dict[str, list[Tracklet]] = {}
        all_embeddings: dict[str, dict[int, np.ndarray]] = {}

        # ---- Stage 1: Detect + embed each camera sequentially ----
        for ci, cam in enumerate(self.camera_chain):
            print(f"[Stage {ci+1}/{len(self.camera_chain)}] Processing {cam.cam_id}")

            # Detect + track
            tracklets = self.detector.track_video(cam.video_path, skip=self.frame_skip)
            all_tracklets[cam.cam_id] = tracklets
            print(f"  [detect]   {len(tracklets)} tracklets")

            # Detect FPS if not set
            if cam.fps is None:
                cap = cv2.VideoCapture(cam.video_path)
                cam.fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                print(f"  [detect]   FPS auto-detected: {cam.fps:.2f}")

            # Extract embeddings
            embeddings: dict[int, np.ndarray] = {}
            cap = cv2.VideoCapture(cam.video_path)
            frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()
            for t in tracklets:
                emb = self.extractor.embed_tracklet(t, cap, frame_cache=frame_cache)
                if emb is not None:
                    embeddings[t.track_id] = emb
            cap.release()
            print(f"  [embed]    {len(embeddings)}/{len(tracklets)} embedded "
                  f"(dim={self.extractor.embed_dim})")

            # Create global tracks from this camera
            new_tracks = self._create_global_tracks_from_camera(
                cam.cam_id, tracklets, embeddings, cam.fps, cam.time_offset_s
            )
            self.unmatched_tracks = new_tracks
            self._per_camera_tracklets[cam.cam_id] = tracklets
            all_embeddings[cam.cam_id] = embeddings

            print(f"  [init]     {len(new_tracks)} global tracks initialized")

            # ---- Stage 2: Match with previous camera ----
            if ci > 0:
                prev_cam = self.camera_chain[ci - 1]
                source_tracks = [gt for gt in self.global_tracks
                                 if prev_cam.cam_id in gt.camera_ids
                                 and not gt.is_complete]

                if source_tracks:
                    print(f"\n  [match]    Matching {len(source_tracks)} tracks from "
                          f"{prev_cam.cam_id} → {cam.cam_id}")
                    print(f"  [match]    Min travel time: {self.min_travel_time_s}s")

                    updated, decisions = self._match_camera(
                        source_tracks, tracklets, embeddings,
                        cam.cam_id, cam.fps, cam.time_offset_s,
                    )
                    self.all_decisions.extend(decisions)

                    n_matched = sum(1 for d in decisions if d.matched)
                    n_time_gated = sum(1 for d in decisions if not d.time_qualified
                                       and d.source_cam)
                    n_sim_rejected = sum(1 for d in decisions
                                         if d.time_qualified and not d.matched
                                         and d.similarity > 0)
                    print(f"  [match]    Results: {n_matched} matched, "
                          f"{n_time_gated} time-gated, "
                          f"{n_sim_rejected} similarity-rejected")

                    # Update global tracks: replace source tracks with updated versions
                    old_ids = {id(s) for s in source_tracks}
                    self.global_tracks = [
                        gt for gt in self.global_tracks if id(gt) not in old_ids
                    ]
                    self.global_tracks.extend(updated)

                    # Mark complete tracks (reached end of chain)
                    if ci == len(self.camera_chain) - 1:
                        for gt in updated:
                            if gt.num_cameras == len(self.camera_chain):
                                gt.is_complete = True
                else:
                    print(f"  [match]    No source tracks from {prev_cam.cam_id} to match")

            # For camera 1, add tracks to global_tracks for first time
            if ci == 0:
                self.global_tracks.extend(new_tracks)

        # Final: mark all tracks that reached the end of chain
        for gt in self.global_tracks:
            if gt.num_cameras == len(self.camera_chain):
                gt.is_complete = True

        return self.global_tracks

    def write_debug_log(self) -> Path:
        """Write detailed match decisions to a log file."""
        log_path = self.output_dir / "match_decisions.log"
        with open(log_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("CROSS-CAMERA MATCH DECISIONS\n")
            f.write(f"  Camera chain: {' → '.join(c.cam_id for c in self.camera_chain)}\n")
            f.write(f"  Min travel time: {self.min_travel_time_s}s\n")
            f.write(f"  Sim threshold: {self.sim_threshold:.2f}\n")
            f.write("=" * 80 + "\n\n")

            for d in self.all_decisions:
                status = "✅ MATCHED" if d.matched else "❌ REJECTED"
                f.write(f"[{status}]\n")
                f.write(f"  Source: {d.source_cam or 'NONE'} track {d.source_local_id} "
                        f"(start {d.source_start_s:.1f}s)\n")
                f.write(f"  Target: {d.target_cam} track {d.target_local_id} "
                        f"(start {d.target_start_s:.1f}s)\n")
                f.write(f"  Travel time: {d.time_diff_s:.1f}s "
                        f"({'✓' if d.time_qualified else '✗'} >= {d.min_travel_time_s:.0f}s)\n")
                if d.similarity > 0:
                    f.write(f"  Similarity: {d.similarity:.4f} "
                            f"({'✓' if d.similarity >= d.sim_threshold else '✗'} "
                            f">= {d.sim_threshold:.2f})\n")
                f.write(f"  Reason: {d.reason}\n\n")

        print(f"\n  [debug] Match decisions written to: {log_path}")
        return log_path

    def write_summary(self) -> dict:
        """Write summary statistics."""
        n_total = len(self.global_tracks)
        n_complete = sum(1 for gt in self.global_tracks if gt.is_complete)
        n_1cam = sum(1 for gt in self.global_tracks if gt.num_cameras == 1)
        n_2cam = sum(1 for gt in self.global_tracks if gt.num_cameras == 2)
        n_3cam = sum(1 for gt in self.global_tracks if gt.num_cameras == 3)

        summary = {
            "total_global_tracks": n_total,
            "complete_3cam": n_3cam,
            "partial_2cam": n_2cam,
            "single_cam": n_1cam,
            "min_travel_time_s": self.min_travel_time_s,
            "sim_threshold": self.sim_threshold,
            "cameras": [c.cam_id for c in self.camera_chain],
        }

        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"  Total global tracks:  {n_total}")
        print(f"  Complete (1→2→3):     {n_3cam}")
        print(f"  Partial (2 cameras):  {n_2cam}")
        print(f"  Single camera only:   {n_1cam}")
        print(f"{'='*70}")

        # Detailed per-track info
        summary_path = self.output_dir / "summary.json"
        import json
        track_details = []
        for gt in self.global_tracks:
            detail = {
                "global_id": gt.global_id,
                "cameras": gt.camera_ids,
                "is_complete": gt.is_complete,
                "num_cameras": gt.num_cameras,
                "start_times": {k: round(v, 1) for k, v in gt.start_times.items()},
                "end_times": {k: round(v, 1) for k, v in gt.end_times.items()},
                "local_ids": gt.local_ids,
                "mean_boxes": {k: v.tolist() for k, v in gt.mean_boxes.items()},
            }
            track_details.append(detail)
        summary["tracks"] = track_details
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"  Summary written to: {summary_path}")

        return summary
