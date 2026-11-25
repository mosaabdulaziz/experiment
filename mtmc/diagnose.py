#!/usr/bin/env python3
"""Diagnose cross-camera association failures for a camera pair.

With GT boxes as detections there are no detection errors, so every GT vehicle
present in BOTH cameras is a merge our association *should* make. For each such
vehicle this script reports why it was or wasn't merged:

  - wall-time overlap between the two cameras' tracklets
  - homography-projected distance (the Tier-1 spatial signal)
  - Re-ID cosine similarity (the Tier-2 appearance signal)
  - the specific gate/threshold that rejected the pair

Run with the same settings as the visualizer so the numbers match the video:

  .venv/bin/python -m mtmc.diagnose --scene S01 --cams c001 c002
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import cv2
import numpy as np

from .associate import TrackletWithEmbedding, _to_obs, associate
from .cameramodel import learn_camera_link_model
from .camgraph import CamGraph, load_cam_graph
from .config import OUTPUT_DIR, PipelineConfig
from .data import load_camera_infos, load_ground_truth
from .reid import VehicleReID
from .run_pipeline import _gt_as_dets, _load_embedding_cache, _save_embedding_cache
from .tracker import track_detections
from .visualize import _match_gt_id

MATCH_FRAC = 0.5  # tracklet must match a GT id on >= this fraction of its frames


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="S01")
    p.add_argument("--cams", nargs="+", default=["c001", "c002"])
    p.add_argument("--min-frames", type=int, default=3)
    p.add_argument("--min-dur", type=float, default=0.2)
    p.add_argument("--sim", type=float, default=0.55, help="Re-ID merge cosine threshold")
    p.add_argument("--spatial-merge", type=float, default=120.0)
    p.add_argument("--spatial-gate", type=float, default=400.0)
    p.add_argument("--time-tol", type=float, default=1.5)
    p.add_argument("--camera-model", action="store_true",
                   help="opt-in: associate with the learned homographies (off by default)")
    p.add_argument("--cam-graph", default=None, metavar="YAML",
                   help="path to camera graph YAML with per-edge association gates")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if len(args.cams) != 2:
        print("need exactly two cameras")
        return 2
    cam_a, cam_b = args.cams
    cfg = PipelineConfig(
        scene=args.scene,
        cameras=args.cams,
        det_conf_threshold=0.5,
        iou_threshold=0.35,
        reid_sim_threshold=args.sim,
        spatial_merge_px=args.spatial_merge,
        spatial_gate_px=args.spatial_gate,
        time_overlap_tolerance_s=args.time_tol,
        min_tracklet_frames=args.min_frames,
        min_tracklet_duration_s=args.min_dur,
    )
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- pipeline (GT boxes as detections, same as the visualizer) ----------
    infos = load_camera_infos(cfg.scene, cfg.cameras, cfg.det_file)
    dets_by_cam = {cam: _gt_as_dets(infos[cam].gt_path) for cam, i in infos.items()}
    tracklets_by_cam: dict[str, dict[int, object]] = {}
    for cam, info in infos.items():
        tls = track_detections(cam, dets_by_cam[cam], cfg, max_frame=info.frames)
        tracklets_by_cam[cam] = {t.track_id: t for t in tls}

    reid = VehicleReID(cfg)
    n_tracklets = sum(len(t) for t in tracklets_by_cam.values())
    cache_path = out_dir / (
        f"embeddings_{cfg.scene}_vis_{'_'.join(cfg.cameras)}_{Path(cfg.det_file).stem}"
        f"_gt_conf{cfg.det_conf_threshold:.2f}_iou{cfg.iou_threshold:.2f}"
        f"_minf{cfg.min_tracklet_frames}.npz"
    )
    embeddings = _load_embedding_cache(cache_path, n_tracklets)
    if embeddings is None:
        embeddings = {}
        for cam, info in infos.items():
            video = cv2.VideoCapture(str(info.video_path))
            frame_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
            for local_id, t in tracklets_by_cam[cam].items():
                emb = reid.embed_tracklet(t, video, frame_cache)
                if emb is None:
                    emb = np.ones(1)
                embeddings[(cam, local_id)] = emb
            video.release()
        _save_embedding_cache(cache_path, embeddings)

    gts = {cam: load_ground_truth(infos[cam].gt_path) for cam in cfg.cameras}
    camera_model = None
    if args.camera_model:
        camera_model = learn_camera_link_model(gts, infos, cfg.fps)
        camera_model.windows = {}

    cam_graph: CamGraph | None = None
    if args.cam_graph:
        cam_graph = load_cam_graph(args.cam_graph, cfg.cameras)
        print(f"[diagnose] {cam_graph.summary()}")

    all_tracklets: list[TrackletWithEmbedding] = []
    for cam, info in infos.items():
        for local_id, t in tracklets_by_cam[cam].items():
            times = info.start_offset_s + (np.asarray(t.frames, dtype=float) - 1) / cfg.fps
            all_tracklets.append(
                TrackletWithEmbedding(
                    cam_id=cam,
                    track_id=local_id,
                    start_s=info.start_offset_s + (t.start_frame - 1) / cfg.fps,
                    end_s=info.start_offset_s + (t.end_frame - 1) / cfg.fps,
                    embedding=embeddings[(cam, local_id)],
                    frames=t.frames,
                    times_s=times,
                    boxes=np.asarray(t.boxes, dtype=float),
                )
            )
    global_ids = associate(all_tracklets, cfg, camera_model, cam_graph)

    gid_of = {}
    for gid, members in global_ids.items():
        for cam, tid in members:
            gid_of[(cam, tid)] = gid
    tl_by_key = {(t.cam_id, t.track_id): t for t in all_tracklets}
    obs_by_key = {(t.cam_id, t.track_id): _to_obs(t) for t in all_tracklets}

    # our tracklets -> GT id (modal IoU match)
    gt_of: dict[tuple[str, int], tuple[int | None, float]] = {}
    for cam in cfg.cameras:
        gt = gts[cam]
        for tid, t in tracklets_by_cam[cam].items():
            gt_of[(cam, tid)] = _match_gt_id(t, gt)

    ids_a = {int(r[0]) for rows in gts[cam_a].values() for r in rows}
    ids_b = {int(r[0]) for rows in gts[cam_b].values() for r in rows}
    shared = sorted(ids_a & ids_b)

    # ---- per-vehicle verdict --------------------------------------------------
    stats: Counter = Counter()
    print(f"\nGT vehicles in BOTH {cam_a}+{cam_b}: {len(shared)} (tracklets in "
          f"{cam_a}={len(tracklets_by_cam[cam_a])}, {cam_b}={len(tracklets_by_cam[cam_b])})\n")
    print(f"{'gt':>4} {'ourA':>5} {'ourB':>5} {'overlap_s':>9} {'proj_px':>8} "
          f"{'sim':>6}  verdict")
    print("-" * 70)

    def cands_for(cam: str, gid: int):
        return [k for k, (g, frac) in gt_of.items()
                if k[0] == cam and g == gid and frac >= MATCH_FRAC]

    def verdict_for(ka, kb):
        """Explain why THIS pair did or didn't merge (replicates associate)."""
        a, b = tl_by_key[ka], tl_by_key[kb]
        oa, ob = obs_by_key[ka], obs_by_key[kb]
        sim = float(np.dot(a.embedding, b.embedding))
        overlap = min(a.end_s, b.end_s) - max(a.start_s, b.start_s)
        d = camera_model.spatial_distance(oa, ob) if camera_model else None
        temporal = camera_model.temporal_gate(oa, ob, cfg.time_overlap_tolerance_s) if camera_model else True
        merged = gid_of[ka] == gid_of[kb]
        if merged:
            return f"MERGED (our id {gid_of[ka]})", merged, sim, overlap, d
        if not temporal:
            return "rejected: NO TIME OVERLAP", merged, sim, overlap, d
        if d is not None and d <= cfg.spatial_merge_px:
            return ("spatial-proof WOULD merge (dist %.0f) but blocked: camera "
                    "conflict (one already merged to another tracklet)" % d, merged, sim, overlap, d)
        if d is not None and d > cfg.spatial_gate_px:
            return (f"rejected: SPATIAL GATE (proj {d:.0f}px > {cfg.spatial_gate_px:.0f}px)",
                    merged, sim, overlap, d)
        return (f"rejected: LOW SIMILARITY ({sim:.3f} < {cfg.reid_sim_threshold})",
                merged, sim, overlap, d)

    for gid in shared:
        ours_a, ours_b = cands_for(cam_a, gid), cands_for(cam_b, gid)
        if not ours_a or not ours_b:
            miss = cam_a if not ours_a else cam_b
            stats["tracklet missing (tracking dropped it)"] += 1
            print(f"{gid:>4} {'-':>5} {'-':>5} {'-':>9} {'-':>8} {'-':>6}  "
                  f"NO TRACKLET in {miss} — tracker never produced one")
            continue

        # did ANY fragment pair get merged under one global id?
        merged_any = any(gid_of[a] == gid_of[b] for a in ours_a for b in ours_b)
        if merged_any:
            stats["merged"] += 1
            gid_ab = next(gid_of[a] for a in ours_a for b in ours_b if gid_of[a] == gid_of[b])
            # report the merged pair
            for a in ours_a:
                for b in ours_b:
                    if gid_of[a] == gid_of[b]:
                        v, _, sim, overlap, d = verdict_for(a, b)
                        d_s = "n/a" if d is None else f"{d:5.0f}"
                        print(f"{gid:>4} {gid_of[a]:>5} {gid_of[b]:>5} {overlap:>9.1f} {d_s:>8} "
                              f"{sim:>6.3f}  {v}")
                        break
                else:
                    continue
                break
            continue

        # not merged: pick the closest pair (highest sim) and explain it
        best = max(((verdict_for(a, b), a, b) for a in ours_a for b in ours_b), key=lambda t: t[0][2])
        (v, _, sim, overlap, d), ka, kb = best
        d_s = "n/a" if d is None else f"{d:5.0f}"
        n_a, n_b = len(ours_a), len(ours_b)
        frag = f"  [{n_a} frag in {cam_a}, {n_b} in {cam_b}]" if (n_a > 1 or n_b > 1) else ""
        print(f"{gid:>4} {gid_of[ka]:>5} {gid_of[kb]:>5} {overlap:>9.1f} {d_s:>8} "
              f"{sim:>6.3f}  {v}{frag}")

    # ---- summary ---------------------------------------------------------------
    print("\n=== SUMMARY ===")
    total = len(shared)
    for k, v in stats.most_common():
        print(f"  {v:>3}/{total}  {k}")
    merged = stats["merged"]
    print(f"\nmerged: {merged}/{total}  ({100 * merged / total:.0f}%)  "
          f"({round(time.time() - t0, 1)}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
