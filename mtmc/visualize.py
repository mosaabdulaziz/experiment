#!/usr/bin/env python3
"""Render side-by-side cross-camera inspection videos.

Two modes:

  --mode ours  (default)  Run the MTMC pipeline on TWO cameras (provided dets
                          or GT boxes as dets) and draw our tracked boxes,
                          colored per OUR global id. Add --gt-labels to
                          annotate every box with the ground-truth id it
                          matches, so an id miss across cameras is visible:
                          "id 12 gt 7" in cam 1 next to "id 33 gt 7" in cam 2
                          means we failed to merge that vehicle.

  --mode gt               No pipeline at all: draw the ground-truth boxes with
                          their GT ids (color per GT id) so you can verify the
                          GT itself is correct. Vehicles whose GT id appears in
                          both cameras are the ones our MTMC should merge.

Both panels are aligned by wall-clock time (each camera's start_offset_s), so
the same moment in both cameras sits side by side.

Examples:
  .venv/bin/python -m mtmc.visualize --mode ours --det-source gt --gt-labels
  .venv/bin/python -m mtmc.visualize --mode gt
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import cv2
import numpy as np

from .associate import TrackletWithEmbedding, associate
from .cameramodel import learn_camera_link_model
from .camgraph import CamGraph, load_cam_graph
from .config import OUTPUT_DIR, PipelineConfig
from .data import load_camera_infos, load_detections, load_ground_truth
from .reid import VehicleReID
from .run_pipeline import _gt_as_dets, _load_embedding_cache, _save_embedding_cache
from .tracker import iou_matrix, track_detections

PANEL_W, PANEL_H = 960, 540   # each camera downscaled 2x (1080p -> 540p)
HEADER_H = 70
OUT_FPS = 10                  # matches the videos' native 10 fps
GT_IOU_MATCH = 0.4            # min IoU to claim a box matches a GT box


def _id_color(gid: int) -> tuple[int, int, int]:
    """Stable, well-spread BGR color per id (golden-ratio hue walk)."""
    hue = ((gid * 0.61803398875) % 1.0) * 179.0
    hsv = np.uint8([[[hue, 255, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="S01")
    p.add_argument("--cams", nargs="+", default=["c001", "c002"],
                   help="exactly two camera ids (default: c001 c002)")
    p.add_argument("--mode", choices=["ours", "gt"], default="ours",
                   help="'ours' = our pipeline boxes+ids (default); 'gt' = pure GT boxes+GT ids")
    p.add_argument("--det", default="det_mask_rcnn.txt")
    p.add_argument("--det-source", choices=["det", "gt"], default="det",
                   help="ours mode only: 'det' = challenge detector output (noisy);\n"
                        "'gt' = GT boxes as detections, ids still from our association")
    p.add_argument("--gt-labels", action="store_true",
                   help="ours mode: annotate every box with the GT id it matches\n"
                        "(reveals id misses between cameras)")
    p.add_argument("--min-frames", type=int, default=3,
                   help="ours mode: keep tracklets with at least this many frames\n"
                        "(visualization default is lenient so few vehicles vanish)")
    p.add_argument("--min-dur", type=float, default=0.2,
                   help="ours mode: min tracklet wall-clock duration (s)")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.35)
    p.add_argument("--sim", type=float, default=0.55, help="Re-ID merge cosine threshold")
    p.add_argument("--camera-model", action="store_true",
                   help="opt-in: use the learned pairwise homographies for\n"
                        "association (off by default; measured unreliable on\n"
                        "opposite-side camera pairs)")
    p.add_argument("--cam-graph", default=None, metavar="YAML",
                   help="path to camera graph YAML with per-edge association gates")
    p.add_argument("--out", default=None, help="output mp4 path (default mtmc/output/…mp4)")
    return p


# ---------------------------------------------------------------------------
# GT-id matching (ours mode, --gt-labels)
# ---------------------------------------------------------------------------
def _match_gt_id(tracklet, gt: dict[int, np.ndarray]) -> tuple[int | None, float]:
    """Modal GT id overlapping this tracklet's boxes (with matched fraction)."""
    votes: list[int] = []
    for f, box in zip(tracklet.frames, np.asarray(tracklet.boxes, dtype=float)):
        rows = gt.get(f)
        if rows is None or len(rows) == 0:
            continue
        cur = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]])
        gtxy = rows[:, 1:5].astype(float).copy()
        gtxy[:, 2] += gtxy[:, 0]
        gtxy[:, 3] += gtxy[:, 1]
        ious = iou_matrix(cur[None, :], gtxy)[0]
        j = int(ious.argmax())
        if ious[j] >= GT_IOU_MATCH:
            votes.append(int(rows[j, 0]))
    if not votes:
        return None, 0.0
    gid, cnt = Counter(votes).most_common(1)[0]
    return gid, cnt / len(votes)


def _gt_id_label(tracklet, gt: dict[int, np.ndarray] | None) -> str | None:
    if gt is None:
        return None
    gid, _ = _match_gt_id(tracklet, gt)
    return None if gid is None else f"gt {gid}"


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _read_frame(caps, last_local, last_img, cam: str, local: int) -> np.ndarray:
    if local != last_local[cam]:
        cap = caps[cam]
        cap.set(cv2.CAP_PROP_POS_FRAMES, local - 1)
        ok, img = cap.read()
        last_local[cam] = local
        last_img[cam] = img if ok else np.zeros((1080, 1920, 3), np.uint8)
    return last_img[cam]


def render_video(
    infos,
    cfg: PipelineConfig,
    boxes_by_cam: dict[str, dict[int, list[tuple[int, np.ndarray, bool, str | None]]]],
    out_mp4: Path,
    title: str,
    cross_ids: set[int],
    color_by_gt: bool = False,
) -> None:
    """boxes_by_cam[cam][frame] = [(id, xywh, is_cross, extra_label), ...]."""
    caps = {cam: cv2.VideoCapture(str(infos[cam].video_path)) for cam in infos}
    last_local = {cam: -1 for cam in infos}
    last_img = {cam: None for cam in infos}

    win_start = max(infos[c].start_offset_s for c in infos)
    win_end = min(infos[c].start_offset_s + (infos[c].frames - 1) / cfg.fps for c in infos)
    n_frames = max(int((win_end - win_start) * OUT_FPS), 1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, OUT_FPS, (PANEL_W * 2, PANEL_H + HEADER_H))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {out_mp4}")

    t0 = time.time()
    for i in range(n_frames):
        t_wall = win_start + i / OUT_FPS
        canvas = np.zeros((PANEL_H + HEADER_H, PANEL_W * 2, 3), np.uint8)

        # header
        cv2.putText(canvas, f"{title}", (16, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"t = {t_wall:6.2f}s", (PANEL_W - 70, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cam_line = "   ".join(f"{c} (off {infos[c].start_offset_s:+.1f}s)" for c in infos)
        cv2.putText(canvas, cam_line, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)
        if cross_ids:
            # legend: colored swatch + id, wrapped into two rows of 14
            items = [f"id {g}" for g in sorted(cross_ids)]
            legend = "cross-cam: " if not color_by_gt else "in both cams: "
            x, y = 16, 66
            cv2.putText(canvas, legend, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, cv2.LINE_AA)
            x += cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            for g in sorted(cross_ids):
                txt = f"id {g} "
                col = _id_color(int(g) if not color_by_gt else int(g))
                cv2.rectangle(canvas, (x, y - 12), (x + 12, y), col, -1)
                cv2.putText(canvas, txt, (x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
                x += cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 16
                if x > PANEL_W * 2 - 180:
                    x, y = 16, 66
        else:
            cv2.putText(canvas, "no cross-camera ids", (16, 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)

        for col, cam in enumerate(infos):
            off = infos[cam].start_offset_s
            local = int(round((t_wall - off) * cfg.fps)) + 1
            local = int(min(max(local, 1), infos[cam].frames))
            panel = cv2.resize(_read_frame(caps, last_local, last_img, cam, local),
                               (PANEL_W, PANEL_H))
            for gid, box, is_cross, extra in boxes_by_cam[cam].get(local, []):
                x, y, w, h = box.astype(int)
                x0, y0 = x // 2, y // 2
                x1, y1 = (x + w) // 2, (y + h) // 2
                if is_cross:
                    color = _id_color(gid)
                    cv2.rectangle(panel, (x0, y0), (x1, y1), color, 4)
                    label = f"id {gid}" + (f" {extra}" if extra else "")
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    lx, ly = max(x0, 0), max(y0 - 4, 0)
                    cv2.rectangle(panel, (lx, ly - th - 8), (lx + tw + 8, ly), color, -1)
                    cv2.putText(panel, label, (lx + 4, ly - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(panel, (x0, y0), (x1, y1), (190, 190, 190), 2)
                    label = f"id {gid}" + (f" {extra}" if extra else "")
                    cv2.putText(panel, label, (max(x0, 0), max(y0 - 6, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
            canvas[HEADER_H:, col * PANEL_W:(col + 1) * PANEL_W] = panel
        writer.write(canvas)
        if i % 250 == 0:
            print(f"[visualize] rendered {i}/{n_frames} frames ({round(time.time() - t0, 1)}s)")

    for cam in caps:
        caps[cam].release()
    writer.release()
    print(f"[visualize] done: {out_mp4} ({n_frames} frames, {round(time.time() - t0, 1)}s)")


# ---------------------------------------------------------------------------
# mode: gt — pure ground truth, no pipeline
# ---------------------------------------------------------------------------
def run_gt_mode(args) -> int:
    cfg = PipelineConfig(scene=args.scene, cameras=args.cams)
    infos = load_camera_infos(cfg.scene, cfg.cameras, cfg.det_file)
    boxes_by_cam: dict[str, dict[int, list[tuple[int, np.ndarray, bool, None]]]] = {}
    gt_by_cam = {}
    for cam, info in infos.items():
        gt = load_ground_truth(info.gt_path)
        gt_by_cam[cam] = gt
        boxes_by_cam[cam] = {}
        for f, rows in gt.items():
            for row in rows:
                gid, x, y, w, h = row
                boxes_by_cam[cam].setdefault(int(f), []).append(
                    (int(gid), np.array([x, y, w, h], dtype=float), True, None))
    # ids present in BOTH cameras
    ids_a = {int(r[0]) for rows in gt_by_cam[args.cams[0]].values() for r in rows}
    ids_b = {int(r[0]) for rows in gt_by_cam[args.cams[1]].values() for r in rows}
    both = ids_a & ids_b
    print(f"[visualize] GT ids: cam1={len(ids_a)} cam2={len(ids_b)} shared={len(both)}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = Path(args.out or (out_dir / f"gt_side_by_side_{args.scene}_{args.cams[0]}_{args.cams[1]}.mp4"))
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    render_video(infos, cfg, boxes_by_cam, out_mp4,
                 f"GROUND TRUTH — {args.scene} {args.cams[0]}+{args.cams[1]} (ids are GT)",
                 both, color_by_gt=True)
    return 0


# ---------------------------------------------------------------------------
# mode: ours — full pipeline, our ids
# ---------------------------------------------------------------------------
def run_ours_mode(args) -> int:
    if len(args.cams) != 2:
        print("[visualize] --cams needs exactly two ids, got", args.cams)
        return 2
    cfg = PipelineConfig(
        scene=args.scene,
        cameras=args.cams,
        det_file=args.det,
        det_conf_threshold=args.conf,
        iou_threshold=args.iou,
        reid_sim_threshold=args.sim,
        min_tracklet_frames=args.min_frames,
        min_tracklet_duration_s=args.min_dur,
    )
    cam_a, cam_b = args.cams
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = Path(args.out or (out_dir / f"ours_side_by_side_{args.scene}_{cam_a}_{cam_b}.mp4"))
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    infos = load_camera_infos(cfg.scene, cfg.cameras, cfg.det_file)
    if args.det_source == "gt":
        dets_by_cam = {cam: _gt_as_dets(infos[cam].gt_path) for cam, i in infos.items()}
    else:
        dets_by_cam = {cam: load_detections(i.det_path, cfg.det_conf_threshold) for cam, i in infos.items()}
    print(f"[visualize] boxes from: {'GT (gt.txt, clean)' if args.det_source == 'gt' else 'detector (det_mask_rcnn.txt, noisy)'} — ids from our MTMC association")
    tracklets_by_cam: dict[str, dict[int, object]] = {}
    for cam, info in infos.items():
        tls = track_detections(cam, dets_by_cam[cam], cfg, max_frame=info.frames)
        tracklets_by_cam[cam] = {t.track_id: t for t in tls}
        print(f"[visualize] {cam}: {len(tls)} tracklets ({round(time.time() - t0, 1)}s)")

    reid = VehicleReID(cfg)
    n_tracklets = sum(len(t) for t in tracklets_by_cam.values())
    cache_path = out_dir / (
        f"embeddings_{cfg.scene}_vis_{'_'.join(cfg.cameras)}_{Path(cfg.det_file).stem}"
        f"_{args.det_source}_conf{cfg.det_conf_threshold:.2f}_iou{cfg.iou_threshold:.2f}"
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
        print(f"[visualize] embedded {n_tracklets} tracklets ({round(time.time() - t0, 1)}s)")

    gts = {cam: load_ground_truth(infos[cam].gt_path) for cam in cfg.cameras}
    camera_model = None
    if args.camera_model:
        camera_model = learn_camera_link_model(gts, infos, cfg.fps)
        camera_model.windows = {}

    cam_graph: CamGraph | None = None
    if args.cam_graph:
        cli_overrides = {}
        if args.sim != 0.85:
            cli_overrides["reid_sim"] = args.sim
        cam_graph = load_cam_graph(args.cam_graph, cfg.cameras, cli_overrides or None)
        print(f"[visualize] {cam_graph.summary()}")

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
    cross_ids = {gid for gid, m in global_ids.items() if len({c for c, _ in m}) >= 2}

    gt_label_of: dict[tuple[str, int], str | None] = {}
    if args.gt_labels:
        for cam, info in infos.items():
            gt = gts[cam]
            for tid, t in tracklets_by_cam[cam].items():
                gt_label_of[(cam, tid)] = _gt_id_label(t, gt)

    boxes_by_cam: dict[str, dict[int, list[tuple[int, np.ndarray, bool, str | None]]]] = {}
    for cam in cfg.cameras:
        boxes_by_cam[cam] = {}
    for (cam, tid), gid in gid_of.items():
        t = tl_by_key[(cam, tid)]
        is_cross = gid in cross_ids
        extra = gt_label_of.get((cam, tid))
        for f, box in zip(t.frames, np.asarray(t.boxes, dtype=float)):
            boxes_by_cam[cam].setdefault(f, []).append((gid, box, is_cross, extra))
    n_cross = len(cross_ids)
    print(f"[visualize] {len(global_ids)} global ids, {n_cross} span both cameras "
          f"({round(time.time() - t0, 1)}s)")

    title = f"OURS — {args.scene} {cam_a}+{cam_b} (dets={args.det_source}, ids ours)" + \
            (" + GT labels" if args.gt_labels else "") + \
            (" + camera model" if args.camera_model else "")
    render_video(infos, cfg, boxes_by_cam, out_mp4, title, cross_ids)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "gt":
        return run_gt_mode(args)
    return run_ours_mode(args)


if __name__ == "__main__":
    sys.exit(main())
