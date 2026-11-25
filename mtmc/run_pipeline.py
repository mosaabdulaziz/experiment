#!/usr/bin/env python3
"""Run the full MTMC pipeline on an AICity22 scene and report IDF1.

Pipeline stages:
  1. Load provided detections + GT + camera timing.
  2. Single-camera tracking (SCT) -> per-camera tracklets.
  3. Re-ID each tracklet (fast-reid VeRi SBS R50-ibn) -> mean embedding.
  4. Cross-camera association (appearance + time gate) -> global IDs.
  5. Write predictions, run the official eval, print IDF1/IDP/IDR.

MLflow (spike Q9, mtmc/tracking/eval.py log_pipeline_eval): each run is logged as
its own run in the experiment ``mtmc-pipeline-eval`` (it scores several models
together, so it belongs to none of their runs): IDF1/IDP/IDR as
``eval/aicity22-<scene>/*``, the settings as params, report.json, and a tag set
per model it used (Re-ID checkpoint, plate detector with ``--det-source plate``):
sha256 and, from the model's sidecar or ``--manifest`` (a deployment's
models.manifest JSON), its registered name/version and run. AICity22 is outside
DVC, so the set is tagged ``unversioned`` with the sha256 of the GT file it read.
``--no-track`` records nothing.

Example:
  .venv/bin/python -m mtmc.run_pipeline            # scene S01, 5 cams
  .venv/bin/python -m mtmc.run_pipeline --scene S01 --det det_mask_rcnn.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from .associate import TrackletWithEmbedding, associate
from .cameramodel import CameraLinkModel, learn_camera_link_model
from .camgraph import CamGraph, load_cam_graph
from .config import EVAL_DIR, OUTPUT_DIR, REID_CHECKPOINT, PipelineConfig
from .data import load_camera_infos, load_detections, load_ground_truth
from .evaluate import (
    ensure_white_rois,
    prepare_scene_gt,
    run_eval,
    write_prediction_file,
)
from .reid import VehicleReID
from .tracker import track_detections
from .tracking.eval import log_pipeline_eval
from .yolo_detector import DEFAULT_PLATE_DETECTOR, PlateVehicleDetector


def _plate_dets_from_video(video_path: Path, model_path: str | None,
                           conf: float, imgsz: int, device: str | None,
                           skip: int = 1,
                           generic_dets: dict[int, np.ndarray] | None = None) -> dict[int, np.ndarray]:
    """Run plate detector on a video file and return merged detections.

    Plate-detected vehicles are the primary vehicle tracker.
    Generic detections (from pre-computed files) provide non-vehicle classes
    and act as fallback for vehicles without visible plates.

    Merge strategy per frame:
      - Plate boxes (expanded to vehicle size) are always included
      - Generic boxes that overlap significantly with plate boxes are suppressed
      - Non-overlapping generic boxes (pedestrians, cyclists, etc.) are kept

    Returns {frame: Nx4 [x,y,w,h]} in MOTChallenge format.
    """
    det = PlateVehicleDetector(
        model_path=model_path or "products/el-mohandes/models/finetuned/best.pt",
        conf_threshold=conf, imgsz=imgsz, device=device,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path}, skipping")
        return {}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    has_generic = generic_dets is not None and len(generic_dets) > 0
    print(f"  [plate-detect] {video_path.name}: {w}x{h}, {total} frames (every {skip}th)"
          f"{' + generic dets' if has_generic else ''}")

    dets: dict[int, np.ndarray] = {}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        frame_num = frame_idx + 1  # 1-indexed
        _, vehicle_boxes, confs = det.detect_frame(frame)

        # Plate detections → xywh
        plate_xywh = []
        if len(confs) > 0:
            xywh = vehicle_boxes.copy()
            xywh[:, 2] = vehicle_boxes[:, 2] - vehicle_boxes[:, 0]
            xywh[:, 3] = vehicle_boxes[:, 3] - vehicle_boxes[:, 1]
            plate_xywh = list(xywh)

        # Generic detections for this frame
        gen_xywh = generic_dets.get(frame_num, np.zeros((0, 4))) if has_generic else np.zeros((0, 4))

        if len(plate_xywh) > 0 and len(gen_xywh) > 0:
            # Suppress generic boxes that overlap with plate boxes
            plate_arr = np.array(plate_xywh)  # Nx4 xywh
            gen_arr = np.array(gen_xywh)      # Mx4 xywh
            # Convert to xyxy for IoU
            p_xyxy = np.c_[plate_arr[:, 0], plate_arr[:, 1],
                           plate_arr[:, 0] + plate_arr[:, 2],
                           plate_arr[:, 1] + plate_arr[:, 3]]
            g_xyxy = np.c_[gen_arr[:, 0], gen_arr[:, 1],
                           gen_arr[:, 0] + gen_arr[:, 2],
                           gen_arr[:, 1] + gen_arr[:, 3]]
            # IoU between each generic box and each plate box
            ix1 = np.maximum(g_xyxy[:, None, 0], p_xyxy[None, :, 0])
            iy1 = np.maximum(g_xyxy[:, None, 1], p_xyxy[None, :, 1])
            ix2 = np.minimum(g_xyxy[:, None, 2], p_xyxy[None, :, 2])
            iy2 = np.minimum(g_xyxy[:, None, 3], p_xyxy[None, :, 3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            g_area = gen_arr[:, 2] * gen_arr[:, 3]
            p_area = plate_arr[:, 2] * plate_arr[:, 3]
            union = g_area[:, None] + p_area[None, :] - inter
            iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            # Keep generic boxes that don't significantly overlap with any plate
            max_iou_per_gen = iou.max(axis=1) if len(iou) > 0 else np.array([])
            keep_mask = max_iou_per_gen < 0.3  # suppress if >30% overlap with a plate
            kept_generic = gen_arr[keep_mask]
            # Merge: plate boxes + kept generic boxes
            all_boxes = np.vstack([plate_arr, kept_generic]) if len(kept_generic) > 0 else plate_arr
        elif len(plate_xywh) > 0:
            all_boxes = np.array(plate_xywh)
        elif len(gen_xywh) > 0:
            all_boxes = gen_arr
        else:
            frame_idx += 1
            continue

        dets[frame_num] = all_boxes
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"    frame {frame_idx}/{total} ({len(dets)} frames with dets)")

    cap.release()
    total_dets = sum(len(v) for v in dets.values())
    print(f"  [plate-detect] done: {total_dets} detections across {len(dets)} frames")
    return dets


def _gt_as_dets(gt_path: Path) -> dict[int, np.ndarray]:
    """Convert a GT file (frame,id,x,y,w,h,...) to the det format used by the
    tracker ({frame: Nx4 [x,y,w,h]}), treating GT boxes as perfect detections.

    Only one box per GT object per frame (GT already is), and conf is not
    needed since we skip the confidence filter for this source.
    """
    out: dict[int, list] = {}
    for line in gt_path.read_text().splitlines():
        p = line.replace(",", " ").split()
        if len(p) < 6:
            continue
        frame = int(float(p[0]))
        x, y, w, h = (float(p[i]) for i in (2, 3, 4, 5))
        if w <= 0 or h <= 0:
            continue
        out.setdefault(frame, []).append([x, y, w, h])
    return {f: np.asarray(v, dtype=float).reshape(-1, 4) for f, v in out.items()}


def _load_embedding_cache(path: Path, expected: int) -> dict | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    keys = data["keys"]
    if len(keys) != expected:
        return None
    return {tuple(k): data["embs"][i] for i, k in enumerate(keys)}


def _save_embedding_cache(path: Path, embeddings: dict) -> None:
    keys = np.array([list(k) for k in embeddings], dtype=object)
    embs = np.stack(list(embeddings.values()))
    np.savez_compressed(path, keys=keys, embs=embs)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="S01")
    p.add_argument("--cams", nargs="+", default=None,
                   help="camera ids, e.g. c001 c002 (default: scene's 5 cams)")
    p.add_argument("--det", default="det_mask_rcnn.txt")
    p.add_argument("--conf", type=float, default=0.5, help="detection confidence threshold")
    p.add_argument("--det-source", choices=["det", "gt", "plate"], default="det",
                   help="'det' = challenge detections (normal); 'gt' = feed the GT\n"
                        "boxes through tracking+association as perfect detections;\n"
                        "'plate' = run fine-tuned plate detector on video files\n"
                        "(primary detector for CCTV deployment)")
    p.add_argument("--plate-model", default=None,
                   help="Path to plate detector model (default: products/el-mohandes/models/finetuned/best.pt)")
    p.add_argument("--plate-conf", type=float, default=0.25,
                   help="Plate detection confidence threshold")
    p.add_argument("--plate-imgsz", type=int, default=1280,
                   help="Plate detector input size")
    p.add_argument("--plate-device", default=None,
                   help="Plate detector device (cuda:0, cpu, etc.)")
    p.add_argument("--iou", type=float, default=0.35, help="SCT association IoU")
    p.add_argument("--sim", type=float, default=0.55, help="Re-ID merge cosine threshold")
    p.add_argument("--stitch-sim", type=float, default=0.65,
                   help="same-camera stitch embedding threshold")
    p.add_argument("--stitch", action="store_true",
                   help="enable same-camera fragment stitching (off by default;\n"
                        "measured to hurt S01)")
    p.add_argument("--time-tol", type=float, default=1.5, help="cross-cam time tolerance (s)")
    p.add_argument("--camera-model", action="store_true",
                   help="opt-in: gate pairs with learned pairwise homographies.\n"
                        "OFF by default (time + Re-ID similarity is the robust\n"
                        "signal for near-synchronous cameras; the homography\n"
                        "was measured to merge wrong vehicles on c001-c002).")
    p.add_argument("--spatial-px", type=float, default=400.0,
                   help="max projected bottom-center distance for spatial gate (px)")
    p.add_argument("--spatial-merge", type=float, default=120.0,
                   help="time-overlapping pairs below this projected distance merge\n"
                        "as a spatial proof (no appearance needed)")
    p.add_argument("--transition-windows", action="store_true",
                   help="also apply per-pair mid->mid transition windows (opt-in)")
    p.add_argument("--cam-graph", default=None, metavar="YAML",
                   help="path to camera graph YAML with per-edge association gates.\n"
                        "See mtmc/examples/cam_graph.yaml for schema and examples.")
    p.add_argument("--skip-reid", action="store_true", help="skip embedding stage (embedding=1s)")
    p.add_argument("--out", default=None, help="output dir (default mtmc/output)")
    p.add_argument("--eval-set", default=None,
                   help="MLflow evaluation set name (default aicity22-<scene>)")
    p.add_argument("--manifest", default=None, metavar="JSON",
                   help="a deployment's model manifest (products/<p>/models.manifest*.json): tags the models used "
                        "with its registered names/versions")
    p.add_argument("--no-track", action="store_true", help="record nothing in MLflow")
    return p


def models_used(args) -> dict[str, str]:
    """The model files this run loaded, by role (Q9): what its score depends on besides the settings."""
    used = {}
    if not args.skip_reid:
        used["reid"] = str(REID_CHECKPOINT)
    if args.det_source == "plate":
        used["plate_detector"] = str(args.plate_model or DEFAULT_PLATE_DETECTOR)
    return used


def track_run(args, report: dict, report_path: Path) -> None:
    """Log this pipeline evaluation to MLflow (mtmc-pipeline-eval). Never fails the run."""
    if args.no_track:
        print("[tracking] off (--no-track)")
        return
    params = {k: v for k, v in report.items() if k not in ("metrics", "prediction", "tracklets", "global_ids")}
    params.update({"det_source": args.det_source, "skip_reid": args.skip_reid, "stitch": args.stitch,
                   "stitch_sim": args.stitch_sim, "spatial_merge": args.spatial_merge})
    try:
        log_pipeline_eval(
            eval_set=args.eval_set or f"aicity22-{report['scene']}",
            gt_path=EVAL_DIR / "ground_truth_train.txt",
            metrics={**report["metrics"], "tracklets": report["tracklets"], "global_ids": report["global_ids"]},
            params=params, models=models_used(args), manifest=args.manifest, artifacts=[report_path],
            run_name=f"{report['scene']}-{args.det_source}", script="mtmc.run_pipeline",
            extra_tags={"detections": report["det"] if args.det_source == "det" else args.det_source})
    except SystemExit as exc:                     # a refused tracking step (bad set name): the score is already out
        print(f"[tracking] not recorded: {exc}", file=sys.stderr)


def main() -> None:
    args = build_parser().parse_args()
    cfg = PipelineConfig(
        scene=args.scene,
        cameras=args.cams or ["c001", "c002", "c003", "c004", "c005"],
        det_file=args.det,
        det_conf_threshold=args.conf,
        iou_threshold=args.iou,
        reid_sim_threshold=args.sim,
        time_overlap_tolerance_s=args.time_tol,
        use_camera_model=args.camera_model,
        spatial_gate_px=args.spatial_px,
        spatial_merge_px=args.spatial_merge,
        use_transition_windows=args.transition_windows,
        stitch=args.stitch or PipelineConfig().stitch,
        stitch_sim_threshold=args.stitch_sim,
    )
    out_dir = Path(args.out or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- 1. load data -----------------------------------------------------
    infos = load_camera_infos(cfg.scene, cfg.cameras, cfg.det_file)
    print(f"[mtmc] scene={cfg.scene} cams={len(infos)} dets={cfg.det_file} "
          f"det_source={args.det_source}")
    if args.det_source == "gt":
        # Feed GT boxes through the pipeline as perfect detections.
        dets_by_cam = {cam: _gt_as_dets(infos[cam].gt_path) for cam, i in infos.items()}
    elif args.det_source == "plate":
        # Run plate detector on each camera's video, merged with generic dets
        print(f"[mtmc] Running plate detector (model={args.plate_model or 'default'}, "
              f"conf={args.plate_conf}, imgsz={args.plate_imgsz})")
        dets_by_cam = {}
        for cam, info in infos.items():
            # Load generic detections as fallback (non-vehicle classes + backup)
            generic = load_detections(info.det_path, cfg.det_conf_threshold)
            dets_by_cam[cam] = _plate_dets_from_video(
                info.video_path, args.plate_model, args.plate_conf,
                args.plate_imgsz, args.plate_device,
                generic_dets=generic,
            )
    else:
        dets_by_cam = {cam: load_detections(i.det_path, cfg.det_conf_threshold) for cam, i in infos.items()}

    # ---- 2. single-camera tracking ----------------------------------------
    tracklets_by_cam: dict[str, dict[int, object]] = {}
    for cam, info in infos.items():
        tls = track_detections(cam, dets_by_cam[cam], cfg, max_frame=info.frames)
        tracklets_by_cam[cam] = {t.track_id: t for t in tls}
        print(f"[mtmc]   {cam}: {len(tls)} tracklets from {len(dets_by_cam[cam])} det-frames")

    # ---- 3. Re-ID ----------------------------------------------------------
    reid = VehicleReID(cfg)
    n_tracklets = sum(len(t) for t in tracklets_by_cam.values())
    cache_path = out_dir / (
        f"embeddings_{cfg.scene}_{args.det_source}_{Path(cfg.det_file).stem}"
        f"_conf{cfg.det_conf_threshold:.2f}_iou{cfg.iou_threshold:.2f}"
        f"_minf{cfg.min_tracklet_frames}.npz"
    )
    embeddings = _load_embedding_cache(cache_path, n_tracklets)
    if embeddings is None:
        embeddings = {}
        for cam, info in infos.items():
            video = cv2.VideoCapture(str(info.video_path))
            frame_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
            for local_id, t in tracklets_by_cam[cam].items():
                emb = None if args.skip_reid else reid.embed_tracklet(t, video, frame_cache)
                if emb is None:
                    emb = np.ones(1)  # placeholder so association still runs without Re-ID
                embeddings[(cam, local_id)] = emb
            video.release()
        _save_embedding_cache(cache_path, embeddings)

    # ---- 4. same-camera stitching (fragments -> one tracklet per vehicle) --
    if cfg.stitch:
        from .stitch import stitch_camera

        for cam in cfg.cameras:
            tl_list = list(tracklets_by_cam[cam].values())
            cam_embs = {t.track_id: embeddings[(cam, t.track_id)] for t in tl_list}
            stitched, _ = stitch_camera(cam, tl_list, cam_embs, cfg)
            tracklets_by_cam[cam] = {t.track_id: t for t in stitched}
        n_st = sum(len(v) for v in tracklets_by_cam.values())
        print(f"[mtmc] stitching: {n_tracklets} -> {n_st} tracklets "
              f"({round(time.time() - t0, 1)}s)")

    # Stitched tracklets keep the base fragment's id, so the cached embedding
    # of that base fragment is reused as-is. That is a fine proxy: fragments of
    # one vehicle have near-identical appearance (same camera, same lighting),
    # so the base fragment's mean embedding represents the merged vehicle well.

    all_tracklets: list[TrackletWithEmbedding] = []
    for cam, info in infos.items():
        for local_id, t in tracklets_by_cam[cam].items():
            emb = embeddings.get((cam, local_id))
            if emb is None:
                emb = np.ones(1)
            times = info.start_offset_s + (np.asarray(t.frames, dtype=float) - 1) / cfg.fps
            all_tracklets.append(
                TrackletWithEmbedding(
                    cam_id=cam,
                    track_id=local_id,
                    start_s=info.start_offset_s + (t.start_frame - 1) / cfg.fps,
                    end_s=info.start_offset_s + (t.end_frame - 1) / cfg.fps,
                    embedding=emb,
                    frames=t.frames,
                    times_s=times,
                    boxes=np.asarray(t.boxes, dtype=float),
                )
            )
    n_emb = sum(1 for t in all_tracklets if t.embedding.shape[0] > 1)
    print(f"[mtmc] embedded {n_emb}/{len(all_tracklets)} tracklets "
          f"({round(time.time() - t0, 1)}s)")

    # ---- 4. association ------------------------------------------------------
    camera_model: CameraLinkModel | None = None
    if cfg.use_camera_model:
        gts = {cam: load_ground_truth(infos[cam].gt_path) for cam in cfg.cameras}
        camera_model = learn_camera_link_model(gts, infos, cfg.fps)
        if not cfg.use_transition_windows:
            camera_model.windows = {}
        n_h = len(camera_model.homographies)
        n_w = len(camera_model.windows)
        print(f"[mtmc] camera-link model: {n_h} homographies, {n_w} transition windows")
    # ---- load camera graph (per-edge config) ----------------------------
    cam_graph: CamGraph | None = None
    if args.cam_graph:
        cli_overrides = {}
        if args.sim != 0.55:  # non-default sim -> override all edges
            cli_overrides["reid_sim"] = args.sim
        if args.time_tol != 1.5:
            cli_overrides["time_tolerance_s"] = args.time_tol
        cam_graph = load_cam_graph(args.cam_graph, cfg.cameras, cli_overrides or None)
        print(f"[mtmc] {cam_graph.summary()}")
    else:
        print(f"[mtmc] no --cam-graph: using global sim={cfg.reid_sim_threshold:.2f} "
              f"tol={cfg.time_overlap_tolerance_s:.1f}s for all pairs")

    global_ids = associate(all_tracklets, cfg, camera_model, cam_graph)
    n_merged = sum(1 for g, m in global_ids.items() if len(m) > 1)
    print(f"[mtmc] {len(global_ids)} global ids ({n_merged} merged across cameras)")

    # ---- 5. evaluate ---------------------------------------------------------
    ensure_white_rois(cfg.cameras, cfg)
    gt_path = prepare_scene_gt(cfg.scene, cfg.cameras, cfg, out_dir)
    pred_path = write_prediction_file(global_ids, tracklets_by_cam, infos, cfg,
                                      out_dir / cfg.prediction_file)
    metrics = run_eval(gt_path, pred_path, cfg)
    print(f"[mtmc] IDF1={metrics.get('idf1', float('nan')):.2f} "
          f"IDP={metrics.get('idp', float('nan')):.2f} "
          f"IDR={metrics.get('idr', float('nan')):.2f} "
          f"({round(time.time() - t0, 1)}s total)")

    report = {
        "scene": cfg.scene,
        "cams": cfg.cameras,
        "det": cfg.det_file,
        "conf": cfg.det_conf_threshold,
        "iou": cfg.iou_threshold,
        "sim": cfg.reid_sim_threshold,
        "time_tol": cfg.time_overlap_tolerance_s,
        "camera_model": cfg.use_camera_model,
        "transition_windows": cfg.use_transition_windows,
        "spatial_px": cfg.spatial_gate_px,
        "cam_graph": args.cam_graph,
        "tracklets": len(all_tracklets),
        "global_ids": len(global_ids),
        "metrics": metrics,
        "prediction": str(pred_path),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[mtmc] report: {out_dir / 'report.json'}")
    track_run(args, report, out_dir / "report.json")


if __name__ == "__main__":
    sys.exit(main())
