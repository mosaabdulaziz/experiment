"""Path and parameter configuration for the MTMC pipeline.

Defaults target the AICity22 Track1 dataset layout and the project venv.
Override via environment variables (MTMC_*).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolvable paths (override with env vars; fall back to known layouts)
# ---------------------------------------------------------------------------
DATASET_ROOT = Path(
    os.environ.get(
        "AICITY22_ROOT",
        "/home/mmtayee/URCA/IRIS/datasets/AICity22_Track1_MTMC_Tracking",
    )
)
FASTREID_ROOT = Path(
    os.environ.get(
        "FASTREID_ROOT",
        "/home/mmtayee/URCA/IRIS/helper-repos/multi-camera-tracking/fast-reid",
    )
)
# Default ReID checkpoint: VeRi-trained SBS(R50-ibn), Rank@1 97% on VeRi.
REID_CHECKPOINT = Path(
    os.environ.get(
        "REID_CHECKPOINT",
        "/home/mmtayee/URCA/IRIS/helper-repos/multi-camera-tracking/fast-reid/checkpoints/veri_sbs_R50-ibn.pth",
    )
)
REID_CONFIG = "configs/VeRi/sbs_R50-ibn.yml"  # relative to FASTREID_ROOT
EVAL_DIR = DATASET_ROOT / "eval"
# ROIs are white so the official eval's ROI-outlier filter is a no-op — we
# deliberately do NOT apply the dataset's road masks (parked cars etc. stay).
ROI_DIR = Path(os.environ.get("MTMC_ROI_DIR", DATASET_ROOT / "ROIs_white"))
OUTPUT_DIR = Path(os.environ.get("MTMC_OUTPUT_DIR", "mtmc/output"))

# ---- Fine-tuned plate detector (primary vehicle detector) ----------------
PLATE_DETECTOR_DIR = Path(os.environ.get(
    "MTMC_PLATE_DETECTOR",
    "products/el-mohandes/models/finetuned",
))
PLATE_DETECTOR_ONNX = PLATE_DETECTOR_DIR / "best.onnx"
PLATE_DETECTOR_PT = PLATE_DETECTOR_DIR / "best.pt"


@dataclass
class PipelineConfig:
    # ---- which scene / cameras -------------------------------------------
    scene: str = "S01"
    cameras: list[str] = field(default_factory=lambda: ["c001", "c002", "c003", "c004", "c005"])
    det_file: str = "det_mask_rcnn.txt"  # provided challenge detections
    det_conf_threshold: float = 0.5      # drop low-confidence dets
    fps: float = 10.0                    # all AIC22 videos are 10 fps (S03/c015 is 8)

    # ---- single-camera tracking (SCT) ------------------------------------
    iou_threshold: float = 0.35          # frame-to-frame association IoU
    max_unmatched_frames: int = 10       # tracklet survives this many empty frames
    min_tracklet_frames: int = 8         # drop shorter tracklets (too noisy for Re-ID)
    min_tracklet_duration_s: float = 0.7  # alternative floor in wall time

    # ---- same-camera stitching -------------------------------------------
    # Measured on S01: stitching over-merges (two similar vehicles merged into
    # one) and HURT IDF1 on both det-fed (0.37 -> 0.36) and GT-fed (0.85 -> 0.78)
    # runs. Kept for experimentation; off by default.
    stitch: bool = False                 # merge fragments of one vehicle per camera
    stitch_sim_threshold: float = 0.65   # same-cam embeddings are reliable; high bar
    stitch_max_gap_frames: int = 25      # fragments within this gap may be one vehicle
    stitch_max_overlap_frames: int = 5   # tiny overlap allowed (occlusion boundary)
    stitch_min_overlap_iou: float = 0.5  # at shared frames, must look like one object

    # ---- Re-ID ------------------------------------------------------------
    reid_crop_stride: int = 4            # embed every Nth frame of a tracklet
    reid_max_crops: int = 32             # cap crops per tracklet
    reid_batch_size: int = 64
    reid_cache_frames: int = 48          # bounded LRU of decoded video frames
    #                                       (each 1080p frame is ~6 MB; keeping
    #                                       all frames per camera was ~12 GB and
    #                                       crashed the host — see reid.py)
    reid_sim_threshold: float = 0.55     # Re-ID merge bar (cosine). Tuned for the
    #                                       appearance-only default on S01:
    #                                       GT-fed 0.61 / det-fed 0.32 at 0.55
    #                                       (0.45-0.55 all ~0.60 GT-fed; higher
    #                                       drops fast since true-pair sims
    #                                       average only ~0.54).

    # ---- cross-camera association -----------------------------------------
    time_overlap_tolerance_s: float = 1.5  # allow tracklets that are within this
    #                                        many seconds of each other to merge
    merge_mode: str = "greedy"           # 'greedy' | 'hierarchical'
    allow_same_camera_merge: bool = False  # never merge tracklets from the same cam
    use_camera_model: bool = False       # gate pairs with learned pairwise
    #                                       homographies (spatial consistency).
    #                                       OFF by default: measured on c001-c002
    #                                       the homography is unreliable (cameras
    #                                       on opposite sides of the square see a
    #                                       car front-on vs back-on, so box
    #                                       bottom-centers are different ground
    #                                       points) — 27/28 "spatial proof" merges
    #                                       joined different vehicles. Wall-time
    #                                       overlap + Re-ID similarity is the
    #                                       robust signal for near-synchronous
    #                                       cameras. Opt-in per pair where the
    #                                       homography is verified tight.
    spatial_gate_px: float = 400.0       # max projected bottom-center distance (px).
    #                                       Tuned on S01: generous enough to keep
    #                                       true cross-cam pairs (tracklet boxes
    #                                       are noisy), tight enough to prune most
    #                                       false candidates (~90% of cross-cam pairs).
    spatial_merge_px: float = 120.0      # pairs below this projected distance AND
    #                                       time-overlapping are merged as a spatial
    #                                       proof (same ground point at same wall
    #                                       time = same vehicle), no appearance needed.
    use_transition_windows: bool = False  # per-pair mid->mid transit windows.
    #                                       Learned from full GT tracks; OFF by
    #                                       default because SCT tracklets are
    #                                       fragments whose midpoints misalign.
    #                                       Opt-in for sequential (non-overlap)
    #                                       transit filtering.

    # ---- evaluation ---------------------------------------------------------
    dstype: str = "train"
    prediction_file: str = "prediction.txt"  # written to OUTPUT_DIR
