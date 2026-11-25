"""Demo video & keyframe visualizer for `products/concealment_detection`.

Selects a zero-miss-detection clip from the calibrated RetailS / PoseLift dataset (or accepts
a custom `--pose-json` and optional `--rgb-video`), runs it through the unmodified production
`ConcealmentMonitor` and `evidence.build_candidate_record`, and renders:
1. An annotated 1920x1080 RGB `.mp4` video showing bounding boxes, 17 COCO pose keypoints,
   calibrated shelf-front zones, wrist reach trajectory, live feature gauges, and the system's
   step-by-step state-machine decision banner.
2. A 6-panel stage-progression keyframe contact sheet (`.png`) for quick inspection.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys
import numpy as np

# Rendering is optional; offline calibration does not require OpenCV.
cv2 = None

PRODUCT_DIR = Path(__file__).resolve().parents[1]
if str(PRODUCT_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCT_DIR))

from adapters import extract_mask_events, load_retails_pose_json  # noqa: E402
from calibration.dataset_utils import parse_camera_id  # noqa: E402
from concealment.features import extract_track_features  # noqa: E402
from concealment.state_machine import ConcealmentMonitor  # noqa: E402
from concealment.types import (  # noqa: E402
    CameraZones,
    CandidateEvent,
    DetectorSettings,
    L_ANKLE,
    L_EAR,
    L_ELBOW,
    L_EYE,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    QualityStatus,
    R_ANKLE,
    R_EAR,
    R_ELBOW,
    R_EYE,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
    TrackFeatures,
    TrackObservation,
)
from config import ServiceConfig, load_config  # noqa: E402
from evidence import build_candidate_record  # noqa: E402

DEFAULT_DATA_ROOT = Path(r"F:\IRIS\data\RetailS")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "reports" / "demo"

COCO_SKELETON_EDGES: tuple[tuple[int, int, tuple[int, int, int]], ...] = (
    (NOSE, L_EYE, (220, 220, 120)),
    (NOSE, R_EYE, (220, 220, 120)),
    (L_EYE, L_EAR, (200, 200, 100)),
    (R_EYE, R_EAR, (200, 200, 100)),
    (L_SHOULDER, R_SHOULDER, (255, 210, 90)),
    (L_SHOULDER, L_HIP, (255, 200, 80)),
    (R_SHOULDER, R_HIP, (255, 200, 80)),
    (L_HIP, R_HIP, (255, 200, 80)),
    (L_SHOULDER, L_ELBOW, (120, 240, 120)),
    (L_ELBOW, L_WRIST, (80, 255, 80)),
    (R_SHOULDER, R_ELBOW, (120, 190, 255)),
    (R_ELBOW, R_WRIST, (70, 160, 255)),
    (L_HIP, L_KNEE, (210, 170, 110)),
    (L_KNEE, L_ANKLE, (190, 150, 95)),
    (R_HIP, R_KNEE, (210, 170, 110)),
    (R_KNEE, R_ANKLE, (190, 150, 95)),
)

STAGE_ORDER = ("IDLE", "DWELL", "REACH", "AT_BODY", "RETURNED", "HIDDEN", "COOLDOWN")

STAGE_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "IDLE": (165, 165, 165),
    "DWELL": (245, 185, 65),
    "REACH": (50, 210, 255),
    "AT_BODY": (45, 135, 255),
    "RETURNED": (65, 75, 255),
    "HIDDEN": (65, 75, 255),
    "COOLDOWN": (90, 95, 245),
}


def find_best_zero_miss_clip(
    data_root: Path,
    cfg: ServiceConfig,
    prefer_split: str = "realworld",
) -> tuple[Path, Path, str]:
    """Find a single-person clip with 0 missed frames and a clean high-tier `DWELL -> REACH -> AT_BODY` detection."""
    inner = data_root / "RetailS" if (data_root / "RetailS").is_dir() else data_root
    splits = [
        (
            "realworld",
            inner / "RetailS_test_realworld" / "pose" / "test",
            inner / "RetailS_test_realworld" / "gt" / "test_frame_mask",
        ),
        (
            "staged",
            inner / "RetailS_test_staged" / "pose" / "test",
            inner / "RetailS_test_staged" / "gt" / "test_frame_mask",
        ),
    ]
    if prefer_split == "staged":
        splits.reverse()

    for split_name, pose_dir, gt_dir in splits:
        if not pose_dir.is_dir():
            continue
        candidates_found: list[tuple[float, Path, Path, str]] = []
        for pose_path in sorted(pose_dir.glob("*.json")):
            gt_path = gt_dir / f"{pose_path.stem}.npy"
            if not gt_path.exists():
                continue
            mask = np.load(gt_path)
            gt_events = extract_mask_events(mask, fps=15.0)
            if not gt_events:
                continue
            raw = json.loads(pose_path.read_text(encoding="utf-8"))
            cam_id = parse_camera_id(pose_path.stem)
            frames = load_retails_pose_json(raw, camera=cam_id, fps=15.0)
            if not (85 <= len(frames) <= 260):
                continue
            # Require zero miss-detections across the entire clip
            if any(len(f.tracks) == 0 for f in frames):
                continue
            track_ids = {t.track_id for f in frames for t in f.tracks}
            if len(track_ids) != 1:
                continue

            cz = cfg.camera_zones.get(cam_id, CameraZones(camera=cam_id, zone_free=True))
            mon = ConcealmentMonitor(cam_id, cfg.detector, cz)
            emitted: list[CandidateEvent] = []
            insuff = 0
            for f in frames:
                res = mon.update(f)
                emitted.extend(res.candidates)
                insuff += len(res.assessments)

            if len(emitted) == 1 and insuff == 0 and emitted[0].tier == "high":
                c = emitted[0]
                stages = [s.stage for s in c.stage_timeline]
                gt_s, gt_e = gt_events[0]
                if (
                    "DWELL" in stages
                    and "REACH" in stages
                    and "AT_BODY" in stages
                    and (gt_s - 0.5 <= c.pts <= gt_e + 1.0)
                ):
                    candidates_found.append((c.score, pose_path, gt_path, split_name))

        if candidates_found:
            candidates_found.sort(key=lambda item: item[0], reverse=True)
            _, best_pose, best_gt, s_name = candidates_found[0]
            return (best_pose, best_gt, s_name)

    raise FileNotFoundError(f"No matching clip found under {data_root}")


def _build_store_canvas(width: int, height: int, camera_zones: CameraZones) -> np.ndarray:
    """Create a clean dark-mode supermarket CCTV floor & shelf-aisle canvas with zone overlays."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    # Subtle vertical gradient for CCTV aisle backdrop
    y_grad = np.linspace(26, 44, height, dtype=np.uint8)[:, None]
    canvas[:, :, 0] = y_grad + 8
    canvas[:, :, 1] = y_grad + 2
    canvas[:, :, 2] = y_grad

    # Subtle perspective floor grid
    grid_color = (52, 48, 44)
    for x in range(0, width, 120):
        cv2.line(canvas, (x, 0), (x, height), grid_color, 1, cv2.LINE_AA)
    for y in range(0, height, 108):
        cv2.line(canvas, (0, y), (width, y), grid_color, 1, cv2.LINE_AA)

    # Draw calibrated shelf_front zones
    for z in camera_zones.zones:
        pts_px = np.array(
            [[int(p[0] * width), int(p[1] * height)] for p in z.points],
            dtype=np.int32,
        )
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts_px], (75, 65, 30))
        cv2.addWeighted(overlay, 0.32, canvas, 0.68, 0, canvas)
        cv2.polylines(canvas, [pts_px], isClosed=True, color=(190, 165, 75), thickness=2, lineType=cv2.LINE_AA)

        cx = int(np.mean(pts_px[:, 0]))
        cy = int(np.mean(pts_px[:, 1]))
        label_pos = ( int(np.min(pts_px[:, 0])) + 12, int(np.min(pts_px[:, 1])) + 26 )
        cv2.putText(
            canvas,
            f"ZONE: {z.id} (shelf_front)",
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (215, 195, 115),
            1,
            cv2.LINE_AA,
        )
        if z.facing is not None:
            end_x = int(cx + z.facing[0] * 70)
            end_y = int(cy + z.facing[1] * 70)
            cv2.arrowedLine(
                canvas,
                (cx, cy),
                (end_x, end_y),
                (220, 200, 110),
                2,
                cv2.LINE_AA,
                tipLength=0.25,
            )
            cv2.putText(
                canvas,
                "shelf facing",
                (end_x + 6, end_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 185, 110),
                1,
                cv2.LINE_AA,
            )
    return canvas


def _draw_person_and_pose(
    img: np.ndarray,
    obs: TrackObservation,
    feat: TrackFeatures,
    stage: str,
    alert_active: bool,
    wrist_trail: deque[tuple[int, int]],
    conf_thresh: float = 0.25,
) -> None:
    """Draw person bounding box, body concealment radius, wrist trail, and COCO-17 skeleton."""
    h, w = img.shape[:2]
    stage_color = (60, 70, 255) if alert_active else STAGE_COLORS_BGR.get(stage, (180, 180, 180))

    x1, y1, x2, y2 = [int(round(v)) for v in obs.box]
    # Soft glow box when in REACH / AT_BODY / ALERT
    if stage in ("REACH", "AT_BODY") or alert_active:
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), stage_color, -1)
        cv2.addWeighted(overlay, 0.10, img, 0.90, 0, img)

    cv2.rectangle(img, (x1, y1), (x2, y2), stage_color, 2, cv2.LINE_AA)

    # Person header badge
    badge_text = f"nvtracker #{obs.track_id} | Stage: {stage}"
    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
    by1 = max(0, y1 - th - 14)
    cv2.rectangle(img, (x1, by1), (x1 + tw + 16, y1), stage_color, -1)
    cv2.putText(
        img,
        badge_text,
        (x1 + 8, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (15, 15, 15) if stage in ("DWELL", "REACH") and not alert_active else (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Draw ground-anchor projection on the floor
    gx_px = int(round(feat.ground_point_norm[0] * w))
    gy_px = int(round(feat.ground_point_norm[1] * h))
    cv2.circle(img, (gx_px, gy_px), 7, (255, 215, 60), -1, cv2.LINE_AA)
    cv2.circle(img, (gx_px, gy_px), 13, (255, 215, 60), 1, cv2.LINE_AA)
    cv2.putText(
        img,
        "floor anchor",
        (gx_px + 14, gy_px + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (235, 205, 95),
        1,
        cv2.LINE_AA,
    )

    # Draw torso center & waist/chest concealment zone radius
    tc_x, tc_y = int(round(feat.torso_center[0])), int(round(feat.torso_center[1]))
    body_r_px = int(round(0.50 * feat.torso_length))
    cv2.circle(
        img,
        (tc_x, tc_y),
        body_r_px,
        (55, 140, 255) if feat.in_at_body_pose else (110, 110, 130),
        1,
        cv2.LINE_AA,
    )

    # Draw active wrist trajectory trail
    if len(wrist_trail) >= 2:
        pts_arr = list(wrist_trail)
        for i in range(1, len(pts_arr)):
            alpha = i / len(pts_arr)
            col = (
                int(60 + 195 * alpha),
                int(220 * alpha),
                int(255 * (1.0 - 0.4 * alpha)),
            )
            cv2.line(img, pts_arr[i - 1], pts_arr[i], col, max(1, int(3 * alpha)), cv2.LINE_AA)

    # Draw COCO-17 skeleton edges
    kps = obs.kps
    for i, j, edge_col in COCO_SKELETON_EDGES:
        if i < len(kps) and j < len(kps):
            x_i, y_i, c_i = kps[i]
            x_j, y_j, c_j = kps[j]
            if c_i >= conf_thresh and c_j >= conf_thresh:
                cv2.line(
                    img,
                    (int(round(x_i)), int(round(y_i))),
                    (int(round(x_j)), int(round(y_j))),
                    edge_col,
                    2,
                    cv2.LINE_AA,
                )

    # Draw keypoints & highlight active wrist
    act_wr_idx = R_WRIST if feat.active_hand == "right" else L_WRIST
    act_el_idx = R_ELBOW if feat.active_hand == "right" else L_ELBOW
    for idx, (kx, ky, kc) in enumerate(kps):
        if kc < conf_thresh:
            continue
        px, py = int(round(kx)), int(round(ky))
        if idx == act_wr_idx:
            wr_col = (50, 60, 255) if feat.in_at_body_pose else (50, 235, 255)
            cv2.circle(img, (px, py), 8, wr_col, -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 13, (255, 255, 255), 2, cv2.LINE_AA)
            tag = f"{feat.active_hand} wrist ({feat.body_zone_name}, {feat.wrist_to_body_torsos:.2f}T)"
            cv2.putText(
                img,
                tag,
                (px + 14, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                wr_col,
                1,
                cv2.LINE_AA,
            )
        elif idx == act_el_idx:
            cv2.circle(img, (px, py), 6, (90, 230, 255), -1, cv2.LINE_AA)
            cv2.putText(
                img,
                f"elbow {feat.min_elbow_deg:.0f} deg",
                (px + 10, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (170, 230, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.circle(img, (px, py), 4, (245, 245, 245), -1, cv2.LINE_AA)


def _draw_hud_and_timeline(
    img: np.ndarray,
    frame_idx: int,
    total_frames: int,
    pts: float,
    duration_s: float,
    camera: str,
    clip_id: str,
    split_name: str,
    stage: str,
    dwell_s: float,
    hold_s: float,
    feat: TrackFeatures,
    q_state: str,
    q_conf: float,
    gt_intervals: list[tuple[float, float]],
    candidate_event: CandidateEvent | None,
    candidate_record: dict | None,
    settings: DetectorSettings,
    missed_frames: int,
) -> None:
    """Draw the right-hand kinematic telemetry panel, top alert banner, and bottom timeline bar."""
    h, w = img.shape[:2]

    # 1. Right-side telemetry HUD panel
    panel_w = 450
    px1 = w - panel_w - 20
    py1 = 20
    py2 = h - 135
    overlay = img.copy()
    cv2.rectangle(overlay, (px1, py1), (w - 20, py2), (22, 20, 18), -1)
    cv2.addWeighted(overlay, 0.84, img, 0.16, 0, img)
    cv2.rectangle(img, (px1, py1), (w - 20, py2), (95, 90, 82), 1, cv2.LINE_AA)

    tx = px1 + 18
    ty = py1 + 32
    cv2.putText(
        img,
        "IRIS CONCEALMENT MONITOR (POSE-FIRST)",
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (240, 215, 120),
        2,
        cv2.LINE_AA,
    )
    ty += 26
    cv2.putText(
        img,
        f"Clip: {clip_id} ({split_name}) | Camera: {camera}",
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    ty += 22
    cv2.putText(
        img,
        f"Frame: {frame_idx + 1}/{total_frames} | PTS: {pts:5.2f}s / {duration_s:.2f}s",
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    ty += 22
    q_col = (100, 235, 120) if q_state == "OK" else (80, 190, 255)
    cv2.putText(
        img,
        f"Quality Gate: {q_state} (core_conf={q_conf:.2f}, {missed_frames} unobserved frames)",
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        q_col,
        1,
        cv2.LINE_AA,
    )

    # State machine progression pills
    ty += 30
    cv2.putText(
        img,
        "Ordered Per-Track State Machine:",
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    ty += 14
    pill_stages = ("IDLE", "DWELL", "REACH", "AT_BODY", "CANDIDATE")
    active_pill = "CANDIDATE" if candidate_event is not None else stage
    pw = (panel_w - 36 - 4 * 6) // 5
    for idx, ps in enumerate(pill_stages):
        bx = tx + idx * (pw + 6)
        by = ty
        is_cur = (ps == active_pill) or (ps == "CANDIDATE" and candidate_event is not None)
        col = STAGE_COLORS_BGR.get(ps, (70, 75, 255)) if is_cur else (55, 55, 55)
        cv2.rectangle(img, (bx, by), (bx + pw, by + 28), col, -1 if is_cur else 1, cv2.LINE_AA)
        cv2.putText(
            img,
            ps,
            (bx + 5, by + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (15, 15, 15) if (is_cur and ps in ("DWELL", "REACH")) else (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

    # Live feature bars
    ty += 52
    cv2.putText(
        img,
        "Kinematic Telemetry vs Calibrated Thresholds:",
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    gauges = [
        (
            "Shelf Dwell (s)",
            dwell_s,
            settings.dwell_min_s,
            settings.dwell_min_s * 2,
            f"{dwell_s:.2f}s (min {settings.dwell_min_s:.2f}s)",
            dwell_s >= settings.dwell_min_s,
        ),
        (
            "Reach Excursion (T)",
            feat.wrist_excursion_torsos,
            settings.reach_excursion_torsos,
            1.2,
            f"{feat.wrist_excursion_torsos:.2f}T (min {settings.reach_excursion_torsos:.2f}T)",
            feat.reach_detected,
        ),
        (
            "Wrist-to-Body Dist (T)",
            feat.wrist_to_body_torsos,
            settings.body_zone_radius_torsos,
            1.2,
            f"{feat.wrist_to_body_torsos:.2f}T (<= {settings.body_zone_radius_torsos:.2f}T: {feat.body_zone_name})",
            feat.wrist_to_body_torsos <= settings.body_zone_radius_torsos,
        ),
        (
            "Elbow Flexion (deg)",
            feat.min_elbow_deg,
            settings.max_conceal_elbow_deg,
            180.0,
            f"{feat.min_elbow_deg:.0f} deg (<= {settings.max_conceal_elbow_deg:.0f} deg)",
            feat.elbow_flexed_or_medial,
        ),
        (
            "Body Hold Duration (s)",
            hold_s,
            settings.min_hold_s,
            settings.min_hold_s * 4,
            f"{hold_s:.2f}s (min {settings.min_hold_s:.2f}s)",
            hold_s >= settings.min_hold_s,
        ),
    ]
    bar_w = panel_w - 36
    for label, val, thresh, max_v, val_str, ok in gauges:
        ty += 24
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (195, 195, 195), 1, cv2.LINE_AA)
        cv2.putText(
            img,
            val_str,
            (tx + 175, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (95, 235, 130) if ok else (175, 175, 175),
            1,
            cv2.LINE_AA,
        )
        ty += 7
        cv2.rectangle(img, (tx, ty), (tx + bar_w, ty + 10), (50, 50, 50), -1)
        fill = int(round(bar_w * min(1.0, max(0.0, val / max(max_v, 1e-6)))))
        bar_col = (75, 215, 110) if ok else (90, 155, 230)
        cv2.rectangle(img, (tx, ty), (tx + fill, ty + 10), bar_col, -1)
        th_x = tx + int(round(bar_w * min(1.0, thresh / max(max_v, 1e-6))))
        cv2.line(img, (th_x, ty - 2), (th_x, ty + 12), (255, 255, 255), 2)
        ty += 12

    # Decision box inside panel
    ty += 26
    cv2.putText(img, "System Decision:", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    ty += 10
    if candidate_event is not None and candidate_record is not None:
        cv2.rectangle(img, (tx, ty), (tx + bar_w, py2 - 16), (35, 42, 135), -1)
        cv2.rectangle(img, (tx, ty), (tx + bar_w, py2 - 16), (70, 95, 255), 2)
        cv2.putText(
            img,
            f"ALERT TRIGGERED @ pts={candidate_event.pts:.2f}s",
            (tx + 12, ty + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (110, 225, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"Tier: {candidate_event.tier.upper()} | Severity: {candidate_record['severity'].upper()} | Score: {candidate_event.score:.2f}",
            (tx + 12, ty + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"Target: {candidate_event.target_kind} ({candidate_event.body_zone_name}) | Terminal: {candidate_event.terminal_stage}",
            (tx + 12, ty + 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
        tl_str = " -> ".join(f"{s.stage}@{s.entered_pts:.1f}s" for s in candidate_event.stage_timeline)
        cv2.putText(
            img,
            f"Timeline: {tl_str}",
            (tx + 12, ty + 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (205, 225, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "Status: SUSPECTED INCIDENT (Requires Human Verification)",
            (tx + 12, ty + 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (130, 245, 160),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.rectangle(img, (tx, ty), (tx + bar_w, ty + 64), (40, 40, 40), -1)
        cv2.putText(
            img,
            f"Monitoring Track (Current Stage: {stage})",
            (tx + 12, ty + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            STAGE_COLORS_BGR.get(stage, (200, 200, 200)),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "No alert emitted yet -- awaiting full ordered sequence.",
            (tx + 12, ty + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (165, 165, 165),
            1,
            cv2.LINE_AA,
        )

    # 2. Top Alert Banner when triggered
    if candidate_event is not None and candidate_record is not None:
        banner_w = w - panel_w - 60
        overlay2 = img.copy()
        cv2.rectangle(overlay2, (20, 20), (20 + banner_w, 96), (30, 35, 150), -1)
        cv2.addWeighted(overlay2, 0.88, img, 0.12, 0, img)
        cv2.rectangle(img, (20, 20), (20 + banner_w, 96), (70, 95, 255), 2)
        cv2.putText(
            img,
            f"REDIS iris:alert:{camera} | kind=suspected_concealment | severity={candidate_record['severity']} | SVR 12s Clip Queued",
            (36, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 230, 255),
            2,
            cv2.LINE_AA,
        )
        desc = candidate_record["description"]
        if len(desc) > 135:
            desc = desc[:132] + "..."
        cv2.putText(
            img,
            desc,
            (36, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

    # 3. Bottom Timeline Bar (Ground Truth vs IRIS Decision)
    by1 = h - 115
    by2 = h - 20
    overlay3 = img.copy()
    cv2.rectangle(overlay3, (20, by1), (w - 20, by2), (22, 20, 18), -1)
    cv2.addWeighted(overlay3, 0.86, img, 0.14, 0, img)
    cv2.rectangle(img, (20, by1), (w - 20, by2), (95, 90, 82), 1, cv2.LINE_AA)

    bar_x1 = 210
    bar_x2 = w - 45
    bar_len = bar_x2 - bar_x1
    dur = max(duration_s, 1e-6)

    # Ground-truth interval track
    gt_y = by1 + 28
    cv2.putText(img, "Dataset Ground Truth:", (34, gt_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.rectangle(img, (bar_x1, gt_y), (bar_x2, gt_y + 16), (48, 48, 48), -1)
    for gs, ge in gt_intervals:
        gx1 = bar_x1 + int(round((gs / dur) * bar_len))
        gx2 = bar_x1 + int(round((ge / dur) * bar_len))
        cv2.rectangle(img, (gx1, gt_y), (gx2, gt_y + 16), (60, 180, 95), -1)
        cv2.putText(
            img,
            f"GT Concealment [{gs:.1f}s - {ge:.1f}s]",
            (gx1 + 6, gt_y + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # System stage & trigger track
    sys_y = by1 + 58
    cv2.putText(img, "IRIS State & Decision:", (34, sys_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.rectangle(img, (bar_x1, sys_y), (bar_x2, sys_y + 16), (48, 48, 48), -1)
    cur_x = bar_x1 + int(round((pts / dur) * bar_len))
    cv2.rectangle(img, (bar_x1, sys_y), (cur_x, sys_y + 16), STAGE_COLORS_BGR.get(stage, (160, 160, 160)), -1)

    if candidate_event is not None:
        trig_x = bar_x1 + int(round((candidate_event.pts / dur) * bar_len))
        cv2.line(img, (trig_x, by1 + 10), (trig_x, by2 - 10), (70, 85, 255), 3, cv2.LINE_AA)
        cv2.circle(img, (trig_x, sys_y + 8), 7, (70, 85, 255), -1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"ALERT @ {candidate_event.pts:.2f}s",
            (min(bar_x2 - 130, trig_x + 10), sys_y + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (110, 225, 255),
            1,
            cv2.LINE_AA,
        )

    # Playhead cursor
    cv2.line(img, (cur_x, by1 + 8), (cur_x, by2 - 8), (255, 255, 255), 2, cv2.LINE_AA)


def render_demo(
    pose_path: Path,
    gt_path: Path | None,
    split_name: str,
    cfg: ServiceConfig,
    out_dir: Path,
    rgb_video_path: Path | None = None,
    fps: float = 15.0,
) -> tuple[Path, Path]:
    """Run `pose_path` through `ConcealmentMonitor` and write the annotated `.mp4` and `.png` keyframe sheet."""
    global cv2
    if cv2 is None:
        import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_json = json.loads(pose_path.read_text(encoding="utf-8"))
    camera = parse_camera_id(pose_path.stem)
    frames = load_retails_pose_json(raw_json, camera=camera, fps=fps)
    if not frames:
        raise ValueError(f"No frames loaded from {pose_path}")

    gt_intervals: list[tuple[float, float]] = []
    if gt_path is not None and gt_path.exists():
        gt_intervals = extract_mask_events(np.load(gt_path), fps=fps)

    cz = cfg.camera_zones.get(camera, CameraZones(camera=camera, zone_free=True))
    mon = ConcealmentMonitor(camera, cfg.detector, cz)

    width, height = frames[0].shape
    base_canvas = _build_store_canvas(width, height, cz)

    rgb_cap = cv2.VideoCapture(str(rgb_video_path)) if rgb_video_path and rgb_video_path.exists() else None
    mp4_path = out_dir / f"demo_{split_name}_{pose_path.stem}.mp4"
    png_path = out_dir / f"demo_{split_name}_{pose_path.stem}_keyframes.png"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (width, height))

    wrist_trail: deque[tuple[int, int]] = deque(maxlen=25)
    latched_candidate: CandidateEvent | None = None
    latched_record: dict | None = None
    duration_s = frames[-1].pts if frames else 1.0
    unobserved_frames = 0

    stage_snapshots: dict[str, np.ndarray] = {}
    # The storyboard needs six evenly spaced images, not the whole clip: holding every 1080p
    # frame cost ~6.2 MB per frame (~4 GB for a 658-frame clip).
    fill_samples: list[np.ndarray] = []
    sample_every = max(1, len(frames) // 6)

    for idx, frame in enumerate(frames):
        if not frame.observed:
            unobserved_frames += 1
        if rgb_cap is not None:
            ok, bgr = rgb_cap.read()
            if ok and bgr is not None:
                img = cv2.resize(bgr, (width, height))
            else:
                img = base_canvas.copy()
        else:
            img = base_canvas.copy()

        res = mon.update(frame)
        if res.candidates and latched_candidate is None:
            latched_candidate = res.candidates[0]
            latched_record = build_candidate_record(latched_candidate)

        if not frame.tracks:
            # A frame the pose model left empty still belongs in the video; it just has no
            # person to annotate.
            writer.write(img)
            if len(fill_samples) < 6 and idx % sample_every == 0:
                fill_samples.append(img.copy())
            continue

        obs = frame.tracks[0]
        cid = obs.track_id
        sm = mon._machines.get(cid)
        track_rec = mon.quality_gate._tracks.get(cid)
        q_status = track_rec.last_status if track_rec is not None else QualityStatus(state="UNKNOWN")
        hist = list(mon._histories.get(cid, ()))
        feat = extract_track_features(obs, hist, cz, frame.shape, cfg.detector,
                                      active_hand=sm.active_hand if sm is not None else None)

        act_wr_idx = R_WRIST if feat.active_hand == "right" else L_WRIST
        wx, wy, wc = obs.kps[act_wr_idx]
        if wc >= 0.25:
            wrist_trail.append((int(round(wx)), int(round(wy))))

        cur_stage = sm.stage if sm is not None else "IDLE"
        dwell_s = sm.dwell_accum_s if sm is not None else 0.0
        hold_s = (
            (frame.pts - sm.at_body_entered_pts)
            if (sm is not None and sm.at_body_entered_pts is not None and cur_stage == "AT_BODY")
            else (latched_candidate.hold_duration_s if latched_candidate is not None else 0.0)
        )

        _draw_person_and_pose(
            img=img,
            obs=obs,
            feat=feat,
            stage=cur_stage,
            alert_active=(latched_candidate is not None),
            wrist_trail=wrist_trail,
        )
        _draw_hud_and_timeline(
            img=img,
            frame_idx=idx,
            total_frames=len(frames),
            pts=frame.pts,
            duration_s=duration_s,
            camera=camera,
            clip_id=pose_path.stem,
            split_name=split_name,
            stage=cur_stage,
            dwell_s=dwell_s,
            hold_s=hold_s,
            feat=feat,
            q_state=q_status.state,
            q_conf=q_status.core_conf,
            gt_intervals=gt_intervals,
            candidate_event=latched_candidate,
            candidate_record=latched_record,
            settings=cfg.detector,
            missed_frames=unobserved_frames,
        )

        writer.write(img)
        if len(fill_samples) < 6 and idx % sample_every == 0:
            fill_samples.append(img.copy())

        if cur_stage not in stage_snapshots:
            stage_snapshots[cur_stage] = img.copy()
        if res.candidates:
            stage_snapshots["ALERT_TRIGGERED"] = img.copy()
        del img

    writer.release()
    if rgb_cap is not None:
        rgb_cap.release()

    # Build a 6-panel keyframe storyboard image (2x3 grid)
    selected_panels: list[np.ndarray] = []
    for key in ("IDLE", "DWELL", "REACH", "AT_BODY", "ALERT_TRIGGERED"):
        if key in stage_snapshots:
            selected_panels.append(stage_snapshots[key])
            del stage_snapshots[key]
    for sample in fill_samples:
        if len(selected_panels) >= 6:
            break
        selected_panels.append(sample)
    while len(selected_panels) < 6 and selected_panels:
        selected_panels.append(selected_panels[-1])

    pw, ph = 640, 360
    grid_rows = []
    for r in range(2):
        row_imgs = [cv2.resize(selected_panels[r * 3 + c], (pw, ph)) for c in range(3)]
        grid_rows.append(np.hstack(row_imgs))
    sheet = np.vstack(grid_rows)
    cv2.imwrite(str(png_path), sheet)

    return (mp4_path, png_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render annotated concealment detection demo video.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pose-json", type=Path, default=None, help="Specific RetailS/PoseLift pose JSON file")
    parser.add_argument("--gt-npy", type=Path, default=None, help="Optional ground-truth .npy frame mask")
    parser.add_argument("--rgb-video", type=Path, default=None, help="Optional RGB video to overlay onto")
    parser.add_argument("--split", choices=("realworld", "staged"), default="realworld")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    cfg = load_config()
    if args.pose_json is not None:
        pose_path = args.pose_json
        gt_path = args.gt_npy
        split_name = args.split
    else:
        pose_path, gt_path, split_name = find_best_zero_miss_clip(
            args.data_root, cfg, prefer_split=args.split
        )

    mp4_path, png_path = render_demo(
        pose_path=pose_path,
        gt_path=gt_path,
        split_name=split_name,
        cfg=cfg,
        out_dir=args.out_dir,
        rgb_video_path=args.rgb_video,
    )
    print(f"Rendered demo video:     {mp4_path}")
    print(f"Rendered keyframe sheet: {png_path}")


if __name__ == "__main__":
    main()
