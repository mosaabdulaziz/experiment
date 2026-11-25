"""Extract tracked COCO-17 poses and bag/basket detections from RGB videos and render annotated demos.

Stores all YOLO weights strictly under `F:\\IRIS\\data\\models` and runs every frame through
the unmodified production `acl.parse_pose_frame` -> `ConcealmentMonitor.update` ->
`evidence.build_candidate_record` pipeline.
"""
from __future__ import annotations

from collections import deque
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

cv2 = None
acl = None

# Ensure Ultralytics / Torch caches stay on F:\IRIS
MODELS_DIR = Path(r"F:\IRIS\data\models")

PRODUCT_DIR = Path(__file__).resolve().parents[1]
if str(PRODUCT_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCT_DIR))

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
    PoseFrame,
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
    ZonePolygon,
)
from config import load_config  # noqa: E402
from evidence import build_candidate_record  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "reports" / "demo"

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

STAGE_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "IDLE": (180, 180, 180),
    "DWELL": (245, 190, 60),
    "REACH": (40, 215, 255),
    "AT_BODY": (35, 130, 255),
    "RETURNED": (55, 65, 255),
    "HIDDEN": (55, 65, 255),
    "COOLDOWN": (65, 75, 250),
}

# COCO class IDs for bags: 24=backpack, 26=handbag, 28=suitcase
COCO_BAG_CLASSES = {24: "backpack", 26: "handbag", 28: "suitcase"}


def _bounded_digest(path: Path, window: int = 8 * 1024 * 1024) -> str:
    """Digest of a file's size plus its first and last `window` bytes.

    Cheap enough to run before every cached demo, and enough to notice a re-encoded clip or a
    swapped checkpoint. This is cache validity, not a security boundary.
    """
    path = Path(path)
    size = path.stat().st_size
    digest = hashlib.sha256(f"{path.name}:{size}".encode())
    with path.open("rb") as stream:
        digest.update(stream.read(window))
        if size > 2 * window:
            stream.seek(size - window)
            digest.update(stream.read(window))
    return digest.hexdigest()


def video_cache_fingerprint(
    video_path: Path,
    camera: str,
    pose_ckpt: Path,
    det_ckpt: Path,
    settings: dict | None = None,
) -> str:
    """Identity of everything that can invalidate a cached `iris:meta` extraction."""
    parts = {
        "video": _bounded_digest(video_path),
        "camera": camera,
        "pose_model": _bounded_digest(Path(pose_ckpt)),
        "det_model": _bounded_digest(Path(det_ckpt)),
        "settings": json.dumps(settings or {}, sort_keys=True, default=str),
        "script": _bounded_digest(Path(__file__)),
    }
    try:
        parts["ultralytics"] = version("ultralytics")
    except PackageNotFoundError:
        parts["ultralytics"] = "absent"
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def extract_video_meta_entries(
    video_path: Path,
    camera: str,
    pose_model: YOLO,
    det_model: YOLO,
    cache_path: Path,
    settings: dict | None = None,
) -> tuple[list[dict], float, tuple[int, int]]:
    """Run YOLO-Pose tracking + YOLO bag detection on `video_path` and cache `iris:meta` entries."""
    fingerprint = video_cache_fingerprint(
        video_path, camera, Path(pose_model.ckpt_path), Path(det_model.ckpt_path), settings
    )
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") != fingerprint:
            raise ValueError(
                f"stale pose cache {cache_path.name}: its fingerprint does not match the current "
                "video, checkpoints, inference settings or this script -- delete it and re-extract"
            )
        return (cached["entries"], float(cached["fps"]), tuple(cached["shape"]))

    cap = cv2.VideoCapture(str(video_path))
    w, h = 1920, 1080
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

    entries: list[dict] = []
    frame_idx = 0
    # Reset tracker state for this video
    if hasattr(pose_model, "predictor") and pose_model.predictor is not None:
        pose_model.predictor = None

    while True:
        ok, raw_frame = cap.read()
        if not ok or raw_frame is None:
            break
        frame = cv2.resize(raw_frame, (w, h), interpolation=cv2.INTER_LINEAR)
        pts = frame_idx / fps

        # 1. Pose + Person Tracking
        pose_res = pose_model.track(
            frame,
            persist=True,
            verbose=False,
            conf=0.25,
            iou=0.5,
            tracker="bytetrack.yaml",
        )[0]

        pose_items: list[dict] = []
        dets_rows: list[list] = []

        if pose_res.boxes is not None and pose_res.keypoints is not None:
            boxes_xyxy = pose_res.boxes.xyxy.cpu().numpy()
            confs = pose_res.boxes.conf.cpu().numpy()
            ids = (
                pose_res.boxes.id.int().cpu().numpy()
                if pose_res.boxes.id is not None
                else np.arange(1, len(boxes_xyxy) + 1)
            )
            kps_xy = pose_res.keypoints.xy.cpu().numpy()
            kps_conf = (
                pose_res.keypoints.conf.cpu().numpy()
                if pose_res.keypoints.conf is not None
                else np.ones((len(boxes_xyxy), 17), dtype=np.float32)
            )

            for idx_p in range(len(boxes_xyxy)):
                tid = int(ids[idx_p])
                x1, y1, x2, y2 = [round(float(v), 1) for v in boxes_xyxy[idx_p]]
                p_conf = round(float(confs[idx_p]), 3)
                kps_list = [
                    [
                        round(float(kps_xy[idx_p, j, 0]), 1),
                        round(float(kps_xy[idx_p, j, 1]), 1),
                        round(float(kps_conf[idx_p, j]), 3),
                    ]
                    for j in range(min(17, kps_xy.shape[1]))
                ]
                pose_items.append({"id": tid, "box": [x1, y1, x2, y2], "kps": kps_list})
                dets_rows.append(["person", p_conf, tid, x1, y1, round(x2 - x1, 1), round(y2 - y1, 1)])

        # 2. Bag / Handbag / Backpack Object Detection
        det_res = det_model.predict(
            frame,
            classes=list(COCO_BAG_CLASSES.keys()),
            conf=0.15,
            verbose=False,
        )[0]
        if det_res.boxes is not None and len(det_res.boxes) > 0:
            b_xyxy = det_res.boxes.xyxy.cpu().numpy()
            b_conf = det_res.boxes.conf.cpu().numpy()
            b_cls = det_res.boxes.cls.int().cpu().numpy()
            for idx_b in range(len(b_xyxy)):
                cls_id = int(b_cls[idx_b])
                label = COCO_BAG_CLASSES.get(cls_id, "handbag")
                bx1, by1, bx2, by2 = [round(float(v), 1) for v in b_xyxy[idx_b]]
                dets_rows.append(
                    [
                        label,
                        round(float(b_conf[idx_b]), 3),
                        900 + idx_b,
                        bx1,
                        by1,
                        round(bx2 - bx1, 1),
                        round(by2 - by1, 1),
                    ]
                )

        entry = {
            "id": f"{frame_idx + 1}-0",
            "fields": {
                "v": "1",
                "frame_num": str(frame_idx),
                "pts": f"{pts:.3f}",
                "pose": json.dumps({"shape": [w, h], "items": pose_items}),
                "dets": json.dumps(dets_rows),
            },
        }
        entries.append(entry)
        frame_idx += 1

    cap.release()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "camera": camera,
                "fps": fps,
                "shape": [w, h],
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    return (entries, fps, (w, h))


def _draw_zone_overlay(img: np.ndarray, camera_zones: CameraZones) -> None:
    h, w = img.shape[:2]
    for z in camera_zones.zones:
        pts_px = np.array(
            [[int(p[0] * w), int(p[1] * h)] for p in z.points],
            dtype=np.int32,
        )
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts_px], (65, 165, 225))
        cv2.addWeighted(overlay, 0.16, img, 0.84, 0, img)
        cv2.polylines(img, [pts_px], isClosed=True, color=(80, 210, 255), thickness=2, lineType=cv2.LINE_AA)

        lx = int(np.min(pts_px[:, 0])) + 10
        ly = max(24, int(np.min(pts_px[:, 1])) + 24)
        cv2.putText(
            img,
            f"SHELF ZONE: {z.id}",
            (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (100, 230, 255),
            2,
            cv2.LINE_AA,
        )
        if z.facing is not None:
            cx = int(np.mean(pts_px[:, 0]))
            cy = int(np.mean(pts_px[:, 1]))
            ex = int(cx + z.facing[0] * 65)
            ey = int(cy + z.facing[1] * 65)
            cv2.arrowedLine(img, (cx, cy), (ex, ey), (90, 225, 255), 2, cv2.LINE_AA, tipLength=0.28)


def _draw_track_on_rgb(
    img: np.ndarray,
    obs: TrackObservation,
    feat: TrackFeatures,
    stage: str,
    is_alerted_track: bool,
    wrist_trail: deque[tuple[int, int]],
    conf_thresh: float = 0.25,
    alert_target_label: str | None = None,
) -> None:
    col = (40, 50, 255) if is_alerted_track else STAGE_COLORS_BGR.get(stage, (185, 185, 185))
    x1, y1, x2, y2 = [int(round(v)) for v in obs.box]

    if stage in ("REACH", "AT_BODY") or is_alerted_track:
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), col, -1)
        cv2.addWeighted(overlay, 0.14, img, 0.86, 0, img)

    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)

    badge = f"Track #{obs.track_id} | {stage}"
    if is_alerted_track:
        t_lbl = alert_target_label or feat.target_kind
        badge = f"Track #{obs.track_id} | SUSPECTED CONCEALMENT ({t_lbl})"
    (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    by1 = max(0, y1 - th - 12)
    cv2.rectangle(img, (x1, by1), (x1 + tw + 14, y1), col, -1)
    txt_col = (15, 15, 15) if (stage in ("DWELL", "REACH") and not is_alerted_track) else (255, 255, 255)
    cv2.putText(img, badge, (x1 + 7, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, txt_col, 2, cv2.LINE_AA)

    # Draw personal bag bounding box if associated with this person
    if obs.has_bag and obs.bag_box is not None:
        bx1, by1_b, bx2, by2_b = [int(round(v)) for v in obs.bag_box]
        cv2.rectangle(img, (bx1, by1_b), (bx2, by2_b), (255, 120, 255), 2, cv2.LINE_AA)
        cv2.putText(
            img,
            "personal_bag",
            (bx1, max(16, by1_b - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 150, 255),
            2,
            cv2.LINE_AA,
        )

    # Draw store basket / cart bounding boxes if associated with this person
    for c_box in obs.cart_boxes:
        cx1, cy1_c, cx2, cy2_c = [int(round(v)) for v in c_box]
        cv2.rectangle(img, (cx1, cy1_c), (cx2, cy2_c), (90, 235, 120), 2, cv2.LINE_AA)
        cv2.putText(
            img,
            "store_basket (exempt)",
            (cx1, max(16, cy1_c - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (90, 235, 120),
            2,
            cv2.LINE_AA,
        )

    # Draw active wrist trail
    if len(wrist_trail) >= 2:
        pts_list = list(wrist_trail)
        for i in range(1, len(pts_list)):
            a = i / len(pts_list)
            c = (int(50 + 200 * a), int(225 * a), 255)
            cv2.line(img, pts_list[i - 1], pts_list[i], c, max(1, int(3 * a)), cv2.LINE_AA)

    # Draw COCO-17 skeleton
    kps = obs.kps
    for i, j, edge_col in COCO_SKELETON_EDGES:
        if i < len(kps) and j < len(kps):
            xi, yi, ci = kps[i]
            xj, yj, cj = kps[j]
            if ci >= conf_thresh and cj >= conf_thresh:
                cv2.line(
                    img,
                    (int(round(xi)), int(round(yi))),
                    (int(round(xj)), int(round(yj))),
                    edge_col,
                    2,
                    cv2.LINE_AA,
                )

    act_wr_idx = R_WRIST if feat.active_hand == "right" else L_WRIST
    for idx, (kx, ky, kc) in enumerate(kps):
        if kc < conf_thresh:
            continue
        px, py = int(round(kx)), int(round(ky))
        if idx == act_wr_idx:
            w_col = (40, 50, 255) if feat.in_at_body_pose else (40, 235, 255)
            cv2.circle(img, (px, py), 7, w_col, -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 11, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                img,
                f"{feat.active_hand} wrist -> {feat.body_zone_name} ({feat.wrist_to_body_torsos:.2f}T)",
                (px + 12, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                w_col,
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.circle(img, (px, py), 4, (245, 245, 245), -1, cv2.LINE_AA)


def _draw_rgb_hud(
    img: np.ndarray,
    video_name: str,
    camera: str,
    frame_idx: int,
    total_frames: int,
    pts: float,
    duration_s: float,
    primary_tid: int,
    stage: str,
    dwell_s: float,
    hold_s: float,
    feat: TrackFeatures,
    q_state: str,
    q_conf: float,
    candidate_event: CandidateEvent | None,
    candidate_record: dict | None,
    settings: DetectorSettings,
) -> None:
    """Draw a sleek translucent HUD overlay and bottom decision timeline onto a 1920x1080 canvas."""
    h, w = img.shape[:2]

    # Top-right translucent telemetry card
    card_w = 470
    cx1 = w - card_w - 20
    cy1 = 20
    cy2 = 435
    over = img.copy()
    cv2.rectangle(over, (cx1, cy1), (w - 20, cy2), (18, 18, 22), -1)
    cv2.addWeighted(over, 0.80, img, 0.20, 0, img)
    cv2.rectangle(img, (cx1, cy1), (w - 20, cy2), (100, 100, 110), 1, cv2.LINE_AA)

    tx = cx1 + 16
    ty = cy1 + 28
    cv2.putText(
        img,
        f"IRIS POSE-FIRST CONCEALMENT MONITOR ({camera})",
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (95, 225, 255),
        2,
        cv2.LINE_AA,
    )
    ty += 24
    cv2.putText(
        img,
        f"Source: {video_name} | Frame {frame_idx + 1}/{total_frames} | PTS: {pts:5.2f}s / {duration_s:.2f}s",
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (215, 215, 215),
        1,
        cv2.LINE_AA,
    )
    ty += 22
    cv2.putText(
        img,
        f"Primary Track #{primary_tid} | Quality Gate: {q_state} (core_conf={q_conf:.2f})",
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (110, 240, 130) if q_state == "OK" else (120, 200, 255),
        1,
        cv2.LINE_AA,
    )

    # State machine pills
    ty += 24
    pill_stages = ("IDLE", "DWELL", "REACH", "AT_BODY", "CANDIDATE")
    active_pill = "CANDIDATE" if candidate_event is not None else stage
    pw = (card_w - 32 - 24) // 5
    for idx, ps in enumerate(pill_stages):
        bx = tx + idx * (pw + 6)
        is_cur = (ps == active_pill) or (ps == "CANDIDATE" and candidate_event is not None)
        col = STAGE_COLORS_BGR.get(ps, (55, 65, 255)) if is_cur else (65, 65, 70)
        cv2.rectangle(img, (bx, ty), (bx + pw, ty + 26), col, -1 if is_cur else 1, cv2.LINE_AA)
        t_col = (15, 15, 15) if (is_cur and ps in ("DWELL", "REACH")) else (245, 245, 245)
        cv2.putText(img, ps, (bx + 5, ty + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, t_col, 1, cv2.LINE_AA)

    # Telemetry bars
    ty += 44
    gauges = [
        ("Shelf Dwell", dwell_s, settings.dwell_min_s, settings.dwell_min_s * 2,
         f"{dwell_s:.2f}s (min {settings.dwell_min_s:.2f}s)", dwell_s >= settings.dwell_min_s),
        ("Reach Excursion", feat.wrist_excursion_torsos, settings.reach_excursion_torsos, 1.2,
         f"{feat.wrist_excursion_torsos:.2f}T (min {settings.reach_excursion_torsos:.2f}T)", feat.reach_detected),
        ("Wrist-to-Body/Bag", feat.wrist_to_body_torsos, settings.body_zone_radius_torsos, 1.2,
         f"{feat.wrist_to_body_torsos:.2f}T (<= {settings.body_zone_radius_torsos:.2f}T: {feat.body_zone_name})",
         feat.wrist_to_body_torsos <= settings.body_zone_radius_torsos),
        ("Elbow Angle", feat.min_elbow_deg, settings.max_conceal_elbow_deg, 180.0,
         f"{feat.min_elbow_deg:.0f} deg (<= {settings.max_conceal_elbow_deg:.0f} deg)", feat.elbow_flexed_or_medial),
        ("Conceal Hold", hold_s, settings.min_hold_s, settings.min_hold_s * 4,
         f"{hold_s:.2f}s (min {settings.min_hold_s:.2f}s)", hold_s >= settings.min_hold_s),
    ]
    bar_w = card_w - 32
    for label, val, th, mv, v_str, ok in gauges:
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (205, 205, 205), 1, cv2.LINE_AA)
        cv2.putText(
            img,
            v_str,
            (tx + 170, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (105, 240, 135) if ok else (185, 185, 185),
            1,
            cv2.LINE_AA,
        )
        ty += 6
        cv2.rectangle(img, (tx, ty), (tx + bar_w, ty + 8), (55, 55, 60), -1)
        fill = int(round(bar_w * min(1.0, max(0.0, val / max(mv, 1e-6)))))
        cv2.rectangle(img, (tx, ty), (tx + fill, ty + 8), (80, 220, 115) if ok else (90, 160, 235), -1)
        th_x = tx + int(round(bar_w * min(1.0, th / max(mv, 1e-6))))
        cv2.line(img, (th_x, ty - 2), (th_x, ty + 10), (255, 255, 255), 2)
        ty += 22

    # Decision box in card
    ty += 4
    if candidate_event is not None and candidate_record is not None:
        cv2.rectangle(img, (tx, ty), (tx + bar_w, cy2 - 14), (35, 40, 155), -1)
        cv2.rectangle(img, (tx, ty), (tx + bar_w, cy2 - 14), (70, 95, 255), 2)
        cv2.putText(
            img,
            f"DECISION: ALERT EMITTED @ pts={candidate_event.pts:.2f}s",
            (tx + 10, ty + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (115, 230, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"Target: {candidate_event.target_kind.upper()} ({candidate_event.body_zone_name}) | Tier: {candidate_event.tier.upper()}",
            (tx + 10, ty + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        tl_str = " -> ".join(f"{s.stage}@{s.entered_pts:.1f}s" for s in candidate_event.stage_timeline)
        cv2.putText(
            img,
            tl_str,
            (tx + 10, ty + 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (205, 225, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.rectangle(img, (tx, ty), (tx + bar_w, cy2 - 14), (38, 38, 44), -1)
        cv2.putText(
            img,
            f"DECISION: MONITORING (Stage={stage})",
            (tx + 10, ty + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            STAGE_COLORS_BGR.get(stage, (200, 200, 200)),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "Requires human verification when candidate triggers.",
            (tx + 10, ty + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )

    # Top-left alert banner once triggered
    if candidate_event is not None and candidate_record is not None:
        bw = w - card_w - 60
        over2 = img.copy()
        cv2.rectangle(over2, (20, 20), (20 + bw, 94), (28, 35, 160), -1)
        cv2.addWeighted(over2, 0.86, img, 0.14, 0, img)
        cv2.rectangle(img, (20, 20), (20 + bw, 94), (75, 100, 255), 2)
        cv2.putText(
            img,
            f"REDIS iris:alert:{camera} | kind=suspected_concealment | target={candidate_event.target_kind} ({candidate_event.body_zone_name}) | severity={candidate_record['severity']}",
            (34, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (115, 235, 255),
            2,
            cv2.LINE_AA,
        )
        desc = candidate_record["description"]
        if len(desc) > 132:
            desc = desc[:129] + "..."
        cv2.putText(img, desc, (34, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (245, 245, 245), 1, cv2.LINE_AA)

    # Bottom progress timeline
    by1 = h - 68
    by2 = h - 18
    over3 = img.copy()
    cv2.rectangle(over3, (20, by1), (w - 20, by2), (18, 18, 22), -1)
    cv2.addWeighted(over3, 0.82, img, 0.18, 0, img)
    cv2.rectangle(img, (20, by1), (w - 20, cy2 if False else by2), (100, 100, 110), 1, cv2.LINE_AA)

    bx1 = 190
    bx2 = w - 40
    blen = bx2 - bx1
    dur = max(duration_s, 1e-6)
    cv2.putText(img, "IRIS Timeline:", (34, by1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (225, 225, 225), 1, cv2.LINE_AA)
    cv2.rectangle(img, (bx1, by1 + 16), (bx2, by1 + 34), (50, 50, 55), -1)
    cur_x = bx1 + int(round((pts / dur) * blen))
    cv2.rectangle(
        img,
        (bx1, by1 + 16),
        (cur_x, by1 + 34),
        (55, 65, 255) if candidate_event is not None else STAGE_COLORS_BGR.get(stage, (170, 170, 170)),
        -1,
    )
    if candidate_event is not None:
        trig_x = bx1 + int(round((candidate_event.pts / dur) * blen))
        cv2.line(img, (trig_x, by1 + 6), (trig_x, by2 - 6), (80, 235, 255), 3, cv2.LINE_AA)
        cv2.putText(
            img,
            f"ALERT @ {candidate_event.pts:.2f}s ({candidate_event.target_kind})",
            (min(bx2 - 220, trig_x + 8), by1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.line(img, (cur_x, by1 + 8), (cur_x, by2 - 8), (255, 255, 255), 2, cv2.LINE_AA)


def _draw_run_banner(img: np.ndarray, run_label: str, overrides: list[str]) -> None:
    """Draw the run identity (model/tracker/config) and every non-default override on the frame."""
    if not run_label and not overrides:
        return
    lines = [run_label] if run_label else []
    lines.extend(overrides)
    line_h = 22
    box_h = 12 + line_h * len(lines)
    y0 = 104
    max_w = 0
    for text in lines:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.47, 1)
        max_w = max(max_w, tw)
    over = img.copy()
    cv2.rectangle(over, (20, y0), (20 + max_w + 24, y0 + box_h), (18, 18, 22), -1)
    cv2.addWeighted(over, 0.78, img, 0.22, 0, img)
    cv2.rectangle(img, (20, y0), (20 + max_w + 24, y0 + box_h), (100, 100, 110), 1, cv2.LINE_AA)
    for i, text in enumerate(lines):
        col = (95, 225, 255) if (i == 0 and run_label) else (200, 200, 255)
        cv2.putText(img, text, (32, y0 + 22 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.47, col, 1, cv2.LINE_AA)


def render_rgb_video_demo(
    video_path: Path,
    camera: str,
    camera_zones: CameraZones,
    detector_settings: DetectorSettings,
    entries: list[dict],
    fps: float,
    out_mp4: Path,
    out_keyframes_png: Path,
    preferred_primary_tid: int | None = None,
    run_label: str = "",
    overrides: list[str] | None = None,
) -> dict:
    """Run extracted `iris:meta` entries through unmodified `acl` + `ConcealmentMonitor` and render RGB demo.

    Every emitted candidate is recorded in the returned summary (not only the one shown in the HUD),
    so false alerts on non-target tracks stay visible.
    """
    global cv2, acl
    if cv2 is None:
        import cv2
    if acl is None:
        import acl
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    monitor = ConcealmentMonitor(camera=camera, settings=detector_settings, camera_zones=camera_zones)

    parsed_frames: list[PoseFrame] = []
    for entry in entries:
        pf = acl.parse_pose_frame(camera=camera, entry=entry)
        if pf is not None:
            parsed_frames.append(pf)

    total_frames = len(parsed_frames)
    duration_s = parsed_frames[-1].pts if parsed_frames else 1.0

    # First pass to identify the alerted or furthest-stage track if not explicitly set
    if preferred_primary_tid is None:
        probe_mon = ConcealmentMonitor(camera=camera, settings=detector_settings, camera_zones=camera_zones)
        stage_rank = {"IDLE": 0, "DWELL": 1, "REACH": 2, "AT_BODY": 3, "RETURNED": 4, "HIDDEN": 4, "COOLDOWN": 5}
        best_tid = 1
        best_score = (-1, 0)
        for pf in parsed_frames:
            res = probe_mon.update(pf)
            if res.candidates:
                best_tid = res.candidates[0].track_id
                break
            for obs in pf.tracks:
                sm = probe_mon._machines.get(obs.track_id)
                rk = stage_rank.get(sm.stage if sm else "IDLE", 0)
                cand = (rk, int(obs.box[3] - obs.box[1]))
                if cand > best_score:
                    best_score = cand
                    best_tid = obs.track_id
        primary_tid = best_tid
    else:
        primary_tid = preferred_primary_tid

    cap = cv2.VideoCapture(str(video_path))
    w, h = 1920, 1080
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h))

    wrist_trails: dict[int, deque[tuple[int, int]]] = {}
    latched_event: CandidateEvent | None = None
    latched_record: dict | None = None
    alert_frame_idx: int | None = None
    stage_first_frame: dict[str, int] = {}
    # The storyboard needs six thumbnails, not the whole clip: keeping every 1080p frame cost
    # ~6.2 MB per frame (~4 GB for a 658-frame clip) and they were only ever resized to 640x360.
    thumb_w, thumb_h = 640, 360
    total_frames = len(parsed_frames)
    panel_thumbs: dict[int, np.ndarray] = {}
    wanted_panels: set[int] = set()
    prev_canvas: np.ndarray | None = None
    # Six evenly spaced panels stand in when the clip never reaches the stages we latch onto.
    fallback_panels = [min(total_frames - 1, int((k + 1) * total_frames / 7)) for k in range(6)] if total_frames else []
    wanted_panels.update(fallback_panels)

    def _keep_panel(index: int, image: np.ndarray) -> None:
        # 13 = the four stage-first panels + the three alert-window panels + the six stand-ins.
        if len(panel_thumbs) < 13:
            panel_thumbs.setdefault(index, cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))
    all_events: list[CandidateEvent] = []
    seen_track_frames: dict[int, int] = {}
    wrist_conf_all: list[float] = []
    wrist_conf_active: list[float] = []

    for idx, pf in enumerate(parsed_frames):
        ok, raw_frame = cap.read()
        if not ok or raw_frame is None:
            break
        canvas = cv2.resize(raw_frame, (w, h), interpolation=cv2.INTER_LINEAR)
        for obs_seen in pf.tracks:
            seen_track_frames[obs_seen.track_id] = seen_track_frames.get(obs_seen.track_id, 0) + 1

        # Run unmodified production state machine
        res = monitor.update(pf)
        all_events.extend(res.candidates)
        if res.candidates and latched_event is None:
            latched_event = res.candidates[0]
            primary_tid = latched_event.track_id
            latched_record = build_candidate_record(latched_event)
            alert_frame_idx = idx
            wanted_panels.add(idx)
            wanted_panels.add(min(total_frames - 1, idx + int(fps)))
            if prev_canvas is not None:
                _keep_panel(idx - 1, prev_canvas)

        _draw_zone_overlay(canvas, camera_zones)

        p_stage = "IDLE"
        p_dwell = 0.0
        p_hold = 0.0
        p_q_state = "OK"
        p_q_conf = 0.0
        p_feat = TrackFeatures(
            track_id=primary_tid,
            pts=pf.pts,
            torso_center=(0.0, 0.0),
            torso_length=1.0,
            ground_point_norm=(0.0, 0.0),
            zone_id=None,
            in_shelf_zone=False,
            left_elbow_deg=180.0,
            right_elbow_deg=180.0,
            min_elbow_deg=180.0,
            active_hand="right",
            wrist_to_body_torsos=99.0,
            body_zone_name="outside",
            target_kind="clothing",
            elbow_flexed_or_medial=False,
            in_at_body_pose=False,
            wrist_excursion_torsos=0.0,
            reach_detected=False,
            active_wrist_conf=0.0,
            wrist_conf_baseline=0.0,
            wrist_hidden_now=False,
        )

        # Sort so primary track draws on top
        tracks_sorted = sorted(pf.tracks, key=lambda o: (o.track_id == primary_tid, o.track_id))
        for obs in tracks_sorted:
            cid = obs.track_id
            sm = monitor._machines.get(cid)
            t_stage = sm.stage if sm is not None else "IDLE"
            hist = list(monitor._histories.get(cid, ()))
            feat = extract_track_features(
                obs=obs,
                history=hist,
                camera_zones=camera_zones,
                shape=pf.shape,
                settings=detector_settings,
                active_hand=sm.active_hand if sm is not None else None,
            )
            trail = wrist_trails.setdefault(cid, deque(maxlen=20))
            wr_idx = R_WRIST if feat.active_hand == "right" else L_WRIST
            if wr_idx < len(obs.kps) and obs.kps[wr_idx][2] >= detector_settings.keypoint_conf_threshold:
                trail.append((int(round(obs.kps[wr_idx][0])), int(round(obs.kps[wr_idx][1]))))

            is_alerted = latched_event is not None and cid == latched_event.track_id
            _draw_track_on_rgb(
                canvas,
                obs=obs,
                feat=feat,
                stage=t_stage,
                is_alerted_track=is_alerted,
                wrist_trail=trail,
                conf_thresh=detector_settings.keypoint_conf_threshold,
                alert_target_label=(
                    f"{latched_event.target_kind}:{latched_event.body_zone_name}"
                    if latched_event is not None
                    else None
                ),
            )

            if cid == primary_tid:
                if wr_idx < len(obs.kps):
                    wrist_conf_all.append(float(obs.kps[wr_idx][2]))
                    if t_stage in ("REACH", "AT_BODY"):
                        wrist_conf_active.append(float(obs.kps[wr_idx][2]))
                p_stage = t_stage
                p_dwell = sm.dwell_accum_s if sm is not None else 0.0
                p_hold = (
                    (pf.pts - sm.at_body_entered_pts)
                    if (sm is not None and sm.at_body_entered_pts is not None and t_stage == "AT_BODY")
                    else (latched_event.hold_duration_s if latched_event is not None else 0.0)
                )
                q_tr = monitor.quality_gate._tracks.get(cid)
                if q_tr is not None and q_tr.last_status is not None:
                    p_q_state = q_tr.last_status.state
                    p_q_conf = q_tr.last_status.core_conf
                p_feat = feat
                if t_stage not in stage_first_frame:
                    stage_first_frame[t_stage] = idx
                    wanted_panels.add(idx)

        _draw_rgb_hud(
            canvas,
            video_name=video_path.name,
            camera=camera,
            frame_idx=idx,
            total_frames=total_frames,
            pts=pf.pts,
            duration_s=duration_s,
            primary_tid=primary_tid,
            stage=p_stage,
            dwell_s=p_dwell,
            hold_s=p_hold,
            feat=p_feat,
            q_state=p_q_state,
            q_conf=p_q_conf,
            candidate_event=latched_event,
            candidate_record=latched_record,
            settings=detector_settings,
        )
        _draw_run_banner(canvas, run_label, overrides or [])

        writer.write(canvas)
        if idx in wanted_panels:
            _keep_panel(idx, canvas)
        prev_canvas = canvas

    cap.release()
    writer.release()

    # Build 6-panel keyframe storyboard (2x3 grid)
    key_indices: list[int] = []
    for st_name in ("IDLE", "DWELL", "REACH", "AT_BODY"):
        if st_name in stage_first_frame:
            key_indices.append(stage_first_frame[st_name])
    if alert_frame_idx is not None:
        if alert_frame_idx > 0 and (alert_frame_idx - 1) not in key_indices:
            key_indices.append(alert_frame_idx - 1)
        key_indices.append(alert_frame_idx)
        post = min(total_frames - 1, alert_frame_idx + int(fps))
        if post not in key_indices:
            key_indices.append(post)

    key_indices = [i for i in sorted(set(key_indices)) if i in panel_thumbs]

    for cand_i in fallback_panels:
        if len(key_indices) >= 6:
            break
        if cand_i not in key_indices and cand_i in panel_thumbs:
            key_indices.append(cand_i)
    key_indices = sorted(key_indices[:6])

    # A clip that never reached a stage we track can still produce fewer than six panels.
    while key_indices and len(key_indices) < 6:
        key_indices.append(key_indices[-1])

    grid = np.zeros((thumb_h * 2, thumb_w * 3, 3), dtype=np.uint8)
    for k_i, f_idx in enumerate(key_indices):
        r = k_i // 3
        c = k_i % 3
        resized = panel_thumbs[f_idx].copy()
        cv2.rectangle(resized, (0, 0), (thumb_w - 1, thumb_h - 1), (95, 95, 105), 1)
        grid[r * thumb_h : (r + 1) * thumb_h, c * thumb_w : (c + 1) * thumb_w] = resized
    cv2.imwrite(str(out_keyframes_png), grid)

    return {
        "video": video_path.name,
        "camera": camera,
        "frames": total_frames,
        "fps": fps,
        "primary_tid": primary_tid,
        "stages_seen": stage_first_frame,
        "alert_emitted": latched_event is not None,
        "all_events": [
            {
                "track_id": ev.track_id,
                "pts": ev.pts,
                "target_kind": ev.target_kind,
                "body_zone_name": ev.body_zone_name,
                "tier": ev.tier,
            }
            for ev in all_events
        ],
        "distinct_person_tracks": len(seen_track_frames),
        "track_frame_counts": dict(sorted(seen_track_frames.items())),
        "primary_wrist_conf_mean": (
            round(float(np.mean(wrist_conf_all)), 3) if wrist_conf_all else None
        ),
        "primary_wrist_conf_mean_reach_or_at_body": (
            round(float(np.mean(wrist_conf_active)), 3) if wrist_conf_active else None
        ),
        "overrides": overrides or [],
        "run_label": run_label,
        "event": (
            {
                "candidate_id": latched_record.get("candidate_id", "") if latched_record else "",
                "pts": latched_event.pts,
                "target_kind": latched_event.target_kind,
                "body_zone_name": latched_event.body_zone_name,
                "tier": latched_event.tier,
                "stage_timeline": [(s.stage, round(s.entered_pts, 2)) for s in latched_event.stage_timeline],
            }
            if latched_event is not None
            else None
        ),
        "out_mp4": str(out_mp4),
        "out_keyframes_png": str(out_keyframes_png),
    }


VIDEOS: tuple[dict, ...] = (
    {
        "key": "gettyimages1",
        "path": Path(r"C:\Users\Maha Mamdouh\Downloads\gettyimages1.mp4"),
        "camera": "getty-cam-1",
        "zone": ZonePolygon(
            id="aisle-1-bottle-shelf",
            points=((0.18, 0.35), (0.68, 0.35), (0.68, 1.00), (0.18, 1.00)),
            facing=(-1.0, 0.0),
        ),
        "tuned_settings": {"short_track_grace_s": 0.0},
        "tuned_relabel_handbag_to_basket": True,
    },
    {
        "key": "gettyimages2",
        "path": Path(r"C:\Users\Maha Mamdouh\Downloads\gettyimages2.mp4"),
        "camera": "getty-cam-2",
        "zone": ZonePolygon(
            id="aisle-2-center-rack",
            points=((0.52, 0.58), (0.72, 0.58), (0.72, 0.82), (0.52, 0.82)),
            facing=(-1.0, 0.0),
        ),
        "tuned_settings": {
            "short_track_grace_s": 0.0,
            "reach_excursion_torsos": 0.38,
            "body_zone_radius_torsos": 0.05,
        },
        "tuned_relabel_handbag_to_basket": False,
    },
)


def _relabel_handbag_to_basket(entries: list[dict]) -> list[dict]:
    """Return a copy of `entries` whose `handbag` detections are relabelled `basket` (store-basket exemption)."""
    out: list[dict] = []
    for e in entries:
        fields = dict(e["fields"])
        rows = json.loads(fields["dets"])
        for r in rows:
            if r[0] == "handbag":
                r[0] = "basket"
        fields["dets"] = json.dumps(rows)
        out.append({"id": e["id"], "fields": fields})
    return out


def _overrides_for(video: dict, config_name: str, defaults: DetectorSettings) -> list[str]:
    """Human-readable list of everything non-default in this run, shown in the HUD and `overrides.json`."""
    lines = [
        f"Zone: hand-drawn '{video['zone'].id}' (deployment input, not from config)",
        "Tracker: ByteTrack (Ultralytics), not DeepStream nvtracker",
    ]
    if config_name == "default":
        lines.append("Thresholds: repo defaults, no overrides")
        return lines
    for key, val in video["tuned_settings"].items():
        lines.append(f"OVERRIDE {key}: {getattr(defaults, key)} -> {val}")
    if video["tuned_relabel_handbag_to_basket"]:
        lines.append("OVERRIDE stream: YOLO 'handbag' relabelled 'basket' (store basket exemption)")
    return lines


def main() -> None:
    import argparse
    from dataclasses import replace

    parser = argparse.ArgumentParser(description="Render RGB concealment demos for real retail videos.")
    parser.add_argument("--pose-model", choices=("yolo11m", "yolo26x", "both"), default="both")
    parser.add_argument("--config", choices=("default", "tuned", "both"), default="both")
    parser.add_argument("--video", choices=("gettyimages1", "gettyimages2", "both"), default="both")
    args = parser.parse_args()

    cfg = load_config()
    defaults = cfg.detector
    models = ("yolo11m", "yolo26x") if args.pose_model == "both" else (args.pose_model,)
    configs = ("default", "tuned") if args.config == "both" else (args.config,)
    videos = [v for v in VIDEOS if args.video in ("both", v["key"])]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(MODELS_DIR)
    os.environ["TORCH_HOME"] = str(MODELS_DIR)
    from ultralytics import YOLO

    print(f"Loading YOLO models from {MODELS_DIR} ...")
    det_model = YOLO(str(MODELS_DIR / "yolo11m.pt"))

    summaries: list[dict] = []
    for model_tag in models:
        pose_model = YOLO(str(MODELS_DIR / f"{model_tag}-pose.pt"))
        for video in videos:
            cache = OUT_DIR / (
                f"cache_{video['key']}_meta.json"
                if model_tag == "yolo11m"
                else f"cache_{model_tag}_{video['key']}_meta.json"
            )
            entries, fps, _ = extract_video_meta_entries(
                video_path=video["path"],
                camera=video["camera"],
                pose_model=pose_model,
                det_model=det_model,
                cache_path=cache,
            )
            zones = CameraZones(camera=video["camera"], zones=(video["zone"],))
            for config_name in configs:
                if config_name == "tuned":
                    settings = replace(defaults, **video["tuned_settings"])
                    run_entries = (
                        _relabel_handbag_to_basket(entries)
                        if video["tuned_relabel_handbag_to_basket"]
                        else entries
                    )
                else:
                    settings = defaults
                    run_entries = entries
                overrides = _overrides_for(video, config_name, defaults)
                stem = f"demo_rgb_{model_tag}_{config_name}_{video['key']}"
                label = f"{video['key']} | {model_tag}-pose + ByteTrack | {config_name.upper()} config"
                summary = render_rgb_video_demo(
                    video_path=video["path"],
                    camera=video["camera"],
                    camera_zones=zones,
                    detector_settings=settings,
                    entries=run_entries,
                    fps=fps,
                    out_mp4=OUT_DIR / f"{stem}.mp4",
                    out_keyframes_png=OUT_DIR / f"{stem}_keyframes.png",
                    run_label=label,
                    overrides=overrides,
                )
                summary["model"] = model_tag
                summary["config"] = config_name
                summaries.append(summary)
                print(
                    f"[{stem}] alerts={[(e['track_id'], e['pts'], e['target_kind']) for e in summary['all_events']]} "
                    f"tracks={summary['distinct_person_tracks']} "
                    f"wrist_conf(reach/at_body)={summary['primary_wrist_conf_mean_reach_or_at_body']}"
                )

    (OUT_DIR / "overrides_and_summary.json").write_text(
        json.dumps(summaries, indent=2, default=str), encoding="utf-8"
    )
    print("Wrote", OUT_DIR / "overrides_and_summary.json")


if __name__ == "__main__":
    main()
