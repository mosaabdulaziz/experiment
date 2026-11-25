"""Read-only viewpoint diagnosis over cached `iris:meta` streams (no production code is touched).

Question: why did the overhead clip (`gettyimages2`) fire clothing/at-body evidence during normal browsing?
Two competing hypotheses:
  H1 torso-scale collapse -- vertical torso `T` shrinks under camera pitch, distorting torso-normalized distances.
  H2 2D projection overlap -- the hand projects onto the torso silhouette from above without touching the body.

Per primary track this reports shoulder-width/torso ratio, wrist distance in px and in `T`, whether the wrist
projects inside the shoulder-hip quad, and how often `in_at_body_pose` is true in IDLE/DWELL frames under the
repo's default settings. The eye-level clip (`gettyimages1`) is the control.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

CAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CAL_DIR))

from render_rgb_demo import OUT_DIR, VIDEOS  # noqa: E402

import acl  # noqa: E402
from concealment.features import extract_track_features, torso_geometry  # noqa: E402
from concealment.state_machine import ConcealmentMonitor  # noqa: E402
from concealment.types import (  # noqa: E402
    CameraZones,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
)
from config import load_config  # noqa: E402

CONF = 0.30


def _pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 3) if values else None


def diagnose(video: dict, cache_path: Path) -> dict:
    settings = load_config().detector
    camera = video["camera"]
    zones = CameraZones(camera=camera, zones=(video["zone"],))
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    frames = [acl.parse_pose_frame(camera=camera, entry=e) for e in data["entries"]]

    counts: dict[int, int] = {}
    for pf in frames:
        for obs in pf.tracks:
            counts[obs.track_id] = counts.get(obs.track_id, 0) + 1
    primary = max(counts, key=counts.get)

    monitor = ConcealmentMonitor(camera=camera, settings=settings, camera_zones=zones)
    ratios: list[float] = []
    torso_px: list[float] = []
    shoulder_px: list[float] = []
    wrist_px_T: list[float] = []
    inside_quad = 0
    quad_frames = 0
    idle_frames = 0
    idle_at_body = 0
    idle_at_body_inside_quad = 0
    all_frames = 0

    for pf in frames:
        monitor.update(pf)
        for obs in pf.tracks:
            if obs.track_id != primary:
                continue
            k = obs.kps
            if min(k[i][2] for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)) < CONF:
                continue
            all_frames += 1
            _, tl, _ = torso_geometry(obs)
            sw = math.hypot(k[L_SHOULDER][0] - k[R_SHOULDER][0], k[L_SHOULDER][1] - k[R_SHOULDER][1])
            torso_px.append(tl)
            shoulder_px.append(sw)
            ratios.append(sw / max(tl, 1e-6))

            quad = np.array(
                [k[L_SHOULDER][:2], k[R_SHOULDER][:2], k[R_HIP][:2], k[L_HIP][:2]], dtype=np.float32
            )
            hist = list(monitor._histories.get(primary, ()))
            feat = extract_track_features(obs, hist, zones, pf.shape, settings)
            wr = R_WRIST if feat.active_hand == "right" else L_WRIST
            wx, wy, wc = k[wr]
            in_quad = False
            if wc >= CONF:
                quad_frames += 1
                in_quad = cv2.pointPolygonTest(quad.reshape(-1, 1, 2), (float(wx), float(wy)), False) >= 0
                inside_quad += int(in_quad)
                wrist_px_T.append(feat.wrist_to_body_torsos)

            sm = monitor._machines.get(primary)
            if sm is not None and sm.stage in ("IDLE", "DWELL"):
                idle_frames += 1
                if feat.in_at_body_pose:
                    idle_at_body += 1
                    idle_at_body_inside_quad += int(in_quad)

    return {
        "video": video["key"],
        "primary_track": primary,
        "frames_used": all_frames,
        "shoulder_to_torso_ratio": {
            "p10": _pct(ratios, 10),
            "median": _pct(ratios, 50),
            "p90": _pct(ratios, 90),
        },
        "torso_length_px": {"p10": _pct(torso_px, 10), "median": _pct(torso_px, 50), "p90": _pct(torso_px, 90)},
        "shoulder_width_px": {
            "p10": _pct(shoulder_px, 10),
            "median": _pct(shoulder_px, 50),
            "p90": _pct(shoulder_px, 90),
        },
        "wrist_to_body_in_T": {"p10": _pct(wrist_px_T, 10), "median": _pct(wrist_px_T, 50), "p90": _pct(wrist_px_T, 90)},
        "wrist_projects_inside_torso_quad_frac": round(inside_quad / quad_frames, 3) if quad_frames else None,
        "idle_or_dwell_frames": idle_frames,
        "idle_or_dwell_at_body_frac_default_settings": round(idle_at_body / idle_frames, 3) if idle_frames else None,
        "idle_or_dwell_at_body_and_inside_quad_frac": (
            round(idle_at_body_inside_quad / idle_at_body, 3) if idle_at_body else None
        ),
    }


def main() -> None:
    results = []
    for video in VIDEOS:
        cache = OUT_DIR / f"cache_{video['key']}_meta.json"
        results.append(diagnose(video, cache))
    out = OUT_DIR / "viewpoint_diagnosis.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
