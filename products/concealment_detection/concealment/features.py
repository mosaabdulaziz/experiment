"""Torso-normalized kinematic, body-zone, elbow-flexion, and wrist-confidence feature extraction."""
from __future__ import annotations

from functools import lru_cache
import math

from .types import (
    CameraZones,
    DetectorSettings,
    L_ELBOW,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    R_ELBOW,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
    TrackFeatures,
    TrackObservation,
)
from .zones import find_containing_zone, ground_anchor, reach_alignment


def joint_angle_deg(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    """Return angle ABC at vertex `b` in degrees in `[0.0, 180.0]`."""
    ux, uy = float(a[0] - b[0]), float(a[1] - b[1])
    vx, vy = float(c[0] - b[0]), float(c[1] - b[1])
    nu = math.hypot(ux, uy)
    nv = math.hypot(vx, vy)
    if nu < 1e-6 or nv < 1e-6:
        return 180.0
    cos_val = max(-1.0, min(1.0, (ux * vx + uy * vy) / (nu * nv)))
    return math.degrees(math.acos(cos_val))


@lru_cache(maxsize=65536)
def torso_geometry(
    obs: TrackObservation,
) -> tuple[tuple[float, float], float, float]:
    """Return `(torso_center, torso_length, shoulder_width)` in capture pixels."""
    kps = obs.kps
    x1, y1, x2, y2 = obs.box
    box_h = max(0.0, float(y2 - y1))
    box_w = max(0.0, float(x2 - x1))

    if len(kps) > R_HIP:
        ls = kps[L_SHOULDER]
        rs = kps[R_SHOULDER]
        lh = kps[L_HIP]
        rh = kps[R_HIP]
        sh_conf = 0.5 * (ls[2] + rs[2])
        hp_conf = 0.5 * (lh[2] + rh[2])
        if sh_conf >= 0.30 and hp_conf >= 0.30:
            mid_sh_x = 0.5 * (ls[0] + rs[0])
            mid_sh_y = 0.5 * (ls[1] + rs[1])
            mid_hp_x = 0.5 * (lh[0] + rh[0])
            mid_hp_y = 0.5 * (lh[1] + rh[1])
            center = (0.5 * (mid_sh_x + mid_hp_x), 0.5 * (mid_sh_y + mid_hp_y))
            raw_torso = math.hypot(mid_hp_x - mid_sh_x, mid_hp_y - mid_sh_y)
            if box_h > 0.0 and raw_torso < 0.18 * box_h:
                torso_len = max(0.30 * box_h, 1.0)
            else:
                torso_len = max(raw_torso, 1.0)
            sh_w = max(math.hypot(rs[0] - ls[0], rs[1] - ls[1]), 0.15 * torso_len)
            return (center, torso_len, sh_w)

    center = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
    torso_len = max(0.35 * box_h, 1.0)
    sh_w = max(0.25 * box_w, 0.15 * torso_len)
    return (center, torso_len, sh_w)


def _point_to_box_dist(px: float, py: float, box: tuple[float, float, float, float]) -> float:
    bx1, by1, bx2, by2 = box
    dx = max(bx1 - px, 0.0, px - bx2)
    dy = max(by1 - py, 0.0, py - by2)
    return math.hypot(dx, dy)


def classify_body_zone(
    wrist_xy: tuple[float, float],
    side: str,
    obs: TrackObservation,
    torso_center: tuple[float, float],
    torso_length: float,
    elbow_deg: float,
    settings: DetectorSettings,
) -> tuple[str, str, float, bool]:
    """Return `(body_zone_name, target_kind, dist_torsos, elbow_flexed_or_medial)`."""
    wx, wy = float(wrist_xy[0]), float(wrist_xy[1])
    t_len = max(float(torso_length), 1.0)
    kps = obs.kps

    if len(kps) > R_HIP:
        ls = kps[L_SHOULDER]
        rs = kps[R_SHOULDER]
        lh = kps[L_HIP]
        rh = kps[R_HIP]
        sh_w = max(math.hypot(rs[0] - ls[0], rs[1] - ls[1]), 0.15 * t_len)
        mid_hp = (0.5 * (lh[0] + rh[0]), 0.5 * (lh[1] + rh[1]))
        opp_sh = ls if side == "right" else rs
        waist_targets = [(lh[0], lh[1]), (rh[0], rh[1]), mid_hp]
        chest_targets = [torso_center, (0.5 * (ls[0] + rs[0]), 0.5 * (ls[1] + rs[1]))]
        under_arm_targets = [
            (opp_sh[0], opp_sh[1] + 0.35 * t_len),
            (0.5 * (opp_sh[0] + torso_center[0]), 0.5 * (opp_sh[1] + torso_center[1])),
        ]
    else:
        sh_w = 0.5 * t_len
        waist_targets = [(torso_center[0], torso_center[1] + 0.5 * t_len)]
        chest_targets = [torso_center]
        under_arm_targets = [torso_center]

    medial_cross = abs(wx - torso_center[0]) <= 0.35 * sh_w
    flexed_or_medial = (elbow_deg <= settings.max_conceal_elbow_deg + 1e-4) or medial_cross

    if obs.has_bag:
        if obs.bag_box is not None:
            bag_dist = _point_to_box_dist(wx, wy, obs.bag_box) / t_len
        else:
            bag_anchor = (
                torso_center[0] + (0.45 * t_len if side == "right" else -0.45 * t_len),
                torso_center[1] + 0.35 * t_len,
            )
            bag_dist = math.hypot(wx - bag_anchor[0], wy - bag_anchor[1]) / t_len
        if bag_dist <= settings.bag_zone_radius_torsos:
            bag_flexed = (elbow_deg <= max(settings.max_conceal_elbow_deg, 150.0)) or medial_cross
            return ("personal_bag", "personal_bag", bag_dist, bag_flexed)

    d_waist = min(math.hypot(wx - tx, wy - ty) for tx, ty in waist_targets) / t_len
    d_chest = min(math.hypot(wx - tx, wy - ty) for tx, ty in chest_targets) / t_len
    d_under = min(math.hypot(wx - tx, wy - ty) for tx, ty in under_arm_targets) / t_len

    best_name, best_dist = min(
        (("waist", d_waist), ("chest", d_chest), ("under_arm", d_under)),
        key=lambda item: item[1],
    )
    if obs.cart_boxes:
        cart_dist = min(_point_to_box_dist(wx, wy, cb) for cb in obs.cart_boxes) / t_len
        if cart_dist <= 0.15:
            return ("outside", "clothing", best_dist, False)

    if best_dist <= settings.body_zone_radius_torsos:
        return (best_name, "clothing", best_dist, flexed_or_medial)

    return ("outside", "clothing", best_dist, flexed_or_medial)


@lru_cache(maxsize=65536)
def _valid_hand_geometry(obs: TrackObservation, side: str, threshold: float) -> bool:
    kps = obs.kps
    el_i = L_ELBOW if side == "left" else R_ELBOW
    wr_i = L_WRIST if side == "left" else R_WRIST
    if len(kps) <= wr_i:
        return False
    for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, el_i, wr_i):
        x, y, c = kps[i]
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(c)) or c < threshold:
            return False
    return True


def _hand_kinematics(
    obs: TrackObservation,
    side: str,
    torso_center: tuple[float, float],
    torso_length: float,
    settings: DetectorSettings,
) -> tuple[float, tuple[float, float], float, str, str, float, bool, bool]:
    kps = obs.kps
    if side == "left":
        sh_i, el_i, wr_i = L_SHOULDER, L_ELBOW, L_WRIST
    else:
        sh_i, el_i, wr_i = R_SHOULDER, R_ELBOW, R_WRIST

    if not _valid_hand_geometry(obs, side, settings.keypoint_conf_threshold):
        conf = float(kps[wr_i][2]) if len(kps) > wr_i else 0.0
        conf = conf if math.isfinite(conf) else 0.0
        return (180.0, torso_center, conf, "outside", "clothing", float("inf"), False, False)

    sh = (kps[sh_i][0], kps[sh_i][1])
    el = (kps[el_i][0], kps[el_i][1])
    wr = (kps[wr_i][0], kps[wr_i][1])
    wr_conf = float(kps[wr_i][2])
    elbow_deg = joint_angle_deg(sh, el, wr)

    zone_name, target_kind, dist_torsos, flexed_ok = classify_body_zone(
        wrist_xy=wr,
        side=side,
        obs=obs,
        torso_center=torso_center,
        torso_length=torso_length,
        elbow_deg=elbow_deg,
        settings=settings,
    )
    in_at_body = (zone_name != "outside") and (flexed_ok or not settings.at_body_requires_flex)
    return (elbow_deg, wr, wr_conf, zone_name, target_kind, dist_torsos, flexed_ok, in_at_body)


def _hand_closer_to_body(right_dist: float, left_dist: float) -> bool:
    """Tie-break the active hand on proximity to the body.

    A wrist that is not confidently seen reports an unknown distance; unknown loses the
    tie-break instead of winning it, which is what `inf` subtraction did to a dropped hand.
    """
    right_known = math.isfinite(right_dist)
    left_known = math.isfinite(left_dist)
    if right_known and left_known:
        return right_dist <= left_dist
    return right_known and not left_known


def _torso_speed(
    history: list[tuple[float, TrackObservation, float]],
    pts: float,
    torso_center: tuple[float, float],
    torso_length: float,
) -> float:
    """Torso-center speed over the last second in torso lengths per second; inf if unknown."""
    window = [(t, o) for t, o, _ in history if 0.0 <= pts - t <= 1.0 + 1e-6]
    if not window:
        return float("inf")
    t0, o0 = min(window, key=lambda item: item[0])
    span = pts - t0
    if span < 0.2 - 1e-6:
        return float("inf")
    c0, _, _ = torso_geometry(o0)
    dist = math.hypot(torso_center[0] - c0[0], torso_center[1] - c0[1])
    return dist / max(torso_length, 1.0) / span


def _hands_met(
    history: list[tuple[float, TrackObservation, float]],
    pts: float,
    settings: DetectorSettings,
) -> bool:
    """Whether both wrists were confidently seen together within the reach window."""
    lookback_s = min(settings.reach_window_s, 1.0)
    for h_pts, h_obs, h_tlen in history:
        if not 0.0 <= pts - h_pts <= lookback_s + 1e-6 or len(h_obs.kps) <= R_WRIST:
            continue
        lw, rw = h_obs.kps[L_WRIST], h_obs.kps[R_WRIST]
        if not all(math.isfinite(v) for v in (*lw, *rw)):
            continue
        if min(lw[2], rw[2]) < settings.keypoint_conf_threshold:
            continue
        tl = max(h_tlen if h_tlen > 0 else torso_geometry(h_obs)[1], 1.0)
        if math.hypot(lw[0] - rw[0], lw[1] - rw[1]) / tl <= settings.hand_transfer_meet_torsos:
            return True
    return False


def extract_track_features(
    obs: TrackObservation,
    history: list[tuple[float, TrackObservation, float]],
    camera_zones: CameraZones,
    shape: tuple[int, int],
    settings: DetectorSettings,
    active_hand: str | None = None,
) -> TrackFeatures:
    """Extract features; history includes the current sample at its timestamp.

    An active sequence supplies its locked hand rather than reselecting each frame.
    """
    locked_hand = active_hand
    pts = history[-1][0] if history else 0.0
    torso_center, torso_length, _ = torso_geometry(obs)
    gp = ground_anchor(obs, shape, conf_threshold=settings.keypoint_conf_threshold)
    zone = find_containing_zone(gp, camera_zones)
    in_shelf_zone = zone is not None
    zone_id = zone.id if zone is not None else None

    l_info = _hand_kinematics(obs, "left", torso_center, torso_length, settings)
    r_info = _hand_kinematics(obs, "right", torso_center, torso_length, settings)

    left_elbow_deg = l_info[0]
    right_elbow_deg = r_info[0]
    min_elbow_deg = min(left_elbow_deg, right_elbow_deg)

    def _side_motion(side: str, cur_wr: tuple[float, float], base_in_at_body: bool) -> tuple[float, bool, bool]:
        wr_idx = L_WRIST if side == "left" else R_WRIST
        sh_idx = L_SHOULDER if side == "left" else R_SHOULDER
        cur_valid = _valid_hand_geometry(obs, side, settings.keypoint_conf_threshold)
        cur_rad = math.hypot(cur_wr[0] - torso_center[0], cur_wr[1] - torso_center[1]) / max(torso_length, 1.0)
        if not cur_valid:
            cur_rad = float("inf")
        if not history:
            reach_ok = cur_valid and cur_rad >= (0.65 + settings.reach_excursion_torsos)
            return (cur_rad if cur_valid else 0.0, reach_ok, base_in_at_body)

        min_radial = float("inf")
        max_radial = 0.0
        max_prior_radial = 0.0
        best_delta = (0.0, 0.0)
        # Replace, rather than duplicate, the current timestamp (legacy callers may
        # provide only past observations, with the final timestamp as the current PTS).
        all_candidates = [(t, o, tl) for t, o, tl in history if t < pts]
        all_candidates.append((pts, obs, torso_length))
        lookback_s = max(settings.reach_window_s, 4.0)
        for h_pts, h_obs, h_tlen in all_candidates:
            if (pts - h_pts) > lookback_s + 1e-6:
                continue
            if not _valid_hand_geometry(h_obs, side, settings.keypoint_conf_threshold):
                continue
            h_tc, h_tl, _ = torso_geometry(h_obs)
            tl = max(h_tlen if h_tlen > 0 else h_tl, 1.0)
            wx, wy, wc = h_obs.kps[wr_idx]
            rad = math.hypot(wx - h_tc[0], wy - h_tc[1]) / tl
            if (pts - h_pts) <= settings.reach_window_s + 1e-6:
                if rad < min_radial:
                    min_radial = rad
                if rad > max_radial:
                    max_radial = rad
                    sx, sy, _ = h_obs.kps[sh_idx]
                    best_delta = (wx - sx, wy - sy)
            if (
                (pts - h_pts) >= 0.06
                and wc >= settings.keypoint_conf_threshold
                and rad > max_prior_radial
            ):
                max_prior_radial = rad

        if min_radial == float("inf"):
            return (0.0, False, False)
        excursion = max(0.0, max_radial - min_radial)
        align = reach_alignment(best_delta, zone)
        reach_ok = (excursion >= settings.reach_excursion_torsos) and (align >= -0.25)
        inward_drop = max(0.0, max_prior_radial - cur_rad)
        inward_ok = base_in_at_body and (inward_drop >= settings.inward_min_torsos)
        return (excursion, reach_ok, inward_ok)

    l_exc, l_reach, l_at_body = _side_motion("left", l_info[1], l_info[7])
    r_exc, r_reach, r_at_body = _side_motion("right", r_info[1], r_info[7])

    # Choose active hand: prefer hand that reached and moved inward to body, then reaching hand
    if locked_hand in ("left", "right"):
        active_hand = locked_hand
    elif r_at_body and not l_at_body:
        active_hand = "right"
    elif l_at_body and not r_at_body:
        active_hand = "left"
    elif r_reach and not l_reach:
        active_hand = "right"
    elif l_reach and not r_reach:
        active_hand = "left"
    elif r_exc != l_exc:
        active_hand = "right" if r_exc > l_exc else "left"
    elif _hand_closer_to_body(r_info[5], l_info[5]):
        active_hand = "right"
    else:
        active_hand = "left"

    act = r_info if active_hand == "right" else l_info
    act_in_at_body = r_at_body if active_hand == "right" else l_at_body
    act_exc = r_exc if active_hand == "right" else l_exc
    act_reach = r_reach if active_hand == "right" else l_reach
    wr_idx = R_WRIST if active_hand == "right" else L_WRIST
    other_hand = "left" if active_hand == "right" else "right"
    oth = l_info if other_hand == "left" else r_info

    # Compute wrist confidence baseline over recent baseline_window_s
    confs: list[float] = []
    for h_pts, h_obs, _ in history:
        if not 0.0 <= pts - h_pts <= settings.baseline_window_s + 1e-6:
            continue
        if len(h_obs.kps) > wr_idx:
            c = float(h_obs.kps[wr_idx][2])
            if math.isfinite(c) and c >= settings.keypoint_conf_threshold:
                confs.append(c)
    if confs:
        s_confs = sorted(confs)
        mid = len(s_confs) // 2
        wrist_conf_baseline = (
            s_confs[mid]
            if len(s_confs) % 2 == 1
            else 0.5 * (s_confs[mid - 1] + s_confs[mid])
        )
    else:
        wrist_conf_baseline = max(act[2], settings.keypoint_conf_threshold)

    active_wrist_conf = act[2]
    wrist_hidden_now = (
        active_wrist_conf < settings.keypoint_conf_threshold
        or (
            wrist_conf_baseline >= settings.keypoint_conf_threshold
            and active_wrist_conf <= settings.hidden_conf_drop_fraction * wrist_conf_baseline
        )
    )

    return TrackFeatures(
        track_id=obs.track_id,
        pts=pts,
        torso_center=torso_center,
        torso_length=torso_length,
        ground_point_norm=gp,
        zone_id=zone_id,
        in_shelf_zone=in_shelf_zone,
        left_elbow_deg=left_elbow_deg,
        right_elbow_deg=right_elbow_deg,
        min_elbow_deg=min_elbow_deg,
        active_hand=active_hand,
        wrist_to_body_torsos=act[5],
        body_zone_name=act[3],
        target_kind=act[4],
        elbow_flexed_or_medial=act[6],
        in_at_body_pose=act_in_at_body,
        wrist_excursion_torsos=act_exc,
        reach_detected=act_reach,
        active_wrist_conf=active_wrist_conf,
        wrist_conf_baseline=wrist_conf_baseline,
        wrist_hidden_now=wrist_hidden_now,
        wrist_geometry_valid=_valid_hand_geometry(obs, active_hand, settings.keypoint_conf_threshold),
        at_body_geometry=act[7],
        torso_speed_torsos_s=_torso_speed(history, pts, torso_center, torso_length),
        other_hand_visible=_valid_hand_geometry(obs, other_hand, settings.keypoint_conf_threshold),
        other_hand_in_at_body_pose=l_at_body if other_hand == "left" else r_at_body,
        other_hand_at_body_geometry=oth[7],
        other_hand_reach_detected=l_reach if other_hand == "left" else r_reach,
        hands_met_recent=settings.hand_transfer and _hands_met(history, pts, settings),
    )
