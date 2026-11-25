"""Shelf-front polygon geometry, ground-anchor projection, and dwell-heatmap zone proposal."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import math

from .types import (
    CameraZones,
    L_ANKLE,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    MAX_OBSERVATION_GAP_S,
    PoseFrame,
    R_ANKLE,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
    TrackObservation,
    ZonePolygon,
)


def _point_on_segment(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    eps: float = 1e-9,
) -> bool:
    dx = x2 - x1
    dy = y2 - y1
    cross = abs((px - x1) * dy - (py - y1) * dx)
    if cross > eps * max(1.0, math.hypot(dx, dy)):
        return False
    return (
        min(x1, x2) - eps <= px <= max(x1, x2) + eps
        and min(y1, y2) - eps <= py <= max(y1, y2) + eps
    )


def point_in_polygon(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    """Return True if normalized `point` lies inside or on the boundary of `polygon`."""
    n = len(polygon)
    if n < 3:
        return False
    px, py = float(point[0]), float(point[1])
    inside = False
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        if _point_on_segment(px, py, x1, y1, x2, y2):
            return True
        if (y1 > py) != (y2 > py):
            x_intersect = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px <= x_intersect:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def ground_anchor(
    obs: TrackObservation,
    shape: tuple[int, int],
    conf_threshold: float = 0.30,
) -> tuple[float, float]:
    """Return normalized floor anchor `(gx, gy)` in `[0.0, 1.0]` for a tracked person."""
    width, height = int(shape[0]), int(shape[1])
    w_denom = float(max(width, 1))
    h_denom = float(max(height, 1))
    kps = obs.kps
    box_x1, _, box_x2, box_y2 = obs.box

    ankles = []
    if len(kps) > R_ANKLE:
        for idx in (L_ANKLE, R_ANKLE):
            kx, ky, kc = kps[idx]
            if kc >= conf_threshold:
                ankles.append((kx, ky))
    if ankles:
        avg_x = sum(p[0] for p in ankles) / len(ankles)
        avg_y = sum(p[1] for p in ankles) / len(ankles)
        return (_clamp01(avg_x / w_denom), _clamp01(avg_y / h_denom))

    hips = []
    if len(kps) > R_HIP:
        for idx in (L_HIP, R_HIP):
            kx, ky, kc = kps[idx]
            if kc >= conf_threshold:
                hips.append((kx, ky))
    if hips:
        hip_x = sum(p[0] for p in hips) / len(hips)
        return (_clamp01(hip_x / w_denom), _clamp01(box_y2 / h_denom))

    mid_x = 0.5 * (box_x1 + box_x2)
    return (_clamp01(mid_x / w_denom), _clamp01(box_y2 / h_denom))


_ZONE_FREE_POLYGON = ZonePolygon(
    id="zone_free",
    points=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    facing=None,
)


def find_containing_zone(
    point: tuple[float, float],
    camera_zones: CameraZones | None,
) -> ZonePolygon | None:
    """Return the first shelf zone containing normalized `point`, or the zone-free fallback."""
    if camera_zones is None or camera_zones.zone_free:
        return _ZONE_FREE_POLYGON
    for zone in camera_zones.zones:
        if point_in_polygon(point, zone.points):
            return zone
    return None


def reach_alignment(
    wrist_delta: tuple[float, float],
    zone: ZonePolygon | None,
) -> float:
    """Return cosine alignment in `[-1.0, 1.0]` with `zone.facing`, or `1.0` if unconstrained."""
    if zone is None or zone.facing is None:
        return 1.0
    dx, dy = float(wrist_delta[0]), float(wrist_delta[1])
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return 0.0
    fx, fy = zone.facing
    fnorm = math.hypot(fx, fy)
    if fnorm < 1e-6:
        return 1.0
    cos_val = (dx * fx + dy * fy) / (norm * fnorm)
    return max(-1.0, min(1.0, cos_val))


def load_zones(raw: dict) -> dict[str, CameraZones]:
    """Validate and parse raw zone configuration dict into `{camera_id: CameraZones}`."""
    if not isinstance(raw, dict):
        raise ValueError("Zone configuration must be a mapping")
    cameras_raw = raw.get("cameras", {})
    if not isinstance(cameras_raw, dict):
        raise ValueError("'cameras' in zone configuration must be a mapping")

    result: dict[str, CameraZones] = {}
    for cam_id, cam_cfg in cameras_raw.items():
        if not isinstance(cam_cfg, dict):
            raise ValueError(f"Camera zone config for '{cam_id}' must be a mapping")
        zone_free_raw = cam_cfg.get("zone_free", False)
        if not isinstance(zone_free_raw, bool):
            raise ValueError(
                f"'zone_free' for camera '{cam_id}' must be a boolean, got {zone_free_raw!r} "
                "(the string 'false' is truthy)"
            )
        zone_free = zone_free_raw
        zones_raw = cam_cfg.get("zones")
        if zones_raw is None:
            zones_raw = cam_cfg.get("shelf_front") or []
        if not isinstance(zones_raw, list):
            raise ValueError(f"'zones' for camera '{cam_id}' must be a list")

        polygons: list[ZonePolygon] = []
        for idx, z in enumerate(zones_raw):
            if not isinstance(z, dict):
                raise ValueError(f"Zone #{idx} for camera '{cam_id}' must be a mapping")
            zone_id = str(z.get("id") or f"zone_{idx + 1}")
            pts_raw = z.get("points") if "points" in z else z.get("polygon")
            if not isinstance(pts_raw, (list, tuple)) or len(pts_raw) < 3:
                raise ValueError(
                    f"Zone '{zone_id}' in camera '{cam_id}' must have at least 3 polygon vertices"
                )
            pts: list[tuple[float, float]] = []
            for v in pts_raw:
                if not isinstance(v, (list, tuple)) or len(v) != 2:
                    raise ValueError(f"Vertex in zone '{zone_id}' must be a 2-element coordinate")
                vx, vy = float(v[0]), float(v[1])
                if not (0.0 <= vx <= 1.0 and 0.0 <= vy <= 1.0):
                    raise ValueError(
                        f"Vertex ({vx}, {vy}) in zone '{zone_id}' must be normalized in [0.0, 1.0]"
                    )
                pts.append((vx, vy))

            facing_raw = z.get("facing")
            facing: tuple[float, float] | None = None
            if facing_raw is not None:
                if not isinstance(facing_raw, (list, tuple)) or len(facing_raw) != 2:
                    raise ValueError(f"'facing' in zone '{zone_id}' must be [dx, dy]")
                fx, fy = float(facing_raw[0]), float(facing_raw[1])
                fnorm = math.hypot(fx, fy)
                if fnorm <= 1e-6:
                    raise ValueError(f"'facing' in zone '{zone_id}' cannot be zero length")
                facing = (fx / fnorm, fy / fnorm)
            kind = str(z.get("kind") or "shelf_front")
            polygons.append(
                ZonePolygon(id=zone_id, points=tuple(pts), facing=facing, kind=kind)
            )

        result[str(cam_id)] = CameraZones(
            camera=str(cam_id),
            zone_free=zone_free,
            zones=tuple(polygons),
        )
    return result


def propose_zones_from_dwell(
    frames: Iterable[PoseFrame],
    grid_size: int = 10,
    min_dwell_s: float = 2.0,
    top_k: int = 2,
) -> CameraZones:
    """Propose up to `top_k` rectangular shelf-front zones from stationary floor dwell on normal frames."""
    g = max(2, int(grid_size))
    cell_dwell: dict[tuple[int, int], float] = defaultdict(float)
    cell_reach_vec: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    last_track_state: dict[int, tuple[float, float, float]] = {}
    camera_name = "cam-1"
    watermark_pts: float | None = None
    anchor_shape: tuple[int, int] | None = None

    for frame in frames:
        camera_name = frame.camera
        # Dwell is only creditable between comparable observations: a frame the model did not
        # see, a clock that went backwards, or a new capture size makes the previous anchors not
        # the same measurement, so the streak restarts instead of accumulating across them.
        comparable = (
            frame.observed
            and (watermark_pts is None or frame.pts > watermark_pts)
            and (anchor_shape is None or frame.shape == anchor_shape)
        )
        if not comparable:
            last_track_state.clear()
            continue
        watermark_pts, anchor_shape = frame.pts, frame.shape
        seen_ids: set[int] = set()
        for obs in frame.tracks:
            seen_ids.add(obs.track_id)
            gx, gy = ground_anchor(obs, frame.shape)
            prev = last_track_state.get(obs.track_id)
            if prev is not None:
                t0, prev_gx, prev_gy = prev
                dt = frame.pts - t0
                if 0.0 < dt <= MAX_OBSERVATION_GAP_S:
                    speed = math.hypot(gx - prev_gx, gy - prev_gy) / dt
                    if speed <= 0.08:
                        ci = min(g - 1, max(0, int(gx * g)))
                        cj = min(g - 1, max(0, int(gy * g)))
                        cell_dwell[(ci, cj)] += dt
                        kps = obs.kps
                        if len(kps) > R_HIP:
                            ls, rs, lh, rh = kps[L_SHOULDER], kps[R_SHOULDER], kps[L_HIP], kps[R_HIP]
                            t_len = max(
                                math.hypot(
                                    0.5 * (lh[0] + rh[0]) - 0.5 * (ls[0] + rs[0]),
                                    0.5 * (lh[1] + rh[1]) - 0.5 * (ls[1] + rs[1]),
                                ),
                                25.0,
                            )
                            for sh_i, wr_i in ((L_SHOULDER, L_WRIST), (R_SHOULDER, R_WRIST)):
                                sx, sy, sc = kps[sh_i]
                                wx, wy, wc = kps[wr_i]
                                if sc >= 0.30 and wc >= 0.30:
                                    dx, dy = wx - sx, wy - sy
                                    ext = math.hypot(dx, dy)
                                    # Outward/elevated reach toward shelf (excludes resting straight arms hanging straight down)
                                    if ext >= 0.70 * t_len and dy <= 0.55 * t_len:
                                        acc = cell_reach_vec[(ci, cj)]
                                        acc[0] += dx / ext
                                        acc[1] += dy / ext
                                        acc[2] += 1.0
            last_track_state[obs.track_id] = (frame.pts, gx, gy)
        # A track absent from a frame it should have been in is not observed standing there;
        # its streak ends rather than resuming with credit for the interval it vanished.
        for lost_id in set(last_track_state) - seen_ids:
            del last_track_state[lost_id]

    qualifying = {cell: dwell for cell, dwell in cell_dwell.items() if dwell >= min_dwell_s}
    if not qualifying:
        return CameraZones(camera=camera_name, zone_free=True, zones=())

    visited: set[tuple[int, int]] = set()
    clusters: list[tuple[float, list[tuple[int, int]]]] = []
    for cell in qualifying:
        if cell in visited:
            continue
        queue = [cell]
        visited.add(cell)
        comp: list[tuple[int, int]] = []
        total_dwell = 0.0
        while queue:
            curr = queue.pop()
            comp.append(curr)
            total_dwell += qualifying[curr]
            cx, cy = curr
            for nx in (cx - 1, cx, cx + 1):
                for ny in (cy - 1, cy, cy + 1):
                    nb = (nx, ny)
                    if nb in qualifying and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
        clusters.append((total_dwell, comp))

    clusters.sort(key=lambda item: item[0], reverse=True)
    polygons: list[ZonePolygon] = []
    pad = 0.25 / g
    for idx, (_, comp) in enumerate(clusters[: max(1, int(top_k))]):
        min_i = min(c[0] for c in comp)
        max_i = max(c[0] for c in comp)
        min_j = min(c[1] for c in comp)
        max_j = max(c[1] for c in comp)
        x1 = _clamp01((min_i / g) - pad)
        y1 = _clamp01((min_j / g) - pad)
        x2 = _clamp01(((max_i + 1) / g) + pad)
        y2 = _clamp01(((max_j + 1) / g) + pad)

        sum_vx = sum(cell_reach_vec[c][0] for c in comp)
        sum_vy = sum(cell_reach_vec[c][1] for c in comp)
        sum_cnt = sum(cell_reach_vec[c][2] for c in comp)
        v_norm = math.hypot(sum_vx, sum_vy)
        facing: tuple[float, float] | None = None
        if sum_cnt >= 5.0 and v_norm > 1e-3:
            facing = (round(sum_vx / v_norm, 4), round(sum_vy / v_norm, 4))

        polygons.append(
            ZonePolygon(
                id=f"proposed_shelf_{idx + 1}",
                points=((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
                facing=facing,
            )
        )

    return CameraZones(camera=camera_name, zone_free=False, zones=tuple(polygons))
