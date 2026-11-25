"""Zero-calibration pinhole ground-plane camera model with dual-anchor projection and exact 2x2 Jacobians."""

from __future__ import annotations

import numpy as np


class PinholeGroundCamera:
    """Zero-calibration pinhole camera mapping 2D boxes to 3D ground-plane poles (X, Z, pole_h, pole_w).

    Coordinate convention (world frame aligned with camera yaw, origin on ground under camera):
      - X: lateral right (meters)
      - Z: horizontal forward depth along ground plane (meters)
      - Y_up: height above ground (meters); ground is Y_up = 0, camera center is at (0, cam_height, 0).
    """

    def __init__(
        self,
        width: int | float = 1920,
        height: int | float = 1080,
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
        cam_height: float = 2.0,
        pitch_deg: float = 8.0,
        min_ray_y: float = 0.02,
        max_depth: float = 80.0,
    ) -> None:
        self.width = float(width)
        self.height = float(height)
        f_default = 0.95 * max(self.width, self.height)
        self.fx = float(fx) if fx is not None else f_default
        self.fy = float(fy) if fy is not None else f_default
        self.cx = float(cx) if cx is not None else 0.5 * self.width
        self.cy = float(cy) if cy is not None else 0.5 * self.height

        self.cam_height = max(float(cam_height), 0.1)
        self.pitch_deg = float(pitch_deg)
        self.pitch_rad = float(np.deg2rad(self.pitch_deg))
        self.sin_pitch = float(np.sin(self.pitch_rad))
        self.cos_pitch = float(np.cos(self.pitch_rad))
        self.min_ray_y = max(float(min_ray_y), 1e-4)
        self.max_depth = max(float(max_depth), 1.0)

    @property
    def v_horizon(self) -> float:
        """Image y-coordinate of the horizontal plane at camera height."""
        if abs(self.cos_pitch) < 1e-6:
            return -1e9
        return self.cy - self.fy * (self.sin_pitch / self.cos_pitch)

    def _pixel_to_world_ray(self, u: float, v: float) -> tuple[float, float, float]:
        """Convert pixel (u, v) to unnormalized ray (rx, ry_down, rz) in horizontal ground-aligned frame."""
        tx = (float(u) - self.cx) / self.fx
        ty = (float(v) - self.cy) / self.fy
        rx = tx
        ry = ty * self.cos_pitch + self.sin_pitch
        rz = -ty * self.sin_pitch + self.cos_pitch
        return rx, ry, rz

    def jacobian_2d_to_3d(self, u: float, v: float, dh: float | None = None) -> np.ndarray:
        """Exact 2x2 analytical Jacobian d(X, Z)/d(u, v) for horizontal plane at vertical drop dh."""
        dh_val = self.cam_height if dh is None else float(dh)
        rx, ry, rz = self._pixel_to_world_ray(u, v)
        rz_safe = max(rz, 1e-4)

        if abs(dh_val) <= 0.05 or dh_val * ry <= 0.0:
            # Near-horizon or sign-mismatched ray: lateral X remains observable, depth Z is ill-conditioned
            ry_safe = max(abs(ry), self.min_ray_y)
            z_approx = min(max(abs(dh_val), 0.1) / ry_safe * rz_safe, self.max_depth)
            return np.array(
                [
                    [z_approx / (rz_safe * self.fx), 0.0],
                    [0.0, 0.0],
                ],
                dtype=np.float64,
            )

        sign_dh = 1.0 if dh_val > 0.0 else -1.0
        ry_floor = max(abs(dh_val) / self.max_depth, 1e-4)
        ry_eff = sign_dh * max(abs(ry), ry_floor)
        z_c = dh_val / ry_eff
        x_val = z_c * rx

        j00 = dh_val / (ry_eff * self.fx)
        j01 = -(x_val * self.cos_pitch) / (ry_eff * self.fy)
        j11 = -dh_val / ((ry_eff ** 2) * self.fy)
        return np.array([[j00, j01], [0.0, j11]], dtype=np.float64)

    def _default_cov_2d(self, w_px: float, h_px: float, conf_scale: float = 1.0) -> np.ndarray:
        sigma_u = max(2.0, 0.008 * float(w_px)) * conf_scale
        sigma_v = max(2.0, 0.006 * float(h_px)) * conf_scale
        return np.diag([sigma_u ** 2, sigma_v ** 2])

    def _intersect_plane(
        self,
        u: float,
        v: float,
        dh: float,
        cov_2d: np.ndarray,
        is_head: bool = False,
        fallback_z: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Intersect pixel ray with horizontal plane at vertical drop dh below camera.

        Returns (pos_3d [X, Z], cov_3d (2x2), was_clamped).
        """
        rx, ry, rz = self._pixel_to_world_ray(u, v)
        rz_safe = max(rz, 1e-4)
        was_clamped = False

        if not is_head:
            # Foot ground plane (dh == cam_height > 0)
            if ry < self.min_ray_y:
                was_clamped = True
                ry_safe = self.min_ray_y
            else:
                ry_safe = ry
            z_c = dh / ry_safe
            z_val = z_c * rz_safe
            if z_val > self.max_depth:
                was_clamped = True
                z_val = self.max_depth
                x_val = z_val * (rx / rz_safe)
            elif z_val < 0.3:
                was_clamped = True
                z_val = 0.3
                x_val = z_val * (rx / rz_safe)
            else:
                x_val = z_c * rx

            j_mat = self.jacobian_2d_to_3d(u, v, dh=dh)
            if ry < self.min_ray_y:
                # Re-evaluate Jacobian at clamped ray so variance stays finite & large
                v_clamped = self.cy + self.fy * ((self.min_ray_y - self.sin_pitch) / max(self.cos_pitch, 1e-4))
                j_mat = self.jacobian_2d_to_3d(u, v_clamped, dh=dh)
            cov_3d = j_mat @ cov_2d @ j_mat.T
            cov_3d = 0.5 * (cov_3d + cov_3d.T)
            if was_clamped:
                cov_3d[1, 1] = max(cov_3d[1, 1], (0.35 * z_val) ** 2)
            cov_3d[0, 0] = max(cov_3d[0, 0], 1e-6)
            cov_3d[1, 1] = max(cov_3d[1, 1], 1e-6)
            return np.array([x_val, z_val], dtype=np.float64), cov_3d, was_clamped

        # Head anchor intersection: handles both cam_height > pole_h (dh > 0) and low-angle cam_height <= pole_h (dh <= 0)
        dh_eps = 0.05
        if abs(dh) <= dh_eps:
            was_clamped = True
            z_val = float(np.clip(fallback_z if fallback_z is not None else min(self.max_depth, 10.0), 0.3, self.max_depth))
            x_val = z_val * (rx / rz_safe)
            j_lat = z_val / (rz_safe * self.fx)
            cov_3d = np.array(
                [
                    [(j_lat ** 2) * cov_2d[0, 0], 0.0],
                    [0.0, max(25.0, (0.5 * z_val) ** 2)],
                ],
                dtype=np.float64,
            )
            return np.array([x_val, z_val], dtype=np.float64), cov_3d, was_clamped

        ry_limit = max(abs(dh) / self.max_depth, 1e-4)
        valid_sign = (dh > 0.0 and ry >= ry_limit) or (dh < 0.0 and ry <= -ry_limit)
        if valid_sign:
            z_c = dh / ry
            z_val = z_c * rz_safe
            if z_val > self.max_depth:
                was_clamped = True
                z_val = self.max_depth
                x_val = z_val * (rx / rz_safe)
            elif z_val < 0.3:
                was_clamped = True
                z_val = 0.3
                x_val = z_val * (rx / rz_safe)
            else:
                x_val = z_c * rx

            j_mat = self.jacobian_2d_to_3d(u, v, dh=dh)
            cov_3d = j_mat @ cov_2d @ j_mat.T
            cov_3d = 0.5 * (cov_3d + cov_3d.T)
            if was_clamped:
                cov_3d[1, 1] = max(cov_3d[1, 1], (0.35 * z_val) ** 2)
        else:
            was_clamped = True
            z_val = float(np.clip(fallback_z if fallback_z is not None else self.max_depth, 0.3, self.max_depth))
            x_val = z_val * (rx / rz_safe)
            j_lat = z_val / (rz_safe * self.fx)
            cov_3d = np.array(
                [
                    [(j_lat ** 2) * cov_2d[0, 0], 0.0],
                    [0.0, max(25.0, (0.5 * z_val) ** 2)],
                ],
                dtype=np.float64,
            )

        cov_3d[0, 0] = max(cov_3d[0, 0], 1e-6)
        cov_3d[1, 1] = max(cov_3d[1, 1], 1e-6)
        return np.array([x_val, z_val], dtype=np.float64), cov_3d, was_clamped

    def foot_to_ground(self, u: float, v: float) -> np.ndarray:
        """Exact ray-ground intersection for foot pixel (u, v) -> [X, Z] in meters."""
        pos_3d, _, _ = self._intersect_plane(u, v, self.cam_height, np.eye(2, dtype=np.float64), is_head=False)
        return pos_3d

    def foot_to_ground_with_cov(
        self,
        u: float,
        v: float,
        cov_2d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ray-ground intersection with 2x2 Jacobian covariance cov_3d = J @ cov_2d @ J.T."""
        pos_3d, cov_3d, _ = self._intersect_plane(u, v, self.cam_height, cov_2d, is_head=False)
        return pos_3d, cov_3d

    def head_to_ground(
        self,
        u_head: float,
        v_head: float,
        pole_h: float,
        fallback_z: float | None = None,
    ) -> np.ndarray:
        """Dual-anchor head ray intersection at height pole_h -> [X, Z].

        Handles both cam_height > pole_h and low-angle cam_height <= pole_h safely
        without negative or infinite depths.
        """
        dh = self.cam_height - float(pole_h)
        pos_3d, _, _ = self._intersect_plane(
            u_head,
            v_head,
            dh,
            np.eye(2, dtype=np.float64),
            is_head=True,
            fallback_z=fallback_z,
        )
        return pos_3d

    def head_to_ground_with_cov(
        self,
        u_head: float,
        v_head: float,
        pole_h: float,
        cov_2d: np.ndarray,
        fallback_z: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dual-anchor head intersection with 2x2 Jacobian covariance."""
        dh = self.cam_height - float(pole_h)
        pos_3d, cov_3d, _ = self._intersect_plane(
            u_head,
            v_head,
            dh,
            cov_2d,
            is_head=True,
            fallback_z=fallback_z,
        )
        return pos_3d, cov_3d

    def head_bbox_to_pole(
        self,
        bbox_xyxy: np.ndarray,
        pole_h: float,
        cov_2d: np.ndarray | None = None,
        fallback_z: float | None = None,
        base_elev: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project a leg-occluded bounding box using head edge y1 and known pole_h.

        Recovers Z from head_to_ground(u_cx, y1, pole_h) and adjusts lateral X to match
        the foot-center projection convention so switching between foot and head anchors
        introduces zero horizontal parallax shift on off-center targets.
        """
        x1, y1, x2, y2 = float(bbox_xyxy[0]), float(bbox_xyxy[1]), float(bbox_xyxy[2]), float(bbox_xyxy[3])
        w_px = max(x2 - x1, 1e-3)
        h_px = max(y2 - y1, 1e-3)
        u_cx = 0.5 * (x1 + x2)
        c2d = self._default_cov_2d(w_px, h_px) if cov_2d is None else np.asarray(cov_2d, dtype=np.float64)

        pos_head, cov_head = self.head_to_ground_with_cov(u_cx, y1, pole_h, c2d, fallback_z=fallback_z)
        z_val = float(pos_head[1])
        z_c_foot = max((self.cam_height - float(base_elev)) * self.sin_pitch + z_val * self.cos_pitch, 0.2)
        x_foot_aligned = z_c_foot * ((u_cx - self.cx) / self.fx)
        return np.array([x_foot_aligned, z_val], dtype=np.float64), cov_head

    def bbox_to_pole(
        self,
        bbox_xyxy: np.ndarray,
        cov_2d: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, float, np.ndarray]:
        """Map [x1, y1, x2, y2] -> (pos_3d [X, Z], pole_h, pole_w, cov_3d (2x2))."""
        x1, y1, x2, y2 = float(bbox_xyxy[0]), float(bbox_xyxy[1]), float(bbox_xyxy[2]), float(bbox_xyxy[3])
        w_px = max(x2 - x1, 1e-3)
        h_px = max(y2 - y1, 1e-3)
        u_cx = 0.5 * (x1 + x2)
        v_foot = y2
        v_head = y1

        c2d = self._default_cov_2d(w_px, h_px) if cov_2d is None else np.asarray(cov_2d, dtype=np.float64)
        pos_3d, cov_3d, _ = self._intersect_plane(u_cx, v_foot, self.cam_height, c2d, is_head=False)
        z_val = float(pos_3d[1])

        _, ry_h, rz_h = self._pixel_to_world_ray(u_cx, v_head)
        rz_h_safe = max(rz_h, 1e-4)
        pole_h = max(self.cam_height - z_val * (ry_h / rz_h_safe), 0.05)

        z_c_mid = max((self.cam_height - 0.5 * pole_h) * self.sin_pitch + z_val * self.cos_pitch, 0.2)
        pole_w = max(w_px * z_c_mid / self.fx, 0.02)
        return pos_3d, float(pole_h), float(pole_w), cov_3d

    def bbox_to_pole_elev(
        self,
        bbox_xyxy: np.ndarray,
        cov_2d: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, float, np.ndarray, float]:
        """Extended bbox_to_pole also returning base_elev (0.0 unless foot hit horizon/depth clamp)."""
        x1, y1, x2, y2 = float(bbox_xyxy[0]), float(bbox_xyxy[1]), float(bbox_xyxy[2]), float(bbox_xyxy[3])
        w_px = max(x2 - x1, 1e-3)
        h_px = max(y2 - y1, 1e-3)
        u_cx = 0.5 * (x1 + x2)
        v_foot = y2
        v_head = y1

        c2d = self._default_cov_2d(w_px, h_px) if cov_2d is None else np.asarray(cov_2d, dtype=np.float64)
        pos_3d, cov_3d, clamped = self._intersect_plane(u_cx, v_foot, self.cam_height, c2d, is_head=False)
        z_val = float(pos_3d[1])

        if clamped:
            _, ry_f, rz_f = self._pixel_to_world_ray(u_cx, v_foot)
            rz_f_safe = max(rz_f, 1e-4)
            base_elev = self.cam_height - z_val * (ry_f / rz_f_safe)
        else:
            base_elev = 0.0

        _, ry_h, rz_h = self._pixel_to_world_ray(u_cx, v_head)
        rz_h_safe = max(rz_h, 1e-4)
        head_elev = self.cam_height - z_val * (ry_h / rz_h_safe)
        pole_h = max(head_elev - base_elev, 0.05)

        z_c_mid = max((self.cam_height - base_elev - 0.5 * pole_h) * self.sin_pitch + z_val * self.cos_pitch, 0.2)
        pole_w = max(w_px * z_c_mid / self.fx, 0.02)
        return pos_3d, float(pole_h), float(pole_w), cov_3d, float(base_elev)

    def bbox_to_pole_batch(
        self,
        bboxes_xyxy: np.ndarray,
        confs: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized batch mapping of (N, 4) boxes -> (pos_3d (N,2), pole_h (N,), pole_w (N,), cov_3d (N,2,2))."""
        boxes = np.asarray(bboxes_xyxy, dtype=np.float64).reshape(-1, 4)
        n = len(boxes)
        if n == 0:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty((0, 2, 2), dtype=np.float64),
            )

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w_px = np.maximum(x2 - x1, 1e-3)
        h_px = np.maximum(y2 - y1, 1e-3)
        u_cx = 0.5 * (x1 + x2)

        tx = (u_cx - self.cx) / self.fx
        ty_f = (y2 - self.cy) / self.fy
        rx = tx
        ry_f = ty_f * self.cos_pitch + self.sin_pitch
        rz_f = np.maximum(-ty_f * self.sin_pitch + self.cos_pitch, 1e-4)

        clamped_ry = ry_f < self.min_ray_y
        ry_safe = np.maximum(ry_f, self.min_ray_y)
        z_c = self.cam_height / ry_safe
        z_val = z_c * rz_f

        over_max = z_val > self.max_depth
        under_min = z_val < 0.3
        clamped_any = clamped_ry | over_max | under_min
        z_val = np.clip(z_val, 0.3, self.max_depth)
        x_val = np.where(over_max | under_min, z_val * (rx / rz_f), z_c * rx)
        pos_3d = np.column_stack([x_val, z_val])

        # Vectorized 2x2 Jacobian J
        j00 = self.cam_height / (ry_safe * self.fx)
        j01 = -(x_val * self.cos_pitch) / (ry_safe * self.fy)
        j11 = -self.cam_height / ((ry_safe ** 2) * self.fy)

        if confs is not None:
            conf_scale = 1.0 + np.maximum(0.0, 1.0 - np.asarray(confs, dtype=np.float64)) * 0.8
        else:
            conf_scale = np.ones(n, dtype=np.float64)
        var_u = (np.maximum(2.0, 0.008 * w_px) * conf_scale) ** 2
        var_v = (np.maximum(2.0, 0.006 * h_px) * conf_scale) ** 2

        c00 = (j00 ** 2) * var_u + (j01 ** 2) * var_v
        c01 = j01 * j11 * var_v
        c11 = (j11 ** 2) * var_v
        c11 = np.where(clamped_any, np.maximum(c11, (0.35 * z_val) ** 2), c11)
        c00 = np.maximum(c00, 1e-6)
        c11 = np.maximum(c11, 1e-6)

        cov_3d = np.zeros((n, 2, 2), dtype=np.float64)
        cov_3d[:, 0, 0] = c00
        cov_3d[:, 0, 1] = c01
        cov_3d[:, 1, 0] = c01
        cov_3d[:, 1, 1] = c11

        # pole_h and pole_w
        ty_h = (y1 - self.cy) / self.fy
        ry_h = ty_h * self.cos_pitch + self.sin_pitch
        rz_h = np.maximum(-ty_h * self.sin_pitch + self.cos_pitch, 1e-4)
        pole_h = np.maximum(self.cam_height - z_val * (ry_h / rz_h), 0.05)

        z_c_mid = np.maximum((self.cam_height - 0.5 * pole_h) * self.sin_pitch + z_val * self.cos_pitch, 0.2)
        pole_w = np.maximum(w_px * z_c_mid / self.fx, 0.02)

        return pos_3d, pole_h, pole_w, cov_3d

    def project_pole_to_bbox(
        self,
        pos_3d: np.ndarray,
        pole_h: float,
        pole_w: float,
        base_elev: float = 0.0,
    ) -> np.ndarray:
        """Project 3D pole (pos_3d [X, Z], pole_h, pole_w) -> [x1, y1, x2, y2] in pixels."""
        x_val = float(pos_3d[0])
        z_val = max(float(pos_3d[1]), 0.2)
        ph = max(float(pole_h), 0.02)
        pw = max(float(pole_w), 0.01)
        elev = float(base_elev)

        dh_foot = self.cam_height - elev
        z_c_foot = max(dh_foot * self.sin_pitch + z_val * self.cos_pitch, 0.2)
        u_cx = self.fx * (x_val / z_c_foot) + self.cx
        v_foot = self.fy * ((dh_foot * self.cos_pitch - z_val * self.sin_pitch) / z_c_foot) + self.cy

        dh_head = self.cam_height - elev - ph
        z_c_head = max(dh_head * self.sin_pitch + z_val * self.cos_pitch, 0.2)
        v_head = self.fy * ((dh_head * self.cos_pitch - z_val * self.sin_pitch) / z_c_head) + self.cy

        z_c_mid = max((dh_foot - 0.5 * ph) * self.sin_pitch + z_val * self.cos_pitch, 0.2)
        w_px = self.fx * (pw / z_c_mid)

        return np.array(
            [u_cx - 0.5 * w_px, v_head, u_cx + 0.5 * w_px, v_foot],
            dtype=np.float64,
        )

    def auto_pitch_for_bboxes(
        self,
        bboxes_xyxy: np.ndarray,
        min_margin_ray_y: float = 0.05,
        max_pitch_deg: float = 35.0,
    ) -> "PinholeGroundCamera":
        """Return a new PinholeGroundCamera with steeper pitch only if boxes sit above current horizon."""
        boxes = np.asarray(bboxes_xyxy, dtype=np.float64).reshape(-1, 4)
        if len(boxes) == 0:
            return self
        min_y2 = float(np.min(boxes[:, 3]))
        ty_min = (min_y2 - self.cy) / self.fy
        ry_min = ty_min * self.cos_pitch + self.sin_pitch
        if ry_min >= self.min_ray_y * 1.5:
            return self

        # Solve ty_min * cos(theta) + sin(theta) = min_margin_ray_y
        # Using sqrt(ty_min^2 + 1) * sin(theta + atan2(ty_min, 1)) = min_margin_ray_y
        amp = float(np.hypot(ty_min, 1.0))
        ratio = float(np.clip(min_margin_ray_y / max(amp, 1e-6), -0.999, 0.999))
        needed_rad = float(np.arcsin(ratio) - np.arctan2(ty_min, 1.0))
        needed_deg = float(np.clip(np.rad2deg(needed_rad), self.pitch_deg, max_pitch_deg))
        if needed_deg <= self.pitch_deg + 0.5:
            return self

        return PinholeGroundCamera(
            width=self.width,
            height=self.height,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            cam_height=self.cam_height,
            pitch_deg=needed_deg,
            min_ray_y=self.min_ray_y,
            max_depth=self.max_depth,
        )
