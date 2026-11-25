"""Shared 8D constant-velocity Kalman filter for SORT / ByteTrack / OC-SORT baselines."""

from __future__ import annotations

import numpy as np


def xyxy_to_xyah(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] -> [cx, cy, aspect_ratio, h]."""
    w = max(float(bbox[2] - bbox[0]), 1e-3)
    h = max(float(bbox[3] - bbox[1]), 1e-3)
    cx = float(bbox[0]) + 0.5 * w
    cy = float(bbox[1]) + 0.5 * h
    return np.array([cx, cy, w / h, h], dtype=np.float64)


def xyah_to_xyxy(xyah: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, aspect_ratio, h] -> [x1, y1, x2, y2]."""
    cx, cy, a, h = float(xyah[0]), float(xyah[1]), float(xyah[2]), max(float(xyah[3]), 1e-3)
    w = max(a * h, 1e-3)
    return np.array(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h],
        dtype=np.float64,
    )


class KalmanFilterXYAH:
    """Standard 8-state Kalman filter [cx, cy, a, h, vx, vy, va, vh] in pure NumPy."""

    def __init__(self) -> None:
        ndim = 4
        dt = 1.0
        self._motion_mat = np.eye(2 * ndim, dtype=np.float64)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float64)
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean_pos = np.asarray(measurement, dtype=np.float64)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        h = max(float(measurement[3]), 1.0)
        std = np.array(
            [
                2.0 * self._std_weight_position * h,
                2.0 * self._std_weight_position * h,
                1e-2,
                2.0 * self._std_weight_position * h,
                10.0 * self._std_weight_velocity * h,
                10.0 * self._std_weight_velocity * h,
                1e-5,
                10.0 * self._std_weight_velocity * h,
            ],
            dtype=np.float64,
        )
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        dt_steps: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        h = max(float(mean[3]), 1.0)
        scale = max(float(dt_steps), 1e-3)
        motion_mat = self._motion_mat.copy()
        for i in range(4):
            motion_mat[i, 4 + i] = scale

        std_pos = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-2,
                self._std_weight_position * h,
            ],
            dtype=np.float64,
        ) * scale
        std_vel = np.array(
            [
                self._std_weight_velocity * h,
                self._std_weight_velocity * h,
                1e-5,
                self._std_weight_velocity * h,
            ],
            dtype=np.float64,
        ) * scale
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = motion_mat @ mean
        covariance = motion_mat @ covariance @ motion_mat.T + motion_cov
        return mean, covariance

    def project(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        h = max(float(mean[3]), 1.0)
        std = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-1,
                self._std_weight_position * h,
            ],
            dtype=np.float64,
        )
        innovation_cov = np.diag(np.square(std))
        proj_mean = self._update_mat @ mean
        proj_cov = self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        return proj_mean, proj_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        proj_mean, proj_cov = self.project(mean, covariance)
        # Solve proj_cov @ K^T = (covariance @ update_mat^T)^T = update_mat @ covariance
        kalman_gain = np.linalg.solve(proj_cov.T, (covariance @ self._update_mat.T).T).T
        innovation = np.asarray(measurement, dtype=np.float64) - proj_mean
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ proj_cov @ kalman_gain.T
        return new_mean, new_covariance
