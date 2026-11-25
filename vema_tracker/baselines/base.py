"""Unified BaseTracker interface for fair SOTA tracker benchmarking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """Single 2D detection in [x1, y1, x2, y2] pixel coordinates."""

    bbox: np.ndarray
    conf: float
    cls_id: int = 0

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=np.float64).reshape(4)
        self.conf = float(self.conf)
        self.cls_id = int(self.cls_id)


@dataclass
class TrackOutput:
    """Single tracker output box in [x1, y1, x2, y2] pixel coordinates."""

    track_id: int
    bbox: np.ndarray
    conf: float
    cls_id: int = 0
    is_shadow: bool = False

    def __post_init__(self) -> None:
        self.track_id = int(self.track_id)
        self.bbox = np.asarray(self.bbox, dtype=np.float64).reshape(4)
        self.conf = float(self.conf)
        self.cls_id = int(self.cls_id)


class BaseTracker(ABC):
    """Abstract adapter implemented by VEMA and all SOTA baseline trackers."""

    name: str = "base"

    @abstractmethod
    def reset(
        self,
        fps: float = 30.0,
        img_size: tuple[int, int] | None = None,
    ) -> None:
        """Reset tracker state before starting a new sequence."""

    @abstractmethod
    def update(
        self,
        detections: list[Detection],
        dt: float | None = None,
    ) -> list[TrackOutput]:
        """Process one frame of detections and return active tracks."""
