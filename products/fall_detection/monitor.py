"""The per-camera monitor: the vendor's FallMonitor, wrapped for this product.

Three things the vendor's copy does not do are done here:

* **the frame's own shape** -- the classifier's tensors are built in it, and only the frame
  knows it (design §5.2);
* **the vendor's config blocks the policy scores against** -- `path_b`, `path_c`, `lying` and
  `immobility` come from the copied `config.yaml`, never from a second copy of the thresholds
  (design §4.1: one source of truth per quantity);
* **the fall paths' soft scores** -- the copy carries which paths latched, and the product's
  policy turns that latch into the scores (design §4.2), so the copy stays self-contained.

The seam itself -- the candidate's `reading` and `fall_paths` -- lives in the copy
(`fall_detection/fall_monitor.py`, see docs/PROVENANCE.md), because the vendor's packaging is the
only place the reading and the latch exist at the same moment.
"""
from __future__ import annotations

import mmaction_compat  # noqa: F401  (installs the import shim before the copy is loaded)
import scoring
from fall_detection.fall_monitor import CFG, FallMonitor

# The blocks the policy scores against: the collapse gates and the lying/immobility thresholds.
VENDOR_BLOCKS = ("path_b", "path_c", "lying", "immobility")


def vendor_blocks(cfg: dict | None = None) -> dict:
    """The vendor's own blocks, as the policy reads them."""
    cfg = CFG if cfg is None else cfg
    return {name: cfg[name] for name in VENDOR_BLOCKS}


def attach_path_scores(candidates: list[dict], vendor_cfg: dict) -> list[dict]:
    """The fall candidate's path scores, from the latch the copy carries (design §4.2)."""
    for rec in candidates:
        if rec.get("type") == "fall":
            rec["path_scores"] = scoring.fall_path_scores(
                rec.get("fall_paths"), rec["evidence"]["kinematics"], vendor_cfg)
    return candidates


class CameraMonitor:
    """One camera's monitor: the vendor's FallMonitor plus the frame's own shape."""

    def __init__(self, classifier, cfg: dict | None = None, device: str = "cuda:0"):
        self.monitor = FallMonitor(cfg=cfg, classifier=classifier, device=device)
        self._vendor_cfg = vendor_blocks(cfg)

    def set_shape(self, shape) -> None:
        """The frame's own shape, as `[capture_w, capture_h]` -> the monitor's `(h, w)`.

        The shape arrives with every frame and changes when the camera does; the monitor reads
        it when it builds the classifier's tensors (design §5.2).
        """
        if not shape or len(shape) < 2:
            return
        self.monitor.img_shape = (int(shape[1]), int(shape[0]))

    def update(self, t, ids, boxes, kps):
        candidates = self.monitor.update(t, ids, boxes, kps) or []
        return attach_path_scores(candidates, self._vendor_cfg)


def build_monitor(classifier, cfg: dict | None = None, device: str = "cuda:0") -> CameraMonitor:
    """The monitor of one camera.

    The tracks are keyed by the track id alone, so each camera gets its own (design §5.1).
    """
    return CameraMonitor(classifier=classifier, cfg=cfg, device=device)
