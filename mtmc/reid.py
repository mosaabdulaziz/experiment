"""Vehicle Re-ID embeddings via fast-reid (VeRi-trained SBS R50-ibn).

The fast-reid checkout has no setup.py, so we add its root to sys.path and use
its DefaultPredictor directly. Model config and checkpoint come from
PipelineConfig / config.py; the checkpoint must exist at REID_CHECKPOINT.
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import FASTREID_ROOT, REID_CHECKPOINT, REID_CONFIG
from .config import PipelineConfig
from .tracker import Tracklet


class VehicleReID:
    """Lazy-loaded fast-reid feature extractor.

    build() is idempotent; the model is constructed on first use so the module
    imports without torch/fast-reid (keeps the rest of the repo importable).
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._model = None
        self._size_test = None

    # -- construction ------------------------------------------------------
    def build(self):
        if self._model is not None:
            return self
        checkpoint = Path(REID_CHECKPOINT)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"ReID checkpoint not found: {checkpoint}\n"
                "Download it with:\n"
                "  curl -L -o <checkpoint> "
                "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth"
            )
        if not (FASTREID_ROOT / REID_CONFIG).exists():
            raise FileNotFoundError(f"fast-reid config missing: {FASTREID_ROOT / REID_CONFIG}")

        sys.path.insert(0, str(FASTREID_ROOT))
        from fastreid.config import get_cfg
        from fastreid.modeling.meta_arch import build_model
        from fastreid.utils.checkpoint import Checkpointer

        cfg = get_cfg()
        cfg.merge_from_file(str(FASTREID_ROOT / REID_CONFIG))
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.MODEL.WEIGHTS = str(checkpoint)
        cfg.freeze()

        model = build_model(cfg)
        model.eval()
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model
        self._size_test = tuple(cfg.INPUT.SIZE_TEST)  # [H, W] e.g. [256, 256]
        return self

    def __call__(self, images_bgr: np.ndarray) -> np.ndarray:
        """Embed a batch of BGR crops (N,H,W,3) -> (N, D) L2-normalized features."""
        if self._model is None:
            self.build()
        h, w = self._size_test
        resized = np.stack([cv2.resize(im, (w, h), interpolation=cv2.INTER_CUBIC) for im in images_bgr])
        rgb = resized[:, :, :, ::-1]  # BGR -> RGB
        tensor = torch.as_tensor(rgb.astype("float32").transpose(0, 3, 1, 2), device=self._model.device)
        with torch.no_grad():
            feats = self._model({"images": tensor})
        feats = torch.nn.functional.normalize(feats, dim=1).cpu().numpy()
        return feats

    # -- tracklet-level -----------------------------------------------------
    def embed_tracklet(
        self,
        tracklet: Tracklet,
        video: cv2.VideoCapture | None,
        frame_cache: dict[int, np.ndarray],
    ) -> np.ndarray | None:
        """Mean-pooled embedding for a tracklet, sampling crops from its frames.

        video: open VideoCapture for the camera (None -> skip, returns None).
        frame_cache: shared {frame: BGR image} so frames reused by several
            tracklets are decoded once.
        Returns None if no usable crops were decoded.
        """
        stride = self.cfg.reid_crop_stride
        frames = tracklet.frames[::stride][: self.cfg.reid_max_crops]
        crops = []
        for f in frames:
            img = frame_cache.get(f)
            if img is None and video is not None:
                video.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
                ok, img = video.read()
                if ok:
                    # Bounded LRU: decoded 1080p frames are ~6 MB each. The old
                    # code retained every frame for the whole camera (~2000
                    # frames => ~12 GB of RAM), which was crashing the host.
                    # Keep only the most recent frames; re-decoding on a miss is
                    # cheap relative to the GPU inference we're doing anyway.
                    frame_cache[f] = img
                    frame_cache.move_to_end(f)
                    while len(frame_cache) > self.cfg.reid_cache_frames:
                        frame_cache.popitem(last=False)
            if img is None:
                continue
            try:
                idx = tracklet.frames.index(f)
            except ValueError:
                continue
            x, y, w, h = tracklet.boxes[idx].astype(int)
            x0, y0 = max(x, 0), max(y, 0)
            x1, y1 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
            if x1 <= x0 or y1 <= y0:
                continue
            crop = img[y0:y1, x0:x1]
            if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                continue
            crops.append(crop)

        if not crops:
            return None
        feats = []
        for i in range(0, len(crops), self.cfg.reid_batch_size):
            feats.append(self(crops[i : i + self.cfg.reid_batch_size]))
        feats = np.vstack(feats)
        mean = feats.mean(axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 0 else None
