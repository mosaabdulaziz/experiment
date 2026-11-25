"""ResNet50-based appearance embedding extractor for vehicle crops.

Uses a pretrained ResNet50 (penultimate layer features, L2-normalized) to
produce a 2048-d visual embedding for each vehicle crop.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms


class AppearanceExtractor:
    """Extract L2-normalized visual embeddings from vehicle crops."""

    def __init__(self, device: str | None = None, cache_size: int = 512):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load pretrained ResNet50, remove final FC layer
        weights = models.ResNet50_Weights.DEFAULT
        base = models.resnet50(weights=weights)
        self.model = torch.nn.Sequential(*list(base.children())[:-1])  # → (B, 2048, 1, 1)
        self.model.eval()
        self.model.to(self.device)

        # Preprocessing matching ImageNet training
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.embed_dim = 2048
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

        print(f"  [reid] ResNet50 loaded on {self.device}, dim={self.embed_dim}")

    @torch.no_grad()
    def embed_crop(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Embed a single BGR crop → (2048,) L2-normalized vector."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < 8 or w < 8:
            return None

        # BGR → RGB
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.preprocess(rgb).unsqueeze(0).to(self.device)
        feat = self.model(tensor).squeeze()        # (2048,)
        feat = F.normalize(feat, dim=0)             # L2 normalize
        return feat.cpu().numpy()

    @torch.no_grad()
    def embed_batch(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Embed a list of BGR crops → (N, 2048) L2-normalized."""
        if not crops_bgr:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        tensors = []
        valid_idx = []
        for i, crop in enumerate(crops_bgr):
            if crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.preprocess(rgb))
            valid_idx.append(i)

        if not tensors:
            return np.zeros((len(crops_bgr), self.embed_dim), dtype=np.float32)

        batch = torch.stack(tensors).to(self.device)
        feats = self.model(batch).squeeze(-1).squeeze(-1)  # (N, 2048)
        feats = F.normalize(feats, dim=1)
        feats_np = feats.cpu().numpy()

        # Fill result (invalid crops get zero vectors)
        result = np.zeros((len(crops_bgr), self.embed_dim), dtype=np.float32)
        for j, vi in enumerate(valid_idx):
            result[vi] = feats_np[j]
        return result

    def embed_tracklet(
        self,
        tracklet,  # Tracklet from yolo_detector
        video: cv2.VideoCapture,
        stride: int = 4,
        max_crops: int = 16,
        frame_cache: OrderedDict[int, np.ndarray] | None = None,
    ) -> np.ndarray | None:
        """Embed a tracklet by sampling crops across its frames.

        Returns mean-pooled (2048,) L2-normalized embedding, or None.
        """
        if frame_cache is None:
            frame_cache = OrderedDict()

        sampled_frames = tracklet.frames[::stride][:max_crops]
        crops = []

        for f in sampled_frames:
            # Check cache
            img = frame_cache.get(f)
            if img is None:
                video.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, img = video.read()
                if ok:
                    frame_cache[f] = img
                    frame_cache.move_to_end(f)
                    while len(frame_cache) > 64:
                        frame_cache.popitem(last=False)
                else:
                    continue

            # Get box for this frame
            box = tracklet.crop_at_frame(f)
            if box is None:
                continue

            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2 = min(x2, img.shape[1])
            y2 = min(y2, img.shape[0])
            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)

        if not crops:
            return None

        # Embed all crops and mean-pool
        all_embs = self.embed_batch(crops)
        mean_emb = all_embs.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        return mean_emb / norm if norm > 0 else None
