"""Vehicle Re-ID engine: embed crops from the CPU frame for cross-camera matching.

Wraps a torchvision ResNet50 (ImageNet-pretrained) appearance extractor. The weights are the repo's DVC-tracked
copy of torchvision's IMAGENET1K_V2 file (REID_WEIGHTS, byte-identical to the torch hub download: sha256 11ad3fa6...);
they are never downloaded at run time. A missing file refuses startup (ReIDWeightsMissing) instead of silently
falling back; any other load failure still degrades to time-only association.

ResNet50 serves as BOTH a coarse gate (reject obviously-different vehicles
below ``reid_gate_threshold``) and the fine matcher (merge above
``reid_sim_threshold``).  On our 13-camera footage it ranked cross-camera
matches far better than the old fast-reid VeRi model (R@1 31% vs 7%) and was
faster (4.3 vs 5.5 ms/crop) — see mtmc/output/reid_bench/ANALYSIS.md.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("iris.mtmc")

REPO_ROOT = Path(__file__).resolve().parents[4]          # perception/modules/mtmc/core/_reid.py -> the repo root
REID_WEIGHTS_REL = "mtmc/models/reid/resnet50-11ad3fa6.pth"
REID_WEIGHTS = REPO_ROOT / REID_WEIGHTS_REL


class ReIDWeightsMissing(FileNotFoundError):
    """The Re-ID weights are not on disk: a deployment error, never retried and never papered over."""


class ReIDEngine:
    """Lazy-loaded ResNet50 (ImageNet) feature extractor for vehicle Re-ID.

    Produces 2048-d L2-normalized embeddings from the penultimate layer,
    matching the preprocessing in mtmc/feature_extractor.py.
    """

    def __init__(self, cfg):
        self._cfg = cfg
        self._model = None
        self._preprocess = None
        self._device = None
        self._load_attempts = 0
        self._load_failed_permanently = False

    def _build(self) -> None:
        """Construct the ResNet50 model (called once on first use)."""
        # The ImageNet weights are the repo's DVC file. Until 2026-11 this path pointed one level too deep
        # (perception/modules/mtmc/models/..., left over from the move out of mtmc/core), so every site silently
        # downloaded the same file from torch hub instead; there is no download fallback any more.
        if not REID_WEIGHTS.is_file():
            raise ReIDWeightsMissing(
                f"the vehicle Re-ID weights {REID_WEIGHTS_REL} are missing ({REID_WEIGHTS}); they are versioned with "
                f"DVC: run `dvc pull {REID_WEIGHTS_REL}.dvc` (or set reid_enabled: false to run without Re-ID)")
        import torch
        import torch.nn.functional as F
        from torchvision import models, transforms

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            # DeepStream container's torch lacks cuDNN 9 runtime sublibraries in LD_LIBRARY_PATH;
            # disabling cuDNN uses native CUDA kernels which run reliably without sublibraries.
            torch.backends.cudnn.enabled = False

        base = models.resnet50(weights=None)
        base.load_state_dict(torch.load(REID_WEIGHTS, map_location="cpu"))
        # Drop avgpool + FC -> (B, 2048) penultimate-layer features.
        model = torch.nn.Sequential(*list(base.children())[:-1])
        model.eval()
        model.to(device)

        # ImageNet preprocessing matching mtmc/feature_extractor.py.
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self._model = model
        self._preprocess = preprocess
        self._device = device
        logger.info("[mtmc] Re-ID model loaded on %s (ResNet50-ImageNet, dim=2048)", device)

    def ensure_loaded(self) -> bool:
        """Ensure the model is loaded. Returns True if available."""
        if self._model is not None:
            return True
        if getattr(self, "_missing", None) is not None:
            raise self._missing
        # Permanent failure flag — stop retrying after max attempts
        if getattr(self, "_load_failed_permanently", False):
            return False
        attempts = getattr(self, "_load_attempts", 0)
        if attempts >= 3:
            self._load_failed_permanently = True
            logger.error("[mtmc] Re-ID model load failed after %d attempts — disabling Re-ID", attempts)
            return False
        self._load_attempts = attempts + 1
        try:
            self._build()
            return True
        except ReIDWeightsMissing as exc:
            self._missing = exc                 # every later call raises it too, never a quiet "unavailable"
            raise
        except Exception as exc:
            logger.warning("[mtmc] Re-ID model load attempt %d failed: %s", self._load_attempts, exc)
            return False

    def embed_crops(self, crops: list[np.ndarray]) -> np.ndarray | None:
        """Embed a batch of BGR crops → (N, 2048) L2-normalized features.

        Returns None if the model isn't loaded or no usable crops.
        """
        if not self.ensure_loaded() or not crops:
            return None

        import torch
        import torch.nn.functional as F

        tensors = []
        for im in crops:
            if im is None or im.size == 0 or im.shape[0] < 8 or im.shape[1] < 8:
                continue
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            tensors.append(self._preprocess(rgb))

        if not tensors:
            return None

        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            feats = self._model(batch).squeeze(-1).squeeze(-1)  # (N, 2048)
        feats = F.normalize(feats, dim=1)
        return feats.cpu().numpy()

    def embed_tracklet(self, tracklet) -> np.ndarray | None:
        """Compute or update the mean-pooled embedding for a tracklet.

        Samples crops from the tracklet's accumulated crop window, embeds
        them in batches, and returns the L2-normalized mean.

        Returns None if no usable crops.
        """
        crops = tracklet.get_crops()
        if not crops:
            return None

        stride = self._cfg.reid_embed_stride
        sampled = crops[::stride][:self._cfg.reid_max_crops]

        all_feats = []
        bs = self._cfg.reid_batch_size
        for i in range(0, len(sampled), bs):
            feats = self.embed_crops(sampled[i:i + bs])
            if feats is not None:
                all_feats.append(feats)

        if not all_feats:
            return None

        combined = np.vstack(all_feats)
        mean = combined.mean(axis=0)
        norm = np.linalg.norm(mean)
        embedding = mean / norm if norm > 0 else mean

        tracklet.embedding = embedding
        tracklet.embedding_count += 1
        return embedding