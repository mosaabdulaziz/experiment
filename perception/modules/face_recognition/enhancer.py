"""Lightweight face enhancement module for IRIS sidecar.

Improves low-quality face crops from CCTV streams before recognition.
Supports multiple enhancement backends:

  - ``preprocess`` (default):
      Lightweight OpenCV-only pipeline: Lanczos upscaling → CLAHE contrast
      enhancement → unsharp masking.  No model files, no extra dependencies.

  - ``gfpgan``:
      Full GFPGAN blind face restoration (StyleGAN2-based).  Requires the
      ``gfpgan`` package + a pretrained model checkpoint downloaded from
      GitHub releases.  One-time ~300 MB download.

  - ``none``:
      Passthrough — no enhancement applied.  Useful for A/B comparison.

Usage
-----
    enhancer = FaceEnhancer(config)
    enhanced = enhancer.enhance(face_crop_bgr)
"""

import logging
from typing import Optional, Dict, Any

import cv2
import numpy as np

logger = logging.getLogger("iris.face_enhancer")


class FaceEnhancer:
    """Enhance face crops before feeding them to the recognition engine.

    Parameters
    ----------
    config : dict
        Global sidecar config dict.  Relevant keys:

        - ``face_enhancement_enabled`` (bool): Master switch.  Default False.
        - ``face_enhancement_backend`` (str): ``"preprocess"`` | ``"gfpgan"`` | ``"none"``.
        - ``face_enhancement_min_face_size`` (int): Minimum face side (pixels)
          below which enhancement is triggered.  Default 48.
        - ``face_enhancement_target_size`` (int): Upscale target for small faces.
          Default 256.
    """

    def __init__(self, config: Dict[str, Any]):
        self._enabled = bool(config.get("face_enhancement_enabled", False))
        self._backend = str(config.get("face_enhancement_backend", "preprocess"))
        self._min_face_size = int(config.get("face_enhancement_min_face_size", 48))
        self._target_size = int(config.get("face_enhancement_target_size", 256))
        self._weight = float(config.get("face_enhancement_weight", 0.5))

        self._gfpgan: Optional[Any] = None
        if self._enabled and self._backend == "gfpgan":
            self._init_gfpgan(config)

        logger.info(
            "FaceEnhancer: enabled=%s backend=%s min_face=%d target=%d",
            self._enabled, self._backend, self._min_face_size, self._target_size,
        )

    # ── public API ────────────────────────────────────────────────────────

    def enhance(self, face_crop: np.ndarray) -> np.ndarray:
        """Enhance a face crop.

        Args:
            face_crop: BGR image of the face region (any size).

        Returns:
            Enhanced BGR image (same or larger size) or the original crop if
            enhancement is disabled / fails.
        """
        if not self._enabled or self._backend == "none":
            return face_crop

        try:
            h, w = face_crop.shape[:2]

            if self._backend == "preprocess":
                return self._enhance_preprocess(face_crop, min(h, w) < self._min_face_size)

            if self._backend == "gfpgan":
                return self._enhance_gfpgan(face_crop)

        except Exception as exc:
            logger.debug("Face enhancement failed (%s), using original crop: %s", self._backend, exc)

        return face_crop

    # ── lightweight OpenCV pipeline ──────────────────────────────────────

    def _enhance_preprocess(self, face_crop: np.ndarray, needs_upscale: bool) -> np.ndarray:
        """OpenCV-only enhancement pipeline.

        1. Lanczos upscaling for small crops (to ``self._target_size``)
        2. CLAHE on the L* channel (Lab colour space)
        3. Mild unsharp masking
        """
        img = face_crop.copy()

        # ── step 1: upscale ────────────────────────────────────────────────
        if needs_upscale:
            h, w = img.shape[:2]
            scale = min(2.0, max(1.0, float(self._target_size) / max(h, w)))
            if scale > 1.05:
                img = cv2.resize(img, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_LANCZOS4)

        # ── step 2: CLAHE on luminance ─────────────────────────────────────
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # ── step 3: unsharp masking ────────────────────────────────────────
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
        img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

        return img

    # ── GFPGAN backend (heavy, lazy-loaded) ───────────────────────────────

    def _init_gfpgan(self, config: Dict[str, Any]) -> None:
        """Download (if needed) and load the GFPGAN model."""
        try:
            from gfpgan.utils import GFPGANer  # type: ignore[import-untyped]

            model_name = config.get("face_enhancement_model", "GFPGANv1.3")
            model_url = (
                f"https://github.com/TencentARC/GFPGAN/releases/download/"
                f"v1.3.0/{model_name}.pth"
            )

            arch = "clean"
            channel_multiplier = 2
            if model_name == "GFPGANv1":
                arch = "original"
                channel_multiplier = 1
            elif model_name == "RestoreFormer":
                arch = "RestoreFormer"
                channel_multiplier = 2

            self._gfpgan = GFPGANer(
                model_path=model_url,
                upscale=2,
                arch=arch,
                channel_multiplier=channel_multiplier,
                bg_upsampler=None,
            )
            logger.info("GFPGAN restorer initialised (%s)", model_name)

        except ImportError:
            logger.warning(
                "GFPGAN backend selected but 'gfpgan' package not installed; "
                "falling back to 'preprocess'"
            )
            self._backend = "preprocess"
        except Exception as exc:
            logger.warning(
                "GFPGAN init failed (%s); falling back to 'preprocess'", exc
            )
            self._backend = "preprocess"

    def _enhance_gfpgan(self, face_crop: np.ndarray) -> np.ndarray:
        """Enhance using full GFPGAN restoration.

        Returns the first restored face (512×512) or the original crop if the
        model does not detect a face in the crop.
        """
        if self._gfpgan is None:
            return face_crop

        # With has_aligned=True, GFPGAN skips its own RetinaFace detection (which
        # often fails on very small CCTV crops) and directly resizes the crop to
        # 512x512 before running the restoration network. This ensures even tiny
        # face crops (e.g. 40x60 px from a distant person) get enhanced.
        _, restored_faces, _ = self._gfpgan.enhance(
            face_crop,
            has_aligned=True,
            only_center_face=True,
            paste_back=False,
            weight=self._weight,
        )
        if restored_faces:
            return restored_faces[0]
        return face_crop
