"""Registry of appearance-embedding models for the vehicle re-ID benchmark.

Every model exposes the same interface::

    handle = get_model("fastreid-veri-sbs-r50-ibn")
    feats = handle.embed_batch(list_of_bgr_crops)   # (N, dim) L2-normalized

Models:

* ``fastreid-veri-sbs-r50-ibn`` — fast-reid SBS(R50-ibn) trained on VeRi
  (the model used in production, sidecar/mtmc).  Requires the fast-reid
  checkout + ``veri_sbs_R50-ibn.pth`` checkpoint.
* ``resnet50-imagenet`` — plain torchvision ResNet50 (ImageNet), penultimate
  layer features (mtmc/feature_extractor.py baseline).
* ``clip-vit-b32`` — OpenAI CLIP ViT-B/32 image tower (HuggingFace download).
* ``dinov2-base`` / ``dinov2-large`` — Meta DINOv2 (HuggingFace download).
* ``hsv-hist`` — CPU color-histogram baseline (no learning) to anchor how much
  the learned embeddings actually buy.

Embedding always returns L2-normalized rows; cosine similarity == dot product.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# registry plumbing
# ---------------------------------------------------------------------------


@dataclass
class ModelHandle:
    name: str
    dim: int
    build: callable          # () -> ready-to-call embedder object
    load_s: float = 0.0
    params: int = 0
    device: str = ""
    _emb: object = field(default=None, init=False, repr=False)

    def ensure(self) -> object:
        if self._emb is None:
            t0 = time.time()
            self._emb = self.build()
            self.load_s = time.time() - t0
            obj = self._emb
            # extractor-style wrappers expose parameters() on an inner model
            if not callable(getattr(obj, "parameters", None)):
                for attr in ("model", "vision_model"):
                    if hasattr(obj, attr) and callable(getattr(getattr(obj, attr), "parameters", None)):
                        obj = getattr(obj, attr)
                        break
            self.params = _count_params(getattr(obj, "parameters", lambda: ()))
        return self._emb

    def embed_batch(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        return self.ensure().embed_batch(crops_bgr)


def _count_params(params_iter) -> int:
    try:
        return sum(p.numel() for p in params_iter() if p.requires_grad)
    except Exception:
        return 0


def _device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def _to_tensor(bgr_list: list[np.ndarray], device: str):
    """Stack BGR crops -> float32 NCHW tensor on device (no resize)."""
    import torch
    arr = np.stack(bgr_list).astype("float32")           # N,H,W,3 (BGR)
    rgb = arr[:, :, :, ::-1].copy()                       # BGR -> RGB
    return torch.as_tensor(rgb.transpose(0, 3, 1, 2), device=device)


def _l2norm(x) -> np.ndarray:
    import torch
    return torch.nn.functional.normalize(x, dim=-1).cpu().numpy()


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

# --- HSV color-histogram baseline ------------------------------------------


def _build_hsv_hist():
    class _HsvHist:
        dim = 128

        def parameters(self):
            return iter(())

        def embed_batch(self, crops_bgr):
            out = np.zeros((len(crops_bgr), self.dim), dtype=np.float32)
            for i, im in enumerate(crops_bgr):
                if im is None or im.size == 0:
                    continue
                small = cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1, 2], None,
                                    [8, 8, 2], [0, 180, 0, 256, 0, 256])
                v = hist.flatten().astype(np.float32)
                n = np.linalg.norm(v)
                if n > 0:
                    out[i] = v / n
            return out
    return _HsvHist()


# --- fast-reid VeRi SBS R50-ibn --------------------------------------------


def _find_fastreid() -> tuple[Path, Path]:
    """Locate (fastreid_root, checkpoint). Searches env + known layouts."""
    import os
    env_root = os.environ.get("FASTREID_ROOT")
    env_ckpt = os.environ.get("REID_CHECKPOINT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates += [
        Path("helper-repos/multi-camera-tracking/fast-reid"),
        Path("/home/mmtayee/URCA/IRIS/helper-repos/multi-camera-tracking/fast-reid"),
        Path("/workspace/iris/helper-repos/multi-camera-tracking/fast-reid"),
        Path("/opt/fast-reid"),
    ]
    root = next((p for p in candidates if (p / "fastreid").exists()), None)
    if root is None:
        raise FileNotFoundError(
            "fast-reid checkout not found — set FASTREID_ROOT or check "
            "helper-repos/multi-camera-tracking/fast-reid"
        )
    ckpt = Path(env_ckpt) if env_ckpt else None
    for cand in ([ckpt] if ckpt else []) + [
        root / "checkpoints" / "veri_sbs_R50-ibn.pth",
        Path("/home/mmtayee/URCA/IRIS/helper-repos/multi-camera-tracking/"
             "fast-reid/checkpoints/veri_sbs_R50-ibn.pth"),
        Path("/workspace/iris/helper-repos/multi-camera-tracking/fast-reid/"
             "checkpoints/veri_sbs_R50-ibn.pth"),
    ]:
        if cand is not None and cand.exists():
            ckpt = cand
            break
    if ckpt is None or not ckpt.exists():
        raise FileNotFoundError(
            "VeRi checkpoint missing — expected veri_sbs_R50-ibn.pth under the "
            "fast-reid checkout (see mtmc/README.md for the download URL)"
        )
    return root, ckpt


def _build_fastreid_veri():
    import torch

    root, ckpt = _find_fastreid()
    cfg_rel = "configs/VeRi/sbs_R50-ibn.yml"
    cfg_path = root / cfg_rel
    if not cfg_path.exists():
        raise FileNotFoundError(f"fast-reid config missing: {cfg_path}")

    sys.path.insert(0, str(root))
    from fastreid.config import get_cfg
    from fastreid.modeling.meta_arch import build_model
    from fastreid.utils.checkpoint import Checkpointer

    cfg = get_cfg()
    cfg.merge_from_file(str(cfg_path))
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.WEIGHTS = str(ckpt)
    cfg.freeze()

    model = build_model(cfg)
    model.eval()
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    device = _device()
    model.to(device)
    size_test = tuple(cfg.INPUT.SIZE_TEST)  # [H, W]

    class _FastReid:
        dim = 2048

        def parameters(self):
            return model.parameters()

        def embed_batch(self, crops_bgr):
            if not crops_bgr:
                return np.zeros((0, self.dim), dtype=np.float32)
            h, w = size_test
            resized = np.stack([
                cv2.resize(im, (w, h), interpolation=cv2.INTER_CUBIC)
                for im in crops_bgr
            ])
            tensor = _to_tensor(list(resized), device)
            with torch.no_grad():
                feats = model({"images": tensor})
            return _l2norm(feats)
    return _FastReid()


# --- ResNet50 (ImageNet) appearance extractor -------------------------------


def _build_resnet50():
    from .feature_extractor import AppearanceExtractor
    ext = AppearanceExtractor(device=_device())
    return ext  # has .embed_batch(crops) -> (N, 2048) L2-normalized


# --- CLIP ViT-B/32 ----------------------------------------------------------


def _build_clip(model_id: str = "openai/clip-vit-base-patch32"):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = _device()
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    dim = model.config.projection_dim or model.config.vision_config.hidden_size

    class _Clip:
        def parameters(self):
            return model.parameters()

        def embed_batch(self, crops_bgr):
            from PIL import Image
            if not crops_bgr:
                return np.zeros((0, dim), dtype=np.float32)
            pil = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                   for im in crops_bgr]
            inputs = processor(images=pil, return_tensors="pt").to(device)
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
            # transformers >= 5 returns a BaseModelOutputWithPooling object
            # (the projected embedding lives in .pooler_output); older versions
            # returned the pooled tensor directly.
            if not isinstance(feats, torch.Tensor):
                for attr in ("image_embeds", "pooler_output"):
                    if hasattr(feats, attr):
                        feats = getattr(feats, attr)
                        break
                else:
                    raise TypeError(
                        f"unexpected CLIP output object: {type(feats).__name__}"
                    )
            return _l2norm(feats)
    return _Clip()


# --- DINOv2 ----------------------------------------------------------------


def _build_dinov2(model_id: str = "facebook/dinov2-base"):
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = _device()
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_id)
    dim = model.config.hidden_size

    class _Dino:
        def parameters(self):
            return model.parameters()

        def embed_batch(self, crops_bgr):
            from PIL import Image
            if not crops_bgr:
                return np.zeros((0, dim), dtype=np.float32)
            pil = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                   for im in crops_bgr]
            inputs = processor(images=pil, return_tensors="pt").to(device)
            with torch.no_grad():
                hidden = model(**inputs).last_hidden_state
            return _l2norm(hidden[:, 0])   # [CLS] token
    return _Dino()


# --- SigLIP (google/siglip-base-patch16-224) --------------------------------


def _build_siglip(model_id: str = "google/siglip-base-patch16-224"):
    import torch
    from transformers import AutoModel, AutoProcessor

    device = _device()
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    dim = model.config.vision_config.hidden_size if hasattr(
        model.config, "vision_config") else model.config.hidden_size

    class _Siglip:
        def parameters(self):
            return model.parameters()

        def embed_batch(self, crops_bgr):
            from PIL import Image
            if not crops_bgr:
                return np.zeros((0, dim), dtype=np.float32)
            pil = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                   for im in crops_bgr]
            inputs = processor(images=pil, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.get_image_features(**inputs)
            feats = out
            if not isinstance(feats, torch.Tensor):
                for attr in ("image_embeds", "pooler_output"):
                    if hasattr(feats, attr):
                        feats = getattr(feats, attr)
                        break
            return _l2norm(feats)
    return _Siglip()


# --- SigLIP2 (google/siglip2-base-patch16-384) ------------------------------


def _build_siglip2(model_id: str = "google/siglip2-base-patch16-384"):
    import torch
    from transformers import AutoModel, AutoProcessor

    device = _device()
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    dim = model.config.vision_config.hidden_size if hasattr(
        model.config, "vision_config") else model.config.hidden_size

    class _Siglip2:
        def parameters(self):
            return model.parameters()

        def embed_batch(self, crops_bgr):
            from PIL import Image
            if not crops_bgr:
                return np.zeros((0, dim), dtype=np.float32)
            pil = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                   for im in crops_bgr]
            inputs = processor(images=pil, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.get_image_features(**inputs)
            feats = out
            if not isinstance(feats, torch.Tensor):
                for attr in ("image_embeds", "pooler_output"):
                    if hasattr(feats, attr):
                        feats = getattr(feats, attr)
                        break
            return _l2norm(feats)
    return _Siglip2()


# --- qwen3-vl:4b vision tower (encoder-only) --------------------------------


def _build_qwen3vl_tower():
    from .qwen3vl_tower import get_qwen3vl_tower
    return get_qwen3vl_tower("qwen3vl-vision-tower")


def _build_qwen3vl_tower_raw():
    from .qwen3vl_tower import build_tower, Qwen3VLTower
    model, config, device, dtype, snapshot = build_tower()

    class _Raw(Qwen3VLTower):
        def embed_batch(self, crops_bgr):
            return self._embed(crops_bgr, pooled=False)

    return _Raw(model, config, device, dtype, snapshot)


# ---------------------------------------------------------------------------
# public registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, ModelHandle] = {
    "hsv-hist": ModelHandle(
        name="hsv-hist",
        dim=128,
        build=_build_hsv_hist,
        device="cpu",
    ),
    "fastreid-veri-sbs-r50-ibn": ModelHandle(
        name="fastreid-veri-sbs-r50-ibn",
        dim=2048,
        build=_build_fastreid_veri,
        device=_device(),
    ),
    "resnet50-imagenet": ModelHandle(
        name="resnet50-imagenet",
        dim=2048,
        build=_build_resnet50,
        device=_device(),
    ),
    "clip-vit-b32": ModelHandle(
        name="clip-vit-b32",
        dim=512,
        build=_build_clip,
        device=_device(),
    ),
    "dinov2-base": ModelHandle(
        name="dinov2-base",
        dim=768,
        build=lambda: _build_dinov2("facebook/dinov2-base"),
        device=_device(),
    ),
    "dinov2-large": ModelHandle(
        name="dinov2-large",
        dim=1024,
        build=lambda: _build_dinov2("facebook/dinov2-large"),
        device=_device(),
    ),
    "siglip-base": ModelHandle(
        name="siglip-base",
        dim=768,
        build=_build_siglip,
        device=_device(),
    ),
    "siglip2-base": ModelHandle(
        name="siglip2-base",
        dim=768,
        build=_build_siglip2,
        device=_device(),
    ),
    "qwen3vl-vision-tower": ModelHandle(
        name="qwen3vl-vision-tower",
        dim=2560,
        build=_build_qwen3vl_tower,
        device=_device(),
    ),
    "qwen3vl-vision-tower-raw": ModelHandle(
        name="qwen3vl-vision-tower-raw",
        dim=1024,
        build=_build_qwen3vl_tower_raw,
        device=_device(),
    ),
}


def get_model(name: str) -> ModelHandle:
    if name not in REGISTRY:
        raise KeyError(
            f"unknown model {name!r}; available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hsv-hist")
    ap.add_argument("--crop", nargs="+", default=[])
    args = ap.parse_args()
    h = get_model(args.model)
    if args.crop:
        crops = [cv2.imread(p) for p in args.crop]
        feats = h.embed_batch(crops)
        print(f"{h.name}: {len(crops)} crops -> {feats.shape} (dim={h.dim})")
        if len(feats) > 1:
            sim = feats @ feats.T
            print("sim matrix:\n", np.round(sim, 3))
