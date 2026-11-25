"""qwen3-vl:4b vision-tower-only embedder.

Loads ONLY the ``model.visual.*`` tensors from the HuggingFace checkpoint
(``Qwen/Qwen3-VL-4B-Instruct``) into a standalone ``Qwen3VLVisionModel`` and
exposes the same ``embed_batch(crops_bgr) -> (N, K) L2-normalized`` interface as
the rest of ``reid_models`` / ``benchmark_encoder_vs_vlm``.

The full LLM (4.4B params) is *not* materialized — the checkpoint safetensors
are streamed and only vision-tower tensors are kept, so this runs in ~1-2GB RAM.

Embedding: forward the image through the tower and mean-pool the per-image
merged patch tokens (``pooler_output``, shape ``(n_tokens, out_hidden_size)``)
into a single vector. ``out_hidden_size`` == the dim the LLM would consume
(2560 for qwen3-vl:4b), so the cosine space is the same one the VLM reasons in.

Vision tower of qwen3-vl:4b (from Ollama modelinfo): 24 blocks, width 1024,
patch 16, spatial_merge_size 2, temporal_patch_size 2, rope freq_base 10000.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

REPO = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_PREFIX = "model.visual."


def _hub_snapshot(repo: str = REPO) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo))


def build_tower(device: str | None = None):
    """Return (model, config, device, dtype, snapshot). Loads only visual weights."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from safetensors import safe_open

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    snapshot = _hub_snapshot(REPO)
    config = _vision_config(snapshot)
    model = Qwen3VLVisionModel(config).to(dtype).to(device).eval()

    state = {}
    for shard in sorted(snapshot.glob("model-*-of-*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith(MODEL_PREFIX):
                    state[k[len(MODEL_PREFIX):]] = f.get_tensor(k).to(dtype)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"missing vision weights: {missing[:8]} ...")
    if unexpected:
        raise RuntimeError(f"unexpected vision keys: {unexpected[:8]} ...")
    return model, config, device, dtype, snapshot


def _vision_config(snapshot: Path):
    import json
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    cfg = json.loads((snapshot / "config.json").read_text())
    return Qwen3VLVisionConfig(**cfg["vision_config"])


class Qwen3VLImagePreprocessor:
    """Image -> (pixel_values, image_grid_thw) using the checkpoint processor."""

    def __init__(self, snapshot: Path):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(str(snapshot))

    def __call__(self, images: list[np.ndarray]) -> dict:
        from PIL import Image
        pil = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in images]
        out = self.processor(images=pil, return_tensors="pt")
        return {
            "pixel_values": out["pixel_values"],
            "grid_thw": out["image_grid_thw"],
        }


class Qwen3VLTower:
    """L2-normalized mean-pooled vision-tower embeddings, reid_models-compatible."""

    def __init__(self, model, config, device, dtype, snapshot: Path):
        self.model = model
        self.config = config
        self.device = device
        self.dtype = dtype
        self.pre = Qwen3VLImagePreprocessor(snapshot)
        self.dim = config.out_hidden_size   # 2560 for qwen3-vl:4b
        self.tower_dim = config.hidden_size  # 1024

    def embed_batch(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        return self._embed(crops_bgr, pooled=True)

    def embed_batch_tower(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Variant using the raw 1024-d tower output (pre-projector)."""
        return self._embed(crops_bgr, pooled=False)

    def _embed(self, crops_bgr: list[np.ndarray], pooled: bool) -> np.ndarray:
        if not crops_bgr:
            return np.zeros((0, self.dim if pooled else self.tower_dim), dtype=np.float32)
        inputs = self.pre(crops_bgr)
        pv = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        grid = inputs["grid_thw"].to(self.device)
        with torch.inference_mode():
            out = self.model(pv, grid_thw=grid)

        tokens = out.pooler_output            # (n_tokens, out_hidden_size)
        split = (grid.prod(-1) // (self.config.spatial_merge_size ** 2)).tolist()
        chunks = torch.split(tokens, split, dim=0)
        means = torch.stack([c.mean(0) for c in chunks], dim=0)
        if not pooled:
            raw = out.last_hidden_state       # (n_raw_tokens, hidden_size=1024)
            raw_split = (grid.prod(-1)).tolist()
            chunks = torch.split(raw, raw_split, dim=0)
            means = torch.stack([c.mean(0) for c in chunks], dim=0)
        return torch.nn.functional.normalize(means.float(), dim=-1).cpu().numpy()


def get_qwen3vl_tower(name: str):
    """reid_models-style builder returning the pooled embedder."""
    model, config, device, dtype, snapshot = build_tower()
    return Qwen3VLTower(model, config, device, dtype, snapshot)