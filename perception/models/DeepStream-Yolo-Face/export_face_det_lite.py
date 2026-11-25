# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Export Qualcomm FaceDetLite (qualcomm/Lightweight-Face-Detection) to DeepStream ONNX.

Builds a single self-contained ONNX graph matching nvdsparseface_Yolo.cpp:
  - input  images: [batch, 3, 480, 640] float32 RGB in [0.0, 1.0]
  - output output: [batch, 4800, 15] float32
    [x1, y1, x2, y2, conf, lx0, ly0, lx1, ly1, lx2, ly2, lx3, ly3, lx4, ly4]
    in 640x480 network pixels, with non-peak cells zeroed via 3x3 maxpool.

Usage:
    python export_face_det_lite.py
    python export_face_det_lite.py --check <image_path>
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

if TYPE_CHECKING:
    import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_OUT = HERE / "face_det_lite.onnx"
DEFAULT_WEIGHTS_NAME = "QFD_V3_superlite_060525_50_pickled.pt"
EXPECTED_WEIGHTS_SHA256 = "7fb1101957f9518fcd7889ab9f558ad058f265cb6d30d2e08184fa71f0bf8f1e"
OFFICIAL_ASSET_URL = (
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/"
    "qai-hub-models/models/face_det_lite/v3/QFD_V3_superlite_060525_50_pickled.pt"
)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "iris_face_det_lite"
INPUT_H = 480
INPUT_W = 640
STRIDE = 8
OPSET = 18


class SeModule(nn.Module):
    """Customized squeeze-excitation module (qai_hub_models v0.66.0)."""

    def __init__(self, in_size: int, reduction: int = 4) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(self.pool(x))


class Block3x3(nn.Module):
    """Modified MobileNetV3 block (qai_hub_models v0.66.0)."""

    def __init__(
        self,
        kernel_size: int,
        in_size: int,
        expand_size: int,
        out_size: int,
        nolinear: nn.Module,
        semodule: Optional[nn.Module],
        stride: int,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.se = semodule
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        if kernel_size == 3:
            self.conv2 = nn.Conv2d(
                expand_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False
            )
        else:
            self.conv2 = nn.Conv2d(expand_size, out_size, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv3 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_size)
            self.nolinear3 = nolinear
        self.bn2 = nn.BatchNorm2d(out_size)
        self.nolinear2 = nolinear
        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        if self.kernel_size == 5:
            out = self.nolinear3(self.bn3(self.conv3(out)))
        if self.se is not None:
            out = self.se(out)
        return out + self.shortcut(x) if self.stride == 1 else out


class CBAModule(nn.Module):
    """Conv-BatchNorm-Activation block (qai_hub_models v0.66.0)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 24,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        act: str = "relu",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act: nn.Module
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "identity":
            self.act = nn.Identity()
        else:
            self.act = nn.PReLU()
        init.xavier_uniform_(self.conv.weight.data)
        if self.conv.bias is not None:
            self.conv.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class UpModule(nn.Module):
    """Upsampling module (qai_hub_models v0.66.0)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        bias: bool = False,
        mode: str = "UCBA",
        act: str = "relu",
    ) -> None:
        super().__init__()
        self.mode = mode
        if self.mode == "UCBA":
            self.up = nn.Upsample(size=None, scale_factor=2, mode="nearest")
            self.conv = CBAModule(in_channels, out_channels, 3, padding=1, bias=bias, act=act)
        elif self.mode == "DeconvBN":
            self.dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=bias)
            self.bn = nn.BatchNorm2d(out_channels)
        elif self.mode == "DeCBA":
            self.dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=bias)
            self.conv = CBAModule(out_channels, out_channels, 3, padding=1, bias=bias)
        else:
            raise RuntimeError(f"Unsupported mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "UCBA":
            return self.conv(self.up(x))
        if self.mode == "DeconvBN":
            return F.relu(self.bn(self.dconv(x)))
        if self.mode == "DeCBA":
            return self.conv(self.dconv(x))
        raise NotImplementedError()


class ContextModule(nn.Module):
    """Single-stage headless context module (qai_hub_models v0.66.0)."""

    def __init__(self, in_channels: int, act: str = "relu") -> None:
        super().__init__()
        block_wide = in_channels // 4
        self.inconv = CBAModule(in_channels, block_wide, 3, 1, padding=1, act=act)
        self.upconv = CBAModule(block_wide, block_wide, 3, 1, padding=1, act=act)
        self.downconv = CBAModule(block_wide, block_wide, 3, 1, padding=1, act="relu")
        self.downconv2 = CBAModule(block_wide, block_wide, 3, 1, padding=1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inconv(x)
        up = self.upconv(x)
        down = self.downconv2(self.downconv(x))
        return torch.cat([up, down], dim=1)


class DetectModule(nn.Module):
    def __init__(self, in_channels: int, act: str = "relu") -> None:
        super().__init__()
        self.upconv = CBAModule(in_channels, in_channels // 2, 3, 1, padding=1, act=act)
        self.context = ContextModule(in_channels, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.upconv(x), self.context(x)], dim=1)


class HeadModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_ext: bool = False,
        act: str = "relu",
    ) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.has_ext = has_ext
        if has_ext:
            self.ext = CBAModule(in_channels, in_channels, kernel_size=3, padding=1, bias=False, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_ext:
            x = self.ext(x)
        return self.head(x)


class Mbv3SmallFast(nn.Module):
    def __init__(self, act: str = "relu", RGB: bool = True, has_se: bool = True) -> None:
        super().__init__()
        self.keep = [2, 7]
        self.uplayer_shape = [32, 64]
        self.output_channels = 96

        in_ch = 3 if RGB else 1
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = nn.ReLU(inplace=True) if act == "relu" else nn.Identity()

        if has_se:
            self.bneck = nn.Sequential(
                Block3x3(3, 16, 16, 16, self.hs1, None, 2),
                Block3x3(3, 16, 64, 32, self.hs1, None, 2),
                Block3x3(3, 32, 96, 32, self.hs1, None, 1),
                Block3x3(5, 32, 96, 32, self.hs1, SeModule(32), 2),
                Block3x3(5, 32, 224, 32, self.hs1, SeModule(32), 1),
                Block3x3(5, 32, 224, 32, self.hs1, SeModule(32), 1),
                Block3x3(5, 32, 128, 64, self.hs1, SeModule(64), 1),
                Block3x3(5, 64, 160, 64, self.hs1, SeModule(64), 1),
                Block3x3(5, 64, 256, 96, self.hs1, SeModule(96), 2),
            )
        else:
            self.bneck = nn.Sequential(
                Block3x3(3, 16, 16, 16, self.hs1, None, 2),
                Block3x3(3, 16, 72, 24, self.hs1, None, 2),
                Block3x3(3, 24, 88, 24, self.hs1, None, 1),
                Block3x3(5, 24, 96, 40, self.hs1, None, 2),
                Block3x3(5, 40, 240, 40, self.hs1, None, 1),
                Block3x3(5, 40, 240, 40, self.hs1, None, 1),
                Block3x3(5, 40, 120, 48, self.hs1, None, 1),
                Block3x3(5, 48, 144, 48, self.hs1, None, 1),
                Block3x3(5, 48, 288, 96, self.hs1, None, 2),
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.hs1(self.bn1(self.conv1(x)))
        outs = []
        for index, item in enumerate(self.bneck):
            x = item(x)
            if index in self.keep:
                outs.append(x)
        outs.append(x)
        return outs


class FaceDetLite(nn.Module):
    """Qualcomm FaceDetLite architecture (qai_hub_models v0.66.0)."""

    def __init__(
        self,
        wide: int = 32,
        upmode: str = "UCBA",
        act: str = "relu",
        RGB: bool = False,
        has_se: bool = True,
        phase: str = "test",
    ) -> None:
        super().__init__()
        self.use_rgb = RGB
        self.has_landmark = True
        self.bb = Mbv3SmallFast(act, RGB, has_se)

        c1, c2 = self.bb.uplayer_shape
        self.conv3 = CBAModule(self.bb.output_channels, wide, kernel_size=1, stride=1, padding=0, bias=False, act="relu")
        self.connect1 = CBAModule(c1, wide, kernel_size=1, act="relu")
        self.connect2 = CBAModule(c2, wide, kernel_size=1, act="relu")

        self.up0 = UpModule(wide, wide, kernel_size=2, stride=2, mode=upmode, act="relu")
        self.up1 = UpModule(wide, wide, kernel_size=2, stride=2, mode=upmode, act="relu")
        self.detect = DetectModule(wide, act="relu")

        self.center = HeadModule(wide, 1, act="relu")
        self.box = HeadModule(wide, 4, act="relu")
        self.landmark = HeadModule(wide, 10, act="relu")
        self.phase = phase
        self.bridge = nn.Conv2d(wide * 2, wide, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not torch.jit.is_tracing():
            *_, h, w = image.shape
            if h % 32 != 0 or w % 32 != 0:
                raise ValueError("Image width/height must both be divisible by 32.")

        image = (image - 0.442) / 0.280
        s8_, s16_, s32_ = self.bb(image)
        s32 = self.conv3(s32_)
        s16 = self.up0(s32) + self.connect2(s16_)
        s8 = self.up1(s16) + self.connect1(s8_)
        x = self.detect(s8)

        center = self.center(x)
        box = self.box(x)
        landmark = self.landmark(x)
        if self.phase == "test":
            return center.sigmoid(), box, landmark
        return center, box, landmark


def decode_maps_to_yolo_face(
    hm: torch.Tensor,
    box: torch.Tensor,
    lmk: torch.Tensor,
    gx: torch.Tensor,
    gy: torch.Tensor,
    stride: int = STRIDE,
) -> torch.Tensor:
    """Decode FaceDetLite stride-8 maps into [B, H*W, 15] DeepStream-Yolo-Face layout.

    Non-peak cells and degenerate (<1px) boxes are zeroed via 3x3 maxpool so
    nvdsparseface_Yolo.cpp skips them in its precluster threshold check while keeping
    output tensor shape static along axis 1.
    """
    pooled = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)

    x1 = (gx - box[:, 0:1]) * float(stride)
    y1 = (gy - box[:, 1:2]) * float(stride)
    x2 = (gx + box[:, 2:3]) * float(stride)
    y2 = (gy + box[:, 3:4]) * float(stride)

    keep = (hm == pooled) & ((x2 - x1) >= 1.0) & ((y2 - y1) >= 1.0)
    conf = torch.where(keep, hm, torch.zeros_like(hm))

    pts = []
    for p in range(5):
        pts.append((lmk[:, p : p + 1] + gx) * float(stride))
        pts.append((lmk[:, 5 + p : 6 + p] + gy) * float(stride))

    packed = torch.cat([x1, y1, x2, y2, conf, *pts], dim=1)
    return packed.flatten(2).transpose(1, 2)


class FaceDetLiteDeployWrapper(nn.Module):
    """3-ch RGB [0,1] frontend + FaceDetLite(phase='test') + [B, 4800, 15] decode tail."""

    gx: torch.Tensor
    gy: torch.Tensor

    def __init__(self, core: FaceDetLite, height: int = INPUT_H, width: int = INPUT_W, stride: int = STRIDE) -> None:
        super().__init__()
        self.rgb_to_gray = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.rgb_to_gray.weight.copy_(torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1))
        for p in self.rgb_to_gray.parameters():
            p.requires_grad = False

        self.core = core
        self.stride = stride
        hm_h, hm_w = height // stride, width // stride
        gy, gx = torch.meshgrid(
            torch.arange(hm_h, dtype=torch.float32),
            torch.arange(hm_w, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("gx", gx.view(1, 1, hm_h, hm_w))
        self.register_buffer("gy", gy.view(1, 1, hm_h, hm_w))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        gray = self.rgb_to_gray(images)
        hm, box, lmk = self.core(gray)
        return decode_maps_to_yolo_face(hm, box, lmk, self.gx, self.gy, stride=self.stride)


def resolve_checkpoint(checkpoint: Optional[Path] = None, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Return local checkpoint path, downloading and SHA-256 verifying Qualcomm's official asset if missing."""
    if checkpoint is not None:
        ckpt = Path(checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / DEFAULT_WEIGHTS_NAME
    if cached.is_file() and cached.stat().st_size > 0:
        digest = hashlib.sha256(cached.read_bytes()).hexdigest()
        if digest == EXPECTED_WEIGHTS_SHA256:
            return cached
        cached.unlink()

    tmp = cached.with_suffix(".pt.part")
    urllib.request.urlretrieve(OFFICIAL_ASSET_URL, str(tmp))
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != EXPECTED_WEIGHTS_SHA256:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded checkpoint SHA-256 mismatch: expected {EXPECTED_WEIGHTS_SHA256}, got {digest}"
        )
    os.replace(tmp, cached)
    return cached


def load_core_model(checkpoint_path: Path) -> FaceDetLite:
    model = FaceDetLite(phase="test")
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    return model


def export(
    checkpoint: Optional[Path] = None,
    out: Path = DEFAULT_OUT,
    opset: int = OPSET,
) -> Path:
    """Export self-contained face_det_lite.onnx with dynamic batch axis and no external data."""
    import onnx

    ckpt_path = resolve_checkpoint(checkpoint)
    core = load_core_model(ckpt_path)
    wrapper = FaceDetLiteDeployWrapper(core).eval()
    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W, dtype=torch.float32)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_tmp = out.with_name(out.stem + ".raw_export.onnx")
    try:
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(raw_tmp),
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
            opset_version=opset,
            dynamo=False,
        )
        proto = onnx.load(str(raw_tmp), load_external_data=True)
        out_dims = proto.graph.output[0].type.tensor_type.shape.dim
        out_dims[0].ClearField("dim_value")
        out_dims[0].dim_param = "batch"
        out_dims[1].ClearField("dim_param")
        out_dims[1].dim_value = (INPUT_H // STRIDE) * (INPUT_W // STRIDE)
        out_dims[2].ClearField("dim_param")
        out_dims[2].dim_value = 15
        onnx.checker.check_model(proto)
        onnx.save_model(proto, str(out), save_as_external_data=False)
    finally:
        for stale in (raw_tmp, raw_tmp.with_suffix(raw_tmp.suffix + ".data")):
            if stale.is_file():
                stale.unlink()

    sidecar = out.with_suffix(out.suffix + ".data")
    if sidecar.is_file():
        sidecar.unlink()
    assert not sidecar.exists(), f"Unexpected external data sidecar found: {sidecar}"
    return out


def _letterbox_bgr(img_bgr: np.ndarray, dst_w: int = INPUT_W, dst_h: int = INPUT_H) -> np.ndarray:
    import cv2
    import numpy as np

    h, w = img_bgr.shape[:2]
    scale = min(dst_w / max(1, w), dst_h / max(1, h))
    pad_x = (dst_w - w * scale) / 2.0
    pad_y = (dst_h - h * scale) / 2.0
    resized = cv2.resize(img_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    x0, y0 = int(round(pad_x)), int(round(pad_y))
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def _match_detections(
    a: list[tuple[np.ndarray, float, Optional[np.ndarray]]],
    b: list[tuple[np.ndarray, float, Optional[np.ndarray]]],
    iou_thresh: float = 0.5,
) -> list[tuple[int, int]]:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from benchmarks.face_bench_adapters import box_iou_xyxy

    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for i, (box_a, _, _) in enumerate(a):
        best_j, best_iou = -1, 0.0
        for j, (box_b, _, _) in enumerate(b):
            if j in used:
                continue
            iou = box_iou_xyxy(box_a, box_b)
            if iou > best_iou:
                best_j, best_iou = j, iou
        if best_j >= 0 and best_iou >= iou_thresh:
            used.add(best_j)
            pairs.append((i, best_j))
    return pairs


def check(
    onnx_path: Path,
    image_path: Path,
    checkpoint: Optional[Path] = None,
    conf_thresh: float = 0.25,
) -> dict[str, Any]:
    """Verify ONNX outputs at b=1 and b=4 match raw decode_face_det_lite() within tolerances."""
    import cv2
    import numpy as np
    import onnxruntime as ort

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from benchmarks.face_bench_adapters import decode_face_det_lite, decode_yolov8_face

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    canvas_bgr = _letterbox_bgr(img_bgr, INPUT_W, INPUT_H)
    rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob_rgb = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])

    # BT.601 luminance matching the ONNX Conv2d frontend.
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.float32)
    core = load_core_model(resolve_checkpoint(checkpoint))
    with torch.no_grad():
        hm, box, lmk = core(torch.from_numpy(gray[None, None, :, :]))
    ref_dets = decode_face_det_lite(
        hm.numpy(),
        box.numpy(),
        lmk.numpy(),
        conf_thresh=conf_thresh,
        nms_iou=0.45,
        stride=STRIDE,
        hm_is_logits=False,
    )
    if not ref_dets:
        return {"ok": False, "reason": f"reference model found 0 faces in {image_path}"}

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_b1 = sess.run(None, {in_name: blob_rgb})[0]
    out_b4 = sess.run(None, {in_name: np.repeat(blob_rgb, 4, axis=0)})[0]

    if out_b1.shape != (1, 4800, 15) or out_b4.shape != (4, 4800, 15):
        return {"ok": False, "reason": f"unexpected shapes b1={out_b1.shape}, b4={out_b4.shape}"}

    batch_diff = float(np.abs(out_b4 - np.repeat(out_b1, 4, axis=0)).max())

    max_box_err = 0.0
    max_lmk_err = 0.0
    max_score_err = 0.0
    for b_idx in range(4):
        onnx_dets = decode_yolov8_face(out_b4[b_idx], conf_thresh=conf_thresh, iou_thresh=0.45)
        pairs = _match_detections(onnx_dets, ref_dets)
        if len(onnx_dets) != len(ref_dets) or len(pairs) != len(ref_dets):
            return {
                "ok": False,
                "reason": f"detection mismatch at batch[{b_idx}]: onnx={len(onnx_dets)}, ref={len(ref_dets)}, paired={len(pairs)}",
            }
        for i, j in pairs:
            o_box, o_score, o_kpts = onnx_dets[i]
            r_box, r_score, r_kpts = ref_dets[j]
            max_box_err = max(max_box_err, float(np.abs(o_box - r_box).max()))
            if o_kpts is not None and r_kpts is not None:
                max_lmk_err = max(max_lmk_err, float(np.abs(o_kpts - r_kpts).max()))
            max_score_err = max(max_score_err, abs(o_score - r_score))

    ok = max_box_err <= 1.0 and max_lmk_err <= 1.0 and max_score_err <= 1e-4 and batch_diff <= 1e-4
    return {
        "ok": ok,
        "onnx": str(onnx_path),
        "image": str(image_path),
        "detections": len(ref_dets),
        "shape_b1": list(out_b1.shape),
        "shape_b4": list(out_b4.shape),
        "max_box_err_px": round(max_box_err, 6),
        "max_landmark_err_px": round(max_lmk_err, 6),
        "max_score_err": round(max_score_err, 8),
        "max_batch_diff": round(batch_diff, 8),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to QFD_V3_superlite_060525_50_pickled.pt")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=OPSET)
    parser.add_argument("--check", type=Path, default=None, help="Image path to verify b=1 and b=4 ONNX parity")
    args = parser.parse_args(argv)

    target_out = Path(args.out)
    export_target = target_out.with_name(target_out.stem + ".proving.onnx") if args.check is not None else target_out
    out_path = export(checkpoint=args.checkpoint, out=export_target, opset=args.opset)

    if args.check is not None:
        report = check(out_path, args.check, checkpoint=args.checkpoint)
        if not report.get("ok"):
            out_path.unlink(missing_ok=True)
            for k, v in report.items():
                print(f"{k}: {v}")
            return 1
        os.replace(out_path, target_out)
        out_path = target_out
        report["onnx"] = str(out_path)

    sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"exported {out_path} ({out_path.stat().st_size} bytes, sha256={sha256})")
    if args.check is not None:
        for k, v in report.items():
            print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
