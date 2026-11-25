#!/usr/bin/env python3
"""Definitive preprocessing comparison: TRT engine vs ultralytics on identical pixels.

Strategy:
1. Read the RAW frame saved by the pgie src-pad probe (NVMM → RGBA → JPEG)
2. Reconstruct the exact input tensor nvinfer would produce (NV12→RGB, letterbox, scale)
3. Feed that SAME tensor to the TRT engine via Python TensorRT
4. Feed the SAME frame to ultralytics (which does its own preprocessing)
5. Compare outputs AND intermediate tensors

Alternatively (simpler): read the RTSP frames directly, run both pipelines,
and compare at the detection level.  This script does both.
"""
import sys, os, glob, json
import numpy as np
import cv2

# ── 1. Read the raw frames saved by the pgie probe ────────────────────────
FRAME_DIR = "/home/mmtayee/URCA/IRIS/repos/surveillance-camera-sys/mtmc/output/diagnose_frames"
if len(sys.argv) > 1:
    FRAME_DIR = sys.argv[1]

print(f"Reading frames from: {FRAME_DIR}")

for cam_id in ["cam-1", "cam-2", "cam-3"]:
    jpgs = sorted(glob.glob(os.path.join(FRAME_DIR, cam_id, "*.jpg")))
    if not jpgs:
        print(f"\n=== {cam_id}: no frames ===")
        continue
    
    print(f"\n{'='*60}")
    print(f"=== {cam_id}: {len(jpgs)} frames ===")
    print(f"{'='*60}")
    
    for jpg in jpgs[:5]:  # first 5 frames
        fname = os.path.basename(jpg)
        frame = cv2.imread(jpg)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        
        # ── 2. Analyze the frame ──────────────────────────────────────────
        # What does nvinfer see?
        # Frame went through: decode → nvstreammux(1280x720) → nvvideoconvert(RGBA) → probe
        # The probe saved: RGBA → BGR → JPEG
        # So this JPEG represents the mux output (1280x720)
        
        print(f"\n  {fname}: {w}x{h}")
        
        # Check for letterbox padding (black bars = value ~0)
        top_row = frame[0, :, :]  # top row
        bot_row = frame[-1, :, :]  # bottom row
        center = frame[h//2, w//2, :]  # center pixel
        
        top_mean = np.mean(top_row)
        bot_mean = np.mean(bot_row)
        center_mean = np.mean(center)
        
        # Check if there are black bars (padding)
        top_is_black = top_mean < 5
        bot_is_black = bot_mean < 5
        top_is_gray = 100 < top_mean < 130
        bot_is_gray = 100 < bot_mean < 130
        
        print(f"    Top row mean: {top_mean:.1f} {'(BLACK)' if top_is_black else '(GRAY)' if top_is_gray else ''}")
        print(f"    Bot row mean: {bot_mean:.1f} {'(BLACK)' if bot_is_black else '(GRAY)' if bot_is_gray else ''}")
        print(f"    Center pixel: R={center[2]} G={center[1]} B={center[0]}")
        
        # ── 3. Simulate nvinfer preprocessing ──────────────────────────────
        # nvinfer does: NV12→RGB, letterbox(1280x1280, pad=0), scale(*1/255)
        # Our frame is already RGB (from JPEG decode), so simulate:
        #   a) Letterbox to 1280x1280 with pad=0 (black) - nvinfer style
        #   b) Letterbox to 1280x1280 with pad=114 (gray) - ultralytics style
        
        def letterbox(img, new_shape=1280, pad_value=114):
            """Letterbox resize preserving aspect ratio."""
            h, w = img.shape[:2]
            r = min(new_shape / h, new_shape / w)
            new_unpad = (int(round(w * r)), int(round(h * r)))
            
            dw = (new_shape - new_unpad[0]) / 2
            dh = (new_shape - new_unpad[1]) / 2
            
            if (w, h) != new_unpad:
                img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=pad_value)
            return img
        
        # nvinfer style (black padding)
        nv_input = letterbox(frame, 1280, pad_value=0)
        # ultralytics style (gray padding)  
        ultra_input = letterbox(frame, 1280, pad_value=114)
        
        # Pixel difference in padding area
        nv_top = nv_input[0:10, :, :]
        ultra_top = ultra_input[0:10, :, :]
        pad_diff = np.mean(np.abs(nv_top.astype(float) - ultra_top.astype(float)))
        
        print(f"    Padding diff (nvinfer=0 vs ultra=114): {pad_diff:.1f} mean abs")
        
        # ── 4. Compare overall input tensors ───────────────────────────────
        # nvinfer scales by 1/255 (net-scale-factor=0.003921569...)
        nv_float = nv_input.astype(np.float32) / 255.0
        ultra_float = ultra_input.astype(np.float32) / 255.0
        
        # Overall MSE between the two input tensors
        mse = np.mean((nv_float - ultra_float) ** 2)
        max_diff = np.max(np.abs(nv_float - ultra_float))
        
        # How much of the image differs
        diff_mask = np.abs(nv_float - ultra_float) > 0.001
        pct_diff = np.mean(diff_mask) * 100
        
        print(f"    Input tensor MSE (nvinfer vs ultra): {mse:.6f}")
        print(f"    Max pixel diff: {max_diff:.4f}")
        print(f"    Pixels differing > 0.001: {pct_diff:.1f}%")
        
        # Check the actual content area (non-padding)
        # For 16:9 → 1:1 letterbox: content is in the middle
        content_top = int((1280 - 720) / 2)  # 280
        content_bot = content_top + 720  # 1000
        nv_content = nv_float[content_top:content_bot, :, :]
        ultra_content = ultra_float[content_top:content_bot, :, :]
        content_mse = np.mean((nv_content - ultra_content) ** 2)
        print(f"    Content area MSE: {content_mse:.8f} (should be ~0 if identical)")

print("\n" + "="*60)
print("CONCLUSION:")
print("If content area MSE > 0: NV12→RGB conversion differs from JPEG→BGR→RGB")
print("If padding diff is high: padding value mismatch (0 vs 114)")
print("="*60)
