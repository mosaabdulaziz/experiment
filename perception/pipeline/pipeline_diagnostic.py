#!/usr/bin/env python3
"""Systematic pipeline diagnostic: capture raw mux output and compare with ultralytics.

This script runs INSIDE the Docker container. It:
1. Reads the raw frames saved by the pgie src-pad probe
2. Reconstructs the nvinfer preprocessing (NV12→RGB, letterbox, scale)
3. Feeds both the nvinfer-processed tensor AND the ultralytics-processed 
   tensor to the ONNX model via onnxruntime
4. Compares outputs

Key question: does the TRT engine (inside nvinfer) produce the same outputs
as onnxruntime on the same pixels?
"""
import os, sys, json, glob
import numpy as np
import cv2

# ── Load model metadata ──────────────────────────────────────────────────
ONNX_PATH = "/workspace/iris/products/el-mohandes/models/finetuned/best.onnx"
FRAME_DIR = os.environ.get("RAW_FRAME_DIR", "/workspace/iris/data/raw_frames")
NATIVE_VIDEOS = {
    "cam-1": "/workspace/iris/products/el-mohandes/videos/experiment/D41.mp4",
    "cam-2": "/workspace/iris/products/el-mohandes/videos/experiment/D52.mp4",
    "cam-3": "/workspace/iris/products/el-mohandes/videos/experiment/CH23.mp4",
}
MUX_W = 1280
MUX_H = 720
MODEL_INPUT = 1280

def letterbox(img, new_shape=MODEL_INPUT, pad_value=114):
    """Letterbox resize preserving aspect ratio (ultralytics style)."""
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

def preprocess_ultralytics(frame_bgr):
    """ultralytics preprocessing: BGR→RGB, letterbox(pad=114), float/255, NCHW."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    padded = letterbox(rgb, MODEL_INPUT, pad_value=114)
    tensor = padded.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(tensor, 0)    # NCHW

def preprocess_nvinfer_style(frame_bgr, pad_value=0):
    """Simulate nvinfer preprocessing: BGR→RGB, letterbox(pad=0), float/255, NCHW."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    padded = letterbox(rgb, MODEL_INPUT, pad_value=pad_value)
    tensor = padded.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(tensor, 0)    # NCHW

def postprocess_onnx(output, conf_thresh=0.10, iou_thresh=0.45):
    """Very basic YOLO output postprocessing for comparison."""
    # output shape: [1, 5, 33600] -> transpose to [1, 33600, 5]
    pred = output[0].T  # [33600, 5]
    # Each row: [cx, cy, w, h, conf]
    # No class dimension since num_classes=1
    scores = pred[:, 4]
    mask = scores > conf_thresh
    filtered = pred[mask]
    if len(filtered) == 0:
        return []
    # Sort by confidence
    order = np.argsort(-filtered[:, 4])
    filtered = filtered[order[:300]]  # top 300
    return filtered.tolist()

def main():
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed. Install with: pip install onnxruntime")
        sys.exit(1)
    
    # Create session
    print(f"Loading ONNX model: {ONNX_PATH}")
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    
    print(f"Model input: {sess.get_inputs()[0].shape}")
    print(f"Model output: {sess.get_outputs()[0].shape}")
    
    results = {}
    
    # ── TEST 1: Compare on the raw mux frames (saved by probe) ───────────
    print("\n" + "="*70)
    print("TEST 1: On mux frames (saved by pgie probe)")
    print("="*70)
    
    for cam_id in ["cam-1", "cam-2", "cam-3"]:
        jpgs = sorted(glob.glob(os.path.join(FRAME_DIR, cam_id, "*.jpg")))
        if not jpgs:
            print(f"\n  {cam_id}: no frames found")
            continue
        
        print(f"\n  {cam_id}: {len(jpgs)} frames")
        ultra_dets = 0
        nv_dets = 0
        total = 0
        
        for jpg in jpgs[:30]:  # first 30 frames
            frame = cv2.imread(jpg)
            if frame is None or frame.shape[1] != MUX_W:
                continue
            total += 1
            
            # ultralytics preprocessing (pad=114)
            ultra_input = preprocess_ultralytics(frame)
            ultra_out = sess.run(None, {input_name: ultra_input})[0]
            ultra_post = postprocess_onnx(ultra_out)
            
            # nvinfer-style preprocessing (pad=0)
            nv_input = preprocess_nvinfer_style(frame, pad_value=0)
            nv_out = sess.run(None, {input_name: nv_input})[0]
            nv_post = postprocess_onnx(nv_out)
            
            # Also try pad=114 (same as ultralytics)
            nv114_input = preprocess_nvinfer_style(frame, pad_value=114)
            nv114_out = sess.run(None, {input_name: nv114_input})[0]
            nv114_post = postprocess_onnx(nv114_out)
            
            if ultra_post: ultra_dets += 1
            if nv_post: nv_dets += 1
            
            # Print comparison for first 5 frames
            if total <= 5:
                ultra_confs = [d[4] for d in ultra_post]
                nv_confs = [d[4] for d in nv_post]
                nv114_confs = [d[4] for d in nv114_post]
                print(f"    frame {total}: ultra={len(ultra_post)}dets{ultra_confs} | "
                      f"nv0={len(nv_post)}dets{nv_confs} | "
                      f"nv114={len(nv114_post)}dets{nv114_confs}")
        
        print(f"\n  SUMMARY ({total} frames):")
        print(f"    ultralytics (pad=114): {ultra_dets}/{total} ({100*ultra_dets/max(total,1):.0f}%)")
        print(f"    nvinfer-style (pad=0): {nv_dets}/{total} ({100*nv_dets/max(total,1):.0f}%)")
        print(f"    nvinfer-style (pad=114): {nv114_dets}/{total} ({100*nv114_dets/max(total,1):.0f}%)" if 'nv114_dets' in dir() else "")
        results[cam_id] = {"ultra": ultra_dets, "nv0": nv_dets, "total": total}
    
    # ── TEST 2: Compare on native-resolution frames (as offline does) ────
    print("\n" + "="*70)
    print("TEST 2: On native video frames (offline-style)")
    print("="*70)
    
    for cam_id, video_path in NATIVE_VIDEOS.items():
        if not os.path.exists(video_path):
            print(f"\n  {cam_id}: video not found at {video_path}")
            continue
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\n  {cam_id}: {video_path}")
        print(f"    Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"    FPS: {fps}, Total frames: {total_frames}")
        
        ultra_dets = 0
        nv_dets = 0
        total = 0
        skip = max(1, int(fps / 10))  # sample at ~10fps
        
        for i in range(0, min(total_frames, 300), skip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            total += 1
            
            # ultralytics on full-res (like offline)
            ultra_input = preprocess_ultralytics(frame)
            ultra_out = sess.run(None, {input_name: ultra_input})[0]
            ultra_post = postprocess_onnx(ultra_out)
            
            # nvinfer-style on full-res
            nv_input = preprocess_nvinfer_style(frame, pad_value=0)
            nv_out = sess.run(None, {input_name: nv_input})[0]
            nv_post = postprocess_onnx(nv_out)
            
            if ultra_post: ultra_dets += 1
            if nv_post: nv_dets += 1
            
            if total <= 3:
                ultra_confs = [d[4] for d in ultra_post]
                nv_confs = [d[4] for d in nv_post]
                print(f"    frame {i}: ultra={len(ultra_post)}dets{ultra_confs} | nv0={len(nv_post)}dets{nv_confs}")
        
        cap.release()
        print(f"\n  SUMMARY ({total} frames sampled):")
        print(f"    ultralytics (pad=114): {ultra_dets}/{total} ({100*ultra_dets/max(total,1):.0f}%)")
        print(f"    nvinfer-style (pad=0): {nv_dets}/{total} ({100*nv_dets/max(total,1):.0f}%)")
    
    print("\n" + "="*70)
    print("DIAGNOSIS:")
    print("  If ultra≈nv114 >> nv0: padding value IS the problem")
    print("  If ultra>>nv114≈nv0:   something ELSE in preprocessing is the problem")
    print("  If ultra≈nv0≈nv114:    preprocessing is fine, problem is in TRT engine itself")
    print("="*70)

if __name__ == "__main__":
    main()
