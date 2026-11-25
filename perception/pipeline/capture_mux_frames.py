#!/usr/bin/env python3
"""Compare ONNX Runtime vs TRT (from probe log) on the EXACT same frames.

The pgie probe saves JPEG frames from the mux output (NVDEC-decoded, scaled
to 1280x720, converted via nvvideoconvert to RGBA). These are the exact frames
nvinfer processes. We also log the TRT raw detections on those same frames.

This script reads both and compares: does ONNX detect on the frames where TRT misses?
"""
import os, sys, json, glob, cv2, numpy as np
import onnxruntime as ort

ONNX = "/workspace/iris/products/el-mohandes/models/finetuned/best.onnx"
PROBE_FRAMES = "/workspace/iris/data/raw_frames"
RAW_DET_LOG = "/workspace/iris/data/raw_detections.jsonl"

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name

def lb(img, ns=1280, pv=114):
    h,w=img.shape[:2]; r=min(ns/h,ns/w); nw,nh=int(round(w*r)),int(round(h*r))
    resized=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_LINEAR) if (w,h)!=(nw,nh) else img
    padded=np.full((ns,ns,3),pv,dtype=np.uint8); dy,dx=(ns-nh)//2,(ns-nw)//2
    padded[dy:dy+nh,dx:dx+nw]=resized
    return padded

def pp(frame, pad=114):
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    p=lb(rgb,1280,pv=pad)
    return np.expand_dims((p.astype(np.float32)/255.0).transpose(2,0,1),0)

def n_dets(out, th=0.10):
    pred=out[0]
    if pred.ndim==3: pred=pred[0]
    return int(np.sum(pred[4]>th))

def top_conf(out, th=0.10):
    pred=out[0]
    if pred.ndim==3: pred=pred[0]
    mask=pred[4]>th
    return float(pred[4][mask].max()) if mask.any() else 0.0

# Load TRT raw detections from the probe log
trt_dets = {}  # (cam, frame_num) -> n_objects
if os.path.exists(RAW_DET_LOG):
    with open(RAW_DET_LOG) as f:
        for line in f:
            e = json.loads(line)
            pad = e["pad"]
            cam = f"cam-{pad+1}"
            frame_num = e["frame"]
            trt_dets[(cam, frame_num)] = e["n_objects"]

print(f"Loaded {len(trt_dets)} TRT detection entries from probe log")

# For each saved probe frame, run ONNX and compare with TRT
for cam in ["cam-1", "cam-2", "cam-3"]:
    jpgs = sorted(glob.glob(os.path.join(PROBE_FRAMES, f"{cam}_frame*.jpg")))
    if not jpgs:
        print(f"\n{cam}: no probe frames found")
        continue
    
    print(f"\n{'='*70}")
    print(f"  {cam}: {len(jpgs)} probe frames")
    print(f"{'='*70}")
    
    # Match probe frames with TRT detections by frame number
    both_match = 0    # TRT detects AND ONNX detects
    trt_only = 0      # TRT detects but ONNX doesn't  
    onnx_only = 0     # ONNX detects but TRT doesn't
    neither = 0        # Neither detects
    total = 0
    
    for jpg in jpgs[:50]:
        fname = os.path.basename(jpg)
        # Extract frame number from filename: cam-1_frame000030.jpg -> 30
        try:
            frame_num = int(fname.split("_frame")[1].split(".")[0])
        except:
            continue
        
        frame = cv2.imread(jpg)
        if frame is None:
            continue
        
        # ONNX detection
        out = sess.run(None, {inp: pp(frame, 114)})[0]
        onnx_n = n_dets(out)
        onnx_c = top_conf(out)
        
        # TRT detection (from probe log)
        trt_n = trt_dets.get((cam, frame_num), -1)
        
        if trt_n < 0:
            continue  # No TRT data for this frame
        
        total += 1
        onnx_det = onnx_n > 0
        trt_det = trt_n > 0
        
        if onnx_det and trt_det: both_match += 1
        elif not onnx_det and trt_det: trt_only += 1
        elif onnx_det and not trt_det: onnx_only += 1
        else: neither += 1
        
        if total <= 20:
            marker = "BOTH" if (onnx_det and trt_det) else ("ONNX" if onnx_det else ("TRT" if trt_det else "---"))
            print(f"  f{frame_num:6d}: TRT={trt_n:2d} ONNX={onnx_n:2d}({onnx_c:.3f}) [{marker}]")
    
    print(f"\n  SUMMARY ({total} matched frames):")
    print(f"    Both detect:     {both_match}/{total} ({100*both_match/max(total,1):.0f}%)")
    print(f"    ONNX only:       {onnx_only}/{total} ({100*onnx_only/max(total,1):.0f}%) ← TRT misses these")
    print(f"    TRT only:        {trt_only}/{total} ({100*trt_only/max(total,1):.0f}%) ← ONNX misses these")
    print(f"    Neither:         {neither}/{total} ({100*neither/max(total,1):.0f}%)")
    
    if onnx_only > 0 and trt_only == 0:
        print(f"\n    >>> TRT misses {onnx_only} frames that ONNX catches")
        print(f"        Same pixels, same preprocessing → TRT ENGINE is the problem")
    elif onnx_only == 0 and trt_only > 0:
        print(f"\n    >>> ONNX misses frames that TRT catches → unlikely")
    elif onnx_only == 0 and trt_only == 0:
        print(f"\n    >>> IDENTICAL results → decode+preprocessing is fine, gap is elsewhere")
    else:
        print(f"\n    >>> Both miss different frames → numerical precision differences")

print("\n" + "="*70)
print("CONCLUSION:")
print("  If ONNX_only >> 0: TRT engine loses detections on the SAME pixels")
print("  If both == 100%: decode+preprocessing is fine, gap must be frame rate (analysis_fps)")
print("  If TRT_only >> 0: TRT hallucinates detections (unlikely)")
print("="*70)
