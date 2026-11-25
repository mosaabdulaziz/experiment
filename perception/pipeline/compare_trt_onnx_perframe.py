#!/usr/bin/env python3
"""Per-frame TRT vs ONNX comparison.

Reads the pipeline JPEGs saved by the probe and the TRT detection log,
then runs ONNX on the same frames and compares.
"""
import json
import glob
import cv2
import numpy as np
import onnxruntime as ort
import redis
import sys

MODEL_PATH = "/workspace/iris/products/el-mohandes/models/finetuned/best.onnx"
RAW_LOG = "/workspace/iris/data/raw_detections.jsonl"
MUX_W, MUX_H = 1280, 720

def preprocess(frame, target=1280):
    h, w = frame.shape[:2]
    scale = min(target / h, target / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    t = canvas.astype(np.float32) / 255.0
    return np.expand_dims(t.transpose(2, 0, 1), 0).astype(np.float32)


def onnx_dets(tensor, sess, inp, conf_thresh=0.10):
    out = sess.run(None, {inp: tensor})[0]
    pred = out[0]
    if pred.ndim == 3:
        pred = pred[0]
    scores = pred[4]
    mask = scores > conf_thresh
    return int(mask.sum()), float(scores[mask].max()) if mask.any() else 0.0


def main():
    # Load ONNX model
    print("Loading ONNX model...")
    sess = ort.InferenceSession(MODEL_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    r = redis.Redis(host="redis")

    # Read TRT raw detections
    trt_by_cam = {}  # cam -> [(frame_num, n_dets, max_conf)]
    try:
        with open(RAW_LOG) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    cam = d.get("cam", "?")
                    fn = d.get("frame", 0)
                    n_det = d.get("n_det", 0)
                    confs = d.get("confs", [])
                    max_c = max(confs) if confs else 0.0
                    if cam not in trt_by_cam:
                        trt_by_cam[cam] = []
                    trt_by_cam[cam].append((fn, n_det, max_c))
                except:
                    pass
    except FileNotFoundError:
        print(f"Raw detection log not found: {RAW_LOG}")

    for cam in ["cam-1", "cam-2", "cam-3"]:
        trt_frames = trt_by_cam.get(cam, [])
        total = len(trt_frames)
        trt_with_dets = sum(1 for _, n, _ in trt_frames if n > 0)

        # Get pipeline JPEGs saved by the probe
        probe_dir = f"/workspace/iris/data/probe_frames/{cam}"
        probe_files = sorted(glob.glob(f"{probe_dir}/*.jpg"))

        # Also try the latest JPEG from Redis
        jpeg = r.get(f"iris:jpeg:{cam}")
        redis_frame = None
        if jpeg:
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            redis_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        print(f"\n{'='*60}")
        print(f"  {cam}")
        print(f"{'='*60}")
        print(f"  TRT raw detections: {trt_with_dets}/{total} frames ({100*trt_with_dets/max(total,1):.0f}%)")

        # Run ONNX on probe frames if available
        if probe_files:
            onnx_hits = 0
            trt_hits_on_same = 0
            print(f"\n  Per-frame comparison ({len(probe_files)} probe frames):")
            print(f"  {'Frame':>8} | {'TRT dets':>8} | {'ONNX dets':>9} | {'ONNX max':>9} | Match")
            print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*6}")
            for pf in probe_files[:30]:
                fname = pf.split("/")[-1]
                # Extract frame number from filename
                try:
                    fn = int(fname.replace("frame_", "").replace(".jpg", ""))
                except:
                    fn = 0

                frame = cv2.imread(pf)
                if frame is None:
                    continue

                # ONNX on this exact frame
                tensor = preprocess(frame)
                n_onnx, max_onnx = onnx_dets(tensor, sess, inp)

                # Find matching TRT entry
                trt_entry = next(((n, c) for f, n, c in trt_frames if f == fn), None)
                if trt_entry:
                    n_trt, conf_trt = trt_entry
                else:
                    n_trt, conf_trt = 0, 0.0

                match = "✅" if (n_onnx > 0) == (n_trt > 0) else "❌"
                if n_onnx > 0 and n_trt == 0:
                    onnx_hits += 1
                if n_trt > 0:
                    trt_hits_on_same += 1

                print(f"  {fn:>8} | {n_trt:>8} | {n_onnx:>9} | {max_onnx:>9.3f} | {match}")

            print(f"\n  TRT detects: {trt_hits_on_same}/{len(probe_files[:30])}")
            print(f"  ONNX detects: {onnx_hits + trt_hits_on_same}/{len(probe_files[:30])}")
            if onnx_hits > 0:
                print(f"  ⚠️ TRT misses {onnx_hits} frames that ONNX catches (same pixels!)")

        elif redis_frame is not None:
            # Fallback: compare on latest frame
            tensor = preprocess(redis_frame)
            n_onnx, max_onnx = onnx_dets(tensor, sess, inp)
            print(f"\n  Latest frame comparison (1 frame):")
            print(f"  ONNX: {n_onnx} detections (max_conf={max_onnx:.3f})")
            print(f"  TRT (latest): {trt_frames[-1] if trt_frames else 'none'}")
        else:
            print(f"  No frames available for comparison")


if __name__ == "__main__":
    main()
