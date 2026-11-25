#!/usr/bin/env python3
"""Definitive frame-by-frame comparison: TRT (DeepStream) vs ultralytics.

Captures frames from RTSP streams, runs ultralytics on them, and cross-references
with the raw nvinfer detection log to find frames where TRT missed but ultralytics detects.
"""
import json, os, sys, time
import numpy as np
import cv2
from ultralytics import YOLO

RTSP_URLS = {
    "cam-1": "rtsp://localhost:8554/cam-1",
    "cam-2": "rtsp://localhost:8554/cam-2",
    "cam-3": "rtsp://localhost:8554/cam-3",
}

ONNX_PATH = "/home/mmtayee/URCA/IRIS/repos/surveillance-camera-sys/products/el-mohandes/models/finetuned/best.onnx"
NFRAMES = 30  # capture this many frames from each stream

def main():
    model = YOLO(ONNX_PATH)
    
    results = {}
    for cam_id, rtsp_url in RTSP_URLS.items():
        print(f"\n=== {cam_id}: capturing {NFRAMES} frames from {rtsp_url} ===")
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print(f"  FAILED to open {rtsp_url}")
            continue
        
        frames_data = []
        for i in range(NFRAMES):
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            h, w = frame.shape[:2]
            
            # Run ultralytics
            ultra_results = model(frame, imgsz=1280, verbose=False)
            ultra_dets = []
            for r in ultra_results:
                for box in r.boxes:
                    ultra_dets.append({
                        "conf": float(box.conf[0]),
                        "cls": int(box.cls[0]),
                        "xyxy": box.xyxy[0].tolist(),
                    })
            
            frames_data.append({
                "frame_idx": i,
                "resolution": f"{w}x{h}",
                "ultra_dets": ultra_dets,
                "ultra_count": len(ultra_dets),
            })
            
            status = f"frame {i}: {w}x{h}, ultra={len(ultra_dets)} dets"
            if ultra_dets:
                status += f" [conf={ultra_dets[0]['conf']:.3f}]"
            print(f"  {status}")
        
        cap.release()
        results[cam_id] = frames_data
    
    # Now compare with TRT raw detection log
    print("\n" + "="*70)
    print("COMPARISON WITH TRT (DeepStream) OUTPUT")
    print("="*70)
    
    raw_log = "data/raw_detections.jsonl"  # the dump is no longer versioned; capture one first
    if os.path.exists(raw_log):
        trt_frames = {}
        with open(raw_log) as f:
            for line in f:
                entry = json.loads(line)
                pad = entry["pad"]
                cam = f"cam-{pad+1}"
                if cam not in trt_frames:
                    trt_frames[cam] = []
                trt_frames[cam].append(entry)
        
        for cam_id in RTSP_URLS:
            ultra = results.get(cam_id, [])
            trt = trt_frames.get(cam_id, [])
            
            ultra_det_count = sum(1 for f in ultra if f["ultra_count"] > 0)
            trt_det_count = sum(1 for f in trt if f["n_objects"] > 0)
            
            print(f"\n  {cam_id}:")
            print(f"    Ultralytics: {ultra_det_count}/{len(ultra)} frames with detections ({100*ultra_det_count/max(len(ultra),1):.0f}%)")
            print(f"    TRT (DeepStream): {trt_det_count}/{len(trt)} frames with detections ({100*trt_det_count/max(len(trt),1):.0f}%)")
            
            # Per-frame comparison
            print(f"\n    Per-frame detail (ultralytics):")
            for f in ultra[:20]:
                dets = f["ultra_dets"]
                if dets:
                    plate_dets = [d for d in dets if d["cls"] == 0]
                    confs = [d["conf"] for d in plate_dets]
                    print(f"      frame {f['frame_idx']}: {len(plate_dets)} plate(s) conf={confs}")
                else:
                    print(f"      frame {f['frame_idx']}: 0 detections")
            
            print(f"\n    Per-frame detail (TRT):")
            for f in trt[:20]:
                objs = f.get("objects", [])
                plate_objs = [o for o in objs if "plate" in o.get("label", "").lower() or o.get("label", "") == "License_Plate"]
                if plate_objs:
                    confs = [o["conf"] for o in plate_objs]
                    print(f"      frame {f['frame']}: {len(plate_objs)} plate(s) conf={confs}")
                else:
                    print(f"      frame {f['frame']}: 0 detections")
    else:
        print(f"\n  No TRT raw log found at {raw_log}")
    
    # Save detailed results
    out_path = "/home/mmtayee/URCA/IRIS/repos/surveillance-camera-sys/mtmc/output/preprocess_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
