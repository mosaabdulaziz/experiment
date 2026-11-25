#!/usr/bin/env python3
"""Capture frames from RTSP via GStreamer NVDEC and save as JPEG.
Also capture via OpenCV (FFmpeg) for comparison.

This isolates the decode stage: NVDEC vs FFmpeg on the same RTSP stream.
"""
import os, sys, time, cv2, numpy as np

MUX_W, MUX_H = 1280, 720
RTSP_URLS = {
    "cam-1": "rtsp://go2rtc:8554/raw-cam-1",
    "cam-2": "rtsp://go2rtc:8554/raw-cam-2",
    "cam-3": "rtsp://go2rtc:8554/raw-cam-3",
}
N_FRAMES = 20

def capture_gstreamer_nvdec(rtsp_url, cam_id, n_frames=20):
    """Capture frames via GStreamer NVDEC pipeline."""
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GLib', '2.0')
    from gi.repository import Gst, GLib
    
    Gst.init(None)
    
    out_dir = f"/workspace/iris/data/nvdec_frames/{cam_id}"
    os.makedirs(out_dir, exist_ok=True)
    
    # Build pipeline: uridecodebin -> videoconvert -> appsink
    pipeline_str = (
        f"uridecodebin uri={rtsp_url} ! "
        f"video/x-raw(memory:NVMM) ! "
        f"nvvideoconvert ! "
        f"video/x-raw,format=RGBA,width={MUX_W},height={MUX_H} ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink emit-signals=True max-buffers=1 drop=True"
    )
    
    pipeline = Gst.parse_launch(pipeline_str)
    appsink = pipeline.get_by_name("appsink0")
    if not appsink:
        # Find the appsink element
        for elem in pipeline.iterate_elements():
            if elem.get_factory().get_name() == "appsink":
                appsink = elem
                break
    
    if not appsink:
        print(f"  {cam_id}: Could not find appsink in pipeline")
        return 0
    
    saved = 0
    loop = GLib.MainLoop()
    
    def on_new_sample(sink):
        nonlocal saved
        if saved >= n_frames:
            return True
        sample = sink.emit("pull-sample")
        if sample is None:
            return True
        buf = sample.get_buffer()
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return True
        
        caps = sample.get_caps()
        struct = caps.get_structure(0)
        w = struct.get_value("width")
        h = struct.get_value("height")
        
        # Convert to numpy array
        frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape(h, w, 3)
        buf.unmap(map_info)
        
        fname = f"{out_dir}/{cam_id}_nvdec_{saved:04d}.jpg"
        cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        saved += 1
        print(f"  {cam_id}: saved NVDEC frame {saved}/{n_frames} ({w}x{h})")
        
        if saved >= n_frames:
            # Stop after capturing enough frames
            GLib.idle_add(loop.quit)
        return True
    
    appsink.connect("new-sample", on_new_sample)
    
    pipeline.set_state(Gst.State.PLAYING)
    
    # Run for up to 30 seconds
    import threading
    def timeout():
        time.sleep(30)
        GLib.idle_add(loop.quit)
    t = threading.Thread(target=timeout, daemon=True)
    t.start()
    
    try:
        loop.run()
    except:
        pass
    
    pipeline.set_state(Gst.State.NULL)
    return saved

def capture_opencv_ffmpeg(rtsp_url, cam_id, n_frames=20):
    """Capture frames via OpenCV (FFmpeg decoder)."""
    out_dir = f"/workspace/iris/data/ffmpeg_frames/{cam_id}"
    os.makedirs(out_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    saved = 0
    for i in range(n_frames * 3):
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        saved += 1
        if saved > n_frames:
            break
        
        # Scale to mux resolution
        scaled = cv2.resize(frame, (MUX_W, MUX_H), interpolation=cv2.INTER_LINEAR)
        fname = f"{out_dir}/{cam_id}_ffmpeg_{saved:04d}.jpg"
        cv2.imwrite(fname, scaled, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
    cap.release()
    return saved

def main():
    print("="*70)
    print("CAPTURING FRAMES: NVDEC (GStreamer) vs FFmpeg (OpenCV)")
    print("="*70)
    
    for cam, url in RTSP_URLS.items():
        print(f"\n--- {cam} ---")
        
        # Capture via NVDEC
        print(f"  Capturing NVDEC frames from {url}...")
        nvdec_n = capture_gstreamer_nvdec(url, cam, N_FRAMES)
        print(f"  NVDEC: saved {nvdec_n} frames")
        
        # Capture via FFmpeg
        print(f"  Capturing FFmpeg frames from {url}...")
        ffmpeg_n = capture_opencv_ffmpeg(url, cam, N_FRAMES)
        print(f"  FFmpeg: saved {ffmpeg_n} frames")
    
    # Now run ONNX on both sets
    print("\n" + "="*70)
    print("RUNNING ONNX ON BOTH SETS")
    print("="*70)
    
    import onnxruntime as ort
    ONNX = "/workspace/iris/products/el-mohandes/models/finetuned/best.onnx"
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
    
    def nd(out, th=0.10):
        pred=out[0]
        if pred.ndim==3: pred=pred[0]
        return int(np.sum(pred[4]>th))
    
    import glob
    
    for cam in RTSP_URLS:
        print(f"\n--- {cam} ---")
        
        # Test NVDEC frames
        nvdec_dir = f"/workspace/iris/data/nvdec_frames/{cam}"
        nvdec_jpgs = sorted(glob.glob(os.path.join(nvdec_dir, "*.jpg")))
        nvdec_dets = 0
        for jpg in nvdec_jpgs:
            f = cv2.imread(jpg)
            if f is None: continue
            if nd(sess.run(None, {inp: pp(f, 114)})) > 0:
                nvdec_dets += 1
        
        # Test FFmpeg frames
        ffmpeg_dir = f"/workspace/iris/data/ffmpeg_frames/{cam}"
        ffmpeg_jpgs = sorted(glob.glob(os.path.join(ffmpeg_dir, "*.jpg")))
        ffmpeg_dets = 0
        for jpg in ffmpeg_jpgs:
            f = cv2.imread(jpg)
            if f is None: continue
            if nd(sess.run(None, {inp: pp(f, 114)})) > 0:
                ffmpeg_dets += 1
        
        print(f"  NVDEC frames: {nvdec_dets}/{len(nvdec_jpgs)} = {100*nvdec_dets/max(len(nvdec_jpgs),1):.0f}%")
        print(f"  FFmpeg frames: {ffmpeg_dets}/{len(ffmpeg_jpgs)} = {100*ffmpeg_dets/max(len(ffmpeg_jpgs),1):.0f}%")
        
        if nvdec_dets == ffmpeg_dets:
            print(f"  >>> IDENTICAL — NVDEC decode is fine, problem is TRT engine")
        elif nvdec_dets < ffmpeg_dets:
            print(f"  >>> NVDEC loses {ffmpeg_dets - nvdec_dets} detections — decode quality matters")
        else:
            print(f"  >>> FFmpeg loses detections (unexpected)")

if __name__ == "__main__":
    main()
