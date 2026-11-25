import cv2, os, glob, numpy as np
import onnxruntime as ort

ONNX = '/workspace/iris/products/el-mohandes/models/finetuned/best.onnx'
FRAME_DIR = '/workspace/iris/data/rtsp_frames'
sess = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
inp_name = sess.get_inputs()[0].name
print(f"Model: {sess.get_inputs()[0].shape}")

def letterbox(img, ns=1280, pv=114):
    h,w=img.shape[:2]; r=min(ns/h,ns/w); nw,nh=int(round(w*r)),int(round(h*r))
    resized=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_LINEAR) if (w,h)!=(nw,nh) else img
    padded=np.full((ns,ns,3),pv,dtype=np.uint8); dy,dx=(ns-nh)//2,(ns-nw)//2
    padded[dy:dy+nh,dx:dx+nw]=resized
    return padded

def pp(frame, pad=114):
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    p=letterbox(rgb,1280,pv=pad)
    return np.expand_dims((p.astype(np.float32)/255.0).transpose(2,0,1),0)

def n_dets(out, th=0.10):
    pred=out[0]
    if pred.ndim==3: pred=pred[0]
    conf = 1.0/(1.0+np.exp(-pred[4]))
    return int(np.sum(conf>th))

# Find uploaded video files in the container
video_paths = {}
for mp4 in glob.glob('/workspace/iris/data/uploads/*.mp4'):
    name = os.path.basename(mp4)
    if 'D41' in name or '1-' in name: video_paths['cam-1'] = mp4
    elif 'D52' in name or '2-' in name: video_paths['cam-2'] = mp4
    elif 'CH23' in name or '3-' in name: video_paths['cam-3'] = mp4

# Also check the go2rtc source videos
for mp4 in glob.glob('/workspace/iris/helper-repos/**/test_streamer/**/*.mp4', recursive=True):
    pass

# Check the actual streamed files
for mp4 in glob.glob('/workspace/iris/data/**/*.mp4', recursive=True):
    name = os.path.basename(mp4).lower()
    if 'd41' in name: video_paths['cam-1'] = mp4
    elif 'd52' in name: video_paths['cam-2'] = mp4
    elif 'ch23' in name: video_paths['cam-3'] = mp4

print(f"\nVideos found: {video_paths}")

print("\n" + "="*70)
print("TEST: Does mux downscaling lose plate detail?")
print("  Path A (ultralytics/RTSP): Read at native res -> letterbox to 1280x1280")
print("  Path B (DeepStream/mux):   Read at native res -> resize to 1280x720 -> letterbox to 1280x1280")
print("  Path C (what nvinfer sees): Read at native res -> resize to 1280x720 -> pad=0 -> letterbox to 1280x1280")
print("="*70)

for cam, vpath in video_paths.items():
    if not os.path.exists(vpath):
        print(f"\n{cam}: video not found at {vpath}")
        continue
    
    cap = cv2.VideoCapture(vpath)
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n{cam}: {vpath}")
    print(f"  Native: {native_w}x{native_h} @ {fps}fps, {total_frames} frames")
    
    skip = max(1, int(fps / 5))  # sample at ~5fps
    a=0; b=0; c=0; t=0
    
    for i in range(0, min(total_frames, 200), skip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: continue
        t+=1
        
        # Path A: ultralytics style (native res -> letterbox to 1280x1280)
        a += 1 if n_dets(sess.run(None,{inp_name:pp(frame,114)}))>0 else 0
        
        # Path B: DeepStream mux downscales to 1280x720, then letterbox
        mux_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        b += 1 if n_dets(sess.run(None,{inp_name:pp(mux_frame,114)}))>0 else 0
        
        # Path C: what nvinfer currently sees (1280x720 + pad=0)
        c += 1 if n_dets(sess.run(None,{inp_name:pp(mux_frame,0)}))>0 else 0
        
        if t <= 3:
            # Show detection details
            out_a = sess.run(None,{inp_name:pp(frame,114)})[0]
            out_b = sess.run(None,{inp_name:pp(mux_frame,114)})[0]
            if out_a.ndim==3: out_a=out_a[0]; out_b=out_b[0]
            ca = 1.0/(1.0+np.exp(-out_a[4])); cb = 1.0/(1.0+np.exp(-out_b[4]))
            top_a = np.max(ca); top_b = np.max(cb)
            print(f"    frame {i}: native={np.sum(ca>0.10)}dets(top={top_a:.3f}) mux={np.sum(cb>0.10)}dets(top={top_b:.3f})")
    
    cap.release()
    print(f"\n  RESULTS ({t} frames):")
    print(f"    A (native res, pad=114): {a}/{t} = {100*a/t:.0f}%")
    print(f"    B (mux 1280x720, pad=114): {b}/{t} = {100*b/t:.0f}%")
    print(f"    C (mux 1280x720, pad=0  ): {c}/{t} = {100*c/t:.0f}%")
    if a > b * 1.3:
        print(f"    >>> MUX DOWNSCALING KILLS DETECTIONS ({a} -> {b})")
    elif b == a:
        print(f"    >>> Mux downscaling is fine — same detections")
    else:
        print(f"    >>> Minor difference from downscaling")

print("\n" + "="*70)
print("DIAGNOSIS SUMMARY:")
print("  If A>>B: mux downscaling loses plate detail -> increase mux resolution")
print("  If B==C: padding is irrelevant (confirmed earlier)")
print("  If B==A: both preprocessing and downscaling are fine -> TRT engine is the issue")
print("="*70)
