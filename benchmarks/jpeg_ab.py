"""A/B: GpuJpegEncoder (nvjpegenc) vs cv2.imencode on identical frames.

Runs inside the DeepStream sidecar container (needs gi/pyds/nvjpegenc):

    docker exec <sidecar> python3 benchmarks/jpeg_ab.py

Prints ms/frame for both paths on 1280x720 noisy frames with drawn boxes, plus
the resulting JPEG byte sizes. The GPU path also uploads the BGR frame, so this
is the full end-to-end cost the publisher pays per published frame.
"""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "perception/pipeline")
from jpeg_encoder import GpuJpegEncoder  # noqa: E402

W, H, QUALITY, N = 1280, 720, 80, 60

rng = np.random.default_rng(0)
frame = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
cv2.rectangle(frame, (100, 100), (300, 300), (0, 0, 255), 2)
cv2.putText(frame, "person 0.95 #3", (100, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

enc = GpuJpegEncoder(W, H, QUALITY)

# warmup both paths
assert enc.encode(frame) is not None
cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])

t0 = time.perf_counter()
for _ in range(N):
    enc.encode(frame)
t_gpu = (time.perf_counter() - t0) / N * 1000

t0 = time.perf_counter()
for _ in range(N):
    cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
t_cpu = (time.perf_counter() - t0) / N * 1000

j_gpu = enc.encode(frame)
ok, j_cpu = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
enc.close()

print(f"gpu nvjpegenc : {t_gpu:6.2f} ms/frame  size={len(j_gpu) if j_gpu else 0} bytes")
print(f"cpu cv2       : {t_cpu:6.2f} ms/frame  size={len(j_cpu)} bytes")
print(f"speedup       : {t_cpu / t_gpu:5.2f}x")
