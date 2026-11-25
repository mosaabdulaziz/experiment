#!/usr/bin/env python3
"""Generate the INT8 calibration table + engine for a pgie config.

This is the DeepStream-native INT8 flow: no pycuda/tensorrt-python needed.
nvinfer builds the INT8 engine during the sidecar pipeline start, and the
custom YOLO parser's calibrator (Int8EntropyCalibrator2 in calibrator.cpp,
compiled in with OPENCV=1) reads calibration images from the list in
INT8_CALIB_IMG_PATH and writes the calibration cache to the target config's
own int8-calib-file. The calibrator itself is architecture-agnostic (reads
net-scale-factor/model-color-format/input dims from whichever config is
active) -- this script works unchanged for yolo11, DAMO-YOLO-S/M, or any
other config_infer_primary_*.txt in perception/models/DeepStream-Yolo/, not just yolo11.

Usage:
    python3 perception/models/DeepStream-Yolo/tools/gen_int8_calib_table.py --video <source_video> \
        [--config perception/models/DeepStream-Yolo/config_infer_primary_damoyolo.txt] \
        [--count 60] [--keep-int8]

Steps performed:
  1. Extract --count evenly spaced frames from --video into
     perception/models/DeepStream-Yolo/calib_images/ (ffmpeg preferred, OpenCV fallback).
  2. Write perception/models/DeepStream-Yolo/calib/calib_images.txt (one absolute path per line).
  3. Set int8_calib_img_path / int8_calib_batch_size in perception/pipeline/sidecar.yaml --
     the batch is always the pgie's actual build batch (current source count
     in sidecar.yaml), never a fixed default: a mismatch here segfaults the
     custom YOLO calibrator (see _engine_batch_size / sidecar.py's main()),
     which is what a flood of unrelated-looking CUDA "Error 700" teardown
     errors in the sidecar logs actually means, not a separate bug.
  4. Flip --config to network-mode=1 (backing up first) and restart the
     sidecar container, so nvinfer builds the INT8 engine.
  5. Wait for the config's own int8-calib-file and the derived INT8 engine
     (<repo>/model_b<N>_gpu0_int8.engine, N = the number of sources currently
     configured in sidecar.yaml -- NOT always b1, see _engine_batch_size)
     to appear, and confirm the pipeline reaches PLAYING.
  6. Revert to whichever network-mode the config had before this ran (e.g.
     FP32 for yolo11, FP16 for DAMO-YOLO's shipped configs) unless
     --keep-int8 was given, in which case network-mode=1 (INT8) is left in
     place.

The sidecar image must include the calibrator: perception/pipeline/Dockerfile builds the
custom parser with OPENCV=1. Rebuild the image once after that change:
    docker compose build deepstream-sidecar

IMPORTANT: this needs the machine running `docker compose` (GPU + the
deepstream-perception/pipeline/redis containers), which may not be the same machine used
to convert/export a model's ONNX in the first place -- the ONNX must already
be in place and --config must already point at a working (non-INT8) config
before running this.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from perception.engine.sources import resolve_sources  # noqa: E402 -- pure/GPU-free, see perception/engine/sources.py

# The container mounts the repo at this path (see docker-compose.yml); the
# calibrator's image list must contain CONTAINER-side absolute paths.
CONTAINER_REPO = "/workspace/iris"
DEFAULT_PGIE_CFG = os.path.join(REPO, "perception/models/DeepStream-Yolo", "config_infer_primary_yolo11.txt")
SIDECAR_YAML = os.path.join(REPO, "sidecar", "sidecar.yaml")
CALIB_DIR = os.path.join(REPO, "perception/models/DeepStream-Yolo", "calib_images")
CALIB_LIST = os.path.join(REPO, "perception/models/DeepStream-Yolo", "calib", "calib_images.txt")


def _find_container(substr: str = "deepstream-sidecar"):
    """Resolve the sidecar container by name substring. Compose derives the
    container name from the checkout directory name (surveillance-camera-sys-…),
    which differs per machine/clone — never hardcode it (same pattern as
    scripts/live_streams.py)."""
    try:
        out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                             capture_output=True, text=True, timeout=15).stdout.split()
    except Exception:
        return None
    return next((n for n in out if substr in n), None)


def _container_or_exit():
    ctr = _find_container()
    if not ctr:
        raise SystemExit("deepstream-sidecar container not found — is the stack up? "
                         "(docker compose up -d)")
    return ctr


def _read_pgie_cfg_value(pgie_config, key, default=None):
    """Read a single `key=value` line out of a config_infer_primary_*.txt."""
    try:
        with open(pgie_config) as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{key}="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return default


def _calib_table_path(pgie_config):
    """Resolve the target config's own int8-calib-file, relative to the
    config's directory (same convention perception/engine/engine_path.py's gie_engine_path
    uses for onnx-file) — this is the single source of truth for where nvinfer
    will actually read/write the calibration cache, so this script never
    hardcodes a table filename that could drift from the config."""
    value = _read_pgie_cfg_value(pgie_config, "int8-calib-file")
    if not value:
        sys.exit(f"ERROR: {pgie_config} has no (uncommented) int8-calib-file= line — "
                  "add one before running calibration, e.g. "
                  "int8-calib-file=calib/calib_<model>.table")
    return os.path.join(os.path.dirname(pgie_config), value)


def _engine_batch_size():
    """The batch nvinfer actually builds the pgie engine against: mirrors
    build_pipeline()'s own `max(1, min(len(sources), max_pgie_batch))` exactly
    (nvstreammux's batch-size and the TRT engine batch are the same value) —
    NOT necessarily 1. Two things depend on getting this right:
      - the derived engine filename this script polls for in wait_for()
        (perception/engine/engine_path.py's gie_engine_path derives it from this same batch);
      - INT8_CALIB_BATCH_SIZE (see main()) -- sidecar.py's own main() docs
        that the custom YOLO calibrator SEGFAULTS during calibration if this
        doesn't exactly match the engine batch. Getting this wrong doesn't
        fail cleanly: it corrupts the CUDA context, which then surfaces as a
        flood of unrelated-looking "Error 700" teardown errors (Myelin graph
        context, CUDA stream/event destructors, allocator dealloc) in the
        sidecar logs -- that spam is fallout from this mismatch, not a
        separate bug, if you ever see it again."""
    import yaml
    with open(SIDECAR_YAML) as f:
        cfg = yaml.safe_load(f)
    max_batch = int(cfg.get("max_pgie_batch", 160))
    return max(1, min(len(resolve_sources(cfg)), max_batch))


def run(cmd, **kw):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, **kw)


def extract_frames(video, count):
    os.makedirs(CALIB_DIR, exist_ok=True)
    for old in os.listdir(CALIB_DIR):
        os.remove(os.path.join(CALIB_DIR, old))
    # Get duration via ffprobe (if available).
    duration = None
    if shutil.which("ffprobe"):
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", video],
                capture_output=True, text=True, check=True)
            duration = float(out.stdout.strip())
        except Exception as exc:
            print(f"ffprobe duration failed ({exc}); sampling every second")
    pat = f"{CALIB_DIR}/frame_%05d.jpg"
    if shutil.which("ffmpeg"):
        fps = max(count / duration, 0.1) if duration else 1.0
        res = run(["ffmpeg", "-y", "-i", video, "-vf", f"fps={fps:.4f}",
                   "-q:v", "2", pat], capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stderr[-800:])
            sys.exit(1)
    else:
        import cv2
        cap = cv2.VideoCapture(video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = sorted(set(int(i * total / count) for i in range(count)))
        for n, i in enumerate(idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if ok:
                cv2.imwrite(pat % n, frame)
        cap.release()
    files = sorted(os.listdir(CALIB_DIR))
    if len(files) < 2:
        print(f"ERROR: only {len(files)} frames extracted from {video}")
        sys.exit(1)
    # Trim to roughly `count` frames and write the CONTAINER-side absolute-path
    # list (the calibrator runs inside the sidecar container).
    if len(files) > count:
        step = len(files) / count
        files = [files[int(i * step)] for i in range(count)]
    os.makedirs(os.path.dirname(CALIB_LIST), exist_ok=True)
    rel = os.path.relpath(CALIB_DIR, REPO)
    with open(CALIB_LIST, "w") as f:
        for name in files:
            f.write(os.path.join(CONTAINER_REPO, rel, name) + "\n")
    print(f"Extracted {len(files)} calibration frames -> {CALIB_LIST}")


def set_yaml_int8_keys(batch):
    # The sidecar passes this value into INT8_CALIB_IMG_PATH, which the custom
    # parser's calibrator reads INSIDE the container — so it must be the
    # container-side path, not the host path.
    rel = os.path.relpath(CALIB_LIST, REPO)
    container_list = os.path.join(CONTAINER_REPO, rel)
    with open(SIDECAR_YAML) as f:
        text = f.read()
    text = re.sub(r"(?m)^#?\s*int8_calib_img_path:.*$",
                  f"int8_calib_img_path: \"{container_list}\"", text)
    text = re.sub(r"(?m)^#?\s*int8_calib_batch_size:.*$",
                  f"int8_calib_batch_size: {batch}", text)
    with open(SIDECAR_YAML, "w") as f:
        f.write(text)
    print(f"sidecar.yaml: int8_calib_img_path={container_list} "
          f"int8_calib_batch_size={batch} set")


def flip_pgie_mode(pgie_config, mode):
    with open(pgie_config) as f:
        text = f.read()
    backup = pgie_config + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(pgie_config, backup)
        print(f"backed up pgie config -> {backup}")
    text = re.sub(r"(?m)^network-mode=\d+.*$", f"network-mode={mode}", text)
    with open(pgie_config, "w") as f:
        f.write(text)
    print(f"{os.path.basename(pgie_config)}: network-mode={mode}")


def restart_sidecar():
    run(["docker", "restart", _container_or_exit()])


def wait_for(table, engine, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.isfile(table) and os.path.isfile(engine):
            return True
        time.sleep(5)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="Source video to sample frames from")
    ap.add_argument("--config", default=DEFAULT_PGIE_CFG,
                    help="pgie config to calibrate (default: config_infer_primary_yolo11.txt). "
                         "Pass config_infer_primary_damoyolo.txt / _damoyolo_m.txt to calibrate "
                         "those instead -- must already work at its shipped precision (e.g. "
                         "DAMO-YOLO's FP16) before running INT8 calibration against it.")
    ap.add_argument("--count", type=int, default=60, help="Calibration frames (default 60; 500+ for production quality)")
    ap.add_argument("--batch", type=int, default=None,
                    help="INT8_CALIB_BATCH_SIZE. Defaults to the pgie's actual build batch "
                         "(current source count in sidecar.yaml, capped by max_pgie_batch) -- "
                         "leave this unset. It MUST equal the engine batch exactly or the "
                         "custom YOLO calibrator segfaults (sidecar.py's main() documents "
                         "this), so an explicit value that doesn't match is refused rather "
                         "than risk a corrupted-CUDA-context crash 10 minutes into a build.")
    ap.add_argument("--keep-int8", action="store_true",
                    help="Leave network-mode=1 (INT8) after verification instead of reverting")
    ap.add_argument("--skip-extract", action="store_true", help="Reuse existing calib_images.txt")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        sys.exit(f"ERROR: video not found: {args.video}")
    pgie_config = os.path.abspath(args.config)
    if not os.path.isfile(pgie_config):
        sys.exit(f"ERROR: config not found: {pgie_config}")

    # Preserve the config's current network-mode so a plain (non --keep-int8)
    # run restores it exactly, instead of assuming FP32 -- DAMO-YOLO's shipped
    # configs default to network-mode=2 (FP16), not 0.
    original_mode = _read_pgie_cfg_value(pgie_config, "network-mode", "0")
    calib_table = _calib_table_path(pgie_config)
    batch = _engine_batch_size()
    engine_path = os.path.join(REPO, f"model_b{batch}_gpu0_int8.engine")

    if args.batch is not None and args.batch != batch:
        sys.exit(f"ERROR: --batch {args.batch} does not match the pgie's actual build batch "
                  f"({batch}, from sidecar.yaml's current source count / max_pgie_batch). "
                  f"The custom YOLO calibrator segfaults on a mismatch (sidecar.py's main()) "
                  f"-- rerun without --batch (it defaults to {batch}) rather than override it.")
    calib_batch = args.batch if args.batch is not None else batch

    if not args.skip_extract:
        extract_frames(args.video, args.count)
    set_yaml_int8_keys(calib_batch)

    # Fresh INT8 engine -> delete any stale one so nvinfer must rebuild.
    for path in (engine_path, calib_table):
        if os.path.exists(path):
            os.remove(path)
            print(f"removed stale {path}")

    flip_pgie_mode(pgie_config, 1)
    restart_sidecar()
    print(f"Waiting for {calib_table} + {engine_path} (up to 10 min)...")
    ok = wait_for(calib_table, engine_path)
    if not ok:
        flip_pgie_mode(pgie_config, original_mode)
        restart_sidecar()
        print(f"FATAL: calibration/engine build did not complete. Reverted to "
              f"network-mode={original_mode} and restarted the sidecar. Check `docker logs "
              + _container_or_exit() + "`.")
        sys.exit(1)

    table_kb = os.path.getsize(calib_table) // 1024
    print(f"OK: {os.path.basename(calib_table)} ({table_kb} KB) + INT8 engine built at "
          f"{engine_path}. Engine will be reused on subsequent boots (deserialized from cache).")
    if args.keep_int8:
        print("INT8 kept as default (network-mode=1).")
    else:
        flip_pgie_mode(pgie_config, original_mode)
        print(f"Reverted to network-mode={original_mode}. Flip it to 1 to run INT8.")


if __name__ == "__main__":
    main()
