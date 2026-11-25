#!/usr/bin/env bash
# Build the sidecar image from its lock and prove it is the image that runs today (DEPENDENCIES.md, "Rebuilding the
# images"):
#
#   tools/deps/verify_sidecar.sh [slim|mtmc|all] [--keep]
#
# For each target: build it, then check
#   - `uv pip freeze` of /opt/iris-venv equals deps/baseline/sidecar-<target>.txt (the diff must be empty);
#   - the files that two wheels share ended up from the right one: cv2 is the GUI opencv-python build, ONNX Runtime the
#     GPU build (install_locked.sh installs them last for this);
#   - on the GPU: ONNX Runtime runs a session on CUDAExecutionProvider, pyds imports; for mtmc, torch sees the GPU and
#     runs a CUDA op.
#
# Builds use a builder of their own (iris-verify, BuildKit pinned below), so its cache is removed with it at the end
# and nobody else's build cache is touched; the images are tagged iris-verify-sidecar:<target> and removed too, unless
# --keep. Needs the docker buildx plugin: the legacy builder silently ignores `COPY --parents` and would build a
# different image. Refuses to start when the Docker disk would drop under MIN_FREE_GB (default 15).
set -euo pipefail
cd "$(dirname "$0")/../.."

targets=all keep=0
for a in "$@"; do case $a in slim|mtmc|all) targets=$a ;; --keep) keep=1 ;; *) echo "usage: $0 [slim|mtmc|all] [--keep]" >&2; exit 2 ;; esac; done
[ "$targets" = all ] && targets="slim mtmc"
MIN_FREE_GB=${MIN_FREE_GB:-15}
# What each build adds to the Docker disk at its peak (estimated 2026-11-04). slim: the builder's own copy of the
# DeepStream base (9.5 GB compressed + ~20 GB unpacked; it cannot reuse Docker's) + ~11.4 GB of new layers (the
# running slim image's unique size), and the same ~11.4 GB again once loaded into Docker. mtmc reuses slim's layers in
# the same builder: ~9 GB more there (CUDA torch and its wheels) and again in Docker. The builder's ~50 GB go away at
# the end.
declare -A NEED_GB=([slim]=55 [mtmc]=20)
BUILDKIT=moby/buildkit:v0.34.1@sha256:99368f3e46177c6405b9740e4461d91cee004128daeecca78d4ceac50ec1da5c
BUILDER=iris-verify
fail=0

if ! docker buildx version >/dev/null 2>&1; then
    echo "docker buildx is not installed: the legacy builder ignores COPY --parents and builds a different image" >&2
    exit 1
fi

free_gb() { df -BG --output=avail "$(docker info --format '{{.DockerRootDir}}')" | tail -1 | tr -dc 0-9; }

had_buildkit=$(docker image inspect "$BUILDKIT" >/dev/null 2>&1 && echo 1 || echo 0)
cleanup() {
    if [ $keep = 0 ]; then
        for t in slim mtmc; do docker image rm "iris-verify-sidecar:$t" >/dev/null 2>&1 || true; done
    fi
    docker buildx rm "$BUILDER" >/dev/null 2>&1 || true      # removes this builder's cache with it, nothing else
    [ "$had_buildkit" = 1 ] || docker image rm "$BUILDKIT" >/dev/null 2>&1 || true
}
trap cleanup EXIT
docker buildx inspect "$BUILDER" >/dev/null 2>&1 \
    || docker buildx create --name "$BUILDER" --driver docker-container --driver-opt "image=$BUILDKIT" >/dev/null

check() { if [ "$2" = ok ]; then echo "  ok    $1"; else echo "  FAIL  $1: $2"; fail=1; fi; }

for t in $targets; do
    image="iris-verify-sidecar:$t"
    have=$(free_gb)
    echo "== $t: ${have} GB free on the Docker disk, the build needs about ${NEED_GB[$t]} GB"
    if [ $((have - NEED_GB[$t])) -lt "$MIN_FREE_GB" ]; then
        echo "   stopping: that would leave under ${MIN_FREE_GB} GB free" >&2; exit 1
    fi
    # The estimate is a guess: while the build runs, stop it if the disk drops under the floor anyway.
    docker buildx build --builder "$BUILDER" --target "$t" --load -t "$image" -f perception/pipeline/Dockerfile . &
    build=$!
    ( while sleep 20; do
          if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then
              echo "   STOPPING the build: under ${MIN_FREE_GB} GB free on the Docker disk" >&2
              kill "$build" 2>/dev/null; docker buildx stop "$BUILDER" >/dev/null 2>&1; exit
          fi
      done ) &
    watchdog=$!
    status=0; wait "$build" || status=$?
    kill "$watchdog" 2>/dev/null || true
    [ "$status" = 0 ] || { echo "   the $t build failed or was stopped ($(free_gb) GB free)" >&2; exit 1; }
    echo "   built; $(free_gb) GB free"

    expected=$(grep -v '^#' "deps/baseline/sidecar-$t.txt" | tr 'A-Z_' 'a-z-' | sort)
    actual=$(docker run --rm --entrypoint bash "$image" -c 'uv pip freeze' 2>/dev/null | tr 'A-Z_' 'a-z-' | sort)
    if d=$(diff <(echo "$expected") <(echo "$actual")); then check "uv pip freeze = deps/baseline/sidecar-$t.txt" ok
    else check "uv pip freeze = deps/baseline/sidecar-$t.txt" "$(echo; echo "$d" | sed 's/^/        /')"; fi

    out=$(docker run --rm -i --gpus all --entrypoint python3 "$image" - <<'EOF' 2>&1
import base64, hashlib, importlib.metadata as md, pathlib
import cv2, numpy as np, onnx, onnxruntime as ort
from onnx import TensorProto, helper

so = next(pathlib.Path(cv2.__file__).parent.glob("cv2*.so"))
digest = base64.urlsafe_b64encode(hashlib.sha256(so.read_bytes()).digest()).decode().rstrip("=")
owner = [d for d in ("opencv-python", "opencv-python-headless")
         if any(str(f).endswith(so.name) and f.hash and f.hash.value.rstrip("=") == digest for f in md.distribution(d).files)]
print("CV2_FILES_FROM", owner)
capi = pathlib.Path(ort.__file__).parent / "capi"
print("ORT_CUDA_LIB", (capi / "libonnxruntime_providers_cuda.so").exists())
graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9
sess = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
y = sess.run(None, {"x": np.array([-1.0, 2.0], np.float32)})[0]
print("ORT_SESSION", sess.get_providers()[0], y.tolist())
import pyds  # noqa: F401  (needs the GPU driver's libcuda, hence --gpus)
print("PYDS ok")
EOF
    ) || true
    check "cv2 is the GUI opencv-python build" "$(grep -q "CV2_FILES_FROM \['opencv-python'\]" <<<"$out" && echo ok || grep CV2_FILES_FROM <<<"$out" || echo "$out" | tail -3)"
    check "ONNX Runtime is the GPU build" "$(grep -q 'ORT_CUDA_LIB True' <<<"$out" && echo ok || echo "no libonnxruntime_providers_cuda.so")"
    check "ONNX Runtime runs on CUDAExecutionProvider" "$(grep -q 'ORT_SESSION CUDAExecutionProvider \[0.0, 2.0\]' <<<"$out" && echo ok || (grep ORT_SESSION <<<"$out" || echo "$out" | tail -3))"
    check "pyds imports" "$(grep -q 'PYDS ok' <<<"$out" && echo ok || echo "$out" | tail -2)"

    if [ "$t" = mtmc ]; then
        out=$(docker run --rm --gpus all --entrypoint python3 "$image" -c '
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
a = torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3)
print("TORCH", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), (a @ a.T).cpu().tolist())' 2>&1) || true
        check "torch sees the GPU and runs a CUDA op" "$(grep -q 'TORCH 2.14.1+cu130 .*\[\[5.0, 14.0\], \[14.0, 50.0\]\]' <<<"$out" && echo ok || echo "$out" | tail -2)"
        grep TORCH <<<"$out" | sed 's/^/        /' || true
    fi
done

[ $fail = 0 ] && echo "verified: $targets" || { echo "VERIFICATION FAILED" >&2; exit 1; }
