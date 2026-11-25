# IRIS DeepStream perception sidecar image.
#
# Built on the DeepStream 9.0 container. Adds:
#   - pyds, BUILT FROM SOURCE against the installed DS 9.0 SDK (no official
#     DS 9.0 pyds wheel exists yet — the latest release targets DS 8.0, so we
#     compile the bindings against /opt/.../deepstream headers for an ABI match)
#   - redis (publish metadata + JPEG), opencv-python-headless (JPEG encode),
#     pyyaml (load sidecar.yaml), minio (upload full-fps preview segments)
# Pinned by digest (DEPENDENCIES.md): NGC build 259401985 (ref c2536fda), the base every running sidecar image was
# built from. The digest is the multi-arch index's; on amd64 it resolves to
# sha256:60888367d4c97ba192411a7694c984080a553f855ad53fc4c5579d70424fafd7.
FROM nvcr.io/nvidia/deepstream:9.0-triton-multiarch@sha256:2e45070ad134b9ab2caa4a97ba4d52fa8744a4f0db30900bd92828d51425a69a AS base

# ── Build deps for the pyds bindings ─────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential \
        python3-dev python3-gi python3-gst-1.0 python-gi-dev \
        libtool m4 autoconf automake \
        libgirepository1.0-dev libcairo2-dev \
        libopencv-dev libjbig0 \
    && rm -rf /var/lib/apt/lists/*

# The base image's libjbig0 install is broken (package registered, but the actual
# libjbig.so.0 file is missing), which makes dlopen fail on the OpenCV-linked
# YOLO parser with "libjbig.so.0: cannot open shared object file" and kills the
# pgie on every fresh container. `--reinstall` is a no-op for a package dpkg
# considers installed, so fetch the .deb from the archive and extract it into
# place — deterministic regardless of the broken package state. apt-get update is
# required here: the previous RUN deleted the lists, without which `download`
# cannot resolve the package.
RUN apt-get update && apt-get download libjbig0 \
    && dpkg -x libjbig0*.deb / && rm -f libjbig0*.deb \
    && rm -rf /var/lib/apt/lists/* && ldconfig

# ── Build + install pyds from source against the installed DS 9.0 SDK ─────────
# At a fixed commit ("Update pyds version to 1.2.3", 2026-04-01), not the default branch's HEAD; `submodule update`
# then checks out the submodule commits that commit records.
ARG PYDS_COMMIT=8ad0349ed7a496fae35ebb21c350641727070b89
RUN cd /opt/nvidia/deepstream/deepstream/sources \
    && git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git \
    && cd deepstream_python_apps \
    && git checkout --quiet "$PYDS_COMMIT" \
    && git submodule update --init \
    && cd bindings && mkdir build && cd build \
    && cmake .. -DDS_PATH=/opt/nvidia/deepstream/deepstream \
    && make -j"$(nproc)" \
    && cp pyds.so "$(python3 -c 'import site; print(site.getsitepackages()[0])')/"

# ── Build every custom nvinfer parser .so ─────────────────────────────────────
# perception/models/build_parsers.sh compiles each perception/models/*/nvdsinfer_custom_impl_*/ against the installed
# DS and installs it under /opt/nvidia/deepstream/deepstream/custom_parsers -- a path outside the host-volume mount
# at /workspace/iris, so it survives the mount. A model with a parser of its own just adds its directory (and a line in
# perception/models/parser_flags for `make` variables, e.g. OPENCV=1 for the YOLO parser's INT8 calibrator); nothing here changes.
# CUDA_VER matches the container's CUDA toolkit (13.1 for DS 9.0). COPY --parents keeps the directory names while
# leaving the models' weights and engines out of the layer.
ENV CUDA_VER=13.1
COPY --parents perception/models/*/nvdsinfer_custom_impl_*/ /src/
COPY perception/models/build_parsers.sh perception/models/parser_flags /src/perception/models/
RUN /src/perception/models/build_parsers.sh /src/perception/models && rm -rf /src
# `just barber-up` rebuilds the image when this label is missing (an image from before the segmentation parser).
LABEL iris.seg-parser="1"

# ── Build every custom nvtracker .so ──────────────────────────────────────────
COPY perception/trackers/ /src/perception/trackers/
COPY perception/pipeline/build_trackers.sh /src/perception/pipeline/
RUN /src/perception/pipeline/build_trackers.sh /src/perception/trackers && rm -rf /src

# ── Install uv, pinned by version and digest (DEPENDENCIES.md, "uv") ─────────
COPY --from=ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 /uv /uvx /usr/local/bin/

# ── Create virtual environment ───────────────────────────────────────────────
# DeepStream 9.0 ships Ubuntu 24.04 with the PEP 668 EXTERNALLY-MANAGED marker.
# Use `uv venv` (does not need ensurepip) with --system-site-packages so that
# pyds.so (installed to system site-packages) remains importable.
ENV VIRTUAL_ENV=/opt/iris-venv
RUN uv venv --system-site-packages "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# ── Python deps: from the lock (perception/uv.lock, DEPENDENCIES.md) ────────
# perception/engine/requirements.txt is what the engine itself imports; perception/modules/<name>/requirements.txt is
# what that module adds (tests/test_module_requirements.py checks nothing it imports is missing). They stay the place
# to declare a dependency: tools/deps/perception_groups.py turns them into the lock's groups, `slim` = the engine +
# BASE_MODULES (the modules every image can run: one image serves attendance, barber and fire) and `mtmc` = slim +
# MTMC_MODULES + CUDA torch. The image installs exactly the lock's versions, hash-checked; install_locked.sh says
# why it installs in three passes (onnxruntime-gpu must land after insightface's CPU onnxruntime). To change a module
# list, change its default here and run `just deps-lock`: a --build-arg that differs from what the lock's group was
# generated from stops the build (install_locked.sh), since the locked group could not honour it.
ARG BASE_MODULES="face_recognition customer_faces appearance proximity"
COPY perception/pyproject.toml perception/uv.lock perception/pipeline/install_locked.sh /req/
RUN /req/install_locked.sh /req slim BASE_MODULES "$BASE_MODULES"

# The InsightFace buffalo_l pack is NOT in the image: face_recognition, customer_faces and the enrollment CLIs load the
# repo's DVC copy, perception/models/insightface/models/buffalo_l, from the mounted repo (perception/insightface_pack.py).
# A missing pack refuses startup with the `dvc pull` to run; nothing is downloaded, at build or at run time.

# Repo root on the shared volume (docker-compose mounts ./:/workspace/iris),
# independent of the checkout directory's name on the host.
WORKDIR /workspace/iris
COPY perception/__init__.py ./perception/
COPY perception/lib/ ./perception/lib/
COPY perception/engine/ ./perception/engine/
COPY perception/modules/ ./perception/modules/
COPY perception/pipeline/ ./perception/pipeline/
CMD ["python3", "-m", "perception.engine.main", "perception/pipeline/sidecar.yaml"]

# ── Build targets ─────────────────────────────────────────────────────────────
# `slim` (the default: it is the LAST stage, so an un-targeted build gives it) is what
# attendance, barber and fire run. `mtmc` adds what multi-camera vehicle tracking needs
# on top: CUDA torch for the Re-ID embedder, the fast-ReID helpers and the VLM clients.
# Keeping them out of `slim` keeps every other product's image several GB smaller and
# stops a change to these layers from rebuilding theirs. docker-compose selects it with
# `build.target: mtmc` on the multicam service only.
FROM base AS mtmc
# The CUDA wheels torch pulls in are hundreds of MB each, from NVIDIA's own host; uv's default 30 s
# HTTP timeout aborts the build on an ordinary link ("operation timed out" on nvidia-cusparselt).
ENV UV_HTTP_TIMEOUT=600
# Installing torch (below) also installs nvidia-cudnn-cu13 9.24 into the venv, while the base image's own
# cuDNN is 9.17. ONNX Runtime picks the venv's libcudnn.so.9 up, and cuDNN then loads its sublibraries BY NAME
# (libcudnn_engines_runtime_compiled.so.9.24.0, ...) through the system loader -- which cannot see the venv's
# directory. Result: every conv fails ("Unable to load any of {libcudnn_engines_runtime_compiled.so.9.24.0...}")
# and ONNX Runtime quietly runs on the CPU, 16x slower (measured: 225 ms vs 14 ms per batch of 8 faces).
# Putting that directory on the loader path makes cuDNN find its own sublibraries. The `slim` target has no
# torch, so it never has this problem.
ENV LD_LIBRARY_PATH=/opt/iris-venv/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
# torch/torchvision on CUDA so MTMC ResNet50 Re-ID runs on the GPU (the
# container carries the CUDA 13.1 runtime from DeepStream 9.0). CPU torch made
# synchronous Re-ID stalls ~2.2s/embed on the shared streaming thread; CUDA
# torch drops a 16-crop embed to tens of ms and the async worker keeps it off
# the streaming thread regardless. The lock's `mtmc` group: the mtmc module's
# requirements.txt (fast-ReID's helpers, the VLM arbiter's clients, ...) plus
# CUDA torch 2.14.1+cu130 from PyTorch's cu130 index.
ARG MTMC_MODULES="mtmc"
RUN /req/install_locked.sh /req mtmc MTMC_MODULES "$MTMC_MODULES"

FROM base AS slim
