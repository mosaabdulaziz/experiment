# Dependency baseline (2026-11-03)

> **A snapshot from before pinning, not kept current.** After the pinning merge request, the locks (`uv.lock` files),
> the Dockerfiles' digests and the compose files' digests are the source of truth (DEPENDENCIES.md); updates change
> those, never these files. They stay as the record of what ran before, and of how the pins were chosen.

What the builds actually installed on 2026-11-03, captured before any of it was pinned. The pinning work (lock files,
base image digests, compose digests) had to reproduce these versions, not upgrade them: each later step rebuilds and
diffs its result against the files here. They are plain `name==version` lists, so they also work as constraint files
(`uv pip compile -c deps/baseline/sidecar-slim.txt ...`).

| file | environment | taken from |
|---|---|---|
| `sidecar-slim.txt` | sidecar `slim` target (`perception/pipeline/Dockerfile`), the venv `/opt/iris-venv` | `km-deepstream-sidecar:latest`, image `sha256:0ae29b17360b`, built 2026-10-04 |
| `sidecar-mtmc.txt` | sidecar with the mtmc packages | `scale-deepstream-sidecar:latest`, image `sha256:9cddcee5f3af`, built 2026-09-30 |
| `app.txt` | app image (root `Dockerfile`), system Python | `ppelive-iris:latest`, image `sha256:c189720e7aca`, built 2026-10-29 |
| `ci-test.txt` | CI `core-tests` (`requirements-test.txt`) | resolved 2026-11-03, see below |
| `dev.txt` | `requirements-dev.txt` | resolved 2026-11-03, see below |

How: `docker run --rm --entrypoint bash <image> -c 'uv pip freeze'` (`--system` for the app image), sorted. Each image
is the newest local build of its kind on the GPU server.

## Sidecar: what is not a Python package

| input | today | where it floats |
|---|---|---|
| base image | `nvcr.io/nvidia/deepstream:9.0-triton-multiarch`, NGC build id `259401985`, ref `c2536fda2e5c6fa09b84b08768f450a39a608761`, DeepStream 9.0.0, Ubuntu 24.04, Python 3.12.3 | `FROM` names a tag |
| | index digest `sha256:2e45070ad134b9ab2caa4a97ba4d52fa8744a4f0db30900bd92828d51425a69a`; amd64 `sha256:60888367d4c97ba192411a7694c984080a553f855ad53fc4c5579d70424fafd7`; arm64 `sha256:1579929e83e5a9bfea5d713022a10c5d4f48100d882d67511ace9ce41f302580` | |
| pyds | `deepstream_python_apps` commit `8ad0349ed7a496fae35ebb21c350641727070b89` (2026-04-01, "Update pyds version to 1.2.3"); its submodules at the commits that commit records (pybind11 `0c69e1eb`, gstreamer `6aa055e9`, git-partial-submodule `bd8bb4e7`) | `git clone` takes the default branch's HEAD |
| uv | 0.12.5 (both sidecar images); the app image has 0.12.10 | `pip3 install uv` / `pip install uv` take the newest |
| apt packages | build tools and `libjbig0` from the Ubuntu archive | not captured: they only build pyds and the parsers |

The NGC tag was checked on 2026-11-03: it still resolves to build `259401985`, and the 66 layers of its amd64 image are
identical (same diff ids) to the first 66 layers of both sidecar images above.

## Compose images (pulled copies on the GPU server)

| compose file | image | digest of the local copy |
|---|---|---|
| `docker-compose.yml` | `alexxit/go2rtc:latest` | `sha256:675c318b23c06fd862a61d262240c9a63436b4050d177ffc68a32710d9e05bae` |
| | `minio/minio:latest` | `sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e` |
| | `prom/prometheus:latest` | `sha256:508729e0e2d18e11fd742a5a5ca70e557b940a93948c3c95fd0123a6fd538b69` |
| | `grafana/grafana:latest` | `sha256:ab5cb380e3ff3172d6c8bd2e7cfd31cce977d2881b260e1f5bc089bf0b759b43` |
| | `nvidia/dcgm-exporter:latest` | `sha256:613ab03c11d442fd960ff515f547e9921537454a712d08160bc8f677f89f1c35` |
| | `prom/node-exporter:latest` | `sha256:1b4e4438faca4dd7e001dd445d161a4a2091b0fededa84093b3a8dfeae1f1be0` |
| | `redis:7-alpine` | `sha256:e7723ff73d963f5cc6d9c4643ea3d989527a402a319239054e9472a7fb9219a2` |

These are this machine's pulls; another server that pulled `latest` at another time may run something else.

## Things the pins must keep (or that need a decision)

- **Both onnxruntime and both opencv wheels are installed**, at the same version: `onnxruntime` and `onnxruntime-gpu`
  1.28.0, `opencv-python` and `opencv-python-headless` 5.0.0.93 (insightface pulls in the CPU/GUI ones). Each pair
  writes the same files; the GPU runtime wins today only because the Dockerfile installs it after insightface. The
  lock keeps this (same versions, GPU wheel installed last); removing the CPU wheels is a separate issue.
- **The mtmc image was not built by the `mtmc` target** but by `Dockerfile.addons` on an older sidecar image. Its
  versions are still the ones running, so the lock follows them.
- **`km-deepstream-sidecar` predates the per-module requirements files** (2026-10-25), but it holds exactly the
  packages they now list; none was added or removed by that change.
- **CI tests newer versions than the app ships.** `ci-test.txt` is what `pip install -r requirements-test.txt` gets
  today; 23 of the 42 packages it shares with `app.txt` differ (e.g. fastapi 0.162.0 vs 0.141.1, pillow 13.0.0 vs
  12.3.0, openai 3.42.1 vs 3.13.0). The root lock has to choose one version per package for both.
- `ci-test.txt` and `dev.txt` were made with `uv pip compile --python-version 3.11 --python-platform
  x86_64-manylinux_2_28` (uv 0.13.6); CI installs with pip, which takes the same newest versions. Neither is a fixed
  environment today: they change whenever a package releases.
