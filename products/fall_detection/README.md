# products/fall_detection

Fall detection for the IRIS cameras: people are tracked by the sidecar, their 17 keypoints are
published with each frame, and this service decides — per track, on the sidecar's own frame
clock — whether what it sees is a fall, a person lying down, a person immobile, or nothing.

It is a **separate deployable**: a FastAPI host process (port **8004**, `just fall-up`) with its
own SQLite database (`data/fall_detection.db`) and its own compose sidecar. It never imports
another product's code; Redis is the only connection.

```
camera ─▶ go2rtc ─▶ deepstream-sidecar-fall ──iris:meta:<cam>──▶ fall-service (:8004)
                   (products/fall_detection/          `pose` field: shape + people + kps
                    sidecar.yaml, role
                    `fall_detection`)                  ├─ iris:alert:<cam>      (the app's alert)
                                                       ├─ iris:svr_request:<cam> (the clip)
                                                       └─ data/fall_detection.db (review queue)
```

The sidecar runs the pose GIE (`perception/modules/pose`) and publishes the `pose` field; the
service reads nothing else from the frame. The decision is the product's: the vendor's monitor
(a copy, see `docs/PROVENANCE.md`) produces candidates, `scoring.py` scores them on three paths, and
`consumer.py` routes them to the platform's alert, a clip request and the review queue.

## Deploying

The **command list for the deployment server** (the 5090) — the bring-up, the liveness checks, the
test suites, the measurements the pilot owes and the recovery table — is
[`docs/PRODUCTION_RUNBOOK.md`](docs/PRODUCTION_RUNBOOK.md).

Prerequisites: the checkpoints. They are versioned with DVC (`just data-pull models-fall`; DATA.md), no longer plain
git blobs. The two the shipped config loads are `mmaction2_joint.pth` and
`mmaction2_jm.pth`; the boot gate verifies their SHA-256 against the vendor's registry and
refuses to start on a mismatch or a missing file. (The vendor's other six `.pth` files — the
`pyskl_*` and the rest of `mmaction2_*` — are only needed if you change `classifier.streams` to
name them.) There is no CPU fallback for the shipped `classifier.device: cuda:0`. The pose ONNX
the sidecar runs (`perception/models/DeepStream-Yolo-Pose/yolo26x-pose.onnx`) comes with the same group; its
TensorRT engine is built on the sidecar's first start.

1. `just bootstrap` once (Python tooling, the frontend, `.env`).
2. Start the frontend with `just up` (the dashboard serves every product, this tab included).
3. `just fall-up` — in this order:
   * `uv sync` in `products/fall_detection/`;
   * the **camera-exclusivity preflight**: every running `deepstream-sidecar-*` container is
     inspected, its sidecar yaml and `IRIS_SIDECAR_ROLE` read, and the start is refused, naming
     the container and the camera, if one of this product's cameras is already held. The
     engine's `roles` tag names a recording's owner but does not stop a second process from
     reading the camera (`perception/engine/sources.py:52-74`), so this check is the guard;
   * the shared containers (redis/minio/go2rtc), then the service — its boot gate is a real
     CUDA op plus both checkpoints' SHA-256, so a GPU-less box or a bad checkpoint refuses in
     seconds, before the sidecar's first start spends minutes building TensorRT engines;
   * the sidecar (`deepstream-sidecar-fall`, compose profile `fall`), then a bounded wait for
     `pipeline PLAYING`.
4. `just fall-log` follows the service log; `just fall-down` stops the service and the sidecar.

Environment (`products/fall_detection/fall_detection.yaml` header): `FALL_DETECTION_CONFIG`
(another yaml), `FALL_DETECTION_DATA_DIR`, `FALL_DETECTION_WEIGHTS_DIR`, `FALL_BACKEND_PORT`,
`REDIS_URL`. Compose variables: `SIDECAR_FALL_TELEMETRY_PORT` (default 9105),
`FALL_SIDECAR_CONFIG` (default `products/fall_detection/sidecar.yaml`).

## The camera rule

The fall deployment **replaces** the other deployment on its cameras; it does not run beside
it. `just fall-up` enforces this (above). The pilot watches `cam-1` (`fall_detection.yaml`); a
config that takes more cameras takes the whole block in `sidecar.yaml`'s `sources`, tagged
`roles: [fall_detection]`.

## In the safety stack

`products/safety` runs this service beside other role sidecars on the same camera. Its settings, all off by
default (products/safety/fall_detection.yaml turns them on):

- `key_role: fall` -- the camera's streams carry `<camera>@fall` (`iris:meta:`, `iris:alert:`, `iris:svr_request:`),
  as the fall role sidecar writes them with role-scoped keys (`acl.key_id`). The preflight compares these key ids, so
  a role-scoped sidecar of another role on the same camera is no conflict.
- `zones: {path, mux: [w, h]}` -- the zones file and the mux size its points are in. A camera with zones whose `fall`
  block is on raises a candidate only for a person inside one (box bottom centre, scaled from capture pixels by the
  frame's `shape`); the alert's `zone` names it. Outside them: no alert, no review row, no history row. A camera with
  no such zone covers the whole frame.
- `alert_details: true` -- the alert's message is a sentence ("Person fell (confidence 82%)", "A person may be lying
  on the floor: needs review (confidence 45%)") and it carries `person_ids` (the track, so the clip draws that
  person) and `evaluation`, `confidence` and `review` as fields. Off: `<evaluation> <confidence>` as before.
- `scoring.alert_above: <confidence>` -- only a candidate whose confidence (two decimals) is above it alerts; below
  it, it still goes to the review queue and the history.

## The API

| Endpoint | What |
|---|---|
| `GET /status` | what the boot gate proved: `device`, `weights_verified`, `classifier_ready`, `paths_live` |
| `GET /events?camera=&since=&limit=` | the scored candidates, newest first; `since` is the tab's cursor (the last id it has seen) |
| `GET /cameras` | the ids the service watches, from this product's config |
| `GET /candidates?camera=&status=` | the review queue, newest first, with each candidate's evidence |
| `POST /review/<id>` | `{"status": "acknowledged" \| "confirmed" \| "dismissed", "note": "..."}` |

The dashboard's dev server proxies `/fall_detection/api` to `FALL_BACKEND_PORT`
(`frontend/vite.config.ts`). The fall tab (`frontend/src/products/fall_detection/`) shows the
camera's published frame — the same JPEG the dashboard wall draws, people boxes included — and
the review queue.

## What is proven, and what is not

Proven on any machine without a GPU or Redis: the policy and routing, the service's start path
and shutdown, the review queue, the vendor copy's seam (reverse-applied against the pristine
manifest), the pose parser (compiled against a stub of DeepStream's header), the export's
tensor contract, the preflight, and the recipe wiring.

Needs the deployment box (see `docs/phases/phase-3-loop-log.md`, Round 5): the boot gate on
the real RTX 5090 with the real checkpoints, the sidecar reaching `PLAYING` with the pose model,
an end-to-end fall through a live camera, the corpus replays (accuracy and latency), and the
frontend's vitest files (Node is not installed on the development box).
