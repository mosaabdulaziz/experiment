# Multi-camera vehicle tracking

One vehicle followed across many cameras: detection (1-class vehicle detector), a per-camera
tracker, cross-camera Re-ID (MTMC), a VLM that settles ambiguous matches, plate reading, and a
feed of "what each camera sees now" to the truck-tracking platform.

This product is **declared, not special-cased**: the shared pipeline file
(`perception/pipeline/sidecar.yaml`) and the shared app config (`config.yaml`) start none of it.
Everything here is switched on by this folder's overlay and by two keys in the app config.

## Where things are

| What | Where |
|---|---|
| Pipeline policy (cameras, vehicle model, tracker, MTMC, VLM arbiter, plates, stabilizer) | `products/multicam/sidecar.yaml` (extends the shared file) |
| Pipeline code | `perception/pipeline/` (`mtmc/`, `plate_*`, `vlm_arbiter.py`, `box_stabilizer.py`, `flight.py`, ...), each a no-op unless its config key is on |
| Models | `products/el-mohandes/models/{vehicle_1280_1cls,plate_640}` (the client's finetuned detectors) |
| Compose service | `deepstream-sidecar-multicam` (profile `multicam`, image target `mtmc`) |
| App side (for now) | `app/tracking.py`, `truck_feed.py`, `vehicle_gallery.py`, `plate_reads.py`, `line_counter.py`, `hazards*.py`, enabled by `mtmc.enabled` / `hazards.enabled` in `config.yaml` |
| Research / training tools | top-level `mtmc/` (detector fine-tuning, audits, Re-ID experiments) |
| Isolation tests | `tests/test_multicam_isolation.py` |

## Run it

```bash
just multi            # 13 recorded clips through the pipeline + app + the truck-tracking UI
just vlm              # the single-camera VLM monitoring preset
docker compose --profile multicam up -d redis minio go2rtc deepstream-sidecar-multicam
```

`just multi` swaps the app's `config.yaml` for `configs/presets/multicam/config.yaml` and runs the
pipeline on `configs/presets/multicam/sidecar.yaml` (this product's overlay with `file://` clips).
It never overwrites `perception/pipeline/sidecar.yaml`; `just up` restores the tracked `config.yaml`.
The scale run (`scripts/scale13_deploy.sh`) uses the plain `deepstream-sidecar` service name on
`products/multicam/sidecar.yaml`.

## The truck-tracking UI

The UI for this product is the separate **truck-tracking** platform
(`~/URCA/IRIS/repos/truck-tracking`, branch `feat/vlm-mtmc-integration`). It talks to this side through
a contract that must not change without it:

* Redis list `iris:camera-feed` (batch schema 1.0, written by `app/truck_feed.py`);
* HTTP on the camera system: `/stream/<cam>/frame`, `/api/vehicles/<gid>/sightings`,
  `/api/vehicles/crops/<cam>/<gid>/image`, `/api/vehicles/plates/<gid>/readings`,
  `/api/vehicles/plates/<gid>/<cam>/image`, `/api/tracking/config`,
  `/api/streams/<cam>/{restricted,filter}-zones`, `/api/hazards/{analyze,status,history}`.

`tests/test_multicam_isolation.py::test_an_app_with_the_use_case_serves_the_routes_truck_tracking_calls`
lists these and fails if one disappears.

## Archived: the old demo UI

Before the monorepo, this branch used the `iris-frontend` submodule with Tracking, Map and Hazards
pages (99 commits beyond the pointer the monorepo base used). The in-tree `frontend/` replaced the
submodule, and truck-tracking replaces those pages, so they were **not ported**. They are recoverable:
repository `ssh://git@gitlab.urca.ai:2222/pyai/iris/iris-frontend.git`, branch `feat/multcam-vlm-demo`,
last pointer `744fc87582031d7d3668a3d048b37f316b8489db`.

## Not done yet (next)

* Move the app-side code above into its own service in this folder, serving the same contract paths.
* ~~Make MTMC / plates / stabilizer plug-in modules rather than branches inside `sidecar.py`.~~ Done: they are modules (`perception/modules/mtmc`, `plates`, `box_stabilizer`); see `perception/ARCHITECTURE.md`.
* The research `mtmc/` folder and `perception/modules/mtmc/core/` share a name; tests that import both
  are order-dependent (`tests/test_visit_episodes.py` errors in a full run, passes alone).
