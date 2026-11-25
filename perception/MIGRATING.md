# Migrating a branch from the old `sidecar.py` to the engine and modules

*For anyone with a branch or a local change written against `perception/pipeline/sidecar.py` (before MR !32). The
architecture itself is explained in [`ARCHITECTURE.md`](ARCHITECTURE.md); this page is only "where did my code go and how
do I port it".*

## What happened, in three lines

- `perception/pipeline/sidecar.py` (4331 lines: the DeepStream engine **and** every use case) is gone.
- The generic part is `perception/engine/` (run as `python3 -m perception.engine.main <yaml>`). Each use case is a
  **module** under `perception/modules/<capability>/`, switched on by a `modules:` list in the product's yaml.
- Behaviour did not change: the golden replay (attendance, zones, multicam identical; fire, barber, barber-faces
  semantically equal) proves it. Your branch's *logic* is still right; it only needs to move to its new home.

## 1. How to run it

| before | now |
|---|---|
| `python3 perception/pipeline/sidecar.py <yaml>` | `python3 -m perception.engine.main <yaml>` (compose and the Dockerfile already do this) |
| `from perception.pipeline.sources import ...` | `from perception.engine.sources import ...` |
| `from perception.pipeline.<helper> import ...` | the new path in section 2 |

## 2. Where each old file went

**Helpers that were their own files** (git tracks these as renames, so `git log --follow <new path>` shows the old history):

| old (`perception/pipeline/`) | new |
|---|---|
| `engine_path.py`, `flight.py`, `jpeg_encoder.py`, `nvenc_probe.py`, `recorder_kind.py`, `segment_index.py`, `source_recovery.py`, `svr_session.py`, `svr_trigger.py`, `telemetry.py`, `sources.py` | `perception/engine/` (same names) |
| `overcrowding_dedup.py` | `perception/lib/overcrowding_dedup.py` |
| `zone_config.py`, `zone_dedup.py`, `box_color.py` | `perception/modules/zones/` (`config.py`, `dedup.py`, `colors.py`) |
| `fire_alerts.py` | `perception/modules/fire_alerts/state.py` |
| `face_owner.py`, `face_regions.py`, `sgie_gallery.py` | `perception/modules/face_detect/` (`owner.py`, `regions.py`, `idle.py`) |
| `face_sample_gate.py` | `perception/engine/gie_gate.py` (now generic: any secondary GIE can use it) |
| `face_recognizer.py`, `face_enhancer.py` | `perception/modules/face_recognition/` (`recognizer.py`, `enhancer.py`) |
| `customer_faces.py` | `perception/modules/customer_faces/collector.py` |
| `appearance.py` | `perception/modules/appearance/worker.py` |
| `plate_hold.py` | `perception/modules/plates/hold.py` |
| `box_stabilizer.py` + `mtmc/detection_postprocess.py` | `perception/modules/box_stabilizer/` (`stabilizer.py`, `postprocess.py`) |
| `mask_outline.py` | `perception/modules/person_masks/outline.py` |
| `mtmc/*`, `mtmc_palette.py`, `plate_reader.py`, `vehicle_gallery.py`, `vlm_arbiter.py` | `perception/modules/mtmc/` (`core/`, `palette.py`, `plate_reader.py`, `vehicle_gallery.py`, `vlm_arbiter.py`) |

Still in `perception/pipeline/`: `sidecar.yaml` (shared engine settings), the `Dockerfile`, tracker configs, and diagnostic
scripts (`register_face.py`, `run_diag.py`, `capture_*`, ...). The diagnostics are not part of the running engine.

**What was inside `sidecar.py`:**

| the old code in `sidecar.py` | now |
|---|---|
| `main()`, startup, shutdown, watchdog | `engine/main.py` |
| `SidecarConfig` (the engine's own keys) | `engine/config.py`; each use case's keys are its module's `Settings` |
| `build_pipeline()` (sources, mux, GIE chain, tracker, sinks) | `engine/gst_build.py`; a module adds its model with `gies()` and its elements with `elements()` |
| `probe()` (the per-frame hot path) | `engine/probe.py`, which calls the modules' hooks; `engine/objects.py` reads the object metadata once |
| `_FramePublisher`, `_draw_and_encode` (preview, JPEG, `iris:meta`) | `engine/publisher.py` |
| recording, smart-record clips | `engine/recording.py`, `svr*.py` |
| zones, overcrowding, filter zones | module `zones` |
| the fire detector and sustained fire/smoke alerts | module `fire_alerts` |
| face detection, ownership, the sampling gate | module `face_detect` (+ `engine/gie_gate.py`) |
| face recognition, capture, enhancement | module `face_recognition` |
| customer face pictures | module `customer_faces` |
| ReID / staff probability / body crops | module `appearance` |
| plate detection and holding | module `plates` |
| MTMC, VLM arbiter, vehicle gallery, plate reading | module `mtmc` |
| box stabilizer, `person_boxes`, `person_masks` | modules `box_stabilizer`, `person_boxes`, `person_masks` |

## 3. Your yaml

A use case is no longer a pile of top-level keys and `*_enabled` flags in the shared file. A product lists the modules it
wants, **in the order they should run**:

```yaml
extends: ../../perception/pipeline/sidecar.yaml
modules:
  zones:         {config_path: ..., overcrowding_enabled: true}
  face_detect:   {owner_mode: tracked, landmarks: true}
  face_recognition: {gallery_key: attendance:face:gallery}
```

| old top-level key(s) | now |
|---|---|
| `face_enabled`, `sgie_face_enabled` | list `face_detect` (its settings: `config`, `confidence_threshold`, `min_pixels`, `owner_mode`/`face_owner`, `sample_only`, `regions_key`, `idle_when_unused`) |
| `face_recognition_enabled`, `face_recognition_*`, `face_gallery_key`, `face_align_*`, `face_capture_*`, `face_enhancement_*` | list `face_recognition` (`threshold`, `margin`, `gallery_key`, `vote_ttl`, `min_votes`, `align`, `capture: {...}`, `enhancement: {...}`, ...) |
| `sgie_auto_*`, `sgie_face_sample_only`, `sgie_face_config` | `face_detect`: `idle_when_unused`, `sample_only`, `config` |
| `customer_faces_enabled`, `customer_crops_dir`, `face_min_sharpness`, `face_max_yaw`, `customer_face_interval_s` | list `customer_faces` (`crops_dir`, `min_sharpness`, `max_yaw`, `interval_s`) |
| `appearance_*`, `body_crop_interval_s` | list `appearance` (`reid_spec`, `staff_model`, `staff_classifier`, `interval_s`, `crops_dir`, `body_interval_s`, ...) |
| `pgie_fire_*`, `fire_alert_*`, `fire_sustain_seconds`, `smoke_sustain_seconds` | list `fire_alerts` (`config`, `interval`, `classes`, `fire_sustain_seconds`, `smoke_sustain_seconds`, `release_seconds`, `min_confidence`, `dropout_grace_seconds`) |
| `analytics_config_path`, `zone_*`, `overcrowding_*` | list `zones` (`config_path`, `analytics_config_path`, `alert_classes`, `grace_seconds`, `match_radius`, `overcrowding_*`) |
| `publish_person_masks` | list `person_masks` (and `person_boxes` for the boxes) |
| the plate and MTMC keys | lists `plates` and `mtmc` |
| engine keys: `mux_*`, `sample_fps`, `pipeline_preview`, `relevant_classes`, `rtsp_*`, `record_*`, `svr_*`, `telemetry_*`, `redis_url`, `pgie_config`, ... | unchanged, still top-level |
| `role_overrides`, `untagged_recording_owner` | declared by the product yaml (attendance's), no longer in the shared file |

Two behaviours to know: **an unknown key, a missing required setting, or an unmet `requires` now refuses to start** (with a
message naming it), and **order is part of the configuration** (a module that changes detections acts on what the modules
above it left).

## 4. Porting a change

1. Find the code you changed in section 2 and apply the same change to its new file; for hunks inside `sidecar.py`, the old
   file is still in git (`git show 572aef5:perception/pipeline/sidecar.py`) to locate the logic by function name.
2. Decide which hook it belongs to ([`modules/README.md`](modules/README.md) lists them in order): a decision every frame is
   `on_frame`; changing what is detected is `transform`; events and extra output are `analyze`; reading metadata off an
   object is `extract`.
3. New tunables go in the module's `Settings` dataclass (never a bare constant); a new alert is
   `self.services.alert(...)`; a new model is a `GieSpec` from `gies()` plus a `requirements.txt` for its Python packages.
4. A module never calls another module: data passes through `ObjectView` (`kind`, `attrs`) and `FrameOutput`. pyds is only
   touched through `perception/engine/nvds.py`.
5. Tests: unit-test the module with stand-ins (see `tests/test_zones_module.py`), then run the golden replay for the products
   that list it (`tools/engine_golden_check.sh <names>`, [`TESTING.md`](TESTING.md)).

Things that surprised people while moving the code:

- Object data lives in one list, `frame.objects`; there are no parallel lists. An object's `draw`/`publish` flags say what
  becomes of it.
- Annotations belong to an object, not to a track id: a face box and the person box beside it share an id.
- A module whose `on_built`/`start` fails refuses startup; a per-frame hook failing is only logged and counted.
- `iris:meta` is still the v1 shape (`"v": "1"`); a v2 is planned and will be announced.

## 5. Merge conflicts

If git reports a conflict in (or a deletion of) `perception/pipeline/sidecar.py` on rebase or merge: keep the deletion, then
re-apply your change by hand with the table in section 2. To extract your hunks:

```bash
git diff $(git merge-base origin/main <your-branch>) <your-branch> -- perception/pipeline/sidecar.py > my-sidecar-change.patch
```

Questions: the architecture is in [`ARCHITECTURE.md`](ARCHITECTURE.md), how to add a capability in
[`modules/README.md`](modules/README.md), and the people who wrote the module you are porting into are in `git log` for its
folder.
