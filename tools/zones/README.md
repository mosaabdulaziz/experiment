# Zones live-reload validation (GPU, opt-in)

`validate_reload.py` checks three things on a real pipeline that the unit tests can only simulate:

| | Question | Pass |
|---|---|---|
| a | How long from saving a zone (app API) until the owning worker's nvdsanalytics applies it? 10 edits, min/median/max | every edit's save→effect ≤ 2 s |
| b | Does only the owning worker reload? | bystander's rendered file unchanged (bytes and mtime), it logs `analytics config unchanged` for every version and never `reloaded`, yet still acks; owner logs exactly one `reloaded` per edit (counted by zones version number, so the setup save's reload is never counted) |
| c | After a camera's ROI section is removed from the file and the element reloads, do its objects still get the old ROI labels? And do re-saved zones come back? | no labels on any object frame from 0.5 s after the ack, over 10 s; then a re-save shows its `z0` again within the edit timeout |

Exit code 0 = all three pass, 2 = a check failed **or any step timed out** (waiting for people, an ack, an effect,
the re-save after the removal), 1 = preflight failed. The JSON report (per-edit timings, log lines, the
owner's file after the removal) goes to `data/zones-reload/report-<time>.json`.

## How it works

Two shard replicas of `tests/configs/zones-reload-gpu.yaml` (`IRIS_SHARD_COUNT=2`): `deepstream-sidecar` runs `cam-a`
(enter-3.mp4), `deepstream-sidecar-1` runs `cam-b` (enter-4.mp4). All edits go to `cam-a`, so shard 0 is the owner
and shard 1 the bystander. Zones are saved with `POST /api/streams/cam-a/zones`, as the zone editor does.

The effect is read from the zones module's `roi_probe` (off by default, on in that config only):
`iris:zones:roi_probe:<camera>` gets the ROI labels nvdsanalytics put on each analyzed frame's objects, whether or not
Python still has zones for the camera, which is what makes (c) observable. Zone ids are positional, so edits alternate
A = `[full frame]` and B = `[2 % corner box, full frame]`: B is in effect once any object carries `z1`; A once two
object frames in a row carry `z0` and no `z1` (`z0` is required: after the removal in (c) frames carry no labels at
all, which must not pass for a re-saved A). Each edit reports both save→ack (the owner's `iris:analytics:zones_applied:cam-a`)
and save→effect (first probe frame showing it, host clock; the frame's arrival time is in the report too).

Every effect is read off frames with people in them, and the clips have long empty stretches (`enter-3.mp4` has
nobody from ~65 s to its end at 78 s and in its first ~17 s). So each timed step first waits for 5 object frames in a
row on cam-a (`--people-timeout`, default 120 s, longer than one loop) and starts its clock only then; the wait is in
the report (`people_wait_s`). The workers are stopped with `docker compose stop -t 30` (`--stop-timeout`). Known: they
still exit 137 after the 30 s, because the image's DeepStream `entrypoint.sh` runs `nvidia_entrypoint.sh $@` without
`exec`, so PID 1 is a shell that never passes SIGTERM to Python (no `SIGTERM: stopping the pipeline` in the log). Harmless
here (the run flushes its own Redis), but in a normal stack it skips main.py's shutdown cleanup.

## What it needs

- **GPU**: one NVIDIA GPU with ≥ 4 GiB free (the preflight refuses less). Expected use about 2.5–3 GiB: each worker
  is one DeepStream process at ~1.2–1.5 GiB (CUDA context ~0.4, yolo11s FP16 batch-1 engine + activations ~0.1–0.2,
  NvDCF tracker ~0.2–0.3, NVDEC surfaces for one 720p clip ~0.2), plus ~1 GiB transient while TensorRT builds the
  engine on a cold machine (only shard 0 builds; shard 1 starts after it and reuses the cached engine). The app uses
  no GPU. Nothing else should be running on that GPU: the timings are the point.
- **Time**: warm (engine cached in `data/deepstream-engines`): ~1 min boot + ~1 min for (a)/(b) (10 × ≤ 3.5 s) + ~30 s
  for (c) ≈ **3 min**. Cold: add the engine build, **~5–10 min** for yolo11s FP16.
- **Files in this checkout** (DVC, `dvc pull <path>.dvc`): `data/uploads/attendance/enter-3.mp4`,
  `data/uploads/attendance/enter-4.mp4`, `perception/models/DeepStream-Yolo/yolo11s.onnx`; in git:
  `config_infer_primary_yolo11.txt`, `labels.txt`. The YOLO parser `.so` is in the sidecar image.
- **Images**: the sidecar image under this run's project name, `zones-reload-deepstream-sidecar:latest` (code is
  bind-mounted, so any recent build of this repo's sidecar works: tag one you have, or `docker compose build
  deepstream-sidecar`), and the app image built **from this branch** (`docker compose build iris`, which makes
  `zones-reload-iris`: app code is copied into the image).
- **Compose services**: `redis`, `iris` (the API), `deepstream-sidecar` and `deepstream-sidecar-1` (profile `shards`),
  each started with `--no-deps`. Not needed: MinIO (nothing is recorded, and the app and sidecar store segments
  locally), go2rtc, monitoring, the product sidecars.
- **Host Python**: the project environment (`uv run`), for `redis`.

## Running it

Use a separate compose project and ports so a stack already on the machine is left alone. The script refuses to start
if its project already runs `iris` or a sidecar, flushes **its own** Redis, and backs up then restores this checkout's
`data/zones_config.txt` / `data/filter_zones.json`.

```bash
cd ~/sarah/zones-per-stream
export COMPOSE_PROJECT_NAME=zones-reload IRIS_HOST_PORT=18100 REDIS_HOST_PORT=6390 \
       MINIO_API_HOST_PORT=19100 MINIO_CONSOLE_HOST_PORT=19101 SIDECAR_TELEMETRY_PORT=9111
docker compose build iris
# the sidecar: tag an existing build of this repo's sidecar (find one: docker images | grep deepstream-sidecar; on
# cmubuntu that is iris-deepstream-sidecar:latest) ...
docker tag iris-deepstream-sidecar:latest zones-reload-deepstream-sidecar:latest
# ... or build it under this project (slow: the DeepStream base image):  docker compose build deepstream-sidecar
dvc pull data/uploads/attendance/enter-3.mp4.dvc data/uploads/attendance/enter-4.mp4.dvc \
         perception/models/DeepStream-Yolo/yolo11s.onnx.dvc
uv run python tools/zones/validate_reload.py --check     # preflight: files, images, free GPU memory, project clash
uv run python tools/zones/validate_reload.py             # --keep-up leaves the stack running to inspect
# cleanup: docker compose --profile shards down -v; docker rmi zones-reload-deepstream-sidecar zones-reload-iris
```

## If (c) says the removed section stays live

nvdsanalytics would then be merging a reloaded file into its previous per-stream configuration, not replacing it. The
fixes, in order of preference:

1. **Explicit disabled section** (proposed default): `render_analytics_config` writes `[roi-filtering-stream-<slot>]`
   `enable=0` for every slot this worker owns that has no active zones, instead of omitting it. A present section
   overwrites the stale one. Cost: the file changes for unzoned cameras (the golden fixture is unaffected: its one
   camera has zones). Re-run (c) to confirm.
2. **Placeholder ROI**, if `enable=0` is also ignored on reload: an enabled section with one degenerate ROI
   (`roi-none=0;0;0;0;0;0`) that no object can be inside. The module ignores the `none` label.
3. **Element reset**, last resort: on a reload that removes a section, cycle the nvdsanalytics element through NULL →
   PLAYING (or rebuild it). It is the only option that is certain, but it stalls every stream of that worker for the
   duration, and slot reuse after detach (#173) would pay it every time.

For slot reuse (#173) the same answer matters: a detached camera's slot keeps its section unless one of these lands.
