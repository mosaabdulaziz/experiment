# Testing and validating the perception sidecar

Four layers, from cheapest to most realistic. A change is not done until the layers it can affect pass.

## 1. Unit tests (no GPU, about a minute)

```bash
python3 -m pytest tests -q            # the repo's suite (`just check` is the full gate: it also runs barber's suite and the frontend build)
just barber-test                      # products/barber's own suite
```

Modules are tested with stand-ins (`EngineServices`, `FrameView`, `ObjectView`), never GStreamer: see
`tests/test_zones_module.py`, `test_face_modules.py`, `test_person_modules.py`. Structural tests worth knowing:

| test | guards |
|---|---|
| `test_engine_boundary.py` | nothing in `perception/engine/` imports use-case code or names a product or a use-case label (the ratchet file is empty and stays empty) |
| `test_engine_modules.py` | the loader: unknown names/settings, `requires`, `after`, hook failure isolation |
| `test_module_requirements.py` | every third-party import in a module is declared in a `requirements.txt`; the Dockerfile builds parsers without naming them |
| `test_module_alerts.py` | the alert contract (`services.alert` -> `iris:alert:<cam>` -> app event) |
| `test_engine_draw.py` | annotation priorities and the `draw` hook |
| `test_dockerfile_targets.py` | what each image target must and must not contain |
| `test_product_declarations.py` | every product yaml loads and lists valid modules |

Known failures that predate this work: `test_sidecar_mtmc.py::test_implicit_inactive_transition_and_fifo_topology_association`
(fails on a clean checkout); and `test_multicam_isolation.py::test_the_shared_app_config_starts_none_of_it` fails only
while `config.yaml` holds the multicam preset (`just multicam` writes it; do not commit that).

## 2. The golden replay: did a change alter what the sidecar publishes?

The sidecar is replayed on a short recorded clip with a Redis recorder attached; every write (`XADD`/`SET`/...) is
normalised (JPEG bytes and blobs by length, timestamps dropped, mask outlines by point count) and compared with a
baseline recorded from the code *before* the change.

```bash
just data-pull                                                                    # once: the clips and baselines come from DVC (DATA.md); `dvc repro golden_clips golden_proximity_clip` re-cuts the clips
tools/engine_golden_check.sh attendance zones multicam fire barber barber-faces   # about 14 minutes, the GPU to itself
tools/engine_golden_run.sh tests/configs/golden-X.yaml out.jsonl                  # one replay, one recording
python3 tools/engine_golden.py diff tests/baselines/golden-X.before.jsonl out.jsonl
```

| config | exercises | must be |
|---|---|---|
| `golden-attendance` | face detect + recognition, box stabilizer | IDENTICAL |
| `golden-zones` | zones (intrusion, overcrowding, filter zones), hot-reload ack | IDENTICAL |
| `golden-multicam` | vehicles, plates, MTMC, stabilizer | IDENTICAL |
| `golden-fire` | fire detector, sustained alerts, clip requests | SEMANTICALLY EQUAL (sustain window runs on wall-clock) |
| `golden-barber`, `golden-barber-faces` | segmentation masks, appearance, customer faces | SEMANTICALLY EQUAL, 8 % count tolerance (appearance samples per wall-clock second) |
| `golden-proximity` | YOLO26l-seg masks, NvDCF without class matching, proximity alerts and overlay (no depth) | SEMANTICALLY EQUAL (NvDCF hands some ids out differently run to run, and the learned scale map follows; alerts and counts match) |

Rules learned the hard way (also in the sidecar's memory notes):

- **The GPU and Redis must be yours.** A leftover demo stack (the 5-camera multicam one) writes into the same Redis, the
  recorder never goes quiet, every config runs to its cap (52 minutes instead of 14) and the diffs are meaningless.
- **Never edit repo code while a batch runs**: each config restarts the sidecar from the bind-mounted tree.
- **A stored baseline can go stale** when the image is rebuilt (package versions change a box by 0.1 px or a re-id
  timing). Prove it: run the *original* code on the current image (below); if it equals the new code, the baseline is
  stale and is re-recorded from that run. Do not loosen the comparison.
- To record a baseline from older code: `git archive <old-commit> perception/pipeline products configs/presets | tar -x -C <scratch>`,
  a temporary `docker-compose.override.yml` for `deepstream-sidecar` that mounts those over the repo paths **and every yaml in
  the config's `extends` chain** (a mix of old code and new yaml cost an hour once) and, for code from before the engine
  split, overrides `command:` to the old entry point; run `tools/engine_golden_run.sh`; delete the override.
- Multicam's golden runs on the slim image (no torch), so its Re-ID is off; that is part of what the baseline records.
- Configs that need Redis state first read it from `<config>.redis` (one `KEY=VALUE` per line).

## 3. GPU budget: what does a model cost?

```bash
tools/engine_gpu_budget.sh tests/configs/golden-attendance.yaml [--max-total 80]
python3 tools/engine_gpu_budget.py sidecar.log          # or any log with [telemetry] lines; `-` reads stdin
```

Prints each GPU stage (`gpu.infer.<gie>`, `gpu.decode`, `gpu.jpeg`) with busy %, average and maximum batch time, and the
total. Run it before and after adding a module; the difference is that module's share. The development laptop (RTX 5050,
8 GB) has run out of memory twice: do not run the 13-camera stack there, and start the truck UI only under
`systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 bash scripts/truck_ui.sh up`.

## 4. Live validation, one product at a time

Bring one product up, watch it in the dashboard, check the logs, bring it down (`just down`) before the next.

**What "healthy" is, for every product:** the sidecar log has no `module <name>.<hook> failed`, no `ModuleStartupError`,
no traceback; `[<cam>] NN fps` is steady; `[telemetry] gauge source_stalls_open=0`; `iris:status:<cam>` exists;
`iris:meta:<cam>` grows; the app log has no errors.

```bash
docker logs iris-deepstream-sidecar-1 2>&1 | grep -E "module .* failed|ModuleStartup|Traceback"
docker logs iris-deepstream-sidecar-1 2>&1 | grep -E "\] [0-9.]+ fps" | tail
docker exec iris-redis-1 redis-cli xlen iris:meta:<cam>
```

| product | bring up | look at | specifics |
|---|---|---|---|
| attendance | `SIDECAR_CONFIG=tests/configs/attendance-chirla-cam15.yaml IRIS_PRODUCT=attendance just up cam-15` | dashboard `http://localhost:5180` | faces appear in `iris:face_rec:cam-15` (`unknown` unless the face is registered); the consumer sessions; 30 fps |
| barber | `just barber-up`, then start the streamer yourself if the site's clip files are missing: `cd test_streamer && python3 stream_recordings.py --ids barber-c,barber-l,barber-r --port 8554 --files 'barber-c=<clip>,barber-l=<clip>,barber-r=<clip>'`; app: `IRIS_PRODUCT=attendance,barber docker compose up -d --no-deps iris`, frontend `npm run dev` in `frontend/` | `http://localhost:5180/?product=barber`, API `:8003/docs` | detections on all three cameras, `iris:appearance:*` filling; customer faces need door regions drawn |
| multicam | 5-camera demo (a laptop cannot hold 13): `bash scripts/switch_preset.sh multicam` (**required**: it turns `mtmc` and the truck feed on in `config.yaml`; without it the sidecar tracks but the Trucks page stays empty), then `SIDECAR_CONFIG=tests/configs/multicam-demo5.yaml IRIS_PRODUCT=iris docker compose --profile multicam up -d redis go2rtc minio iris deepstream-sidecar-multicam`, and once the sidecar logs `pipeline PLAYING`, the truck UI under the memory cap above | dashboard `:18000`, trucks `:5173/trucks` (`admin@example.com` / `Admin123!`) | 20 fps, `MTMC initialized`, `Re-ID model loaded on cuda`, `iris:mtmc:assigned/plates/direction` growing, app log `iris.truck_feed: publishing`, `iris.tracking` transitions. `data/filter_zones.json` (gitignored) can hide a truck: a thin zone on cam-01 once made a second truck "untracked" |

Pitfalls: one file source per golden run (the image has no AAC decoder, so clips are 720p without audio); `just up` with
no argument needs the default CHIRLA video; `.env` without `COMPOSE_PROJECT_NAME` names images `iris-*`; on the shared
server never `pkill` by name, kill exact PIDs.

## 5. The regression harness: one command for all of the above, filed per machine

`just regress` runs the unit suites, the golden replays (section 2) and the build-size measurements, takes the GPU cost
of every replay (section 3) plus fps, drops, boot time and engine rebuilds from the sidecar's own telemetry, and writes
one results file under `benchmarks/runs/<machine_id>/`. It refuses to run on a GPU that something else is using (or marks
the run `noisy` with `--allow-noisy`). What each number means, what "correct" is for each solution, and what is not covered
yet is in `benchmarks/SCORECARD.md`; the story is #174. `just regress --only unit,build` needs no GPU (CI runs it as
`regress-cpu`).
