# rtx-5050-laptop: reference benchmark (recorded 2026-11-17/18)

The reference for this machine. A new run is compared with it by `python3 -m regress compare <results.json>` (tolerances in
`tolerances.json`, derived from the noise of the five identical runs below). Raw data of the recording (run files, the 3 h soak, the host
recorder with container logs, provenance snapshots) is kept in DVC, not git; see "Where the raw data is". The golden outputs (`golden/`) are in DVC too: `just data-pull golden`.

## The machine and the conditions
- NVIDIA GeForce RTX 5050 Laptop GPU, driver 595.99.02, CUDA 13.2, 8151 MiB; AMD Ryzen 7 250 w/ Radeon 780M Graphics; 16 threads; 14.9 GB RAM.
- Code: `6e3deb1e` (main `873fa6fa` with the regress harness); the safety soak ran the stack from main `873fa6fa`.
- On AC for 100 % of the recording, GPU idle before every run (the harness checks), no other job started. Desktop apps (Teams and others) were
  open but idle, and about 1.1 GB of swap was already in use at the start (almost no swap traffic: one short burst during the soak). The laptop GPU runs into its
  software power cap near 60 W: the `throttled` hints in the results say how often; it is part of this machine's behaviour, not noise.
- The host recorder (`regress hostsense`, branch feat/soak-hostsense) ran alongside at the lowest priority; the runs themselves used
  the harness at `6e3deb1e`.
- This is a laptop: its numbers say how the pipeline behaves on THIS machine and are for comparing a change with its own past. They are
  not the speed of the server or the Orin.

## What was recorded
| part | what | size |
|---|---|---|
| golden replays | the default six configs x 11 runs (the last 3 about 3 hours after the others: the baseline holds, all three passed); PPE, fall, shelf x 3; proximity x 2 warm runs | `baseline.json` (medians), `golden/` (outputs) |
| noise | spread of every number between identical runs; 11 unstable metrics are not gated | `tolerances.json` |
| build and unit | image sizes, frontend bundle, lock files, 2810 core tests (0 failing, ~92 s) | in `baseline.json` |
| cold engines | all goldens with the engines moved aside (the first config to need an engine pays its build) | table below |
| profile replay | every config once under the 5 s container sampler: CPU, memory, network, I/O | table below |
| two 3 h soaks | the safety stack (PPE + fire + zones, 3 cameras, looping demo clips) sampled every 30 s | `soak-reference.json`, `soak-reference-2.json`, `soak-variability.txt` |
| host recorder | pressure, swap, CPU split and frequency, temperatures, fan, AC, NVMe I/O, container events | `host-reference.json` |
| manifest | sha256 of every engine, model, clip, config; image ids | `reference-manifest.json` |

## Golden replays (medians of the identical runs)
| config | entries | fps p10 | boot s | drops | GPU util % | SM active | power W | GPU MiB added | output reproducible |
|---|---|---|---|---|---|---|---|---|---|
| attendance | 796 | 20.0 | 8.0 | 0 | 58 | 0.61 | 55 | 1936 | yes |
| zones | 806 | 20.0 | 7.9 | 0 | 58 | 0.61 | 54 | 1936 | yes |
| fire | 624 | 30.0 | 7.9 | 0 | 96 | 0.88 | 60 | 1932 | yes |
| multicam | 168 | 19.9 | 19.7 | 0 | 6 | 0.00 | 11 | 1402 | yes |
| barber | 1146 | 25.0 | 9.8 | 0 | 27 | 0.33 | 28 | 2824 | yes |
| barber-faces | 2632 | 25.0 | 9.7 | 0 | 34 | 0.44 | 37 | 2978 | **no** |
| ppe | 207 | 25.0 | 6.7 | 0 | 98 | 0.92 | 60 | 1294 | **no** |
| fall_detection | 214 | 17.4 | 6.8 | 0 | 98 | 0.93 | 60 | 2960 | **no** |
| shelf | 214 | 29.0 | 6.6 | 0 | 98 | 0.90 | 60 | 1064 | **no** |
| proximity | 437 | 20.0 | 6.5 | 0 | 28 | 0.17 | 25 | 1526 | yes |

- **Output reproducible = no** (barber-faces, PPE, fall, shelf): identical runs analyse other frames or hand out random ids, so their
  per-frame output is not a correctness oracle. A changed output only WARNS for them; their fps, drops, GPU counters and stage timers repeat
  tightly (spread 0.1-5 %) and are gated.
- attendance (796 entries) and zones (806) are `IDENTICAL` to the repo baselines in all five runs; fire is `SEMANTICALLY EQUAL` x5 (624 or 623
  entries). multicam (168 entries every run) and barber show the known difference against the repo baselines, which were recorded on another
  GPU; barber's entry count moves between identical runs (1142 to 1146), so for it the comparison is semantic with an 8 % count tolerance.
- Proximity replays on the laptop only when its clip and baseline are pulled (`just data-pull golden`) and its 736x1280 segmentation model is exported
  (`just proximity-models`); it is then replayed by default. Its output is `SEMANTICALLY EQUAL` to the repo's own baseline (recorded on the server) and
  gave the same 437 entries in all three runs. Its first run built the TensorRT engine (2.7 GB of RAM), so the baseline uses the two warm runs.
- Not covered: the host fall service (the replays run the sidecar only).

## Cold engine builds (boot to first frame, seconds)
| config | warm | cold | engines rebuilt |
|---|---|---|---|
| attendance | 7.9 | 73.9 | ?, pgie, pgie_fire, sgie_face |
| multicam | 40.4 | 59.2 | ?, pgie |
| barber | 9.8 | 32.2 | pgie |
| ppe | 6.7 | 31.9 | pgie_ppe |
| fall_detection | 6.8 | 132.3 | sgie_pose |
| shelf | 6.7 | 22.0 | pgie_shelf |

A rebuilt TensorRT engine is not bit-reproducible: after the cold run the golden outputs of attendance, zones and multicam differ from the
repo baselines (`WARN: a TensorRT engine was rebuilt`), and the restored engines give `IDENTICAL` again. The manifest records the engine hashes so a
changed output can be tied to a changed engine.

## Resources and CPU-side stage times per replay (gated; medians of 3 recorded runs, stage times of 8)
The sidecar container while the replay ran, and the time per call of its CPU-side stages (the GPU stages are in the table above).
| config | CPU cores | anon MiB (max) | threads | probe ms | publish ms | publish.draw ms | Redis write ms |
|---|---|---|---|---|---|---|---|
| attendance | 0.62 | 1270 | 109 | 2.29 | 25.64 | 24.82 | 0.85 |
| zones | 0.63 | 1267 | 109 | 2.35 | 25.63 | 24.91 | 0.83 |
| fire | 0.57 | 1214 | 110 | 1.23 | 25.80 | 33.16 | 0.75 |
| multicam | 0.28 | 1329 | 82 | 1.38 | 0.11 | - | 0.09 |
| barber | 0.39 | 1386 | 128 | 1.02 | 0.72 | - | 0.71 |
| barber-faces | 0.67 | 1402 | 128 | 1.43 | 0.81 | - | 0.80 |
| proximity | 0.49 | 608 | 74 | 2.46 | 0.29 | - | 0.26 |

- CPU cores, anon memory and the stage times per call are gated (anon: 8 % and 108 MiB; probe 18 %). The cgroup peak (it includes page cache), a stage's
  share of the replay and the flight-recorder CPU are recorded and shown, not gated: they vary too much between identical runs.
- File descriptors are not recorded for the replays: the container runs as root and its `/proc` is closed to this user. The soak records them for the safety stack.
- publish.draw is the preview drawing (CPU); in attendance, zones and fire it is about 25 ms (fire 33 ms) per published frame; barber, multicam and barber-faces have no preview draw stage, their whole publish step takes under 1 ms.

Network and disk of one profile pass (not gated):
| config | net MB/s | disk write MB |
|---|---|---|
| attendance | 1.82 | 16 |
| zones | 1.8 | 13 |
| fire | 0.99 | 13 |
| multicam | 0.0 | 16 |
| barber | 0.04 | 20 |
| barber-faces | 0.06 | 23 |

## Safety-stack soak (two 3 h runs, 20 min warm-up cut; 3 cameras: PPE, fire, zones)
Two identical soaks of the same stack on the same laptop, the second about 20 hours after the first. They agree closely, so the gate could be calibrated.
| series | soak 1 | soak 2 | ratio |
|---|---|---|---|
| `alert_clips` growth, MiB/h | 3618 | 3599 | 1.01 |
| `alert_svr` growth (unclaimed smart-record files), MiB/h | 129 | 152 | 1.18 |
| zones sidecar anon memory, MiB/h (fitted) | 42 | 35 | 1.20 |

- Disk is what grows: `alert_clips` +3.6 GB/h in both (about 8-9 MB per incident event, roughly 400 events/h at the demo rate, far above a real site), `alert_svr`
  +129 and +152 MiB/h (unclaimed smart-record files, 24 to 27 files/h). The 6 h prune of `alert_clips` could not be observed in 3 h.
- Memory: ppe, redis, go2rtc and minio are flat. The zones role grows in both soaks, about 130 MiB over 3 h (anon fitted +42 and +35 MiB/h, R2 0.7;
  two steps of about 66 MiB in the first, a noisier drift in the second): reproducible, worth watching on a longer run, not a proven leak. fire's container
  memory grows by page cache (+99 to +125 MiB/h file-backed); its anon swings between 720 and 1040 MiB with no trend.
- Environment effects in the second soak, not the pipeline: the app container's page cache grew (+610 MiB/h file-backed, from building alert clips) and the host
  swapped (about 800 MiB of swap was already in use at the start and grew by 450 MiB), which showed up as 65 MiB/h of swap in the ppe sidecar.
- Threads (72-83 per sidecar), file descriptors (76-93) and Redis memory (12-16 MiB) stayed bounded; GPU memory per container grew once by about
  26 MiB in fire and in ppe and not at all in zones; 8 Redis streams filled to their caps and stayed there; no restarts, no OOM, in both soaks.
- The soak gate (`regress soak-compare`) is calibrated on these two: slopes differed by at most 1.20x (anon, 7 MiB/h), 1.18x (disk bytes) and 1.16x (disk files),
  so it warns at 2x for anon memory and 1.5x for disk. Two soaks are still two: check again after a machine or stack change.
- Host (first soak): CPU was busy (tasks waited for a CPU 24 % of the time on average, p95
  40 %), the CPU ran at 74 C (max 77),
  memory stall about 0 %, swap activity negligible. Read the CPU figures with this in mind: the demo cameras are three `ffmpeg` processes on the same laptop
  re-encoding the clips into camera-like H.264 streams (a real site has none), so part of that CPU load and heat is the test rig, not the pipeline. The streams
  were RTSP through go2rtc at 720p H.264; 1080p, 4K and H.265 sources were not measured.

## Limits (un-paced replay, 720p H.264 file sources; see `capacity-reference.json`)
How fast each use case can go on this laptop, and how many cameras at the source rate it keeps up with (every camera at 95 % of its source rate,
no drops, no stall).
| use case | source fps | 1-camera ceiling fps | headroom | cameras kept up with | aggregate ceiling |
|---|---|---|---|---|---|
| attendance | 20.0 | 28.5 | 1.43x | 1 (tried [1, 2]) | 26.1 fps at 2 cameras |
| zones | 20.0 | 28.8 | 1.44x | 1 (tried [1, 2]) | 26.3 fps at 2 cameras |
| fire | 30.0 | 30.6 | 1.02x | 1 (tried [1, 2]) | 27.4 fps at 2 cameras |
| barber | 25.0 | 93.95 | 3.76x | 2 (tried [1, 2, 4]) | 86.0 fps at 4 cameras |
| shelf | 30.0 | 29.2 | 0.97x | 1 (tried [1, 2]) | 25.8 fps at 2 cameras |
| ppe | 30.0 | 25.0 | 0.83x | 0 (tried [1]) | 25.0 fps at 1 cameras |
| fall_detection | 25.0 | 17.9 | 0.72x | 0 (tried [1]) | 17.9 fps at 1 cameras |
| proximity | 20.0 | 76.95 | 3.85x | 2 (tried [1, 2, 4]) | 74.8 fps at 4 cameras |

- PPE (25 fps) and fall (17.9 fps) cannot keep up with one full-rate camera; fire and shelf have almost no headroom.
- If a role only needs a lower analysis rate (the pipeline has `analysis_fps`), the same GPU serves proportionally more cameras.
- Barber reaches 94 fps: it has no preview (JPEG) stage and its detector stage takes about 10 ms per frame against 27 ms in attendance.

## Cameras at a 5 fps analysis rate (the story's cells; `regress capacity --analysis-fps 5`)
Each camera analysed at 5 frames a second instead of every frame; the same clip as 1, 4 and 8 cameras (the ladder stops at the first count not kept up with).
| use case | cameras kept up with | tried | note |
|---|---|---|---|
| attendance | 4 | [1, 4, 8] | 8 cameras did not |
| zones | 4 | [1, 4, 8] | 8 cameras did not |
| fire | 4 | [1, 4, 8] | 8 cameras did not |
| barber | 8 | [1, 4, 8] | all tried counts held |
| shelf | 4 | [1, 4, 8] | 8 cameras did not |
| ppe | 4 | [1, 4, 8] | 8 cameras did not |
| fall_detection | 1 | [1, 4] | 4 cameras did not |
| proximity | 4 | [1, 4, 8] | 8 cameras did not |

- Gated metric: `capacity.<use case>@af5.realtime_cameras` (compare runs with the same camera counts). The un-paced 1-camera ceilings above are gated separately.
- Fall is the only use case that cannot take 4 cameras at 5 fps: its pose model (yolo26x-pose) is the heaviest per frame.

## What limits attendance, zones and fire: the detector, not the preview
Attendance, zones and fire all run yolo11x at FP32 and all stop at about 26 to 28 fps. Swapping the attendance detector (same clip, un-paced):
| detector | precision | ceiling fps | aggregate with 2 cameras | model alone, GPU ms per frame (trtexec) | first boot with engine build, s |
|---|---|---|---|---|---|
| yolo11x | fp32 (the product's default) | 28.35 | 26.0 at 2 cameras | 23.1482 | 8.1 |
| yolo11l | fp32 | 52.1 | 48.2 at 2 cameras | 10.8198 | 29.5 |
| yolo11m | fp16 | 188.0 | 139.4 at 2 cameras | 3.125 | 24.9 |
| yolo11s | fp16 | 237.8 | 241.8 at 2 cameras | 1.39856 | 23.6 |
| damoyolo_s | fp16 | 232.4 | 239.3 at 2 cameras | 1.87244 | 22.1 |
| yolo11x-fp16 | fp16 (forced) | 90.4 | 82.2 at 2 cameras | 7.77692 | 30.2 |
| yolo11l-fp16 | fp16 (forced) | 150.95 | 134.5 at 2 cameras | 4.10742 | 27.6 |

- The FP32 yolo11x and yolo11l set the ceiling (the model alone takes 23 and 11 ms per frame). The same yolo11x at FP16 reaches 90 fps, yolo11s about
  240. The small models stop near 235 fps, where something other than the detector limits (decode, probe, publish: not attributed here).
- This answers a suspicion of an earlier draft: the preview JPEG encode is not what caps these use cases.
- The product config says the X-tier FP32 detector "started as a one-off tier comparison". A precision change alters outputs and was not checked for
  accuracy; this is a finding, not a change.

## Using it
```
python3 -m regress run --only golden --configs zones --out /tmp/zones.json       # one config
python3 -m regress compare /tmp/zones.json                                       # judged on what it covered; --strict wants everything
python3 -m regress compare /tmp/all.json --strict                                # full run
python3 -m regress soak --project <p> --duration 3h ...  &&  python3 -m regress soak-analyze <dir>
python3 -m regress hostsense --out <dir>                                         # what else the machine was doing
```
Efficiency metrics (fps, drops, GPU counters, stage timers) only warn by default; correctness (unit tests, golden outputs) fails.
After a change of GPU driver, DeepStream, TensorRT or the machine itself the reference has to be re-recorded.

## Where the raw data is
In DVC (remote `work`), not in git, so the repository stays small:
- `benchmarks/raw/rtx-5050-laptop.dvc` -> `benchmarks/raw/rtx-5050-laptop/` (82 MB, 288 files): the run files, `soak/`, `hostsense/` with container logs,
  `profile-soak/`, capacity and detector-size results, the overnight run, provenance snapshots and the drivers that produced all of it. Get it with
  `just data-pull benchmark-raw`. Its own README says what each folder is. Recording took about 17 h on the laptop (2026-11-17 to 11-18).
- `benchmarks/baselines/rtx-5050-laptop/golden.dvc` -> `golden/` (3.7 MB, 10 outputs): what `regress compare` diffs a changed output against.
  Get it with `just data-pull golden`. Without it a changed output still warns, it just cannot say what changed.
Everything else here (`baseline.json`, `tolerances.json`, the references and the manifest) is small and stays in git.
