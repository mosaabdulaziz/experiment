# Scorecard: what "still works" and "not slower" mean, per solution

The regression harness (`just regress`, issue #174) records the numbers below for a checkout on one machine, so a merge
request can be compared with `main` and nothing gets worse unnoticed. This page says **what is measured, what counts as
correct, and what is not covered yet.** The commands are at the end.

## What one results file holds

One JSON file per run, filed under the machine's id (`benchmarks/runs/<machine_id>/`, not committed; the baselines that
are committed live in `benchmarks/baselines/<machine_id>/`, task #228).

| section | metric | unit | direction | gate (planned, tasks #228/#230) |
|---|---|---|---|---|
| **fingerprint** | machine_id, GPU, driver, CUDA / L4T, CPU, RAM, git commit and dirty flag, power mode (Jetson) | | | decides whether two results are comparable |
| **guardrails** | GPU busy before the run (median of 5 readings), other compute processes, other stacks, battery | % | | a run that is not idle is marked `noisy` |
| **correctness** | unit suites: tests, failures, errors, skipped, time, failing ids, 10 slowest | count, s | failures must stay 0 | **fails** |
| | golden replay per config: entries recorded, sha256 of the output, diff against the baseline | | identical (or semantically equal) | **fails** |
| **runtime**, per golden config | stage timers from the sidecar: time each GPU stage was executing (`gpu.infer.*`, `gpu.decode`, `gpu.jpeg`) and their sum | % of time | lower is better | warns |
| | fps per camera: median, p10 | fps | higher | warns |
| | drops: `qos_dropped_total`, `publisher_drops`, `pub_shed_total`, stall episodes | count | 0 | warns |
| | boot to the first fps line; which TensorRT engines were cached or rebuilt | s | lower | warns (a rebuild is flagged, not compared) |
| | GPU memory added while it ran | MiB | lower | warns |
| | CPU-side stage timers from the sidecar's telemetry (probe, publish, draw, Redis write, upload...): time per call and share of the replay | ms, % | lower | warns |
| | the sidecar container while it ran: CPU cores, RAM (anon, cgroup peak), threads, file descriptors | cores, MiB, count | lower | warns |
| | flight recorder: process CPU %, main-loop lag (a sample every 30 s: one or two per replay) | %, s | lower | warns |
| **capacity**, per use case (`regress capacity`) | un-paced 1-camera ceiling, and the cameras kept up with (every camera at 95 % of its source rate, no drop, no stall); also at a 5 fps analysis rate | fps, cameras | higher | warns |
| **hardware counters**, per golden config, while the pipeline ran (1 Hz; median, p95, max) | **SM active** (share of time SMs had a warp resident), **SM occupancy**, **tensor / FP16 / FP32 pipe active**, **DRAM active**, graphics-engine active (DCGM profiling) | ratio | see the hints | warns |
| | GPU time and memory-controller time (NVML; the number nvidia-smi shows), **NVDEC, NVENC, NVJPG, OFA** engine load | % | lower | warns |
| | SM and memory clock against the maximum, power draw, temperature, **throttle reasons** (power cap, thermal), PCIe throughput | MHz, W, % of samples | clocks near max | warns |
| | on a Jetson: GR3D, EMC (memory) load, NVDEC / NVENC / NVJPG / VIC / OFA clocks, power rails (tegrastats) | % , MHz, mW | | warns |
| | **hints**: where to look first (see below) | text | | information |
| **build** | image sizes (sidecar slim, mtmc, app) | MB | lower | warns |
| | frontend bundle: total, JS, CSS, largest chunk | bytes | lower | warns |
| | packages in every `uv.lock` | count | stable | warns |

Efficiency only **warns**; correctness **fails**. How the comparison decides (`tools/regress/compare.py`):

- **FAIL (exit 1):** a unit suite with more failures than the baseline or fewer tests; a golden replay that did not run or whose
  output differs from the baseline's (byte-identical, or the replay's own semantic diff for the configs compared that way).
- **WARN:** an efficiency metric that moved the wrong way by more than its tolerance (`benchmarks/tolerances.json`, proposed by
  `just regress-noise` from repeated identical runs, never tighter than the noise they showed). `--strict` makes warnings fail.
  A move the right way is listed as IMPROVED.
- **Not directly comparable:** efficiency means something only against a baseline from the same `machine_id`. Against another
  machine's baseline the efficiency rows are INFO and the golden outputs are not judged (a different GPU or TensorRT changes
  detections without anything being wrong); unit suites still are. A run marked NOISY is advisory the same way.
- A baseline is the median of N runs of one clean commit (`just regress-baseline`), stored with the golden outputs so a changed
  output can be diffed, not just flagged.

## Reading the numbers

- **nvidia-smi's "GPU utilisation" and the sidecar's stage "busy %" are both TIME measures.** They say how long something
  was running, not how much of the GPU it used: a GPU reads 90% while one small kernel at a time runs on a few SMs. On the
  RTX 5050 laptop with only the desktop running, nvidia-smi says 39%, while DCGM says the SMs were active 0% of the time
  and the graphics engine 3.6%. During the multicam replay the plate stage's timer says 89% busy while the GPU was busy only
  15-19% of the time and the SMs were active in short bursts (median 0%, p95 about 46%); two runs on 2026-11-10/11 gave
  the same picture, so it is the workload, not a bad counter window. So efficiency is judged on the **hardware counters**: SM active, SM occupancy, the tensor / FP16 /
  FP32 pipes, DRAM, and each hardware engine (NVDEC, NVENC, NVJPG, OFA) separately; the timers say *where* time goes, the
  counters say *how well the hardware is used*.
- **The stage timers overlap and a replay runs faster than live.** The stages run in parallel, so their shares can add up
  past 100%, and a golden clip is read as fast as the pipeline allows. Compare a branch with `main` on the same machine;
  do not read a camera count from them: the limits per use case are measured un-paced by `regress capacity` (below), and the
  team's `benchmarks/stream_capacity.py` sweep (#149) stays the server's N-stream saturation grid.
- **Clocks and power change the numbers.** A laptop GPU runs at its power cap under load (the RTX 5050 held about 56 W and
  2.3 GHz of 3.1 GHz max, power-capped in every sample of the attendance replay), so the same code gives different times at
  different temperatures. The counters record clocks, power and throttle reasons so a difference can be explained.
- **Compare on the same machine, preferably idle.** A different GPU changes every efficiency number and can change the
  output of a replay (the baselines recorded in `tests/baselines/` came from the GPU server). Hence one baseline per
  `machine_id`, and a `noisy` flag when something else was using the GPU.
- **A rebuilt TensorRT engine makes the boot time meaningless** (minutes instead of seconds): the file says
  `engine: rebuilt` so it is not mistaken for a regression.
- **The RTX 5050 laptop drives its display on the same GPU** (about 40% busy with nothing running on 2026-11-10; only
  `Xorg` attached). Its idle limit is raised with `REGRESS_IDLE_GPU_UTIL=60` and the background load is written into
  every result. A baseline and the run compared with it must be taken with a similar background.

## The hardware counter layers, and what each machine supports

| layer | what it adds | needs | RTX 5050 laptop | Jetson AGX Orin | 5090-work, iris-work |
|---|---|---|---|---|---|
| **NVML** (read through nvidia-smi) | GPU / memory-controller time, NVDEC / NVENC / NVJPG / OFA load, clocks, power, temperature, throttle reasons, PCIe | any NVIDIA GPU | yes | no (tegrastats instead) | not probed yet |
| **DCGM profiling** (the repo's pinned dcgm-exporter image) | SM active, SM occupancy, tensor / FP16 / FP32 pipes, DRAM active, graphics engine, PCIe bytes | docker, `--gpus all`, `--cap-add SYS_ADMIN`; the GPU and driver must expose profiling | **yes** (tested 2026-11-10) | not supported on Jetson | not probed yet |
| **tegrastats** | GR3D, EMC, media engines, power rails | a Jetson | n/a | yes | n/a |
| **Nsight Systems / Compute** (`nsys`, `ncu`, with CUDA) | per-kernel timelines, memory-bound or compute-bound, why one kernel is slow | profiling permission | installed | JetPack tools | not probed |

`just regress-gpu-probe` samples for a few seconds and prints which layers answered; run it when a machine is onboarded and
write the answer into this table. A layer that is not supported is recorded as "not available" with the reason in every
result, never as a zero. The first three layers run on every `just regress`; Nsight is the **deep dive** you start when a
hint points somewhere (`nsys profile` on one golden config), not something to run on every merge request.

**Hints** are heuristics that turn the numbers into "look here first": decode or JPEG engine near its limit (p95 >= 80%), the
GPU busy by time but the SMs mostly idle (small batches or serial launches), a stage busy by its timer while SMs idle,
memory-bound (DRAM >= 60%), tensor cores idle while SMs are busy (models on the FP32 path), throttling (>= 5% of the
samples), clocks well under the maximum while busy. They point a profiler at the right place; they are not verdicts.

The idle check uses the graphics-engine activity DCGM measures (nvidia-smi's utilisation where DCGM is not available):
the desktop on the RTX 5050 laptop reads 39% on nvidia-smi and about 4% on the engine counter, and only the second says
something else could contend with a measurement.

## Per solution

"Efficient" is the same everywhere (the runtime and build rows above, from the solution's golden config); the interesting
column is what "correct" can be checked against today.

| solution | "correct" means | what checks it today | not covered yet |
|---|---|---|---|
| **Engine** (`perception/engine`) | no use-case code in it; every golden config unchanged | `tests/test_engine_boundary.py` (ratchet at 0), engine unit tests, all golden replays | none |
| **Attendance** (+ zones) | the same faces named, the same zone alerts, the same segments | `golden-attendance`, `golden-zones` (exact); `tests/test_attendance_*`, face-module tests; CI `attendance-service` imports the service | accuracy against labelled faces: no ground truth in the repo (`capture_baseline.py` records counts for one clip, by hand) |
| **Fire** | the same fire / smoke alerts at the same times | `golden-fire` (semantic: kind, class, time within 0.5 s); 5 unit-test files | accuracy on labelled clips |
| **Barber** | the same appearance and customer-face counts | `golden-barber`, `golden-barber-faces` (semantic, 8% on counts); the product's own suite (923 tests, CI `barber-tests`) | none beyond counts |
| **Multicam** (vehicles, plates, MTMC) | the same metadata, MTMC assignments, plate reads | `golden-multicam` (exact, on the mtmc image); MTMC and plate unit tests | IDF1 (`mtmc/evaluate.py`) and plate accuracy (`mtmc/eval_ocr_quality.py`) are manual, with no stored expected numbers |
| **Fall detection** | the same detections and alerts | the product's own suite (193 tests, CI `fall-detection-tests`); `golden-fall_detection` replays a 31 s clip (laptop reference, 3 repeats: 17.4 fps, GPU 98 %) | the replay's output differs between identical runs, so a changed output only warns; precision / recall / false alarms per hour (`evaluate_benchmark.py`) is manual and needs an outside corpus. The host fall service is not benchmarked (accepted risk) |
| **PPE** | the same violations on the demo clips | 14 unit-test files; `golden-ppe` (laptop reference, 3 repeats: 25 fps, GPU 98 %) | the output differs between identical runs, so a changed output only warns; accuracy (#199) |
| **Shelf void** | the same voids raised and cleared | 3 unit-test files; `golden-shelf` (laptop reference, 3 repeats: 29 fps, GPU 98 %) | the output differs between identical runs, so a changed output only warns |
| **Proximity** | the same close-pair alerts | 12 unit-test files; `golden-proximity` (semantic: ids differ run to run, alerts and counts match), replayed by `just regress` and `engine_golden_check.sh` once `just data-pull golden` has fetched its clip and baseline and `just proximity-models` has exported its segmentation model. Laptop reference: 3 replays, 437 entries each, `SEMANTICALLY EQUAL` to the repo baseline recorded on the server, 20 fps, ceiling 77 fps | a machine that has not pulled the clip and exported the model skips it (the run says so); accuracy (#199) |
| **Plates, VLM** | the same plate reads / VLM verdicts | plates: the plate stage runs in the multicam golden (exact); VLM: unit tests only | accepted risk: no stand-alone plate or VLM replay (the VLM is an external, non-deterministic model; its soak is #153) |
| **el-mohandes, othaim** (client sites) | the same output on their cameras | covered through the multicam config; no client-specific replay | accepted risk (recorded in task #229) |
| **App and dashboard** | the API, events, clips and zones behave as before | about 2,300 unit tests under `tests/` | the frontend has no tests: `npm run build` (type check) only; the bundle size is recorded. Accepted risk |

Accuracy numbers (precision, recall, false alarms per camera-hour on labelled clips) are the job of story #199; its
harness will write them into the same results file, and the planned correctness gate will compare them too.

## Which check for which change

`python3 -m regress plan` reads the changed files (against the merge-base with `origin/main`) and prints the cheapest checks that can notice them,
from `benchmarks/tiers.json` (first matching rule per file; a file no rule matches counts as a shared change and says so):

| tier | for | runs | about |
|---|---|---|---|
| 0 | docs, frontend, tooling, app code the replays do not run | unit suites and build sizes | 2 min |
| 1 | one use case's module, product or golden config | its golden replay and 1-camera ceiling, compared with the baseline | 9 min per use case |
| 2 | the shared engine, pipeline, models, images, dependencies | every golden replay and ceiling | 50 min |
| 3 | recording, alert clips, incidents, the safety stack | tier 2 plus a 1 h soak gated by `soak-compare` | 2 h |

Durations are what the laptop reference measured. After an image, dependency or driver change see "Refreshing a baseline".

## Accepted risks (not covered, and why)

A use case or layer that has no check today, so nobody assumes otherwise:

| not covered | why |
|---|---|
| **Host fall service** (the product service, not the sidecar role) | the replays run the sidecar only; its own suite has 193 tests |
| **Plates stand-alone, VLM** | plates run inside the multicam golden; the VLM is an external model with non-deterministic output (its soak is #153) |
| **App timings** (API latency, clip building, MinIO throughput, database growth) | behaviour is covered by about 2,300 unit tests; the soak watches the app's memory and disk, not its speed |
| **Stream types** | the clips are 720p H.264 files (fall 320x240); 1080p, 4K, H.265 and RTSP ingest are not measured, so the camera counts are specific to that |
| **Other machines** | only the RTX 5050 laptop has a recorded reference; the GPU server, the 5090 and the Orin each need their own (the Orin runs the attendance replay on `feat/orin-sidecar`) |
| **Output of PPE, fall, shelf and barber-faces** | identical runs analyse other frames or hand out random ids, so their per-frame output is not an oracle: a change only warns; their speed, drops, GPU counters and resources are gated |
| **Soak gate** | calibrated on two identical 3 h soaks of the safety stack (slopes differed by at most 1.20x for anon memory, 1.18x for disk bytes), so `soak-compare` warns at 2x for anon memory and 1.5x for disk; two soaks are still two, so check it again after a machine or stack change |
| **Accuracy** | story #199 |
| **Which stage limits a use case** | the resource numbers say what is full, not what limits the rate: story #268 |

## Refreshing a baseline

A baseline belongs to a machine, a GPU driver, a CUDA / DeepStream / TensorRT version and a set of TensorRT engines. Record it again when any of
those change (a rebuilt engine changes golden outputs slightly: the compare command warns instead of failing, and `reference-manifest.json` holds the
engine hashes so a difference can be tied to one). On a quiet machine: `just regress-baseline` (the default six, five runs), the PPE, fall and shelf
repeats, `regress capacity` for the ceilings, a 3 h soak with `regress hostsense` running, then `regress noise` for the tolerances. The laptop's
reference (`benchmarks/baselines/rtx-5050-laptop/README.md`) lists what was recorded and how. The golden outputs and the raw run data of a baseline go to
DVC, not git: `dvc add benchmarks/baselines/<machine>/golden benchmarks/raw/<machine>`, `dvc push`, commit the `.dvc` pointers (the catalog group
`benchmark-raw` and `golden` in `data/catalog.yaml` already match them).

## Commands

```bash
just regress                         # everything on this machine (about 25 minutes, GPU to itself)
just regress --only unit,build       # the part that needs no GPU (CI runs it as `regress-cpu`)
just regress --only golden --configs attendance,zones
just regress-idle                    # is this machine idle enough? exit 0 = yes
just regress-gpu-probe               # which GPU counter layers does this machine answer?
just regress-show benchmarks/runs/<machine_id>/<file>.json
just regress-baseline                # this machine's baseline: 3 runs of a clean commit, the median (about 1 hour)
just regress-compare benchmarks/runs/<machine_id>/<file>.json   # PASS / WARN / FAIL against that baseline
just regress-noise benchmarks/runs/<machine_id>/*.json --tolerances benchmarks/tolerances.json   # tolerances from 5+ identical runs

python3 -m regress plan --base origin/main      # which checks THIS change needs: tier, configs, exact commands, minutes
python3 -m regress capacity --configs ppe --cameras 1 [--analysis-fps 5]   # un-paced ceiling, cameras kept up with
python3 -m regress capacity --models all --cameras 1,2                      # the attendance detector swapped per size (engine slots protected and restored)
python3 -m regress capacity --engines                                        # trtexec: GPU time per frame of each detector alone
python3 -m regress soak --project safety --duration 3h ...                  # long-run growth of a running stack; soak-analyze, soak-compare against the reference
python3 -m regress hostsense --out <dir>                                    # what else the machine was doing: pressure, temperatures, throttling, container events and logs
python3 -m regress backfill --log <sidecar log> --out <dir> run-*.json      # add the CPU-side stage timers to older results from a saved log
```

`benchmarks/known_failures.txt` lists tests that are known to fail (empty today); they do not fail the verdict but are
printed on every run. A test must not be added there to make a merge request pass.

Reused, not rebuilt: `tools/engine_golden_run.sh` and `engine_golden.py` (the replay and its comparison),
`tools/engine_gpu_budget.py` (GPU % per stage), the boot and engine-rebuild rules of `benchmarks/run_bench.py`, the
sidecar's own telemetry, and the junit file CI already writes. Capacity (how many cameras fit) stays with
`benchmarks/stream_capacity.py` and `just bench-regress` (#149).
