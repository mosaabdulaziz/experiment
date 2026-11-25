# Pedestrian-vehicle proximity

A safety alert when a person stays too close to a moving vehicle inside a zone, on the yard, warehouse and
loading-bay cameras of the multicam sites. The sidecar overlay is `sidecar.yaml`; the rule is the `proximity` module
(`perception/modules/proximity/README.md`); the design and the measurements behind it are
`RESEARCH_pedestrian-vehicle-proximity_2026-10-26.md` (branch `analysis/ped-vehicle-proximity`).

## What runs

- **Detector:** YOLO26l-seg COCO at 1280x736: persons, cars, trucks, buses, motorcycles and their instance masks in one
  model. Forklifts come through as truck or car. YOLO26x-seg is the upgrade for cameras where distant people or
  forklifts matter (2.2x the GPU).
- **Tracker:** the attendance-tuned NvDCF (long shadow tracking) with class matching off, so a forklift that flips
  between truck and car keeps its id, and NVIDIA's accuracy-preset visual tracker (NvDCF_VPI with HOG and larger
  feature images), so a person briefly missed is picked up again under the same id
  (`perception/pipeline/config_tracker_nvdcf_proximity.yml`).
- **Modules:** `proximity` only. It tests zone membership itself, so this product does not run `zones` (no duplicate
  intrusion alerts when the multicam sidecar watches the same cameras).
- **Cadence:** the detector on every 3rd frame, ~6.7 analysed frames/s per 20 fps camera (`process_fps: 2.5`;
  nvinfer's `interval` counts the frames it skips). The module runs on the frames nvinfer actually inferred.

## First run

```bash
just proximity-models          # once: both ONNX files (weights cached in models/proximity/)
docker compose --profile proximity up -d redis minio go2rtc deepstream-sidecar-proximity
```

Turn proximity on per zone in the dashboard's zone editor (its **Proximity** section: vehicles, max gap, the depth
check and its cut-off, and the rest under Advanced); saving applies it live. The depth check is on by default, at a
cut-off of 0.25 (Ticket 6); a pair it holds back raises no alert but is kept, with its clip and depth image, on the
Incidents page behind **Held back by depth**, to tune a zone's cut-off by. The zone file keeps only what differs from the defaults
(`perception/modules/proximity/zone_defaults.json`), so zones follow later default changes. Each incident is one card
on the Incidents page: severity, ongoing or resolved, its facts, a snapshot at the minimum gap and its clip(s) with the
distance line drawn.

Where a person's feet are hidden, their line starts at a ring (the estimated foot) joined to them by a dashed leg. In a
zone with the depth check, each close pair's line shows what the check measured (green: same depth; red, the line
dashed: one behind the other, no alert; grey: no answer yet). **Preview settings → Depth map** blends the model's
latest depth map under the drawings; while it is on, depth-zone cameras get a depth frame every 0.5 s (about 12 ms of
GPU each). A depth-confirmed incident's card shows its depth gap and a small depth image of the pair.

## Cameras

| id | camera | notes |
|---|---|---|
| cam-01 | D41 | gate; depth check recommended (occlusions) |
| cam-02 | D16 | gate |
| cam-03 | D52 | weighbridge |
| cam-06 | CH23 | yard |
| cam-10 | D15 | warehouse aisle |
| cam-11 | D18 | warehouse aisle |
| cam-14 | D05 (Camera189) | loading area; depth check recommended (occlusions) |

CH05 and D09 are in scope once they have streams; leave the depth check off on D09 (bird's-eye view). D01, D11, D20
and D21 are out of scope.

## Defaults and measured accuracy (Ticket 3, 2026-11-01)

Footage: the 8 in-scope cameras, 1 h each (13 Sep), and 12 July clips (0.68 h). Labels: 1130 candidate clips reviewed
by hand, 139 incidents (29 critical, 110 warning). Zones cover the whole view, the depth check is off, and info
alerts count.
- **Catching:** an alert catches an incident when it comes within [start − 1 s, end + 2 s] and is the labelled
  person's track or within 150 px of their feet. An incident with no alert of its own also counts when an open
  incident of the same vehicle lists its person.
- **Choosing:** best F1, picked on the first 30 min of each hour and checked on the last 30 min, then the reverse.
- **July** was never used to choose.

| setting | before | now |
|---|---|---|
| gap measured to | the vehicle's lower outline | the lowest 25 % of its mask (`vehicle_band`) |
| a vehicle is moving when | its reference point moved | its reference point and its box moved (`box_motion`) |
| `max_gap` (ph) | 0.75 | 0.75 |
| `persist_s` (s) | 1.0 | 0.25 |
| `stationary_window_s` (s) | 2.0 | 2.0 |
| `stationary_motion` (ph) | 0.5 | 0.75 |

Old → new defaults, replayed over three engine recordings of the same footage. Each cell: incidents caught, F1, false
alarms per camera-hour.

| recording | 13 Sep, first 30 min | 13 Sep, last 30 min | July (other days) |
|---|---|---|---|
| 1 | 23 → 33 of 53 · 0.49 → 0.66 · 3.75 → 3.00 | 26 → 31 of 58 · 0.51 → 0.59 · 3.75 → 3.25 | 18 → 15 of 28 · 0.68 → 0.63 · 7.39 → 4.43 |
| 2 | 20 → 30 · 0.44 → 0.62 · 3.75 → 2.75 | 26 → 29 · 0.51 → 0.56 · 3.50 → 3.50 | 15 → 13 · 0.63 → 0.58 · 5.91 → 4.43 |
| 3 | 21 → 28 · 0.50 → 0.60 · 2.25 → 2.50 | 24 → 25 · 0.49 → 0.53 · 3.50 → 2.50 | 16 → 14 · 0.65 → 0.61 · 5.91 → 4.43 |

- **The result:** better on 13 Sep, with equal or fewer false alarms. On July, 2–3 fewer of 28 incidents and fewer
  false alarms.
- **Time to alert:** median 1.3–1.9 s. Duplicates per incident: 0.
- **Ticket 4 (hidden feet from the person's own height; under half in view not measured):** 3–6 fewer false alarms per
  recording, 1 real incident lost on two recordings and 1 gained on the third; held-out F1 up or equal.
- **The box gate** keeps a parked vehicle whose mask merges with a neighbour for a few frames from reading as moving
  (C189). It costs about 5 of 139 incidents with slow-creeping trucks; `box_motion: 0` turns it off.
- **The same rule** moves by up to 6 incidents per set between recordings; recording 3 also lost about 20 % of its
  frames to a saturated shared GPU.
- **Not adopted:**
  - smoothing the gap over several frames: it helped only on the footage it was tuned on;
  - measuring to the vehicle's front or ignoring vehicles that have passed;
  - time-to-contact, a gap that widens with speed, hysteresis, and a band per mask column.
- **Limits:**
  - an incident whose person or vehicle the detector never saw had no clip to label;
  - CH05 and D09 had no scale seed;
  - the 13 Sep seed was learned from the same hours;
  - the depth check was off.

## GPU budget (measured 2026-10-28)

On the shared development host (NVIDIA RTX PRO 6000 Blackwell Max-Q, already ~81 % busy with other stacks when the
run started), 4 site clips (3200x1800, 20 fps) read from disk faster than real time, detector on every 3rd frame,
batch 4, preview on. `tools/engine_gpu_budget.py` over the run's telemetry:

| stage | busy % | avg ms | max ms | batches |
|---|---|---|---|---|
| gpu.decode (not decode work: see below) | 138.8 | 5.33 | 32.10 | 9122 |
| gpu.infer.pgie (YOLO26l-seg 1280x736) | 54.5 | 8.98 | 29.46 | 2126 |
| gpu.jpeg | 15.8 | 10.27 | 24.99 | 539 |

- **Detector:** ~9 ms per batch of 4, so ~2.2 ms per inferred frame; at 6.7 inferred frames/s a camera costs ~15 ms of
  GPU per second (~1.5 %).
- **Decode is cheap.** It runs on the GPU's video decoders (NVDEC), not its compute: measured with `nvidia-smi` at
  real-time playback, the 4 cameras at 3200x1800, 20 fps take ~2.4 % of NVDEC (0.6 % a camera), and 1280x720
  sub-streams ~0.3 %; GPU compute (SM) was ~18 % either way. The engine's `gpu.decode` stage times the decoder element
  including its waits, so it reads the same ~5.4 ms a frame at 720p as at 1800p and is not a decode cost. Keep the main
  streams: an emulated 1280x720 sub-stream (1.5 Mbps) found 2-9 % fewer people and up to 25 % fewer far ones, and
  10 fps (1 Mbps) up to 17 % and 43 % fewer, for a saving of ~2 % of a decoder.
- **Depth check** (YOLO26m-depth, onnxruntime CUDA): 12.3 ms per frame, only on frames with a close pair, at most 2/s
  per camera.
- **Streaming thread (CPU):** the probe, with the proximity module, averages 1.45 ms a frame (~20 % of the thread for 7
  cameras at 20 fps).

Numbers on a shared GPU: re-measure on the deployment host before choosing how many cameras one host carries.
