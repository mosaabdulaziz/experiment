# DeepStream sidecar — before/after benchmark results

Measured on this machine (RTX 5050 Laptop GPU, DeepStream 9.0, single 1280x720
file source `camera_5_2023-06-08-12-29-08.avi`, ~281 s of footage per run).

- **before** = commit `32aaf69` (pre-enhancement sidecar) + FP32 pgie
- **after**  = HEAD (`015b069`, the enhancements) + INT8 pgie
- **after_fp32** = HEAD code + FP32 pgie (isolates INT8's contribution)

Every run replays the same video from the start under identical `sidecar.yaml`
(sources, fps, thresholds, local segment storage). The three hot-path functions
(`probe`, splitmuxsink rotation callback, `FaceRecognizer._process_frame`) are
wrapped by `benchmarks/instrument.py` with the **same** timing instrumentation in
both code versions. GPU sampled every 3 s for 90 s after first fps.

## Summary

| metric | before (old code, FP32) | after (new code, INT8) | Δ |
|---|---|---|---|
| boot to first fps | **28.5 s** | **6.8 s** | **−21.7 s** (−76%) |
| face engine at boot | rebuilt from ONNX (~22 s) | deserialized from cache (~1 s) | rebuild → cache |
| pipeline fps | 30.0 | 30.0 | equal (source-capped) |
| probe busy % (streaming thread) | 25.1 % | 21.7 % | −14 % relative |
| probe avg per buffer | 8.36 ms | 7.21 ms | −1.2 ms |
| segment rotation callback | 1.99 ms | **0.06 ms** | **~33× faster** |
| recognizer per frame (1 face) | 30.5 ms | 7.7 ms | **~4× faster** |
| recognizer per crop (micro-bench) | 17.3 ms | 5.7 ms | **3.0× faster** |
| GPU util | 31.5 % | 21.5 % | −10 pts |
| GPU memory | 2011 MiB | 1535 MiB | −476 MiB |
| pgie engine size | 39.6 MB (fp32) | 16.6 MB (int8) | −58 % |

## Per change

1. **Face SGIE engine caching** — the before probe pointed at
   `data/deepstream-engines/` (a hand-placed file); with that file absent it
   rebuilt the face engine from ONNX every boot: log shows `Trying to create
   engine` → `Load new model` ~22 s later. The after code probes nvinfer's real
   serialize path (`yolov8n-face.onnx_b16_gpu0_fp32.engine`) and deserializes in
   ~1 s. Boot: 28.5 s → 6.8 s.

2. **Async segment uploader** — `format-location-full` (recording streaming
   thread) did the local copy + Redis index + delete inline at ~2 ms per
   rotation; it now just enqueues at 0.06 ms (**~33×**). With MinIO/object-store
   backend the inline cost — and therefore the win — is much larger; here
   `storage_backend: local` keeps the absolute numbers small.

3. **Frame publisher worker** — streaming-thread load (probe busy) drops
   25.1 % → 21.7 % and per-buffer time 8.36 → 7.21 ms. At one 720p camera with a
   fast local Redis the remaining probe cost is dominated by the shared CPU
   surface copy + zone logic, so the encode/Redis offload shows a modest but
   real 14 % reduction; the benefit scales with camera count, resolution and
   remote object stores (the code also reports `frame publisher saturated`
   warnings if it ever backs up — none observed).

4. **Recognition-only face path** — per-crop: `app.get()` re-detection
   17.3 ms → `_embed_crop` 5.7 ms (**3.0×**, measured on identical crops from
   the test video with the same GPU). Whole-frame recognition drops 30.5 → 7.7 ms
   (~4×). The before path also *fails* on some SGIE crops (`app.get` returns 0
   faces and the face is skipped); the after path never re-detects, so it cannot
   miss an already-located crop. (Embedding agreement between the two paths was
   checked in the earlier session: same-face sim 0.69 vs background 0.05.)

5. **INT8 calibration** — aligned-window comparison of the identical code on
   FP32 vs INT8: GPU util **36.0 % → 21.5 %** (−40 %), GPU memory 1654 → 1535 MiB,
   engine 39.6 → 16.6 MB, pipeline fps and detection confidences unchanged
   (person 0.74–0.78, same band as FP32). Both engines deserialize from cache.

## Caveats

- Single camera, 1280x720, 30 fps source cap — inference is not the bottleneck,
  so fps is a wash either way; the wins show in GPU headroom and thread time.
- `before` vs `after` GPU util windows are not exactly aligned (the before boot
  is 22 s longer), so treat the INT8 isolation row (after_fp32 vs after) as the
  clean GPU comparison.
- Local segment storage keeps the rotation-callback numbers small.


---

## Stream-capacity sweep (max streams on this laptop)

Same test video × N sources, full production config (INT8 pgie + face sgie +
zones + alert clips + recognition), GPU-util < 95% as the pass criterion.
Each cell = fresh sidecar restart, 30 s sampling window. Measured on this
laptop: **RTX 5050 Laptop (8 GB VRAM), 16 cores, 14 GB RAM**.

### Correction: the first sweep measured the wrong axis

The first version of this sweep set `stream_fps` per cell and expected the
pipeline load to change. It did not: `stream_fps` (sidecar.py:80) only
throttles the **analysis/JPEG publishing cadence** — the GStreamer chain
(decode → nvstreammux → nvinfer → face sgie) always ran at the source video's
native 30 fps. That is why the original numbers were flat (~99–100% GPU at 8
streams for every fps setting): the fps axis was not throttling inference at
all.

The fixed harness transcodes the test video once per target rate into
`benchmarks/fixtures/fps{25,10,5}.mp4` (H.264, `ffmpeg -vf fps=N`) and points
all N sources at the matching fixture, so **decode AND inference genuinely run
at the target frame rate** — the same load as real cameras configured to those
rates.

### The real fix: `analysis_fps` (videorate cap on the analysis branch)

`stream_fps` was only one of two bottlenecks. Even with the fps fixtures, the
**analysis branch** (nvinfer → nvtracker → nvvideoconvert → probe) still ran on
every decoded frame at the fixture's native rate — `process_fps`/`infer_interval`
only throttled nvinfer, so the tracker/convert/probe burned GPU on full-rate
frames and the GPU scaled with total frames/s through the pipeline (that is the
legacy matrix below: 100 % GPU at 10-12 streams @ 25 fps).

Commit `b02f90c` inserts a **videorate capped to `analysis_fps` (default 5)**
between each source's tee and the mux, so the entire analysis branch only sees
5 fps/source regardless of source rate. The record branch (off the same tee) is
untouched, so recordings stay full-rate. All numbers below are measured with
`analysis_fps: 5`.

### Capacity matrix (median GPU util %, median VRAM MiB, median per-stream analysis fps)

Per-stream fps is now the **analysis** rate (capped at 5 fps/source by
videorate); decode runs at the fixture's native rate underneath. Grid swept past
12 streams until saturation. **Important — the cap is nominal, not achieved:**
the videorate→mux path delivers only ~0.84× the set cap (25 fps source / 6 ≈
4.17), so the rows below read ~4.2 with `analysis_fps: 5` even though the cap
says 5. That flat 4.2 at just 8 streams (22 % GPU) is a batch-phase alignment
drop inside nvstreammux (40 ms batched-push-timeout vs the videorate cadence),**
not a GPU limit**. To get a *solid* analysis rate you set the cap ~20 % higher
(see the target-based sweep below).

| streams | fps=25 | | | fps=10 | | | fps=5 | | |
|---|---|---|---|---|---|---|---|---|---|
| | GPU% | VRAM | fps | GPU% | VRAM | fps | GPU% | VRAM | fps |
| 1 | 12 | 1398 | 5.0 | 3 | 1363 | 5.0 | 2 | 1367 | 5.0 |
| 2 | 17 | 1484 | 5.0 | 6 | 1454 | 5.0 | 3 | 1458 | 5.0 |
| 4 | 18 | 1664 | 5.0 | 11 | 1632 | 5.0 | 5 | 1632 | 5.0 |
| 6 | 17 | 1817 | 5.0 | 13 | 1817 | 4.2 | 8 | 1843 | 5.0 |
| 8 | 25 | 2008 | 4.2 | 12 | 1981 | 4.2 | 3 | 1981 | 4.2 |
| 10 | 13 | 2296 | 4.2 | 26 | 2330 | 5.0 | 11 | 2300 | 4.2 |
| 12 | 16 | 2599 | 5.0 | 16 | 2470 | 4.2 | 23 | 2493 | 4.2 |
| 16 | 45 | 2842 | 4.2 | — | | | — | | |
| 20 | 24 | 3308 | 4.1 | — | | | — | | |
| 24 | 39 | 3635 | 4.1 | — | | | — | | |
| 28 | 68 | 4296 | 4.2 | — | | | — | | |
| 32 | **96** | 4609 | 4.1 | — | | | — | | |
| 36 | 100 | 5217 | 4.0 | 95 | 5227 | 4.0 | — | | |
| 40 | — | | | 99 | 5634 | 4.0 | 95 | 5620 | 4.1 |
| 44 | — | | | — | | | 98 | 6002 | 4.0 |
| 48 | — | | | — | | | 100 | 6514 | 3.9 |

The nominal-cap matrix above answers "how many streams fit before the GPU
saturates" (decode + VRAM bound) — but because the cap under-delivers (~0.84×),
its per-stream fps column is **not** the analysis rate you would actually get.
The sweep below answers the question that matters for analysis quality: for a
**guaranteed** analysis rate, how many streams?

### Target-based capacity: solid stream counts for 25 / 15 / 10 / 5 fps

Each cell sets `analysis_fps` to the **compensated cap** (`cap = target / 0.84`
≈ 1.20× target) and reads the **achieved** per-stream rate off the sidecar's
`[cam-N] X fps` log (the probe-arrival rate, post-videorate — the number the
user actually sees). 25 fps sources, fresh restart per cell, 30 s window.

| Analysis target | Cap to set | Safety factor | **Max solid streams** | Achieved med | Min | GPU @ max |
|---|---|---|---|---|---|---|
| **5 fps** | 6 | 1.20× | **24** | 5.0 | 4.9 | 65 % |
| **10 fps** | 12 | 1.20× | **16** | 10.0 | 9.8 | 81 % |
| **15 fps** | 18 | 1.20× | **8** | 15.0 | 15.0 | 68 % |
| **25 fps** (full rate) | 25 | 1.00× | **6** | 25.0 | 25.0 | 99 % |

Supporting cells (achieved med fps @ median GPU %):

| cap → streams ↓ | 6 | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 36 |
|---|---|---|---|---|---|---|---|---|---|
| 6 (tgt 5) | — | 5.0 @ 33 % | — | 6.0 @ 65 % | — | 5.0 @ 65 % | 4.8 @ 97 % | 4.9 @ 97 % | 4.7 @ 100 % |
| 12 (tgt 10) | — | 10.0 @ 31 % | 10.0 @ 73 % | 10.0 @ 81 % | 9.9 @ 100 % | — | — | — | — |
| 18 (tgt 15) | — | 15.0 @ 68 % | 14.8 @ 100 % | 14.8 @ 100 % | — | — | — | — | — |
| 25 (tgt 25) | 25.0 @ 99 % | 22.8 @ 100 % | — | — | — | — | — | — | — |

Reads:
- **Solid 5 fps → 24 streams** (cap 6, GPU 65 %). At 28+ the GPU hits ~97 %
  and achieved dips to 4.7-4.9. This is the corrected answer to "how many
  streams at 5 fps, not below": the earlier "32 streams @ 5 fps" claim was
  actually ~4.1 fps (cap 5 under-delivers).
- **Solid 10 fps → 16 streams** (cap 12; dips to 9.9 @ 20, GPU 100 %).
- **Solid 15 fps → 8 streams** (cap 18; 14.8 @ 12, GPU 100 %).
- **Solid 25 fps → 6 streams** — full-rate analysis; the legacy decode+analysis
  ceiling (99 % GPU @ 6, degrades to 22.8 @ 8).

> **Live wall confirmation (24 streams, cap 6):** 24/24 tiles advancing, GPU
> 48 %, achieved median **5.0** / p75 5.0 (the uploads are lighter than the 720p
> fixture). Occasional sub-5 readings (min 3.7) come from the short 9-46 s
> looping clips stalling at each loop boundary — a content artifact, not
> pipeline saturation; the 284 s fixture at the same config holds 5.0 (min 4.9).

### Verdict

- **At a guaranteed (solid) analysis rate:** 5 fps → **24 streams**, 10 fps →
  **16**, 15 fps → **8**, full-rate 25 fps → **6** (cap = target × ~1.20 to
  compensate the ~0.84 nominal-cap factor).
- **Raw decode + VRAM wall (nominal cap, what fits before saturation):**
  ≈28-32 streams @ 25 fps source, ~36 @ 10 fps, ~40 @ 5 fps — GPU 96-100 % and
  per-source fps slips to 3.9-4.0 at the wall. **Those counts run analysis at
  ~4 fps, not 5** (the nominal-cap numbers above).
- **VRAM:** ~4.6 GiB @ 32 streams; co-limiting only past ~44 streams.
- **CPU:** ≤ ~29 % median across all cells — never the limit.

> **Legacy baseline (no `analysis_fps`, same GPU, archived in
> `results/legacy-no-analysis-fps/`):** 8×25 = 98 % GPU, 10×25 = 100 %, 12×25 =
> 100 % — analysis ran at full source rate. So the videorate cap roughly
> **quadrupled capacity at full source rate**: ~6-8 streams → ~28-32 @ 25 fps
> (decode+VRAM-bound), and a **guaranteed 5 fps** analysis rate is sustainable
> for **24 streams** (vs ~6-8 before the cap at full-rate analysis).

> **Caveat:** all N sources point at the same fixture file and start in lockstep
> (no independent phase jitter). Real cameras have independent timing; this does
> not change average GPU load materially, but is slightly optimistic for bursty
> sources. The fixture files are generated from `data/uploads/`'s test video by
> `python3 benchmarks/stream_capacity.py fixtures` (or on the first `sweep`).

**Bottom line:** with the `analysis_fps` videorate cap, this RTX 5050 serves a
**guaranteed 5 fps analysis rate for ~24 streams**, 10 fps for ~16, 15 fps for
~8, and full-rate 25 fps for ~6; at the raw decode+VRAM ceiling it fits
**~28-32 streams @ 25 fps** (analysis there is ~4 fps, not 5). Set
`analysis_fps` to **target × 1.20** (e.g. 6 for a solid 5 fps) because the
videorate→mux path delivers only ~0.84× the nominal cap. For an always-on
monitoring deployment on this class of laptop GPU, ~24 cameras at a true 5 fps
analysis rate is the practical ceiling.


---

## Laptop re-measured — videorate mechanism (2026-08-09, commit `bc63e14`)

> The two sweeps above used the **wall-clock throttle** (exact per-stream rates
> at every N). Commit `bc63e14` restored **videorate** as the `analysis_fps`
> mechanism (chosen as the production default: preserves native PTS so previews
> stay coherent; the wall-clock throttle's PTS re-stamping made file sources
> decode ~2× realtime and publish incoherent frames). This section is the first
> capacity measurement with the videorate mechanism, on the same laptop (RTX
> 5050, 25 fps fixture, 1280×720, INT8 pgie + face sgie, recording off).

Same harness, `--analysis-fps {5,10,15}`, one row per value, fresh restart per
cell, 30 s window, stop at ≥ 90 % GPU or per-stream fps < 75 % of the cap. The
**cap is nominal, not achieved**: videorate + the mux's batched-push alignment
under-deliver ~0.84× (25 fps source → af=5 gives ~4.2, af=10 gives ~8.4, af=15
gives ~12.4) — identical to the factor documented above.

### Capacity matrix (median GPU util %, median VRAM MiB, median per-stream fps)

| n | af=5 GPU% | VRAM | perSrc | af=10 GPU% | VRAM | perSrc | af=15 GPU% | VRAM | perSrc |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 1442 | 5.0 | 14 | 1474 | 10.0 | 13 | 1444 | 15.0 |
| 2 | 10 | 1570 | 4.2 | 16 | 1564 | 10.0 | 15 | 1502 | 15.0 |
| 4 | 10 | 1765 | 4.2 | 22 | 1711 | 8.4 | 27 | 1686 | 12.5 |
| 6 | 20 | 1941 | 4.2 | 38 | 1887 | 10.0 | 27 | 1857 | 12.5 |
| 8 | 33 | 2122 | 4.2 | 39 | 2096 | 10.0 | 33 | 2029 | 12.4 |
| 10 | 18 | 2452 | 5.0 | 63 | 2380 | 10.0 | 53 | 2342 | 12.4 |
| 12 | 42 | 2627 | 5.0 | 49 | 2547 | 8.2 | **92** | 2516 | 12.4 ↓ |
| 16 | 42 | 2874 | 4.2 | 74 | 3000 | 8.4 | — | — | — |
| 24 | 50 | 3820 | 4.2 | **99** | 3851 | 8.1 ↓ | — | — | — |
| 32 | 84 | 4692 | 4.1 | — | — | — | — | — | — |

`↓` = row stopped here (GPU ≥ 90 %).

### Verdict (videorate mechanism)

| analysis_fps cap | achieved per-stream | max solid streams | GPU @ max |
|---|---|---|---|
| 5 | ~4.2 fps | **32+** (no saturation at n=32) | 84 % |
| 10 | ~8.4 fps | **16** (74 %); wall at 24 (99 %) | 74 % |
| 15 | ~12.4 fps | **10** (53 %); wall at 12 (92 %) | 53 % |

- **af=5 → 32+ streams** — the grid ended at n=32 with GPU at 84 %; the wall
  is past 32 on this laptop.
- **af=10 → 16 solid, wall at 24.** The current production config (af=10, 14
  sources) sits at ~63 % GPU in this measurement — healthy headroom.
- **af=15 → 10 solid, wall at 12.**
- Zero ERROR lines across all 26 cells. The 11 `frame publisher saturated`
  warnings appear only at the wall cells (n=16/24 @ af=10, n=12 @ af=15,
  n=32 @ af=5) — a symptom of hitting the wall, not a fault.
- These numbers are not directly comparable to the throttle-era tables above
  (different mechanism and no cap-compensation): videorate quantizes the
  achieved rate to ~0.84× the nominal cap instead of hitting the target
  exactly, but it delivers coherent previews, which is why it is production.
- Harness change in this sweep: saturation floor 0.85 → 0.75 of target, so
  videorate's PTS quantization doesn't false-saturate rows. Per-cell JSONs in
  `benchmarks/results/capacity_*_af{5,10,15}_norec.json`.

### Final check — committed config, recording OFF by default (2026-08-10)

`record_enabled` is now `false` by default (the committed production config), so
this is the ceiling users actually hit out of the box. One cell at the grid's
edge, production analysis_fps, no recording:

| n | fps | af | record | boot_s | GPU% med/p95 | VRAM MiB | CPU% | per-stream fps mean/min |
|---|---|---|---|---|---|---|---|---|
| 32 | 25 | 10 | off | 18.2 | 61 / 92 | 4439 | 86.5 | 5.28 / 5.0 |

- **All 32 sources reported fps** (`fps_line_count: 32`), spread 5.0–5.4 — the
  mux's batched-push alignment smooths every source to the same cadence at high
  N, so per-stream rate (not GPU) is what degrades; nothing drops or errors.
- **GPU is not the ceiling** (61 % median; p95 92 % bursty) — CPU (86.5 %) and
  the batched mux are now the limiting factors, which is why per-stream fps
  lands at ~5.3 rather than the ~8.4 seen at n≤16.
- **0 ERROR / GStreamer-CRITICAL lines.** 8 `frame publisher saturated`
  warnings (queue drops) at this wall — same symptom as the prior wall cells,
  not a fault.
- Verdict: the committed no-record default runs **32+ streams**; the grid's
  previous af=5 wall at 84 % GPU vs this af=10 run at 61 % median suggests the
  true wall is well past 32 even at production `analysis_fps: 10`.


---

## M1 process-sharding measurement (2026-08-10) — sharding does NOT extend this laptop's wall

M1 (R1) is implemented: `sidecar.py` slices sources by
`IRIS_SHARD_ID`/`IRIS_SHARD_COUNT` (env overrides `shard_id`/`shard_count` yaml),
shard 0 alone runs the stale-status purge against the full pre-slice source list
(the old per-process purge would have deleted other shards' status keys at boot),
and `docker-compose.yml` adds `deepstream-sidecar-1/2` behind the `shards`
profile via a top-level `x-sidecar-shard` anchor (explicit `image:` so replicas
share one built image; `IRIS_SHARD_COUNT` required via `:?` interpolation). The
harness gained `--shards N` (`n % SHARDS == 0`, per-shard engines get batch
`n // SHARDS`), reconciles the replica containers, and merges per-camera fps
across shards. Same laptop (RTX 5050, 8 GB), same cells as the single-process
baseline, fresh restart per cell.

| Cell | 1 shard | 2 shards | Δ |
|---|---|---|---|
| n=24 af=6 | 5.0 fps solid · GPU 71/73 % · VRAM ~3.8 GB · 0 sat | 5.0 fps solid · GPU **63/80 %** · VRAM ~5.0 GB · 0 sat | fps equal; GPU **−8 pts** at the same rate (half-batch engines per shard) |
| n=16 af=12 | 10.0 fps · GPU 85/100 % · 3 sat | **10.3 fps** (min 10.0) · GPU 97/100 % · 6 sat | fps holds/improves; both at the GPU's ceiling |
| n=48 af=6 | (single-process GPU-bound; n=32 @ af=10 already 98 %) | **4.6 fps** (target 5) · GPU **100 %** · VRAM **7.6/8 GB** · 14 sat · sgie `output bufferpool resize failed` errors | ❌ capacity did NOT extend |

**What this proves:**

1. **The mechanism is correct.** Two sidecar processes coexist on one GPU;
   status/meta keys don't collide (purge fix holds — no shard wiped another's
   keys at boot); per-camera fps merges cleanly across shards
   (`fps_line_count == n` in every cell, e.g. 48/48 at n=48). Boot time is
   unchanged (~9 s, both shards' engines deserialize from the b24/b8 caches).
2. **This laptop's wall is the GPU, not the probe thread.** n=48 in 2 shards
   pins the GPU at 100 % with VRAM at 7.6/8 GB — a second process just adds a
   second engine + decode onto the same saturated GPU. The 2-shard config
   cannot beat the single-process wall because there is no CPU-side wall to
   parallelize (CPU stayed 18–45 % in every cell).
3. **M1's payoff is machine-dependent.** Its target is the *workstation's*
   wall — the ~600–640 fps aggregate probe serialization at 38–57 % GPU, where
   N processes each own 1/N streams and parallel probe threads multiply the
   wall (620 → ~1100 → ~1600+). On this GPU-bound laptop it is capacity-neutral
   (equal fps at equal n, slightly lower GPU util); the laptop's next lever is
   **M2** (inference headroom: yolo11n / 960×540 / engine A/B), which cuts GPU
   work directly.
4. **Sharding costs VRAM per process** (full engine + decode buffers each):
   ~1.2 GB extra at n=24 (3.8 → 5.0 GB), ~2× engines at n=48. On 8 GB the
   headroom is thin — a third shard would not fit.

Rollback is one env var: `IRIS_SHARD_COUNT=1` (or the compose `shards` profile
left off) restores today's single-process behavior; the purge fix is
backward-compatible by construction.

---

## M1 process-sharding measurement (2026-08-10) — this workstation (RTX PRO 6000 Blackwell)

M1 (R1) A/B on the probe-bound workstation — the machine where sharding was
expected to pay off. Same harness (`--shards S`), same 25 fps fixtures, videorate
mechanism (achieved ≈ 0.84× the af cap: af=25 → ~20.8, af=15 → ~12.4, af=10 →
~8.4, af=5 → ~4.2), `--no-record`, fresh restart per cell, 30 s window. Cells:
the 4-cell × 3-shard A/B from CONTEXT.md §4 + the two shard-only proof cells.
All 14 cells: **0 ERROR lines, `fps_line_count == n`** (cross-shard merge works).

### A/B grid (median per-stream fps · aggregate fps · GPU med/p95 · sat)

| Cell | 1 shard | | 2 shards | | 3 shards | |
|---|---|---|---|---|---|---|
| | fps (agg) · GPU · sat | | fps (agg) · GPU · sat | | fps (agg) · GPU · sat | |
| **n=24 af=25** (full-rate control) | 20.8 (499) · 31/41 % · 0 | | 20.8 (499) · 37/39 % · 0 | | 20.8 (499) · 39/40 % · 0 | |
| **n=48 af=15** (control, < wall) | 11.5 (552) · 50/53 % · 0 | | 12.5 (600) · 55/69 % · 0 | | 12.4 (595) · 60/71 % · 0 | |
| **n=96 af=10** (discriminator) | 5.8 (557) · 57/68 % · 0 | | **8.3 (797)** · 79/85 % · 16 | | **8.4 (806)** · 85/89 % · 21 | |
| **n=192 af=5** (scale-proof) | 3.0 (576) · 40/83 % · 4 | | **4.2 (806)** · 80/90 % · 14 | | **4.2 (806)** · 91/99 % · 21 | |
| **n=256 af=5 s2** (proof) | — | | 3.4 (870) · 94/98 % · 14 | | — | |
| **n=384 af=5 s3** (proof) | — | | — | | 2.8 (1075) · 100/100 % · 28 | |

VRAM: 4.2 → 6.0 → 7.6 GB at n=24 (s1→s2→s3) — **~1.8 GB per extra process**;
n=384 s3 peaks at 43.6 GB of 97.9 GB. Boot 12–57 s, all engines cached/quick.

### Verdict vs the pass criteria (SCALE_ARCHITECTURE.md §8.3)

1. **✅ The ~600 fps aggregate wall is broken.** The two cells that degraded
   single-process now hold at the videorate ceiling when sharded: n=96 af=10
   goes **5.8 → 8.3/8.4 fps (557 → 797/806 agg)**, n=192 af=5 goes **3.0 → 4.2
   (576 → 806 agg)**. Per-shard probe load is only 403/shard (2 shards) or
   269/shard (3 shards) — far under the old ~620 single-process wall.
2. **⚠️ The ~1100 / ~1600 headline targets were NOT reached — the GPU becomes
   the wall first.** n=256 af=5 s2: 3.4 fps → **870 agg** (vs ~1100 target, GPU
   94/98 %); n=384 af=5 s3: 2.8 fps → **1075 agg** (vs ~1600 target, GPU
   100/100 %). Both still beat single-process (~538 and ~614 agg would have
   been the wall) — +1.6× and +1.75× — but at 256–384 sources the *decode*
   load alone (25 fps × N = 6 400–9 600 decoded fps) pins the GPU, so the
   probe-parallelism win runs out of GPU headroom. Reaching ~1100/~1600 needs
   the M2 lever (yolo11n / 960×540 / engine A-B), which cuts GPU work
   directly.
3. **✅ No regression.** n=24 af=25 holds 20.8 fps at every shard count; n=48
   af=15 actually *improves* sharded (11.5 → 12.5/12.4, the videorate ceiling).
4. **✅ GPU p95 < 90 % in the solid region** (all cells up to n=96; n=192 s2
   sits at the 90 edge). Only the past-the-wall proof cells exceed it.
5. **⚠️ Publisher saturation persists at high total-n even sharded** (0 at
   n≤48, 14–28/window at n≥96). Each shard's publisher sees 1/S streams, but
   the JPEG-encode + Redis queue still backs up at ≥96 sources total; analysis
   fps holds regardless (it is a queue-drop warning, not a frame loss in the
   pipeline).
6. **✅ Zero `gst_poll_write_control` / double-free / ERROR lines** in any
   cell — per-process FD headroom (nofile 65536) holds to n=384.
7. **✅ Cross-shard state holds** — `fps_line_count == n` in every cell
   (24/48/96/192/256/384), no status-key collisions; the shard-0-only purge
   fix works.

### Bottom line

On the workstation, M1 process sharding **breaks the ~600–640 fps probe wall**
that capped single-process capacity: cells that degraded to ~557–576 agg now
hold the videorate ceiling at **~797–806 agg with 2 shards** (3 shards adds GPU
headroom but no aggregate — the shared GPU/mux is the new limit). The
higher-n proofs land at **870 agg (n=256, 2 shards)** and **1075 agg (n=384, 3
shards)**, GPU-bound rather than probe-bound — a real ~1.4–1.75× capacity gain
over single-process, though short of the ~1100/~1600 projections because
decode of 256–384 sources at 25 fps saturates the GPU before the probe would.
Next lever for the remaining headroom: **M2** (inference-compute reduction).
Per-cell JSONs: `benchmarks/results/capacity_*_af{25,15,10,5}_norec[_s{2,3}].json`.

### af=6 follow-up — max analysis rate the 192-stream wall can hold (2026-08-10)

After the M1 A/B, the **live 192-stream × 2-shard wall** (now with viewer-gated
preview publishing, `viewer_gated_preview: true`) was raised from
`analysis_fps: 5` to **6** — the documented cap-compensation rule
(cap = target / 0.84 ≈ target × 1.20) that turns the ~0.84×-quantized 4.2 into a
measured ~5.0. Same config as the n=192 af=5 s2 cell (25 fps fixtures,
`--no-record`, 2 shards), 30 s sampling window:

| Cell | per-stream fps (med / min / max) | aggregate | GPU med / p95 | sat | errors |
|---|---|---|---|---|---|
| n=192 af=5 s2 (baseline) | 4.2 / 4.1 | 806 | 80 / 90 % | 14 | 0 |
| **n=192 af=6 s2** | **4.8 / 4.3 / 5.6** | **~922** | **70–89 / 95–96 %** | **0** | **0** |

- **192/192 sources reported** (`fps_line_count: 192`), boot 32 s, 0 ERROR lines.
- **0 `frame publisher saturated`** — the viewer-gated encoder eliminated the
  A/B's 14/window at the same total-n (only watched streams are encoded now).
- GPU median read 70–89 % across sampling windows (batched-mux burstiness);
  p95 ~95–96 % in both — **the GPU is now the binding wall, not the cap**.
- The cap-compensation rule holds (af=6 → ~4.8), but the *solid* 5.0 it targets
  is **not reachable at 192 streams**: the A/B grid already showed the collapse
  pattern past ~94 % GPU (n=256 af=5 s2: cap 5 → only 3.4 achieved). af=7/8
  would pin the GPU at ~100 % and drop achieved fps back below the cap.
- **≥5.0 fps at 192 streams needs the M2 lever** (cheaper inference: yolo11n /
  960×540 / engine A-B), which cuts GPU work per frame instead of racing the cap
  into the wall.
- Data: `benchmarks/results/capacity_192s_25fps_af6_norec_s2.json` + a
  `capacity_summary.jsonl` row. (The live wall currently runs this config; the
  committed `sidecar.yaml` stays at the 14-source production af=10 default.)

### R7 A/B — custom YOLO builder vs stock ONNX parser (2026-08-10, this workstation)

M2's first sub-lever tested the SCALE_ARCHITECTURE.md §2.1 hypothesis that the
custom `NvDsInferYoloCudaEngineGet` engine is 2–3× off TRT's theoretical.
**Disproven on this GPU.** Same `yolo11s.onnx` + same `calib/calib.table` (INT8),
engine B built by the stock TRT ONNX parser (`trtexec --onnx`), both benchmarked
with `trtexec --loadEngine --shapes=input:<B>x3x640x640` on a **quiet GPU**
(192-stream wall paused), 5 s window after 1 s warmup:

| Batch | Engine A (custom builder) | Engine B (ONNX parser) | Winner |
|---|---|---|---|
| **96** (live per-shard) | **27.6 qps · 35.97 ms** GPU compute | 23.9 qps · 41.49 ms | **A +15 %** |
| **160** (max_pgie_batch) | **15.7 qps · 62.78 ms** | 13.4 qps · 73.83 ms | **A +18 %** |

- Both engines expose the identical I/O (`input 1×3×640×640` → `output 1×8400×6`,
  decoded YOLO11), so the pipeline swap would be mechanical — but it would be a
  **regression**, so the custom builder stays.
- Engine B build sanity: b1 INT8 engine built cleanly from the ONNX + existing
  calib cache (`&&&& PASSED`, 63 MB); the b96 engine came in leaner (13.8 MB vs
  the custom builder's engine) — smaller file, slower inference.
- **Implication:** the inference lever is not the builder but the **work per
  frame** — R5 (960×540 mux, −44 % pixels) then R6 (yolo11n, ~⅓ FLOPs). R7 is
  closed as a negative result.

### R8 experiment — `decoder_drop_interval` (2026-08-10, this workstation)

Architecture-only lever (no model/resolution change): throttle decode to ~6 fps
via `decoder_drop_interval` instead of full-rate 25 fps, since the analysis
branch only consumes ~5 fps. Measured on the live 192-stream × 2-shard wall
(af=6), 30 s window, values anchored to boot:

| Config | per-stream fps (min/p50/max) | GPU med/p95 | decoder med (burst) | errors |
|---|---|---|---|---|
| af=6 (baseline) | 4.30 / 4.80 / 5.60 | 70–89 / 95–96 % | 23 % | 0 |
| af=6 + drop=4 | 4.30 / 4.80 / 5.90 | 80 / 96 % | 21 % | 0 |
| af=6 + drop=25 (extreme) | 4.60 / 4.90 / 6.90 | 90 / 97 % | 8 % (**bursts to 84 %**) | 0 |

**Verdict: no-op for file sources — ❌ no gain, reverted.** Two hard facts:

1. **`drop-frame-interval` is not honored by the internal decoder for `file://`
   looped sources.** At `drop=25` (should decode ~1 fps) analysis still ran at
   ~4.9 fps — impossible if decode were truly throttled — and the decoder still
   burst to 74–84 % during the window. The property appears to apply to live
   RTSP paths only; the code path (`srcbin.set_property("drop-frame-interval")`,
   sidecar.py:1642) fires but the file-loop decoder ignores it.
2. **Even conceptually, decode was not the wall.** The engine alone sustains
   ~2 650 fps aggregate inference (trtexec b96 = 27.6 qps × 96) while the
   pipeline delivers only ~940 agg at GPU p95 96–97 % — the GPU is busy with the
   whole pipeline (bursty decode + tracker + convert + 2-shard inference), and
   per-stream rate is capped by the mux batched-push / videorate quantization
   interaction (~0.84×), not by decode.

**Implication for the architecture survey:** the mux→videorate quantization is
now the per-stream cap mechanism; remaining structural levers are the
**batched-push-timeout** interaction, viewer-gated surface copy, tracker state
(R12), and multi-node (M5). R8 is closed as a negative result for file sources.

### Overhead audit — what is processed that should not be (2026-08-10, live 192-wall)

Static + live inspection of the running pipeline (`grep` config/state, container
logs, Redis keyspace, trtexec on the face engine) for work whose output is
**discarded** — pure overhead that can be switched off without changing any
result the deployment actually consumes:

| # | Overhead | Evidence (live) | Cost | Fix |
|---|---|---|---|---|
| 1 | **Face SGIE (`yolov8n-face`, FP32) runs a second full detection pass on every analyzed frame of all 192 streams — with an EMPTY gallery** | `sgie_face_enabled: true`, sgie config `interval=0` (no skip) + `operate-on-gie-id=1` (processes pgie output per frame); recognizer log: `Loaded 0 faces from gallery` every 30 s reload; Redis `face_gallery:*` = 0, `face_emb:*` = 0 | **0.483 ms/frame (trtexec b16 fp32, wall contention) — MORE than the pgie's 0.375 ms/frame (b96 int8)**. At ~960 analyzed fps that is a double-digit % of the GPU, spent detecting faces whose result is only ever published as `"unknown", confidence 0.0` | `sgie_face_enabled: false` (and `face_recognition_enabled: false`) until a gallery is registered |
| 2 | **nvdsanalytics element in the pipeline with ZERO ROIs** | `analytics_enabled: true`; `data/nvdsanalytics_config.txt` contains only `[property] enable=1 …` — no `[group]`/ROI lines; Redis `zone_alert:*` = 0, `overcrowd_alert:*` = 0 | Per-frame element + probe `roiStatus` iteration that can never fire | `analytics_enabled: false` until zones are configured (the app auto-writes this empty config at boot) |
| 3 | **Face-recognition thread publishes `unknown` results for every detected face** (empty-gallery path) | `_process_frame` short-circuits to `_publish_results([… "unknown" …])` per frame | Redis xadds of no-value rows (at face-detection rate, bounded by #1) | Falls away with #1; alternatively skip publishing when gallery empty |
| 4 | Viewer-gated surface copy (known) | Only **16/192** streams watched; probe still `_map_frame_bgr` copies every analyzed frame of all 192 | ~3.7 MB/frame × ~960 fps CPU memcpy (CPU at 7 %, hygiene not wall) | Gate the copy on `is_watched` too (R4-adjacent) |

**Headline:** #1 — the pipeline is running *two* detection models on the same
frames, and the second (FP32 face, per-frame cost **above** the INT8 pgie) feeds
an empty gallery. For analysis-only deployments this is the single largest
removable GPU overhead identified so far — worth an A/B (`sgie_face_enabled:
false` vs `true`, same 192-wall) to quantify.

---

## Stream-capacity sweep — this workstation (RTX PRO 6000 Blackwell Max-Q)

Ran on **2026-08-05** on the production workstation, full production config
(INT8 pgie + face sgie + zones + alert clips), same harness as the laptop
sweep. Source: **Wisenet 1280×720@30 CCTV** (different footage than the laptop's
clip), transcoded once per rate into `benchmarks/fixtures/fps{25,10,5}.mp4` so
decode AND inference genuinely run at the target rate. Every cell = fresh
sidecar restart + its own batch-N INT8 engine (built once, cached), 30 s
sampling window. `--no-record` throughout (see NVENC finding below).

### Hardware

- **GPU:** NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (96 GB VRAM)
- Engine builds: batch-N INT8 deserialize in ~1 s; each new N builds once (~1-2 min)

### Capacity matrix (median GPU util %, median VRAM MiB, median per-stream fps)

| streams | fps=25 | | | fps=10 | | | fps=5 | | |
|---|---|---|---|---|---|---|---|---|---|
| | GPU% | VRAM | fps | GPU% | VRAM | fps | GPU% | VRAM | fps |
| 1 | 6 | 1944 | 25.0 | 2 | 1944 | 10.0 | 1 | 1948 | 5.0 |
| 2 | 4 | 1845 | 25.0 | 3 | 1837 | 10.0 | 1 | 1835 | 5.0 |
| 4 | 5 | 2101 | 25.0 | 3 | 2101 | 10.0 | 2 | 2101 | 5.0 |
| 8 | 10 | 2398 | 25.0 | 6 | 2398 | 10.0 | 3 | 2398 | 5.0 |
| 12 | 16 | 2856 | 25.0 | 8 | 2856 | 10.0 | 3 | 2858 | 5.0 |
| 16 | 23 | 3173 | 25.0 | 12 | 3189 | 10.0 | 4 | 3173 | 5.0 |
| 24 | **36** | 4178 | **25.0** | 17 | 3936 | 10.0 | 0* | 3944 | 5.0 |
| 32 | 45 | 4991 | 18.5 ↓ | 22 | 4717 | 10.0 | 11 | 4715 | 5.0 |
| 48 | — | — | — | 32 | 6659 | 10.0 | 14 | 6305 | 5.0 |
| 64 | — | — | — | **44** | 8381 | **10.0** | 11 | 7919 | 5.0 |
| 96 | — | — | — | 52 | 11503 | 6.8 ↓ | **61** | 10931 | **5.0** |
| 128 | — | — | — | — | — | — | boot fail | — | (NVDEC) |

\* n=24 @ 5 fps sampled 0% GPU — a sampling artifact (the 3 s window landed
between inference bursts at near-idle load); the cell held 5.0 fps everywhere.
`↓` marks the first cell where per-stream fps dropped below 85% of target.

### Verdict (max streams at full target rate, inference on)

- **25 fps (typical camera rate): 24 streams** — GPU 36 %, VRAM 4.2 GB. At 32
  streams per-stream fps drops to ~18.5 (aggregate inference wall).
- **10 fps: 64 streams** — GPU 44 %, VRAM 8.4 GB. At 96 streams it degrades
  to ~6.8.
- **5 fps: 96 streams** — GPU 61 %, VRAM 10.9 GB, every stream exactly 5.0.
  128 sources will not boot (decode limit, below), so this is the practical
  ceiling measured.

### Why it saturates: aggregate batch throughput, not GPU% or VRAM

GPU util never approached 90 % — the rows stop because **per-stream fps drops
when the aggregated frame rate exceeds the pgie's batch-inference wall**, not
because the GPU is pegged. Every source shares one nvinfer engine that
processes N frames per inference call, so per-stream fps ≈ (engine fps) / N.
Across rows the wall sits around **~600 fps aggregate** (720p INT8 YOLO11):

- 24 × 25.0 = 600 fps; 32 × 18.5 = 592 fps
- 64 × 10.0 = 640 fps; 96 × 6.8 = 653 fps
- 96 × 5.0 = 480 fps (below the wall — hence held)

CPU stayed ≤ 6 % median in every cell; VRAM peaked at 11.5 GB of 96 GB. So the
headline numbers are not GPU-compute or memory limits — they are the single
batch-inference engine's throughput at 1280×720.

### Real-world ceilings this sweep surfaced (not inference)

1. **NVENC recording cap ≈ 15 streams.** With `record_enabled: true`, n=16
   reaches PLAYING but silently delivers zero frames (pipeline stalls past the
   driver's concurrent-session cap). This is why the sweep ran `--no-record`
   and why the earlier 22-stream live wall needed it too. Practical deployment
   limit for continuous recording, regardless of GPU headroom.
2. **NVDEC decode-surface cap: between 96 and 128 streams.** n=128 @ 5 fps
   built and deserialized its b128 engine, then died at boot with
   `nvv4l2decoder: Failed to allocate required memory` (decode-surface pool
   exhaustion), while GPU sat at 61 %.

   > **Corrected 2026-08-06:** re-measuring the n=128 failure showed it is the
   > container **file-descriptor limit** (`ulimit -n = 1024`, Docker default;
   ~9-10 FDs per source → `gst_poll_write_control` assertion + double-free
   crash), not NVDEC. With `nofile: 65536` on the sidecar, 128 → 256 streams
   boot and run. See the follow-up section below.
3. **Frame-publisher saturation** warnings appear past ~24 streams @ 25 fps /
   ~48 @ 10-5 fps (analysis/JPEG publishing backs up, per-stream fps holds).

### Headline

| target rate | max streams at full rate | what happens after |
|---|---|---|
| 25 fps | **24** | 32 → ~18.5 fps |
| 10 fps | **64** | 96 → ~6.8 fps |
| 5 fps | **96** | 128 won't boot (NVDEC) |

**Bottom line:** this workstation runs ~4× the laptop's stream count at every
rate (6→24 @ 25 fps, 12→64 @ 10 fps, 12→96 @ 5 fps). For always-on monitoring
cameras at 5 fps it sustains **96 streams**; the next ceiling past that is the
hardware decoder, not inference. Laptop numbers and the original per-cell JSONs
are archived under `benchmarks/results/laptop_rtx5050/`.

---

## Stream-capacity follow-up — 2026-08-06: the 96-stream cap was two framework walls (both fixed)

After the wall-clock throttle replaced videorate (exact per-stream rates at every
N, see commit log), the sweep on this workstation again capped at **96-108
streams** — this time for reasons that were NOT the GPU. Two framework walls
were found, fixed, and re-measured.

### Wall 1 — container FD limit (the "128 won't boot" crash)

The sidecar container ran with Docker's default `ulimit -n = 1024`. Each source
needs ~9-10 file descriptors (decoder threads, queue pipes, mux pads), so:

- n=108 × ~9.4 ≈ **1015 FDs** → works
- n=110 × ~9.5 ≈ **1045 FDs > 1024** → GStreamer's `gst_poll` cannot allocate a
  poll set → `gst_poll_write_control: assertion 'set != NULL'` + `free(): double
  free detected in tcache 2` heap corruption → crash-loop with zero frames.

That crash — previously attributed to an NVDEC decode-surface cap — is a process
FD limit. Fix: `ulimits: nofile: 65536` on the sidecar in `docker-compose.yml`
(verified in-container: 1024 → 65536). n=110 → 256 now boot and run.

### Wall 2 — TensorRT engine batch build limit

The engine batch was derived from the source count, so n=192 tried to BUILD a
b192 INT8 engine, which TensorRT refuses: `CASK5 CONV offset must not exceed
int32_max` (yolo11s INT8 on this GPU builds cleanly only through ~b160). Fix:
`max_pgie_batch` (default 160, yaml-configurable) caps the mux + pgie batch and
the engine path; the INT8 calibrator batch clamps to match. Sources beyond the
cap batch in multiple mux pushes (batched-push-timeout flushes partial batches)
— the documented DeepStream pattern for more sources than the engine batch.

### Re-measured capacity (target-based, no-record, 25 fps sources)

Same harness (fresh restart per cell, 30 s window, achieved rate read from the
sidecar fps log). The throttle holds per-stream rates within ~93-96 % of target
until the **~600-640 total-fps aggregate wall** (the batch-inference wall
measured in the original sweep) is hit.

| af | n | achieved per-stream fps | % of target | GPU med / p95 | VRAM |
|---|---|---|---|---|---|
| 25 | 24 | 23.2-23.3 | 93 % | 48 / 51 | 3.9 GB |
| 25 | 32 | 17.9 | 72 % | 45 / 48 | 4.7 GB |
| 15 | 32 | 14.3-14.6 | 96 % | 52 / 56 | 4.7 GB |
| 15 | 40 | 14.0 | 93 % | 46 / 49 | 5.4 GB |
| 15 | 48 | 12.3-12.4 | 83 % | 47 / 57 | 6.2 GB |
| 10 | 48 | 9.5-9.7 | 96 % | 51 / 54 | 6.2 GB |
| 10 | 64 | 8.7-8.9 | 88 % | 36 / 47 | 7.8 GB |
| 10 | 96 | 6.5 | 65 % | 41 / 64 | 10.9 GB |
| 5 | 110 | 3.9-4.4 | 82 % | 53 / 67 | 12.0 GB |
| 5 | 128 | 3.8-4.1 | 79 % | 55 / 76 | 13.7 GB |
| 5 | 160 | 4.0 | 80 % | 49 / 73 | 17.6 GB |
| 5 | 192 | 3.1-3.3 | 64 % | 38 / 97 | 18.0 GB |
| 5 | 256 | 2.0-2.2 | 42 % | 54 / 96 | 20.0 GB |

### Bottom line (2026-08-06, throttle + fixes)

| analysis rate | recommended max streams | note |
|---|---|---|
| 25 fps | **24** | 23.2 fps (93 %); 32 → 17.9 |
| 15 fps | **40** | 14.0 (93 %); 48 → 12.3 |
| 10 fps | **48** (edge 64) | 9.6 (96 %); 64 → 8.8 |
| 5 fps | **110** (edge 160) | 4.1 avg (82 %); runs to 256 |

- **128 streams now boot and run** at af=5 (3.8-4.1 fps, 55 % GPU) — the wall
  was the container FD limit, not NVDEC. n=256 also runs; n>160 uses the capped
  batch.
- The ~600-640 fps aggregate wall is unchanged (24×25=600, 64×10=640,
  96×10=960 exceeds it → 6.5, 160×4=640, 192×3.2=614): capacity(af) ≈ 620/af.
- `frame publisher saturated` warnings (8 per window) at every cell: the JPEG
  encode + Redis worker is the CPU-side cost at ~600 fps total while the GPU
  sits at ~50 %. Reducing per-frame CPU cost (JPEG size/quality, stream_fps
  cadence) is the lever to push the ~93 % thresholds higher.
- All engines reused from cache (b24/32/40/48/64/96); per-cell JSONs in
  `benchmarks/results/`.

## Re-running

```bash
# restore HEAD + INT8 (default state)
git restore perception/pipeline/sidecar.py perception/pipeline/face_recognizer.py \
        perception/models/DeepStream-Yolo/config_infer_primary_yolo11.txt
docker restart surveillance-camera-sys-deepstream-sidecar-1

# optional: register gallery faces so the recognizer does real work
docker exec surveillance-camera-sys-deepstream-sidecar-1 \
    python3 /workspace/iris/benchmarks/bench_gallery.py

# per-crop recognition micro-benchmark (Change 4)
docker exec surveillance-camera-sys-deepstream-sidecar-1 \
    python3 /workspace/iris/benchmarks/bench_face_crop.py

# pipeline runs (one ~3.5 min each) + comparison table
python3 benchmarks/run_bench.py before
python3 benchmarks/run_bench.py after
python3 benchmarks/run_bench.py after_fp32
python3 benchmarks/run_bench.py compare
```

## Live wall at scale (22 streams) — two real-world bugs found & fixed

Proving the 21-22 stream wall on the dashboard exposed two issues unrelated to
inference throughput. Both are fixed and verified live.

### 1. Frontend: black tiles past ~6 streams (browser connection limit)

- Every grid tile held a permanent MJPEG connection (`multipart/x-mixed-replace`).
  Browsers cap ~6 concurrent HTTP/1.1 connections per origin, so a 21-tile wall
  rendered most tiles black. Not a pipeline issue — Redis had live frames for
  every stream the whole time.
- Fix: new `GET /stream/{id}/frame` snapshot endpoint (single JPEG, `no-store`),
  and grid tiles poll it ~2.5 fps with a cache-buster instead of holding a stream.
  21 short-lived requests multiplex over the same ~6 sockets — scales to any
  tile count. The expanded single-stream view keeps smooth MJPEG.
- Also: `generate_mjpeg` did a blocking Redis `get()` inside the async generator
  every 40 ms; now offloaded to the executor so concurrent viewers can't stall
  the event loop.

### 2. Source codec: `.avi` (mpeg4) streams froze at EOS; `.mp4` (h264) looped

- Perfect correlation: all advancing cams were `.mp4`/h264; all 18 frozen cams
  were `.avi`/mpeg4. Modern NVDEC no longer decodes MPEG-4, so those sources fell
  back to software decode where `nvurisrcbin`'s `file-loop` never fires — each
  played once to EOS and froze.
- Fix: `live_streams.py transcode` converts non-h264 uploads to h264 MP4 twins
  (`<stem>_h264.mp4`); `list`/`run` prefer the twin automatically. All sources
  now decode on NVDEC and loop like real CCTV streams.

### Verified live (22 streams, process_fps=5, no-record)

- All 22 cams ADVANCING past EOS in a 10 s window (loop confirmed).
- 0 errors, 0 publisher drops; all JPEG keys fresh (idletime 0 s).
- NVDEC is now the wall's bottleneck: GPU 99-100% (all-h264 hardware decode),
  ~12-15 fps realtime per cam, process ~2.4-3 fps (interval-throttled).
- The earlier low-GPU numbers (15-34%) were measured with mpeg4 AVIs that
  software-decoded on CPU — not comparable to an all-h264 production feed.


---

## Per-stage telemetry (2026-08-12, new benchmark mode)

`perception/pipeline/telemetry.py` + `stream_capacity.py --telemetry` add a per-stage
busy%/avg breakdown to every benchmark cell: `probe` (with
`probe.surface_copy` / `.frame_objects` / `.dets`), `publish` (`.draw` /
`.redis`), `recognizer` (`.embed` / `.stabilize` / `.publish`), `upload`, and
`rotation`, plus backpressure gauges (publisher/recognition/uploader queue
depths, per-source mux-queue fill). Off by default (`telemetry_enabled:
false`); zero hot-path cost when off. GPU elements stay on the host-side
nvidia-smi + per-source fps sampling (see benchmarks/README.md).

### Current-state reference — clever sweep with per-stage telemetry (2026-08-12)

Branch `feat/recording-pipeline-optimization`, same RTX 5050 laptop, 25 fps
fixtures, `--no-record`, fresh restart per cell, 30 s window, `--telemetry` on.
Grid starts near the previous videorate walls and pushes past them (the
"clever sweep"): af=5 {24,32}, af=10 {16,20,24}, af=25 {4,6,8,10,12}.

### Capacity (median GPU %, VRAM MiB, median per-stream analysis fps)

| af | n | GPU% | VRAM | perSrc | publisher drops | verdict |
|---|---|---|---|---|---|---|
| 5 | 24 | 66 | 3922 | 4.2 | 0 | solid |
| 5 | 32 | **97** | 4554 | 4.2 | 196 | **wall** (GPU) |
| 10 | 16 | 77 | 2898 | 8.3 | 363 | solid (publisher drops start) |
| 10 | 20 | **99** | 3515 | 8.3 | 1329 | **wall** (GPU) |
| 25 | 4 | 32 | 1711 | 20.8 | 0 | solid |
| 25 | 6 | 53 | 1887 | 20.8 | 0 | solid |
| 25 | 8 | 80 | 2064 | 20.7 | 995 | edge |
| 25 | 10 | 89 | 2410 | 20.6 | — | edge |
| 25 | 12 | **97** | 2549 | 20.7 | 1921 | **wall** (GPU) |

0 ERROR lines in every cell. vs the 2026-08-09 videorate reference the GPU is
~10–15 pts busier at the same cells (af=5 n=32: 84→97 %; af=10 wall moves 24→20),
so the current-state walls are: **af=5 → 32, af=10 → 20, af=25 → 12.**

### Per-stage consumption (what the telemetry found — what to tackle)

| stage | n=24 af=5 | n=32 af=5 | n=16 af=10 | n=12 af=25 | note |
|---|---|---|---|---|---|
| `probe` (streaming thread busy%) | 25.0 | 31.6 | 29.8 | 36.6 | never the wall |
| ↳ `probe.surface_copy` | 23.9 | 30.2 | 28.0 | 35.0 | **~95 % of probe**; 2.1–2.4 ms/frame |
| ↳ `probe.dets` + `.frame_objects` | 0.1 | 0.1 | 0.1 | 0.5 | negligible |
| `publish` (Σ across workers) | 297 | 708 | 278 | 704 | the only Python backpressure |
| ↳ `publish.draw` | 288 | 702 | 265 | 690 | ~95 % of publish; avg 33→117 ms/call at the wall (nvjpegenc contention) |
| ↳ `publish.redis` | 12 | 14 | 12 | 13 | ~1 ms/call, fine |
| mux_queue gauges | 0 | 0 | 0 | 0 | analysis queues never back up — not a bottleneck |
| recognizer | — | — | — | — | no faces in fixture/gallery → zero cost here |

**Reads for the enhancement backlog:**

1. **GPU is the binding wall at every rate** (97–99 % at the walls) — decode +
   inference + NVJPEG all on one GPU; matches the laptop's prior verdicts.
2. **Streaming thread has ~2× headroom** (25–37 % busy) and is ~95 %
   `probe.surface_copy` — the known "viewer-gated surface copy" item from the
   overhead audit; gating the copy on watched streams is the single biggest
   Python-side reclaim available.
3. **The frame publisher is the only Python-side ceiling**: drops start at
   ~120–160 aggregate published fps (n=16 af=10: 363; n=8 af=25: 995) and the
   shared nvjpegenc draw cost per frame balloons at GPU saturation (33→117 ms).
   Secondary to the GPU wall on this laptop, but it is the bottleneck that will
   bind first on the 96 GB workstation at high source counts (already seen in
   the M1 A/B as `publisher saturated` at ≥96 sources).
4. **mux queues never fill** (all gauges 0) — the `analysis_queue_buffers`
   mechanism is not adding backpressure anywhere measured.
5. **Recognition is free in this benchmark** (empty gallery) — the empty-gallery
   sgie/recognizer overhead noted in the audit does not show up when no faces
   are present; re-check with `bench_gallery.py` + a face-bearing fixture.

Per-cell JSONs: `benchmarks/results/capacity_{24,32}s_25fps_af5_norec.json`,
`capacity_{16,20}s_25fps_af10_norec.json`, `capacity_{4,6,8,10,12}s_25fps_af25_norec.json`
(each carries a `telemetry` block; re-print with
`python3 benchmarks/stream_capacity.py telemetry <cell.json>`).

### GPU-element attribution — pad-probe rows (2026-08-13, current code)

Re-ran the wall cells (plus one solid cell per rate) with the `gpu.*` pad-probe
rows: `gpu.decode.<cam>` (per-frame decode, sink→src delta on the decoder),
`gpu.infer.pgie` / `gpu.infer.sgie_face` (per-batch nvinfer), `gpu.jpeg`
(per-encode nvjpegenc). busy% = share of wall-clock window the element was
active (median of 5 s flushes).

| stage | n=24 af=5 | n=32 af=5 | n=16 af=10 | n=20 af=10 | n=8 af=25 | n=12 af=25 |
|---|---|---|---|---|---|---|
| GPU util (nvidia-smi med / p95) | 44/85 | 62/98 | 84/97 | 91/100 | 65/99 | 98/99 |
| `gpu.decode.<cam>` per-cam busy% (median) | 49.2 | 47.0 | 47.2 | 46.7 | 40.6 | 39.3 |
| ↳ Σ decode busy% across cams | 1180 | 1488 | 758 | 930 | 330 | 482 |
| ↳ avg per decode call | ~38 ms | ~38 ms | ~37 ms | ~38 ms | ~34 ms | ~38 ms |
| `gpu.infer.pgie` busy% | 18.3 | 24.9 | 23.1 | 26.1 | 28.6 | 38.6 |
| `gpu.infer.sgie_face` busy% | 51.2 | 27.8 | 38.3 | 32.1 | 35.3 | 20.6 |
| `gpu.jpeg` busy% | 20.7 | 27.1 | 30.3 | 37.9 | 25.1 | 64.4 |
| `probe.surface_copy` busy% | 21.3 | 31.0 | 27.0 | 33.2 | 22.3 | 30.9 |
| `publish.draw` Σ busy% (avg ms/call) | 344 (33) | 591 (118) | 295 (62) | 623 (90) | 99 (44) | 782 (77) |

**What the GPU rows attribute the wall to:**

1. **Decode is the dominant GPU consumer — not inference.** Every decoder runs
   ~40–50 % busy regardless of rate or count (the NVDEC engines are shared and
   stay contended), so decode busy% *scales linearly with N*: Σ 330 % at n=8 →
   1488 % at n=32. nvinfer (pgie) is only 18–39 % busy even at full-rate
   af=25 — inference is *not* the wall. This is the first measurement that
   separates the two; the 2026-08-09/08-12 runs only saw aggregate GPU util.
2. **NVJPEG grows with publish load.** `gpu.jpeg` tracks `publish.draw` (it *is*
   the encode half of it): 21 % at n=24 af=5 → 64 % at n=12 af=25. Its avg
   stays ~3–7 ms/call; the multi-hundred-ms `publish.draw` wall cost is queue
   wait (nvjpegenc contention), not the encode itself.
3. **The single-decode latency is dominated by contention**: avg ~35–38 ms per
   decode call with multi-second maxes (GPU saturation spikes, e.g. 3.5 s at
   n=32 af=5). Per-frame decode time itself is ~1–2 ms on this GPU — the rest
   is queueing behind the other N decoders.
4. **Per-cam decode busy% is flat across all cells** (~40–50 %) because decode
   runs at the source rate (25 fps) regardless of the analysis-rate cap — so
   lowering `analysis_fps` buys no decode headroom. The levers that would move
   the decode wall are *decode-side*: lower decode resolution/fps on
   non-analyzed streams, or fewer sources per GPU.

Per-cell JSONs now carry the `gpu.*` rows: re-print any cell with
`python3 benchmarks/stream_capacity.py telemetry <cell.json>` (e.g. the
n=32 af=5 cell shows all 32 `gpu.decode.cam-*` rows individually).

---

## Inference-only preview mode — verification sweep (2026-08-13)

`pipeline_preview: false` (inference-only / raw preview) turns the sidecar into
pure inference: the probe skips the surface copy, the publisher skips box
drawing + JPEG encode, and the frontend plays the source's raw RTSP
(`preview_uri` in `iris:status:<id>`) with boxes overlaid from `iris:meta`.
This sweep re-ran the af=5 row in that mode (`--telemetry --no-record`, same
laptop, same 25 fps fixtures, engine rebuilt once per boot) and pushed past the
old wall to find the new one.

### Capacity (median GPU %, VRAM MiB, median per-stream analysis fps)

| af | n | GPU% | VRAM | perSrc | publisher drops | verdict |
|---|---|---|---|---|---|---|
| 5 | 24 | 18 | 3722 | 4.2 | 0 | solid |
| 5 | 32 | 39 | 4436 | 4.0 | 0 | solid (was the **wall** at 97 %) |
| 5 | 40 | 42 | 5250 | 4.0 | 0 | solid |
| 5 | 48 | 61 | 6084 | 4.0 | 0 | solid (was 96 % in pipeline mode) |
| 5 | 56 | 80 | 7200 | 4.0 | 0 | solid |
| 5 | 64 | 76 | — | 4.0 | 0 | solid (p95 100 %, spiky) |
| 5 | 72 | **100** | — | 3.1 | 0 | **wall** (GPU + per-src fps floor) |

0 ERROR lines, 0 publisher drops in every cell — the Python-side ceiling is
gone entirely (`publisher_drops=0`, `publisher_pending=0` even at the wall).

### The expected bottlenecks are gone (per-stage telemetry)

| stage | pipeline mode n=32 af=5 | inference-only n=32 af=5 | n=64 af=5 |
|---|---|---|---|
| `probe` (streaming thread busy%) | 31.6 | 0.4 | ~0.5 |
| `probe.surface_copy` | 30.2 | **0** | **0** |
| `publish` Σ busy% | 708 | 16.7 | 32.4 |
| `publish.draw` | 702 | **absent** | **absent** |
| `publish.redis` | 14 | 16.7 | 32.4 |
| `gpu.jpeg` (NVJPEG) | 27.1 | **absent** | **absent** |
| publisher saturation warnings | 2–196 | **0** | **0** |

### Reads

1. **The wall at af=5 doubles: n=32 → n=64–72.** GPU at the old wall point
   (n=32) dropped 97 % → 39 %. The remaining wall is pure decode + inference:
   Σ `gpu.decode` busy scales with N (as before) and pgie grows with N.
2. **Decode remains the binding resource** — unchanged by preview mode; the
   structural lever (lower decode resolution) still applies on top of this.
3. **The sgie toggle (corrected mechanism).** The original auto-toggle used
   nvinfer `interval`, which only gates the full-frame path — a no-op for a
   secondary GIE. The corrected toggle sets `infer-on-class-ids` to a
   never-matching class (`"999"`): `should_infer_object()` then skips every
   person, so the sgie queues zero inferences. Effect on the bench: GPU util
   at n=48 dropped 96 % → 61 %, and `probe.surface_copy` went to zero (no face
   detections → no recognition crops). The `gpu.infer.sgie_face` telemetry row
   *does not* drop — it measures the element's sink→src buffer transit, which
   is dominated by fixed per-batch framework overhead, not the face inference
   (the inference itself is what the GPU util drop shows).

Archived pipeline-mode af=5 cells: `benchmarks/results/pre_rawmode/`. New
inference-only cells: `capacity_{24,32,40,48,56,64,72}s_25fps_af5_norec.json`
(each with a `telemetry` block; re-print with
`python3 benchmarks/stream_capacity.py telemetry <cell.json>`).

---

## Genuine 5 fps A/B — pipeline preview vs raw preview (2026-08-16)

Every af=5 row before this used **25 fps fixtures capped to af=5**, which has
two effects: (1) the videorate PTS quantization measures **~4.0–4.2 fps**, not
5.0 (the "4 not 5" the operator flagged), and (2) decode runs at a heavy 25 fps
per source even though inference is capped. This sweep re-measured both preview
modes with the **genuine 5 fps fixture** (`benchmarks/fixtures/fps5.mp4`, `-vf
fps=5`, verified `avg_frame_rate=5/1`): decode AND inference truly run at 5 fps,
and the 1:1 source/analysis ratio means the probe reports **exactly 5.00 fps**
with no PTS quantization. Same laptop (RTX 5050 Laptop), `--no-record
--telemetry`, `mux_sync_inputs: 1` (the production flicker-fix config) in every
cell.

### Capacity at true 5 fps (median GPU %, median CPU %, median per-stream fps)

| preview mode | n | GPU% | CPU% | perSrc fps | drops | verdict |
|---|---|---|---|---|---|---|
| pipeline | 48 | 50 | 57 | 5.00 | 4 | solid |
| pipeline | 52 | 80 | 54 | 5.00 | 2 | solid |
| pipeline | 54 | 68 | 49 | 5.00 | 4 | **max at 5.0** (publish.draw ~100 %/thread) |
| pipeline | 56 | 89 | 70 | 3.60 | 7 | **wall** (GPU + CPU publisher) |
| pipeline | 64 | 94 | 72 | 2.80 | — | saturated |
| pipeline | 72 | 28* | 85 | 2.40 | 0 | saturated (CPU-starved) |
| raw | 56 | 58 | 17 | 5.00 | 0 | solid (pipeline mode was 3.6 @ 89 %) |
| raw | 64 | 66 | 16 | 5.00 | 0 | **max at 5.0** |
| raw | 66 | 100 | 17 | 4.60 | 0 | **wall** (pure GPU) |
| raw | 68 | 100 | 19 | 4.40 | 0 | saturated |
| raw | 72 | 100 | 26 | 4.00 | 0 | saturated |

*GPU% at n=72 pipeline is misleading: CPU publisher saturation starves the
pipeline, so the GPU idles while CPU pins at 85 %.

### The walls, in one line

- **Pipeline preview: ~54 streams** at a true 5.00 fps.
- **Raw preview: ~64 streams** at a true 5.00 fps — **+10 streams (+18 %)**.

### Why each mode hits its wall

| | pipeline preview (n=54) | raw preview (n=64) |
|---|---|---|
| binding resource | **CPU** — `publish.draw` ~100 %/thread × 54 threads on 16 cores (JPEG box-drawing) | **GPU** — decode + pgie at ~100 % util |
| CPU% | 49–72 | 16–26 |
| `publish.draw` busy | ~5 400 % (Σ, 54 threads) | **absent** (no JPEG path) |
| `gpu.jpeg` (NVJPEG) | ~69 % busy | **absent** |
| `probe.surface_copy` | present | **absent** |
| publisher drops | 2–7 | **0** |
| GPU util at the other mode's wall | 89 % @ n=56 | 66 % @ n=64 |

Raw preview removes the Python/CPU-side ceiling entirely (drops=0 even at the
wall): the remaining wall is pure decode + inference, i.e. the same structural
lever as always (lower decode resolution / more GPU). Per-stage telemetry
confirms the expected stages vanish: `publish.draw`, `gpu.jpeg` and
`probe.surface_copy` are absent from every raw-mode cell (re-print with
`python3 benchmarks/stream_capacity.py telemetry capacity_64s_5fps_af5_norec.json`).

### sync-inputs isolation (does the flicker fix cost capacity?)

Reran n=64 with `mux_sync_inputs: 0` (the pre-fix behavior): 3.6 fps vs 2.8 fps
with sync=1 — sync-inputs=1 costs some per-stream throughput at high N because
the mux waits for every source before closing a batch, but it is NOT the main
limiter at n=64 (the CPU publisher is, in pipeline mode). And sync=0 is the
mode that produced the cross-stream flicker bug, so it is not shippable anyway:
the production config (sync=1) is what both columns above measure.

New cells: `capacity_{48,52,54,56,64,72}s_5fps_af5_norec.json` (pipeline mode)
and `capacity_{56,64,66,68,72}s_5fps_af5_norec.json` (raw mode), each with a
`telemetry` block where noted. The old af=5-on-25fps rows measured 4.0–4.2 fps
and are superseded for capacity purposes by the genuine-5fps numbers above.

## Customer spec matrix — 5 / 10 / 15 fps × preview mode × SVR recording (2026-08-16)

**Question this answers:** how many streams can we honestly sell at a given
target fps, with and without smart-video-record (SVR), keeping ≥ 90 % of the
target rate (the 4.5–5.5 band for a 5 fps target) and GPU headroom for peaks?

**Method — production path, not fixtures:** every cell runs the real RTSP path
(go2rtc ingests host ffmpeg pushers looping the genuine `fps{5,10,15}.mp4`
fixtures; the sidecar pulls `rtsp://go2rtc:8554/raw-cam-N` via `nvurisrcbin`;
SVR is the production `recorder_kind: svr` on every source, engaging on real
detection triggers — verified segments were written). `mux_sync_inputs: 1`
(the shippable flicker fix) is on in every cell. GPU % is a 60 s window sampled
every 2 s; per-source fps is the median over the window after the harness waits
for all N sources to reach ≥ 0.9 × target (no boot stragglers counted).

Hardware: **RTX 5050 Laptop (8 GB VRAM), 16 cores, 64 GB RAM.** Full matrix:
38 cells (5/10/15 fps × pipeline/raw × rec off/on × 4–6 N values), RTSP path,
files `benchmarks/results/capacity_{n}s_{fps}fps_af{fps}_{rec,norec}.json`.

### The honest spec table (largest N that still holds ≥ 90 % target fps)

| target | preview | record | max N @ fps hold | GPU p95 @ that N | VRAM @ that N | binding resource |
|---|---|---|---|---|---|---|
| 5 fps | pipeline | off | **52** | 97 % | 6.5 GB | **CPU** — publisher draw (JPEG box-drawing) |
| 5 fps | pipeline | on | **48** | 92 % | 6.2 GB | **CPU** + SVR bookkeeping |
| 5 fps | raw | off | **64** | 82 % | 7.7 GB | GPU — decode + pgie |
| 5 fps | raw | on | **64** | 72 % | 7.7 GB | GPU (SVR adds ~nothing on RTSP) |
| 10 fps | raw | off | **24–28** | 84 % @24 | 3.7 GB | GPU |
| 10 fps | raw | on | **24–28** | 83 % @24 | 3.8 GB | GPU |
| 15 fps | raw | off | **20** | 77 % @20 | 3.3 GB | GPU |
| 15 fps | raw | on | **20** | 73 % @20 | 3.3 GB | GPU |

N at which the target fps actually breaks (the fps wall, not the spec):

| target | preview | record | fps wall (drops below 90 %) |
|---|---|---|---|
| 5 fps | pipeline | off | n=56 → 4.7 fps |
| 5 fps | pipeline | on | n=52 → 4.8 fps |
| 5 fps | raw | off | n=68 → 4.7 fps |
| 5 fps | raw | on | n=68+ (64 still holds 5.00) |
| 10 fps | raw | off | beyond n=40 (40 still holds 10.00) |
| 10 fps | raw | on | beyond n=36 (36 still holds 10.00) |
| 15 fps | raw | off | beyond n=28 (28 still holds 15.00) |
| 15 fps | raw | on | beyond n=24 (24 still holds 15.00) |

### What to tell the customer (the conservative sellable numbers)

With **raw preview** (the production default) and SVR recording on:

- **5 fps → 48–56 streams.** fps holds 5.00 at n=64, but by the 85 % GPU-p95
  cap the honest number is ~48–56 (p95 63–92 % across the noisy burst cells;
  VRAM 6.2–7.7 GB). Above n=56, VRAM is 90–94 % of the 8 GB card — no peak
  headroom. Recommend **48** for guaranteed headroom, **56** if accepting
  ~86 % VRAM.
- **10 fps → 24 streams.** fps holds 10.00 even at n=40, but GPU p95 crosses
  85 % between n=24 (84 %) and n=28 (58 % — burst noise; a dirty window).
  Recommend **24**.
- **15 fps → 20 streams.** fps holds 15.00 to n=28+, but p95 crosses 85 % at
  n=24 (97 %). Recommend **20** (p95 73–77 %, VRAM 3.3 GB).

Pipeline (annotated JPEG) preview: **48 streams at 5 fps** with recording, 52
without. Do NOT sell pipeline preview on the GPU-p95 criterion — it cannot meet
a 85 % p95 cap at ANY stream count (95–99 % from n=40) because the
sync-inputs batch-close bursts peg the GPU during every batch; its real
constraint is the CPU publisher, and its real wall is 52–56 streams.

### SVR recording cost (smart-record, not raw-stream recording)

- **Raw preview: ~0 streams.** SVR on RTSP adds nothing measurable to the
  wall (64 holds both ways) and no VRAM (7.7 GB both ways) — it only engages
  on detection triggers.
- **Pipeline preview: ~4 streams** (52 → 48) — SVR bookkeeping on the same
  CPU-publisher-bound path.
- This is the honest answer to "with and without recording": the record
  branch costs ~0 % on the GPU path and ~8 % of the 5 fps wall on the CPU path.

### The bimodal-GPU caveat (why p95 is not the whole story)

Every sync-inputs=1 cell shows a bimodal GPU series: ~30–70 % median with
95–100 % bursts during batch closes (e.g. n=60 rec-on: 31/96 alternating).
The **median** is the sustained level; **p95** is the burst level. A cell can
hold the target fps at p95=100 % (10 fps n=40, 15 fps n=28) because the
batches still close in time. Spec on the fps hold first, use p95 ≤ 85 % as
the headroom rule, and expect run-to-run p95 noise of ±15 points (n=56 rec-on
read 92 % once, 72 % at n=64) — re-verify the headline cells before quoting.
