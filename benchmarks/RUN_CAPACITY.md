# IRIS capacity benchmark — this machine (RTX PRO 6000 Blackwell)

Run these from the repo root. Everything is logged, so at the end you can just
paste back the summary + log.

> **M1 process-sharding A/B (single vs 2 vs 3 sidecar processes) is the current
> validation — see `CONTEXT.md` at the repo root for the handoff protocol.**
> This file documents the single-process grid this machine ran historically.

**Machine being measured:** RTX PRO 6000 Blackwell (96 GB VRAM), 128 threads.
**Fixture source:** `data/datasets/Wisenet/video_sets/set_2/video2_1.avi`
(1280×720, transcoded to 25/10/5 fps H.264 in `benchmarks/fixtures/`).

---

## 0. Preflight (10 s)

```bash
cd /home/n1-master-station-urca/IRIS/surveillance-camera-sys

# all 4 containers must be Up
docker compose ps

# GPU must be idle (no dashboard/other jobs running) — record this baseline
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# fixtures + tool sanity
ls -la benchmarks/fixtures/
python3 benchmarks/stream_capacity.py summary   # will be empty or partial — fine
```

---

## 1. Clean slate (optional but recommended)

Earlier validation runs left some `capacity_*.json` rows. Archive them so the
final grid is this machine's clean sweep:

```bash
mkdir -p benchmarks/results/pre_sweep
mv benchmarks/results/capacity_*.json  benchmarks/results/pre_sweep/ 2>/dev/null
mv benchmarks/results/capacity_summary.jsonl benchmarks/results/pre_sweep/ 2>/dev/null
echo "archived old rows"
```

---

## 2. Full capacity sweep (~45–75 min, run it and walk away)

> **`--no-record` is required.** Each source's recording branch opens a
> concurrent NVENC session; past ~15 sources the driver caps sessions and the
> pipeline reaches PLAYING but never delivers a frame (observed at n=16). The
> sweep measures the inference pipeline, so recording is turned off per cell —
> the ~15-stream recording cap is itself a reported finding.

```bash
nohup python3 benchmarks/stream_capacity.py sweep \
  --ns 1,2,4,8,12,16,24,32,48,64,96,128,192 \
  --fpss 25,10,5 \
  --saturate 90 \
  --no-record \
  > /tmp/iris_capacity_sweep.log 2>&1 &

echo "started PID $! — log: /tmp/iris_capacity_sweep.log"
```

What it does per cell: rewrites `perception/pipeline/sidecar.yaml` to N copies of the
matching-fps fixture (with `record_enabled: false`) → restarts the sidecar →
waits for the first fps line → samples GPU/VRAM/CPU for 30 s → writes
`benchmarks/results/capacity_<N>s_<fps>fps.json` + one row in
`capacity_summary.jsonl`. Each row stops early once median GPU util ≥ 90 % or
a stream can't keep its target fps ("SATURATED").

**Timing note:** the first time each stream count (1,2,4,…,192) is hit, the
sidecar builds a batch-`N` INT8 engine from ONNX (calibration, ~1–2 min).
That's expected — later cells with the same N deserialize in seconds. So the
25 fps row is the slowest; 10/5 fps rows fly.

> Earlier runs failed at n=16 with `record_enabled: true` — that was the NVENC
> recording-session cap, not a measurement. Always pass `--no-record`.

### Monitor (any time)

```bash
tail -20 /tmp/iris_capacity_sweep.log          # per-cell progress
pgrep -af stream_capacity                       # still running?
```

Each cell prints a line like:

```
n=16 fps=25: gpu_util_med=23.0% vram=3741MiB stream_fps={'cam-1': 25.0, ...}
  -> n=64 @ 25fps SATURATED (gpu%=93, min_src_fps=24.1); stopping row
```

### When it's done

The last line is `sweep done; container restarted on restored yaml`. Then:

```bash
python3 benchmarks/stream_capacity.py summary
```

---

## 3. What to paste back

Paste these back (or attach the files):

1. Output of `python3 benchmarks/stream_capacity.py summary` — the full grid.
2. The sweep log: `cat /tmp/iris_capacity_sweep.log` (or just the last ~40 lines).
3. The row files: `ls benchmarks/results/capacity_*.json` (optional).

---

## 4. (Optional) Suite C — live wall at scale

After the sweep, we know the sustainable stream count. To watch the wall on
**http://localhost:8000**:

```bash
# 1. copy a good mix of dataset videos into data/uploads
python3 scripts/live_streams.py copy 'data/datasets/Avenue/training_videos/*.avi'  --max 12
python3 scripts/live_streams.py copy 'data/datasets/Wisenet/video_sets/*/video*.avi' --max 24
python3 scripts/live_streams.py copy 'data/datasets/CHIRLA0/videos/videos/seq_000/*.avi' --max 12

# 2. show what's usable, then transcode non-h264 AVIs to h264 so they loop
python3 scripts/live_streams.py list
python3 scripts/live_streams.py transcode

# 3. run the wall: every usable upload, no NVENC recording, ~5 fps inference
python3 scripts/live_streams.py run --all --no-record --process-fps 5 --no-rtsp

# 4. verify + GPU under the wall
python3 scripts/live_streams.py status
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# 5. when done, put the committed single-source config back
python3 scripts/live_streams.py restore
```

Adjust `--max` counts to hit roughly the max sustainable stream count from the
sweep (e.g. 48–96 streams).

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| A cell times out with `no 25fps reading within 240.0s` | Engine build hung/failed. Check `docker logs --tail 40 <sidecar-container>`. The tool restores the yaml and continues. |
| Container crash-loops with `Segmentation fault` + `calibrateEngine` | `int8_calib_batch_size` doesn't match the source count — the fixed tool mirrors it automatically. If you see this, the sweep is running an OLD copy of `stream_capacity.py`. |
| Sweep killed mid-cell leaves `sidecar.yaml` with many sources | `git checkout -- perception/pipeline/sidecar.yaml` then `docker restart <sidecar-container>`. |
| `summary` mixes laptop rows | Rows were archived to `benchmarks/results/pre_sweep/`; check nothing was moved back. |
