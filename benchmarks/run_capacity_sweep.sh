#!/usr/bin/env bash
# One-shot capacity sweep for this machine. Run from the repo root:
#   bash benchmarks/run_capacity_sweep.sh
# Logs to /tmp/iris_capacity_sweep.log and prints the grid at the end.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

NS="1,2,4,8,12,16,24,32,48,64,96,128,192"
FPSS="25,10,5"
SAT=90
LOG=/tmp/iris_capacity_sweep.log

echo "== preflight =="
docker compose ps --format '{{.Name}} {{.Status}}' | grep -E 'sidecar|iris' || { echo "stack not up — run: docker compose up -d"; exit 1; }
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
ls -la benchmarks/fixtures/ 2>/dev/null || { echo "fixtures missing — run: python3 benchmarks/stream_capacity.py fixtures --source data/datasets/Wisenet/video_sets/set_2/video2_1.avi"; exit 1; }

echo "== archiving any partial rows from earlier runs =="
mkdir -p benchmarks/results/pre_sweep
mv benchmarks/results/capacity_*.json      benchmarks/results/pre_sweep/ 2>/dev/null
mv benchmarks/results/capacity_summary.jsonl benchmarks/results/pre_sweep/ 2>/dev/null
echo "archived (pre_sweep/)."

echo "== starting sweep: N in {$NS} x fps in {$FPSS}, stop at >=${SAT}% GPU =="
echo "   (--no-record: recording off — NVENC caps ~15 concurrent recording sessions,"
echo "    past which the pipeline stalls; the cap itself is a finding)"
python3 benchmarks/stream_capacity.py sweep --ns "$NS" --fpss "$FPSS" --saturate "$SAT" --no-record 2>&1 | tee "$LOG"

echo ""
echo "== sweep finished. summary grid: =="
python3 benchmarks/stream_capacity.py summary
echo ""
echo "Full per-cell log: $LOG"
