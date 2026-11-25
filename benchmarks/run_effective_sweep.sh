#!/usr/bin/env bash
# Effective-analysis-rate capacity sweep (RTX PRO 6000 workstation).
#
# Rows: analysis_fps in {5, 10, 15, 25} x record {off, on}, with camera
# sources always at native 25 fps (decode stays at source rate; only the
# analysis branch is capped). 8 rows total; each row runs N up to 128 and
# stops early on saturation (median GPU >= 90% or min per-stream fps < 85% of
# the effective target).
#
# Usage (from the repo root):
#   bash benchmarks/run_effective_sweep.sh 2>&1 | tee /tmp/iris_effective_sweep.log
#   # or, to survive terminal close:
#   nohup bash benchmarks/run_effective_sweep.sh > /tmp/iris_effective_sweep.log 2>&1 &
#
# When it finishes, the final `summary` grid prints; the per-cell JSONs land
# in benchmarks/results/capacity_<n>s_25fps_af<rate>_{rec,norec}.json.
set -u
cd "$(dirname "$0")/.."

LOG=/tmp/iris_effective_sweep.log
GRID=1,2,4,8,12,16,24,32,40,48,64,96,128

echo "== preflight =="
docker compose ps | grep sidecar || true
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader

echo "== archive previous summary + any prior effective-run cells =="
mkdir -p benchmarks/results/pre_effective
if [ -f benchmarks/results/capacity_summary.jsonl ]; then
  mv benchmarks/results/capacity_summary.jsonl benchmarks/results/pre_effective/
fi
mv benchmarks/results/capacity_*_af*_rec.json benchmarks/results/pre_effective/ 2>/dev/null || true
mv benchmarks/results/capacity_*_af*_norec.json benchmarks/results/pre_effective/ 2>/dev/null || true

echo "== sweep: 8 rows (analysis_fps x record), camera at 25 fps =="
for af in 5 10 15 25; do
  for mode in norec rec; do
    extra=""
    [ "$mode" = norec ] && extra="--no-record"
    echo "--- row: analysis_fps=$af record=$mode ---"
    # -u: unbuffered so the log shows per-cell progress live
    python3 -u benchmarks/stream_capacity.py sweep \
      --fpss 25 --ns "$GRID" --analysis-fps "$af" $extra
    echo "--- row done (af=$af, record=$mode) ---"
  done
done

echo "== restoring tracked bookkeeping + restarting on the restored config =="
python3 benchmarks/stream_capacity.py restore >/dev/null 2>&1 || true
git checkout -- benchmarks/results/_sidecar.yaml.bak 2>/dev/null || true

echo "== final summary grid =="
python3 benchmarks/stream_capacity.py summary
echo "full per-cell log: $LOG"
