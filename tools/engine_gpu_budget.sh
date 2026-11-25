#!/usr/bin/env bash
# What a config costs on the GPU, per model: replay it on its golden clip and print the busy% of every GPU stage.
#
#   tools/engine_gpu_budget.sh tests/configs/golden-attendance.yaml [--max-total 80]
#
# Needs the GPU to itself (like the golden run). The sidecar's [telemetry] lines are read from the container log
# after the run; see tools/engine_gpu_budget.py for what they mean. Use it before and after adding a module: the
# difference is that module's share.
set -euo pipefail
cfg=${1:?usage: engine_gpu_budget.sh <sidecar-config.yaml> [engine_gpu_budget.py options]}
shift
cd "$(dirname "$0")/.."
out=$(mktemp --suffix=.jsonl)
trap 'rm -f "$out"' EXIT
tools/engine_golden_run.sh "$cfg" "$out" >/dev/null
docker compose logs --no-color deepstream-sidecar 2>&1 | python3 tools/engine_gpu_budget.py - "$@"
