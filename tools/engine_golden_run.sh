#!/usr/bin/env bash
# Replay one golden config through the sidecar and record what it writes to Redis.
#
#   tools/engine_golden_run.sh tests/configs/golden-attendance.yaml tests/baselines/golden-attendance.before.jsonl
#
# Starts from an empty Redis (this checkout's compose redis only), starts the recorder BEFORE the sidecar (streams
# are capped, so a late recorder misses the start), and stops the sidecar when the clip has ended.
set -euo pipefail
cfg=${1:?usage: engine_golden_run.sh <sidecar-config.yaml> <out.jsonl>}
out=${2:?usage: engine_golden_run.sh <sidecar-config.yaml> <out.jsonl>}
cd "$(dirname "$0")/.."

docker compose up -d --no-build redis >/dev/null
docker compose stop deepstream-sidecar >/dev/null 2>&1 || true
docker compose exec -T redis redis-cli flushall >/dev/null
# Keys the config needs in Redis before the sidecar starts, one KEY=VALUE per line in <config>.redis (e.g. the door
# regions the barber face stage reads); a flushed Redis has none.
redis_seed="${cfg%.yaml}.redis"
if [ -f "$redis_seed" ]; then
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    docker compose exec -T redis redis-cli set "${line%%=*}" "${line#*=}" >/dev/null
  done < "$redis_seed"
fi

url="redis://localhost:${REDIS_HOST_PORT:-6379}/0"
python3 tools/engine_golden.py record --redis "$url" --out "$out" --idle "${GOLDEN_IDLE:-25}" --max-seconds "${GOLDEN_MAX:-900}" &
rec=$!
sleep 1
SIDECAR_CONFIG="$cfg" docker compose up -d --no-build deepstream-sidecar >/dev/null
# The sidecar exits at the end of the clip and compose would restart it (replaying the clip forever, so the
# recorder never goes quiet). Turn the restart policy off for this container only.
docker update --restart=no "$(docker compose ps -q deepstream-sidecar)" >/dev/null
wait "$rec"
docker compose stop deepstream-sidecar >/dev/null
