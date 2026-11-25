#!/usr/bin/env bash
# Collect everything a full-day run produced into ONE folder, to be copied to the laptop and imported there with
# scripts/scale_day_import.sh. Runs on the host of the stack (the server), from the run directory, after the day ended.
#
#   scripts/scale_day_export.sh [OUT_DIR]        default OUT_DIR: <run dir>/../day-export-<date>-<time>
#
# What it does, in this order (nothing is deleted; the stack is left stopped, its data in place):
#   1. waits WAIT seconds (default 120) so exits held for 30 s and the last plate readings reach the platform,
#      stops the player, the monitor and the sidecar, waits until the platform has consumed the feed queue,
#      stops iris (it writes the vehicles' stays on the way out)
#   2. dumps the platform database; saves Redis; stops the monitoring containers and Redis
#   3. copies data/ (crops, plates, VLM records, events, flight records, Redis files, stays, zones, metrics, heals),
#      the Prometheus/Loki/Grafana volumes, the container and player logs, the configs, the commits
#   4. writes manifest.json (row counts, file counts, sizes, times, heals) and SHA256SUMS
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$PWD"
OUT="${1:-$(dirname "$RUN_DIR")/day-export-$(date +%Y%m%d-%H%M)}"
WAIT="${WAIT:-120}"
PROJECT=scale
DC="docker compose -p $PROJECT -f docker-compose.yml -f docker-compose.scale.yml -f docker-compose.scale-ui.yml -f docker-compose.scale-monitoring.yml"
mkdir -p "$OUT"/{db,volumes,logs,config}
log() { echo "[$(date +%T)] $*"; }

log "waiting ${WAIT}s for held exits and the last readings"
sleep "$WAIT"
python3 scripts/scale13_monitor.py stop >/dev/null 2>&1 || true
SCALE13_STATE="$RUN_DIR/data/day-player" python3 scripts/scale13_streamer.py stop >/dev/null 2>&1 || true
cp data/day-player/clock.json "$OUT/config/player_clock.json" 2>/dev/null || cp /tmp/scale13-streamer/clock.json "$OUT/config/player_clock.json" 2>/dev/null || true
END_WALL=$(date +%FT%T)
$DC stop deepstream-sidecar >/dev/null
log "sidecar stopped; waiting for the platform to consume the feed queue"
for _ in $(seq 60); do
    n=$(docker exec ${PROJECT}-redis-1 redis-cli -n 1 llen iris:camera-feed 2>/dev/null | tr -d '\r' || true)
    [ "${n:-0}" = 0 ] && break
    sleep 2
done
log "feed queue length: ${n:-?}"
sleep 20
docker stop -t 60 ${PROJECT}-iris-1 >/dev/null 2>&1 || true     # iris saves the vehicles' stays on shutdown
log "iris stopped"

scripts/scale_db.sh counts > "$OUT/db/counts.txt"
scripts/scale_db.sh dump "$OUT/db/platform.dump"
docker exec ${PROJECT}-redis-1 redis-cli SAVE >/dev/null 2>&1 || true   # (fails harmlessly when a rerun finds it stopped)
$DC stop prometheus loki promtail grafana iris-exporter node-exporter >/dev/null 2>&1 || true
docker stop -t 30 ${PROJECT}-redis-1 >/dev/null 2>&1 || true
log "database dumped, Redis saved and stopped"

log "copying data/ (without the previous runs, the footage links and uploads)"
# from a container: the containers wrote most of it (Redis files too) as root, unreadable for this user
mkdir -p "$OUT/data"
docker run --rm -v "$RUN_DIR/data":/src:ro -v "$OUT/data":/dst alpine sh -c \
    'cd /src && for f in * .[!.]*; do case "$f" in _before_*|scale13_videos|uploads) continue;; esac; [ -e "$f" ] && cp -a "$f" /dst/; done; chmod -R a+rX /dst'
for v in prometheus-data loki-data grafana-data; do
    docker run --rm -v ${PROJECT}_$v:/v:ro -v "$OUT/volumes":/out alpine tar czf /out/$v.tgz -C /v . 2>/dev/null && log "volume $v saved" || log "volume $v: not saved"
done
for c in $(docker ps -a --format '{{.Names}}' | grep "^${PROJECT}-"); do docker logs "$c" > "$OUT/logs/$c.log" 2>&1 || true; done
cp /tmp/scale13-streamer/*.log "$OUT/logs/" 2>/dev/null || true   # (the day run keeps them in data/day-player, copied with data/)
cp /tmp/scale13-streamer.log /tmp/scale13-monitor.log "$OUT/logs/" 2>/dev/null || true
cp config.yaml "$OUT/config/"; cp products/multicam/sidecar.yaml "$OUT/config/"; cp -r configs/presets/scale13 "$OUT/config/preset-scale13"
grep -v '^QWEN_' .env > "$OUT/config/env.txt" || true
cp .synced_commit "$OUT/config/" 2>/dev/null || true
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | grep "^${PROJECT}-" > "$OUT/config/images.txt" || true

python3 - "$OUT" "$END_WALL" <<'PY'
import json, os, sys, glob, time
out, end_wall = sys.argv[1], sys.argv[2]
def count(pattern):
    return len(glob.glob(os.path.join(out, pattern), recursive=True))
def lines(path):
    try:
        return open(os.path.join(out, path)).read().splitlines()
    except OSError:
        return []
clock = {}
try:
    clock = json.load(open(os.path.join(out, "config/player_clock.json")))
except (OSError, ValueError):
    pass
m = {
    "exported_at": time.strftime("%FT%T"),
    "run_ended_wall": end_wall,
    "footage_start": clock.get("start"),
    "footage_began_wall": time.strftime("%FT%T", time.localtime(clock["t0"])) if clock.get("t0") else None,
    "db_counts": {l.split()[0]: int(l.split()[1]) for l in lines("db/counts.txt") if len(l.split()) == 2},
    "heals": lines("data/scale13_heals.log"),
    "commits": lines("config/.synced_commit"),
    "files": {
        "vehicle_crops": count("data/vehicle_crops/**/*.*"),
        "plates": count("data/plates/*"),
        "vlm_arbitration": count("data/vlm_arbitration/**/*.*"),
        "flight": count("data/flight/*"),
        "redis_files": count("data/redis/**/*"),
    },
    "stays_file": os.path.exists(os.path.join(out, "data/tracking_stays.json")),
    "metrics_rows": max(0, len(lines("data/scale13_metrics.csv")) - 1),
}
json.dump(m, open(os.path.join(out, "manifest.json"), "w"), indent=2)
print(json.dumps(m, indent=2))
PY
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
log "export ready: $OUT ($(du -sh "$OUT" | cut -f1), $(wc -l < "$OUT/SHA256SUMS") files)"
