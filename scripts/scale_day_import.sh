#!/usr/bin/env bash
# Bring a full-day run (made by scripts/scale_day_export.sh on the server) into THIS machine, so the platform,
# the VLM page, the crops/plates and Grafana show it as if it had run here.
#
#   scripts/scale_day_import.sh <export dir on the server>  [HOST=iris-work]   copy it, then import
#   scripts/scale_day_import.sh --local <folder>                               import an export already on this disk
#
# SHIFT_TO=2026-10-06T00:00:00 scripts/scale_day_import.sh ...  also moves every run timestamp so the footage appears
# to have begun then (scripts/scale_day_shift.py; Grafana history keeps the real times).
#
# Before anything is replaced, the laptop's current run state is backed up to ~/URCA/IRIS/backups/<time>/
# (database dump, data/, monitoring volumes). Afterwards the stack is opened in browse mode.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PROJECT=scale
if [ "${ON_SERVER:-0}" = 1 ]; then       # run ON the shared server itself (scripts/scale_demo_v3.sh): its own paths and port
    DEST="${DEST:-$REPO}"
    export LOCAL=0 ON_SERVER=1 UI_HTTPS_PORT="${UI_HTTPS_PORT:-35443}" SCALE_DATA="${SCALE_DATA:-$DEST/data}"
else
    DEST="${DEST:-$HOME/URCA/IRIS/scale-local}"
    export LOCAL=1 UI_HTTPS_PORT="${UI_HTTPS_PORT:-35444}"
fi
DEPLOY="$REPO/scripts/scale13_deploy.sh"
DC="docker compose -p $PROJECT -f docker-compose.yml -f docker-compose.scale.yml -f docker-compose.scale-ui.yml -f docker-compose.scale-monitoring.yml"
STAMP=$(date +%Y%m%d-%H%M%S)
log() { echo "[$(date +%T)] $*"; }

if [ "${1:-}" = "--local" ]; then
    SRC="${2:?usage: --local <folder>}"
else
    REMOTE="${1:?usage: scale_day_import.sh <export dir on the server> | --local <folder>}"
    SRC="$HOME/URCA/IRIS/imports/$(basename "$REMOTE")"
    mkdir -p "$SRC"
    log "copying $REMOTE from ${HOST:-iris-work} -> $SRC (resumable: run it again if interrupted)"
    rsync -a --partial --info=progress2 -e "ssh -o RequestTTY=no -o RemoteCommand=none" "${HOST:-iris-work}:$REMOTE/" "$SRC/"
fi
[ -f "$SRC/manifest.json" ] || { echo "$SRC is not a day export (no manifest.json)" >&2; exit 1; }
log "verifying checksums"
(cd "$SRC" && sha256sum -c --quiet SHA256SUMS) || { echo "checksum mismatch: the copy is damaged, run it again" >&2; exit 1; }

# the code the run used is the one this machine runs, but say so when it differs
grep -h . "$SRC/config/.synced_commit" 2>/dev/null | sed 's/^/    run commit: /' || true

BK="$HOME/URCA/IRIS/backups/$STAMP"
if [ "${SKIP_BACKUP:-0}" != 1 ]; then
log "backing up the current state to $BK"
mkdir -p "$BK/volumes"
(cd "$DEST" && $DC up -d --no-deps ui-postgres >/dev/null 2>&1) || true
for _ in $(seq 30); do docker exec ${PROJECT}-ui-postgres-1 pg_isready -U iris -d iris >/dev/null 2>&1 && break; sleep 2; done
"$REPO/scripts/scale_db.sh" dump "$BK/platform.dump" || echo "    (no database to back up)"
"$DEPLOY" down >/dev/null 2>&1 || true
# (from a container: the containers wrote part of it as root, unreadable for this user)
mkdir -p "$BK/data"
docker run --rm -v "$DEST/data":/src:ro -v "$BK/data":/dst alpine sh -c \
    'cd /src && for f in * .[!.]*; do case "$f" in _before_*|uploads) continue;; esac; [ -e "$f" ] && cp -a "$f" /dst/; done; chmod -R a+rX /dst'
for v in prometheus-data loki-data grafana-data; do
    docker run --rm -v ${PROJECT}_$v:/v:ro -v "$BK/volumes":/out alpine tar czf /out/$v.tgz -C /v . 2>/dev/null || true
done
log "backup done"
fi
"$DEPLOY" down >/dev/null 2>&1 || true      # nothing may run while its state is replaced (Redis would keep the old state in memory)

log "replacing the run state in $DEST/data"
# (the containers wrote these as root: remove them from a container)
docker run --rm -v "$DEST/data":/d alpine sh -c 'cd /d && rm -rf vehicle_crops plates vlm_arbitration flight segments alert_svr alert_clips clips redis tracking_stays.json raw_detections.jsonl scale13_metrics.csv scale13_metrics_run1.csv scale13_heals.log events.db .time_shift'
rsync -a --chmod=Du+w,Fu+w "$SRC/data/" "$DEST/data/"      # (u+w: a read-only archive must not make the run data read-only)

log "restoring the platform database"
(cd "$DEST" && $DC up -d --no-deps ui-postgres >/dev/null 2>&1)
for _ in $(seq 30); do docker exec ${PROJECT}-ui-postgres-1 pg_isready -U iris -d iris >/dev/null 2>&1 && break; sleep 2; done
"$REPO/scripts/scale_db.sh" restore "$SRC/db/platform.dump"

log "restoring the monitoring history (Grafana, Prometheus, Loki)"
for v in prometheus-data loki-data grafana-data; do
    [ -f "$SRC/volumes/$v.tgz" ] || continue
    docker volume rm -f ${PROJECT}_$v >/dev/null 2>&1 || true
    docker volume create ${PROJECT}_$v >/dev/null
    docker run --rm -v ${PROJECT}_$v:/v -v "$SRC/volumes":/in:ro alpine sh -c "tar xzf /in/$v.tgz -C /v"
done

log "opening it (browse mode: no sidecar, no players, truck feed off)"
"$DEPLOY" browse

if [ -n "${SHIFT_TO:-}" ]; then
    FROM=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['footage_began_wall'])" "$SRC/manifest.json")
    log "moving the run in time: footage began $FROM, shown as beginning $SHIFT_TO"
    "$REPO/scripts/scale_day_shift.py" --from "$FROM" --to "$SHIFT_TO"
    docker restart ${PROJECT}-iris-1 >/dev/null
fi

log "checking the database against the manifest"
python3 - "$SRC/manifest.json" <<'PY'
import json, subprocess, sys
want = json.load(open(sys.argv[1]))["db_counts"]
out = subprocess.run(["scripts/scale_db.sh", "counts"], capture_output=True, text=True, check=True).stdout
have = {l.split()[0]: int(l.split()[1]) for l in out.splitlines() if len(l.split()) == 2}
bad = {k: (want[k], have.get(k)) for k in want if have.get(k) != want[k]}
print("  all counts match the server's:", {k: have[k] for k in sorted(have)} if not bad else f"MISMATCH (server, here): {bad}")
sys.exit(1 if bad else 0)
PY
log "done. Platform https://localhost:$UI_HTTPS_PORT  VLM page /iris/vlm  Grafana /grafana/   (backup of the previous state: $BK)"
