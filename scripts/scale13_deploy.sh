#!/usr/bin/env bash
# The 15-camera scale test on the shared GPU server: its own directory, compose project ("scale")
# and ports, so it never touches anybody else's stack.
#
#   scale13_deploy.sh sync        code (committed files of HEAD, incl. the vehicle model), preset, .env
#                                 (with the VLM key from the local .env), images. Starts nothing.
#   scale13_deploy.sh up          sync, then start: stack -> the 13 streams -> sidecar -> UI
#   scale13_deploy.sh down        stop everything of this project (containers, the streamer)
#   scale13_deploy.sh stream start|stop|status      only the 24 h real-time playback
#   scale13_deploy.sh ui up|reset|down             only the truck-tracking UI (reset clears its trucks)
#   scale13_deploy.sh mon up|down|reset                  only the health monitoring (Grafana at <ui>/grafana/)
#   scale13_deploy.sh status
#   scale13_deploy.sh day-arm      a full-day run on the footage clock (see "Full-day run" below)
#   scale13_deploy.sh day-status   how that run is doing (footage clock, cameras, heals, platform counts, disk)
#   scale13_deploy.sh continue     keep what is in the platform (e.g. an imported day) and start a LIVE run beside it:
#                                  the footage plays from START (default 11:49) as new live footage, stamped with today
#   scale13_deploy.sh browse       (LOCAL=1) open an imported run: platform, camera-system app, monitoring; no sidecar,
#                                  no players, and the truck feed switched off so nothing is re-sent
#
# Full-day run: BEGIN_AT=2026-10-06T00:00:00 scale13_deploy.sh day-arm. The footage of START (default 00:00 of the
# recorded day) begins at that local wall time, every file at its recorded time, holes filled with black, stopping
# at the end of the day. Until BEGIN_AT the cameras show black. Redis is persistent and the previous run's data is
# moved to data/_before_<time>/. Afterwards scripts/scale_day_export.sh (server) and scale_day_import.sh (laptop).
#
# ON_SERVER=1: run this script ON the server itself (a scheduled day-arm must not depend on the laptop being
# awake): same server paths and ports, commands run locally, no code sync (the copy it runs from is the synced one).
#
# Everything is overridable: HOST, DEST, UI_DEST, VIDEOS (the recordings on the server), START.
#
# LOCAL=1 runs the same stack on THIS machine instead of over ssh (DEST/UI_DEST default to
# ~/URCA/IRIS/scale-local and truck-tracking-local, recordings from /media/mmtayee/WD_1, UI at
# https://localhost:35443). CAMS="cam-01,cam-02,..." limits it to those cameras (default: all 15),
# useful on a smaller GPU. Example:  LOCAL=1 CAMS=cam-01,cam-02,cam-03,cam-04,cam-05,cam-06 \
#                                    START=2026-09-13T11:49:00 scripts/scale13_deploy.sh up
set -euo pipefail
LOCAL="${LOCAL:-0}"
CAMS="${CAMS:-}"
if [ "$LOCAL" = 1 ]; then
    HOST="local"
    DEST="${DEST:-$HOME/URCA/IRIS/scale-local}"
    UI_DEST="${UI_DEST:-$HOME/URCA/IRIS/truck-tracking-local}"
    # the recordings drive: its mount point changes (WD_1, WD_11 after a re-plug), so find it by label
    VIDEOS="${VIDEOS:-$(findmnt -rn -S LABEL=WD_1 -o TARGET 2>/dev/null | head -1 || true)}"
    VIDEOS="${VIDEOS:-/media/mmtayee/WD_1}"
    ls "$VIDEOS"/D41_*.mp4 >/dev/null 2>&1 || { echo "no recordings in $VIDEOS: is the WD_1 drive mounted? (set VIDEOS=...)" >&2; case "${1:-}" in down|status|browse|monitor|day-status) ;; *) exit 1 ;; esac; }
    case "${1:-}" in up|continue|day-arm)
        nvidia-smi -L >/dev/null 2>&1 || { echo "the NVIDIA driver is not usable here: $(nvidia-smi 2>&1 | head -1)" >&2
            echo "after a driver update ('Driver/library version mismatch') reboot the machine, then run this again" >&2; exit 1; } ;;
    esac
    # The platform is served on this machine's LAN address, so another device on the network can open it, and on localhost
    # (PUBHOST=... to force an address; no network: localhost only).
    LAN_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i == "src") { print $(i + 1); exit }}')"
    PUBHOST="${PUBHOST:-${LAN_IP:-localhost}}"
    UI_HTTPS_PORT="${UI_HTTPS_PORT:-35444}"      # 35443 may be an ssh tunnel to the server
    # images this machine already has, used as the base of the project's own tags
    IRIS_BASE_IMAGE="${IRIS_BASE_IMAGE:-iris-iris:latest}"
    SIDECAR_BASE_IMAGE="${SIDECAR_BASE_IMAGE:-iris-deepstream-sidecar:latest}"
elif [ "${ON_SERVER:-0}" = 1 ]; then
    HOST="server"
    DEST="${DEST:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    UI_DEST="${UI_DEST:-/data/IRIS/TESTING_PLAYGROUND/truck-tracking-scale}"
    VIDEOS="${VIDEOS:-/data/IRIS/data/el-mohandes}"
    PUBHOST="${PUBHOST:-10.2.0.60}"
    UI_HTTPS_PORT="${UI_HTTPS_PORT:-35443}"
    IRIS_BASE_IMAGE="${IRIS_BASE_IMAGE:-kw-iris:latest}"
    SIDECAR_BASE_IMAGE="${SIDECAR_BASE_IMAGE:-surveillance-camera-sys-deepstream-sidecar:latest}"
else
    HOST="${HOST:-iris@10.2.0.60}"
    DEST="${DEST:-/data/IRIS/TESTING_PLAYGROUND/surveillance-camera-sys-scale}"
    UI_DEST="${UI_DEST:-/data/IRIS/TESTING_PLAYGROUND/truck-tracking-scale}"
    VIDEOS="${VIDEOS:-/data/IRIS/data/el-mohandes}"
    PUBHOST="${HOST#*@}"
    UI_HTTPS_PORT="${UI_HTTPS_PORT:-35443}"
    IRIS_BASE_IMAGE="${IRIS_BASE_IMAGE:-kw-iris:latest}"
    SIDECAR_BASE_IMAGE="${SIDECAR_BASE_IMAGE:-surveillance-camera-sys-deepstream-sidecar:latest}"
fi
# the second address the https proxy answers on: localhost for the laptop; on the server the other of its office and Tailscale addresses
if [ "$LOCAL" = 1 ]; then ALT_HOST="${ALT_HOST:-$([ "$PUBHOST" = localhost ] && echo 127.0.0.1 || echo localhost)}"
else ALT_HOST="${ALT_HOST:-$([ "$PUBHOST" = 10.2.0.60 ] && echo 100.102.162.114 || echo 10.2.0.60)}"; fi
START="${START:-2026-09-13T00:00:00}"     # the footage clock time every camera starts at
# START may be given as just a time of the recorded day, "14:30" or "14:30:15": it then means that time on the footage day
FOOTAGE_DAY="${FOOTAGE_DAY:-2026-09-13}"
if [[ "$START" =~ ^([0-9]{1,2}):([0-9]{2})(:([0-9]{2}))?$ ]]; then
    START=$(printf '%sT%02d:%02d:%02d' "$FOOTAGE_DAY" "$((10#${BASH_REMATCH[1]}))" "$((10#${BASH_REMATCH[2]}))" "$((10#${BASH_REMATCH[4]:-0}))")
fi
RTSP_PORT=38554
PROJECT=scale
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI_SRC="${UI_SRC:-$REPO/../truck-tracking}"
cd "$REPO"
if [ "$LOCAL" = 1 ] || [ "${ON_SERVER:-0}" = 1 ]; then SSH=(bash -c); else SSH=(ssh -o BatchMode=yes -o ServerAliveInterval=30 "$HOST"); fi
DC="docker compose -p $PROJECT -f docker-compose.yml -f docker-compose.scale.yml -f docker-compose.scale-ui.yml -f docker-compose.scale-monitoring.yml"
STACK="redis minio go2rtc deepstream-sidecar iris"
# streamer --skip for every camera not in CAMS
SKIP_ARG=""
PER_FILE_ARG=""
[ "$LOCAL" = 1 ] && PER_FILE_ARG="--per-file all"   # the concat seek fails on this machine
if [ -n "$CAMS" ]; then
    SKIP=""; for n in $(seq -w 1 15); do case ",$CAMS," in *",cam-$n,"*) ;; *) SKIP="$SKIP,cam-$n" ;; esac; done
    SKIP_ARG="--skip ${SKIP#,}"
fi

# true while a full-day run is armed on this host (its Redis then persists, see docker-compose.scale.yml)
day_run_marked() { "${SSH[@]}" "test -f '$DEST/data/.day_run'"; }

sync() {
    if [ "${ON_SERVER:-0}" = 1 ]; then echo "==> code: running on the server, using what was synced"; return 0; fi
    local SSH_TEST_DAY=false
    day_run_marked 2>/dev/null && SSH_TEST_DAY=true
    echo "==> code (committed files of HEAD) -> $DEST"
    "${SSH[@]}" "mkdir -p '$DEST'"
    if command -v rsync >/dev/null && git diff --quiet HEAD -- . 2>/dev/null; then
        # Send only the files whose content differs (the full archive is ~2.6 GB, slow over a
        # remote link). The tree equals HEAD here, so this is the same set of files.
        if [ "$LOCAL" = 1 ]; then
            mkdir -p "$DEST"
            git ls-files -z | rsync -a --checksum --from0 --files-from=- ./ "$DEST/"
        else
            git ls-files -z | rsync -a --checksum --from0 --files-from=- -e "ssh -o BatchMode=yes" ./ "$HOST:$DEST/"
        fi
    else
        git archive HEAD | "${SSH[@]}" "tar -x -C '$DEST'"
    fi
    echo "==> .env (own project name and ports)"
    "${SSH[@]}" "cat > '$DEST/.env.new'" <<ENV
COMPOSE_PROJECT_NAME=$PROJECT
TRUCK_UI_DIR=$UI_DEST
SCALE_PUBLIC_URL=/iris
SCALE_GO2RTC_URL=/go2rtc
IRIS_PRODUCT=iris
PUBLIC_HOST=$PUBHOST
ALT_HOST=$ALT_HOST
UI_HTTPS_PORT=$UI_HTTPS_PORT
UI_WEB_PORT=35173
IRIS_UI_PORT=$((UI_HTTPS_PORT + 1))
IRIS_HOST_PORT=38000
REDIS_HOST_PORT=36379
GO2RTC_RTSP_PORT=$RTSP_PORT
GO2RTC_API_PORT=31984
GO2RTC_WEBRTC_PORT=38555
MINIO_API_HOST_PORT=39000
MINIO_CONSOLE_HOST_PORT=39001
SIDECAR_TELEMETRY_PORT=39101
GRAFANA_PORT=${GRAFANA_PORT:-33000}
PROMETHEUS_PORT=39090
NODE_EXPORTER_PORT=39100
FOOTAGE_DIR=$VIDEOS
REDIS_AOF=no
SCALE_KEEP_RUN_FILES=$($SSH_TEST_DAY && echo 1 || echo 0)
REDIS_SAVE=$($SSH_TEST_DAY && echo '60 1' || true)
ENV
    # The hosted VLM's key, base URL and model go from the local .env to the server's, readable by
    # its owner only. Never printed, never committed.
    if grep -q '^QWEN_API_KEY=' .env 2>/dev/null; then
        grep -E '^(QWEN_API_KEY|QWEN_BASE_URL|QWEN_MODEL)=' .env | "${SSH[@]}" "umask 077; cat >> '$DEST/.env.new'"
        echo "    VLM key copied (QWEN_*), mode 600"
    else
        echo "    WARNING: no QWEN_API_KEY in the local .env; the VLM would not work"
    fi
    "${SSH[@]}" "chmod 600 '$DEST/.env.new' && mv '$DEST/.env.new' '$DEST/.env'"
    # which commits this is, for the manifest of a day-run export (the server copy has no .git)
    "${SSH[@]}" "cat > '$DEST/.synced_commit'" <<COMMITS
camera_system=$(git rev-parse HEAD)$(git diff --quiet HEAD -- . 2>/dev/null || echo ' (+uncommitted changes, not shipped)')
truck_tracking=$(git -C "$UI_SRC" rev-parse HEAD 2>/dev/null || echo unknown)
COMMITS
    echo "==> preset scale13"
    "${SSH[@]}" "cd '$DEST' && bash scripts/switch_preset.sh scale13 >/dev/null && echo applied"
    if [ -n "$CAMS" ]; then
        # keep only the listed cameras as sidecar sources (the others are not played either)
        "${SSH[@]}" "cd '$DEST' && python3 - '$CAMS' <<'PY'
import re, sys
keep = set(sys.argv[1].split(','))
out = []
for line in open('products/multicam/sidecar.yaml').read().splitlines():
    m = re.match(r'\s+- \{id: (cam-\d+),', line)
    out.append(line if not m or m.group(1) in keep else '    # (not run here) ' + line.strip())
open('products/multicam/sidecar.yaml', 'w').write('\n'.join(out) + '\n')
print('    cameras kept:', ','.join(sorted(keep)))
PY"
    fi
    # The images this repo builds already exist on the server (same Dockerfiles); reuse them
    # instead of rebuilding 30 GB. The sidecar image there predates the Re-ID dependencies, so a
    # thin layer adds torch and friends when `import torch` fails (perception/pipeline/Dockerfile.addons).
    # on the laptop the project's iris image is a tag of the one `docker compose build iris` makes (it carries the frontend):
    # refresh it, so a rebuilt frontend (e.g. the Hazard Detection tab) is what the demo runs
    [ "$LOCAL" = 1 ] && docker tag "$IRIS_BASE_IMAGE" $PROJECT-iris:latest 2>/dev/null
    "${SSH[@]}" "docker image inspect $PROJECT-iris:latest >/dev/null 2>&1 || docker tag $IRIS_BASE_IMAGE $PROJECT-iris:latest 2>/dev/null || { echo '==> building the iris image from the repo (no image to reuse; first time takes a while)'; cd '$DEST' && $DC build iris; }"
    "${SSH[@]}" "docker image inspect $PROJECT-deepstream-sidecar:latest >/dev/null 2>&1 || docker tag $SIDECAR_BASE_IMAGE $PROJECT-deepstream-sidecar:latest 2>/dev/null || { echo '==> building the sidecar image from the repo (no image to reuse; first time takes a long while)'; cd '$DEST' && $DC build deepstream-sidecar; }"
    if ! "${SSH[@]}" "docker run --rm --entrypoint python3 $PROJECT-deepstream-sidecar:latest -c 'import torch' >/dev/null 2>&1"; then
        echo "==> adding torch to the sidecar image (thin layer, a few minutes)"
        if [ "$LOCAL" = 1 ]; then cp perception/pipeline/Dockerfile.addons "/tmp/Dockerfile.$PROJECT-addons"; else scp -q -o BatchMode=yes perception/pipeline/Dockerfile.addons "$HOST:/tmp/Dockerfile.$PROJECT-addons"; fi
        "${SSH[@]}" "mkdir -p /tmp/$PROJECT-empty-ctx && cd /tmp/$PROJECT-empty-ctx && docker build -q -f /tmp/Dockerfile.$PROJECT-addons --build-arg BASE=$SIDECAR_BASE_IMAGE -t $PROJECT-deepstream-sidecar:latest ."
    fi
    echo "synced."
}

stream() {
    # during a day run the players keep their clock and playlists on disk (they survive a reboot of the machine)
    local ST="mkdir -p '$DEST/data/day-player' && export SCALE13_STATE='$DEST/data/day-player';"
    day_run_marked 2>/dev/null || ST=":;"
    case "${1:-status}" in
        start)
            local TIMELINE_ARG=""    # decided when the players start (TIMELINE=1 BEGIN_AT=... is set per call)
            [ "${TIMELINE:-0}" = 1 ] && TIMELINE_ARG="--timeline --stop-at-end ${BEGIN_AT:+--begin-at $BEGIN_AT}"
            "${SSH[@]}" "cd '$DEST' && $ST python3 scripts/scale13_streamer.py status | head -1 | grep -q '^running' && echo 'streams already running' || { nohup setsid python3 scripts/scale13_streamer.py run --videos '$VIDEOS' --port $RTSP_PORT --start '$START' $SKIP_ARG $PER_FILE_ARG $TIMELINE_ARG > /tmp/scale13-streamer.log 2>&1 < /dev/null & sleep 4; head -14 /tmp/scale13-streamer.log; }" ;;
        stop) "${SSH[@]}" "cd '$DEST' && $ST python3 scripts/scale13_streamer.py stop" ;;
        status) "${SSH[@]}" "cd '$DEST' && $ST python3 scripts/scale13_streamer.py status | head -1; pgrep -fc 'ffmpeg.*raw-cam' || true" ;;
    esac
}

ui_sync() {
    if [ "${ON_SERVER:-0}" = 1 ]; then return 0; fi
    [ -d "$UI_SRC/apps/web" ] || { echo "no truck-tracking checkout at $UI_SRC (set UI_SRC)"; return 1; }
    echo "==> truck-tracking UI (committed files of its HEAD: $(git -C "$UI_SRC" branch --show-current)) -> $UI_DEST"
    "${SSH[@]}" "mkdir -p '$UI_DEST'"
    git -C "$UI_SRC" archive HEAD | "${SSH[@]}" "tar -x -C '$UI_DEST'"
    # Its own .env: database in the ui-postgres container, redis db 1 of this stack, fresh secrets.
    "${SSH[@]}" "test -f '$UI_DEST/.env'" 2>/dev/null || {
        J1=$(python3 -c 'import secrets;print(secrets.token_urlsafe(48))'); J2=$(python3 -c 'import secrets;print(secrets.token_urlsafe(48))')
        "${SSH[@]}" "cd '$UI_DEST' && cp .env.example .env && sed -i \
          -e 's#@localhost:5432#@ui-postgres:5432#g' \
          -e 's#^REDIS_URL=.*#REDIS_URL=redis://redis:6379/1#' \
          -e 's#^WEB_ORIGIN=.*#WEB_ORIGIN=https://$PUBHOST:$UI_HTTPS_PORT#' \
          -e 's#^COOKIE_SECURE=.*#COOKIE_SECURE=true#' \
          -e 's#^COOKIE_DOMAIN=.*#COOKIE_DOMAIN=#' \
          -e 's#^JWT_ACCESS_SECRET=.*#JWT_ACCESS_SECRET=$J1#' \
          -e 's#^JWT_REFRESH_SECRET=.*#JWT_REFRESH_SECRET=$J2#' .env && chmod 600 .env"
    }
}

ui() {
    case "${1:-up}" in
        up)
            ui_sync
            "${SSH[@]}" "cd '$DEST' && $DC up -d ui-postgres ui https-proxy 2>&1 | tail -3"
            echo "==> waiting for the UI (first start installs and builds, a few minutes)"
            for _ in $(seq 120); do
                code=$("${SSH[@]}" "curl -s -o /dev/null -w '%{http_code}' -X POST localhost:35173/api/v1/auth/login -H 'content-type: application/json' -d '{}'" 2>/dev/null || true)
                case "$code" in 4*|2*) break ;; esac; sleep 5
            done
            echo "    UI answering (HTTP $code)"
            echo "==> registering our company, stops and cameras in it"
            "${SSH[@]}" "cd '$DEST' && python3 scripts/provision_truck_tracking.py --container $PROJECT-ui-postgres-1 2>&1 | tail -2"
            ;;
        reset) "${SSH[@]}" "cd '$DEST' && python3 - <<'PY'
import subprocess, sys, yaml
sys.path.insert(0, '.')
from app.truck_feed import tenant_uuid
tf = yaml.safe_load(open('config.yaml'))['truck_feed']
sql = open('scripts/reset_truck_tracking_history.sql').read()
subprocess.run(['docker','exec','-i','$PROJECT-ui-postgres-1','psql','-U','iris','-d','iris','-v','ON_ERROR_STOP=1',
                '-v','drop_trucks=1','-v',\"company='%s'\" % tenant_uuid(tf['company_id'])], input=sql, text=True, check=True)
PY" ;;
        down) "${SSH[@]}" "cd '$DEST' && $DC stop ui ui-postgres" ;;
    esac
}

mon() {
    case "${1:-up}" in
        up)
            # the Caddyfile is a bind-mounted file that the sync replaced: recreate the proxy to see it
            "${SSH[@]}" "cd '$DEST' && $DC up -d --build iris-exporter prometheus grafana node-exporter loki promtail 2>&1 | tail -4 && $DC up -d --force-recreate --no-deps https-proxy 2>&1 | tail -1"
            echo "    health dashboard: https://$PUBHOST:$UI_HTTPS_PORT/grafana/   (Prometheus: localhost:39090 on the host)" ;;
        down) "${SSH[@]}" "cd '$DEST' && $DC stop iris-exporter prometheus grafana node-exporter loki promtail" ;;
        reset)
            # a fresh run starts with empty graphs and logs: drop the metric history (Prometheus) and the
            # log history (Loki), and move the events log and the stall dumps of the last run into an
            # archive folder (they are evidence, so they are kept, not deleted)
            "${SSH[@]}" "cd '$DEST' && $DC rm -sf iris-exporter prometheus grafana node-exporter loki promtail >/dev/null 2>&1; docker volume rm -f ${PROJECT}_prometheus-data ${PROJECT}_loki-data ${PROJECT}_promtail-positions >/dev/null 2>&1; true"
            "${SSH[@]}" "mkdir -p '$DEST/data/flight' && docker run --rm -v '$DEST/data/flight:/f' -v '$DEST/scripts/archive_flight.py:/a.py:ro' python:3.12-slim python /a.py /f"
            echo "    monitoring history cleared (metrics, logs)" ;;
    esac
}

up_all() {
    sync
    "${SSH[@]}" "cd '$DEST' && $DC up -d --no-build redis minio go2rtc 2>&1 | tail -3"
    stream start
    "${SSH[@]}" "cd '$DEST' && $DC up -d --no-build --force-recreate deepstream-sidecar iris 2>&1 | tail -3"
    ui up
    mon up
}

# Keep the platform's history (an imported historical day: trucks, journeys, sightings, crops, stays) and start a live
# run beside it. Unlike `up` it first clears the camera system's Redis STREAMS (the last frames, assignments, exits and
# plate events of the previous run), which a restart would otherwise replay into the platform at their old times; the
# id counter (iris:mtmc:next_gid) and the crop index (iris:vehicle:crops) are kept, so new trucks get new ids and the
# old trucks' crops still show. Not cleared: the database, Grafana/Prometheus history, data/ files.
continue_run() {
    echo "==> stopping the current stack (nothing is deleted)"
    stream stop || true
    "${SSH[@]}" "cd '$DEST' && $DC down 2>&1 | tail -2"
    echo "==> clearing the previous run's Redis streams (keeping the id counter and the crop index)"
    "${SSH[@]}" "cd '$DEST' && $DC up -d --no-build redis 2>&1 | tail -1"
    "${SSH[@]}" "for _ in \$(seq 30); do docker exec ${PROJECT}-redis-1 redis-cli ping 2>/dev/null | grep -q PONG && break; sleep 1; done; n=0; for k in \$(docker exec ${PROJECT}-redis-1 redis-cli --scan --pattern 'iris:*' | grep -vE '^(iris:mtmc:next_gid|iris:vehicle:crops|iris:ctrl:zones_version|iris:analytics:zones_applied)\$'); do docker exec ${PROJECT}-redis-1 redis-cli unlink \"\$k\" >/dev/null; n=\$((n+1)); done; echo \"    cleared \$n keys; next global id: \$(docker exec ${PROJECT}-redis-1 redis-cli get iris:mtmc:next_gid)\""
    echo "==> starting the live run: footage $START plays now"
    # browse mode (an imported day) switched the truck feed off; a live run needs it, and on the server `sync` (which rewrites .env) is skipped
    "${SSH[@]}" "cd '$DEST' && sed -i '/^TRUCK_FEED_ENABLED=/d;/^TRACKING_REPLAY=/d' .env"
    up_all
    echo "continuing. The platform keeps its earlier trucks; the new footage adds new ones."
}

# ---- Full-day run --------------------------------------------------------------------------------
# Run state a new run must start without. Moved (not deleted) to data/_before_<time>/: it is evidence of the last run.
RUN_STATE="vehicle_crops plates vlm_arbitration flight segments alert_svr alert_clips clips redis tracking_stays.json raw_detections.jsonl scale13_metrics.csv scale13_metrics_run1.csv scale13_heals.log events.db"

monitor() {
    case "${1:-start}" in
        start) "${SSH[@]}" "cd '$DEST' && python3 scripts/scale13_monitor.py stop >/dev/null 2>&1; SCALE_DEST='$DEST' nohup setsid python3 scripts/scale13_monitor.py run --heal-after 120 --heal-cooldown 600 --out '$DEST/data/scale13_metrics.csv' > /tmp/scale13-monitor.log 2>&1 < /dev/null & sleep 2; echo \"monitor pid \$(cat /tmp/scale13-monitor.pid)\"" ;;
        stop) "${SSH[@]}" "cd '$DEST' && python3 scripts/scale13_monitor.py stop" ;;
    esac
}

# The run's settings go on disk and a cron watchdog (scripts/scale_day_ensure.sh: @reboot and every 5 minutes, for the
# user of this host) keeps the run going after a reboot and exports it when the day is over. It removes its own
# cron entries after the export.
day_watchdog_install() {
    local day="${START%%T*}"
    "${SSH[@]}" "cd '$DEST' && printf 'START=%s\nBEGIN_AT=%s\nDAY=%s\nEXPORT_DIR=%s\n' '$START' '$BEGIN_AT' '$day' '$(dirname "$DEST")/day-export-$day' > data/day_run.env && rm -f data/.day_exported && touch data/.day_armed && cat /proc/sys/kernel/random/boot_id > data/day-boot-id"
    "${SSH[@]}" "( crontab -l 2>/dev/null | grep -v scale_day_ensure; echo '@reboot sleep 90 && $DEST/scripts/scale_day_ensure.sh'; echo '*/5 * * * * $DEST/scripts/scale_day_ensure.sh' ) | crontab - && echo 'watchdog installed in the crontab of this user (@reboot and every 5 min); it removes itself after the export'"
}

day_arm() {
    : "${BEGIN_AT:?set BEGIN_AT=YYYY-MM-DDTHH:MM:SS, the local wall time at which footage $START begins (e.g. the coming midnight)}"
    local stamp; stamp=$(date +%Y%m%d-%H%M%S)
    echo "==> stopping the current run and moving its data to data/_before_$stamp"
    monitor stop || true
    stream stop || true
    mon reset
    "${SSH[@]}" "cd '$DEST' && $DC down 2>&1 | tail -2"
    # (the containers wrote most of these as root, so they are moved from a container; a rename, nothing is copied or deleted)
    "${SSH[@]}" "docker run --rm -v '$DEST/data':/d alpine sh -c 'cd /d && mkdir -p _before_$stamp && for f in $RUN_STATE; do if [ -e \$f ]; then mv \$f _before_$stamp/; fi; done; mkdir -p redis; touch .day_run' && echo archived"
    # persistence for this run's Redis (a sync normally writes this; on the server itself there is no sync)
    "${SSH[@]}" "cd '$DEST' && sed -i '/^REDIS_AOF=/d;/^REDIS_SAVE=/d;/^SCALE_KEEP_RUN_FILES=/d' .env && printf 'REDIS_AOF=no\nREDIS_SAVE=60 1\nSCALE_KEEP_RUN_FILES=1\n' >> .env"
    echo "==> starting: footage $START begins at $BEGIN_AT (black until then); Redis snapshots every 60 s"
    TIMELINE=1 up_all
    echo "==> clean platform history (trucks, visits, journeys, sightings, alerts, raw events)"
    ui reset
    "${SSH[@]}" "docker exec -i ${PROJECT}-ui-postgres-1 psql -U iris -d iris -v ON_ERROR_STOP=1 -c 'TRUNCATE alerts, notifications, feed_events_raw, ingest_batch_receipts, camera_status_intervals, analytics_anomalies RESTART IDENTITY'"
    "${SSH[@]}" "docker restart ${PROJECT}-iris-1 >/dev/null && echo 'iris restarted (feed bridge starts clean)'"
    monitor start
    day_watchdog_install
    echo "armed. Follow it with: $0 day-status"
}

day_status() { "${SSH[@]}" "cd '$DEST' && python3 scripts/scale_day_status.py"; }

# Open an imported run on this machine: no sidecar, no footage players, and the truck feed off so the camera system
# cannot re-send its replayed history into the (imported) platform database.
browse() {
    [ "$LOCAL" = 1 ] || [ "${ON_SERVER:-0}" = 1 ] || { echo "browse is for the laptop (LOCAL=1) or the server itself (ON_SERVER=1)"; exit 2; }
    ui_sync
    sync
    "${SSH[@]}" "cd '$DEST' && sed -i '/^TRUCK_FEED_ENABLED=/d;/^TRACKING_REPLAY=/d' .env && printf 'TRUCK_FEED_ENABLED=0\nTRACKING_REPLAY=0\n' >> .env && rm -f data/.day_run"
    "${SSH[@]}" "cd '$DEST' && $DC up -d --no-build --no-deps redis minio iris ui-postgres ui https-proxy 2>&1 | tail -4"
    mon up
    echo "browsing: https://$PUBHOST:$UI_HTTPS_PORT  (VLM page /iris/vlm, health /grafana/); live streams are offline"
}

case "${1:-sync}" in
    sync) sync ;;
    up) up_all ;;
    day-arm) day_arm ;;
    continue) continue_run ;;
    day-status) day_status ;;
    browse) browse ;;
    monitor) monitor "${2:-start}" ;;
    down) stream stop || true; "${SSH[@]}" "cd '$DEST' && $DC down" ;;
    stream) stream "${2:-status}" ;;
    ui) ui "${2:-up}" ;;
    mon) mon "${2:-up}" ;;
    status) "${SSH[@]}" "cd '$DEST' && $DC ps; nvidia-smi --query-gpu=utilization.gpu,utilization.decoder,memory.used --format=csv,noheader"; stream status ;;
    *) echo "usage: $0 [sync|up|down|stream start|stop|status|ui up|reset|down|mon up|down|reset|day-arm|day-status|browse|continue|monitor start|stop|status]"; exit 2 ;;
esac
