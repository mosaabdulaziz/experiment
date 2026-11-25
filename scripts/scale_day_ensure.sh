#!/usr/bin/env bash
# Keeps an armed full-day run going and finishes it. Idempotent; runs on the server from cron (@reboot and every
# 5 minutes, installed by `scale13_deploy.sh day-arm`), so a hard reset of the machine costs only its downtime.
#
# Reads data/day_run.env (START, BEGIN_AT, DAY, EXPORT_DIR written by day-arm). From the arming until 10 minutes
# after the footage ends it makes sure that: the stack is up, the footage players run on the ORIGINAL footage clock
# (they rejoin it where it is now), the health monitor runs, and, once after a reboot, the global-id counter is
# moved forward (Redis snapshots are up to 60 s old; ids handed out after the last one must never be handed out
# again) and the sidecar restarted to read it. After the end it runs the export once and removes its cron entries.
# Everything it does is logged to data/day-ensure.log.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
RUN_DIR="$PWD"
exec 9>/tmp/scale-day-ensure.lock; flock -n 9 || exit 0
# everything started below must not inherit the lock (fd 9), or the long-running players would hold it for good
# and every later run of this script would find it taken: each launch ends with 9>&-
[ -f data/day_run.env ] && [ -f data/.day_armed ] || exit 0
# shellcheck disable=SC1091
. data/day_run.env
LOG="$RUN_DIR/data/day-ensure.log"
log() { echo "[$(date +%FT%T)] $*" >> "$LOG"; }
DC="docker compose -p scale -f docker-compose.yml -f docker-compose.scale.yml -f docker-compose.scale-ui.yml -f docker-compose.scale-monitoring.yml"
DEPLOY="env ON_SERVER=1 START=$START BEGIN_AT=$BEGIN_AT TIMELINE=1 $RUN_DIR/scripts/scale13_deploy.sh"

now=$(date +%s)
begin=$(date -d "$BEGIN_AT" +%s)
start_s=$(date -d "$START" +%s); day0_s=$(date -d "${START%%T*}" +%s)
end=$((begin + 86400 - (start_s - day0_s)))      # when the last footage second is played

if [ "$now" -ge $((end + 600)) ]; then
    [ -f data/.day_exported ] && exit 0
    log "the day is over: exporting to $EXPORT_DIR"
    if WAIT=120 "$RUN_DIR/scripts/scale_day_export.sh" "$EXPORT_DIR" >> "$LOG" 2>&1 9>&-; then
        touch data/.day_exported; log "export done"
        crontab -l 2>/dev/null | grep -v scale_day_ensure | crontab - ; log "cron entries removed"
    else
        log "EXPORT FAILED (will be tried again in 5 minutes)"
    fi
    exit 0
fi

# 1. the stack: start whatever is not running (restart policies normally did it after a reboot)
want="redis minio go2rtc ui-postgres ui https-proxy iris deepstream-sidecar"
down=""
for s in $want; do
    [ "$(docker inspect -f '{{.State.Running}}' scale-$s-1 2>/dev/null)" = true ] || down="$down $s"
done
if [ -n "$down" ]; then log "not running:$down: starting them"; $DC up -d --no-build $down >> "$LOG" 2>&1 9>&-; fi

# 2. after a reboot: move the id counter forward, restart the sidecar once so it reads it
boot=$(cat /proc/sys/kernel/random/boot_id)
if [ "$(cat data/day-boot-id 2>/dev/null)" != "$boot" ]; then
    for _ in $(seq 30); do docker exec scale-redis-1 redis-cli ping 2>/dev/null | grep -q PONG && break; sleep 2; done
    if docker exec scale-redis-1 redis-cli ping 2>/dev/null | grep -q PONG; then
        n=$(docker exec scale-redis-1 redis-cli INCRBY iris:mtmc:next_gid 500 | tr -d '\r')
        log "new boot ($boot): next global id moved to $n; restarting the sidecar"
        docker restart scale-deepstream-sidecar-1 >> "$LOG" 2>&1 9>&-
        echo "$boot" > data/day-boot-id
    else
        log "new boot but Redis does not answer: cannot move the id counter yet (tries again in 5 minutes)"
    fi
fi

# 3. the footage players (on the original clock) and the monitor
export SCALE13_STATE="$RUN_DIR/data/day-player"
sup=$(cat "$SCALE13_STATE/supervisor.pid" 2>/dev/null)
if [ -z "$sup" ] || [ ! -d "/proc/$sup" ]; then
    log "footage players not running: starting them on the footage clock"
    $DEPLOY stream start >> "$LOG" 2>&1 9>&-
fi
mp=$(cat /tmp/scale13-monitor.pid 2>/dev/null)
if [ -z "$mp" ] || [ ! -d "/proc/$mp" ]; then
    log "monitor not running: starting it"
    $DEPLOY monitor start >> "$LOG" 2>&1 9>&-
fi
exit 0
