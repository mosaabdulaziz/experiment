#!/usr/bin/env bash
# The central MLflow server on cmubuntu: rclone mount (artifacts on SharePoint) -> mount check -> Postgres (Docker)
# -> mlflow server (host process, user vlm). See README.md beside this file.
#
#   serverctl.sh start      mount, check, Postgres, server (idempotent: a running piece is left alone)
#   serverctl.sh stop       stop the server               stop --all: also unmount and stop Postgres
#                           (either leaves a hold: the watchdog does not restart what an operator stopped)
#   serverctl.sh ensure     the watchdog (cron, every 2 min): a running server whose mount fails its check is
#                           stopped at once; a stopped server is started again (which refuses without a good mount),
#                           unless held by `stop`
#   serverctl.sh boot       @reboot: wait for Docker and the network, then start (a reboot clears the hold)
#   serverctl.sh server     start only the server (never mounts; refuses unless the mount check passes)
#   serverctl.sh check      the mount check alone
#   serverctl.sh status     mount, server /health, Postgres, pending/in-progress uploads to SharePoint, cache size
#                           (exit 0: all up and nothing waiting to upload; 1: something down; 3: uploads pending)
#   serverctl.sh create-user <name>   a non-admin user, its credentials written to secrets/client-<name>.env
#
# The server never starts on an unmounted artifact folder: `mountpoint -q` AND a write/read/delete round trip through
# the mount must pass first. And even a server that outlives its mount cannot fill the local disk: MLflow writes under
# artifacts/store/, which on SharePoint is a folder but in the unmounted local folder is a read-only regular FILE (the
# guard), so every artifact write there fails ("Not a directory") instead of landing on the local disk.
set -euo pipefail

IRIS_MLFLOW_HOME="${IRIS_MLFLOW_HOME:-$HOME/mlflow-server}"
# Non-secret settings, overridable in $IRIS_MLFLOW_HOME/config.env (written by install.sh).
MLFLOW_HOST=10.2.0.50                     # the LAN interface (enp6s18), not 0.0.0.0
MLFLOW_PORT=15050
MLFLOW_WORKERS=4
PG_PORT=15432                             # 127.0.0.1 only (docker-compose.yml)
REMOTE=iris-onedrive:IRIS/repo-data/mlflow-artifacts
VFS_CACHE_MAX_SIZE=3G
VFS_CACHE_MAX_AGE=1h
ALLOWED_HOSTS="10.2.0.50,10.2.0.50:15050,cmubuntu,cmubuntu:15050,cmubuntu.cairomotive.com,cmubuntu.cairomotive.com:15050,localhost:*,127.0.0.1:*"
RCLONE="$HOME/.local/bin/rclone"
# shellcheck disable=SC1091
[ -f "$IRIS_MLFLOW_HOME/config.env" ] && . "$IRIS_MLFLOW_HOME/config.env"

APP="$IRIS_MLFLOW_HOME/app"
MOUNT="$IRIS_MLFLOW_HOME/artifacts"
STORE="$MOUNT/store"                      # --artifacts-destination: a folder on SharePoint, a guard file locally
CACHE="$IRIS_MLFLOW_HOME/rclone-cache"
SECRETS="$IRIS_MLFLOW_HOME/secrets"
LOGS="$IRIS_MLFLOW_HOME/logs"
RUN="$IRIS_MLFLOW_HOME/run"
PIDFILE="$RUN/mlflow-server.pid"
HOLD="$RUN/held-by-operator"
RCDIR="$RUN/rc"                           # mode 700: only vlm can talk to the mount's remote control
RCSOCK="$RCDIR/rclone.sock"               # rclone rc over a unix socket, never a TCP port
VENV="$IRIS_MLFLOW_HOME/venv"
COMPOSE=(docker compose -f "$APP/docker-compose.yml" --project-name iris-mlflow)
export IRIS_MLFLOW_HOME PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin"

mkdir -p "$LOGS" "$RUN"
log() { printf '%s [serverctl %s] %s\n' "$(date -Is)" "$CMD" "$*" | tee -a "$LOGS/serverctl.log" >&2; }
die() { log "REFUSED: $*"; exit 1; }

# -- the artifact mount ----------------------------------------------------------------------------------------
mounted() { mountpoint -q "$MOUNT"; }
# A mount-table entry that does not answer: rclone died ("Transport endpoint is not connected"). mountpoint -q says
# "not mounted" for it, yet nothing can be created there until it is lazily unmounted.
dead_mount() { ! mounted && findmnt -rn -M "$MOUNT" >/dev/null 2>&1; }
clear_dead() { if dead_mount; then log "clearing the dead FUSE mount on $MOUNT"; fusermount3 -uz "$MOUNT"; fi; }

guard() {   # the unmounted folder's read-only file named like the artifact folder
    mounted && return 0
    [ -f "$STORE" ] || { rm -rf "$STORE" 2>/dev/null; : > "$STORE"; }
    chmod 444 "$STORE"
}

do_mount() {
    if mounted; then return 0; fi
    clear_dead
    mkdir -p "$MOUNT" "$CACHE" "$RCDIR"
    chmod 700 "$RCDIR"; rm -f "$RCSOCK"
    chmod 755 "$MOUNT"                        # fusermount3 needs write access to the mountpoint itself
    guard
    log "mounting $REMOTE on $MOUNT (vfs cache $CACHE, max $VFS_CACHE_MAX_SIZE)"
    "$RCLONE" mount "$REMOTE" "$MOUNT" --daemon --allow-non-empty \
        --vfs-cache-mode writes --cache-dir "$CACHE" \
        --vfs-cache-max-size "$VFS_CACHE_MAX_SIZE" --vfs-cache-max-age "$VFS_CACHE_MAX_AGE" \
        --dir-cache-time 30s --transfers 4 \
        --rc --rc-addr "unix://$RCSOCK" \
        --log-file "$LOGS/rclone.log" --log-level INFO \
        </dev/null >>"$LOGS/rclone.log" 2>&1 9>&-     # the daemon keeps no fd of ours: not the lock (9), not a pipe
    for _ in $(seq 1 40); do mounted && return 0; sleep 0.5; done
    die "the rclone mount of $REMOTE did not come up on $MOUNT (see $LOGS/rclone.log)"
}

# The safety check: a real mountpoint of the expected remote, and a file that round-trips through it.
check_mount() {
    mountpoint -q "$MOUNT" \
        || { echo "$MOUNT is not a mountpoint: the rclone mount of $REMOTE is not up" >&2; return 1; }
    local source
    source="$(findmnt -n -o SOURCE --target "$MOUNT" 2>/dev/null || true)"
    [ "$source" = "${REMOTE}" ] || [ "$source" = "${REMOTE#*:}" ] || [[ "$source" == "${REMOTE%%:*}:"* ]] \
        || { echo "$MOUNT is mounted from '$source', not $REMOTE" >&2; return 1; }
    [ -d "$STORE" ] || { echo "$STORE is not a folder: the mount does not show $REMOTE/store" >&2; return 1; }
    local probe="$STORE/.mount-check-$(hostname -s)-$$" token
    token="$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')"
    if ! timeout 30 sh -c 'printf %s "$1" > "$2"' _ "$token" "$probe" 2>/dev/null; then
        echo "cannot write a test file into $MOUNT (the mount is not usable)" >&2; return 1
    fi
    if [ "$(timeout 30 cat "$probe" 2>/dev/null)" != "$token" ]; then
        timeout 10 rm -f "$probe" 2>/dev/null || true
        echo "a test file written into $MOUNT did not read back (the mount is not usable)" >&2; return 1
    fi
    timeout 30 rm -f "$probe" 2>/dev/null && [ ! -e "$probe" ] \
        || { echo "cannot delete a test file from $MOUNT (the mount is not usable)" >&2; return 1; }
}

do_check() {
    local why
    if ! why="$(check_mount 2>&1)"; then
        die "$why. The MLflow server does not start without it: its artifacts would land in the local folder and fill the disk. Run: $0 start (it mounts first), or check '$RCLONE lsf $REMOTE' and $LOGS/rclone.log"
    fi
    log "mount check passed: $MOUNT is $REMOTE, write/read/delete round-trip in $STORE ok"
}

do_unmount() {
    if mounted; then
        log "unmounting $MOUNT"
        fusermount3 -u "$MOUNT" 2>/dev/null || fusermount3 -uz "$MOUNT"
    fi
    clear_dead
    guard
}

# -- Postgres -----------------------------------------------------------------------------------------------------
do_postgres() {
    "${COMPOSE[@]}" up -d --quiet-pull postgres >>"$LOGS/serverctl.log" 2>&1 9>&- \
        || die "docker compose could not start Postgres (see $LOGS/serverctl.log)"
    for _ in $(seq 1 60); do
        [ "$(docker inspect -f '{{.State.Health.Status}}' iris-mlflow-postgres 2>/dev/null)" = healthy ] && return 0
        sleep 1
    done
    die "Postgres (iris-mlflow-postgres) did not become healthy in 60 s"
}

# -- the server ---------------------------------------------------------------------------------------------------
server_pid() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null && cat "$PIDFILE"; }
healthy() { curl -fsS -m 5 "http://$MLFLOW_HOST:$MLFLOW_PORT/health" >/dev/null 2>&1; }

do_server() {
    if server_pid >/dev/null; then log "server already running (pid $(server_pid))"; return 0; fi
    do_check                                  # the last word before anything can write artifacts
    log "starting mlflow server on http://$MLFLOW_HOST:$MLFLOW_PORT (artifacts -> $STORE)"
    # No password on the command line or in a URI: libpq reads it from secrets/pgpass (mode 600); the CSRF key of the
    # basic-auth app comes from secrets/server.env (stable across restarts and workers).
    set -a; . "$SECRETS/server.env"; set +a
    PGPASSFILE="$SECRETS/pgpass" \
    MLFLOW_AUTH_CONFIG_PATH="$SECRETS/basic_auth.ini" \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    setsid nohup "$VENV/bin/mlflow" server \
        --app-name basic-auth \
        --host "$MLFLOW_HOST" --port "$MLFLOW_PORT" --workers "$MLFLOW_WORKERS" \
        --backend-store-uri "postgresql://mlflow@127.0.0.1:$PG_PORT/mlflow" \
        --serve-artifacts --artifacts-destination "$STORE" \
        --allowed-hosts "$ALLOWED_HOSTS" \
        >>"$LOGS/mlflow-server.log" 2>&1 < /dev/null 9>&- &
    echo $! > "$PIDFILE"
    for _ in $(seq 1 90); do
        healthy && { log "server up (pid $(cat "$PIDFILE"))"; return 0; }
        kill -0 "$(cat "$PIDFILE")" 2>/dev/null || die "mlflow server exited at startup (see $LOGS/mlflow-server.log)"
        sleep 1
    done
    die "mlflow server did not answer /health in 90 s (see $LOGS/mlflow-server.log)"
}

stop_server() {
    local pid
    if pid="$(server_pid)"; then
        log "stopping mlflow server (pid $pid)"
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
        for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -0 "$pid" 2>/dev/null && { kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid"; }
    fi
    rm -f "$PIDFILE"
}

# -- status -------------------------------------------------------------------------------------------------------
# Uploads: the mount's `vfs/stats` over its rc socket (uploadsInProgress / uploadsQueued / erroredFiles). A mount
# started without --rc (before this was added) has no socket: then count the cache entries rclone marks Dirty (written
# locally, not yet uploaded) in its metadata, which is what vfs/stats counts too, minus the in-progress split.
upload_status() {   # prints "<pending> <text>"
    local json
    if [ -S "$RCSOCK" ] && json="$(timeout 10 "$RCLONE" rc --unix-socket "$RCSOCK" vfs/stats 2>/dev/null)"; then
        printf '%s' "$json" | python3 -c '
import json, sys
d = json.load(sys.stdin).get("diskCache", {})
busy, queued, err = d.get("uploadsInProgress", 0), d.get("uploadsQueued", 0), d.get("erroredFiles", 0)
pending = busy + queued
text = "none pending" if pending == 0 else f"{busy} in progress, {queued} queued"
if err: text += f", {err} ERRORED (see logs/rclone.log)"
files = d.get("files", 0)
print(pending + err, f"{text}  [rclone vfs/stats; {files} file(s) in the cache]")'
        return
    fi
    local n
    n="$(grep -rlE '"Dirty": *true' "$CACHE/vfsMeta" 2>/dev/null | wc -l)"
    if mounted; then
        echo "$n $( [ "$n" = 0 ] && echo "none pending" || echo "$n file(s) not yet uploaded") [no rc socket: from the cache metadata]"
    else
        echo "$n $( [ "$n" = 0 ] && echo "none pending" || echo "$n file(s) not yet uploaded: they upload once the mount is back") [mount down: from the cache metadata]"
    fi
}

do_status() {
    local rc=0 up pending
    if mounted; then echo "mount:    up ($REMOTE) on $MOUNT"
    else rc=1; echo "mount:    $(dead_mount && echo "DEAD (rclone gone)" || echo "NOT mounted") on $MOUNT"; fi
    if healthy; then echo "server:   /health OK at http://$MLFLOW_HOST:$MLFLOW_PORT ($(server_pid >/dev/null && echo "pid $(server_pid)" || echo "no pid file"))"
    else rc=1; echo "server:   /health NOT answering at http://$MLFLOW_HOST:$MLFLOW_PORT ($(server_pid >/dev/null && echo "pid $(server_pid)" || echo stopped))"; fi
    local pg; pg="$(docker inspect -f '{{.State.Status}} ({{.State.Health.Status}})' iris-mlflow-postgres 2>/dev/null || echo absent)"
    [ "$pg" = "running (healthy)" ] && echo "postgres: up, $pg" || { rc=1; echo "postgres: DOWN, $pg"; }
    up="$(upload_status)"; pending="${up%% *}"
    echo "uploads:  ${up#* }"
    echo "cache:    $(du -sh "$CACHE" 2>/dev/null | cut -f1) on disk in $CACHE (max $VFS_CACHE_MAX_SIZE, more during one bigger upload)"
    if [ "$rc" = 0 ] && [ "$pending" != 0 ]; then rc=3; fi
    [ "$pending" != 0 ] && echo "=> uploads pending: do not shut down or reboot until this says none pending"
    return "$rc"
}

with_lock() { exec 9>"$RUN/serverctl.lock"; flock -w 300 9 || die "another serverctl is running"; }

CMD="${1:-status}"
case "$CMD" in
    start) with_lock; rm -f "$HOLD"; do_mount; do_postgres; do_server ;;      # do_server runs the mount check
    stop)
        with_lock; date -Is > "$HOLD"; stop_server
        if [ "${2:-}" = --all ]; then do_unmount; "${COMPOSE[@]}" stop postgres >>"$LOGS/serverctl.log" 2>&1; log "stopped all"; fi ;;
    ensure)
        with_lock
        if [ -f "$HOLD" ] && ! server_pid >/dev/null; then exit 0; fi       # stopped on purpose: leave it
        if server_pid >/dev/null && ! why="$(check_mount 2>&1)"; then
            log "mount failed its check while the server runs ($why): stopping the server"
            stop_server
            mounted && ! check_mount 2>/dev/null && do_unmount     # mounted but unusable: unmount to remount
            clear_dead
        fi
        if ! server_pid >/dev/null || ! healthy; then
            server_pid >/dev/null && { log "server not healthy: restarting"; stop_server; }
            do_mount; do_postgres; do_server
        fi ;;
    boot)
        for _ in $(seq 1 60); do docker info >/dev/null 2>&1 && "$RCLONE" lsf --max-depth 1 "${REMOTE%/*}" >/dev/null 2>&1 && break; sleep 5; done
        rm -f "$HOLD"; "$0" start ;;
    server) with_lock; rm -f "$HOLD"; do_postgres; do_server ;;   # the server step alone: never mounts, only checks
    check) do_check ;;
    mount) with_lock; do_mount; do_check ;;
    unmount) with_lock; server_pid >/dev/null && die "stop the server first"; do_unmount ;;
    status) do_status ;;
    create-user)
        name="${2:?usage: serverctl.sh create-user <name>}"
        pw="$(head -c 24 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 24)"
        admin_pw="$(sed -n 's/^admin_password *= *//p' "$SECRETS/basic_auth.ini")"
        # credentials through curl's stdin config, never on a command line (`ps` shows those to every local user)
        code="$(printf 'user = "admin:%s"\ndata = "{\\"username\\": \\"%s\\", \\"password\\": \\"%s\\"}"\n' \
                    "$admin_pw" "$name" "$pw" \
                | curl -sS -K - -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' \
                    "http://$MLFLOW_HOST:$MLFLOW_PORT/api/2.0/mlflow/users/create")"
        [ "$code" = 200 ] || die "creating user $name failed (HTTP $code)"
        umask 077
        { echo "MLFLOW_TRACKING_URI=http://$MLFLOW_HOST:$MLFLOW_PORT"; echo "MLFLOW_TRACKING_USERNAME=$name"
          echo "MLFLOW_TRACKING_PASSWORD=$pw"; } > "$SECRETS/client-$name.env"
        log "created user $name; its credentials are in $SECRETS/client-$name.env" ;;
    *) echo "usage: $0 start|stop [--all]|ensure|boot|server|check|mount|unmount|status|create-user <name>" >&2; exit 2 ;;
esac
