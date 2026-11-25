#!/usr/bin/env bash
# Run / stop / reset the truck-tracking web UI that sits beside this stack.
#
#   scripts/truck_ui.sh up       start its database, migrate, provision our cameras, run API + web
#   scripts/truck_ui.sh down     stop the dev servers, CLEAR its tracked trucks/history, stop its database
#   scripts/truck_ui.sh reset    clear its tracked trucks/history (leave it running)
#   scripts/truck_ui.sh status
#
# Wired into `just down` (down), `just multi` (up, after the stack is ready) and the
# `just truck-ui*` recipes. The repo is found at $TRUCK_UI_DIR (default ../truck-tracking);
# when it isn't there every command is a no-op, so checkouts without it are unaffected.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
set -a; [ -f .env ] && source .env; set +a

UI_DIR="${TRUCK_UI_DIR:-$REPO_DIR/../truck-tracking}"
if [ ! -d "$UI_DIR/apps/web" ]; then
    echo "truck-ui: $UI_DIR is not a truck-tracking checkout -- skipping (set TRUCK_UI_DIR)"
    exit 0
fi
UI_DIR="$(cd "$UI_DIR" && pwd)"
COMPOSE="$UI_DIR/infra/docker/docker-compose.yml"
PG=iris-postgres
RUN_DIR="$REPO_DIR/.run"
PIDFILE="$RUN_DIR/truck_ui.pid"
LOG="$RUN_DIR/truck_ui.log"
BIN="$RUN_DIR/bin"
API_PORT=3100
WEB_PORT=5173
REDIS_PORT="${REDIS_HOST_PORT:-6379}"
DB_ADMIN="postgres://iris:iris@localhost:5432/iris"
mkdir -p "$RUN_DIR" "$BIN" "$REPO_DIR/data"

# pnpm comes through corepack; nested `pnpm` calls (turbo, the db scripts) need it on PATH.
printf '#!/bin/sh\nexec corepack pnpm "$@"\n' > "$BIN/pnpm"; chmod +x "$BIN/pnpm"
export PATH="$BIN:$PATH" COREPACK_ENABLE_DOWNLOAD_PROMPT=0

psql_ui() { docker exec -i "$PG" psql -U iris -d iris -At -v ON_ERROR_STOP=1 "$@"; }
pg_running() { [ "$(docker inspect -f '{{.State.Running}}' "$PG" 2>/dev/null)" = "true" ]; }

# Our company id as the feed bridge sends it (placeholder labels map to a stable UUID).
company_id() {
    python3 - <<'PY' 2>/dev/null
import sys, yaml
sys.path.insert(0, ".")
from app.truck_feed import tenant_uuid
for path in ("config.yaml", "configs/presets/multicam/config.yaml"):
    try:
        tf = (yaml.safe_load(open(path)) or {}).get("truck_feed") or {}
    except OSError:
        continue
    if tf.get("company_id"):
        print(tenant_uuid(str(tf["company_id"])))
        break
PY
}

listener_pid() { ss -ltnp 2>/dev/null | awk -v p=":$1" '$4 ~ p"$" {print}' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2; }

stop_dev() {
    local stopped=0
    if [ -f "$PIDFILE" ]; then
        local pid; pid=$(cat "$PIDFILE")
        kill -TERM "-$pid" 2>/dev/null && stopped=1
        rm -f "$PIDFILE"
    fi
    # dev servers started some other way: match on this UI checkout's own processes only
    for port in $API_PORT $WEB_PORT; do
        local pid; pid=$(listener_pid "$port")
        if [ -n "$pid" ] && [ "$(readlink "/proc/$pid/cwd" 2>/dev/null)" != "" ] \
           && readlink "/proc/$pid/cwd" | grep -q "^$UI_DIR"; then
            kill -TERM "$pid" 2>/dev/null && stopped=1
        fi
    done
    if [ "$stopped" = 1 ]; then
        for _ in $(seq 20); do
            [ -z "$(listener_pid $API_PORT)$(listener_pid $WEB_PORT)" ] && break
            sleep 0.5
        done
        echo "truck-ui: stopped the dev servers"
    else
        echo "truck-ui: dev servers were not running"
    fi
}

reset_history() {
    if ! pg_running; then echo "truck-ui: database is not running -- nothing to clear"; return 0; fi
    local company; company=$(company_id)
    if [ -z "$company" ]; then echo "truck-ui: no truck_feed.company_id in config -- nothing to clear"; return 0; fi
    psql_ui -v drop_trucks=1 -v company="'$company'" < "$REPO_DIR/scripts/reset_truck_tracking_history.sql" \
        > /dev/null && echo "truck-ui: cleared tracked trucks and history for company $company" \
        || echo "truck-ui: WARNING could not clear the history"
}

cmd_down() {
    stop_dev
    reset_history
    if pg_running; then
        docker compose -f "$COMPOSE" stop postgres > /dev/null 2>&1 && echo "truck-ui: database stopped (data kept)"
    fi
}

cmd_up() {
    # nothing may still be running from an earlier start
    stop_dev > /dev/null
    echo "truck-ui: starting ($UI_DIR, branch $(git -C "$UI_DIR" branch --show-current))"

    pg_out=$(docker compose -f "$COMPOSE" up -d --no-deps postgres 2>&1) \
        || { echo "truck-ui: could not start the database:"; echo "$pg_out" | tail -4 | sed 's/^/    /'; return 1; }
    for _ in $(seq 60); do
        [ "$(docker inspect -f '{{.State.Health.Status}}' "$PG" 2>/dev/null)" = "healthy" ] && break
        sleep 2
    done

    [ -f "$UI_DIR/.env" ] || cp "$UI_DIR/.env.example" "$UI_DIR/.env"
    # Its Redis is db 1 of THIS stack's Redis: where the feed bridge publishes (truck_feed.redis_url).
    sed -i "s#^REDIS_URL=.*#REDIS_URL=redis://localhost:${REDIS_PORT}/1#" "$UI_DIR/.env"

    cd "$UI_DIR" || return 1
    [ -d node_modules ] || { echo "truck-ui: installing dependencies..."; pnpm install > "$LOG" 2>&1 || { echo "truck-ui: pnpm install failed, see $LOG"; return 1; }; }
    [ -f packages/shared/dist/index.js ] || { echo "truck-ui: building shared packages..."; pnpm turbo run build --filter='./packages/*' >> "$LOG" 2>&1 || { echo "truck-ui: build failed, see $LOG"; return 1; }; }

    DATABASE_ADMIN_URL="$DB_ADMIN" pnpm db:migrate >> "$LOG" 2>&1 || { echo "truck-ui: migrations failed, see $LOG"; return 1; }
    # The demo login. The seed makes a NEW company each run, so only when there's no admin yet.
    if [ "$(psql_ui -c "select count(*) from users where email='admin@example.com'")" = "0" ]; then
        DATABASE_ADMIN_URL="$DB_ADMIN" pnpm db:seed >> "$LOG" 2>&1
    fi

    cd "$REPO_DIR" || return 1
    if [ -n "$(company_id)" ]; then
        python3 scripts/provision_truck_tracking.py > /dev/null 2>&1 \
            && echo "truck-ui: provisioned company/stops/cameras" \
            || echo "truck-ui: WARNING provisioning failed (is config.yaml from the multicam preset?)"
    else
        echo "truck-ui: no truck_feed config -- run \`just multi\` first so cameras can be provisioned"
    fi

    cd "$UI_DIR" || return 1
    : > "$LOG"
    setsid nohup bash -c "exec pnpm dev" >> "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    for _ in $(seq 90); do
        [ -n "$(listener_pid $API_PORT)" ] && [ -n "$(listener_pid $WEB_PORT)" ] && break
        sleep 1
    done
    if [ -n "$(listener_pid $API_PORT)" ] && [ -n "$(listener_pid $WEB_PORT)" ]; then
        echo "truck-ui: ready  ->  http://localhost:$WEB_PORT  (admin@example.com / Admin123!)   log: $LOG"
    else
        echo "truck-ui: WARNING the servers did not come up in time, see $LOG"
        return 1
    fi
}

cmd_status() {
    pg_running && echo "database : running" || echo "database : stopped"
    [ -n "$(listener_pid $API_PORT)" ] && echo "api      : up (:$API_PORT)" || echo "api      : down"
    [ -n "$(listener_pid $WEB_PORT)" ] && echo "web      : up (http://localhost:$WEB_PORT)" || echo "web      : down"
    if pg_running; then
        echo "trucks   : $(psql_ui -c 'select count(*) from trucks')"
    fi
}

case "${1:-}" in
    up) cmd_up ;;
    down) cmd_down ;;
    reset) reset_history ;;
    status) cmd_status ;;
    *) echo "usage: $0 up|down|reset|status"; exit 2 ;;
esac
