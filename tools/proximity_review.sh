#!/usr/bin/env bash
# The proximity review stack: the proximity pipeline on recorded clips, the app and the dashboard -- isolated from any
# compose stack on the host (its own container names, network, Redis and ports), so it can run beside a live product
# on a shared machine. Driven by `just proximity-review <action>`:
#
#   up [clips]            start everything; clips = "id=/path/a.mp4,id2=/path/b.mp4" (default: the golden clip)
#   down                  stop everything (zones, incidents and clips are kept under data/<name>/)
#   restart [app|ui|all]  restart the app (after a code change), the dashboard, or both with the pipeline
#   status                what runs, where to look, how many incidents
#   logs [pipeline|app|ui]  follow a log
#   reset                 down, then delete data/<name>/ (zones, incidents, recordings)
#
# Settings (environment, all optional): PROXIMITY_REVIEW_NAME (proximity-review), PROXIMITY_REVIEW_REDIS_PORT (16399),
# PROXIMITY_REVIEW_APP_PORT (18093), PROXIMITY_REVIEW_UI_PORT (5193), PROXIMITY_REVIEW_IMAGE (the sidecar image,
# <compose project>-deepstream-sidecar:latest), PROXIMITY_REVIEW_MODELS (perception/models; `just proximity-models`).
# Each camera gets a whole-frame proximity zone the first time; edit it in the dashboard's zone editor.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && source .env; set +a

NAME=${PROXIMITY_REVIEW_NAME:-proximity-review}
DIR=data/$NAME
REDIS_PORT=${PROXIMITY_REVIEW_REDIS_PORT:-16399}
APP_PORT=${PROXIMITY_REVIEW_APP_PORT:-18093}
UI_PORT=${PROXIMITY_REVIEW_UI_PORT:-5193}
IMAGE=${PROXIMITY_REVIEW_IMAGE:-${COMPOSE_PROJECT_NAME:-$(basename "$PWD")}-deepstream-sidecar:latest}
MODELS=${PROXIMITY_REVIEW_MODELS:-$PWD/perception/models}
GOLDEN=$PWD/data/uploads/golden/proximity.mp4

alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

stop_group() {          # the process group a pidfile names (npm run dev forks vite; uv forks python), gone on return
    if alive "$1"; then
        local pg; pg=$(cat "$1")
        kill -- "-$pg" 2>/dev/null || kill "$pg" 2>/dev/null || true
        # an app shutting down closes its port at once but keeps handling alerts until its browsers disconnect:
        # a new app started meanwhile would make every alert twice (the old one on the old code)
        for _ in $(seq 1 30); do kill -0 -- "-$pg" 2>/dev/null || break; sleep 1; done
        kill -9 -- "-$pg" 2>/dev/null || true
    fi
    rm -f "$1"
}

write_configs() {       # $1 = clips spec
    mkdir -p "$DIR"
    python3 - "$1" "$DIR" "$NAME" "$REDIS_PORT" "$PWD" <<'EOF'
import json, os, sys
import yaml
spec, out, name, redis_port, root = sys.argv[1:6]
sources = []
for item in filter(None, spec.split(",")):
    cam, _, path = item.partition("=") if "=" in item else (None, None, item)
    path = os.path.abspath(path)
    cam = cam or os.path.splitext(os.path.basename(path))[0]
    if not os.path.isfile(path):
        sys.exit(f"proximity-review: no such clip {path}")
    sources.append({"id": cam, "uri": f"file://{path}"})
inside = f"/workspace/iris/{out}"
sidecar = {"extends": "../../products/proximity/sidecar.yaml",      # relative: the same file serves pipeline and app
           "redis_url": f"redis://{name}-redis:6379/0", "sources": sources, "file_loop": True, "sink_sync": True,
           "storage_backend": "local", "local_segments_dir": f"{inside}/segments",
           "modules": {"proximity": {"config_path": f"{inside}/zones_config.txt"}}}
yaml.safe_dump(sidecar, open(f"{out}/sidecar.yaml", "w"), sort_keys=False)
zones_file = f"{out}/zones_config.txt"
zones = json.load(open(zones_file)) if os.path.isfile(zones_file) else {}
for s in sources:                                     # a whole-frame proximity zone for a camera seen the first time
    zones.setdefault(s["id"], {"zones": [{"id": "z0", "name": "Whole view",
                                          "points": [[0, 0], [1280, 0], [1280, 720], [0, 720]],
                                          "intrusion": {"mode": "off", "classes": ["person"]},
                                          "overcrowding": {"enabled": False, "threshold": 5, "classes": ["person"]},
                                          "proximity": {"enabled": True}}]})
json.dump(zones, open(zones_file, "w"))
cfg = yaml.safe_load(open(f"{root}/config.yaml"))
cfg["perception"]["backend"] = "deepstream"
cfg["perception"]["redis_url"] = f"redis://127.0.0.1:{redis_port}/0"
cfg.setdefault("segments", {}).update({"enabled": True, "storage_backend": "local", "local_dir": f"{out}/segments"})
cfg.setdefault("alerts", {}).setdefault("clip", {})["enabled"] = True
cfg["zones"]["file_path"] = zones_file
cfg["zones"]["sidecar_config_path"] = f"{root}/{out}/sidecar.yaml"
cfg.setdefault("monitoring", {})["vlm_enabled"] = False
cfg.setdefault("api", {})["api_key"] = cfg.get("api", {}).get("api_key") or "unused"   # the VLM is off
cfg.setdefault("events", {})["db_path"] = f"{out}/events.db"
yaml.safe_dump(cfg, open(f"{out}/config.yaml", "w"), sort_keys=False)
print(" ".join(f"{s['id']}={s['uri'][7:]}" for s in sources))
EOF
}

start_pipeline() {
    docker network inspect "$NAME" >/dev/null 2>&1 || docker network create "$NAME" >/dev/null
    docker ps -q -f "name=^$NAME-redis$" | grep -q . || docker run -d --rm --name "$NAME-redis" --network "$NAME" \
        -p "127.0.0.1:$REDIS_PORT:6379" redis:7-alpine >/dev/null
    local mounts=() d
    for d in $(python3 -c "import yaml,os; print(' '.join(sorted({os.path.dirname(s['uri'][7:]) for s in yaml.safe_load(open('$DIR/sidecar.yaml'))['sources']})))"); do
        case "$d" in "$PWD"*) ;; *) mounts+=(-v "$d:$d:ro") ;; esac
    done
    # the container records as root; the app (this user) prunes recordings past segments.retention_seconds, which it
    # can only do in folders it owns -- in root's folders it could not, and they filled the disk
    for d in $(python3 -c "import yaml; print(' '.join(s['id'] for s in yaml.safe_load(open('$DIR/sidecar.yaml'))['sources']))"); do
        mkdir -p "$DIR/segments/$d" 2>/dev/null || true       # in an older stack's root-owned segments/: warn below
        [ -O "$DIR/segments/$d" ] || echo "proximity-review: $DIR/segments/$d is not yours, its recordings will not" \
            "be pruned (just proximity-review reset)" >&2
    done
    docker run -d --rm --name "$NAME-sidecar" --network "$NAME" --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgcc_s.so.1 --ulimit nofile=65536:65536 --ulimit core=0:0 \
        -v "$PWD:/workspace/iris" -v "$MODELS:/workspace/iris/perception/models" "${mounts[@]}" -w /workspace/iris \
        "$IMAGE" python3 -m perception.engine.main "$DIR/sidecar.yaml" >/dev/null
}

start_app() {
    # the app's own test environment (everything create_app needs) + uvicorn: no VLM or detector extras
    setsid bash -c 'echo $$ > "$0"; exec env -u VIRTUAL_ENV -u SIDECAR_CONFIG IRIS_PRODUCT=iris uv run --quiet \
        --python 3.11 --with-requirements requirements-test.txt --with uvicorn --with python-dotenv --with minio \
        --with openai python tools/proximity_review_app.py "$1" "$2"' "$DIR/app.pid" "$DIR/config.yaml" "$APP_PORT" \
        > "$DIR/app.log" 2>&1 &
}

start_ui() {
    [ -d frontend/node_modules ] || (cd frontend && npm ci)
    (cd frontend && IRIS_HOST_PORT=$APP_PORT ATTENDANCE_FRONTEND_PORT=$UI_PORT setsid bash -c \
        'echo $$ > ../'"$DIR/ui.pid"'; exec npm run dev -- --strictPort' > "../$DIR/ui.log" 2>&1 &)
}

wait_for_app() {
    for _ in $(seq 1 90); do
        curl -sf -o /dev/null "http://127.0.0.1:$APP_PORT/api/cameras" && return 0
        sleep 2
    done
    echo "proximity-review: the app did not answer within 3 minutes -- see $DIR/app.log" >&2; return 1
}

urls() {
    echo "dashboard: http://$(hostname -I | awk '{print $1}'):$UI_PORT  (or http://localhost:$UI_PORT on this machine)"
    echo "app:       http://127.0.0.1:$APP_PORT   data: $DIR/   logs: just proximity-review logs [pipeline|app|ui]"
}

case "${1:-status}" in
up)
    docker image inspect "$IMAGE" >/dev/null 2>&1 \
        || { echo "proximity-review: no sidecar image $IMAGE (set PROXIMITY_REVIEW_IMAGE)" >&2; exit 1; }
    [ -f "$MODELS/DeepStream-Yolo-Seg/yolo26l-seg-736x1280.onnx" ] \
        || { echo "proximity-review: no detector in $MODELS -- run 'just proximity-models' (or set PROXIMITY_REVIEW_MODELS)" >&2; exit 1; }
    cams=$(write_configs "${2:-$GOLDEN}")
    echo "cameras: $cams"
    docker ps -q -f "name=^$NAME-sidecar$" | grep -q . || start_pipeline
    alive "$DIR/app.pid" || start_app
    alive "$DIR/ui.pid" || start_ui
    wait_for_app
    urls
    ;;
down)
    stop_group "$DIR/ui.pid"; stop_group "$DIR/app.pid"
    docker rm -f "$NAME-sidecar" "$NAME-redis" >/dev/null 2>&1 || true
    docker network rm "$NAME" >/dev/null 2>&1 || true
    echo "proximity-review: stopped (zones, incidents and clips kept in $DIR/)"
    ;;
restart)
    case "${2:-app}" in
    app) stop_group "$DIR/app.pid"; start_app; wait_for_app ;;
    ui) stop_group "$DIR/ui.pid"; start_ui ;;
    all) "$0" down; "$0" up "$(python3 -c "import yaml; print(','.join(s['id'] + '=' + s['uri'][7:] for s in yaml.safe_load(open('$DIR/sidecar.yaml'))['sources']))")" ;;
    *) echo "restart app|ui|all" >&2; exit 2 ;;
    esac
    urls
    ;;
status)
    docker ps -a -f "name=^$NAME-" --format '{{.Names}}: {{.Status}}'
    for p in app ui; do alive "$DIR/$p.pid" && echo "$p: running (pid $(cat "$DIR/$p.pid"))" || echo "$p: stopped"; done
    if curl -sf -o /dev/null "http://127.0.0.1:$APP_PORT/api/cameras"; then
        curl -s "http://127.0.0.1:$APP_PORT/api/events?limit=200" | python3 -c "
import json, sys
evs = json.load(sys.stdin); evs = evs if isinstance(evs, list) else evs.get('events', [])
prox = [e for e in evs if 'proximity' in (e.get('tags') or [])]
by = {}
for e in prox: by[e['incident_status'] or '-'] = by.get(e['incident_status'] or '-', 0) + 1
print(f'proximity incidents (last 200 events): {len(prox)} {by}')"
        urls
    fi
    ;;
logs)
    case "${2:-pipeline}" in
    pipeline) docker logs -f --tail 50 "$NAME-sidecar" ;;
    app|ui) tail -n 50 -f "$DIR/$2.log" ;;
    *) echo "logs pipeline|app|ui" >&2; exit 2 ;;
    esac
    ;;
reset)
    "$0" down
    # the pipeline container wrote its recordings as root
    docker run --rm -v "$PWD/data:/d" alpine rm -rf "/d/$NAME" && echo "proximity-review: $DIR/ deleted"
    ;;
*)
    sed -n '2,20p' "$0"; exit 2
    ;;
esac
