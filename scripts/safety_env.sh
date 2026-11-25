# Settings of the safety stack (`just safety-*`, docker-compose.safety.yml). Sourced by those recipes, not run.
# Each SAFETY_* setting comes from the environment, else from .env, else the default below (.env.example lists them).
# Only the SAFETY_* lines of .env are read: sourcing all of it would give this stack the main stack's project name
# and ports.
_safety_setting() {
    local v="${!1-}"
    if [[ -z "$v" && -f .env ]]; then
        v="$(sed -nE "s/^[[:space:]]*(export[[:space:]]+)?$1=([^#]*).*/\2/p" .env | tail -n1)"
        v="$(sed -E 's/[[:space:]]+$//; s/^"(.*)"$/\1/; s/^'\''(.*)'\''$/\1/' <<< "$v")"
    fi
    printf -v "$1" '%s' "${v:-$2}"
    export "$1"
}
_safety_setting SAFETY_COMPOSE_PROJECT safety
_safety_setting SAFETY_DIR scratch/safety                # the stack's data dir (mounted at data/ in its containers)
_safety_setting SAFETY_CAMERAS products/safety/cameras.yaml
_safety_setting SAFETY_DEMO_CLIPS /data/IRIS/data/safety-demo   # the demo cameras' `clip:` files
# The models the role sidecars load; a fresh worktree has only the DVC pointers (`just data-pull`), so this can name
# another checkout's perception/models instead.
_safety_setting SAFETY_MODELS_DIR perception/models
# A DVC store on this machine (a local remote) `just safety-models` copies the models from; else it runs dvc pull.
_safety_setting SAFETY_DVC_STORE /data/IRIS/repo-data/dvc-store
# Absolute: a relative bind-mount source without ./ is a named volume to compose
SAFETY_DIR="$(realpath -m "$SAFETY_DIR")"
SAFETY_MODELS_DIR="$(realpath -m "$SAFETY_MODELS_DIR")"
# Host ports, beside every other stack on the machine
_safety_setting SAFETY_IRIS_PORT 58000
_safety_setting SAFETY_REDIS_PORT 56379
_safety_setting SAFETY_GO2RTC_RTSP_PORT 58554
_safety_setting SAFETY_GO2RTC_API_PORT 51984
_safety_setting SAFETY_GO2RTC_WEBRTC_PORT 58555
_safety_setting SAFETY_MINIO_API_PORT 59000
_safety_setting SAFETY_MINIO_CONSOLE_PORT 59001
_safety_setting SAFETY_FALL_PORT 58014
# The sidecar image (any image built from perception/pipeline/Dockerfile: the code is mounted, not baked in)
_safety_setting SAFETY_SIDECAR_IMAGE "$SAFETY_COMPOSE_PROJECT-deepstream-sidecar:latest"
SAFETY_IRIS_IMAGE="$SAFETY_COMPOSE_PROJECT-iris:latest"
# The sidecars and the app run as the user who runs the recipe: files they write stay theirs
export SAFETY_UID="$(id -u)" SAFETY_GID="$(id -g)"

source "$(dirname "${BASH_SOURCE[0]}")/safety_pids.sh"   # safety_pid / safety_stop: the host processes' pid files

safety_compose_cmd() {
    printf '%s' "docker compose -p $SAFETY_COMPOSE_PROJECT -f docker-compose.yml -f docker-compose.safety.yml"
}

# The variables docker-compose.yml interpolates, set to this stack's values (shared by compose and the controller).
_safety_compose_env() {
    printf '%s\n' "COMPOSE_PROJECT_NAME=$SAFETY_COMPOSE_PROJECT" "IRIS_HOST_PORT=$SAFETY_IRIS_PORT" \
        "REDIS_HOST_PORT=$SAFETY_REDIS_PORT" "GO2RTC_RTSP_PORT=$SAFETY_GO2RTC_RTSP_PORT" \
        "GO2RTC_API_PORT=$SAFETY_GO2RTC_API_PORT" "GO2RTC_WEBRTC_PORT=$SAFETY_GO2RTC_WEBRTC_PORT" \
        "MINIO_API_HOST_PORT=$SAFETY_MINIO_API_PORT" "MINIO_CONSOLE_HOST_PORT=$SAFETY_MINIO_CONSOLE_PORT"
}

safety_compose() {
    local vars; mapfile -t vars < <(_safety_compose_env)
    env "${vars[@]}" $(safety_compose_cmd) "$@"
}

# The controller (tools/safety_controller.py), with this stack's compose command (the same env as safety_compose),
# data dir, cameras and clips; its PyYAML and redis come from the locked `safety` group.
safety_controller() {
    local cmd="$1"; shift
    local vars; mapfile -t vars < <(_safety_compose_env)
    env -u VIRTUAL_ENV "${vars[@]}" \
        uv run --quiet --locked --isolated --only-group safety python tools/safety_controller.py "$cmd" \
        --data "$SAFETY_DIR/data" --cameras "$SAFETY_CAMERAS" --clips "$SAFETY_DEMO_CLIPS" \
        --compose "$(safety_compose_cmd)" --redis "redis://localhost:$SAFETY_REDIS_PORT/0" \
        --go2rtc-api "http://localhost:$SAFETY_GO2RTC_API_PORT" --fall-restart "just safety-fall-restart" "$@"
}
