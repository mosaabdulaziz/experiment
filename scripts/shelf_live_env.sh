# Settings of the shelf demo (`just shelf-*`, docker-compose.shelf-live.yml). Sourced by those recipes, not run.
# Each SHELF_* setting comes from the environment, else from .env, else the default below. Only the SHELF_* lines of
# .env are read: sourcing all of it would give this stack the main stack's project name and ports.
# Same shape as scripts/ppe_live_env.sh.
_shelf_setting() {
    local v="${!1-}"
    if [[ -z "$v" && -f .env ]]; then
        v="$(sed -nE "s/^[[:space:]]*(export[[:space:]]+)?$1=([^#]*).*/\2/p" .env | tail -n1)"
        v="$(sed -E 's/[[:space:]]+$//; s/^"(.*)"$/\1/; s/^'\''(.*)'\''$/\1/' <<< "$v")"
    fi
    printf -v "$1" '%s' "${v:-$2}"
    export "$1"
}
_shelf_setting SHELF_COMPOSE_PROJECT shelflive
_shelf_setting SHELF_LIVE_VIDEOS data/uploads/shelf      # DVC: just data-pull samples-shelf
_shelf_setting SHELF_LIVE_DIR scratch/shelf-live
# Absolute: a relative bind-mount source without ./ is a named volume to compose
SHELF_LIVE_DIR="$(realpath -m "$SHELF_LIVE_DIR")"
# Host ports, beside every other stack on the machine (the PPE demo uses 48xxx/49xxx)
_shelf_setting SHELF_IRIS_PORT 47000
_shelf_setting SHELF_REDIS_PORT 47379
_shelf_setting SHELF_GO2RTC_RTSP_PORT 47554
_shelf_setting SHELF_GO2RTC_API_PORT 47984
_shelf_setting SHELF_MINIO_API_PORT 47900
_shelf_setting SHELF_MINIO_CONSOLE_PORT 47901
_shelf_setting SHELF_SIDECAR_TELEMETRY_PORT 47101
# The images docker-compose.yml builds for this compose project (`just shelf-build`). The sidecar runs the checkout's
# own code (the repo is mounted at /workspace/iris), so any sidecar image built from this repo's
# perception/pipeline/Dockerfile can be named here instead of building one.
_shelf_setting SHELF_SIDECAR_IMAGE "$SHELF_COMPOSE_PROJECT-deepstream-sidecar:latest"
_shelf_setting SHELF_IRIS_IMAGE "$SHELF_COMPOSE_PROJECT-iris:latest"
# The sidecar and the app run as the user who runs the recipe: files they write stay theirs
export SHELF_UID="$(id -u)" SHELF_GID="$(id -g)"

shelf_compose() {
    env COMPOSE_PROJECT_NAME="$SHELF_COMPOSE_PROJECT" IRIS_HOST_PORT="$SHELF_IRIS_PORT" REDIS_HOST_PORT="$SHELF_REDIS_PORT" \
        GO2RTC_RTSP_PORT="$SHELF_GO2RTC_RTSP_PORT" GO2RTC_API_PORT="$SHELF_GO2RTC_API_PORT" \
        MINIO_API_HOST_PORT="$SHELF_MINIO_API_PORT" MINIO_CONSOLE_HOST_PORT="$SHELF_MINIO_CONSOLE_PORT" \
        SIDECAR_TELEMETRY_PORT="$SHELF_SIDECAR_TELEMETRY_PORT" \
        docker compose -p "$SHELF_COMPOSE_PROJECT" -f docker-compose.yml -f docker-compose.shelf-live.yml "$@"
}
