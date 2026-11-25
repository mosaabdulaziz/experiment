# Settings of the PPE demo (`just ppe-*`, docker-compose.ppe-live.yml). Sourced by those recipes, not run.
# Each PPE_* setting comes from the environment, else from .env, else the default below (.env.example lists them).
# Only the PPE_* lines of .env are read: sourcing all of it would give this stack the main stack's project name
# and ports.
_ppe_setting() {
    local v="${!1-}"
    if [[ -z "$v" && -f .env ]]; then
        v="$(sed -nE "s/^[[:space:]]*(export[[:space:]]+)?$1=([^#]*).*/\2/p" .env | tail -n1)"
        v="$(sed -E 's/[[:space:]]+$//; s/^"(.*)"$/\1/; s/^'\''(.*)'\''$/\1/' <<< "$v")"
    fi
    printf -v "$1" '%s' "${v:-$2}"
    export "$1"
}
_ppe_setting PPE_COMPOSE_PROJECT ppelive
_ppe_setting PPE_LIVE_VIDEOS data/uploads/ppe        # DVC: just data-pull samples-ppe
_ppe_setting PPE_LIVE_DIR scratch/ppe-live
# Absolute: a relative bind-mount source without ./ is a named volume to compose
PPE_LIVE_DIR="$(realpath -m "$PPE_LIVE_DIR")"
# Host ports, beside every other stack on the machine
_ppe_setting PPE_IRIS_PORT 48000
_ppe_setting PPE_REDIS_PORT 46379
_ppe_setting PPE_GO2RTC_RTSP_PORT 48554
_ppe_setting PPE_GO2RTC_API_PORT 41984
_ppe_setting PPE_MINIO_API_PORT 49000
_ppe_setting PPE_MINIO_CONSOLE_PORT 49001
_ppe_setting PPE_SIDECAR_TELEMETRY_PORT 49101
# The sidecar image docker-compose.yml builds for this compose project (`just ppe-build`)
_ppe_setting PPE_SIDECAR_IMAGE "$PPE_COMPOSE_PROJECT-deepstream-sidecar:latest"
PPE_IRIS_IMAGE="$PPE_COMPOSE_PROJECT-iris:latest"
# The sidecar and the app run as the user who runs the recipe: files they write stay theirs
export PPE_UID="$(id -u)" PPE_GID="$(id -g)"

ppe_compose() {
    env COMPOSE_PROJECT_NAME="$PPE_COMPOSE_PROJECT" IRIS_HOST_PORT="$PPE_IRIS_PORT" REDIS_HOST_PORT="$PPE_REDIS_PORT" \
        GO2RTC_RTSP_PORT="$PPE_GO2RTC_RTSP_PORT" GO2RTC_API_PORT="$PPE_GO2RTC_API_PORT" \
        MINIO_API_HOST_PORT="$PPE_MINIO_API_PORT" MINIO_CONSOLE_HOST_PORT="$PPE_MINIO_CONSOLE_PORT" \
        SIDECAR_TELEMETRY_PORT="$PPE_SIDECAR_TELEMETRY_PORT" \
        docker compose -p "$PPE_COMPOSE_PROJECT" -f docker-compose.yml -f docker-compose.ppe-live.yml "$@"
}
