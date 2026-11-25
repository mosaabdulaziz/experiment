#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-vlm}"
RUN_STREAMER="${2:-true}"

echo "========================================================="
echo " Starting IRIS: Profile = $PROFILE, Streamer = $RUN_STREAMER"
echo "========================================================="

# 1. Clean shutdown of any existing runs
bash "$REPO_DIR/scripts/clean_down.sh"

# 2. Switch configuration preset
bash "$REPO_DIR/scripts/switch_preset.sh" "$PROFILE"

# Free GPU VRAM for multicam profile (Ollama VLM models hold 3-4GB of GPU memory)
if [[ "$PROFILE" == "multicam" ]] && command -v ollama >/dev/null 2>&1; then
    echo "==> Freeing GPU memory for 13-camera DeepStream pipeline (stopping Ollama models)..."
    for model in $(ollama ps 2>/dev/null | awk 'NR>1 {print $1}'); do
        echo "    Stopping Ollama model: $model"
        ollama stop "$model" >/dev/null 2>&1 || true
    done
fi

# 3. Bring up stack
cd "$REPO_DIR"

# The pipeline runs on the preset's overlay, never on a copy of the shared file. multicam uses
# the product's own compose service (the `mtmc` image target, profile `multicam`); vlm uses the
# plain service pointed at its overlay.
if [[ "$PROFILE" == "multicam" ]]; then
    export SIDECAR_CONFIG="configs/presets/multicam/sidecar.yaml"
    SIDECAR_SVC="deepstream-sidecar-multicam"
    COMPOSE_UP=(docker compose --profile multicam)
    BASE_SVCS=(redis go2rtc minio)
    REST_SVCS=(iris "$SIDECAR_SVC")
else
    export SIDECAR_CONFIG="configs/presets/$PROFILE/sidecar.yaml"
    SIDECAR_SVC="deepstream-sidecar"
    COMPOSE_UP=(docker compose)
    BASE_SVCS=(redis go2rtc minio)
    REST_SVCS=(iris "$SIDECAR_SVC")
fi
echo "==> Pipeline overlay: $SIDECAR_CONFIG (service: $SIDECAR_SVC)"

# Models and clips are versioned with DVC (DATA.md). A preset whose files were never pulled would come up "ready"
# and process nothing, so say what to pull before starting anything.
"$REPO_DIR/tools/data/ensure_data.sh" "$SIDECAR_CONFIG" || exit 1

# Which product the dashboard on the app's port opens with. Unset, the app reports "every product"
# and the shell opens the first one -- attendance, whose backend these presets do not run (its
# /attendance/ws fails and the page never connects). These presets are the surveillance UI;
# the multi-truck tracking view is the truck-tracking platform. IRIS_PRODUCT in .env still wins.
export IRIS_PRODUCT="${IRIS_PRODUCT:-iris}"
echo "==> Dashboard product: $IRIS_PRODUCT"

if [[ "$RUN_STREAMER" == "true" ]]; then
    echo "==> Bringing up base services (redis, go2rtc, minio)..."
    "${COMPOSE_UP[@]}" up -d "${BASE_SVCS[@]}"

    echo "==> Launching test streamer for profile '$PROFILE'..."
    bash "$REPO_DIR/scripts/start_streamer.sh" "$PROFILE"

    echo "==> Bringing up remaining services ($SIDECAR_SVC, iris)..."
    "${COMPOSE_UP[@]}" up -d "${REST_SVCS[@]}"
else
    echo "==> Bringing up the stack without the test streamer..."
    "${COMPOSE_UP[@]}" up -d "${BASE_SVCS[@]}" "${REST_SVCS[@]}"
fi

echo ""
echo "==> Verifying running services:"
"${COMPOSE_UP[@]}" ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

# The multi-camera preset feeds the truck-tracking web UI; start it with the stack.
TRUCK_UI_URL=""
if [[ "$PROFILE" == "multicam" ]]; then
    echo "==> Starting the truck-tracking web UI..."
    # The script exits 0 when it skips (no ../truck-tracking checkout), so ask the port, not the exit status.
    bash "$REPO_DIR/scripts/truck_ui.sh" up || true
    if curl -sf -m 5 -o /dev/null "http://localhost:5173/"; then
        TRUCK_UI_URL="http://localhost:5173"
    else
        echo "==> The truck UI is not running (see above; it needs the truck-tracking repo next to this one, or TRUCK_UI_DIR)."
    fi
fi

echo ""
set -a
[[ -f .env ]] && source .env
set +a
echo "========================================================="
echo " Stack ready! Active profile: $PROFILE"
echo " Web UI:    http://localhost:${IRIS_HOST_PORT:-18000}"
echo " go2rtc UI: http://localhost:${GO2RTC_API_PORT:-1984}"
echo " MinIO UI:  http://localhost:${MINIO_CONSOLE_HOST_PORT:-19001}"
[[ -n "$TRUCK_UI_URL" ]] && echo " Trucks UI: $TRUCK_UI_URL  (admin@example.com / Admin123!)"
echo "========================================================="
