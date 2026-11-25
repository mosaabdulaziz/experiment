#!/usr/bin/env bash
# From the laptop: bring demo-v3 up on the server (ssh): sync the camera-system code and the UI (committed files of HEAD of both
# repos), copy the removal lists, then run scripts/scale_demo_v3.sh there. The server run keeps going after this script ends.
#
#   scripts/scale_demo_v3_server.sh [START]        START = footage time, default 11:49:00
# SERVER=iris@10.2.0.60 (office) instead of the Tailscale address iris@100.102.162.114 (from home).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
SERVER="${SERVER:-iris@100.102.162.114}"
DEST=/data/IRIS/TESTING_PLAYGROUND/surveillance-camera-sys-scale
UI_DEST=/data/IRIS/TESTING_PLAYGROUND/truck-tracking-scale
UI_SRC="${UI_SRC:-$REPO/../truck-tracking}"
SSHO=(-o RequestTTY=no -o RemoteCommand=none -o BatchMode=yes -o ServerAliveInterval=30)
START="${1:-2026-09-13T11:49:00}"
git diff --quiet HEAD -- . || echo "note: uncommitted changes are NOT deployed (sync ships the committed files of HEAD)"
echo "== 1/4 camera-system code, preset and .env -> $SERVER:$DEST"
HOST="$SERVER" scripts/scale13_deploy.sh sync
echo "== 2/4 truck-tracking UI -> $UI_DEST"
ssh "${SSHO[@]}" "$SERVER" "mkdir -p '$UI_DEST'"
git -C "$UI_SRC" archive HEAD | ssh "${SSHO[@]}" "$SERVER" "tar -x -C '$UI_DEST'"
echo "== 3/4 the removal lists"
ssh "${SSHO[@]}" "$SERVER" "mkdir -p '$DEST/review'"
scp -q "${SSHO[@]}" "$HOME/URCA/IRIS/review/removed_trucks.json" "$HOME/URCA/IRIS/review/whatif/removed_trucks_newrules.json" "$SERVER:$DEST/review/"
echo "== 4/4 demo-v3 on the server (import, removals, live run)"
PH="${SERVER#*@}"
ssh "${SSHO[@]}" "$SERVER" "cd '$DEST' && PUBLIC_HOST=$PH bash scripts/scale_demo_v3.sh '$START'"
