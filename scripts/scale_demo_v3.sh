#!/usr/bin/env bash
# demo-v3 ON the server (what `just demo-v3` does on the laptop): the day the NEW start rules keep, as yesterday, plus a LIVE run
# beside it, from a clean state. Run ON the server, from the synced copy: scripts/scale_demo_v3.sh [START]   (START: footage time,
# default 11:49). From the laptop: `just demo-v3-server`, which syncs the code and the removal lists first.
#
# The original export (the finished day-run, on the server) is only read. The platform database and the run state of the scale
# project (its own compose project, ports 35443 ...) are replaced; nobody else's stack is touched.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export ON_SERVER=1 DEST="$REPO" SCALE_DATA="$REPO/data" UI_HTTPS_PORT="${UI_HTTPS_PORT:-35443}"
EXPORT="${EXPORT:-$(dirname "$REPO")/day-export-2026-09-13}"
LISTS="$REPO/review"                      # the removal lists, copied here by the laptop wrapper
START="${1:-${START:-2026-09-13T11:49:00}}"
[ -f "$EXPORT/manifest.json" ] || { echo "the original export is missing: $EXPORT" >&2; exit 1; }
[ -f "$LISTS/removed_trucks.json" ] && [ -f "$LISTS/removed_trucks_newrules.json" ] || { echo "the removal lists are missing in $LISTS" >&2; exit 1; }
nvidia-smi -L >/dev/null 2>&1 || { echo "the NVIDIA driver is not usable here: $(nvidia-smi 2>&1 | head -1)" >&2; exit 1; }
DAY=$(date -d yesterday +%F)
echo "== 1/3 the original export as $DAY (its footage 00:00 = $DAY 00:00)"
SKIP_BACKUP=1 SHIFT_TO="${DAY}T00:00:00" scripts/scale_day_import.sh --local "$EXPORT"
echo "== 2/3 the trucks the new start rules would not track are removed ($DAY becomes the new-rules day)"
REMOVED_TRUCKS="$LISTS/removed_trucks.json" python3 scripts/scale_remove_trucks.py --from-record --apply
REMOVED_TRUCKS="$LISTS/removed_trucks_newrules.json" python3 scripts/scale_remove_trucks.py --from-record --apply
echo "== 3/3 live from footage $START beside it"
START="$START" scripts/scale13_deploy.sh continue
echo "== waiting (up to 8 min) for the cameras"
for _ in $(seq 48); do python3 scripts/scale_day_status.py --brief 2>/dev/null | grep -q "cameras dead 0;" && break; sleep 10; done
python3 scripts/scale_day_status.py | head -12
echo "demo-v3 ready on the server: https://${PUBLIC_HOST:-100.102.162.114}:$UI_HTTPS_PORT ($DAY = what the new rules keep, today = live)"
