#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESET="${1:-}"

if [[ -z "$PRESET" ]]; then
    echo "Usage: $0 <vlm|multicam>" >&2
    exit 1
fi

PRESET_DIR="$REPO_DIR/configs/presets/$PRESET"

if [[ ! -d "$PRESET_DIR" ]]; then
    echo "Error: preset '$PRESET' not found in $REPO_DIR/configs/presets" >&2
    exit 1
fi

echo "==> Switching active configuration to: $PRESET"
cp "$PRESET_DIR/config.yaml" "$REPO_DIR/config.yaml"
echo "$PRESET" > "$REPO_DIR/.active_profile"
echo "    Applied $PRESET_DIR/config.yaml -> config.yaml"
# The pipeline side is NOT copied over perception/pipeline/sidecar.yaml any more (the shared
# file stays neutral). The preset runs on an overlay named through SIDECAR_CONFIG, which
# scripts/run_setup.sh exports for compose and the preset's config.yaml names for the app.
case "$PRESET" in
    scale13) echo "    Pipeline overlay: products/multicam/sidecar.yaml" ;;
    *)       echo "    Pipeline overlay: configs/presets/$PRESET/sidecar.yaml" ;;
esac
# Per-site files the preset carries (filter zones, gate counting lines): put in place only when the run
# has none yet, so zones edited later in the UI are not overwritten.
mkdir -p "$REPO_DIR/data"
for f in filter_zones.json count_lines.json; do
    if [[ -f "$PRESET_DIR/$f" && ! -f "$REPO_DIR/data/$f" ]]; then
        cp "$PRESET_DIR/$f" "$REPO_DIR/data/$f"
        echo "    Applied $PRESET_DIR/$f -> data/$f"
    fi
done
