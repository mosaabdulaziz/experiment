#!/usr/bin/env bash
# Cut the proximity golden clip (tests/configs/golden-proximity.yaml): 40 s of the Camera189 placement clip around the
# forklift incident at 1:36. dvc.yaml's golden_proximity_clip stage: a fresh machine pulls the clip (`just data-pull`);
# re-cutting it needs the site recording (`just data-pull samples-proximity`). Stream copy: same source, same bytes.
set -euo pipefail
cd "$(dirname "$0")/.."
src=data/uploads/proximity/procimity_placement_combined.mp4
out=data/uploads/golden
[ -f "$src" ] || { echo "the proximity site recording is not on this machine ($src): no clip was cut" >&2; exit 1; }
mkdir -p "$out"
ffmpeg -nostdin -loglevel error -n -ss 80 -t 40 -i "$src" -an -c copy "$out/proximity.mp4"
ls -la "$out/proximity.mp4"
