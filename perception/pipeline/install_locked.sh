#!/usr/bin/env bash
# Install one dependency group of perception/uv.lock into the active venv, hash-checked (DEPENDENCIES.md).
#
#   install_locked.sh <project dir> <group> <ARG name> <ARG value>
#                                     e.g. install_locked.sh /req slim BASE_MODULES "$BASE_MODULES"
#
# The group was generated from the Dockerfile's default module list (tools/deps/perception_groups.py records it in
# pyproject.toml as `# <group>: <ARG name>="..."`). A build whose --build-arg differs stops here, before installing
# anything: the lock's group cannot install another module list, and ignoring the override silently would build an
# image without the modules that were asked for.
#
# Exact versions and files come from the lock (`uv export --frozen`, `uv pip install --require-hashes`), in three
# passes, because a few wheels write the same files and the one installed LAST wins:
#   1. everything else;
#   2. CUDA torch/torchvision (mtmc only), from PyTorch's cu130 index (an export carries no index URLs; the hash still
#      pins the exact file);
#   3. LATE, one at a time, in this order: opencv-python after opencv-python-headless (the GUI build's cv2 is the one
#      that ran before the lock: insightface installed it after the engine's headless one), then onnxruntime-gpu after
#      onnxruntime (the GPU runtime must win; insightface pulls in the CPU one).
# Removing those duplicate CPU/GUI wheels is a follow-up (DEPENDENCIES.md, "Follow-ups").
set -euo pipefail
[ $# -eq 4 ] || { echo "usage: install_locked.sh <project dir> <group> <ARG name> <ARG value>" >&2; exit 2; }
project=$1 group=$2 arg=$3
read -ra words <<< "$4"; requested="${words[*]}"   # whitespace-normalised, no glob expansion
record=$(grep -m1 "^# $group: $arg=\"" "$project/pyproject.toml" || true)
if [ -z "$record" ]; then
    echo "error: perception/pyproject.toml records no $arg for the '$group' group: run \`just deps-lock\`" >&2
    exit 1
fi
recorded=${record#*=\"}; recorded=${recorded%\"}
if [ "$requested" != "$recorded" ]; then
    echo "error: $arg=\"$requested\", but perception/uv.lock's '$group' group was generated from $arg=\"$recorded\"." >&2
    echo "The image installs that locked group, so a different module list cannot take effect. To change it, edit the" >&2
    echo "default of ARG $arg in perception/pipeline/Dockerfile, run \`just deps-lock\`, commit perception/pyproject.toml" >&2
    echo "and perception/uv.lock, and build without --build-arg $arg." >&2
    exit 1
fi
LATE="opencv-python onnxruntime-gpu"
CUDA="torch torchvision"
CUDA_INDEX=https://download.pytorch.org/whl/cu130   # perception/pyproject.toml, [[tool.uv.index]] (a test keeps them equal)
work=$(mktemp -d)

uv export --quiet --frozen --project "$project" --only-group "$group" --no-emit-project --format requirements.txt \
    -o "$work/all.txt"
skip=()
for p in $LATE $CUDA; do skip+=(--no-emit-package "$p"); done
uv export --quiet --frozen --project "$project" --only-group "$group" --no-emit-project --format requirements.txt \
    "${skip[@]}" -o "$work/main.txt"

# One package's entry from the full export: its `name==version` line and the indented --hash lines under it.
entry() { awk -v n="$1" 'index($0, n "==") == 1 { p = 1; print; next } p && /^ / { print; next } { p = 0 }' "$work/all.txt"; }

uv pip install --no-cache --require-hashes --no-deps -r "$work/main.txt"   # the export is the whole closure

: > "$work/cuda.txt"
for p in $CUDA; do entry "$p" >> "$work/cuda.txt"; done
if [ -s "$work/cuda.txt" ]; then
    uv pip install --no-cache --require-hashes --no-deps --index-url "$CUDA_INDEX" -r "$work/cuda.txt"
fi

for p in $LATE; do
    entry "$p" > "$work/late.txt"
    if [ -s "$work/late.txt" ]; then
        uv pip install --no-cache --require-hashes --no-deps -r "$work/late.txt"
    fi
done
rm -rf "$work"
