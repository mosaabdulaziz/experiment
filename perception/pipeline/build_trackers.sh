#!/bin/sh
# Build every custom nvtracker library under perception/trackers/*/ and install it
# under /opt/nvidia/deepstream/deepstream/custom_trackers, mirroring build_parsers.sh.
# Each library must export NvMOT_Init, NvMOT_Process, and NvMOT_Query so nvtracker
# symbol resolution is verified at image build time.
set -eu
root=${1:-/src/perception/trackers}
dest=/opt/nvidia/deepstream/deepstream/custom_trackers
mkdir -p "$dest"
found=0
for dir in "$root"/*/; do
  [ -f "$dir/Makefile" ] || continue
  found=1
  echo "== tracker $dir"
  (
    cd "$dir"
    make clean
    make -j"$(nproc)"
    for lib in lib*.so; do
      for sym in NvMOT_Init NvMOT_Process NvMOT_Query; do
        nm -D --defined-only "$lib" | grep -Eq "[[:space:]]${sym}$" || {
          echo "$lib exports no $sym symbol" >&2
          exit 1
        }
      done
      cp "$lib" "$dest/"
    done
  )
done
[ "$found" = 1 ] || { echo "no tracker directories under $root" >&2; exit 1; }
