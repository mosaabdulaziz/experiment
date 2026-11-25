#!/bin/sh
# Build every custom nvinfer output parser under perception/models/*/nvdsinfer_custom_impl_*/ and install it where the
# models' configs expect it (/opt/nvidia/deepstream/deepstream/custom_parsers). A new model that needs a parser of
# its own only adds its directory: nothing in the Dockerfile changes.
#
# `parser_flags` (next to this script) lists extra `make` variables per parser directory (e.g. OPENCV=1 for the YOLO
# parser's INT8 calibrator). CUDA_VER comes from the environment (the Dockerfile sets it for the container's CUDA).
# `make clean` runs first: objects built on a host must never leak into the image. Each library must export at
# least one NvDsInferParse* symbol, so a parser that built but cannot be found by nvinfer fails the image build.
set -eu
root=${1:-/src/perception/models}
dest=/opt/nvidia/deepstream/deepstream/custom_parsers
mkdir -p "$dest"
found=0
for dir in "$root"/*/nvdsinfer_custom_impl_*/; do
  [ -f "$dir/Makefile" ] || continue
  found=1
  echo "== parser $dir"
  (
    cd "$dir"
    flags=$(grep -v '^[[:space:]]*#' "$root/parser_flags" 2>/dev/null | awk -v d="$(basename "$dir")" '$1 == d { $1 = ""; print }')
    make clean
    make -j"$(nproc)" $flags
    for lib in lib*.so; do
      nm -D --defined-only "$lib" | grep -q ' NvDsInferParse' || { echo "$lib exports no NvDsInferParse* function" >&2; exit 1; }
      cp "$lib" "$dest/"
    done
  )
done
[ "$found" = 1 ] || { echo "no parser directories under $root" >&2; exit 1; }
