#!/usr/bin/env bash
# Cut the clips of the golden configs added for task #229: PPE, shelf void and fall (tests/configs/golden-{ppe,shelf,fall_detection}.yaml).
# Sources are the sample videos of the demos (`just data-pull samples-ppe samples-shelf samples-fall`). Same rules as
# make_golden_clips.sh: no audio (the sidecar image has no AAC decoder, and a source with an audio track it cannot decode stalls
# before the first frame), 1280x720 at most (a 1080p clip stalled the development laptop's GPU), re-encoded ONCE: runs replay
# the resulting file, so the pulled clip, not a re-cut one, is the reference. NOT yet a dvc.yaml stage: add it (and the
# baselines) once the three replays have been recorded and checked on the GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
out=data/uploads/golden
mkdir -p "$out"
enc=(-an -vf "scale='min(1280,iw)':-2" -r 30 -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -g 30)
cut() {   # <source> <out> [extra ffmpeg input options]
    local src=$1 dst=$2; shift 2
    if [ -f "$src" ]; then ffmpeg -nostdin -loglevel error -n "$@" -i "$src" "${enc[@]}" "$dst"
    else echo "missing $src: no $(basename "$dst") was cut (just data-pull)"; fi
}
cut data/uploads/ppe/test.mp4   "$out/ppe.mp4"            # 22 s, 1080p with audio in the original
cut data/uploads/shelf/demo.mp4 "$out/shelf.mp4" -t 40
# fall: the 31 s, 320x240 sample (not the Le2i corpus clip make_golden_clips.sh cuts when the corpus is present: that one has
# ground truth, this one is just a short fall-product replay). Not upscaled; stream copy, audio dropped.
if [ -f data/uploads/fall/video.mp4 ] && [ ! -f "$out/fall.mp4" ]; then
    ffmpeg -nostdin -loglevel error -n -i data/uploads/fall/video.mp4 -an -c:v copy "$out/fall.mp4"
fi
ls -la "$out"
