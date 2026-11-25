#!/usr/bin/env bash
# Cut the short clips the golden configs (tests/configs/golden-*.yaml) replay. This is the `golden_clips` stage of
# dvc.yaml: the sources are dvc-tracked (`just data-pull`) and the clips are its outputs, so a fresh machine pulls them
# instead of running this. Stream copy keeps the same source giving the same bytes; the fire clip is re-encoded.
set -euo pipefail
cd "$(dirname "$0")/.."
out=data/uploads/golden
mkdir -p "$out"
# fire/smoke: the whole 61 s sample, 1280x720 without audio. The sidecar image has no AAC decoder (a source with an
# audio track it cannot decode stalls before the first frame), and the 1080p original stalled the pipeline on the
# development laptop's GPU after ~2.5 s, while every 720p clip runs. Re-encoded ONCE: runs replay the resulting file.
ffmpeg -nostdin -loglevel error -n -i data/uploads/fire/part1/bucketfire.mp4 -an -vf scale=1280:720 -r 30 \
    -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -g 30 "$out/fire.mp4"
# barber shop: 90 s from the 32-minute recording
ffmpeg -nostdin -loglevel error -n -ss 600 -t 90 -i data/uploads/barber/9Hwju1yZBAM.mp4 -an -c copy "$out/barber.mp4"
# vehicles/plates/MTMC: first 90 s of one MTMC camera
ffmpeg -nostdin -loglevel error -n -t 90 -i products/el-mohandes/videos/5-CH23-enter.mp4 -an -c copy "$out/vehicles.mp4"
# fall: a clip with a fall in it, cut from the labelled corpus. The corpus is NOT in git (it is
# hundreds of videos); FALL_CORPUS points at it. The clip is Le2i Coffee_room_01 video (1), whose
# ground truth puts the fall between frames 48 and 80 (25 fps), and it is re-encoded to h264/mp4
# because the sidecar image has no AVI/Xvid decoder. Without the corpus the step says so and the
# golden fall replay is simply not available on this machine.
# The default is this machine's Git-Bash spelling of F:\fall_detection\data -- the same folder
# products/fall_detection/measure.py reads by default. On the server, set FALL_CORPUS.
corpus="${FALL_CORPUS:-/f/fall_detection/data}"
fall_src="$corpus/videos/le2i/Coffee_room_01/Coffee_room_01/Videos/video (1).avi"
if [ -f "$fall_src" ]; then
    ffmpeg -nostdin -loglevel error -n -ss 0 -t 20 -i "$fall_src" -an -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -g 30 "$out/fall.mp4"
else
    echo "the fall corpus is not on this machine ($corpus): no golden fall clip was cut"
fi
ls -la "$out"
