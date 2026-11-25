#!/usr/bin/env bash
# Replay golden configs on the CURRENT code and diff each against its recorded baseline.
#
#   tools/engine_golden_check.sh attendance zones fire multicam barber
#
# Baselines are tests/baselines/golden-<name>.before.jsonl, recorded from the pre-refactor sidecar. Exit status is
# non-zero if any config differs. One camera per config: two file sources at once stall each other on a small GPU.
#
# Exact comparison for the configs that replay byte-for-byte (attendance, zones, multicam). `fire`, `barber` and `proximity`
# (NvDCF hands some ids out differently run to run; perception/TESTING.md) are
# compared semantically (SEMANTIC below): the fire alert's sustain window runs on wall-clock time, so it lands a few
# frames apart run to run, and barber's ReID tracker on the GPU jitters box values -- measured by diffing two runs of
# the SAME code, which differed on ~565 of 750 frames, as much as new-vs-original did. Override with SEMANTIC="...".
# The barber configs also sample people for appearance once per track per second of WALL-clock time, so that stream's
# entry count follows how long the replay took: the original code gave 337 and 339, the modules 343 to 352. They get a
# looser count tolerance (8%); the meta stream, which is deterministic, still has to match in counts and labels.
set -uo pipefail
cd "$(dirname "$0")/.."
status=0
SEMANTIC="${SEMANTIC:-fire barber barber-faces proximity}"
for name in "$@"; do
  echo "=== $name $(date +%T)"
  rm -f "tests/baselines/golden-$name.after.jsonl"     # never diff a stale recording from an earlier run
  run_log=$(mktemp)
  GOLDEN_IDLE="${GOLDEN_IDLE:-40}" GOLDEN_MAX="${GOLDEN_MAX:-1500}" \
    tools/engine_golden_run.sh "tests/configs/golden-$name.yaml" "tests/baselines/golden-$name.after.jsonl" > "$run_log" 2>&1
  run_rc=$?
  grep -E "recorded|^ +[0-9]+ +iris|rror" "$run_log"
  # a replay that never started (e.g. docker compose refusing the project name) used to show up only as a missing file
  if [ "$run_rc" -ne 0 ]; then echo "replay failed (exit $run_rc); its last lines:"; tail -6 "$run_log" | sed 's/^/    /'; status=1; rm -f "$run_log"; continue; fi
  rm -f "$run_log"
  docker logs iris-deepstream-sidecar-1 > "${GOLDEN_LOGS:-/tmp}/golden-$name.log" 2>&1 || true
  mode=""
  case " $SEMANTIC " in *" $name "*) mode="--semantic" ;; esac
  tol=""
  case "$name" in barber*) tol="--tol 0.08" ;; esac
  python3 tools/engine_golden.py diff $mode $tol "tests/baselines/golden-$name.before.jsonl" \
      "tests/baselines/golden-$name.after.jsonl" || status=1
done
exit $status
