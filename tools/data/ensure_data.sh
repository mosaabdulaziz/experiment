#!/usr/bin/env bash
# Make sure the models and clips a sidecar config uses are pulled (DATA.md). When they are not and a person is at the
# terminal, offer to pull the right DVC group now; in a script or CI it just fails with the instruction.
#
#   tools/data/ensure_data.sh [sidecar.yaml]      default: $SIDECAR_CONFIG, else perception/pipeline/sidecar.yaml
set -uo pipefail
cd "$(dirname "$0")/../.."
cfg="${1:-}"
python3 -I tools/data/check_models.py $cfg && exit 0
groups=$(python3 -I tools/data/check_models.py --groups $cfg 2>/dev/null)
if [ -t 0 ] && [ -t 1 ] && [ -n "$groups" ]; then
    printf 'Pull %s now? [Y/n] ' "$groups"
    read -r answer
    case "${answer:-y}" in n|N|no|NO) exit 1 ;; esac
    just data-pull $groups && python3 -I tools/data/check_models.py $cfg
else
    exit 1
fi
