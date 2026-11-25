#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Performing clean shutdown..."
# `just down` stops everything this checkout may have started: the preset
# streamer (.streamer.pid), the attendance host processes (.run/*.pid,
# including the sidecar watchdog), stray publishers, and every compose
# service including the `roles` profile.
(cd "$REPO_DIR" && just down)

echo "==> Clean shutdown complete."
