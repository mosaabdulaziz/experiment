#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$REPO_DIR/.streamer.pid"

echo "==> Stopping test streamer processes (if any)..."
if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE" || true)"
    if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
        echo "    Sending SIGTERM to test streamer PID $PID..."
        kill -TERM "$PID" 2>/dev/null || true
        for i in {1..10}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                break
            fi
            sleep 0.3
        done
        if kill -0 "$PID" 2>/dev/null; then
            echo "    Force killing PID $PID..."
            kill -KILL "$PID" 2>/dev/null || true
        fi
    fi
    rm -f "$PID_FILE"
fi

# Clean up any orphaned streamer or ffmpeg RTSP pusher processes
pkill -f "python3.*test_streamer/stream_recordings.py" 2>/dev/null || true
pkill -f "ffmpeg.*-f rtsp.*raw-cam" 2>/dev/null || true
echo "    Test streamer stopped cleanly."
