#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-multicam}"
shift || true
EXTRA_ARGS="$*"

PID_FILE="$REPO_DIR/.streamer.pid"
LOG_FILE="$REPO_DIR/data/streamer.log"
mkdir -p "$REPO_DIR/data"

# go2rtc's RTSP ingest port -- .env's GO2RTC_RTSP_PORT, same as docker-compose.yml
set -a
[[ -f "$REPO_DIR/.env" ]] && source "$REPO_DIR/.env"
set +a
RTSP_PORT="${GO2RTC_RTSP_PORT:-8554}"

# Stop any running instance first
bash "$REPO_DIR/scripts/stop_streamer.sh"

echo "==> Waiting for go2rtc RTSP service on port $RTSP_PORT..."
python3 -c '
import socket, time, sys
for _ in range(30):
    try:
        with socket.create_connection(("localhost", int(sys.argv[1])), timeout=1):
            print("    go2rtc RTSP port %s is ready." % sys.argv[1])
            sys.exit(0)
    except Exception:
        time.sleep(0.5)
print("    warning: go2rtc RTSP port %s timed out, starting streamer anyway..." % sys.argv[1])
' "$RTSP_PORT"

echo "==> Starting test streamer daemon (profile: $PROFILE)..."
python3 -c "
import os, sys

repo = os.path.abspath('$REPO_DIR')
log_file = os.path.join(repo, 'data', 'streamer.log')
pid_file = os.path.join(repo, '.streamer.pid')

cmd = [sys.executable, os.path.join(repo, 'test_streamer', 'stream_recordings.py'),
       '--profile', '$PROFILE', '--port', '$RTSP_PORT', '--pid-file', pid_file]
extra = '''$EXTRA_ARGS'''.split()
cmd.extend(extra)

pid = os.fork()
if pid > 0:
    sys.exit(0)

os.setsid()
pid2 = os.fork()
if pid2 > 0:
    sys.exit(0)

with open(log_file, 'a') as f:
    os.dup2(f.fileno(), 1)
    os.dup2(f.fileno(), 2)

os.execv(sys.executable, cmd)
"

sleep 1.5

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    STREAMER_PID="$(cat "$PID_FILE")"
    echo "    Test streamer is running in background (PID: $STREAMER_PID)."
    echo "    Logs available at: data/streamer.log"
else
    echo "    Error: Test streamer failed to start. Last log lines:" >&2
    tail -n 20 "$LOG_FILE" >&2 || true
    exit 1
fi
