# Vendored from shelf-demo (~/sarah/shelf-demo), branch feat/iris-integration, commit 195390728433235b0daced2c0f4dc8614a2d212e,
# service/log.py. Copied unchanged below this header: fix it there first, then copy it again.
"""One JSON object per line on stdout. Never pass camera passwords or the AMQP URL."""
import json
import sys
import threading
from datetime import datetime, timezone

_lock = threading.Lock()


def log(msg, **fields):
    rec = {"time": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
           "msg": msg, **fields}
    line = json.dumps(rec, ensure_ascii=False, default=str)
    with _lock:
        print(line, file=sys.stdout, flush=True)
