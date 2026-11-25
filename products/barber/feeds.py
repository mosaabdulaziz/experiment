"""go2rtc's camera feeds for a barber site (2026-10-15, Palm Hills on its real cameras). barber-up runs it.

live:  point each raw stream at the shop's recorder, in go2rtc's memory only. The channels come from site.sh;
       the recorder's address with its credentials from .env (--base-env names the variable), never from git or
       go2rtc.yaml. go2rtc answers 400 when it cannot save its read-only config file but still applies the source,
       so the result is checked on /api/streams, not on the status codes.
clips: exit 3 when go2rtc still pulls a camera into one of the clip streams (left from a live run): a pushed clip
       would mix with it. go2rtc cannot turn a stream back into an empty placeholder over its API, so barber-up
       restarts go2rtc (= go2rtc.yaml's placeholders again).
Standard library only: barber-up runs it with the system python3."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

Call = Callable[[str, dict], dict]


def sources(feeds: str, base: str) -> list[tuple[str, str]]:
    """'raw-ph-1=101 raw-ph-2=1101' -> [(stream, <base>/<channel>#transport=tcp), ...] (TCP: UDP tears on loss)."""
    out = []
    for item in feeds.split():
        name, channel = item.split("=", 1)
        out.append((name, f"{base.rstrip('/')}/{channel}#transport=tcp"))
    return out


def _urls(streams: dict, name: str) -> list[str]:
    return [p["url"] for p in ((streams.get(name) or {}).get("producers") or []) if p.get("url")]


def apply(pairs: list[tuple[str, str]], call: Call) -> list[str]:
    """Replace each stream's source (DELETE, then PUT: a PATCH emptied them, 2026-10-15); the names go2rtc did not take."""
    for name, src in pairs:
        call("DELETE", {"src": name})
        call("PUT", {"name": name, "src": src})
    streams = call("GET", {})
    return [name for name, src in pairs if src not in _urls(streams, name)]


def pulled(streams: dict, names: list[str]) -> list[str]:
    """The streams go2rtc pulls from a source of its own (a producer with a url); a push carries none."""
    return [n for n in names if _urls(streams, n)]


def http(api: str) -> Call:
    def call(method: str, params: dict) -> dict:
        url = f"{api.rstrip('/')}/api/streams" + (f"?{urllib.parse.urlencode(params)}" if params else "")
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method=method), timeout=10) as r:
                body = r.read()
        except urllib.error.HTTPError as exc:     # 400 "read-only file system": applied, not saved
            body = exc.read()
        try:
            return json.loads(body or b"{}") if method == "GET" else {}
        except ValueError:
            return {}
    return call


def _wait(call: Call, seconds: float = 30.0) -> dict:
    end = time.monotonic() + seconds
    while True:
        try:
            return call("GET", {})
        except OSError:
            if time.monotonic() > end:
                raise
            time.sleep(1)


def main(argv: list[str] | None = None, env: dict | None = None) -> int:
    env = os.environ if env is None else env
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("live", "clips"))
    ap.add_argument("--api", required=True)
    ap.add_argument("--feeds", default="", help="live: 'raw-ph-1=101 raw-ph-2=1101 ...'")
    ap.add_argument("--base-env", default="", help="live: the .env variable holding rtsp://user:pass@host:port/path")
    ap.add_argument("--ids", default="", help="clips: the clip ids (raw-<id> streams), comma separated")
    a = ap.parse_args(argv)
    call = http(a.api)
    if a.mode == "live":
        base = env.get(a.base_env, "") if a.base_env else ""
        if not base:
            print(f"feeds: set {a.base_env or 'the recorder variable'} in .env (rtsp://user:pass@host:554/...)",
                  file=sys.stderr)
            raise SystemExit(2)
        pairs = sources(a.feeds, base)
        _wait(call)
        missing = apply(pairs, call)
        for name, src in pairs:
            print(f"feeds: {name} <- {src.rsplit('/', 1)[-1]}" + ("  NOT TAKEN" if name in missing else ""))
        return 1 if missing else 0
    names = [f"raw-{i}" for i in a.ids.split(",") if i]
    left = pulled(_wait(call), names)
    if left:
        print(f"feeds: go2rtc still pulls a camera into {', '.join(left)}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
