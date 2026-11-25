"""Record and compare what the DeepStream sidecar writes to Redis, for before/after refactor checks.

The pipeline engine is being separated from the use cases (see the plan in docs/). Each step must leave
the wire output unchanged. That is checked by replaying the same recorded clip and comparing what lands in Redis.

    # run a clip config (tests/configs/golden-*.yaml: file:// source, no loop), then:
    python3 tools/engine_golden.py record --out tests/baselines/golden-attendance.before.jsonl
    # ... refactor ...
    python3 tools/engine_golden.py record --out tests/baselines/golden-attendance.after.jsonl
    python3 tools/engine_golden.py diff tests/baselines/golden-attendance.before.jsonl \
                                        tests/baselines/golden-attendance.after.jsonl

`record` tails every iris:* stream until it has been quiet for --idle seconds (the clip ended), then snapshots
string and hash keys. Output is normalised: entry ids, wall-clock fields and JPEG bytes are dropped; floats are
rounded. `diff` compares per (key, frame) and tolerates frames present in only one run (the publisher may shed
under load) up to --min-coverage; any differing value in a shared frame fails.

Runs that cannot be byte-identical (a GPU tracker with ReID, a sustain window measured in wall-clock time) are compared
with `diff --semantic`: counts, per-label detection totals and alert timing within tolerances. Prove first that the
config really is that noisy by diffing two runs of the SAME code; the semantic mode is only as strict as the noise.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

# "v" is the version stamp of an iris:meta entry, added after the baselines were recorded.
VOLATILE_FIELDS = {"ntp_ts", "recorded_at", "boot_id", "jpeg", "start_pts", "end_pts", "object_key", "path",
                   "crop_path", "v"}
# Not compared: the zones module acknowledges the zones version it loaded; a product that does not list that module
# (it moved out of the engine) no longer writes the key, and nothing of that product reads it. PPE, fire and fall
# acknowledge their own zones the same way (perception/lib/feature_zones.py): keys a product gains with the module.
IGNORED_SNAPSHOTS = {"iris:analytics:zones_applied", "iris:analytics:ppe_zones_applied",
                     "iris:analytics:fire_zones_applied", "iris:analytics:fall_zones_applied"}
# An overlay entry's `at` is its frame's arrival on the wall clock (EngineServices.overlay): different every run.
OVERLAY_PREFIX, OVERLAY_VOLATILE = "iris:overlay:", {"at"}
LONG_STRING = 200            # a longer string is a blob (an embedding, a crop): kept as its length, not its bytes
SKIP_KEY_PREFIXES = ("iris:jpeg:", "iris:ctrl:", "iris:watched:", "iris:flight", "iris:status:")
FLOAT_DIGITS = 3


def _norm(value):
    if isinstance(value, float):
        return round(value, FLOAT_DIGITS)
    if isinstance(value, str) and len(value) > LONG_STRING:
        return f"<str:{len(value)}>"
    if isinstance(value, list):
        return [_norm(v) for v in value]
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in value.items() if k not in VOLATILE_FIELDS}
    return value


def _summarise_masks(masks):
    """person_masks entries are [track id, confidence, outline | None]; an outline is dozens of floats that only the
    semantic comparisons would ignore anyway, so keep how many points it has."""
    return [[m[0], m[1], None if m[2] is None else len(m[2])] if isinstance(m, list) and len(m) == 3 else m
            for m in masks]


def _decode(field: str, raw: bytes):
    if field == "jpeg":
        return "<jpeg>"
    text = raw.decode("utf-8", "replace")
    if text[:1] in "[{":
        try:
            return _norm(json.loads(text))
        except ValueError:
            pass
    try:
        return round(float(text), FLOAT_DIGITS)
    except ValueError:
        return text


def _entry(key: str, fields: dict) -> dict:
    out = {}
    for f, raw in fields.items():
        name = f.decode() if isinstance(f, bytes) else f
        if name in VOLATILE_FIELDS or (key.startswith(OVERLAY_PREFIX) and name in OVERLAY_VOLATILE):
            continue
        value = _decode(name, raw)
        out[name] = _summarise_masks(value) if name == "person_masks" and isinstance(value, list) else value
    return {"key": key, "fields": out}


def record(url: str, out: Path, idle: float, max_seconds: float, pattern: str) -> int:
    import redis

    r = redis.Redis.from_url(url)
    cursors: dict[str, bytes] = {}
    entries: list[dict] = []
    last_new = time.monotonic()
    started = time.monotonic()
    streams: list[bytes] = []
    rescanned = 0.0
    while True:
        now = time.monotonic()
        # Streams are capped (maxlen ~64) and a file source decodes faster than real time, so poll the known
        # streams tightly and only re-discover keys about once a second.
        if now - rescanned >= 1.0:
            streams = [k for k in r.scan_iter(pattern, count=500)
                       if not k.decode().startswith(SKIP_KEY_PREFIXES) and r.type(k) == b"stream"]
            rescanned = now
        got = 0
        for raw_key in streams:
            key = raw_key.decode()
            for _id, fields in _read(r, raw_key, cursors, key):
                entries.append(_entry(key, fields))
                got += 1
        now = time.monotonic()
        if got:
            last_new = now
        if (entries and now - last_new >= idle) or now - started >= max_seconds:
            break
        time.sleep(0.02 if got else 0.1)

    snapshots = []
    for raw_key in sorted(r.scan_iter(pattern, count=500)):
        key = raw_key.decode()
        if key.startswith(SKIP_KEY_PREFIXES):
            continue
        kind = r.type(raw_key)
        if kind == b"string":
            val = r.get(raw_key)
            snapshots.append({"key": key, "snapshot": "string", "value": _decode(key.split(":")[1], val) if val and len(val) < 4096 else hashlib.sha1(val or b"").hexdigest()})
        elif kind == b"hash":
            # field names only: the values are digests of image blobs that differ from run to run
            snapshots.append({"key": key, "snapshot": "hash", "value": sorted(f.decode() for f in r.hkeys(raw_key))})

    with out.open("w") as fh:
        for e in entries + snapshots:
            fh.write(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n")
    per_key = collections.Counter(e["key"] for e in entries)
    print(f"recorded {len(entries)} stream entries across {len(per_key)} keys, {len(snapshots)} snapshots -> {out}")
    for k, n in sorted(per_key.items()):
        print(f"  {n:6d}  {k}")
    return 0 if entries else 1


def _read(r, raw_key, cursors, key):
    resp = r.xread({raw_key: cursors.get(key, b"0-0")}, count=500)
    for _stream, items in resp:
        for _id, fields in items:
            cursors[key] = _id
            yield _id, fields


def _frame_id(e: dict, ordinal: int):
    f = e.get("fields", {})
    for name in ("pts", "frame_num"):
        if name in f:
            return (name, f[name])
    return ("n", ordinal)


def _load(path: Path):
    streams: dict[str, dict] = collections.defaultdict(dict)
    snaps: dict[str, dict] = {}
    counts: collections.Counter = collections.Counter()
    for line in path.read_text().splitlines():
        e = json.loads(line)
        if "snapshot" in e:
            if e["key"] not in IGNORED_SNAPSHOTS:
                snaps[e["key"]] = e
            continue
        counts[e["key"]] += 1
        streams[e["key"]][_frame_id(e, counts[e["key"]])] = e["fields"]
    return streams, snaps


def diff(a: Path, b: Path, min_coverage: float, show: int) -> int:
    sa, na = _load(a)
    sb, nb = _load(b)
    bad = 0
    for key in sorted(set(sa) | set(sb)):
        fa, fb = sa.get(key, {}), sb.get(key, {})
        if not fa or not fb:
            print(f"FAIL {key}: present only in {'A' if fa else 'B'} ({len(fa or fb)} entries)")
            bad += 1
            continue
        shared = fa.keys() & fb.keys()
        coverage = len(shared) / max(len(fa), len(fb))
        mism = [(fid, {k: (fa[fid].get(k), fb[fid].get(k)) for k in fa[fid].keys() | fb[fid].keys()
                       if fa[fid].get(k) != fb[fid].get(k)}) for fid in sorted(shared, key=str) if fa[fid] != fb[fid]]
        status = "ok  " if not mism and coverage >= min_coverage else "FAIL"
        print(f"{status} {key}: A={len(fa)} B={len(fb)} shared={len(shared)} coverage={coverage:.3f} mismatched={len(mism)}")
        if status == "FAIL":
            bad += 1
            for fid, d in mism[:show]:
                print(f"       {fid}: {json.dumps(d, ensure_ascii=False)[:300]}")
    for key in sorted(set(na) | set(nb)):
        if na.get(key, {}).get("value") != nb.get(key, {}).get("value"):
            print(f"FAIL snapshot {key}: {str(na.get(key, {}).get('value'))[:150]} != {str(nb.get(key, {}).get('value'))[:150]}")
            bad += 1
    print("IDENTICAL" if not bad else f"{bad} difference(s)")
    return 1 if bad else 0


def _alerts(entries: dict) -> list:
    """(kind-ish signature, pts) per alert entry, in time order."""
    out = []
    for fields in entries.values():
        sig = tuple(str(fields.get(k)) for k in ("kind", "class", "zone", "object_label") if k in fields)
        out.append((sig, float(fields.get("pts", 0.0))))
    return sorted(out, key=lambda x: x[1])


def _close(x, y, tol: float) -> bool:
    """Equal, except that numbers may differ by `tol` relatively (a value learned from tracked objects, such as a fitted
    scale, jitters run to run as the GPU tracker does); strings and structure must match."""
    if isinstance(x, bool) or isinstance(y, bool):
        return x == y
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return abs(x - y) <= tol * max(abs(x), abs(y), 1e-9)
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(_close(x[k], y[k], tol) for k in x)
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(_close(u, v, tol) for u, v in zip(x, y))
    return x == y


def semantic_diff(a: Path, b: Path, tol: float, pts_window: float, slack: int = 2) -> int:
    """For runs that cannot be byte-identical (GPU tracker jitter, wall-clock sustain windows): compare what the
    run MEANT instead of every value. Per stream: entry count and per-label detection totals within `tol` of each
    other (and within `slack` entries, so a stream of two entries is not held to a percentage); alert streams must carry
    the same alerts (kind/class/zone), each within `pts_window` seconds; snapshots equal, numbers within `tol`."""
    sa, na = _load(a)
    sb, nb = _load(b)
    bad = 0
    for key in sorted(set(sa) | set(sb)):
        fa, fb = sa.get(key, {}), sb.get(key, {})
        if not fa or not fb:
            print(f"FAIL {key}: present only in {'A' if fa else 'B'} ({len(fa or fb)} entries)")
            bad += 1
            continue
        if "alert" in key:
            la, lb = _alerts(fa), _alerts(fb)
            ok = len(la) == len(lb) and all(x[0] == y[0] and abs(x[1] - y[1]) <= pts_window for x, y in zip(la, lb))
            print(f"{'ok  ' if ok else 'FAIL'} {key}: A={la} B={lb}")
            bad += not ok
            continue
        msgs = []
        if abs(len(fa) - len(fb)) > max(tol * max(len(fa), len(fb)), slack):
            msgs.append(f"entries {len(fa)} vs {len(fb)}")
        if key.startswith("iris:meta:"):
            def totals(entries):
                t: dict = collections.Counter()
                for f in entries.values():
                    for det in f.get("dets", []):
                        t[det[0]] += 1
                return t
            ta, tb = totals(fa), totals(fb)
            for label in sorted(set(ta) | set(tb)):
                x, y = ta.get(label, 0), tb.get(label, 0)
                if abs(x - y) > max(tol * max(x, y, 1), slack):
                    msgs.append(f"{label} detections {x} vs {y}")
        print(f"{'ok  ' if not msgs else 'FAIL'} {key}: A={len(fa)} B={len(fb)} {'; '.join(msgs)}")
        bad += bool(msgs)
    for key in sorted(set(na) | set(nb)):
        if not _close(na.get(key, {}).get("value"), nb.get(key, {}).get("value"), tol):
            print(f"FAIL snapshot {key}: {str(na.get(key, {}).get('value'))[:150]} != {str(nb.get(key, {}).get('value'))[:150]}")
            bad += 1
    print("SEMANTICALLY EQUAL" if not bad else f"{bad} difference(s)")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--redis", default="redis://localhost:6379/0")
    rec.add_argument("--out", required=True, type=Path)
    rec.add_argument("--idle", type=float, default=20.0, help="seconds without new entries that end the run")
    rec.add_argument("--max-seconds", type=float, default=900.0)
    rec.add_argument("--pattern", default="iris:*")
    d = sub.add_parser("diff")
    d.add_argument("a", type=Path)
    d.add_argument("b", type=Path)
    d.add_argument("--min-coverage", type=float, default=0.98)
    d.add_argument("--show", type=int, default=5)
    d.add_argument("--semantic", action="store_true",
                   help="compare counts, per-label totals and alert timing within tolerances instead of every value")
    d.add_argument("--tol", type=float, default=0.03, help="--semantic: relative tolerance on counts")
    d.add_argument("--pts-window", type=float, default=0.5, help="--semantic: seconds an alert's pts may move")
    d.add_argument("--slack", type=int, default=2, help="--semantic: entries a count may differ by regardless of tol")
    args = ap.parse_args()
    if args.cmd == "record":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        return record(args.redis, args.out, args.idle, args.max_seconds, args.pattern)
    if args.semantic:
        return semantic_diff(args.a, args.b, args.tol, args.pts_window, args.slack)
    return diff(args.a, args.b, args.min_coverage, args.show)


if __name__ == "__main__":
    sys.exit(main())
