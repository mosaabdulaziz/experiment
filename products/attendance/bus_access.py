#!/usr/bin/env python3
"""Bus whitelist: only allowed people may be on the bus (2026-11-02).

Nothing new is detected here -- this drives the restricted-zone machinery attendance already has (zones.py,
violations.py): an access check on the bus camera (the attendance policy's `access_control`), and a role on every
employee. Someone whose role is on the allow-list is fine; a recognised person with any other role (or none) opens a
`role_not_allowed` violation; a body nobody recognises opens an `unidentified` one. Violations show on the
attendance dashboard (alerts page, popup) and in /api/monitoring.

WHERE it applies: inside the restricted zones drawn on the camera (Cameras page), each with its own allowed roles; and
when none is drawn, on the WHOLE stream. This script draws none, so the bus is checked everywhere in the frame.

The whitelist is therefore "who has the role `passenger`". It is applied through the running attendance server's API,
so it takes effect on the next consumer cycle with no restart:

    bus_access.py prepare [--camera cam-22]           # BEFORE the stack starts: switch the saved check off (see below)
    bus_access.py setup [--camera cam-22] [--wait]   # access check on the camera + every employee without a role allowed
    bus_access.py deny  "Person 2"                    # role -> visitor: not on the whitelist
    bus_access.py allow "Person 2"                    # role -> passenger again
    bus_access.py status                              # who is allowed, and the violations of today

`--wait` first waits until every employee has been recognised since the command started: the sidecar loads the face
gallery about 30 s after the stack starts, and until then every body is "unidentified", which would open a violation
for people who are simply not recognised yet. The check is saved in products/attendance/data/attendance_policy.json
and so is live again the moment the attendance server starts, a minute or so before the sidecar has the gallery:
`prepare` takes it out of that file first (the recipe runs it before `just up`). Plain host python (urllib only).
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ALLOWED_ROLE, DENIED_ROLE = "passenger", "visitor"
ZONE_ID = "bus-cabin"
WHOLE_FRAME = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def call(base, method, path, body=None):
    req = urllib.request.Request(base + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path}: HTTP {e.code} {e.read().decode()[:200]}")
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach the attendance server at {base} ({e.reason}) -- is the stack up? (just attendance-bus)")


def employees(base):
    return call(base, "GET", "/api/admin/employees")


def find(base, name):
    matches = [e for e in employees(base) if e["name"].lower() == name.lower() or e["id"] == name]
    if not matches:
        sys.exit(f'no employee "{name}" -- known: {", ".join(e["name"] for e in employees(base))}')
    return matches[0]


def set_role(base, emp, role):
    call(base, "PATCH", f"/api/admin/employees/{emp['id']}", {"role": role})
    print(f"{emp['name']} ({emp['id']}): role {emp.get('role')!r} -> {role!r}"
          f"  [{'allowed' if role == ALLOWED_ROLE else 'NOT allowed'}]")


def wait_recognised(base, since, timeout):
    """Every active employee has a sighting newer than `since` (naive local time, as the API reports it)."""
    deadline = time.time() + timeout
    while True:
        rows = call(base, "GET", "/api/attendance")
        seen = [r for r in rows if r.get("last_departure") and datetime.fromisoformat(r["last_departure"]) >= since]
        if rows and len(seen) == len(rows):
            return True
        if time.time() > deadline:
            missing = [r["name"] for r in rows if r not in seen]
            print(f"WARNING: not recognised after {timeout:.0f}s: {', '.join(missing)} -- continuing anyway")
            return False
        time.sleep(3)


def set_access_check(base, camera, roles):
    """Turn the access check on (roles) or off (None) for one camera, keeping the other cameras' settings."""
    ac = dict(call(base, "GET", "/api/attendance/policy").get("access_control") or {})
    if roles is None:
        ac.pop(camera, None)
    else:
        ac[camera] = {"allowed_roles": roles}
    call(base, "PUT", "/api/attendance/policy", {"access_control": ac})


def cmd_setup(base, args):
    # an older version drew a whole-cabin zone; with a zone drawn the check would only apply inside it
    zones = call(base, "GET", "/api/monitoring/zones").get(args.camera, [])
    others = [z for z in zones if z.get("id") != ZONE_ID]
    if len(others) != len(zones):
        call(base, "PUT", f"/api/monitoring/zones/{args.camera}", others)
    if args.wait:
        set_access_check(base, args.camera, None)          # still on from an earlier run? off until recognised
        print("waiting for everyone to be recognised (the gallery loads ~30 s after start)...")
        wait_recognised(base, args.since, args.timeout)
    set_access_check(base, args.camera, [ALLOWED_ROLE])
    drawn = [z["id"] for z in others if z.get("type") == "restricted"]
    print(f"{args.camera}: access check on, allowed roles [{ALLOWED_ROLE}], applies to "
          + (f"the drawn zone(s) {drawn}" if drawn else "the WHOLE STREAM (no zone drawn)"))
    for e in employees(base):
        if not e.get("role"):
            set_role(base, e, ALLOWED_ROLE)


def cmd_prepare(args):
    data = Path(__file__).resolve().parent / "data"
    pol = data / "attendance_policy.json"
    if pol.is_file():
        policy = json.loads(pol.read_text())
        if args.camera in (policy.get("access_control") or {}):
            del policy["access_control"][args.camera]
            pol.write_text(json.dumps(policy, indent=2, ensure_ascii=False))
            print(f"{args.camera}: access check off until everyone is recognised")
    zf = data / "zones.json"
    if zf.is_file():                                        # a cabin zone saved by an older version
        zones = json.loads(zf.read_text())
        kept = [z for z in zones.get(args.camera, []) if z.get("id") != ZONE_ID]
        if len(kept) != len(zones.get(args.camera, [])):
            zones[args.camera] = kept
            zf.write_text(json.dumps(zones, indent=2, ensure_ascii=False))


def cmd_status(base, args):
    for e in sorted(employees(base), key=lambda e: e["id"]):
        role = e.get("role")
        print(f"{e['id']:<8} {e['name']:<14} role={role or '-':<10} "
              f"{'allowed' if role == ALLOWED_ROLE else 'NOT allowed'}")
    mon = call(base, "GET", "/api/monitoring")
    drawn = {cam: [z["id"] for z in zs if z.get("type") == "restricted"] for cam, zs in mon["zones"].items()
             if any(z.get("type") == "restricted" for z in zs)}
    for cam, cfg in (call(base, "GET", "/api/attendance/policy").get("access_control") or {}).items():
        print(f"access check {cam}: allowed roles {cfg.get('allowed_roles')}, applies to "
              + (f"zones {drawn[cam]}" if cam in drawn else "the whole stream"))
    print("restricted zones drawn:", drawn or "none")
    rows = mon["violations"]
    print(f"violations today: {len(rows)}")
    for v in rows[:10]:
        who = v.get("name") or "unidentified person"
        print(f"  {v['started_at']}  {v['violation_type']:<16} {who:<14} {v['camera']}/{v['zone_id']}  "
              f"{'OPEN' if v['open'] else 'closed'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://localhost:8001", help="attendance server")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--camera", default="cam-22")
    s.add_argument("--wait", action="store_true")
    s.add_argument("--timeout", type=float, default=120.0)
    for name in ("allow", "deny"):
        p = sub.add_parser(name)
        p.add_argument("who", help='employee name or id, e.g. "Person 2"')
    sub.add_parser("status")
    pp = sub.add_parser("prepare")
    pp.add_argument("--camera", default="cam-22")
    args = ap.parse_args()
    args.since = datetime.now()
    if args.cmd == "prepare":
        cmd_prepare(args)
    elif args.cmd == "setup":
        cmd_setup(args.base, args)
    elif args.cmd == "status":
        cmd_status(args.base, args)
    else:
        set_role(args.base, find(args.base, args.who), ALLOWED_ROLE if args.cmd == "allow" else DENIED_ROLE)


if __name__ == "__main__":
    main()
