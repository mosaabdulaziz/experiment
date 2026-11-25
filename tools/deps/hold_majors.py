#!/usr/bin/env python3
"""Keep major version changes out of automated dependency updates (DEPENDENCIES.md, "Updates").

  hold_majors.py [--base REF]           re-lock every package that crossed a major version (or is HELD) back to its
                                        version in REF's lock, letting everything else keep its update
  hold_majors.py --check [--base REF]   exit 1 if any lock has such a change against REF
  hold_majors.py --report [--base REF]  list every package each lock changes against REF, and how (for review)

Renovate's lock-file maintenance runs `uv lock --upgrade`, which can move any package, transitive ones and the sidecar's
unversioned requirements included, to a new major version; its "no majors" rules only see direct dependencies. Renovate
runs this after every update (postUpgradeTasks), and CI runs it with --check on Renovate's merge requests. A major update
is done by hand, deliberately, in its own merge request (`just deps-update`).

"Major" is the first release number, or the first two for 0.x versions (0.4 -> 0.5 may break). Calendar versions
(2026.x, a first number >= 1000) are never held: a new year is not a breaking change.
"""
import argparse
import os
import re
import subprocess
import sys
import tomllib

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
LOCKS = [".", "perception", "products/attendance", "products/barber", "products/fall_detection",
         "tools/onnx-export", "tools/mtmc-research", "tools/mlflow-server"]
# Held at their exact version: the sidecar's cuDNN workaround and the app Dockerfile's CPU pin are tied to them.
HELD = {"torch", "torchvision"}


def major(version):
    release = [int(x) for x in re.match(r"(\d+(?:\.\d+)*)", version).group(1).split(".")]
    if release[0] >= 1000:
        return None
    return tuple(release[:2]) if release[0] == 0 else (release[0],)


def versions(text):
    """name -> set of locked versions (a package can have one per platform, e.g. torch 2.14.0 and 2.14.0+cpu)."""
    out = {}
    for p in tomllib.loads(text)["package"]:
        if "version" in p:
            out.setdefault(p["name"], set()).add(p["version"])
    return out


def base_lock(ref, lock_dir):
    path = os.path.normpath(os.path.join(lock_dir, "uv.lock"))
    r = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def violations(base, new):
    """[(name, base version to go back to, why)] for every package that crossed a major or is HELD and changed."""
    out = []
    for name, now in sorted(new.items()):
        before = base.get(name)
        if not before or now == before:
            continue
        back = max(before, key=_key)
        if name in HELD:
            out.append((name, back.split("+")[0], f"held: {sorted(before)} -> {sorted(now)}"))
            continue
        old_majors = {major(v) for v in before} - {None}
        new_majors = {major(v) for v in now} - {None}
        if new_majors - old_majors and old_majors and max(new_majors) > max(old_majors):
            out.append((name, back.split("+")[0], f"major: {sorted(before)} -> {sorted(now)}"))
    return out


def kind(before, now):
    if not before:
        return "added"
    if not now:
        return "removed"
    b, n = max(before, key=_key), max(now, key=_key)
    if major(b) != major(n) and None not in (major(b), major(n)):
        return "MAJOR"
    rb, rn = _key(b), _key(n)
    return "minor" if rb[:2] != rn[:2] else "patch"


def _key(v):
    return [int(x) for x in re.findall(r"\d+", v.split("+")[0])]


def report(ref):
    for lock_dir in LOCKS:
        base_text = base_lock(ref, lock_dir)
        with open(os.path.join(ROOT, lock_dir, "uv.lock")) as f:
            new = versions(f.read())
        base = versions(base_text) if base_text else {}
        rows = [(name, base.get(name, set()), new.get(name, set())) for name in sorted(set(base) | set(new))
                if base.get(name) != new.get(name)]
        if rows:
            print(f"== {lock_dir}/uv.lock: {len(rows)} change(s) against {ref}")
            for name, before, now in rows:
                tag = "HELD" if name in HELD else kind(before, now)
                print(f"  {tag:7} {name:32} {', '.join(sorted(before)) or '-':>20} -> {', '.join(sorted(now)) or '-'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--report", action="store_true")
    ap.add_argument("--base", default="HEAD", help="the git ref whose locks are the starting point (default HEAD)")
    args = ap.parse_args()
    if args.report:
        report(args.base)
        return 0
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    failed = False
    for lock_dir in LOCKS:
        base_text = base_lock(args.base, lock_dir)
        if base_text is None:                              # a lock that is new in this change: nothing to hold
            continue
        base = versions(base_text)
        for attempt in range(4):
            with open(os.path.join(ROOT, lock_dir, "uv.lock")) as f:
                found = violations(base, versions(f.read()))
            if not found:
                break
            for name, back, why in found:
                print(f"{lock_dir}/uv.lock: {name} {why}", file=sys.stderr)
            if args.check or attempt == 3:
                failed = True
                break
            cmd = ["uv", "lock", "--quiet"] + [a for name, back, _ in found for a in ("--upgrade-package", f"{name}=={back}")]
            print(f"{lock_dir}: {' '.join(cmd)}", file=sys.stderr)
            subprocess.run(cmd, cwd=os.path.join(ROOT, lock_dir), env=env, check=True)
    if failed:
        print("a lock still crosses a major version (or moves a held package): do it by hand in its own merge request "
              "(DEPENDENCIES.md, \"Updates\")", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
