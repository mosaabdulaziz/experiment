#!/usr/bin/env python3
"""Fail when a .dvc pointer (or dvc.lock) names data that is not on the remote yet: someone forgot `just data-push`.

  check_pushed.py --rclone iris-onedrive:IRIS/repo-data/dvc-store   CI: list the store with rclone
  check_pushed.py --dir /path/to/dvc-store              a local directory remote (tests, a mounted store)

Once the remote is MinIO, `dvc status -c` does the same job and this script is not needed.
"""
import argparse
import json
import os
import subprocess
import sys

import yaml

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SKIP_DIRS = {".git", ".dvc", "node_modules", "helper-repos", ".data-remote", "frontend", "scratch"}


def pointers():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.endswith(".dvc") or (name == "dvc.lock" and dirpath == ROOT):
                yield os.path.join(dirpath, name)


def wanted_hashes(path):
    """md5 of every out (file or dir manifest) a pointer or lock file names."""
    with open(path) as f:
        doc = yaml.safe_load(f)
    outs = doc.get("outs", [])
    for stage in (doc.get("stages") or {}).values():
        outs += stage.get("outs", [])
    return [o["md5"] for o in outs if o.get("md5")]


def object_path(md5):
    return f"files/md5/{md5[:2]}/{md5[2:]}"


class Store:
    def __init__(self, rclone=None, directory=None):
        self.rclone, self.directory = rclone, directory
        self.present = set(self._list())

    def _list(self):
        if self.directory:
            for dirpath, _, names in os.walk(self.directory):
                for n in names:
                    yield os.path.relpath(os.path.join(dirpath, n), self.directory)
        else:
            out = subprocess.run(["rclone", "lsf", "-R", "--files-only", self.rclone], check=True, capture_output=True, text=True)
            yield from out.stdout.splitlines()

    def read(self, rel):
        if self.directory:
            with open(os.path.join(self.directory, rel)) as f:
                return f.read()
        return subprocess.run(["rclone", "cat", f"{self.rclone}/{rel}"], check=True, capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rclone")
    g.add_argument("--dir")
    args = ap.parse_args()
    store = Store(args.rclone, args.dir)

    missing = []
    for path in sorted(pointers()):
        for md5 in wanted_hashes(path):
            rel = object_path(md5)
            if rel not in store.present:
                missing.append((os.path.relpath(path, ROOT), rel))
            elif md5.endswith(".dir"):
                for member in json.loads(store.read(rel)):
                    if object_path(member["md5"]) not in store.present:
                        missing.append((os.path.relpath(path, ROOT), f"{member['relpath']} ({member['md5']})"))
    if missing:
        print(f"{len(missing)} object(s) named in git are not on the remote -- run `just data-push`:", file=sys.stderr)
        for pointer, what in missing[:50]:
            print(f"  {pointer}: {what}", file=sys.stderr)
        sys.exit(1)
    print(f"ok: every pointer's data is on the remote ({len(store.present)} objects)")


if __name__ == "__main__":
    main()
