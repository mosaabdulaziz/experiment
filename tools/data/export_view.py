#!/usr/bin/env python3
"""A browsable, per-use-case view of the DVC store.

The store is content-addressed (files/md5/ab/cdef...), which is right for DVC and useless for a person. This builds a
second tree next to it, `<use case>/<kind>/<the file's path in the repo>`, out of hardlinks to the store's objects, so it
costs no disk space. It is read-only reference: DVC never reads it, and it is rebuilt from scratch each time.

  export_view.py manifest > manifest.json                         in the repo: what goes where (needs PyYAML)
  export_view.py apply --store S --out O --manifest manifest.json  next to the store, e.g. on the server (standard library only)

`just data-view` does both: it copies this file and the manifest next to the store on the server and runs it there.
"""
import argparse
import json
import os
import shutil
import sys

MARKER = ".iris-view"


def manifest():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import catalog
    import yaml
    groups = catalog.load()
    owner = catalog.assign(groups)
    entries = []
    for pointer, group in sorted(owner.items()):
        with open(os.path.join(catalog.ROOT, pointer)) as f:
            out = yaml.safe_load(f)["outs"][0]
        entries.append({"folder": groups[group]["folder"], "path": os.path.join(os.path.dirname(pointer), out["path"]),
                        "md5": out["md5"]})
    # the golden_clips stage outputs are not .dvc pointers: take them from dvc.lock
    with open(os.path.join(catalog.ROOT, "dvc.lock")) as f:
        lock = yaml.safe_load(f)
    for group, g in groups.items():
        for stage in g.get("stages", []):
            for out in lock["stages"][stage]["outs"]:
                entries.append({"folder": g["folder"], "path": out["path"], "md5": out["md5"]})
    return entries


def obj(store, md5):
    return os.path.join(store, "files", "md5", md5[:2], md5[2:])


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:                     # another filesystem: fall back to a symlink rather than copy gigabytes
        os.symlink(src, dst)


def apply(store, out, entries):
    store, out = os.path.abspath(store), os.path.abspath(out)
    if out == store or out.startswith(store + os.sep) or store.startswith(out + os.sep):
        sys.exit(f"refusing: --out ({out}) overlaps the store ({store})")
    marker = os.path.join(out, MARKER)
    if os.path.isdir(out) and os.listdir(out) and not os.path.exists(marker):
        sys.exit(f"refusing: {out} is not empty and was not made by this tool (no {MARKER})")
    if os.path.isdir(out):
        for name in os.listdir(out):    # rebuilt from scratch; only this view's own links are in here
            path = os.path.join(out, name)
            shutil.rmtree(path) if os.path.isdir(path) and not os.path.islink(path) else os.remove(path)
    os.makedirs(out, exist_ok=True)
    with open(marker, "w") as f:
        f.write("made by tools/data/export_view.py; safe to delete, rebuilt by `just data-view`\n")
    missing, count = [], 0
    for e in entries:
        dst_root = os.path.join(out, e["folder"], e["path"])
        if e["md5"].endswith(".dir"):
            manifest_path = obj(store, e["md5"])
            if not os.path.exists(manifest_path):
                missing.append(e["path"]); continue
            with open(manifest_path) as f:
                members = json.load(f)
            for m in members:
                src = obj(store, m["md5"])
                if os.path.exists(src):
                    link(src, os.path.join(dst_root, m["relpath"])); count += 1
                else:
                    missing.append(os.path.join(e["path"], m["relpath"]))
        else:
            src = obj(store, e["md5"])
            if os.path.exists(src):
                link(src, dst_root); count += 1
            else:
                missing.append(e["path"])
    print(f"view: {count} files linked under {out}")
    if missing:
        print(f"{len(missing)} not in the store yet (not pushed?):", *missing[:20], sep="\n  ", file=sys.stderr)
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest")
    a = sub.add_parser("apply")
    a.add_argument("--store", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--manifest", required=True)
    args = ap.parse_args()
    if args.cmd == "manifest":
        json.dump(manifest(), sys.stdout, indent=1)
    else:
        with open(args.manifest) as f:
            apply(args.store, args.out, json.load(f))


if __name__ == "__main__":
    main()
