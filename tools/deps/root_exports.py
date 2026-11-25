#!/usr/bin/env python3
"""Write requirements.txt, requirements-test.txt and requirements-dev.txt from the root pyproject.toml and uv.lock.

  root_exports.py          rewrite the three files
  root_exports.py --check  exit 1 when one is not what the lock says

The root pyproject.toml is the source; these files are thin exports for what still reads a requirements file (the app
Dockerfile, `pip install -r`): each requirement of the group, in the group's order, pinned to its version in uv.lock.
Only the direct requirements are listed, so the app Dockerfile's opencv sed and its install order keep working; an
environment that must match the lock exactly installs with `uv sync --locked` (DEPENDENCIES.md).
"""
import argparse
import os
import re
import sys
import tomllib

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PYPROJECT = os.path.join(ROOT, "pyproject.toml")
LOCK = os.path.join(ROOT, "uv.lock")
EXPORTS = {"requirements.txt": "app", "requirements-test.txt": "test", "requirements-dev.txt": "dev"}


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_versions():
    with open(LOCK, "rb") as f:
        lock = tomllib.load(f)
    versions = {}
    for p in lock["package"]:
        if "version" in p:
            versions.setdefault(norm(p["name"]), set()).add(p["version"])
    return versions


def render(group, groups, versions):
    lines = [f"# Generated from pyproject.toml (group `{group}`) and uv.lock by tools/deps/root_exports.py. Do not edit:",
             "# change the group in pyproject.toml, then `just deps-lock`. DEPENDENCIES.md."]
    for req in groups[group]:
        if isinstance(req, dict):
            sys.exit(f"pyproject.toml: group {group!r} includes another group; an export lists plain requirements only")
        m = re.match(r"\s*([A-Za-z0-9._-]+)(\[[^\]]*\])?", req)
        name, extras = m.group(1), m.group(2) or ""
        found = versions.get(norm(name), set())
        if len(found) != 1:
            sys.exit(f"uv.lock has {sorted(found) or 'no'} version(s) of {name}: cannot pin it in one line")
        lines.append(f"{name}{extras}=={next(iter(found))}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    with open(PYPROJECT, "rb") as f:
        groups = tomllib.load(f)["dependency-groups"]
    versions = locked_versions()
    stale = []
    for path, group in EXPORTS.items():
        full = os.path.join(ROOT, path)
        text = render(group, groups, versions)
        old = open(full).read() if os.path.exists(full) else None
        if old == text:
            continue
        stale.append(path)
        if not args.check:
            with open(full, "w") as f:
                f.write(text)
            print(f"{path}: rewritten from group `{group}`")
    if args.check and stale:
        print(f"out of date with pyproject.toml / uv.lock: {', '.join(stale)}: run `just deps-lock`", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
