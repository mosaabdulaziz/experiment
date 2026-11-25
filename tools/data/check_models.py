#!/usr/bin/env python3
"""Fail when a model or a video the chosen sidecar config uses is under DVC but not pulled yet.

  check_models.py [sidecar.yaml]     default: $SIDECAR_CONFIG, else perception/pipeline/sidecar.yaml

Reads the files the yaml and the files it `extends:` name (.onnx directly, the `onnx-file=` of every nvinfer .txt it names, and file:// video
sources) and looks for a `<file>.dvc` pointer next to each one that is missing. A file with no pointer is not our
business here. The message names the DVC group to pull (data/catalog.yaml).
"""
import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CONTAINER = "/workspace/iris/"
PATH = re.compile(r"""["'\s=](?:file://)?(?:/workspace/iris/)?([\w./-]+\.(?:txt|onnx|pth|yml|yaml|mp4|avi|mkv))\b""")
MEDIA = (".mp4", ".avi", ".mkv")


def local(path):
    return os.path.join(ROOT, path[len(CONTAINER):] if path.startswith(CONTAINER) else path)


EXTENDS = re.compile(r"^extends:\s*[\"']?([^\s\"'#]+)", re.M)


def chain_text(path, seen=None):
    """The text of a settings file and of every file it `extends:`, so a preset that only overrides the sources still
    sees the detector its base names (configs/presets/ppe/live-test.yaml -> sidecar.yaml -> perception/pipeline)."""
    seen = set() if seen is None else seen
    path = os.path.normpath(path)
    if path in seen or not os.path.isfile(path):
        return ""
    seen.add(path)
    with open(path) as f:
        text = f.read()
    for parent in EXTENDS.findall(text):
        text += "\n" + chain_text(os.path.join(os.path.dirname(path), parent), seen)
    return text


def models_in(text):
    for match in PATH.finditer(text):
        path = local(match.group(1))
        if path.endswith(".txt") and os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    key, _, value = line.partition("=")
                    if key.strip() == "onnx-file" and value.strip():
                        yield os.path.join(os.path.dirname(path), value.strip())
        elif path.endswith((".yml", ".yaml")) and os.path.isfile(path):
            with open(path) as f:                       # e.g. products/barber/tracker.yml names its re-ID model
                for inner in PATH.finditer(f.read()):
                    if inner.group(1).endswith((".onnx", ".pth")):
                        yield local(inner.group(1))
        elif path.endswith((".onnx", ".pth") + MEDIA):
            yield path


def groups_for(missing):
    """The DVC groups that own the missing files, or [] when the catalog cannot be read."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import catalog
        owner = catalog.assign(catalog.load())
        return sorted({owner[os.path.relpath(m, ROOT) + ".dvc"] for m in missing if os.path.relpath(m, ROOT) + ".dvc" in owner})
    except Exception:
        return []


def main():
    args = [a for a in sys.argv[1:] if a != "--groups"]
    only_groups = "--groups" in sys.argv[1:]          # print the groups to pull, for tools/data/ensure_data.sh
    cfg = args[0] if args else os.environ.get("SIDECAR_CONFIG") or "perception/pipeline/sidecar.yaml"
    path = local(cfg)
    if not os.path.isfile(path):
        sys.exit(0)  # nothing to inspect; the sidecar reports its own bad config
    text = chain_text(path)
    missing = sorted({m for m in models_in(text) if not os.path.exists(m) and os.path.exists(m + ".dvc")})
    if missing and only_groups:
        print(" ".join(groups_for(missing)))
        sys.exit(1)
    if missing:
        print(f"{cfg} needs files that are versioned in DVC but not pulled:", file=sys.stderr)
        print(*[f"  {os.path.relpath(m, ROOT)}" for m in missing[:12]], sep="\n", file=sys.stderr)
        if len(missing) > 12:
            print(f"  ... and {len(missing) - 12} more", file=sys.stderr)
        groups = groups_for(missing)
        print(f"run: just data-pull {' '.join(groups)}" if groups else "run: just data-pull",
              "  (just data-mount first if the OneDrive remote is the default)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
