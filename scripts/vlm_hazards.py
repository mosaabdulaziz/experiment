#!/usr/bin/env python3
"""Send images to a Qwen VLM (DashScope, OpenAI-compatible) and draw the hazards it finds.

Usage: python scripts/vlm_hazards.py [IMAGE_OR_DIR ...]   (default: analysis/)
Reads QWEN_API_KEY / QWEN_BASE_URL / QWEN_MODEL from .env.
Writes <out>/<name>_hazards.png and <out>/<name>_hazards.json (default out: analysis/annotated).

The prompt, the call, the parsing and the drawing live in app/hazards.py (shared with the Hazard Detection tab's API).
"""
import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.hazards import HazardError, annotate, ask as _ask, settings  # noqa: E402,F401  (annotate re-exported for callers)


def load_env(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def ask(img, model, base_url, key):
    try:
        return _ask(img, model, base_url, key)
    except HazardError as exc:
        sys.exit(str(exc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", default=[str(ROOT / "analysis")])
    ap.add_argument("--out", default=str(ROOT / "analysis" / "annotated"))
    a = ap.parse_args()
    load_env(ROOT / ".env")
    cfg = settings()
    if cfg is None:
        sys.exit("QWEN_API_KEY / QWEN_BASE_URL / QWEN_MODEL missing from .env")
    files = []
    for p in map(Path, a.inputs):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")) if p.is_dir() else [p]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        print(f"{f.name} -> {cfg.model} ...", flush=True)
        img = Image.open(f)
        res = ask(img, cfg.model, cfg.base_url, cfg.key)
        stem = f.stem.replace(" ", "_")
        annotate(img, res).save(out / f"{stem}_hazards.png")
        (out / f"{stem}_hazards.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
        for i, hz in enumerate(res.get("hazards", []), 1):
            print(f"  {i}. [{hz.get('severity')}] {hz.get('label')}: {hz.get('description')}")


if __name__ == "__main__":
    main()
