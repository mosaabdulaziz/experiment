#!/usr/bin/env python3
"""Extract vehicle features from crop images using Qwen2.5VL via Ollama.

Hardened against the known ollama qwen2.5-vl vision flakiness (ollama#15828):
garbage/"?" outputs and OOM that worsen with sustained use. Mitigations:
- JSON schema output via format= (no prompt injection, no parse errors)
- keep_alive=0 so the model is unloaded after every request (fresh state)
- bounded image size (downscale oversized crops)
- retry with model unload on garbage/error responses
"""

import json
import base64
import io
import subprocess
import sys
from pathlib import Path

from PIL import Image
from ollama import chat

CROPS_DIR = Path(__file__).parent / "output" / "vehicle_crops" / "chosen crops"
OUTPUT_FILE = Path(__file__).parent / "vehicle_features.json"
MODEL = "qwen2.5vl:3b"
MAX_SIDE = 1280  # downscale oversized crops to keep vision tokens bounded
SHRINK_STEPS = [1280, 1024, 896, 768, 640]  # progressively shrink and retry on failure

PROMPT = (
    "Analyze this vehicle crop image. It may be a close-up or partial view taken "
    "by a surveillance camera. Extract every vehicle feature you can confidently "
    "determine. If a feature cannot be determined, return a sensible default or "
    "null. This data will be used to match the same vehicle across different "
    "cameras, so pay attention to distinguishing details."
)

VEHICLE_SCHEMA = {
    "type": "object",
    "properties": {
        "vehicle_type": {"type": "string", "enum": ["car", "truck", "van", "bus", "motorcycle", "suv", "pickup", "other"]},
        "vehicle_subtype": {"type": "string", "enum": ["sedan", "hatchback", "suv", "pickup_truck", "box_truck", "flatbed", "tanker", "container", "bus", "van", "motorcycle", "other"]},
        "color": {"type": "string", "description": "primary color, e.g. white, black, red, blue, silver, gray, green"},
        "color_secondary": {"type": ["string", "null"]},
        "has_cargo": {"type": "boolean"},
        "cargo_type": {"type": "string", "enum": ["none", "covered", "open", "container", "tanker", "flatbed_load", "other"]},
        "has_trailer": {"type": "boolean"},
        "trailer_type": {"type": "string", "enum": ["none", "box", "flatbed", "tanker", "container", "other"]},
        "has_roof_rack": {"type": "boolean"},
        "has_cover": {"type": "boolean"},
        "license_plate_visible": {"type": "boolean"},
        "condition": {"type": "string", "enum": ["good", "fair", "poor", "damaged"]},
        "estimated_size": {"type": "string", "enum": ["small", "medium", "large", "extra_large"]},
        "distinguishing_marks": {"type": "string", "description": "visible marks, stickers, damage, brand logos, or unique features"},
        "confidence": {"type": "number", "description": "0.0 to 1.0, overall clarity/confidence"},
    },
    "required": ["vehicle_type", "vehicle_subtype", "color", "color_secondary", "has_cargo", "cargo_type", "has_trailer", "trailer_type", "has_roof_rack", "has_cover", "license_plate_visible", "condition", "estimated_size", "distinguishing_marks", "confidence"],
}


def to_b64(path: Path, max_side: int) -> str:
    img = Image.open(path)
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def is_garbage(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) > 0 and set(stripped) <= {"?", "\ufffd", " "}


def unload() -> None:
    try:
        subprocess.run(["ollama", "stop", MODEL], capture_output=True, timeout=30)
    except Exception:
        pass


def analyze_image(image_path: Path) -> dict:
    last_err = None
    for max_side in SHRINK_STEPS:
        b64 = to_b64(image_path, max_side)
        for attempt in range(3):
            try:
                response = chat(
                    model=MODEL,
                    messages=[{"role": "user", "content": PROMPT, "images": [b64]}],
                    format=VEHICLE_SCHEMA,
                    options={"temperature": 0.0, "num_ctx": 8192},
                    keep_alive=0,
                )
                text = response.message.content
                if is_garbage(text):
                    unload()
                    continue
                return json.loads(text)
            except json.JSONDecodeError:
                unload()
                continue
            except Exception as e:
                last_err = str(e)
                unload()
                continue
    # Last resort: free-form prompt without schema enforcement
    try:
        b64 = to_b64(image_path, SHRINK_STEPS[-1])
        response = chat(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "Describe this vehicle crop and answer in VALID JSON matching exactly: "
                    "{\"vehicle_type\":\"...\",\"vehicle_subtype\":\"...\",\"color\":\"...\","
                    "\"color_secondary\":null,\"has_cargo\":true,\"cargo_type\":\"...\","
                    "\"has_trailer\":false,\"trailer_type\":\"none\",\"has_roof_rack\":false,"
                    "\"has_cover\":false,\"license_plate_visible\":false,\"condition\":\"good\","
                    "\"estimated_size\":\"medium\",\"distinguishing_marks\":\"...\",\"confidence\":0.0}"
                    " No other text."
                ),
                "images": [b64],
            }],
            options={"temperature": 0.0, "num_ctx": 8192},
            keep_alive=0,
        )
        text = response.message.content
        if is_garbage(text):
            return {"error": "failed across all schema and free-form attempts"}
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        return json.loads(text)
    except Exception:
        return {"error": "failed across all schema and free-form attempts"}


def main():
    vehicle_dirs = sorted(
        [d for d in CROPS_DIR.iterdir() if d.is_dir() and d.name.startswith("vehicle_")]
    )
    total = sum(1 for d in vehicle_dirs for _ in d.glob("*.jpg"))
    print(f"Found {len(vehicle_dirs)} vehicle folders, {total} images total")
    print(f"Using model: {MODEL}\n", flush=True)

    results = {}
    idx = 0
    for vdir in vehicle_dirs:
        vid = vdir.name
        results[vid] = {}
        for img in sorted(vdir.glob("*.jpg")):
            idx += 1
            print(f"[{idx}/{total}] {vid}/{img.name} ... ", end="", flush=True)
            f = analyze_image(img)
            if "error" in f:
                print(f"FAILED: {f['error']} {f.get('detail') or ''}")
                results[vid][img.name] = f
            else:
                results[vid][img.name] = f
                vtype = f.get("vehicle_type", "?")
                color = f.get("color", "?")
                cargo = "cargo" if f.get("has_cargo") else "no-cargo"
                trailer = "trailer" if f.get("has_trailer") else "no-trailer"
                print(f"{vtype} | {color} | {cargo} | {trailer}", flush=True)

    with open(OUTPUT_FILE, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")

    fails = [(v, i) for v in results for i, ft in results[v].items() if "error" in ft]
    print(f"Total images: {total}, failed: {len(fails)}")
    for v, i in fails:
        print(f"  {v}/{i}")


if __name__ == "__main__":
    main()