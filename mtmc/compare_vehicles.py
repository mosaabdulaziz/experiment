#!/usr/bin/env python3
"""Pairwise vehicle matching using Qwen2.5VL via Ollama.

Given two vehicle crop images, asks the vision model whether they show the SAME
vehicle. Hardened the same way as extract_vehicle_features.py (JSON schema,
keep_alive=KEEP_ALIVE, shrink+retry).

Comparisons performed:
- explicit pairs (user-supplied, e.g. vehicle_6/t0038_f00726 vs vehicle_5/t0008_f00187)
- within-folder: representative crop of a vehicle vs every other crop of the same vehicle
  (different camera views of the same physical vehicle -> expect SAME)
- cross-folder: representative crop of one vehicle vs representative of another
  (different physical vehicles -> expect DIFFERENT)
"""

import json
import base64
import io
import os
from pathlib import Path

from PIL import Image
from ollama import Client

BASE = Path(__file__).parent / "output" / "vehicle_crops" / "chosen crops"
OUTPUT_FILE = Path(__file__).parent / "vehicle_comparisons.json"
MODEL = "qwen3-vl:4b"   # module-global; overridden by main() --model
HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
NUM_CTX = 8192
KEEP_ALIVE = -1          # keep the model resident across calls (huge speedup on
                         # batch runs; -1 = until ollama stops it)
USE_FORMAT = False       # if True, force the JSON grammar (format=COMPARE_SCHEMA) like the
                         # original baseline run; if False, rely on the prompt instruction
SHRINK_STEPS = [1024, 896, 768, 640]

PROMPT = (
    "You are a vehicle matching system for a multi-camera surveillance setup. "
    "Image 1 and Image 2 are crops of vehicles captured by (possibly different) cameras "
    "looking at the same scene from DIFFERENT VIEW ANGLES and DISTANCES. "
    "Determine whether they show the SAME physical vehicle.\n"
    "Compare visible traits: color(s), body type/silhouette, brand/logo, grille, lights, "
    "window/decal layout, cargo, trailer, roof rack, visible marks, damage, stickers.\n"
    "MULTI-CAMERA RULES (missing trait is NOT a contradiction):\n"
    "- A brand/logo/text visible in one crop but NOT in the other is simply out of frame "
    "  or too small at that angle — never treat 'brand not visible' as evidence of a "
    "  different vehicle. Only a clearly DIFFERENT brand in the same visible location counts.\n"
    "- A vehicle appearing smaller in one crop just means that camera was farther away — "
    "  never call it a 'smaller vehicle' because of zoom/distance.\n"
    "- A crop may show only PART of the vehicle (e.g. just the cab, or a corner). Compare "
    "  the parts that ARE visible; if the visible parts are compatible, they can still match.\n"
    "DECIDE:\n"
    "- 'same': visible attributes are compatible and no genuinely incompatible trait is seen.\n"
    "- 'different': a GENUINELY incompatible trait is visible (clearly different color, "
    "  different body type/silhouette, clearly different brand in the same location, "
    "  incompatible grille/lights/window layout).\n"
    "- 'uncertain': you cannot decide from these crops — too little detail, too occluded, "
    "  or only generic portions visible (e.g. both crops show only a plain white box wall); "
    "  another crop/view would be needed to decide. Prefer 'uncertain' over forcing a wrong "
    "  answer when the evidence is insufficient."
)

COMPARE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["same", "different", "uncertain"]},
        "confidence": {"type": "number", "description": "0.0 to 1.0"},
        "matched_features": {"type": "array", "items": {"type": "string"}, "description": "traits that are compatible/consistent between the two, e.g. 'color: white/red', 'same grille style', 'body type: truck'. 'brand not visible in image 2' is NOT a conflict: write it under a note not here."},
        "conflicting_features": {"type": "array", "items": {"type": "string"}, "description": "only GENUINE conflicts, e.g. 'different body type (sedan vs truck)', 'clearly different grille', 'different color'. 'brand not visible on image 2' is absolutely NOT a conflict."},
        "note_missing_visibility": {"type": "array", "items": {"type": "string"}, "description": "traits visible in one image but not in the other (e.g. 'brand not in image 2', 'trailer not visible in image 1') — recorded but NOT treated as conflicts"},
        "explanation": {"type": "string", "description": "brief verdict reasoning"},
    },
    "required": ["decision", "confidence", "matched_features", "conflicting_features", "explanation"],
}


def to_b64(rel: str, max_side: int) -> str:
    img = Image.open(BASE / rel)
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def is_garbage(text: str) -> bool:
    s = text.strip()
    return len(s) > 0 and set(s) <= {"?", "\ufffd", " "}


def parse_verdict(text: str):
    """Parse the model's JSON output, tolerating markdown fences, stray prose or a bare verdict word."""
    if not text:
        return None
    s = text.strip()
    bare = s.strip(" .\n\t\"'`")
    if bare.lower() in ("same", "different", "uncertain"):
        return {"decision": bare.lower(), "confidence": None, "explanation": "bare verdict (no JSON)"}
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _client():
    return Client(host=HOST)


def unload() -> None:
    try:
        _client().stop(MODEL)
    except Exception:
        pass


def stacked_b64(a: str, b: str, max_side: int) -> str:
    """Combine two crops side-by-side into ONE image (workaround for the flaky 2-image path)."""
    im1 = Image.open(BASE / a).convert("RGB")
    im2 = Image.open(BASE / b).convert("RGB")
    step = max_side / 2
    im1.thumbnail((step, step), Image.LANCZOS)
    im2.thumbnail((step, step), Image.LANCZOS)
    H = max(im1.height, im2.height)
    cat = Image.new("RGB", (im1.width + im2.width, H), (0, 0, 0))
    cat.paste(im1, (0, int((H - im1.height) / 2)))
    cat.paste(im2, (im1.width, int((H - im2.height) / 2)))
    buf = io.BytesIO()
    cat.save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def compare(a: str, b: str) -> dict:
    """Compare two crop paths; returns matching verdict dict or error dict."""
    last_err = None
    prompt_suffix = ("\nImage 1 is the first attached image, Image 2 is the second attached image."
                     if USE_FORMAT else
                     "\nImage 1 is the first attached image, Image 2 is the second attached image. Return ONLY valid JSON matching the required schema.")
    for side in SHRINK_STEPS:
        b64a, b64b = to_b64(a, side), to_b64(b, side)
        for _ in range(3):
            try:
                response = _client().chat(
                    model=MODEL,
                    messages=[{
                        "role": "user",
                        "content": PROMPT + prompt_suffix,
                        "images": [b64a, b64b],
                    }],
                    options={"temperature": 0.0, "num_ctx": NUM_CTX},
                    format=COMPARE_SCHEMA if USE_FORMAT else None,
                    keep_alive=KEEP_ALIVE,
                )
                text = response.message.content
                if is_garbage(text):
                    unload()
                    continue
                verdict = parse_verdict(text)
                if verdict is not None:
                    return verdict
                unload()
                continue
            except json.JSONDecodeError:
                unload()
                continue
            except Exception as e:
                last_err = str(e)
                unload()
                continue
    # Flaky 2-image path failed; fall back to a single stacked image.
    return compare_stacked(a, b, last_err)


def compare_stacked(a: str, b: str, last_err: str = None) -> dict:
    """Fallback: render both crops side-by-side in one image so a single vision call is used."""
    prompt_suffix = ("\nThe two crops are shown side by side in ONE image: LEFT half is Image 1, RIGHT half is Image 2."
                     if USE_FORMAT else
                     "\nThe two crops are shown side by side in ONE image: LEFT half is Image 1, RIGHT half is Image 2. Return ONLY valid JSON matching the required schema.")
    for side in SHRINK_STEPS:
        for _ in range(3):
            try:
                response = _client().chat(
                    model=MODEL,
                    messages=[{
                        "role": "user",
                        "content": PROMPT + prompt_suffix,
                        "images": [stacked_b64(a, b, side)],
                    }],
                    options={"temperature": 0.0, "num_ctx": NUM_CTX},
                    format=COMPARE_SCHEMA if USE_FORMAT else None,
                    keep_alive=KEEP_ALIVE,
                )
                text = response.message.content
                if is_garbage(text):
                    unload()
                    continue
                verdict = parse_verdict(text)
                if verdict is not None:
                    verdict.setdefault("fallback", "stacked-single-image")
                    return verdict
                unload()
                continue
            except Exception as e:
                last_err = str(e)
                unload()
                continue
    return {"error": f"failed in stacked fallback too ({last_err})"}


def images_in(folder: str):
    return sorted((BASE / folder).glob("*.jpg"))


def representative(folder: str) -> Path:
    """Pick the largest crop of a folder as the representative clear view."""
    return max(images_in(folder), key=lambda p: p.stat().st_size)


def rel_of(p: Path) -> str:
    return f"{p.parent.name}/{p.name}"


def load_existing() -> dict:
    """Load previously computed comparisons so interrupted runs can resume."""
    if not OUTPUT_FILE.exists():
        return {}
    try:
        data = json.loads(OUTPUT_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not data or "within_vehicle" not in data:
        return {}
    return data


def main():
    global MODEL, OUTPUT_FILE, HOST, NUM_CTX, KEEP_ALIVE, USE_FORMAT
    import argparse
    ap = argparse.ArgumentParser(description="Pairwise vehicle VLM comparisons")
    ap.add_argument("--model", default="qwen3-vl:4b")
    ap.add_argument("--out", default=str(OUTPUT_FILE))
    ap.add_argument("--host", default=HOST, help="ollama server base URL")
    ap.add_argument("--num-ctx", type=int, default=NUM_CTX,
                    help="context window size (tokens) for each call")
    ap.add_argument("--keep-alive", type=float, default=KEEP_ALIVE,
                    help="seconds model stays loaded after a call (-1 = forever)")
    ap.add_argument("--format", action="store_true",
                    help="force the JSON grammar (format=COMPARE_SCHEMA) like the baseline run")
    args = ap.parse_args()
    MODEL = args.model
    OUTPUT_FILE = Path(args.out)
    HOST = args.host
    NUM_CTX = args.num_ctx
    KEEP_ALIVE = args.keep_alive
    USE_FORMAT = args.format

    folders = sorted(p.name for p in BASE.iterdir() if p.is_dir() and p.name.startswith("vehicle_"))
    reps = {f: rel_of(representative(f)) for f in folders}
    print("Representatives:", json.dumps(reps, indent=2))

    out = load_existing() or {"explicit_pairs": {}, "within_vehicle": {}, "cross_vehicle": {}}

    def save():
        with open(OUTPUT_FILE, "w") as fp:
            json.dump(out, fp, indent=2)

    def need(key: str, section: str) -> bool:
        text = out[section][key]
        return "error" in text or "decision" not in text

    # 1) Explicit user-requested pairs
    explicit = [
        ("vehicle_6/t0038_f00726.jpg", "vehicle_5/t0008_f00187.jpg"),
    ]
    for a, b in explicit:
        key = f"{a} <-> {b}"
        if key not in out["explicit_pairs"] or need(key, "explicit_pairs"):
            print(f"\nEXPLICIT: {a} vs {b}", flush=True)
            out["explicit_pairs"][key] = compare(a, b)
            save()
        else:
            print(f"\nSKIP EXPLICIT: {key}", flush=True)

    # also verify each image against itself (sanity: must be SAME)
    for a, b in explicit:
        key = f"{a} <-> {a}"
        if key not in out["explicit_pairs"] or need(key, "explicit_pairs"):
            print(f"SELF-CHECK: {a} vs itself", flush=True)
            out["explicit_pairs"][key] = compare(a, a)
            save()
        else:
            print(f"SKIP SELF-CHECK: {key}", flush=True)

    # 2) Within-vehicle matching (rep vs every other crop of the same vehicle)
    for f in folders:
        others = [rel_of(p) for p in images_in(f) if rel_of(p) != reps[f]]
        if not others:
            continue
        out["within_vehicle"].setdefault(f, {})
        for o in others:
            if o in out["within_vehicle"][f] and "decision" in out["within_vehicle"][f][o] and "error" not in out["within_vehicle"][f][o]:
                print(f"SKIP WITHIN {f}: rep({reps[f]}) vs {o}", flush=True)
                continue
            print(f"WITHIN {f}: rep({reps[f]}) vs {o}", flush=True)
            out["within_vehicle"][f][o] = compare(reps[f], o)
            save()

    # 3) Cross-vehicle discrimination
    pairs = set()
    for f in folders:
        for g in folders:
            if f == g:
                continue
            pairs.add(tuple(sorted((f, g))))

    # requested highlights always included; also run the full matrix for completeness
    for f in ["vehicle_1", "vehicle_8", "vehicle_7"]:
        for g in folders:
            if f == g:
                continue
            pair = tuple(sorted((f, g)))
            a, b = reps[pair[0]], reps[pair[1]]
            key = f"{a} <-> {b}"
            if key in out["cross_vehicle"] and "error" not in out["cross_vehicle"][key] and "decision" in out["cross_vehicle"][key]:
                print(f"SKIP CROSS {key}", flush=True)
                continue
            print(f"CROSS {pair[0]}({a}) vs {pair[1]}({b})", flush=True)
            out["cross_vehicle"][key] = compare(a, b)
            save()

    save()

    print("\n=== SUMMARY ===")
    for k, v in out["explicit_pairs"].items():
        print(f"EXPLICIT  {k}: decision={v.get('decision')} conf={v.get('confidence')}")
    for f, d in out["within_vehicle"].items():
        same = sum(1 for v in d.values() if v.get("decision") == "same")
        diff = sum(1 for v in d.values() if v.get("decision") == "different")
        unc  = sum(1 for v in d.values() if v.get("decision") == "uncertain")
        errs = sum(1 for v in d.values() if "error" in v)
        print(f"WITHIN {f}: same={same} different={diff} uncertain={unc} errors={errs} (of {len(d)})")
    for k, v in out["cross_vehicle"].items():
        print(f"CROSS     {k}: decision={v.get('decision')} conf={v.get('confidence')}")

    print(f"\nAll comparisons saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()