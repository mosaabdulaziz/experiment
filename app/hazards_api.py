"""The Hazard Detection tab's API: POST /api/hazards/analyze, GET /api/hazards/status.

An image (an upload, or the current frame of one of the camera streams) is sent to the Qwen VLM (app/hazards.py); the answer is the
hazards, the model's one-sentence scene and the image with the hazards drawn on it. The call takes tens of seconds, so the endpoint is a
plain (threaded) function: it blocks only its own worker thread. The inputs and results are kept in data/hazards/.
"""
import base64
import io
import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from . import hazards

logger = logging.getLogger(__name__)
router = APIRouter(tags=["hazards"])

MAX_BYTES = 15 * 1024 * 1024                  # an uploaded image
MAX_PIXELS = 60_000_000
SAVE_DIR = Path(os.environ.get("HAZARDS_DIR", os.path.join("data", "hazards")))
# each analysis is one paid call: only a few at a time
_slots = threading.BoundedSemaphore(int(os.environ.get("HAZARDS_MAX_CONCURRENT", "2")))


def _error(message: str, status: int):
    return JSONResponse({"error": message}, status_code=status)


@router.get("/api/hazards/status")
def hazards_status():
    cfg = hazards.settings()
    return {"configured": cfg is not None, "model": cfg.model if cfg else None}


def _frame_of(request: Request, stream_id: str):
    """The current frame of a camera stream as JPEG bytes, or (None, error response)."""
    sm = getattr(request.app.state, "stream_manager", None)
    src = sm.get_source(stream_id) if sm is not None else None
    if src is None:
        return None, _error("unknown stream", 404)
    frame = src.grab_display_frame()
    if not frame:
        return None, JSONResponse({"error": "no frame yet"}, status_code=503, headers={"Retry-After": "3"})
    return frame, None


@router.post("/api/hazards/analyze")
def hazards_analyze(request: Request, file: UploadFile | None = File(None), stream_id: str | None = Form(None)):
    cfg = hazards.settings()
    if cfg is None:
        return _error("The VLM is not configured: QWEN_API_KEY / QWEN_BASE_URL / QWEN_MODEL are missing", 503)
    if file is not None and stream_id:
        return _error("Send either an image or a stream_id, not both", 400)
    if file is not None:
        data = file.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            return _error(f"The image is larger than {MAX_BYTES // (1024 * 1024)} MB", 413)
        source, name = "upload", Path(file.filename or "upload").stem
    elif stream_id:
        data, err = _frame_of(request, stream_id)
        if err is not None:
            return err
        source, name = "camera", stream_id
    else:
        return _error("Send an image (file) or a camera (stream_id)", 400)
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        return _error("That file is not an image the server can read (png, jpg, webp)", 400)
    if img.width * img.height > MAX_PIXELS:
        return _error("The image is too large to analyse", 413)

    if not _slots.acquire(blocking=False):
        return _error("Too many analyses are running; try again in a moment", 429)
    t0 = time.time()
    try:
        result, annotated = hazards.analyze(img, cfg)
    except hazards.HazardError as exc:
        logger.warning("[hazards] %s: %s", name, exc)
        return _error(str(exc), exc.status)
    finally:
        _slots.release()
    elapsed = round(time.time() - t0, 1)

    now = datetime.now()
    stem = f"{now:%Y%m%d_%H%M%S}_{_safe(name)}"
    hz = []
    for h in result.get("hazards", []) if isinstance(result.get("hazards"), list) else []:
        if isinstance(h, dict):
            hz.append({"label": str(h.get("label", "hazard")), "severity": str(h.get("severity", "")).lower(),
                       "description": str(h.get("description", "")), "bbox": h.get("bbox")})
    out = {
        "id": stem, "scene": str(result.get("scene", "")), "hazards": hz,
        "width": img.width, "height": img.height,
        "model": cfg.model, "source": source, "name": name, "elapsed_s": elapsed, "created": now.isoformat(timespec="seconds"),
    }
    try:                                       # the Arabic wording (a text-only call): the English result stands without it
        out["ar"] = hazards.translate_ar(out, cfg)
    except hazards.HazardError as exc:
        logger.warning("[hazards] %s: no Arabic translation: %s", name, exc)
    saved = {}
    try:                                       # the results are kept like the prototype's: <name>_hazards.png / .json (the history reads them)
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        annotated.save(SAVE_DIR / f"{stem}_hazards.png")
        (SAVE_DIR / f"{stem}_hazards.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        saved = {"png": f"{stem}_hazards.png", "json": f"{stem}_hazards.json"}
    except OSError as exc:
        logger.warning("[hazards] could not save the result: %s", exc)
    return {**out, "image": _jpeg_data_url(annotated), "saved": saved}


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name).strip("_") or "image"


def _jpeg_data_url(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


_ID = re.compile(r"^\d{8}_\d{6}_[\w.-]+$")


def _history_entry(jpath: Path) -> dict | None:
    """One saved analysis (its json). Results saved before the history existed hold only scene+hazards: the rest comes from the name."""
    try:
        d = json.loads(jpath.read_text())
    except (OSError, ValueError):
        return None
    stem = jpath.name[: -len("_hazards.json")]
    m = re.match(r"^(\d{8})_(\d{6})_(.+)$", stem)
    if not m or not isinstance(d, dict):
        return None
    name = d.get("name") or m.group(3)
    created = d.get("created") or datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").isoformat(timespec="seconds")
    hz = d.get("hazards") if isinstance(d.get("hazards"), list) else []
    return {
        "id": stem, "name": name, "created": created, "scene": str(d.get("scene", "")),
        "source": d.get("source") or ("camera" if re.match(r"^cam-\d+$", name) else "upload"),
        "ar": d.get("ar") if isinstance(d.get("ar"), dict) else None,
        "model": d.get("model"), "elapsed_s": d.get("elapsed_s"), "width": d.get("width"), "height": d.get("height"),
        "hazards": [{"label": str(h.get("label", "hazard")), "severity": str(h.get("severity", "")).lower(),
                     "description": str(h.get("description", "")), "bbox": h.get("bbox")} for h in hz if isinstance(h, dict)],
    }


@router.get("/api/hazards/history")
def hazards_history(stream_id: str | None = None, limit: int = 30):
    """The saved analyses, newest first; with stream_id only that camera's. No images (see /history/{id})."""
    if not SAVE_DIR.is_dir():
        return {"items": []}
    items = []
    for jp in sorted(SAVE_DIR.glob("*_hazards.json"), reverse=True):
        e = _history_entry(jp)
        if e is None or (stream_id and e["name"] != stream_id):
            continue
        items.append(e)
        if len(items) >= max(1, min(limit, 200)):
            break
    return {"items": items}


@router.get("/api/hazards/history/{item_id}")
def hazards_history_item(item_id: str):
    """One saved analysis with its annotated image."""
    jp, pp = SAVE_DIR / f"{item_id}_hazards.json", SAVE_DIR / f"{item_id}_hazards.png"
    entry = _history_entry(jp) if _ID.match(item_id) and jp.is_file() and pp.is_file() else None
    if entry is None:
        return _error("unknown analysis", 404)
    if entry["ar"] is None and (cfg := hazards.settings()) is not None:       # saved before the Arabic wording existed: translate it once
        try:
            entry["ar"] = hazards.translate_ar(entry, cfg)
            data = json.loads(jp.read_text())
            data["ar"] = entry["ar"]
            jp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except (hazards.HazardError, OSError, ValueError) as exc:
            logger.warning("[hazards] %s: no Arabic translation: %s", item_id, exc)
    with Image.open(pp) as im:
        return {**entry, "image": _jpeg_data_url(im)}
