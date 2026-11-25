"""Hazard annotation with a Qwen VLM (DashScope's OpenAI-compatible /chat/completions).

One image goes in; the model returns the hazards it sees (label, severity high/medium/low, one-sentence description and a bbox on
a 0-1000 normalised grid) and `annotate` draws them as severity-coloured boxes with numbered labels. This is the logic of the
standalone prototype scripts/vlm_hazards.py (which now imports it) and what the Hazard Detection tab's API (app/hazards_api.py) runs.

Credentials: QWEN_API_KEY / QWEN_BASE_URL / QWEN_MODEL in the environment.
"""
import base64
import io
import json
import os
import re
from dataclasses import dataclass

import httpx
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SEVERITY_COLOR = {"high": (230, 30, 30), "medium": (255, 150, 0), "low": (240, 200, 0)}

PROMPT = """You are an industrial HSE (health, safety, environment) inspector reviewing a frame from a fixed CCTV camera in a chemical drum storage / loading yard.
Identify every plausible safety hazard visible: e.g. unsafe stacking of drums/IBCs, drums stored without pallets or bunding, spills or stains, loose hoses/cables on walkways, forklift/vehicle-pedestrian proximity, workers without visible PPE, blocked aisles or fire equipment, combustible material (rags, wood pallets, cardboard) next to chemicals, damaged or leaking containers, unlabeled containers, fire/smoke, ignition sources.
Only report what you can actually see; do not invent hazards. Prefer tight boxes.

Reply with ONLY a JSON object, no markdown:
{"scene": "<one sentence>", "hazards": [{"label": "<short name>", "severity": "high|medium|low", "description": "<one sentence why it is a hazard>", "bbox": [x1, y1, x2, y2]}]}
bbox coordinates are integers normalised to a 0-1000 grid over the image width/height (x1,y1 top-left; x2,y2 bottom-right)."""


class HazardError(Exception):
    """Something the caller can report: `status` is the HTTP status an API should answer with."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Settings:
    key: str
    base_url: str
    model: str


def settings() -> "Settings | None":
    """The VLM settings from the environment, or None when one of the three is missing."""
    key, base, model = (os.environ.get(k) for k in ("QWEN_API_KEY", "QWEN_BASE_URL", "QWEN_MODEL"))
    return Settings(key, base, model) if key and base and model else None


def encode(img, max_side=1600):
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def parse_reply(text):
    """The JSON object in the model's reply (it may wrap it in prose or a code fence)."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise HazardError(f"No JSON in reply: {(text or '')[:500]}")
    try:
        res = json.loads(m.group(0))
    except ValueError as exc:
        raise HazardError(f"The reply is not valid JSON ({exc}): {m.group(0)[:300]}") from exc
    if not isinstance(res, dict):
        raise HazardError(f"Unexpected reply: {m.group(0)[:300]}")
    return res


def ask(img, model, base_url, key, timeout=300):
    body = {
        "model": model,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encode(img)}},
            {"type": "text", "text": PROMPT},
        ]}],
    }
    try:
        r = httpx.post(base_url.rstrip("/") + "/chat/completions", json=body, timeout=timeout,
                       headers={"Authorization": f"Bearer {key}"})
    except httpx.HTTPError as exc:
        raise HazardError(f"The VLM could not be reached: {exc}") from exc
    if r.status_code >= 400:
        raise HazardError(f"API error {r.status_code}: {r.text[:500]}")
    try:
        text = r.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise HazardError(f"Unexpected API reply: {r.text[:300]}") from exc
    return parse_reply(text)


TRANSLATE_PROMPT = """Translate this workplace-safety report into Modern Standard Arabic, as a safety officer would write it.
Translate only the text values ("scene", and each hazard's "label" and "description"); keep the same keys and the same number and order of hazards.
Reply with ONLY the JSON object, no markdown.

"""


def translate_ar(result, cfg: Settings, timeout=120):
    """The scene and each hazard's label/description in Arabic: {"scene": str, "hazards": [{"label", "description"}]} (text only, no image)."""
    hz = [h for h in result.get("hazards", []) if isinstance(h, dict)]
    src = {"scene": str(result.get("scene", "")),
           "hazards": [{"label": str(h.get("label", "")), "description": str(h.get("description", ""))} for h in hz]}
    body = {"model": cfg.model, "temperature": 0.1,
            "messages": [{"role": "user", "content": TRANSLATE_PROMPT + json.dumps(src, ensure_ascii=False)}]}
    try:
        r = httpx.post(cfg.base_url.rstrip("/") + "/chat/completions", json=body, timeout=timeout,
                       headers={"Authorization": f"Bearer {cfg.key}"})
    except httpx.HTTPError as exc:
        raise HazardError(f"The VLM could not be reached: {exc}") from exc
    if r.status_code >= 400:
        raise HazardError(f"API error {r.status_code}: {r.text[:500]}")
    try:
        out = parse_reply(r.json()["choices"][0]["message"]["content"])
        ar = {"scene": str(out["scene"]),
              "hazards": [{"label": str(a["label"]), "description": str(a["description"])} for a in out["hazards"]]}
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise HazardError(f"Unexpected translation reply: {r.text[:300]}") from exc
    if len(ar["hazards"]) != len(hz):
        raise HazardError("The translation does not match the hazards")
    return ar


def _font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:                                   # no DejaVu in this image: Pillow's built-in font
        return ImageFont.load_default(size=size)


def annotate(img, result):
    img = img.convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img)
    font = _font(max(14, h // 55))
    lw = max(2, h // 300)
    for i, hz in enumerate(result.get("hazards", []), 1):
        try:
            x1, y1, x2, y2 = [float(v) for v in hz["bbox"]]
        except (KeyError, ValueError, TypeError):
            continue
        box = [x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h]
        col = SEVERITY_COLOR.get(str(hz.get("severity", "")).lower(), (255, 0, 255))
        d.rectangle(box, outline=col, width=lw)
        tag = f"{i}. {hz.get('label', 'hazard')}"
        tb = d.textbbox((0, 0), tag, font=font)
        ty = max(0, box[1] - (tb[3] - tb[1]) - 6)
        d.rectangle([box[0], ty, box[0] + tb[2] - tb[0] + 8, ty + tb[3] - tb[1] + 6], fill=col)
        d.text((box[0] + 4, ty + 1), tag, fill=(0, 0, 0), font=font)
    return img


def analyze(img, cfg: Settings):
    """Ask the model about `img`; returns (result dict, annotated PIL image)."""
    res = ask(img, cfg.model, cfg.base_url, cfg.key)
    return res, annotate(img, res)
