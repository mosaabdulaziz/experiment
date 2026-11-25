"""What a module asked to draw on recorded clips (`iris:overlay:<camera>`, EngineServices.overlay), placed on a clip.

Every overlay entry carries the frame time (`pts`, the pipeline's clock) of the analysed frame it belongs to, and so
does every recorded segment (the recorder indexes its segments on the same clock), so a clip knows the frame time of its
first frame (`start_pts`, from the segment it was cut from) and an entry shows `pts - start_pts` seconds in -- exactly.
(The wall clock is no use here: segments are labelled with the time they were indexed minus their duration, which runs
seconds off, by a different amount per camera.) Each entry shows until the next one or the hold, whichever is first
(the live preview holds the last analysed frame's lines the same way; a clip bridges a short detector miss).

The proximity module writes an entry for every analysed frame with a person and a vehicle in a proximity zone (its
README): each pair's line with its style and ids, each box with its mask's outline and the vehicle's state. A clip and
a snapshot draw their incident's pair from it (`select`), in the live preview's colours.

The other features write boxes (perception/lib/overlay_boxes.py): style "alert" for an object an alert can be about,
by its tracker id -- PPE's workers, the pose module's people (the fall service names its track), people in a zone --
drawn when it is the alert's subject (`incident_pair`: person_ids, else track_id or object_id), or all of them when
the alert names none (overcrowding); "fire" / "smoke" always. The alert's object is drawn in the alert colour."""
from __future__ import annotations

import json
import math
import os

import cv2
import numpy as np

OVERLAY_PREFIX = "iris:overlay:"
HOLD_S = 0.5
CLIP_HOLD_S = 1.0   # a clip's entry shows this long without a next one: a frame or two the detector missed
REACH_S = 5.0       # entries further than the clip's pad plus this from the alert belong to another timeline
# the proximity module's live-preview colours (BGR; not imported: the module package loads the pipeline)
PERSON_BGR = (80, 220, 80)
VEHICLE_BGR = {"moving": (230, 60, 230), "parked": (160, 160, 160), "unknown": (255, 150, 30)}
LINE_BGR = {"latched": (0, 0, 255), "close": (0, 165, 255), "far": (200, 255, 170)}
LINE_SHADOW = (25, 25, 25)      # the dark outline under every line and gap
DEPTH_BGR = {"same": (80, 200, 80), "apart": (60, 60, 230), "wait": (170, 170, 170)}   # the module's DEPTH_* colours
ALERT_BGR = (98, 94, 255)       # the alert's own object: IRIS's alert red (#ff5e62)
HAZARD_BGR = {"fire": (40, 110, 255), "smoke": (190, 190, 190)}   # boxes drawn whatever the alert names
MASK_ALPHA = 0.18
# the proximity preview setting the UI writes (StreamManager; the module's PREVIEW_KEY): clips take `clip_style` from it
PREVIEW_KEY = "iris:ctrl:proximity_preview"
CLIP_STYLES = ("masks", "boxes")        # the first is the default


def _text(v) -> str:
    return v.decode() if isinstance(v, bytes) else v


def _pair(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(c, (int, float)) for c in v)


def _well_formed(shape) -> bool:
    """A shape `draw` can use: a line with two points, or a box with four numbers (and a mask outline of three
    points or more, when it has one)."""
    if not isinstance(shape, dict):
        return False
    if shape.get("type") == "line":
        return _pair(shape.get("p1")) and _pair(shape.get("p2"))
    rect, mask = shape.get("rect"), shape.get("mask")
    return (shape.get("type") == "box" and isinstance(rect, (list, tuple)) and len(rect) == 4
            and all(isinstance(c, (int, float)) for c in rect)
            and (mask is None or (isinstance(mask, list) and len(mask) >= 3 and all(_pair(pt) for pt in mask))))


def _ids(raw) -> set:
    out = set()
    for part in str(raw or "").split(","):
        try:
            out.add(int(part))
        except ValueError:
            continue
    return out


def incident_pair(details) -> "tuple | None":
    """(person ids, vehicle ids) an alert is about, from its details: a proximity alert's `person_ids` / `vehicle_ids`
    ("6,8"), else the one object of a PPE alert (`track_id`) or a zone intrusion (`object_id`); None for an alert that
    names none."""
    if not isinstance(details, dict):
        return None
    if "person_ids" in details or "vehicle_ids" in details:
        return _ids(details.get("person_ids")), _ids(details.get("vehicle_ids"))
    for key in ("track_id", "object_id"):
        if _ids(details.get(key)):
            return _ids(details.get(key)), set()
    return None


def merge_pairs(a, b) -> "tuple | None":
    """The pairs of two incidents sharing a clip; an alert without one adds nothing."""
    if a is None or b is None:
        return a if b is None else b
    return a[0] | b[0], a[1] | b[1]


def clip_style(redis_client) -> str:
    """How clips and snapshots draw their objects, as chosen in the UI: "masks" (the default) or "boxes". Never
    raises: an unreadable setting is the default."""
    try:
        chosen = json.loads(redis_client.get(PREVIEW_KEY) or "{}")
        style = chosen.get("clip_style") if isinstance(chosen, dict) else None
    except Exception:  # noqa: BLE001 -- the setting is cosmetic: a clip is never lost to it
        style = None
    return style if style in CLIP_STYLES else CLIP_STYLES[0]


def select(entry: dict, pair) -> "dict | None":
    """The entry with only what a clip or snapshot draws, or None when that is nothing: the incident's pair -- its
    lines, close or not, and the boxes of its persons and vehicles -- or, without a pair, the close lines and their
    objects; and of the other features' boxes, the alert's own object (all of them without a subject), fire and
    smoke."""
    if pair is None:
        lines = [s for s in entry["shapes"] if s.get("type") == "line" and s.get("style") in ("latched", "close")]
        persons, vehicles = {s.get("person") for s in lines}, {s.get("vehicle") for s in lines}
    else:
        persons, vehicles = pair
        lines = [s for s in entry["shapes"]
                 if s.get("type") == "line" and s.get("person") in persons and s.get("vehicle") in vehicles]
    subjects = None if pair is None else pair[0]
    boxes = [s for s in entry["shapes"] if s.get("type") == "box" and (
        s.get("style") in HAZARD_BGR                                                   # fire, smoke: always
        or (s.get("style") == "alert" and (subjects is None or s.get("id") in subjects))  # the alert's own object
        or s.get("id") in {"person": persons, "vehicle": vehicles}.get(s.get("style"), ()))]
    return {**entry, "shapes": lines + boxes} if lines or boxes else None


def read_entries(redis_client, sid: str, around_pts: "float | None", reach: "float | None",
                 wall_lo: "float | None" = None, wall_hi: "float | None" = None) -> list:
    """The camera's overlay entries within `reach` seconds of `around_pts` (frame time; None: any), oldest first;
    with a wall window, only those written inside it -- an entry's id is its write time in ms, kept as its `wall` (s),
    which places it on a smart-record clip. Malformed entries are skipped."""
    lo = "-" if wall_lo is None else str(int(wall_lo * 1000))
    hi = "+" if wall_hi is None else str(int(wall_hi * 1000))
    out = []
    for _id, fields in redis_client.xrange(OVERLAY_PREFIX + sid, lo, hi):
        f = {_text(k): _text(v) for k, v in fields.items()}
        try:
            pts = float(f["pts"])
            if around_pts is not None and abs(pts - around_pts) > reach:
                continue
            shapes = [sh for sh in json.loads(f["shapes"]) if _well_formed(sh)]   # a bad shape costs only itself
            # the frame's arrival when the entry carries it (its write time trails it by the inference delay)
            wall = float(f["at"]) if f.get("at") else int(_text(_id).split("-")[0]) / 1000.0
            out.append({"pts": pts, "wall": wall, "w": int(f["w"]), "h": int(f["h"]), "shapes": shapes})
        except (KeyError, ValueError, TypeError):
            continue
    return sorted(out, key=lambda e: e["pts"])


def timeline(entries: list, clip_start_pts: float, clip_len: float, hold: float = HOLD_S) -> list:
    """[(start s into the clip, duration s, entry or None), ...] covering the whole clip, the clip's first frame being
    at frame time `clip_start_pts`; None = nothing drawn."""
    placed = sorted(((e["pts"] - clip_start_pts, e) for e in entries), key=lambda p: p[0])
    placed = [(rel, e) for rel, e in placed if 0.0 <= rel < clip_len]
    out, cursor = [], 0.0
    for i, (rel, e) in enumerate(placed):
        if rel > cursor:
            out.append((cursor, rel - cursor, None))
        nxt = placed[i + 1][0] if i + 1 < len(placed) else clip_len
        shown = min(nxt - rel, hold)
        if shown > 0:
            out.append((rel, shown, e))
            cursor = rel + shown
    if cursor < clip_len:
        out.append((cursor, clip_len - cursor, None))
    return out


def _fill(img: np.ndarray, pts: np.ndarray, color: tuple) -> None:
    """A see-through fill: on a transparent image (a clip's overlay) through its alpha, on a frame (a snapshot) by
    blending."""
    if img.ndim == 3 and img.shape[2] == 4:
        cv2.fillPoly(img, [pts], color + (round(MASK_ALPHA * 255),))
        return
    h, w = img.shape[:2]
    x0, y0 = np.maximum(pts.min(axis=0), 0)
    x1, y1 = np.minimum(pts.max(axis=0) + 1, (w, h))
    if x1 <= x0 or y1 <= y0:
        return
    region = img[y0:y1, x0:x1]
    filled = region.copy()
    cv2.fillPoly(filled, [pts - (x0, y0)], color)
    cv2.addWeighted(filled, MASK_ALPHA, region, 1 - MASK_ALPHA, 0, dst=region)


def _dashed(img, a, b, color, width, on, off, pad) -> None:
    """A dashed segment a -> b; each gap widened by `pad` (the coloured line's width) so round caps leave it open and
    an outline's dashes stay in step with its line's."""
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    if length < 1:
        return
    ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
    s = 0.0
    while s < length:
        e = min(s + on, length)
        cv2.line(img, (int(a[0] + ux * s), int(a[1] + uy * s)), (int(a[0] + ux * e), int(a[1] + uy * e)), color, width,
                 cv2.LINE_AA)
        s += on + off + pad


def _depth_part(depth):
    """(text, colour key) of a line's depth reading, or None."""
    if not isinstance(depth, dict):
        return None
    state, gap = depth.get("state"), depth.get("gap")
    if state in ("same", "apart") and isinstance(gap, (int, float)):
        return f"d {gap:.2f}", state
    if state in ("pending", "timeout"):
        return ("d ..." if state == "pending" else "d -"), "wait"
    return None


def draw(img: np.ndarray, entry: dict, style: str = "masks") -> np.ndarray:
    """Draw an entry (`select`ed) on a BGR or BGRA image, scaled from the entry's coordinate space: each object as its
    mask (see-through, outlined; its box when it has no mask) or, in the "boxes" style, as its box -- labelled, persons
    green, vehicles by state -- then the lines with their gaps, coloured by how close."""
    h, w = img.shape[:2]
    sx, sy = w / max(entry["w"], 1), h / max(entry["h"], 1)
    thick = max(round(w / 400), 2)
    scale = max(w / 1280 * 0.6, 0.4)
    alpha = (255,) if img.ndim == 3 and img.shape[2] == 4 else ()

    def label(text, x, top, color) -> None:
        text_h = int(22 * scale) + thick
        y = top - 2 * thick if top - 2 * thick >= text_h else top + text_h          # above, else just inside
        cv2.putText(img, str(text), (x + thick, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, max(thick // 2, 1),
                    cv2.LINE_AA)

    shapes = [s for s in entry["shapes"] if _well_formed(s)]
    for s in shapes:
        if s.get("type") != "box":
            continue
        color = _box_color(s)
        if style == "masks" and s.get("mask"):
            pts = np.round(np.array(s["mask"], np.float32) * (sx, sy)).astype(np.int32)
            _fill(img, pts, color)
            cv2.polylines(img, [pts], True, color + alpha, max(thick // 2, 1), cv2.LINE_AA)
            if s.get("label"):
                label(s["label"], int(pts[:, 0].min()), int(pts[:, 1].min()), color + alpha)
            continue
        left, top, bw, bh = s["rect"]
        p1, p2 = (int(left * sx), int(top * sy)), (int((left + bw) * sx), int((top + bh) * sy))
        cv2.rectangle(img, p1, p2, color + alpha, thick)
        if s.get("label"):
            label(s["label"], p1[0], p1[1], color + alpha)
    ring = max(3 * thick, 6)
    on, off = 2 * thick, thick + 1
    lines = []
    for s in shapes:
        if s.get("type") != "line":
            continue
        a = (int(s["p1"][0] * sx), int(s["p1"][1] * sy))
        b = (int(s["p2"][0] * sx), int(s["p2"][1] * sy))
        seen = None
        if s.get("hidden") and _pair(s.get("seen")):
            seen = (int(s["seen"][0] * sx), int(s["seen"][1] * sy))
        start, length = a, math.hypot(b[0] - a[0], b[1] - a[1])
        if seen and length > ring:                         # the line leaves from the ring's edge
            start = (int(a[0] + (b[0] - a[0]) * ring / length), int(a[1] + (b[1] - a[1]) * ring / length))
        width = thick if s.get("style") == "far" else thick + max(thick // 2, 2)
        apart = isinstance(s.get("depth"), dict) and s["depth"].get("state") == "apart"
        lines.append((s, a, start, b, seen, width, apart))
    edge = max(thick // 3, 1)
    leg = max(thick // 2, 1)
    for _s, a, start, b, seen, width, apart in lines:      # all outlines first: lines that cross stay whole
        if apart:
            _dashed(img, start, b, LINE_SHADOW + alpha, width + 2 * edge, on, off, width)
        else:
            cv2.line(img, start, b, LINE_SHADOW + alpha, width + 2 * edge, cv2.LINE_AA)
        if seen:
            _dashed(img, seen, (a[0], a[1] - ring), LINE_SHADOW + alpha, leg + 2 * edge, on, off, leg)
            cv2.circle(img, a, ring, LINE_SHADOW + alpha, leg + 1 + 2 * edge, cv2.LINE_AA)
        else:
            cv2.circle(img, a, ring // 2 + edge, LINE_SHADOW + alpha, -1, cv2.LINE_AA)
    for s, a, start, b, seen, width, apart in lines:
        color = LINE_BGR.get(s.get("style"), LINE_BGR["latched"]) + alpha
        if apart:
            _dashed(img, start, b, color, width, on, off, width)
        else:
            cv2.line(img, start, b, color, width, cv2.LINE_AA)
        if seen:
            _dashed(img, seen, (a[0], a[1] - ring), color, leg, on, off, leg)
            cv2.circle(img, a, ring, color, leg + 1, cv2.LINE_AA)
        else:
            cv2.circle(img, a, ring // 2, color, -1, cv2.LINE_AA)
        at = ((a[0] + b[0]) // 2 + 2 * thick, (a[1] + b[1]) // 2 - thick)
        text_w = max(thick // 2, 1)
        label = str(s.get("label", ""))
        texts = [(label, at[0], color)]
        part = _depth_part(s.get("depth"))
        if part:
            shift = cv2.getTextSize(label + " ", cv2.FONT_HERSHEY_SIMPLEX, scale, text_w)[0][0]
            texts.append((part[0], at[0] + shift, DEPTH_BGR[part[1]] + alpha))
        for text, x, col in texts:
            cv2.putText(img, text, (x, at[1]), cv2.FONT_HERSHEY_SIMPLEX, scale, LINE_SHADOW + alpha,
                        text_w + 2 * edge, cv2.LINE_AA)
            cv2.putText(img, text, (x, at[1]), cv2.FONT_HERSHEY_SIMPLEX, scale, col, text_w, cv2.LINE_AA)
    return img


def _box_color(shape: dict) -> tuple:
    if shape.get("style") == "person":
        return PERSON_BGR
    if shape.get("style") == "alert":
        return ALERT_BGR
    if shape.get("style") in HAZARD_BGR:
        return HAZARD_BGR[shape["style"]]
    return VEHICLE_BGR.get(shape.get("state"), VEHICLE_BGR["unknown"])


def write_sequence(segments: list, width: int, height: int, tmp_dir: str, style: str = "masks") -> "str | None":
    """The timeline as transparent PNGs (objects drawn in `style`) and an ffconcat list for
    clip_builder.apply_overlays; None when nothing is drawn anywhere."""
    if not any(e is not None for _s, _d, e in segments):
        return None
    blank = os.path.join(tmp_dir, "lines_blank.png")
    cv2.imwrite(blank, np.zeros((height, width, 4), np.uint8))
    rows, last = ["ffconcat version 1.0"], blank
    for i, (_start, duration, e) in enumerate(segments):
        path = blank
        if e is not None:
            path = os.path.join(tmp_dir, f"lines_{i:04d}.png")
            cv2.imwrite(path, draw(np.zeros((height, width, 4), np.uint8), e, style))
        rows += [f"file '{path}'", f"duration {duration:.3f}"]
        last = path
    rows.append(f"file '{last}'")       # the concat demuxer honours the last duration only when a file follows it
    out = os.path.join(tmp_dir, "lines.ffconcat")
    with open(out, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    return out


def pick_clip(clips: list, at_pts: float):
    """(clip id, seconds into it, exact) for the frame at `at_pts` (frame time): the clip whose range holds that
    moment, else the nearest frame of the nearest clip (0.1 s inside its end: a seek to the very end may find no
    frame); None without a usable clip. `clips` = [(clip id, meta)], meta as the alert clip service stores it
    ({"range", "start_pts"}, or for an older clip {"anchor", "pts", "range"})."""
    best = None
    for clip_id, meta in clips:
        try:
            lo, hi = (float(v) for v in meta["range"])
            if meta.get("start_pts") is not None:           # the clip's own frame times (exact)
                at = lo + (at_pts - float(meta["start_pts"]))
            else:                                           # an older clip: placed from its alert
                at = float(meta["anchor"]) + (at_pts - float(meta["pts"]))
        except (KeyError, TypeError, ValueError):
            continue
        if lo <= at <= hi:
            return clip_id, at - lo, True
        distance = lo - at if at < lo else at - hi
        if best is None or distance < best[0]:
            best = (distance, clip_id, min(max(at, lo), max(hi - 0.1, lo)) - lo)
    return None if best is None else (best[1], best[2], False)


def nearest(entries: list, at_pts: float, hold: float = HOLD_S) -> "dict | None":
    """The entry closest to `at_pts`, if one lies within `hold` of it."""
    best = min(entries, key=lambda e: abs(e["pts"] - at_pts), default=None)
    return best if best is not None and abs(best["pts"] - at_pts) <= hold else None
