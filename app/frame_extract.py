import cv2
import numpy as np

_RESTRICTED_COLOR_BGR = (0, 0, 255)     # red
_VALID_COLOR_BGR = (168, 242, 107)      # green — BGR encoding of #6bf2a8 (frontend's --green)
# The zone editor's colours (zoneDraft.ts zoneColor: --accent, --green, --yellow) and its fill (20%), for clips
_EDITOR_COLOR_BGR = {"restricted": (108, 153, 249), "valid": (168, 242, 107), "off": (102, 209, 255)}
_EDITOR_FILL_ALPHA = 51


def annotate_zone(frame: bytes, points_norm: list[tuple[float, float]],
                  zone_type: str = "restricted") -> bytes:
    img = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    if img is None or len(points_norm) < 3:
        return frame
    h, w = img.shape[:2]
    pts = np.array([[round(x * w), round(y * h)] for x, y in points_norm], dtype=np.int32)
    thickness = max(round(w / 400), 2)
    color = _VALID_COLOR_BGR if zone_type == "valid" else _RESTRICTED_COLOR_BGR
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else frame


def zone_overlay_png(points_norm: list[tuple[float, float]], width: int, height: int,
                     dest_path: str, zone_type: str = "restricted") -> "str | None":
    """One zone: zones_overlay_png's."""
    return zones_overlay_png([(points_norm, zone_type)], width, height, dest_path)


def zones_overlay_png(zones: list, width: int, height: int, dest_path: str) -> "str | None":
    """Write a transparent BGRA PNG of width x height carrying the zone as the zone editor draws it: its outline over
    a 20% fill, in the editor's colour for its purpose (ZoneEditorOverlay.tsx, zoneDraft.ts zoneColor) -- restricted
    the accent, valid green, crowding only (`zone_type` "off") yellow.

    Burned onto an alert preview clip by clip_builder.apply_overlay(s). (A description frame's zone, annotate_zone,
    keeps its own red / green outline: it is drawn for the model, not the operator.) `zones` is [(points, zone_type),
    ...]: several alerts sharing one clip show every zone they are about. Returns dest_path, or None when every polygon
    is degenerate, the canvas is empty, or the write failed -- callers treat None as "no overlay" and keep going
    rather than losing the clip.
    """
    drawable = [(points, kind) for points, kind in zones if len(points) >= 3]
    if not drawable or width <= 0 or height <= 0:
        return None
    img = np.zeros((height, width, 4), dtype=np.uint8)      # fully transparent
    thickness = max(round(width / 400), 2)
    for points_norm, zone_type in drawable:
        pts = np.array([[round(x * width), round(y * height)] for x, y in points_norm], dtype=np.int32)
        b, g, r = _EDITOR_COLOR_BGR.get(zone_type, _EDITOR_COLOR_BGR["restricted"])
        fill = img.copy()
        cv2.fillPoly(fill, [pts], color=(b, g, r, _EDITOR_FILL_ALPHA))
        img = np.maximum(img, fill)                         # overlapping fills do not stack up
        cv2.polylines(img, [pts], isClosed=True, color=(b, g, r, 255), thickness=thickness)
    try:
        return dest_path if cv2.imwrite(dest_path, img) else None
    except Exception:
        return None
