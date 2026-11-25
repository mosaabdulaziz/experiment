import os

import cv2
import numpy as np

from app.frame_extract import annotate_zone, zone_overlay_png


def _solid_jpeg(width: int, height: int, color=(0, 0, 0)) -> bytes:
    img = np.full((height, width, 3), color, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_annotate_zone_draws_red_outline_at_scaled_position():
    frame = _solid_jpeg(400, 200, color=(0, 0, 0))  # solid black frame, width=400 height=200
    # A square covering the left half of the frame, in normalized fractions.
    points_norm = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]

    result = annotate_zone(frame, points_norm)

    decoded = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (200, 400, 3)
    # The right edge of the drawn square is at x=0.5*400=200 -- check a pixel on that
    # vertical line (well inside the frame, away from corners) is no longer solid black.
    assert not np.array_equal(decoded[100, 200], [0, 0, 0])
    # A point far from any edge of the polygon stays untouched.
    assert np.array_equal(decoded[100, 350], [0, 0, 0])


def test_annotate_zone_draws_green_outline_for_valid_zone_type():
    frame = _solid_jpeg(400, 200, color=(0, 0, 0))
    points_norm = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]

    result = annotate_zone(frame, points_norm, "valid")

    decoded = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
    # BGR encoding of the frontend's --green token (#6bf2a8); JPEG re-encoding
    # introduces a few units of compression noise, so allow tolerance.
    assert np.allclose(decoded[100, 200], [168, 242, 107], atol=25)


def test_annotate_zone_defaults_to_red_outline_for_restricted_zone_type():
    frame = _solid_jpeg(400, 200, color=(0, 0, 0))
    points_norm = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]

    result = annotate_zone(frame, points_norm, "restricted")

    decoded = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
    assert np.allclose(decoded[100, 200], [0, 0, 255], atol=25)


def test_annotate_zone_returns_frame_unchanged_when_fewer_than_3_points():
    frame = _solid_jpeg(100, 100)
    result = annotate_zone(frame, [(0.1, 0.1), (0.9, 0.9)])
    assert result == frame


def test_annotate_zone_returns_frame_unchanged_when_undecodable():
    garbage = b"not a real jpeg"
    assert annotate_zone(garbage, [(0.1, 0.1), (0.5, 0.1), (0.5, 0.9)]) == garbage


# ── zone_overlay_png ────────────────────────────────────────────────────────

_SQUARE = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]


def _edge_pixel(path):
    """The drawn pixel on the square's top edge, as (b, g, r, a).

    Reads a 3-row band because the outline is 2px thick and cv2.polylines centres
    the stroke on the path, so the exact row it lands on is not worth pinning.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    band = img[49:52, 200]
    drawn = [px for px in band if px[3] == 255]
    assert drawn, "expected the top edge of the square to be drawn"
    return tuple(int(c) for c in drawn[0])


def test_zone_overlay_png_draws_the_zone_as_the_zone_editor_does(tmp_path):
    """An opaque outline over a 20% fill, like the zone editor's canvas (ZoneEditorOverlay.tsx)."""
    out = str(tmp_path / "z.png")
    assert zone_overlay_png(_SQUARE, 400, 200, out) == out
    img = cv2.imread(out, cv2.IMREAD_UNCHANGED)
    assert img.shape == (200, 400, 4)          # height, width, BGRA
    assert img[100, 200, 3] == 51              # the inside: the editor's 20% fill
    assert img[10, 10, 3] == 0                 # outside the zone: transparent
    assert _edge_pixel(out)[3] == 255          # the edge itself is opaque


def test_zone_overlay_png_colours_by_zone_purpose_as_the_editor(tmp_path):
    """The editor's zoneColor: restricted --accent, valid --green, crowding only (intrusion off) --yellow."""
    paths = {mode: str(tmp_path / f"{mode}.png") for mode in ("restricted", "valid", "off")}
    for mode, path in paths.items():
        zone_overlay_png(_SQUARE, 400, 200, path, mode)
    assert _edge_pixel(paths["restricted"])[:3] == (108, 153, 249)     # #f9996c
    assert _edge_pixel(paths["valid"])[:3] == (168, 242, 107)          # #6bf2a8
    assert _edge_pixel(paths["off"])[:3] == (102, 209, 255)            # #ffd166


def test_zone_overlay_png_rejects_a_degenerate_polygon(tmp_path):
    out = str(tmp_path / "z.png")
    assert zone_overlay_png([(0.1, 0.1), (0.9, 0.9)], 400, 200, out) is None
    assert not os.path.exists(out)


def test_zone_overlay_png_rejects_a_zero_size_canvas(tmp_path):
    out = str(tmp_path / "z.png")
    assert zone_overlay_png(_SQUARE, 0, 200, out) is None
