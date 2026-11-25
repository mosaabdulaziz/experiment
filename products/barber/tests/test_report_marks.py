import io

from PIL import Image

from report_marks import mark


def jpeg(w=320, h=180):
    out = io.BytesIO()
    Image.new("RGB", (w, h), (255, 255, 255)).save(out, "JPEG")
    return out.getvalue()


def red(im, x, y):
    r, g, b = im.getpixel((x, y))
    return r > 180 and g < 120 and b < 120


def test_a_wait_gets_a_red_box_round_the_customer():
    """2026-10-18 (user): the report's alert pictures carry the Preview's mark."""
    im = Image.open(io.BytesIO(mark(jpeg(), (0.25, 0.5, 0.5, 0.25), None, "#12")))
    assert red(im, 80, 112) and red(im, 239, 112)             # its left and right edges, half way down
    assert not red(im, 160, 112)                              # the customer is not painted over


def test_a_long_haircut_gets_its_chair_zone_outline():
    zone = [[0.5, 0.2], [0.9, 0.2], [0.9, 0.8], [0.5, 0.8]]
    im = Image.open(io.BytesIO(mark(jpeg(), (0.5, 0.2, 0.4, 0.6), zone, "VIP2")))
    assert red(im, 160, 90) and red(im, 288, 90) and not red(im, 224, 90)


def test_without_a_mark_or_a_picture_it_changes_nothing():
    raw = jpeg()
    assert mark(raw, None, None, None) == raw
    assert mark(b"not a jpeg", (0.1, 0.1, 0.2, 0.2), None, None) == b"not a jpeg"
