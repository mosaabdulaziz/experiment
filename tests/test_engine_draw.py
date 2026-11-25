"""How the preview is drawn: annotations (priorities) and modules' own draw functions. perception/engine/publisher.py
can only be imported inside the DeepStream image, so the one function is lifted out of the source."""
import ast
import pathlib
import types

import numpy as np
import cv2

from perception.engine.frame import FrameOutput, ObjectView

SRC = (pathlib.Path(__file__).resolve().parents[1] / "perception" / "engine" / "publisher.py").read_text()


def _draw():
    node = next(n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == "_draw_and_encode")
    node.decorator_list = []
    ns = {"cv2": cv2, "BOX_DEFAULT": (0, 255, 0), "sys": __import__("sys")}
    exec(compile(ast.Module([node], []), "publisher.py", "exec"), ns)
    return ns["_draw_and_encode"]


GCFG = types.SimpleNamespace(capture_w=200, capture_h=100, mux_w=100, mux_h=50, jpeg_quality=80)


def _obj(label="person", oid=3):
    return ObjectView(oid, label, 0.9, 0.9, 10, 10, 30, 20)


def _texts(monkeypatch):
    seen = []
    monkeypatch.setattr(cv2, "putText", lambda img, text, *a, **k: seen.append(text))
    return seen


def test_an_ordinary_box_gets_the_default_caption(monkeypatch):
    seen = _texts(monkeypatch)
    _draw()(np.zeros((100, 200, 3), np.uint8), [_obj()], GCFG)
    assert seen == ["person 0.90 #3"]


def test_a_module_can_suppress_the_caption_and_another_module_can_replace_it_whatever_the_order(monkeypatch):
    seen = _texts(monkeypatch)
    face = _obj("face")
    for order in ((("", -1), ("Ann", 0)), (("Ann", 0), ("", -1))):
        out = FrameOutput()
        for text, prio in order:
            out.annotate(face, text=text, priority=prio)
        seen.clear()
        _draw()(np.zeros((100, 200, 3), np.uint8), [face], GCFG, annotations=out.annotations)
        assert seen == ["Ann"]
    out = FrameOutput()
    out.annotate(face, text="", priority=-1)
    seen.clear()
    _draw()(np.zeros((100, 200, 3), np.uint8), [face], GCFG, annotations=out.annotations)
    assert seen == []


def test_a_module_can_have_a_box_drawn_where_it_says_instead_of_the_detection():
    o = _obj()                                          # mux (10, 10, 30, 20): capture (20, 20) - (80, 60)
    out = FrameOutput()
    out.annotate(o, box=(20.0, 5.0, 30.0, 20.0))        # a steadier box: capture (40, 10) - (100, 50)
    img = np.zeros((100, 200, 3), np.uint8)
    _draw()(img, [o], GCFG, annotations=out.annotations)
    assert img[10, 40:100].any() and img[49, 40:100].any()
    assert not img[59, 20:39].any()                     # nothing where the detection itself was


def test_module_draw_functions_run_after_the_boxes_with_the_capture_scale():
    calls = []
    out = FrameOutput()
    out.draw(lambda img, sx, sy: calls.append((img.shape, sx, sy)))
    img = np.zeros((100, 200, 3), np.uint8)
    assert _draw()(img, [_obj()], GCFG, draws=out.draws)
    assert calls == [((100, 200, 3), 2.0, 2.0)]
    assert img[20:22, 20:60].any()                       # the box was already on the image when the module drew
