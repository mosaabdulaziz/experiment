"""The module contract (perception/engine/modules.py): loading, settings, ordering, and failure isolation."""
import sys
import textwrap
import types

import pytest

from perception.engine.frame import FrameOutput, FrameView, ObjectView, detection_rows
from perception.engine.modules import (EngineServices, Module, ModuleConfigError, ModuleSet, ModuleStartupError,
                                       PipelineHandle, build_settings,
                                       load_modules)

PKG = "fake_modules_pkg"


def _services():
    return EngineServices(redis=None, redis_factory=lambda: None, emit=lambda *a, **k: None,
                          request_clip=lambda *a, **k: None, stream_ids=["cam-1"], timer=lambda *a: None,
                          telemetry=None)


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    """A throwaway modules package on disk; tests add module folders to it."""
    root = tmp_path / PKG
    root.mkdir()
    (root / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [n for n in sys.modules if n == PKG or n.startswith(PKG + ".")]:
        monkeypatch.delitem(sys.modules, name)

    def add(name, body):
        d = root / name
        d.mkdir()
        (d / "__init__.py").write_text(textwrap.dedent(body))
    return add


SIMPLE = """
    from dataclasses import dataclass
    from perception.engine.modules import Module
    @dataclass
    class Settings:
        label: str
        threshold: int = 3
    class M(Module):
        name = "{name}"
        Settings = Settings
        requires = {requires}
        after = {after}
        calls = []
        def analyze(self, frame, out):
            M.calls.append(self.name)
    MODULE = M
"""


def _add(pkg, name, requires=(), after=()):
    pkg(name, SIMPLE.format(name=name, requires=repr(tuple(requires)), after=repr(tuple(after))))


def test_loads_in_the_order_given_and_builds_settings(pkg):
    _add(pkg, "alpha")
    _add(pkg, "beta")
    ms = load_modules({"beta": {"label": "x"}, "alpha": {"label": "y", "threshold": 9}}, _services(), PKG)
    assert ms.names() == ["beta", "alpha"]
    assert ms.get("alpha").settings.threshold == 9 and ms.get("beta").settings.threshold == 3


def test_empty_or_missing_modules_section_loads_nothing():
    assert not load_modules(None, _services(), PKG)
    assert not load_modules({}, _services(), PKG)


def test_unknown_setting_refuses_startup(pkg):
    _add(pkg, "alpha")
    with pytest.raises(ModuleConfigError, match="unknown setting.*thresh"):
        load_modules({"alpha": {"label": "x", "thresh": 1}}, _services(), PKG)


def test_missing_required_setting_refuses_startup(pkg):
    _add(pkg, "alpha")
    with pytest.raises(ModuleConfigError, match="missing required setting.*label"):
        load_modules({"alpha": {}}, _services(), PKG)


def test_unknown_module_refuses_startup(pkg):
    with pytest.raises(ModuleConfigError, match="no such module"):
        load_modules({"ghost": {}}, _services(), PKG)


@pytest.mark.parametrize("name", ["../etc", "Upper", "_hidden", "a.b", "1abc", "a-b"])
def test_a_module_name_cannot_escape_the_package(name):
    with pytest.raises(ModuleConfigError, match="not a valid module name"):
        load_modules({name: {}}, _services(), PKG)


def test_an_import_error_inside_a_module_is_not_reported_as_unknown_module(pkg):
    pkg("broken", "import not_a_real_dependency_xyz\n")
    with pytest.raises(ModuleNotFoundError, match="not_a_real_dependency_xyz"):
        load_modules({"broken": {}}, _services(), PKG)


def test_package_must_expose_a_module_class(pkg):
    pkg("plain", "x = 1\n")
    with pytest.raises(ModuleConfigError, match="must expose MODULE"):
        load_modules({"plain": {}}, _services(), PKG)


def test_package_name_must_match_the_module_name(pkg):
    pkg("one", SIMPLE.format(name="other", requires="()", after="()"))
    with pytest.raises(ModuleConfigError, match="named 'other'"):
        load_modules({"one": {"label": "x"}}, _services(), PKG)


def test_requires_must_be_present_and_earlier(pkg):
    _add(pkg, "base")
    _add(pkg, "dependent", requires=["base"])
    with pytest.raises(ModuleConfigError, match="requires .*base.*not configured"):
        load_modules({"dependent": {"label": "x"}}, _services(), PKG)
    with pytest.raises(ModuleConfigError, match="must be listed before"):
        load_modules({"dependent": {"label": "x"}, "base": {"label": "x"}}, _services(), PKG)
    assert load_modules({"base": {"label": "x"}, "dependent": {"label": "x"}}, _services(), PKG).names() == \
        ["base", "dependent"]


def test_after_only_constrains_the_order_when_the_other_module_is_there(pkg):
    _add(pkg, "base")
    _add(pkg, "reader", after=["base"])
    assert load_modules({"reader": {"label": "x"}}, _services(), PKG).names() == ["reader"]
    with pytest.raises(ModuleConfigError, match="must be listed after .*base"):
        load_modules({"reader": {"label": "x"}, "base": {"label": "x"}}, _services(), PKG)
    assert load_modules({"base": {"label": "x"}, "reader": {"label": "x"}}, _services(), PKG).names() == \
        ["base", "reader"]


def test_settings_none_means_no_settings():
    assert build_settings(None, None, "m") is None
    with pytest.raises(ModuleConfigError, match="takes no settings"):
        build_settings(None, {"a": 1}, "m")


class _Boom(Module):
    name = "boom"

    def analyze(self, frame, out):
        raise RuntimeError("bug")


class _Ok(Module):
    name = "ok"
    seen = 0

    def analyze(self, frame, out):
        _Ok.seen += 1


def test_one_failing_module_does_not_stop_the_others_or_the_caller():
    _Ok.seen = 0
    ms = ModuleSet([_Boom(None, _services()), _Ok(None, _services())])
    frame, out = FrameView("cam-1", None, 0, 1, 0.1, 0.0), FrameOutput()
    for _ in range(3):
        ms.run("analyze", frame, out)
    assert _Ok.seen == 3 and ms.failures == {("boom", "analyze"): 3}


def test_predicate_hooks_ask_every_module():
    class Yes(Module):
        name = "yes"
        asked = 0

        def wants_pixels(self, frame):
            Yes.asked += 1
            return True

    class No(Module):
        name = "no"
        asked = 0

        def wants_pixels(self, frame):
            No.asked += 1
            return False

    ms = ModuleSet([Yes(None, None), No(None, None)])
    assert ms.any("wants_pixels", None) is True and Yes.asked == 1 and No.asked == 1


def test_modules_that_listen_to_one_control_key_all_hear_it_in_order():
    """One signal, several listeners: the zones version tells the zones module and the PPE / fire zones (perception/lib
    /feature_zones.py) alike that the zones changed -- a process running zones and fire (attendance) has both."""
    heard = []

    class A(Module):
        name = "a"

        def control(self):
            return {"iris:ctrl:x": lambda v: heard.append(("a", v))}

    class B(Module):
        name = "b"

        def control(self):
            return {"iris:ctrl:x": lambda v: heard.append(("b", v)), "iris:ctrl:y": lambda v: heard.append(("y", v))}

    controls = ModuleSet([A(None, None), B(None, None)]).controls()
    controls["iris:ctrl:x"]("7")
    assert heard == [("a", "7"), ("b", "7")] and set(controls) == {"iris:ctrl:x", "iris:ctrl:y"}


def test_one_listener_failing_does_not_silence_the_others_and_the_change_is_retried():
    heard = []

    class Broken(Module):
        name = "broken"

        def control(self):
            return {"iris:ctrl:x": lambda v: 1 / 0}

    class Fine(Module):
        name = "fine"

        def control(self):
            return {"iris:ctrl:x": lambda v: heard.append(v)}

    controls = ModuleSet([Broken(None, None), Fine(None, None)]).controls()
    with pytest.raises(ZeroDivisionError):                   # the poller retries the change on its next tick
        controls["iris:ctrl:x"]("7")
    assert heard == ["7"]


def test_objects_are_extracted_once_and_lazily():
    calls = []
    f = FrameView("cam-1", None, 0, 1, 0.1, 0.0,
                  _objects_loader=lambda: calls.append(1) or [ObjectView(1, "person", 0.9, 0.9, 0, 0, 10, 10)])
    assert calls == []
    assert f.objects is f.objects and len(calls) == 1


def test_a_shadow_track_shows_the_trackers_confidence():
    o = ObjectView(1, "person", -0.1, 0.62, 0, 0, 1, 1)
    assert not o.detected and o.shown_conf == 0.62
    assert ObjectView(1, "person", 0.8, 0.1, 0, 0, 1, 1).shown_conf == 0.8


def _obj(oid=1, label="person"):
    return ObjectView(oid, label, 0.9, 0.9, 0, 0, 1, 1)


def test_frame_output_collects_annotations_and_asks_for_pixels():
    out = FrameOutput()
    o = _obj(7)
    out.annotate(o, color=(255, 0, 0))
    out.annotate(o, text="x")
    out.force_pixels()
    assert out.annotations[o].color == (255, 0, 0) and out.annotations[o].text == "x" and out.need_pixels


def test_annotations_belong_to_an_object_not_to_its_id():
    # a face box is drawn on its owner's id, so it shares the id of the person box; they must style apart
    person, face = _obj(7, "person"), _obj(7, "face")
    out = FrameOutput()
    out.annotate(face, color=(1, 2, 3))
    assert person not in out.annotations and out.annotations[face].color == (1, 2, 3)


def test_detection_rows_follow_the_publish_flag_and_the_wire_shape():
    plain, shadow, hidden = _obj(1), ObjectView(2, "person", -0.1, 0.62, 10, 20, 30, 40), _obj(3)
    hidden.publish = False
    assert detection_rows([plain, shadow, hidden], full=False) == [["person", 0.9, 1], ["person", -0.1, 2]]
    assert detection_rows([shadow, hidden], full=True) == [["person", 0.62, 2, 10, 20, 30, 40]]


def test_the_template_loads_and_is_not_configurable_by_name():
    from perception.modules import _template
    assert issubclass(_template.MODULE, Module)
    with pytest.raises(ModuleConfigError, match="not a valid module name"):
        load_modules({"_template": {}}, _services())


def test_annotation_priority_does_not_depend_on_module_order():
    for order in ((0, 10), (10, 0)):
        out, o = FrameOutput(), _obj()
        for prio in order:
            out.annotate(o, color=(prio, prio, prio), text=f"p{prio}", priority=prio)
        assert out.annotations[o].color == (10, 10, 10) and out.annotations[o].text == "p10"


def test_an_equal_priority_annotation_replaces_and_empty_text_suppresses_the_label():
    out, o = FrameOutput(), _obj()
    out.annotate(o, text="first")
    out.annotate(o, text="")
    assert out.annotations[o].text == ""


def test_extra_text_accumulates_and_fields_are_free_form():
    out, o = FrameOutput(), _obj()
    out.annotate(o, extra=("gid:7", (1, 2, 3), 0.75, 2))
    out.annotate(o, extra=("more", (1, 2, 3), 0.5, 1))
    out.fields["plates"] = [[1, 2]]
    assert len(out.annotations[o].extra) == 2 and out.fields == {"plates": [[1, 2]]}


def test_first_returns_the_first_opinion_in_module_order_and_skips_failures():
    class Silent(Module):
        name = "silent"

    class Boom(Module):
        name = "boom"

        def sample(self, frame):
            raise RuntimeError("x")

    class Says(Module):
        name = "says"

        def sample(self, frame):
            return False

    class Late(Module):
        name = "late"

        def sample(self, frame):
            return True

    ms = ModuleSet([Silent(None, None), Boom(None, None), Says(None, None), Late(None, None)])
    assert ms.first("sample", None) is False
    assert ModuleSet([Silent(None, None)]).first("sample", None) is None


def test_a_build_time_hook_that_fails_refuses_startup_but_a_frame_hook_does_not():
    class Broken(Module):
        name = "broken"

        def on_built(self, pipeline):
            raise RuntimeError("no element")

        def on_frame(self, frame, out):
            raise RuntimeError("bad frame")

    modules = ModuleSet([Broken(None, _services())])
    with pytest.raises(ModuleStartupError, match="broken.on_built failed: no element"):
        modules.setup("on_built", PipelineHandle(pipeline=None, get=lambda n: None))
    modules.run("on_frame", None, None)                       # a bad frame is logged and counted, never fatal
    assert modules.failures == {("broken", "on_frame"): 1}
