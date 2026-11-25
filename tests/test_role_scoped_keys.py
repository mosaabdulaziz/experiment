"""Role-scoped Redis keys: several role sidecars on one camera without colliding (safety integration, ticket 1).

With `role_scoped_keys: true` a role process writes every per-camera key under `<camera>@<role>` instead of
`<camera>`, so a proximity sidecar and a PPE sidecar on the same camera keep separate status (boot_id), meta,
preview, overlay, alert and clip streams. Off (the default) every key name is exactly what it was."""
import json
import pathlib
import re
import sys
from unittest import mock

import pytest

try:                                    # the DeepStream image has GStreamer; the CI test image does not
    import gi  # noqa: F401
except ImportError:
    sys.modules["gi"] = mock.MagicMock()
    sys.modules["gi.repository"] = mock.MagicMock()

from perception.engine.config import SidecarConfig
from perception.engine.modules import EngineServices
from perception.engine.modules import camera_keyed_modules
from perception.engine.sources import (make_stream_state, role_scope_problem, split_stream_key_id,
                                       stale_status_keys, stream_key_id)
from perception.engine.status import _write_status

ROOT = pathlib.Path(__file__).resolve().parents[1]


class _SetRedis:
    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value


def _services(**kw):
    sent = []
    services = EngineServices(redis=None, redis_factory=lambda: None,
                              emit=lambda key, fields, maxlen=64: sent.append((key, fields)) or True,
                              request_clip=lambda *a, **k: None, stream_ids=["cam-1"], timer=lambda *a: None,
                              telemetry=None, **kw)
    return services, sent


def test_unscoped_key_id_is_the_camera_id_even_with_a_role():
    assert stream_key_id("cam-1", "barber", scoped=False) == "cam-1"
    assert stream_key_id("cam-1", None, scoped=False) == "cam-1"


def test_scoped_key_id_names_camera_and_role():
    assert stream_key_id("cam-1", "ppe", scoped=True) == "cam-1@ppe"


def test_scoped_keys_need_a_role():
    with pytest.raises(ValueError, match="role"):
        stream_key_id("cam-1", None, scoped=True)


def test_scoped_keys_refuse_a_camera_id_with_the_separator():
    with pytest.raises(ValueError, match="@"):
        stream_key_id("cam@1", "ppe", scoped=True)


def test_split_stream_key_id():
    assert split_stream_key_id("cam-1@ppe") == ("cam-1", "ppe")
    assert split_stream_key_id("cam-1") == ("cam-1", None)


def test_stream_state_keys_default_to_the_camera_id():
    st = make_stream_state("cam-1")
    assert (st.id, st.key_id, st.meta_key, st.jpeg_key, st.status_key) == (
        "cam-1", "cam-1", "iris:meta:cam-1", "iris:jpeg:cam-1", "iris:status:cam-1")


def test_stream_state_keys_follow_the_key_id_while_the_id_stays_the_camera():
    st = make_stream_state("cam-1", key_id="cam-1@ppe")
    assert st.id == "cam-1"
    assert (st.key_id, st.meta_key, st.jpeg_key, st.status_key) == (
        "cam-1@ppe", "iris:meta:cam-1@ppe", "iris:jpeg:cam-1@ppe", "iris:status:cam-1@ppe")


def test_status_written_under_the_key_id():
    r = _SetRedis()
    _write_status(r, {"id": "cam-1", "uri": "rtsp://go2rtc:8554/raw-cam-1"}, running=True, boot_id="b1",
                  key_id="cam-1@ppe")
    assert list(r.values) == ["iris:status:cam-1@ppe"]
    assert json.loads(r.values["iris:status:cam-1@ppe"])["boot_id"] == "b1"


def test_status_key_unchanged_without_a_key_id():
    r = _SetRedis()
    _write_status(r, {"id": "cam-1", "uri": "file:///x.mp4"}, running=False)
    assert list(r.values) == ["iris:status:cam-1"]


@pytest.fixture
def quiet_telemetry(monkeypatch):
    """SidecarConfig configures the process-wide telemetry; keep these tests from switching it for later ones."""
    from perception.engine import telemetry
    monkeypatch.setattr(telemetry, "configure", lambda *a, **k: False)


def test_sidecar_config_key_id_is_off_by_default(quiet_telemetry):
    gcfg = SidecarConfig({"redis_url": "redis://127.0.0.1:1/0", "record_enabled": False}, role="barber")
    assert gcfg.key_id("cam-1") == "cam-1"


def test_sidecar_config_key_id_when_scoped(quiet_telemetry):
    gcfg = SidecarConfig({"redis_url": "redis://127.0.0.1:1/0", "record_enabled": False, "role_scoped_keys": True},
                         role="ppe")
    assert gcfg.key_id("cam-1") == "cam-1@ppe"


def test_module_alert_and_overlay_go_to_the_key_id():
    services, sent = _services(key_id=lambda sid: f"{sid}@ppe")
    services.alert("cam-1", "ppe_violation")
    services.overlay("cam-1", "ppe", 1.0, (10, 10), [])
    assert [key for key, _ in sent] == ["iris:alert:cam-1@ppe", "iris:overlay:cam-1@ppe"]


def test_module_alert_key_unchanged_by_default():
    services, sent = _services()
    services.alert("cam-1", "ppe_violation")
    assert sent[0][0] == "iris:alert:cam-1"


def test_scoped_purge_only_touches_its_own_role():
    existing = ["iris:status:cam-1@ppe", "iris:status:cam-9@ppe", "iris:status:cam-9@fall", "iris:status:cam-9"]
    assert stale_status_keys(existing, {"cam-1"}, role="ppe", scoped=True) == ["iris:status:cam-9@ppe"]


def test_unscoped_purge_is_unchanged():
    existing = [b"iris:status:cam-1", b"iris:status:cam-9", b"iris:status:cam-9@ppe"]
    assert stale_status_keys(existing, {"cam-1"}, role=None, scoped=False) == [
        b"iris:status:cam-9", b"iris:status:cam-9@ppe"]


def test_modules_that_write_camera_keyed_streams_say_so_and_are_named_under_scoped_keys():
    """A module whose own workers write per-camera streams named by the camera id (not services.key_id) declares
    `camera_keyed_streams`; the engine refuses it under role_scoped_keys. The engine itself names no module."""
    import types
    loaded = types.SimpleNamespace(modules=[types.SimpleNamespace(name="zones", camera_keyed_streams=False),
                                            types.SimpleNamespace(name="face_recognition", camera_keyed_streams=True)])
    assert camera_keyed_modules(loaded) == ["face_recognition"]


@pytest.mark.parametrize("module", ["face_recognition", "customer_faces", "appearance"])
def test_the_camera_keyed_modules_declare_it(module):
    source = (ROOT / "perception" / "modules" / module / "__init__.py").read_text()
    assert re.search(r"^\s+camera_keyed_streams = True", source, re.M), module


def test_the_engine_names_no_module():
    source = (ROOT / "perception" / "engine" / "sources.py").read_text()
    for name in ("face_recognition", "customer_faces", "appearance", "safety product", "products/safety"):
        assert name not in source, name


def test_startup_check_passes_an_unscoped_process():
    assert role_scope_problem({"modules": {"face_recognition": {}}}, None, [{"id": "cam@1"}]) is None


def test_startup_check_passes_a_scoped_role_process():
    assert role_scope_problem({"role_scoped_keys": True, "modules": {"ppe": {}}}, "ppe", [{"id": "cam-1"}]) is None


def test_startup_check_refuses_scoped_keys_without_a_role():
    assert "role" in role_scope_problem({"role_scoped_keys": True}, None, [{"id": "cam-1"}])


def test_startup_check_refuses_a_camera_id_with_the_separator():
    assert "@" in role_scope_problem({"role_scoped_keys": True}, "ppe", [{"id": "cam@1"}])
