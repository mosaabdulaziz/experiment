"""tools/safety_controller.py: the safety stack's cameras and per-camera toggles -> each role sidecar's config, and
only the sidecars whose cameras changed restarted (safety integration, ticket 1, decisions D11/D16/D29)."""
import importlib.util
import json
import pathlib
import sys

import pytest
import yaml

from perception.engine.sources import load_sidecar_config

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("safety_controller", ROOT / "tools" / "safety_controller.py")
sc = importlib.util.module_from_spec(_spec)
sys.modules["safety_controller"] = sc
_spec.loader.exec_module(sc)
PRODUCT = sc.load_product(ROOT / "products/safety/safety.yaml")

CAMERAS = [
    {"id": "bay-a", "name": "Bay A", "clip": "a.mp4", "features": ["proximity", "zones", "ppe"]},
    {"id": "store", "name": "Store", "rtsp": "rtsp://${CAM_USER}:${CAM_PASS}@10.0.0.5/s1", "features": ["fire"]},
]


def test_toggles_saved_by_the_ui_override_the_file():
    eff = sc.effective_cameras(CAMERAS, {"bay-a": {"features": ["ppe"], "name": "Loading bay"}}, PRODUCT)
    assert [(c["id"], c["name"], c["features"]) for c in eff] == [
        ("bay-a", "Loading bay", ["ppe"]), ("store", "Store", ["fire"])]


def test_an_unknown_feature_is_refused():
    with pytest.raises(ValueError, match="teleport"):
        sc.effective_cameras([{"id": "x", "clip": "x.mp4", "features": ["teleport"]}], {}, PRODUCT)


def test_a_camera_id_with_the_role_separator_is_refused():
    with pytest.raises(ValueError, match="@"):
        sc.effective_cameras([{"id": "a@b", "clip": "x.mp4", "features": []}], {}, PRODUCT)


def test_each_role_gets_the_cameras_that_have_its_feature_through_go2rtc():
    sources = sc.role_sources(sc.effective_cameras(CAMERAS, {}, PRODUCT), PRODUCT)
    assert sources["proximity"] == [{"id": "bay-a", "uri": "rtsp://go2rtc:8554/raw-bay-a", "roles": ["proximity"]}]
    assert [s["id"] for s in sources["fire"]] == ["store"]
    assert sources["fall"] == []


def test_a_rendered_role_config_extends_the_role_and_carries_the_common_settings(tmp_path):
    rendered = sc.render_role("ppe", [{"id": "bay-a", "uri": "rtsp://go2rtc:8554/raw-bay-a", "roles": ["ppe"]}],
                              PRODUCT)
    assert rendered["extends"] == "../../../products/safety/roles/ppe.yaml"
    assert rendered["sidecar_role"] == "ppe" and rendered["role_scoped_keys"] is True
    assert (rendered["mux_width"], rendered["mux_height"], rendered["record_enabled"]) == (1280, 720, False)
    assert rendered["svr_dir"] == "/workspace/iris/data/alert_svr/ppe"
    assert rendered["svr_dir_app"] == "data/alert_svr/ppe"


def test_every_rendered_role_draws_its_alerts_objects_and_stamps_frame_arrival():
    """The safety roles write their objects' boxes for the clips (overlay_boxes; no other product does) and stamp each
    frame's arrival on this machine's clock (rtcp_camera_clock off: go2rtc's relay drops the camera's RTCP), which
    places those boxes on a smart-record clip without the inference delay."""
    for role in PRODUCT["roles"]:
        rendered = sc.render_role(role, [{"id": "c", "uri": "rtsp://go2rtc:8554/raw-c", "roles": [role]}], PRODUCT)
        assert (rendered.get("overlay_boxes"), rendered.get("rtcp_camera_clock")) == (True, False), role
    base = yaml.safe_load((sc.REPO / "perception/pipeline/sidecar.yaml").read_text())
    assert not base.get("overlay_boxes")                       # every other product: off


def test_a_rendered_role_config_loads_through_the_engine_s_extends_chain(tmp_path):
    # Laid out as in a container: the repo at the root, the stack's data dir at <root>/data.
    root = tmp_path / "root"
    (root / "data" / "safety" / "generated").mkdir(parents=True)
    for name in ("products", "perception", "configs"):
        (root / name).symlink_to(ROOT / name)
    rendered = sc.render_role("zones", [{"id": "office", "uri": "rtsp://go2rtc:8554/raw-office", "roles": ["zones"]}],
                              PRODUCT)
    path = root / "data" / "safety" / "generated" / "zones.yaml"
    path.write_text(yaml.safe_dump(rendered))
    cfg = load_sidecar_config(str(path))
    assert cfg["modules"]["zones"]["end_markers"] is True
    assert cfg["sources"] == rendered["sources"] and cfg["role_scoped_keys"] is True


@pytest.mark.parametrize("role", ["proximity", "zones", "ppe", "fall", "fire"])
def test_every_role_file_loads(role):
    cfg = load_sidecar_config(str(ROOT / f"products/safety/roles/{role}.yaml"))
    assert cfg.get("modules"), f"role {role} runs no module"


def test_the_fall_service_config_takes_the_fall_role_s_cameras_and_the_stack_s_zones(tmp_path):
    rendered = sc.render_fall_service([{"id": "rest-room"}], PRODUCT, ROOT, tmp_path)
    assert rendered["extends"] == str(ROOT / "products/safety/fall_detection.yaml")
    assert rendered["cameras"] == [{"id": "rest-room"}]
    assert rendered["zones"] == {"path": str(tmp_path / "zones_config.txt"), "mux": [1280, 720]}


def test_go2rtc_publishes_demo_clips_and_pulls_real_cameras():
    g = sc.render_go2rtc(sc.effective_cameras(CAMERAS, {}, PRODUCT))
    assert g["streams"]["raw-bay-a"] is None                       # the demo streamer publishes into it
    assert g["streams"]["raw-store"] == ["rtsp://${CAM_USER}:${CAM_PASS}@10.0.0.5/s1"]
    assert g["streams"]["bay-a"][0] == "rtsp://127.0.0.1:8554/raw-bay-a"


def test_writing_reports_only_the_files_that_changed(tmp_path):
    first = sc.write_generated(tmp_path, {"ppe": {"a": 1}, "fire": {"b": 2}})
    assert first == {"ppe", "fire"}
    again = sc.write_generated(tmp_path, {"ppe": {"a": 1}, "fire": {"b": 3}})
    assert again == {"fire"}
    assert yaml.safe_load((tmp_path / "fire.yaml").read_text()) == {"b": 3}


def test_the_plan_restarts_changed_roles_starts_missing_ones_and_stops_empty_ones():
    sources = {"proximity": [{"id": "a"}], "zones": [{"id": "a"}], "ppe": [], "fall": [], "fire": [{"id": "s"}]}
    plan = sc.plan(sources, changed={"zones", "ppe"}, running={"proximity", "zones", "ppe"})
    assert plan == {"restart": ["zones"], "start": ["fire"], "stop": ["ppe"]}


def test_nothing_changed_and_everything_running_is_an_empty_plan():
    sources = {"proximity": [{"id": "a"}], "zones": [], "ppe": [], "fall": [], "fire": []}
    assert sc.plan(sources, changed=set(), running={"proximity"}) == {"restart": [], "start": [], "stop": []}


class _Runner:
    def __init__(self, running=""):
        self.calls, self.running = [], running

    def __call__(self, args):
        self.calls.append(list(args))
        return self.running if "ps" in args else ""


def test_applying_runs_compose_only_for_the_roles_in_the_plan():
    runner = _Runner()
    sc.apply_plan({"restart": ["zones"], "start": ["fire"], "stop": ["ppe"]}, ["docker", "compose", "-p", "s"], runner)
    assert runner.calls == [
        ["docker", "compose", "-p", "s", "stop", "safety-ppe"],
        ["docker", "compose", "-p", "s", "up", "-d", "--no-build", "safety-fire"],
        ["docker", "compose", "-p", "s", "restart", "safety-zones"]]


def test_running_roles_are_read_from_compose():
    runner = _Runner(running="iris\nsafety-ppe\nsafety-zones\nredis\n")
    assert sc.running_roles(["docker", "compose"], runner) == {"ppe", "zones"}


def test_the_state_file_is_optional(tmp_path):
    assert sc.load_state(tmp_path / "absent.json") == {}
    (tmp_path / "s.json").write_text(json.dumps({"bay-a": {"features": ["ppe"]}}))
    assert sc.load_state(tmp_path / "s.json") == {"bay-a": {"features": ["ppe"]}}


def test_the_demo_clips_are_listed_per_camera(tmp_path):
    eff = sc.effective_cameras(CAMERAS, {}, PRODUCT)
    assert sc.demo_clips(eff, tmp_path) == [("bay-a", str(tmp_path / "a.mp4"))]


# ── go2rtc picks up new cameras through its API, not a restart (a restart would cut every camera's pull) ──────────

def test_the_streams_go2rtc_lacks_are_the_new_ones():
    generated = sc.render_go2rtc(sc.effective_cameras(CAMERAS, {}, PRODUCT))
    live = {"raw-store", "store"}
    assert set(sc.go2rtc_missing(generated, live)) == {"raw-bay-a", "bay-a"}


class _Http:
    def __init__(self, live=()):
        self.live, self.puts = {name: {} for name in live}, []

    def __call__(self, method, url):
        if method == "GET":
            return json.dumps(self.live)
        self.puts.append(url)
        return ""


def test_a_publish_only_stream_is_registered_by_name_alone():
    http = _Http()
    sent, skipped = sc.register_streams({"raw-bay-a": None}, "http://localhost:51984", http)
    assert http.puts == ["http://localhost:51984/api/streams?name=raw-bay-a"]
    assert (sent, skipped) == (["raw-bay-a"], [])


def test_a_stream_with_sources_sends_each_one_encoded():
    http = _Http()
    sc.register_streams({"bay-a": ["rtsp://127.0.0.1:8554/raw-bay-a", "ffmpeg:rtsp://127.0.0.1:8554/raw-bay-a#video=h264"]},
                        "http://localhost:51984", http)
    (url,) = http.puts
    assert url.startswith("http://localhost:51984/api/streams?name=bay-a&src=rtsp%3A%2F%2F127.0.0.1%3A8554%2Fraw-bay-a")
    assert "&src=ffmpeg%3Artsp%3A%2F%2F127.0.0.1%3A8554%2Fraw-bay-a%23video%3Dh264" in url


def test_a_source_whose_credentials_are_not_in_this_environment_is_not_sent(monkeypatch):
    monkeypatch.delenv("CAM_USER", raising=False)
    http = _Http()
    sent, skipped = sc.register_streams({"raw-store": ["rtsp://${CAM_USER}:${CAM_PASS}@10.0.0.5/s1"]},
                                        "http://localhost:51984", http)
    assert http.puts == [] and (sent, skipped) == ([], ["raw-store"])


def test_apply_registers_new_streams_before_starting_the_roles(tmp_path):
    calls = []

    class Runner(_Runner):
        def __call__(self, args):
            calls.append(("compose", args[-1]))
            return super().__call__(args)

    def http(method, url):
        calls.append((method, url.split("?")[0] if method == "GET" else url.split("name=")[1].split("&")[0]))
        return "{}" if method == "GET" else ""

    cams = tmp_path / "cameras.yaml"
    cams.write_text(yaml.safe_dump({"cameras": [{"id": "yard", "clip": "y.mp4", "features": ["ppe"]}]}))
    args = sc.parse_args(["apply", "--data", str(tmp_path / "data"), "--cameras", str(cams),
                          "--compose", "docker compose -p s", "--go2rtc-api", "http://localhost:51984"])
    sc.apply(args, runner=Runner(), http=http)
    puts = [c for c in calls if c[0] == "PUT"]
    assert {name for _m, name in puts} == {"raw-yard", "yard"}
    assert calls.index(puts[-1]) < calls.index(("compose", "safety-ppe"))


# ── what was applied, not what this run rewrote, decides (a failed apply is retried; render alone changes nothing) ──

def _args_for(tmp_path, cameras):
    cams = tmp_path / "cameras.yaml"
    cams.write_text(yaml.safe_dump({"cameras": cameras}))
    return sc.parse_args(["apply", "--data", str(tmp_path / "data"), "--cameras", str(cams),
                          "--compose", "docker compose -p s"])


class _Failing(_Runner):
    def __call__(self, args):
        if "up" in args or "restart" in args:
            raise RuntimeError("docker said no")
        return super().__call__(args)


UP_PPE = ["docker", "compose", "-p", "s", "up", "-d", "--no-build", "safety-ppe"]
RESTART_PPE = ["docker", "compose", "-p", "s", "restart", "safety-ppe"]


def test_a_failed_restart_is_retried(tmp_path):
    args = _args_for(tmp_path, [{"id": "yard", "clip": "y.mp4", "features": ["ppe"]}])
    sc.apply(args, runner=_Runner(), http=None)                                       # ppe started with yard
    args = _args_for(tmp_path, [{"id": "yard", "clip": "y.mp4", "features": ["ppe"]},
                                {"id": "gate", "clip": "g.mp4", "features": ["ppe"]}])
    failing = _Failing(running="safety-ppe\n")
    with pytest.raises(RuntimeError):
        sc.apply(args, runner=failing, http=None)                                    # the restart fails
    retry = _Runner(running="safety-ppe\n")
    sc.apply(args, runner=retry, http=None)
    assert RESTART_PPE in retry.calls


def test_a_camera_go2rtc_cannot_get_is_not_applied_and_no_role_starts_for_it(tmp_path, monkeypatch):
    monkeypatch.delenv("CAM_USER", raising=False)
    monkeypatch.delenv("CAM_PASS", raising=False)
    cams = tmp_path / "cameras.yaml"
    cams.write_text(yaml.safe_dump({"cameras": [
        {"id": "store", "rtsp": "rtsp://${CAM_USER}:${CAM_PASS}@10.0.0.5/s1", "features": ["fire"]}]}))
    args = sc.parse_args(["apply", "--data", str(tmp_path / "data"), "--cameras", str(cams),
                          "--compose", "docker compose -p s", "--go2rtc-api", "http://localhost:51984"])
    runner = _Runner()
    with pytest.raises(sc.IncompleteApply, match="raw-store"):
        sc.apply(args, runner=runner, http=_Http())
    assert runner.calls == []                                    # no role started on a stream go2rtc lacks
    assert "go2rtc" not in sc.load_applied(args.data)            # nor recorded: the next apply tries again
    sc.apply(args, runner=runner, http=_Http(live=["raw-store"]))    # go2rtc restarted: it has the stream now
    assert ["docker", "compose", "-p", "s", "up", "-d", "--no-build", "safety-fire"] in runner.calls


class _Versions:
    """The app's side of an apply in Redis: the version asked for, and the one marked applied."""

    def __init__(self, version):
        self.values = {sc.APPLY_VERSION_KEY: str(version)}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = str(value)


def test_a_requested_version_is_applied_once_and_marked_applied():
    r, runs = _Versions(3), []
    poll = sc.ApplyWatch(r, lambda: runs.append(1) or {}, now=lambda: 0.0).poll
    poll()
    poll()
    assert runs == [1] and r.get(sc.APPLY_APPLIED_KEY) == "3"


def test_a_failing_apply_backs_off_gives_up_and_starts_afresh_when_apply_is_pressed_again():
    clock, runs = [0.0], []

    def run():
        runs.append(clock[0])
        raise RuntimeError("the fall service did not come up")
    r = _Versions(1)
    watch = sc.ApplyWatch(r, run, now=lambda: clock[0])
    while clock[0] < 1000:                                       # polled every 2 s
        watch.poll()
        clock[0] += 2.0
    assert runs == [0.0, 10.0, 30.0, 70.0, 150.0]                # 10, 20, 40, 80 s apart, then given up
    assert r.get(sc.APPLY_APPLIED_KEY) is None
    r.set(sc.APPLY_VERSION_KEY, 2)                               # Apply pressed again
    watch.poll()
    assert runs[-1] == 1000.0


def test_rendering_alone_leaves_the_change_pending(tmp_path):
    args = _args_for(tmp_path, [{"id": "yard", "clip": "y.mp4", "features": ["ppe"]}])
    sc.apply(args, runner=_Runner(), http=None)                                       # ppe started with yard
    args = _args_for(tmp_path, [{"id": "yard", "clip": "y.mp4", "features": ["ppe"]},
                                {"id": "gate", "clip": "g.mp4", "features": ["ppe"]}])
    sc.render(sc.load_product(args.product), sc.load_cameras(args.cameras), {}, args.data)    # `just safety-status`
    running = _Runner(running="safety-ppe\n")
    sc.apply(args, runner=running, http=None)
    assert RESTART_PPE in running.calls


def test_an_apply_after_a_successful_one_does_nothing(tmp_path):
    args = _args_for(tmp_path, [{"id": "yard", "clip": "y.mp4", "features": ["ppe"]}])
    sc.apply(args, runner=_Runner(), http=None)
    again = _Runner(running="safety-ppe\n")
    assert sc.apply(args, runner=again, http=None) == {"restart": [], "start": [], "stop": []}
    assert [c for c in again.calls if "ps" not in c] == []


def test_generated_files_are_replaced_whole(tmp_path):
    sc.write_generated(tmp_path, {"ppe": {"a": 1}})
    sc.write_generated(tmp_path, {"ppe": {"a": 2}})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ppe.yaml"]             # no temporary left behind
    assert yaml.safe_load((tmp_path / "ppe.yaml").read_text()) == {"a": 2}


# ── camera health is held while a role restarts, the hold ends when it is live; a stopped role leaves no status ──

import fakeredis  # noqa: E402


def test_apply_marks_the_roles_it_starts_or_restarts_so_camera_health_holds_them(tmp_path):
    args = _args_for(tmp_path, [{"id": "yard", "clip": "y.mp4", "features": ["ppe"]}])
    r = fakeredis.FakeStrictRedis(decode_responses=True)
    sc.apply(args, runner=_Runner(), http=None, redis_client=r)
    assert r.exists("iris:safety:applying:ppe") and 0 < r.ttl("iris:safety:applying:ppe") <= sc.APPLY_HOLD_S


def test_a_stopped_role_leaves_no_status_behind(tmp_path):
    args = _args_for(tmp_path, [{"id": "yard", "clip": "y.mp4", "features": ["ppe"]}])
    r = fakeredis.FakeStrictRedis(decode_responses=True)
    sc.apply(args, runner=_Runner(), http=None, redis_client=r)
    r.set("iris:status:yard@ppe", json.dumps({"running": True}))
    r.set("iris:status:yard@zones", json.dumps({"running": True}))
    args = _args_for(tmp_path, [{"id": "yard", "clip": "y.mp4", "features": []}])
    plan = sc.apply(args, runner=_Runner(running="safety-ppe\n"), http=None, redis_client=r)
    assert plan["stop"] == ["ppe"]
    assert not r.exists("iris:status:yard@ppe") and r.exists("iris:status:yard@zones")


class _Clock:
    def __init__(self, on_sleep=None):
        self.t, self.on_sleep = 0.0, on_sleep

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s
        if self.on_sleep:
            self.on_sleep()


def test_waiting_ends_when_every_restarted_stream_is_back_with_a_new_boot_and_releases_the_hold():
    r = fakeredis.FakeStrictRedis(decode_responses=True)
    r.set("iris:status:yard@ppe", json.dumps({"running": True, "boot_id": "old"}))
    r.set("iris:safety:applying:ppe", "1", ex=300)
    sources = {"ppe": [{"id": "yard"}]}
    clock = _Clock(on_sleep=lambda: r.set("iris:status:yard@ppe", json.dumps({"running": True, "boot_id": "new"})))
    late = sc.wait_until_live(r, sources, ["ppe"], {"yard@ppe": "old"}, timeout=60, now=clock.now, sleep=clock.sleep)
    assert late == set() and not r.exists("iris:safety:applying:ppe")


def test_a_role_that_does_not_come_back_keeps_its_hold_until_it_expires():
    r = fakeredis.FakeStrictRedis(decode_responses=True)
    r.set("iris:safety:applying:ppe", "1", ex=300)
    clock = _Clock()
    late = sc.wait_until_live(r, {"ppe": [{"id": "yard"}]}, ["ppe"], {}, timeout=10, now=clock.now, sleep=clock.sleep)
    assert late == {"ppe"} and r.exists("iris:safety:applying:ppe")


def test_models_lists_the_pointers_of_the_models_the_stack_loads(capsys):
    assert sc.main(["models", "--data", "unused"]) == 0
    pointers = capsys.readouterr().out.split()
    assert "perception/models/DeepStream-Yolo/yolo11x.onnx.dvc" in pointers
    assert "perception/models/DeepStream-Yolo-Pose/yolo26x-pose.onnx.dvc" in pointers
    assert "products/fall_detection/weights/pyskl_j.pth.dvc" in pointers        # a listed folder: each of its files
    assert all((ROOT / p).is_file() for p in pointers)                          # every one is a real DVC pointer


def test_reporting_what_is_pending_writes_nothing_the_app_reads(tmp_path, capsys):
    """`just safety-status` lists what each role runs and what is still to apply. The app reads the generated files as
    the running config (camera health expects their streams), so a report must not write toggles nobody applied."""
    args = _args_for(tmp_path, [{"id": "yard", "clip": "y.mp4", "features": ["ppe"]}])
    sc.apply(args, runner=_Runner(), http=None)
    generated = tmp_path / "data" / "safety" / "generated"
    before = {p.name: p.read_text() for p in generated.iterdir()}
    cams = tmp_path / "cameras.yaml"
    cams.write_text(yaml.safe_dump({"cameras": [{"id": "yard", "clip": "y.mp4", "features": ["ppe", "fire"]}]}))
    assert sc.main(["render", "--dry-run", "--data", str(tmp_path / "data"), "--cameras", str(cams)]) == 0
    assert "fire       yard  (not applied yet)" in capsys.readouterr().out
    assert {p.name: p.read_text() for p in generated.iterdir()} == before


def test_applying_by_hand_waits_for_the_roles_and_so_releases_camera_health_s_hold(monkeypatch, capsys):
    """`just safety-up` / `just safety-apply` apply once by hand: they hold camera health for the roles they start, so
    they must also release it once those roles are live -- not leave health blind until the hold expires."""
    calls = []
    monkeypatch.setattr(sc, "apply", lambda args, **kw: calls.append(kw) or {"restart": [], "start": [], "stop": []})
    assert sc.main(["apply", "--data", "unused"]) == 0
    assert calls == [{"wait": True}]
