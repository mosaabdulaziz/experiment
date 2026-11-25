"""The safety stack's files agree with each other: every feature of products/safety/safety.yaml has its role file and
a compose service that runs the config the controller generates for it (docker-compose.safety.yml)."""
import os
import pathlib
import subprocess
import time

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRODUCT = yaml.safe_load((ROOT / "products/safety/safety.yaml").read_text())


def _compose():
    class Loader(yaml.SafeLoader):
        pass
    # compose's merge tags (!override, !reset) are not plain YAML
    Loader.add_constructor("!override", lambda loader, node: loader.construct_sequence(node))
    return yaml.load((ROOT / "docker-compose.safety.yml").read_text(), Loader=Loader)


def test_every_feature_has_a_role_file():
    for feature in PRODUCT["features"]:
        assert (ROOT / PRODUCT["roles"][feature]).is_file(), feature


def test_every_feature_has_a_compose_service_running_its_generated_config():
    services = _compose()["services"]
    for feature in PRODUCT["features"]:
        service = services[f"safety-{feature}"]
        assert service["command"][-1] == f"data/safety/generated/{feature}.yaml"


def test_the_role_sidecars_share_one_mux_size_and_scoped_keys():
    common = PRODUCT["common"]
    assert common["role_scoped_keys"] is True
    assert (common["mux_width"], common["mux_height"]) == (common["capture_width"], common["capture_height"])


def test_every_role_stamps_frame_arrival_on_this_machine_s_clock():
    """Every safety camera comes through go2rtc's relay, which drops a camera's RTCP clock: rtcp_camera_clock off
    lets nvstreammux stamp each frame's arrival (attach-sys-ts), which places the alert clips' boxes without the
    inference delay (FrameView.arrived, app/alert_clips.py _maybe_overlay_svr)."""
    assert PRODUCT["common"]["rtcp_camera_clock"] is False


def test_a_camera_that_stays_dead_after_its_source_restarts_ends_its_role_sidecar_so_docker_restarts_it():
    """2026-11-10 live demo: the store camera's feed stopped and came back; the fire sidecar got its RTSP bytes again
    but no frame came out of the mux through every in-process source restart (barber saw the same, 2026-09-26). The
    stall watch then ends the sidecar (exit 71) and the role's restart policy brings it back."""
    assert PRODUCT["common"]["source_stall_max_failed_restarts"] > 0
    services = _compose()["services"]
    for feature in PRODUCT["features"]:
        assert services[f"safety-{feature}"].get("restart") == "unless-stopped", feature


def test_the_demo_cameras_only_name_known_features():
    cameras = yaml.safe_load((ROOT / "products/safety/cameras.yaml").read_text())["cameras"]
    assert cameras and all(set(c["features"]) <= set(PRODUCT["features"]) for c in cameras)
    assert any(len(c["features"]) >= 2 for c in cameras)        # the demo shows two features sharing one camera


def test_the_app_reads_every_role_s_generated_config():
    env = _compose()["services"]["iris"]["environment"]
    (value,) = [e.split("=", 1)[1] for e in env if e.startswith("SIDECAR_CONFIG=")]
    assert value.split(",") == [f"data/safety/generated/{f}.yaml" for f in PRODUCT["features"]]


def test_the_app_reaches_the_fall_service_and_reads_the_same_cameras_file():
    env = dict(e.split("=", 1) for e in _compose()["services"]["iris"]["environment"])
    assert env["SAFETY_FALL_URL"] == "http://host.docker.internal:${SAFETY_FALL_PORT}"
    assert env["SAFETY_CAMERAS"] == "${SAFETY_CAMERAS}"


# ── the host processes' pid files (scripts/safety_pids.sh): only the process that wrote one is ever signalled ─────────

def _bash(script):
    return subprocess.run(["bash", "-c", f'source "{ROOT / "scripts/safety_pids.sh"}"; {script}'],
                          capture_output=True, text=True, timeout=20)


def test_a_pid_file_is_trusted_only_while_it_names_the_process_group_that_wrote_it(tmp_path):
    pidf = tmp_path / "fall.pid"
    leader = subprocess.Popen(["setsid", "bash", "-c", 'echo $$ > "$1"; exec sleep 60', "_", str(pidf)])
    other = subprocess.Popen(["sleep", "60"])            # alive, but in this process's group: no group of its own
    try:
        for _ in range(100):
            if pidf.exists() and pidf.read_text().strip():
                break
            time.sleep(0.05)
        pid = pidf.read_text().strip()
        assert _bash(f'safety_pid "{pidf}"').stdout.strip() == pid
        # the same pid in a file older than the process (a pid reused after the writer ended, or a reboot): not ours
        stale = tmp_path / "stale.pid"
        stale.write_text(pid + "\n")
        os.utime(stale, (time.time() - 3600,) * 2)
        for junk in ("", "1", "0", "-1", "12abc", str(other.pid)):
            bad = tmp_path / "bad.pid"
            bad.write_text(junk)
            assert _bash(f'safety_pid "{bad}"').returncode != 0, junk
        out = _bash(f'safety_stop "{stale}"')
        assert out.returncode == 1 and "nothing stopped" in out.stderr and not stale.exists()
        assert leader.poll() is None and other.poll() is None          # neither was signalled
        assert _bash(f'safety_stop "{pidf}"').returncode == 0 and not pidf.exists()
        assert leader.wait(timeout=5) is not None                      # the group it led is stopped
        assert other.poll() is None
    finally:
        for proc in (leader, other):
            if proc.poll() is None:
                proc.kill()
                proc.wait()
