import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "iris_events", pathlib.Path(__file__).resolve().parents[1] / "monitoring/exporter/events.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

CTX = {"power_w": 71.0, "power_limit_w": 54.0, "sm_mhz": 1900, "sm_max_mhz": 3090, "temp_c": 66,
       "cpu_temp_c": 83.0, "ac_online": True}


def test_throttle_start_end_with_duration_and_the_worst_context():
    t = ev.ThrottleTracker(min_s=3.0)
    assert t.update(0.0, {"sw_power_cap": 0, "gpu_idle": 1}, CTX) == []          # idle is not trouble
    assert t.update(10.0, {"sw_power_cap": 1}, CTX) == []                         # not yet: could be a blip
    assert t.update(11.0, {"sw_power_cap": 1}, {**CTX, "power_limit_w": 40.0, "temp_c": 72}) == []
    start = t.update(13.0, {"sw_power_cap": 1}, CTX)
    assert [(e["kind"], e["reason"]) for e in start] == [("start", "sw_power_cap")] and start[0]["since"] == 10.0
    assert t.update(20.0, {"sw_power_cap": 1}, CTX) == []                         # announced once
    end = t.update(58.0, {"sw_power_cap": 0}, CTX)
    assert end[0]["kind"] == "end" and round(end[0]["duration_s"]) == 48
    assert end[0]["ctx"]["power_limit_w"] == 40.0 and end[0]["ctx"]["temp_c"] == 72
    line = ev.format_throttle(end[0])
    assert "THROTTLE END sw_power_cap after 48 s" in line and "limit 40 W" in line and "AC adapter online" in line
    assert "limit 40 W" in ev.format_throttle(start[0]) and "CPU 83 C" in ev.format_throttle(start[0])


def test_a_one_second_flicker_is_a_blip_not_a_logged_throttle():
    t = ev.ThrottleTracker(min_s=3.0)
    assert t.update(0.0, {"sw_power_cap": 1}, CTX) == []
    e = t.update(1.0, {"sw_power_cap": 0}, CTX)
    assert [(x["kind"], x["reason"]) for x in e] == [("blip", "sw_power_cap")]
    assert t.update(2.0, {"sw_power_cap": 0}, CTX) == []


def test_two_reasons_are_tracked_separately():
    t = ev.ThrottleTracker(min_s=0.0)
    e = t.update(0.0, {"sw_power_cap": 1, "hw_thermal": 1}, CTX)
    assert {x["reason"] for x in e} == {"sw_power_cap", "hw_thermal"}
    e = t.update(5.0, {"sw_power_cap": 1, "hw_thermal": 0}, CTX)
    assert [(x["kind"], x["reason"]) for x in e] == [("end", "hw_thermal")]


def test_alert_firing_then_resolved_with_how_long():
    a = {"state": "firing", "labels": {"alertname": "CameraSilent", "severity": "critical", "component": "camera",
                                      "cam": "cam-07", "summary": "no frames"}}
    pend = {"state": "pending", "labels": {"alertname": "X"}}
    state, events = ev.diff_alerts({}, [a, pend], 100.0)
    assert [e["kind"] for e in events] == ["firing"] and "CameraSilent" in ev.format_alert(events[0])
    assert "cam=cam-07" in ev.format_alert(events[0]) and "no frames" in ev.format_alert(events[0])
    state, events = ev.diff_alerts(state, [a], 130.0)
    assert events == []                                                    # still firing: nothing new
    state, events = ev.diff_alerts(state, [], 372.0)
    assert events[0]["kind"] == "resolved" and ev.format_alert(events[0]).endswith("after 272 s")
    assert state == {}


def test_event_line_format_and_context_readers(tmp_path):
    line = ev.event_line("gpu", "THROTTLE START", now=1791100863)
    assert line == "2026-10-04T08:01:03Z [event] gpu: THROTTLE START"
    ac = tmp_path / "AC0"; ac.mkdir(); (ac / "type").write_text("Mains\n"); (ac / "online").write_text("0\n")
    bat = tmp_path / "BAT0"; bat.mkdir(); (bat / "type").write_text("Battery\n")
    assert ev.read_ac_online(str(tmp_path)) is False
    (ac / "online").write_text("1\n")
    assert ev.read_ac_online(str(tmp_path)) is True
    assert ev.read_ac_online(str(tmp_path / "nope")) is None
    hw = tmp_path / "hwmon3"; hw.mkdir(); (hw / "name").write_text("k10temp\n"); (hw / "temp1_input").write_text("83125\n")
    assert ev.read_cpu_temp(str(tmp_path)) == 83.125
    path = tmp_path / "sub" / "events.log"
    ev.append_event(str(path), line); ev.append_event(str(path), "second")
    assert path.read_text().splitlines() == [line, "second"]
