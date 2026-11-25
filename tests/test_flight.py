import logging
import sys
import pathlib
import threading
import time

from perception.engine import flight


def test_bus_messages_are_bucketed_and_mapped_to_cameras():
    assert flight.classify_bus("No data from source since last 10 sec. Trying reconnection (1)") == "nodata"
    assert flight.classify_bus("Could not write to resource. (10)") == "writefail"
    assert flight.classify_bus("something", True) == "error" and flight.classify_bus("something") == "warn"
    assert flight.cam_of("src_4", {4: "cam-05"}) == "cam-05" and flight.cam_of("src", {}) == "other"
    assert flight.cam_of("src_9", {}) == "src_9"


def test_a_silent_main_loop_leaves_every_threads_stack_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(flight, "_TICK", 0.05)
    monkeypatch.setattr(flight.shutil, "which", lambda _x: None)   # never attach gdb to the test run
    rec = flight.FlightRecorder(str(tmp_path), logging.getLogger("t"), stall_s=0.2,
                                describe=lambda: ["cam-01: last frame never"])
    gate = threading.Event()
    threading.Thread(target=gate.wait, name="blocked-worker", daemon=True).start()
    rec.start()
    for _ in range(60):                     # the "main loop" never beats
        files = list(tmp_path.glob("stall-*.txt"))
        if files and "blocked-worker" in files[0].read_text():
            break
        time.sleep(0.1)
    gate.set()
    files = list(tmp_path.glob("stall-*.txt"))
    assert files and rec.stalls >= 1
    text = files[0].read_text()
    assert "## sources" in text and "cam-01: last frame never" in text
    assert "MAIN" in text and "blocked-worker" in text


def test_beating_main_loop_is_not_a_stall(tmp_path, monkeypatch):
    monkeypatch.setattr(flight, "_TICK", 0.05)
    rec = flight.FlightRecorder(str(tmp_path), logging.getLogger("t"), stall_s=0.3)
    rec.start()
    end = time.monotonic() + 0.8
    while time.monotonic() < end:
        rec.beat()
        time.sleep(0.05)
    assert rec.stalls == 0 and not list(tmp_path.glob("stall-*.txt"))
    rec.note("cam-01", "nodata"); rec.note("cam-01", "nodata")
    assert rec.count("cam-01", "nodata") == 2 and rec.count("cam-02", "nodata") == 0


class _Log:
    def __init__(self):
        self.lines = []

    def warning(self, fmt, *a):
        self.lines.append(fmt % a)

    error = info = warning


def test_a_counter_that_keeps_rising_is_reported_once_a_minute(tmp_path):
    log, total = _Log(), [0]
    rec = flight.FlightRecorder(str(tmp_path), log)
    rec.watch_drops("frame publisher", lambda: total[0], 0.5, "cannot keep up")
    t = 1000.0
    for i in range(12):                       # 3 drops/s for 24 s
        total[0] += 6
        rec._check_pressure(t, 2.0)
        t += 2.0
    assert len([l for l in log.lines if "[backpressure] frame publisher" in l]) == 1
    assert "3.0 dropped/s" in log.lines[0] and "cannot keep up" in log.lines[0]
    assert rec.pressure_events == 1
    for i in range(25):                       # still rising: not repeated within the minute, then again after it
        total[0] += 6
        rec._check_pressure(t, 2.0)
        t += 2.0
    assert len(log.lines) == 2


def test_a_slow_counter_or_a_short_burst_is_not_reported(tmp_path):
    log, total = _Log(), [0]
    rec = flight.FlightRecorder(str(tmp_path), log)
    rec.watch_drops("qos", lambda: total[0], 0.5, "x")
    t = 0.0
    for i in range(30):
        total[0] += 1 if i % 10 == 0 else 0   # a single drop now and then
        rec._check_pressure(t, 2.0)
        t += 2.0
    for i in range(3):                        # a short burst
        total[0] += 20
        rec._check_pressure(t, 2.0)
        t += 2.0
    assert log.lines == []


def test_a_queue_pinned_at_its_cap_is_reported_after_the_grace_period(tmp_path):
    log, depth = _Log(), [8]
    rec = flight.FlightRecorder(str(tmp_path), log)
    rec.watch_depth("cam-01 publisher queue", lambda: depth[0], 8, "not draining", secs=30.0)
    for t in range(1000, 1029, 2):
        rec._check_pressure(float(t), 2.0)
    assert log.lines == []
    rec._check_pressure(1032.0, 2.0)
    assert len(log.lines) == 1 and "8 of 8" in log.lines[0]
    depth[0] = 1                              # drained: the clock restarts
    rec._check_pressure(1040.0, 2.0)
    assert rec._depth_watch[0]["since"] is None


def test_qos_drops_are_summed_per_element_and_only_grow(tmp_path):
    rec = flight.FlightRecorder(str(tmp_path), _Log())
    rec.note_qos("sink0", 5); rec.note_qos("sink1", 3); rec.note_qos("sink0", 2)
    assert rec.qos_dropped() == 8
    rec.note_qos("sink0", 9)
    assert rec.qos_dropped() == 12


def test_events_go_to_one_shared_log_file(tmp_path):
    log, total = _Log(), [0]
    rec = flight.FlightRecorder(str(tmp_path), log)
    rec.watch_drops("frame publisher", lambda: total[0], 0.5, "cannot keep up")
    t = 1000.0
    for _ in range(8):
        total[0] += 6
        rec._check_pressure(t, 2.0)
        t += 2.0
    lines = (tmp_path / "events.log").read_text().splitlines()
    assert len(lines) == 1 and "[event] backpressure: frame publisher: 3.0 dropped/s" in lines[0]
    assert lines[0].endswith("cannot keep up") and lines[0][4] == "-" and lines[0].split()[0].endswith("Z")


def test_a_camera_silent_for_four_seconds_opens_one_episode_with_evidence_and_closes_when_frames_return(tmp_path):
    log = _Log()
    rec = flight.FlightRecorder(str(tmp_path), log)
    ages = {"cam-01": 0.1, "cam-14": 0.2, "cam-15": None}
    rec.watch_sources(lambda: dict(ages), lambda cam: f"mux queue 2/2 for {cam}")
    rec._check_sources(100.0)
    assert log.lines == []
    ages["cam-14"] = 4.5
    rec._check_sources(102.0)
    rec._check_sources(104.0)                              # still silent: no second START
    starts = [l for l in log.lines if "START" in l]
    assert len(starts) == 1 and "cam-14: no frame for 4 s" in starts[0] and "mux queue 2/2 for cam-14" in starts[0]
    assert rec.episodes("cam-14") == 1 and rec.open_episodes() == 1
    ages["cam-14"] = 0.1
    rec._check_sources(110.0)
    assert any("END cam-14" in l for l in log.lines) and rec.open_episodes() == 0
    events = (tmp_path / "events.log").read_text()
    assert "STALL START cam-14" in events and "STALL END cam-14" in events
    ages["cam-14"] = 9.0                                   # a second episode counts again
    rec._check_sources(120.0)
    assert rec.episodes("cam-14") == 2


def test_blocked_evidence_gathering_never_blocks_the_recorder(tmp_path):
    import threading as th
    log = _Log()
    rec = flight.FlightRecorder(str(tmp_path), log)
    block = th.Event()
    rec.watch_sources(lambda: {"cam-02": 6.0}, lambda cam: block.wait(30))
    t0 = time.monotonic()
    rec._check_sources(50.0)
    block.set()
    assert time.monotonic() - t0 < 3.0 and "evidence unavailable" in log.lines[0]


def test_decoder_thread_census_tells_a_sleep_loop_from_normal_waiting():
    tasks = [("python3", "futex_do_wait", "S")] + [("dec_que:src", "futex_do_wait", "S")] * 13 + [
        ("dec_que:src", "hrtimer_nanosleep", "S"), ("queue:src", "futex_do_wait", "S")]
    c = flight.census_from(tasks)
    assert (c["decoders"], c["sleeping"]) == (14, 1) and "1 in a nanosleep loop" in c["text"]
    assert flight.census_from([])["decoders"] == 0
    assert flight.read_tasks("/nonexistent") == []
