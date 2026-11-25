"""CPU-side stage timers, flight lines, per-replay resources and the log backfill (parsers.py, metrics.py, collectors.py, backfill.py)."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from regress import backfill, collectors, metrics, parsers  # noqa: E402

BASE = 1794948000.0        # an epoch second; the lines below carry docker timestamps around it


def ts(offset):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(BASE + offset, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


def line(offset, msg):
    return f"{ts(offset)} 12:00:00 [INFO] iris.sidecar: {msg}"


LOG = "\n".join([
    line(1, "[telemetry] probe n=1 total=5.1ms busy=0.1% avg=5.09ms max=5.09ms thread=Dummy-3 window=5.2s"),        # before the window: boot
    line(10, "[telemetry] probe n=100 total=200.0ms busy=4.0% avg=2.00ms max=6.00ms thread=Dummy-3 window=5.0s"),
    line(15, "[telemetry] probe n=100 total=300.0ms busy=6.0% avg=3.00ms max=9.00ms thread=Dummy-3 window=5.0s"),
    line(10, "[telemetry] publish n=50 total=500.0ms busy=10.0% avg=10.00ms max=20.00ms thread=frame-publisher-cam-1 window=5.0s"),
    line(15, "[telemetry] publish.redis n=50 total=25.0ms busy=0.5% avg=0.50ms max=2.00ms thread=frame-publisher-cam-1 window=5.0s"),
    line(12, "[telemetry] gpu.infer.pgie n=100 total=1800.0ms busy=36.0% avg=18.00ms max=30.00ms thread=gpu window=5.0s"),
    line(12, "[telemetry] gauge fps_cam-1=20.0"),
    line(20, "[flight] cpu 65% in 109 threads: python 48%, Dummy 6% | main-loop lag 0.0s, sched delay 3ms"),
    line(50, "[flight] cpu 20% in 109 threads: Dummy 12% | main-loop lag 0.4s, sched delay 9ms"),
    line(80, "[flight] cpu 99% in 109 threads: python 90% | main-loop lag 2.0s, sched delay 90ms"),                  # after the window
    line(200, "[telemetry] probe n=100 total=900.0ms busy=18.0% avg=9.00ms max=19.00ms thread=Dummy-3 window=5.0s"),  # after the window
])


def test_cpu_stages_are_summed_over_the_window_and_leave_out_gpu_stages():
    s = parsers.parse_cpu_stages(LOG, BASE + 5, BASE + 65)
    assert set(s) == {"probe", "publish", "publish.redis"}
    assert s["probe"]["n"] == 200 and s["probe"]["avg_ms"] == 2.5 and s["probe"]["max_ms"] == 9.0
    assert s["probe"]["busy"] == round(100 * 500.0 / (60 * 1000), 2)        # 500 ms in a 60 s window
    assert s["publish"]["avg_ms"] == 10.0


def test_without_a_window_every_line_counts_and_the_longest_stage_sets_the_duration():
    s = parsers.parse_cpu_stages(LOG)
    assert s["probe"]["n"] == 1 + 100 + 100 + 100
    assert s["probe"]["busy"] is not None


def test_flight_lines_in_the_window():
    f = parsers.parse_flight(LOG, BASE + 5, BASE + 65)
    assert f == {"samples": 2, "cpu_pct_max": 65, "top_thread_pct_max": 48, "threads_max": 109, "main_loop_lag_max_s": 0.4, "sched_delay_max_ms": 9}
    assert parsers.parse_flight(LOG, BASE + 300, BASE + 400) == {}
    idle = line(1, "[flight] cpu 0% in 92 threads: - | main-loop lag 0.0s, sched delay 0ms")
    assert parsers.parse_flight(idle)["top_thread_pct_max"] == 0


def test_parse_runtime_carries_both_for_the_steady_window():
    log = "\n".join([line(0, "sidecar starting shard=0/1"), line(2, "[cam-1] 20.0 fps"), line(10, "[cam-1] 20.0 fps"), line(15, "[cam-1] 20.0 fps"),
                     line(11, "[telemetry] probe n=100 total=200.0ms busy=4.0% avg=2.00ms max=6.00ms thread=Dummy-3 window=5.0s"),
                     line(20, "[flight] cpu 30% in 80 threads: python 25% | main-loop lag 0.1s, sched delay 1ms"),
                     line(60, "EOS — pipeline reached end of stream")])
    rt = parsers.parse_runtime(log)
    assert rt["cpu_stages"]["probe"]["avg_ms"] == 2.0 and rt["flight"]["cpu_pct_max"] == 30


def result_with(cfg="att", **rt_extra):
    rt = {"boot_to_first_fps_s": 8.0, "engine": {"rebuilt": False}, "fps": {}, "drops": {}, "gpu_stages": {}, "gpu_counters": {}}
    rt.update(rt_extra)
    return {"correctness": {"golden": {}}, "runtime": {cfg: rt}}


def test_flatten_has_the_new_metric_ids_and_families():
    flat = metrics.flatten(result_with(cpu_stages={"probe.dets": {"avg_ms": 0.01, "busy": 0.2}}, flight={"cpu_pct_max": 65, "main_loop_lag_max_s": 0.4},
                                       resources={"cpu_cores": 0.6, "rss_anon_mib": 1200.0, "cgroup_peak_mib": 1500.0, "threads": 109, "fds": 80}))
    assert flat["runtime.att.cpu_stage.probe.dets.avg_ms"].value == 0.01 and flat["runtime.att.cpu_stage.probe.dets.busy"].direction == "lower"
    assert flat["runtime.att.flight_cpu_pct"].value == 65 and flat["runtime.att.main_loop_lag_s"].value == 0.4
    assert flat["runtime.att.res.rss_anon_mib"].value == 1200.0 and flat["runtime.att.res.threads"].unit == "threads"
    assert metrics.family("runtime.att.cpu_stage.probe.dets.avg_ms") == "runtime.*.cpu_stage.probe.dets.avg_ms"
    assert metrics.family("runtime.att.res.rss_anon_mib") == "runtime.*.res.rss_anon_mib"


def row(t, cpu_usec, anon, peak, threads=80, fds=70):
    return (BASE + t, {"cgroup": {"cpu_usec": cpu_usec, "anon": anon, "peak_bytes": peak}, "procs": {"main": {"threads": threads, "fds": fds}}})


def test_summarize_resources_over_the_window():
    rows = [row(0, 0, 100 * 2 ** 20, 200 * 2 ** 20), row(10, 6_000_000, 1200 * 2 ** 20, 1500 * 2 ** 20, threads=109), row(20, 12_000_000, 1100 * 2 ** 20, 1500 * 2 ** 20, fds=90)]
    r = collectors.summarize_resources(rows, BASE + 5, BASE + 25)
    assert r["cpu_cores"] == 0.6 and r["rss_anon_mib"] == 1200.0 and r["cgroup_peak_mib"] == 1500.0 and r["threads"] == 109 and r["fds"] == 90
    whole = collectors.summarize_resources(rows)
    assert whole["cpu_cores"] == 0.6 and whole["samples"] == 3
    assert collectors.summarize_resources(rows[:1]) == {}


class FakeSampler:
    def __init__(self):
        self.n = 0

    def _containers(self, now):
        self.n += 1
        return {"regress-deepstream-sidecar-1": {"cgroup": {"cpu_usec": self.n * 1_000_000, "anon": 2 ** 30, "peak_bytes": 2 ** 31}, "procs": {"main": {"threads": 90, "fds": 70}}},
                "regress-redis-1": {"cgroup": {"cpu_usec": 5}}}, {}


def test_the_resource_sampler_keeps_only_the_sidecar_and_survives_a_broken_read():
    rs = collectors.ResourceSampler(interval=0.01, sampler=FakeSampler())
    rs.start()
    time.sleep(0.15)
    out = rs.finish()
    assert out["samples"] >= 3 and out["rss_anon_mib"] == 1024.0 and out["threads"] == 90

    class Broken:
        def _containers(self, now):
            raise RuntimeError("docker is gone")
    rs2 = collectors.ResourceSampler(interval=0.01, sampler=Broken())
    rs2.start()
    time.sleep(0.05)
    assert rs2.finish() == {}


def test_backfill_adds_stages_to_every_config_with_a_window(tmp_path):
    res = {"runtime": {"att": {"steady_window": [BASE + 5, BASE + 65]}, "nowin": {"steady_window": None}}}
    new, n = backfill.backfill(res, LOG)
    assert n == 1 and new["runtime"]["att"]["cpu_stages"]["probe"]["n"] == 200 and new["runtime"]["att"]["flight"]["samples"] == 2
    assert "cpu_stages" not in new["runtime"]["nowin"] and "cpu_stages" not in res["runtime"]["att"]          # the input is untouched
    log = tmp_path / "side.log"
    log.write_text(LOG)
    run = tmp_path / "run-1.json"
    run.write_text(json.dumps(res))
    assert backfill.run(str(log), [str(run)], str(tmp_path / "out")) == 0
    assert json.load(open(tmp_path / "out" / "run-1.json"))["runtime"]["att"]["cpu_stages"]["publish"]["avg_ms"] == 10.0


def two_replays_log():
    return "\n".join([
        line(0, "sidecar starting shard=0/1"), line(3, "[cam-1] 20.0 fps"), line(8, "[cam-1] 20.0 fps"),
        line(5, "[telemetry] probe n=100 total=200.0ms busy=4.0% avg=2.00ms max=6.00ms thread=Dummy-3 window=5.0s"),
        line(10, "[telemetry] probe n=100 total=400.0ms busy=8.0% avg=4.00ms max=8.00ms thread=Dummy-3 window=5.0s"),
        line(100, "sidecar starting shard=0/1"), line(103, "[cam-1] 25.0 fps"),
        line(105, "[telemetry] probe n=100 total=600.0ms busy=12.0% avg=6.00ms max=9.00ms thread=Dummy-3 window=5.0s"),
        line(140, "[flight] cpu 5% in 80 threads: - | main-loop lag 0.0s, sched delay 0ms"),       # the idle tail after the last probe line
    ])


def test_a_config_without_a_recorded_window_gets_one_from_its_neighbours_and_the_log():
    res = {"runtime": {"a": {"steady_window": [BASE + 3, BASE + 12]}, "b": {"steady_window": None}, "c": {"steady_window": [BASE + 200, BASE + 260]}}}
    log = two_replays_log()
    new, n = backfill.backfill(res, log)
    assert n == 3                                                       # a (recorded), b (inferred) and c (recorded, but no lines fall in it)
    assert new["runtime"]["c"]["cpu_stages"] == {}
    assert new["runtime"]["b"]["backfill_window"] == [BASE + 103, BASE + 105]
    assert new["runtime"]["b"]["cpu_stages"]["probe"]["avg_ms"] == 6.0 and new["runtime"]["b"]["steady_window"] is None
    assert "backfill_window" not in new["runtime"]["a"] and new["runtime"]["a"]["cpu_stages"]["probe"]["n"] == 200


def test_no_window_can_be_inferred_when_the_log_has_no_replay_in_the_gap():
    res = {"runtime": {"a": {"steady_window": [BASE + 3, BASE + 12]}, "b": {"steady_window": None}, "c": {"steady_window": [BASE + 30, BASE + 60]}}}
    new, n = backfill.backfill(res, two_replays_log())
    assert "cpu_stages" not in new["runtime"]["b"]


def test_the_last_config_of_a_run_is_not_matched_to_a_replay_far_later_in_the_log():
    log = "\n".join([two_replays_log(),
                     line(5000, "sidecar starting shard=0/1"), line(5003, "[cam-1] 30.0 fps"),                      # an unrelated replay much later
                     line(5005, "[telemetry] probe n=100 total=900.0ms busy=18.0% avg=9.00ms max=19.00ms thread=Dummy-3 window=5.0s")])
    res = {"correctness": {"golden": {"b": {"wall_s": 60.0}}}, "runtime": {"a": {"steady_window": [BASE + 3, BASE + 12]}, "b": {"steady_window": None}}}
    new, _ = backfill.backfill(res, log)
    assert new["runtime"]["b"]["backfill_window"] == [BASE + 103, BASE + 105]                                      # the nearby replay, not the one at 5000 s
    far = {"correctness": {"golden": {"b": {"wall_s": 60.0}}}, "runtime": {"a": {"steady_window": [BASE + 3000, BASE + 3010]}, "b": {"steady_window": None}}}
    assert "backfill_window" not in backfill.backfill(far, two_replays_log())[0]["runtime"]["b"]               # nothing within reach: none, not a wrong one
