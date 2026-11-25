import importlib.util
import pathlib
import sys
import pytest

pytest.importorskip("prometheus_client")


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "monitoring/exporter"))   # `import events`
_spec = importlib.util.spec_from_file_location(
    "iris_exporter", pathlib.Path(__file__).resolve().parents[1] / "monitoring/exporter/exporter.py")
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)


def test_throttle_decode_flags_only_the_set_reasons():
    d = ex.decode_throttle(0x4 | 0x40)
    assert d["sw_power_cap"] == 1 and d["hw_thermal"] == 1 and d["hw_slowdown"] == 0
    assert ex.decode_throttle(0) == {k: 0 for k in ex.THROTTLE_BITS}


def test_stream_age_from_the_last_id():
    now = 1_000_000.0
    assert ex.stream_entry_age("999990000-3", now) == 10.0
    assert ex.stream_entry_age(b"999999500-0", now) == 0.5
    assert ex.stream_entry_age(None) is None and ex.stream_entry_age("0-0", now) == now
    assert ex.stream_entry_age("garbage") is None


def test_go2rtc_parse_pads_names_and_tells_live_inputs_from_idle_ones():
    streams = {
        "cam-1": {"producers": [{"url": "rtsp://127.0.0.1:8554/raw-cam-1"}], "consumers": None},
        "raw-cam-1": {"producers": [{"remote_addr": "1.2.3.4:5", "bytes_recv": 900}],
                      "consumers": [{"id": 1}, {"id": 2}]},
        "raw-cam-12": {"producers": [{"url": "x"}], "consumers": None},   # nobody pushing
        "raw-cam-3": {"producers": None, "consumers": None},
    }
    out = ex.parse_go2rtc(streams)
    assert set(out) == {"cam-01", "cam-12", "cam-03"}
    assert out["cam-01"] == {"alive": True, "consumers": 2, "bytes": 900, "sent": 0}
    assert out["cam-12"]["alive"] is False and out["cam-03"]["alive"] is False


def test_player_cams_counts_ffmpeg_pushers_only():
    cmds = ["ffmpeg -re -i a.mp4 -f rtsp rtsp://127.0.0.1:38554/raw-cam-6",
            "ffmpeg -re -i b.mp4 -f rtsp rtsp://127.0.0.1:38554/raw-cam-14",
            "python3 scale13_streamer.py raw-cam-9"]
    assert ex.player_cams(cmds) == {"cam-06": 1, "cam-14": 1}


def test_cpu_and_memory_from_docker_stats():
    s = {"cpu_stats": {"cpu_usage": {"total_usage": 300}, "system_cpu_usage": 2000, "online_cpus": 4},
         "precpu_stats": {"cpu_usage": {"total_usage": 100}, "system_cpu_usage": 1000},
         "memory_stats": {"usage": 1000, "limit": 4000, "stats": {"inactive_file": 200}}}
    assert round(ex.cpu_percent(s), 1) == 80.0
    assert ex.mem_used(s) == (800, 4000)
    assert ex.cpu_percent({}) == 0.0


def test_go2rtc_parse_sums_what_it_sent_on_to_its_readers():
    streams = {"raw-cam-4": {"producers": [{"remote_addr": "x", "bytes_recv": 500}],
                             "consumers": [{"senders": [{"bytes": 300}, {"bytes": 20}]}, {"senders": [{"bytes": 7}]}, {}]}}
    assert ex.parse_go2rtc(streams)["cam-04"]["sent"] == 327
