import pytest

import feeds

BASE = "rtsp://u:p@10.0.0.9:554/Streaming/Channels"


def test_sources_put_each_channel_under_the_base_over_tcp():
    assert feeds.sources("raw-ph-1=101 raw-ph-2=1101", BASE) == [
        ("raw-ph-1", BASE + "/101#transport=tcp"), ("raw-ph-2", BASE + "/1101#transport=tcp")]


def test_apply_replaces_each_stream_then_checks_go2rtc_pulls_it():
    calls, state = [], {}

    def call(method, params):
        calls.append((method, dict(params)))
        if method == "DELETE":
            state.pop(params["src"], None)
        elif method == "PUT":
            state[params["name"]] = {"producers": [{"url": params["src"]}]}
        return state

    assert feeds.apply([("raw-ph-1", BASE + "/101#transport=tcp")], call) == []
    assert calls[:2] == [("DELETE", {"src": "raw-ph-1"}), ("PUT", {"name": "raw-ph-1", "src": BASE + "/101#transport=tcp"})]
    assert feeds.apply([("raw-ph-2", "x")], lambda m, p: {}) == ["raw-ph-2"]          # go2rtc did not take it


def test_pulled_names_the_clip_streams_go2rtc_still_pulls_from_a_camera():
    streams = {"raw-ph-1": {"producers": [{"url": BASE + "/101"}]},                    # left from a live run
               "raw-ph-2": {"producers": [{"remote_addr": "172.19.0.1:1"}]},           # the test streamer's push
               "raw-ph-3": None}
    assert feeds.pulled(streams, ["raw-ph-1", "raw-ph-2", "raw-ph-3", "raw-ph-4"]) == ["raw-ph-1"]


def test_live_without_the_base_in_the_environment_refuses(capsys):
    with pytest.raises(SystemExit) as e:
        feeds.main(["live", "--api", "http://x", "--feeds", "raw-ph-1=101", "--base-env", "PALM_HILLS_RTSP"], env={})
    assert e.value.code == 2 and "PALM_HILLS_RTSP" in capsys.readouterr().err
