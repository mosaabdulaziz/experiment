from datetime import datetime, timedelta

from footage_clock import footage_clock

FOOT = datetime(2026, 10, 16, 9, 29, 58)                # a recording of 16 Oct from 09:29:58
S = datetime(2026, 10, 18, 14, 53, 12)                  # the streamer started it then (device clock)
LEN = 34200.0                                           # 9.5 h


def run(lap=0, lag=0.0):
    return {"ph-1": [S.timestamp(), LEN, lap], "ph-3": [S.timestamp() + lag, LEN, lap]}


def test_device_time_maps_to_the_footage_from_the_laps_start():
    """2026-10-18 (user): a run playing 16 Oct 09:30-19:00 reports that date and those times, not the device's."""
    c = footage_clock(FOOT, run(lag=0.4))
    assert c.footage(S + timedelta(minutes=30)) == datetime(2026, 10, 16, 9, 59, 58)
    assert c.device(datetime(2026, 10, 16, 9, 59, 58)) == S + timedelta(minutes=30)
    assert (c.start, c.end) == (FOOT, datetime(2026, 10, 16, 18, 59, 58))


def test_the_lap_is_the_one_the_consumer_saw():
    # looped once (the consumer cleared the rows and moved to lap 1): footage time from that lap's start
    c = footage_clock(FOOT, run(lap=1))
    assert c.footage(S + timedelta(seconds=LEN + 300)) == FOOT + timedelta(seconds=300)
    # ended without looping: lap 0 still, so its times stay right after the run (no wall-clock lap counting)
    assert footage_clock(FOOT, run(lap=0)).footage(S + timedelta(hours=9)) == FOOT + timedelta(hours=9)


def test_no_footage_start_or_no_clips_keep_the_device_clock():
    assert footage_clock(None, run()) is None
    assert footage_clock(FOOT, {}) is None and footage_clock(FOOT, None) is None
