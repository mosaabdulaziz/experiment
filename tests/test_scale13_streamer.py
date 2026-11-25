"""Timeline planning of the full-day player (scripts/scale13_streamer.py)."""
import datetime as dt
import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location("streamer", pathlib.Path(__file__).parent.parent / "scripts" / "scale13_streamer.py")
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)

T = lambda h, m=0, s=0: dt.datetime(2026, 9, 13, h, m, s)
H265 = ("hevc", 3200, 1800, 20.0)


def entry(t, name, dur, meta=H265):
    return (t, name, (dur, *meta))


def test_contiguous_files_have_no_fill():
    pieces = st.plan_timeline([entry(T(0), "a", 3600), entry(T(1), "b", 3600)], T(0))
    assert [p["kind"] for p in pieces] == ["file", "file"]
    assert pieces[0]["end"] == T(1)


def test_hole_is_filled_in_the_codec_of_the_file_before():
    pieces = st.plan_timeline([entry(T(0), "a", 119.5), entry(T(0, 5, 50), "b", 3600, ("h264", 2560, 1440, 20.0))], T(0))
    assert [p["kind"] for p in pieces] == ["file", "fill", "file"]
    fill = pieces[1]
    assert (fill["codec"], fill["w"], fill["h"]) == ("hevc", 3200, 1800)
    assert fill["start"] == T(0, 1, 59, ) + dt.timedelta(seconds=0.5) and fill["end"] == T(0, 5, 50)


def test_overlap_is_cut_where_the_next_file_starts():
    pieces = st.plan_timeline([entry(T(22, 40, 42), "a", 4759.5), entry(T(22, 59, 59), "b", 3421.9)], T(22))
    assert pieces[0]["end"] == T(22, 59, 59)
    assert pieces[1]["start"] == T(22, 59, 59)


def test_late_start_seeks_into_the_file_and_skips_finished_pieces():
    entries = [entry(T(0), "a", 3600), entry(T(1), "b", 3600)]
    pieces = st.plan_timeline(entries, T(1, 10))
    assert len(pieces) == 1 and pieces[0]["path"] == "b"
    assert pieces[0]["seek"] == 600 and pieces[0]["start"] == T(1, 10)


def test_start_inside_a_hole_begins_with_the_fill():
    entries = [entry(T(0), "a", 100), entry(T(0, 10), "b", 3600)]
    pieces = st.plan_timeline(entries, T(0, 3))
    assert pieces[0]["kind"] == "fill" and pieces[0]["start"] == T(0, 3)


def test_piece_commands():
    fill = {"kind": "fill", "start": T(0, 2), "end": T(0, 6), "path": None, "seek": 0.0, "codec": "hevc", "w": 3200, "h": 1800, "fps": 20.0}
    c = st.piece_cmd(fill, "h", 1, 3, T(0, 2))
    assert "libx265" in c and "color=c=black:s=3200x1800:r=20" in c and c[c.index("-t") + 1] == "240.000"
    f = {"kind": "file", "start": T(0), "end": T(1), "path": "x.mp4", "seek": 12.5, "codec": "h264", "w": 1, "h": 1, "fps": 20.0}
    c = st.piece_cmd(f, "h", 1, 3, T(0))
    assert c[c.index("-ss") + 1] == "12.500" and "copy" in c and c[-1] == "rtsp://h:1/raw-cam-3"
