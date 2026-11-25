"""recorder_kind: which recorder a source gets, and why.

Smart Video Record is RTSP-only -- DS 9.0's nvurisrcbin states "Sources must be
of type source-type-rtsp" against both smart-record and smart-rec-container -- so
a file:// source has no SVR path to fall back to and must keep the encode branch."""

import pytest

from perception.engine.recorder_kind import recorder_kind


def test_rtsp_defaults_to_svr():
    assert recorder_kind("rtsp://cam.local/stream1") == "svr"


def test_file_defaults_to_encode_because_svr_is_rtsp_only():
    assert recorder_kind("file:///workspace/iris/data/uploads/clip.avi") == "encode"


def test_http_and_unknown_schemes_default_to_encode():
    # Anything that is not RTSP cannot use SVR; encode is the safe fallback.
    assert recorder_kind("http://host/video.mp4") == "encode"
    assert recorder_kind("/dev/video0") == "encode"


def test_scheme_match_is_case_insensitive():
    assert recorder_kind("RTSP://cam.local/stream1") == "svr"


def test_rtsp_lookalikes_do_not_match():
    # A path containing "rtsp" is not an rtsp:// URI.
    assert recorder_kind("file:///data/rtsp-archive/clip.mp4") == "encode"


def test_explicit_override_wins_over_the_scheme():
    assert recorder_kind("rtsp://cam.local/s", override="encode") == "encode"
    assert recorder_kind("file:///a.mp4", override="none") == "none"


def test_override_svr_on_a_non_rtsp_source_is_rejected():
    """SVR cannot work on a file source, so silently honouring the override would
    produce a stream that records nothing and reports no error."""
    with pytest.raises(ValueError, match="rtsp"):
        recorder_kind("file:///a.mp4", override="svr")


def test_unknown_override_is_rejected():
    with pytest.raises(ValueError, match="unknown recorder"):
        recorder_kind("rtsp://cam.local/s", override="magnetic-tape")


def test_auto_override_is_the_same_as_no_override():
    assert recorder_kind("rtsp://cam.local/s", override="auto") == "svr"
    assert recorder_kind("file:///a.mp4", override="auto") == "encode"
