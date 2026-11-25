"""perception/modules/zones/analytics.render_analytics_config: the ROI config each zones process renders for itself.
Written against the app's former renderer, numbered by {camera id: index}; the shim keeps that shape."""
from perception.modules.zones.analytics import render_analytics_config


def render_nvdsanalytics_config(zones, id_to_index, mux_width, mux_height):
    return render_analytics_config(zones, {i: sid for sid, i in id_to_index.items()}, mux_width, mux_height)

_HEADER = (
    "[property]\n"
    "enable=1\n"
    "config-width=1280\n"
    "config-height=720\n"
    "osd-mode=0\n"
)


def test_no_zones_renders_property_section_only():
    assert render_nvdsanalytics_config({}, {}, 1280, 720) == _HEADER


def test_one_zoned_stream_renders_single_non_inverted_group():
    zones = {"cam0": {"zones": [{"id": "z0", "name": "A",
                                 "points": [[100, 100], [300, 100], [300, 300]]}]}}
    out = render_nvdsanalytics_config(zones, {"cam0": 0}, 1280, 720)
    assert out == (
        _HEADER
        + "\n"
        "[roi-filtering-stream-0]\n"
        "enable=1\n"
        "roi-z0=100;100;300;100;300;300\n"
        "inverse-roi=0\n"
        "class-id=-1\n"
    )


def test_valid_zone_is_still_non_inverted():
    # Inversion for valid zones moved into the sidecar (Python), so the INI is always
    # non-inverted regardless of mode.
    zones = {"cam0": {"zones": [{"id": "z0", "name": "A",
                                 "points": [[1, 1], [2, 2], [3, 3]],
                                 "intrusion": {"mode": "valid", "classes": ["person"]}}]}}
    out = render_nvdsanalytics_config(zones, {"cam0": 0}, 1280, 720)
    assert "inverse-roi=0\n" in out
    assert "inverse-roi=1" not in out


def test_multiple_zones_on_one_stream_share_one_group():
    zones = {"cam0": {"zones": [
        {"id": "z0", "name": "A", "points": [[1, 1], [2, 2], [3, 3]],
         "intrusion": {"mode": "restricted", "classes": ["person"]}},
        {"id": "z1", "name": "B", "points": [[4, 4], [5, 5], [6, 6]],
         "intrusion": {"mode": "valid", "classes": ["person"]}},
    ]}}
    out = render_nvdsanalytics_config(zones, {"cam0": 0}, 1280, 720)
    assert out.count("[roi-filtering-stream-0]") == 1
    assert "roi-z0=1;1;2;2;3;3\n" in out
    assert "roi-z1=4;4;5;5;6;6\n" in out
    assert "inverse-roi=0\n" in out


def test_overcrowding_only_zone_still_gets_an_roi_but_no_overcrowding_section():
    # A zone with intrusion off but overcrowding on needs a named ROI so the sidecar can
    # read its membership; overcrowding counting is done in Python off roiStatus, so there
    # is no separate [overcrowding-stream-N] block anymore.
    zones = {"cam0": {"zones": [{"id": "z0", "name": "Lobby",
                                 "points": [[1, 1], [2, 2], [3, 3]],
                                 "intrusion": {"mode": "off", "classes": ["person"]},
                                 "overcrowding": {"enabled": True, "threshold": 3,
                                                  "classes": ["person"]}}]}}
    out = render_nvdsanalytics_config(zones, {"cam0": 0}, 1280, 720)
    assert "[roi-filtering-stream-0]" in out
    assert "roi-z0=1;1;2;2;3;3\n" in out
    assert "overcrowding-stream" not in out
    assert "object-threshold" not in out


def test_fully_off_zone_contributes_no_roi():
    zones = {"cam0": {"zones": [{"id": "z0", "name": "A",
                                 "points": [[1, 1], [2, 2], [3, 3]],
                                 "intrusion": {"mode": "off"},
                                 "overcrowding": {"enabled": False}}]}}
    out = render_nvdsanalytics_config(zones, {"cam0": 0}, 1280, 720)
    assert "roi-filtering-stream" not in out


def test_zone_with_no_active_features_but_a_second_active_zone_keeps_the_group():
    zones = {"cam0": {"zones": [
        {"id": "z0", "name": "Off", "points": [[1, 1], [2, 2], [3, 3]],
         "intrusion": {"mode": "off"}, "overcrowding": {"enabled": False}},
        {"id": "z1", "name": "On", "points": [[4, 4], [5, 5], [6, 6]],
         "intrusion": {"mode": "restricted", "classes": ["person"]}},
    ]}}
    out = render_nvdsanalytics_config(zones, {"cam0": 0}, 1280, 720)
    assert "roi-z0" not in out           # the off zone has no ROI
    assert "roi-z1=4;4;5;5;6;6\n" in out  # the active one keeps its id


def test_legacy_single_zone_entry_renders_with_z0_label():
    zones = {"cam0": {"points": [[100, 100], [300, 100], [300, 300]], "type": "valid"}}
    out = render_nvdsanalytics_config(zones, {"cam0": 0}, 1280, 720)
    assert "roi-z0=100;100;300;100;300;300\n" in out
    assert "inverse-roi=0\n" in out


def test_multiple_streams_ordered_by_numeric_index_not_dict_order():
    zones = {
        "cam0": {"zones": [{"id": "z0", "name": "A", "points": [[1, 1], [2, 2], [3, 3]]}]},
        "cam1": {"zones": [{"id": "z0", "name": "B", "points": [[4, 4], [5, 5], [6, 6]]}]},
    }
    id_to_index = {"cam0": 1, "cam1": 0}   # cam1 has the lower index despite coming second
    out = render_nvdsanalytics_config(zones, id_to_index, 640, 480)
    assert out.index("[roi-filtering-stream-0]") < out.index("[roi-filtering-stream-1]")
    stream0 = out.split("[roi-filtering-stream-0]")[1].split("[roi-filtering-stream-1]")[0]
    assert "roi-z0=4;4;5;5;6;6" in stream0   # cam1 -> stream-0


def test_zone_for_id_absent_from_id_to_index_is_skipped():
    zones = {"stale-cam": {"zones": [{"id": "z0", "name": "A",
                                      "points": [[1, 1], [2, 2], [3, 3]]}]}}
    out = render_nvdsanalytics_config(zones, {}, 1280, 720)
    assert "roi-filtering-stream" not in out


def test_source_without_a_saved_zone_is_omitted():
    out = render_nvdsanalytics_config({}, {"cam0": 0}, 1280, 720)
    assert "roi-filtering-stream" not in out
