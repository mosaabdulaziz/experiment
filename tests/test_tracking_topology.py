"""The app expands camera groups for its own checks; it must agree with the sidecar's matcher."""
import yaml

from app.tracking import TrackingService
from perception.modules.mtmc.core._topology import expand_camera_groups


def test_app_and_sidecar_expand_the_scale13_layout_identically():
    for name in ("scale13",):
        side = yaml.safe_load(open("products/multicam/sidecar.yaml"))["modules"]["mtmc"]["config"]
        app = yaml.safe_load(open(f"configs/presets/{name}/config.yaml"))["mtmc"]
        assert side["camera_groups"] == app["camera_groups"] and side["group_links"] == app["group_links"]
        topo, gaps, overlap = expand_camera_groups(side["camera_groups"], side["group_links"])
        assert TrackingService._load_topology(app) == topo


def test_the_real_layout_has_the_expected_shape():
    side = yaml.safe_load(open("products/multicam/sidecar.yaml"))["modules"]["mtmc"]["config"]
    topo, gaps, overlap = expand_camera_groups(side["camera_groups"], side["group_links"])
    assert len(topo) == 15
    assert topo["cam-01"] == {"cam-02", "cam-03"}                       # D41: its gate neighbour and D52
    assert topo["cam-03"] == {"cam-01", "cam-02", "cam-04", "cam-05", "cam-06", "cam-07", "cam-08", "cam-09"}
    assert gaps[("cam-03", "cam-06")] == 3.0 and gaps[("cam-10", "cam-12")] == 2.0
    assert ("cam-04", "cam-09") in overlap and ("cam-03", "cam-04") not in overlap
