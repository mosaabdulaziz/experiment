"""Unit tests for `concealment/zones.py`."""
from __future__ import annotations

import math
import pytest

from concealment.types import CameraZones, ZonePolygon
from concealment.zones import (
    find_containing_zone,
    ground_anchor,
    load_zones,
    point_in_polygon,
    propose_zones_from_dwell,
    reach_alignment,
)
from conftest import make_frame, make_skeleton_obs


def test_point_in_polygon_interior_exterior_and_boundary():
    poly = ((0.1, 0.2), (0.9, 0.2), (0.9, 0.8), (0.1, 0.8))
    assert point_in_polygon((0.5, 0.5), poly) is True
    assert point_in_polygon((0.1, 0.2), poly) is True      # vertex
    assert point_in_polygon((0.5, 0.2), poly) is True      # top edge
    assert point_in_polygon((0.05, 0.5), poly) is False
    assert point_in_polygon((0.5, 0.95), poly) is False


def test_ground_anchor_uses_confident_ankles_then_hips_then_box():
    shape = (1920, 1080)
    # 1. Confident ankles at y=780, center_x=960 -> (0.5, 780/1080)
    obs = make_skeleton_obs(center_x=960.0, ankle_y=780.0, ankle_conf=0.9)
    gx, gy = ground_anchor(obs, shape, conf_threshold=0.30)
    assert math.isclose(gx, 0.5, abs_tol=1e-3)
    assert math.isclose(gy, 780.0 / 1080.0, abs_tol=1e-3)

    # 2. Low-confidence ankles -> projects downward from confident hips (hip_y=540, shoulder_y=400 -> torso=140)
    obs_no_ank = make_skeleton_obs(center_x=960.0, shoulder_y=400.0, hip_y=540.0, ankle_conf=0.05)
    gx2, gy2 = ground_anchor(obs_no_ank, shape, conf_threshold=0.30)
    assert math.isclose(gx2, 0.5, abs_tol=1e-3)
    assert gy2 > 540.0 / 1080.0

    # 3. Low-confidence everywhere -> falls back to bbox bottom-center
    obs_box_only = make_skeleton_obs(
        center_x=480.0,
        conf=0.05,
        ankle_conf=0.05,
        box=(400.0, 200.0, 560.0, 810.0),
    )
    gx3, gy3 = ground_anchor(obs_box_only, shape, conf_threshold=0.30)
    assert math.isclose(gx3, 480.0 / 1920.0, abs_tol=1e-3)
    assert math.isclose(gy3, 810.0 / 1080.0, abs_tol=1e-3)


def test_find_containing_zone_and_zone_free_fallback():
    z1 = ZonePolygon(id="shelf-a", points=((0.1, 0.2), (0.5, 0.2), (0.5, 0.8), (0.1, 0.8)))
    cz = CameraZones(camera="cam-1", zone_free=False, zones=(z1,))
    assert find_containing_zone((0.3, 0.5), cz) == z1
    assert find_containing_zone((0.8, 0.5), cz) is None

    cz_free = CameraZones(camera="cam-1", zone_free=True, zones=())
    fallback = find_containing_zone((0.8, 0.5), cz_free)
    assert fallback is not None
    assert fallback.id == "zone_free"


def test_reach_alignment_with_and_without_facing():
    zone_up = ZonePolygon(
        id="shelf-a",
        points=((0.1, 0.2), (0.9, 0.2), (0.9, 0.8), (0.1, 0.8)),
        facing=(0.0, -1.0),
    )
    assert math.isclose(reach_alignment((0.0, -25.0), zone_up), 1.0, abs_tol=1e-5)
    assert math.isclose(reach_alignment((0.0, 25.0), zone_up), -1.0, abs_tol=1e-5)
    assert math.isclose(reach_alignment((25.0, 0.0), zone_up), 0.0, abs_tol=1e-5)
    assert math.isclose(reach_alignment((0.0, 0.0), zone_up), 0.0, abs_tol=1e-5)

    zone_none = ZonePolygon(id="shelf-b", points=((0.1, 0.2), (0.9, 0.2), (0.9, 0.8)))
    assert math.isclose(reach_alignment((12.0, 34.0), zone_none), 1.0, abs_tol=1e-5)
    assert math.isclose(reach_alignment((12.0, 34.0), None), 1.0, abs_tol=1e-5)


def test_load_zones_validates_and_normalizes():
    raw = {
        "cameras": {
            "cam-1": {
                "zone_free": False,
                "zones": [
                    {
                        "id": "shelf-1",
                        "facing": [3.0, 4.0],
                        "points": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]],
                    }
                ],
            }
        }
    }
    loaded = load_zones(raw)
    assert "cam-1" in loaded
    z = loaded["cam-1"].zones[0]
    assert z.id == "shelf-1"
    assert math.isclose(z.facing[0], 0.6, abs_tol=1e-5)
    assert math.isclose(z.facing[1], 0.8, abs_tol=1e-5)

    # Fewer than 3 points -> ValueError
    with pytest.raises(ValueError):
        load_zones({"cameras": {"cam-1": {"zones": [{"id": "bad", "points": [[0.1, 0.1], [0.2, 0.2]]}]}}})

    # Out of [0, 1] range -> ValueError
    with pytest.raises(ValueError):
        load_zones(
            {
                "cameras": {
                    "cam-1": {
                        "zones": [{"id": "bad", "points": [[0.1, 0.1], [1.2, 0.1], [0.5, 0.5]]}]
                    }
                }
            }
        )

    # Zero-length facing vector -> ValueError
    with pytest.raises(ValueError):
        load_zones(
            {
                "cameras": {
                    "cam-1": {
                        "zones": [
                            {
                                "id": "bad",
                                "facing": [0.0, 0.0],
                                "points": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]],
                            }
                        ]
                    }
                }
            }
        )


def test_propose_zones_from_dwell_finds_stationary_region():
    # Track 1 stands still at (960, 780) -> normalized (0.50, 0.722) for 4.0 seconds
    frames = []
    for i in range(41):
        pts = i * 0.1
        obs = make_skeleton_obs(track_id=1, center_x=960.0, ankle_y=780.0)
        frames.append(make_frame(pts, [obs], camera="cam-1", frame_num=i))
    proposed = propose_zones_from_dwell(frames, grid_size=10, min_dwell_s=2.0, top_k=1)
    assert proposed.camera == "cam-1"
    assert len(proposed.zones) == 1
    assert point_in_polygon((0.50, 780.0 / 1080.0), proposed.zones[0].points) is True
