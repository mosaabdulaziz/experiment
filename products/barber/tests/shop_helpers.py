"""Scaffolding for the engine tests: a 1000x1000 frame, boxes whose zone
anchor (80% down the box) lands on a given point, times from T0, and a small
shop -- centre camera with door/sofa/chair 1/wash, left camera owning chair 1
with its seat."""
from datetime import datetime, timedelta

from domain import Appearance, Frame, TrackObs
from policy import effective
from tracklets import unit
from zones import FrameGeometry

T0 = datetime(2026, 9, 24, 10, 0, 0)
GEO = FrameGeometry(1000, 1000)
POLICY = effective({
    "track_gone_s": 2, "still_window_s": 3, "still_radius": 0.02, "seat_dwell_s": 10, "seat_jitter_s": 2,
    "chair_gap_s": 20, "min_service_s": 60, "wash_return_s": 300, "min_visit_s": 3, "waiting_debounce_s": 5,
    "exit_confirm_s": 20, "lost_timeout_s": 60, "sync_window_s": 3, "reid_veto": 0.3, "reid_margin": 0.05,
    "reid_match_min": 0.7, "reattach_s": 120, "staff_min_samples": 3, "staff_threshold": 0.6,
    "line_band": 0.015, "door_band": 0.08, "wait_target_s": 100, "service_target_s": 400,
})


def sq(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


ZONES = {
    "barber-c": [
        {"id": "door", "name": "door", "type": "entrance_line", "points": [[0.4, 0.1], [0.6, 0.1]],
         "inside": [0.5, 0.5]},
        {"id": "sofa", "name": "sofa", "type": "waiting", "points": sq(0.3, 0.3, 0.7, 0.6)},
        {"id": "c-ch1", "name": "chair 1 (centre)", "type": "chair", "points": sq(0.0, 0.3, 0.2, 0.5), "pair": "ch1"},
        {"id": "wash", "name": "wash", "type": "washing", "points": sq(0.3, 0.8, 0.7, 1.0)},
    ],
    "barber-l": [
        {"id": "l-ch1", "name": "chair 1", "type": "chair", "points": sq(0.4, 0.4, 0.8, 0.9), "pair": "ch1",
         "owner": True},
        {"id": "l-s1", "name": "seat 1", "type": "seat", "points": sq(0.5, 0.5, 0.7, 0.8), "seat_of": "l-ch1"},
    ],
}
# centre camera spots (zone anchors); the door line is y=0.1, the feet sit 0.03 below the anchor
C_STREET, C_DOOR, C_SOFA, C_CHAIR, C_AISLE, C_WASH = (0.5, 0.0), (0.5, 0.2), (0.5, 0.45), (0.1, 0.4), (0.25, 0.7), (0.5, 0.9)
# left camera spots
L_AISLE, L_EDGE, L_SEAT = (0.2, 0.2), (0.45, 0.45), (0.6, 0.65)


def ts(s: float) -> float:
    return T0.timestamp() + s


def at(s: float) -> datetime:
    return T0 + timedelta(seconds=s)


def box_at(pt):
    return (pt[0] * 1000 - 50, pt[1] * 1000 - 120, 100.0, 150.0)


def frame(cam: str, s: float, tracks: dict) -> Frame:
    return Frame(cam, ts(s), tuple(TrackObs(tid, box_at(p)) for tid, p in tracks.items()))


def staff_app(cam: str, tid: int, s: float, p: float) -> Appearance:
    return Appearance(cam, tid, ts(s), None, p)


def reid_app(cam: str, tid: int, s: float, vec) -> Appearance:
    return Appearance(cam, tid, ts(s), unit(vec), None)
