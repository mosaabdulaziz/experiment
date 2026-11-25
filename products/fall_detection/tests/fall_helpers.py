"""Shared stand-ins for the fall product's tests: FakeRedis, FakeStorage, candidate/policy factories.

The candidate dicts mirror the vendor's packaging exactly (fall_monitor.py:1007-1024) plus the
seam's two keys (design §4.2): `reading` and `fall_paths`. `path_scores` is the product's own
(monitor.attach_path_scores) -- the factory takes it as a parameter because some tests start from
a candidate that already crossed that step.
"""
import copy

DECISIVE_FALL = [0.005, 0.005, 0.005, 0.02, 0.005, 0.005, 0.95, 0.005]
AMBIGUOUS_FALL_LYING = [0.01, 0.01, 0.01, 0.42, 0.01, 0.01, 0.45, 0.07]
DECISIVE_LYING = [0.005, 0.005, 0.01, 0.95, 0.005, 0.005, 0.015, 0.005]
DECISIVE_UPRIGHT = [0.90, 0.02, 0.02, 0.01, 0.02, 0.02, 0.005, 0.005]

POLICY = {
    "scoring": {
        "confirm_threshold": 0.70,
        "margin_span": 0.80,
        "trust": {"action": 1.00, "kinematic": 0.50, "cold": 0.40},
        "evaluations": {
            "falling": [6],
            "lying": [3],
            "immobile": [3, 2],
            "unstable": [7],
            "upright": [0, 1, 2, 4, 5],
        },
    },
    "suppress_after_fall_s": 10.0,
}

VENDOR_CFG = {
    "path_b": {"drop": 0.14, "vdown": 0.16, "low_hip": 0.35, "angle": 40.0, "aspect": 0.90},
    "path_c": {"drop": 0.14, "vdown": 0.16, "low_hip": 0.32, "angle": 45.0, "aspect": 1.05},
    "lying": {"angle_in": 35.0, "aspect_in": 0.50, "max_motion": 1.0},
    "immobility": {"disp_thresh": 0.035},
}

KINEMATIC_AT_EMISSION = {
    "drop": 0.01, "vdown": 0.02, "hip_rel": 0.15, "angle": 75.0, "aspect": 1.6,
    "knee_angle": 0.0, "disp_win": 0.01, "motion": 0.2,
}


def make_rec(kind="fall", track=1, t_start=100.0, t_alert=103.0, probs=None,
             action=None, kinematic=None, cold=None, kinematics=None, event_id=None):
    top = max(range(len(probs)), key=lambda i: probs[i]) if probs else None
    return {
        "type": kind,
        "track": track,
        "t_start": t_start,
        "t_candidate": t_start + 2.0,
        "t_alert": t_alert,
        "priority": "high" if kind == "fall" else "normal",
        "event_id": event_id or f"ev-{kind}-{track}-{t_start}",
        "status": "candidate",
        "reading": None if probs is None else {"probs": list(probs), "top": top},
        "path_scores": {"action": action, "kinematic": kinematic, "cold": cold},
        "evidence": {
            "clip_window": [t_start - 2.0, t_alert + 2.0],
            "tsstg_probs": {},
            "ntu60_top5": [],
            "kinematics": dict(kinematics or KINEMATIC_AT_EMISSION),
            "snapshot_path": "",
        },
    }


def detailed(policy=None):
    """The policy with `alert_details` on (the safety stack's alert: a sentence, the person and the fields)."""
    return {**copy.deepcopy(policy or POLICY), "alert_details": True}


def policy_with(**scoring_overrides):
    p = copy.deepcopy(POLICY)
    for k, v in scoring_overrides.items():
        if isinstance(v, dict):
            p["scoring"][k].update(v)
        else:
            p["scoring"][k] = v
    return p


class FakeRedis:
    def __init__(self):
        self.added = []          # (key, fields) in write order

    def xadd(self, key, fields, *a, **k):
        self.added.append((key, dict(fields)))
        return f"{len(self.added)}-0"

    def entries(self, key):
        return [f for k, f in self.added if k == key]


class FakeStorage:
    def __init__(self):
        self.rows = []
        self.events = []

    def add_review(self, row):
        self.rows.append(dict(row))
        return len(self.rows) - 1

    def set_review_status(self, row_id, status, note=""):
        self.rows[row_id].update(status=status, note=note)

    def add_event(self, row):
        self.events.append(dict(row))
        return len(self.events)          # the real add_event returns SQLite's lastrowid (1-based)
