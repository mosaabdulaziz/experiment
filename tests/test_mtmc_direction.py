"""The VLM tells entering from leaving at a camera that looks at vehicles from one side (D16 faces the
cab of vehicles that are leaving)."""
import numpy as np

from perception.modules.mtmc.core import MTMCConfig, MTMCManager
from perception.modules.mtmc.vlm_arbiter import _parse_direction

FRAME = np.full((720, 1280, 3), 90, np.uint8)


class DirVlm:
    """Scripted stand-in for VLMArbiter: compare jobs and direction jobs."""
    enabled = True
    min_confidence = 0.5
    max_wait_s = 12.0

    def __init__(self, orientation="front", confidence=0.95, same_gids=(0,), answer_direction=True):
        self.orientation, self.confidence = orientation, confidence
        self.same_gids, self.answer_direction = set(same_gids), answer_direction
        self.results, self.n, self.direction_jobs, self.compare_jobs = {}, 0, [], []

    def submit(self, a, b, context=None):
        self.n += 1
        self.compare_jobs.append(context["candidate_global_id"])
        d = "same" if context["candidate_global_id"] in self.same_gids else "different"
        self.results[self.n] = {"decision": d, "confidence": 0.95}
        return self.n

    def submit_direction(self, views, context=None):
        self.n += 1
        self.direction_jobs.append((context["cam_id"], context["local_id"], len(views)))
        if self.answer_direction:
            self.results[self.n] = {"decision": self.orientation, "confidence": self.confidence}
        return self.n

    def take_result(self, job):
        return self.results.pop(job, None)


def _mgr(vlm, **cfg):
    c = MTMCConfig(enabled=True, reid_enabled=False, topology="any", min_disappearance_s=2.0,
                   origin_cameras=["cam-01", "cam-02"], vlm_views=1, vlm_collect_max_s=0.0,
                   direction_cameras={"cam-02": "front_is_exit"}, direction_views=2, direction_wait_s=25.0,
                   tracked_classes=("vehicle",), **cfg)
    return MTMCManager(c, vlm_arbiter=vlm)


def _frames(m, cam, oid, t0, n, step=0.5, x=100.0):
    out = {}
    for i in range(n):
        box = (x, 200.0, 300.0, 200.0, "vehicle", 0.9, oid)
        out = m.process_frame(cam, int(t0 * 10) + i, t0 + i * step, [["vehicle", 0.9, oid]], boxes=[box], frame_bgr=FRAME)
    return out


def _gid(m, cam, oid):
    """The matcher's own global id of a tracklet (-1 = none yet). Not the tracklet's global_id, which the
    no-Re-ID fallback fills with its own counter."""
    for g in m.association.global_tracks.values():
        if oid in g.local_track_aliases.get(cam, ()):
            return g.global_id
    return -1


def _vehicle_inside(m):
    """A vehicle that entered on cam-01 and is out of view now (gid 0)."""
    m.association._vlm = None                              # no compare jobs for the setup
    _frames(m, "cam-01", 5, 0.0, 6)
    m.association._vlm = m.vlm_arbiter
    assert _gid(m, "cam-01", 5) == 0
    m.association.mark_tracklet_inactive("cam-01", 5, 3.0)


def test_front_view_on_the_exit_camera_exits_the_vehicle_that_is_inside():
    vlm = DirVlm("front", same_gids={0})
    m = _mgr(vlm)
    _vehicle_inside(m)
    _frames(m, "cam-02", 9, 100.0, 10)
    assert vlm.direction_jobs and vlm.direction_jobs[0][:2] == ("cam-02", 9)
    assert _gid(m, "cam-02", 9) == 0                         # matched to the vehicle inside
    ev = m.pop_exit_events()
    assert len(ev) == 1 and ev[0]["global_id"] == 0 and m.association.global_tracks[0].exited
    assert m._direction_total["exiting"] == 1


def test_rear_view_is_recorded_as_entering_and_does_not_skip_matching():
    vlm = DirVlm("rear", same_gids=())
    m = _mgr(vlm)
    _frames(m, "cam-02", 9, 100.0, 10)
    assert _gid(m, "cam-02", 9) >= 0                          # a new vehicle at the entrance
    assert m.pop_exit_events() == []
    ev = m.pop_direction_events()
    assert len(ev) == 1 and ev[0]["direction"] == "entering" and ev[0]["orientation"] == "rear"
    assert m.association.gate_hint("cam-02", 9) is None       # nothing was forced on the matcher


def test_an_unclear_or_unsure_answer_changes_nothing_and_stops_the_waiting():
    for orientation, conf in (("side", 0.9), ("unclear", 0.9), ("front", 0.4)):
        vlm = DirVlm(orientation, conf)
        m = _mgr(vlm)
        _frames(m, "cam-02", 9, 100.0, 10)
        assert _gid(m, "cam-02", 9) >= 0 and m.pop_exit_events() == [] and m.pop_direction_events() == []
        assert m._direction_total["undecided"] == 1


def test_a_new_vehicle_waits_for_the_verdict_and_is_judged_anyway_after_the_wait():
    vlm = DirVlm(answer_direction=False)
    m = _mgr(vlm, )
    _frames(m, "cam-02", 9, 100.0, 6)                         # 2.5 s of video: still waiting
    assert _gid(m, "cam-02", 9) == -1
    _frames(m, "cam-02", 9, 100.0 + 3, 60)                    # past direction_wait_s (25 s)
    assert _gid(m, "cam-02", 9) >= 0
    assert len(vlm.direction_jobs) == 1                       # asked once, not on every frame


def test_other_cameras_and_a_missing_vlm_are_never_held_up():
    vlm = DirVlm()
    m = _mgr(vlm)
    _frames(m, "cam-01", 4, 0.0, 6)
    assert _gid(m, "cam-01", 4) >= 0 and vlm.direction_jobs == []
    m2 = _mgr(None)
    _frames(m2, "cam-02", 9, 100.0, 6)
    assert m2._direction_map == {} and _gid(m2, "cam-02", 9) >= 0


def test_the_direction_answer_is_parsed_from_json_fences_prose_and_fragments():
    assert _parse_direction('{"orientation": "front", "confidence": 0.9, "evidence": ["grille", "headlights"]}') == \
        {"decision": "front", "confidence": 0.9, "evidence": ["grille", "headlights"]}
    assert _parse_direction('```json\n{"orientation":"REAR","confidence":"0.8"}\n```')["decision"] == "rear"
    assert _parse_direction('Sure: {"orientation": "side", "confidence": 0.6} done')["decision"] == "side"
    assert _parse_direction('orientation: front, confidence: 0.75')["confidence"] == 0.75
    assert _parse_direction('{"orientation": "upside down"}') is None and _parse_direction("") is None


class SeqVlm(DirVlm):
    """Answers the direction question from a list, one answer per ask (the last one repeats)."""
    def __init__(self, answers, **kw):
        super().__init__(**kw)
        self.answers = list(answers)

    def submit_direction(self, views, context=None):
        self.n += 1
        self.direction_jobs.append((context["cam_id"], context["local_id"], len(views)))
        o, c = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        self.results[self.n] = {"decision": o, "confidence": c}
        return self.n


def _strict(vlm, **cfg):
    c = MTMCConfig(enabled=True, reid_enabled=False, topology="any", min_disappearance_s=2.0,
                   origin_cameras=["cam-01", "cam-02"], vlm_views=1, vlm_collect_max_s=0.0,
                   direction_cameras={"cam-01": "front_is_enter", "cam-02": "front_is_exit"}, direction_views=2,
                   direction_wait_s=25.0, direction_retry_s=4.0, direction_require=True,
                   tracked_classes=("vehicle",), **cfg)
    return MTMCManager(c, vlm_arbiter=vlm)


def test_on_a_front_is_enter_camera_the_front_enters_and_the_rear_leaves():
    m = _strict(DirVlm("front", same_gids=()))
    _frames(m, "cam-01", 4, 0.0, 10)
    assert _gid(m, "cam-01", 4) >= 0                          # front at D41 = entering: a track starts
    assert m.pop_direction_events()[0]["direction"] == "entering"


def test_the_rear_at_a_front_is_enter_camera_closes_the_vehicle_that_is_inside():
    m = _strict(DirVlm("rear", same_gids=(0,)))
    _frames(m, "cam-02", 5, 0.0, 10)                           # rear at D16 = entering: gid 0 starts
    assert _gid(m, "cam-02", 5) == 0
    m.association.mark_tracklet_inactive("cam-02", 5, 6.0)
    _frames(m, "cam-01", 9, 100.0, 10)                         # rear at D41 = leaving: matched to gid 0 and closed
    assert _gid(m, "cam-01", 9) == 0 and m.association.global_tracks[0].exited


def test_a_leaving_vehicle_that_matches_nothing_inside_never_starts_a_track():
    m = _strict(DirVlm("rear", same_gids=()))                 # rear at D41 = leaving
    _frames(m, "cam-01", 4, 0.0, 10)
    assert _gid(m, "cam-01", 4) == -1 and not m.association.global_tracks


def test_an_unclear_answer_is_asked_again_and_a_later_entering_starts_the_track():
    vlm = SeqVlm([("side", 0.9), ("side", 0.9), ("front", 0.95)])
    m = _strict(vlm)
    _frames(m, "cam-01", 4, 0.0, 6)
    assert _gid(m, "cam-01", 4) == -1                         # still undecided
    _frames(m, "cam-01", 4, 3.0, 40)                          # more video: asked again (every 4 s), third answer decides
    assert len(vlm.direction_jobs) == 3
    assert _gid(m, "cam-01", 4) >= 0


def test_a_vehicle_whose_direction_never_gets_decided_does_not_start_a_track():
    vlm = SeqVlm([("side", 0.9)])
    m = _strict(vlm)
    _frames(m, "cam-01", 4, 0.0, 80)                          # 40 s of video, past direction_wait_s
    assert _gid(m, "cam-01", 4) == -1 and not m.association.global_tracks
    assert len(vlm.direction_jobs) > 1                        # it was asked again, not once


def _moving(m, cam, oid, t0, n, dx, step=0.5):
    """A vehicle that moves dx pixels per frame across the picture."""
    out = {}
    for i in range(n):
        box = (400.0 + dx * i, 200.0, 300.0, 200.0, "vehicle", 0.9, oid)
        out = m.process_frame(cam, int(t0 * 10) + i, t0 + i * step, [["vehicle", 0.9, oid]], boxes=[box], frame_bgr=FRAME)
    return out


def _side(vlm, **cfg):
    c = MTMCConfig(enabled=True, reid_enabled=False, topology="any", min_disappearance_s=2.0,
                   origin_cameras=["cam-01", "cam-02"], vlm_views=1, vlm_collect_max_s=0.0,
                   direction_cameras={"cam-01": "side_right_to_left_enters"}, direction_views=2,
                   direction_wait_s=25.0, direction_retry_s=4.0, direction_require=True,
                   tracked_classes=("vehicle",), **cfg)
    return MTMCManager(c, vlm_arbiter=vlm)


def test_side_view_front_pointing_left_and_moving_right_to_left_enters():
    m = _side(DirVlm("left", same_gids=()))
    _moving(m, "cam-01", 4, 0.0, 40, dx=-40)         # the first answer comes too early to see the movement: asked again
    assert _gid(m, "cam-01", 4) >= 0
    assert m.pop_direction_events()[0]["direction"] == "entering"


def test_side_view_front_pointing_right_means_leaving_and_starts_no_track():
    m = _side(DirVlm("right", same_gids=()))
    _moving(m, "cam-01", 4, 0.0, 40, dx=40)
    assert _gid(m, "cam-01", 4) == -1 and not m.association.global_tracks


def test_side_view_vlm_and_movement_that_disagree_are_not_decided():
    vlm = SeqVlm([("left", 0.95)])
    m = _side(vlm)
    _moving(m, "cam-01", 4, 0.0, 80, dx=40)                   # says left, drives left to right: never decided
    assert _gid(m, "cam-01", 4) == -1 and len(vlm.direction_jobs) > 1


def test_side_view_toward_the_camera_enters_when_the_vehicle_hardly_moves():
    m = _side(DirVlm("toward", same_gids=()))
    _moving(m, "cam-01", 4, 0.0, 10, dx=0)
    assert _gid(m, "cam-01", 4) >= 0


def test_none_means_no_vehicle_is_parsed_for_both_questions():
    assert _parse_direction('{"orientation": "none", "confidence": 0.9, "evidence": ["bare ground"]}')["decision"] == "none"
    assert _parse_direction('{"orientation": "NONE", "confidence": 0.9}', side=True)["decision"] == "none"
    assert _parse_direction('orientation: none, confidence: 0.8', side=True)["decision"] == "none"
    assert _parse_direction('{"orientation": "left"}')is None                 # left is only a side-view word


def test_a_picture_with_no_vehicle_is_final_never_starts_a_track_and_is_asked_once():
    for mk in (_strict, _side):
        vlm = SeqVlm([("none", 0.9)])
        m = mk(vlm)
        cam = "cam-02" if mk is _strict else "cam-01"
        _moving(m, cam, 4, 0.0, 80, dx=0)                       # 40 s of video, far past the retries
        assert _gid(m, cam, 4) == -1 and not m.association.global_tracks
        assert len(vlm.direction_jobs) == 1 and m._direction_total["not_vehicle"] == 1
        assert m.pop_direction_events() == [] and m.pop_exit_events() == []


def test_a_none_answer_that_is_not_sure_is_treated_as_unclear_and_asked_again():
    vlm = SeqVlm([("none", 0.3), ("rear", 0.95)])
    m = _strict(vlm)
    _moving(m, "cam-02", 4, 0.0, 40, dx=0)
    assert len(vlm.direction_jobs) == 2 and _gid(m, "cam-02", 4) >= 0
