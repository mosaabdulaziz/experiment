import sys
import pathlib

from perception.modules.mtmc.vlm_arbiter import VLMArbiter


def test_health_counts_outcomes_latency_and_backlog():
    v = VLMArbiter(enabled=False)
    assert v.health()["answered_total"] == 0 and v.health()["last_answer_age_s"] == -1.0
    v._queue.put(object())
    v._in_flight = 2
    v._note_done({"decision": "same"}, 3.0, 1.0)
    v._note_done({"decision": "different"}, 5.0, 0.5)
    v._note_done({"decision": "uncertain", "error": "timeout"}, 40.0, 2.0)
    h = v.health()
    assert (h["answered_total"], h["same_total"], h["different_total"], h["uncertain_total"]) == (3, 1, 1, 1)
    assert h["errors_total"] == 1 and h["in_flight"] == 0.0 and h["queue_depth"] == 1.0
    assert h["compare_max_s"] == 40.0 and h["compare_last_s"] == 40.0 and h["wait_max_s"] == 2.0
    assert 0 <= h["last_answer_age_s"] < 5


def test_a_direction_job_returns_the_orientation_and_is_counted_apart_from_comparisons():
    import numpy as np
    v = VLMArbiter(enabled=False)
    sent = {}

    def fake_call(prompt, imgs):
        sent["prompt"], sent["n"] = prompt, len(imgs)
        return '{"orientation": "front", "confidence": 0.92, "evidence": ["grille", "headlights"]}'
    v._call = fake_call
    out = v._direction([np.zeros((40, 60, 3), np.uint8)] * 2, {"cam_id": "cam-02"})
    assert out["decision"] == "front" and out["confidence"] == 0.92 and out["explanation"] == "grille; headlights"
    assert sent["n"] == 2 and "which end of the vehicle faces the camera" in sent["prompt"].lower()
    v._call = lambda p, i: "I cannot tell"
    assert v._direction([np.zeros((40, 60, 3), np.uint8)])["decision"] == "unclear"
    v._note_done(out, 2.0, 0.1, True)
    v._note_done({"decision": "same"}, 2.0, 0.1)
    h = v.health()
    assert h["direction_total"] == 1 and h["same_total"] == 1 and h["answered_total"] == 2
