"""Moving a run in time (scripts/scale_day_shift.py): only run timestamps under known keys, only after the threshold."""
import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location("shift", pathlib.Path(__file__).parent.parent / "scripts" / "scale_day_shift.py")
sh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sh)

THR = 1_790_000_000.0


def test_shifts_run_times_and_leaves_the_rest():
    row = {"submitted_at": THR + 100, "resolved_at": THR + 105, "pts": THR + 5, "job_id": 3, "conf": 0.9, "text": "x"}
    out = sh.shift_value(row, -1000.0, THR)
    assert out["submitted_at"] == THR - 900 and out["resolved_at"] == THR - 895
    assert out["pts"] == THR + 5 and out["job_id"] == 3 and out["conf"] == 0.9     # not a time key


def test_stays_episodes_nested_in_lists():
    state = {"saved_at": THR + 50, "cameras": {"cam-01": {"7": {"wall_first": THR + 10, "wall_last": THR + 20,
                                                                  "episodes": [[THR + 1, THR + 2]], "count": 4}}}}
    out = sh.shift_value(state, 60.0, THR)
    v = out["cameras"]["cam-01"]["7"]
    assert (v["wall_first"], v["wall_last"], v["episodes"], v["count"]) == (THR + 70, THR + 80, [[THR + 61, THR + 62]], 4)
    assert out["saved_at"] == THR + 110


def test_values_before_the_threshold_are_left_alone():
    assert sh.shift_value({"updated_at": THR - 5}, 99.0, THR) == {"updated_at": THR - 5}
