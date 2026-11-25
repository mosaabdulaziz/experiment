import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location(
    "engine_gpu_budget", pathlib.Path(__file__).resolve().parents[1] / "tools" / "engine_gpu_budget.py")
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


def _line(name, n, total, busy, avg, mx, window=5.0):
    return (f"deepstream-sidecar  | 2026 INFO [telemetry] {name} n={n} total={total}ms busy={busy}% avg={avg}ms "
            f"max={mx}ms thread=gpu window={window}s")


LOG = "\n".join([
    "[telemetry] probe n=10 total=5.0ms busy=1.0% avg=0.50ms max=1.00ms thread=streaming window=5.0s",   # not a GPU stage
    _line("gpu.infer.pgie", 100, 1000.0, 20.0, 10.0, 15.0), _line("gpu.decode.cam-1", 150, 500.0, 10.0, 3.33, 5.0),
    _line("gpu.decode.cam-2", 150, 500.0, 10.0, 3.33, 5.0),            # window 1: model load, skipped
    _line("gpu.infer.pgie", 100, 1500.0, 30.0, 15.0, 20.0), _line("gpu.infer.sgie_face", 50, 500.0, 10.0, 10.0, 12.0),
    _line("gpu.decode.cam-1", 150, 500.0, 10.0, 3.33, 5.0), _line("gpu.decode.cam-2", 150, 500.0, 10.0, 3.33, 5.0),
    _line("gpu.infer.pgie", 100, 1000.0, 20.0, 10.0, 25.0), _line("gpu.infer.sgie_face", 50, 700.0, 14.0, 14.0, 18.0),
    _line("gpu.decode.cam-1", 150, 500.0, 10.0, 3.33, 5.0),
])


def test_the_first_window_is_dropped_and_the_rest_averaged_per_stage():
    s = budget.parse(LOG)
    assert round(s["gpu.infer.pgie"]["busy"], 1) == 25.0 and s["gpu.infer.pgie"]["max_ms"] == 25.0
    assert round(s["gpu.infer.sgie_face"]["busy"], 1) == 12.0
    assert round(s["gpu.infer.sgie_face"]["avg_ms"], 1) == 12.0          # 1200 ms over 100 batches


def test_cameras_decoders_are_one_stage_whose_busy_adds_up():
    s = budget.parse(LOG)
    assert round(s["gpu.decode"]["busy"], 1) == 15.0                      # (20 + 10) / 2 windows


def test_the_total_and_the_table_name_every_stage_biggest_first():
    s = budget.parse(LOG)
    assert round(budget.total(s), 1) == 52.0
    lines = budget.render(s).splitlines()
    assert lines[1].startswith("gpu.infer.pgie") and lines[-1].startswith("total")


def test_a_log_without_gpu_lines_reports_nothing_to_budget(tmp_path, capsys):
    f = tmp_path / "log"
    f.write_text("nothing here")
    assert budget.main([str(f)]) == 2
    f.write_text(LOG)
    assert budget.main([str(f), "--max-total", "40"]) == 1
    assert budget.main([str(f), "--max-total", "90"]) == 0
