import importlib.util
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    "gpu_exp", pathlib.Path(__file__).resolve().parents[1] / "scripts/scale_gpu_experiment.py")
ge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ge)

YAML = """stream_fps: 10            # preview rate
pipeline_preview: True
process_fps: 5
analysis_fps: 0
gpu_jpeg: true
"""


def test_overrides_change_only_the_named_keys_and_keep_comments():
    out = ge.apply_overrides(YAML, {"process_fps": "2.5", "pipeline_preview": "false", "new_key": "1"})
    assert "process_fps: 2.5\n" in out and "pipeline_preview: false\n" in out
    assert "stream_fps: 10            # preview rate" in out and "gpu_jpeg: true" in out
    assert out.rstrip().endswith("new_key: 1")


def test_the_original_file_comes_back_byte_for_byte_even_when_the_block_raises(tmp_path):
    p = tmp_path / "sidecar.yaml"
    p.write_bytes(YAML.encode())
    with pytest.raises(RuntimeError):
        with ge.overridden(str(p), {"analysis_fps": "10"}):
            assert "analysis_fps: 10" in p.read_text()
            raise RuntimeError("test failed halfway")
    assert p.read_bytes() == YAML.encode()


def test_dmon_rows_become_stats():
    text = "# gpu sm mem enc dec jpg ofa fb bar1 ccpm\n# Idx % % % % % % MB MB MB\n" + "".join(
        f"    0 {sm:4d} {sm // 2:4d}    0   24    0    0 4682    6    0\n" for sm in [50, 60, 70, 80, 90, 95, 100, 100, 70, 85])
    s = ge.gpu_stats(ge.parse_dmon(text))
    assert s["samples"] == 10 and s["sm_avg"] == 80 and s["sm_ge90_pct"] == 40.0 and s["dec_avg"] == 24
    assert ge.gpu_stats([]) == {}


def test_table_has_one_row_per_test_and_dashes_for_missing_values():
    t = ge.render_table([{"test": "T0 baseline", "sm_avg": 77.3, "fps_min": 19.6},
                         {"test": "T1 previews off", "sm_avg": 55.0}])
    lines = t.splitlines()
    assert len(lines) == 4 and "T0 baseline | 77.3 |" in lines[2] and "| 19.6 |" in lines[2] and lines[3].count("-") >= 5
