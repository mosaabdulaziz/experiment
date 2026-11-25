"""Print a compact one-line summary of a capacity cell JSON.

Usage: python3 benchmarks/_cell_summary.py <path-to-cell.json>
"""
import json
import sys

d = json.load(open(sys.argv[1]))
gu = d.get("gpu_util") or {}
gm = d.get("gpu_mem_mib") or {}
vs = sorted((d.get("per_source_fps") or {}).values())
med = vs[len(vs) // 2] if vs else None
mn = vs[0] if vs else None
print(
    f"n={d.get('n')} fps={d.get('fps')} af={d.get('analysis_fps')} "
    f"achieved_med={med} min={mn} "
    f"gpu_med={gu.get('median')} gpu_p95={gu.get('p95')} "
    f"vram={gm.get('median')} cpu={(d.get('cpu_pct') or {}).get('median')} "
    f"sat={d.get('publisher_saturated_warnings')} errs={len(d.get('error_lines') or [])}"
)
