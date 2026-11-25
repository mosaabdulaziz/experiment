"""Set analysis_fps in sidecar.yaml to a given value (in place).

Usage: python3 benchmarks/_set_af.py <analysis_fps>
"""
import re
import sys

p = "perception/pipeline/sidecar.yaml"
t = open(p).read()
t2 = re.sub(r"(?m)^analysis_fps:.*$", f"analysis_fps: {sys.argv[1]}", t)
open(p, "w").write(t2)
print(f"analysis_fps set to {sys.argv[1]}")
