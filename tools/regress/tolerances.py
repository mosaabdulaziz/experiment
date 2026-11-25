"""How far an efficiency metric may move from its baseline before it is reported (benchmarks/tolerances.json).

A rule is {"match": <fnmatch on a metric id or family>, "rel": <fraction of the baseline>, "abs": <absolute amount>}; the
allowed deviation is the LARGER of the two (a relative tolerance alone is useless for a baseline near 0: drops 0 -> 1).
The first rule that matches wins; `default` covers the rest. The numbers are proposed by `regress noise` from repeated
identical runs of one commit and are never tighter than what that noise showed: a tolerance inside the noise would make
every merge request a coin toss. Correctness has no tolerance (it is exact, or the golden replay's own semantic diff).
"""
import fnmatch
import json
import os

DEFAULT = {"rel": 0.10, "abs": 0.0}


def load(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {"default": dict(DEFAULT), "rules": [], "source": None}
    with open(path) as f:
        data = json.load(f)
    data.setdefault("default", dict(DEFAULT))
    data.setdefault("rules", [])
    data["source"] = path
    return data


def rule_for(tol: dict, metric_id: str, fam: str) -> dict:
    return next((r for r in tol["rules"] if fnmatch.fnmatchcase(metric_id, r["match"]) or fnmatch.fnmatchcase(fam, r["match"])),
                tol["default"])


def allowed(tol: dict, metric_id: str, fam: str, baseline: float) -> float:
    """The deviation from `baseline` that is still within tolerance."""
    rule = rule_for(tol, metric_id, fam)
    return max(abs(baseline) * float(rule.get("rel", 0.0)), float(rule.get("abs", 0.0)))


def gated(tol: dict, metric_id: str, fam: str) -> bool:
    """False for a metric that proved unstable between identical runs: it is shown, never judged."""
    return rule_for(tol, metric_id, fam).get("gate", True) is not False
