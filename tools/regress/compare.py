"""Is this run worse than the baseline? PASS / WARN / FAIL (task #230).

    python3 -m regress compare benchmarks/runs/<machine>/<run>.json [--baseline <file>] [--strict] [--all]

Rules, in the order they matter:
  * Correctness FAILS: a unit suite with more failures than the baseline, fewer tests than the baseline, a golden replay
    that did not run, or whose output differs from the baseline's (exactly, or per the replay's own semantic diff for the
    configs that are compared that way). There is no tolerance on these.
  * Efficiency WARNS (FAILS only with --strict): a metric that moved the wrong way by more than its tolerance
    (benchmarks/tolerances.json). A move the right way is reported as IMPROVED, not hidden.
  * "Not directly comparable": efficiency only means something against a baseline from the SAME machine. On another
    machine the efficiency rows are shown for information (INFO) and the golden outputs are not judged either, because a
    different GPU or TensorRT build changes detections without anything being wrong. Unit suites are still judged.
  * A run marked NOISY (something else was using the GPU) is advisory: its efficiency rows are INFO.
  * A rebuilt TensorRT engine, or counters that were invalid, are left out by the metric list itself (metrics.py).
"""
import json
import os

from . import metrics, tolerances

PASS, WARN, FAIL, INFO, IMPROVED = "PASS", "WARN", "FAIL", "INFO", "IMPROVED"


def load_baseline(path: str) -> dict:
    """A baseline is `baseline.json` (written by `regress baseline`) or any results file (compare two runs directly)."""
    with open(path) as f:
        doc = json.load(f)
    if doc.get("schema") == "regress.baseline.v1":
        doc["_dir"] = os.path.dirname(os.path.abspath(path))
        return doc
    flat = metrics.flatten(doc)
    golden = {}
    for cfg, g in (doc.get("correctness", {}).get("golden") or {}).items():
        if g.get("ran"):
            golden[cfg] = {"sha256": g["output_sha256"], "output": g.get("output_file")}
    return {"schema": "regress.baseline.v1", "machine_id": doc.get("machine_id"), "fingerprint": doc.get("fingerprint", {}),
            "noisy": doc.get("noisy", False), "metrics": {k: {"value": m.value, "kind": m.kind, "direction": m.direction, "unit": m.unit}
                                                          for k, m in flat.items()},
            "golden": golden, "_dir": None, "_from_results": True}


def _row(mid, kind, base, new, status, note="", unit=""):
    return {"id": mid, "kind": kind, "baseline": base, "new": new, "status": status, "note": note, "unit": unit}


def _change(base, new) -> str:
    if isinstance(base, (int, float)) and isinstance(new, (int, float)) and not isinstance(base, bool):
        d = new - base
        rel = f" ({100 * d / abs(base):+.1f}%)" if base else ""
        return f"{d:+g}{rel}"
    return ""


def _judge_efficiency(mid, b, n, direction, tol):
    """(status, note) of one efficiency metric."""
    if not isinstance(b, (int, float)) or isinstance(b, bool) or not isinstance(n, (int, float)):
        return (PASS, "") if b == n else (WARN, "changed")
    allow = tolerances.allowed(tol, mid, metrics.family(mid), float(b))
    d = n - b
    if direction == "equal":
        return (PASS, "") if d == 0 else (WARN, "changed")
    if abs(d) <= allow:
        return PASS, ""
    worse = (d > 0) if direction in ("lower",) else (d < 0) if direction == "higher" else True
    if direction == "both":
        return WARN, f"moved by more than {allow:g}"
    return (WARN, f"worse by more than {allow:g}") if worse else (IMPROVED, "")


def _resolve(path, root):
    """A results file records its golden outputs relative to the checkout that wrote it: `root` is that checkout."""
    if path and root and not os.path.isabs(path):
        return os.path.join(root, path)
    return path


def compare(new: dict, base: dict, tol: dict, golden_diff=None, strict: bool = False, outputs_root: str = None) -> dict:
    """{"rows": [...], "notes": [...], "comparable": bool, "counts": {status: n}, "verdict": PASS|WARN|FAIL, "exit": 0|1}.
    `golden_diff(cfg, before_path, after_path) -> (ok, status_text)` judges two golden outputs that are not byte-identical.
    `outputs_root`: the checkout the results file was written in (its golden outputs are paths relative to it)."""
    rows, notes = [], []
    new_fp, base_fp = new.get("fingerprint", {}), base.get("fingerprint", {})
    comparable = bool(new.get("machine_id")) and new.get("machine_id") == base.get("machine_id")
    if not comparable:
        notes.append(f"not directly comparable: the baseline is from {base.get('machine_id')}, this run from "
                     f"{new.get('machine_id')}. Efficiency rows are shown for information and the golden outputs are not "
                     "judged; unit suites still are.")
    for key, label in (("driver", "driver"), ("cuda", "CUDA"), ("gpu", "GPU")):
        if comparable and base_fp.get(key) != new_fp.get(key) and base_fp.get(key) and new_fp.get(key):
            notes.append(f"{label} differs from the baseline's ({base_fp.get(key)} -> {new_fp.get(key)}): efficiency may differ for that reason")
    if new.get("noisy"):
        notes.append("this run is marked NOISY (something else was using the machine): efficiency rows are advisory")
    if base.get("noisy"):
        notes.append("the baseline itself was recorded on a noisy machine: re-record it")
    advisory = (not comparable) or bool(new.get("noisy"))

    flat = metrics.flatten(new)
    for mid, b in sorted(base["metrics"].items()):
        if mid.startswith("golden."):
            continue                                   # judged below, against the stored output
        if mid not in flat:
            continue                                   # that section was not run this time
        n = flat[mid]
        if b["kind"] == "correctness":
            if b["direction"] == "zero":
                st, note = (PASS, "") if n.value <= b["value"] else (FAIL, f"{n.value} failing, the baseline had {b['value']}")
            elif b["direction"] == "higher":
                st, note = (PASS, "") if n.value >= b["value"] else (FAIL, f"fewer tests than the baseline ({n.value} < {b['value']})")
            else:
                st, note = (PASS, "") if n.value == b["value"] else (FAIL, "changed")
            rows.append(_row(mid, "correctness", b["value"], n.value, st, note, b.get("unit", "")))
            continue
        if not tolerances.gated(tol, mid, metrics.family(mid)):
            rows.append(_row(mid, "efficiency", b["value"], n.value, INFO, "not gated: unstable between identical runs", b.get("unit", "")))
            continue
        st, note = _judge_efficiency(mid, b["value"], n.value, b["direction"], tol)
        if advisory and st in (WARN, IMPROVED):
            st, note = INFO, (note + "; " if note else "") + ("different machine" if not comparable else "noisy run")
        rows.append(_row(mid, "efficiency", b["value"], n.value, st, note, b.get("unit", "")))

    gnew = (new.get("correctness", {}).get("golden") or {})
    if base.get("golden") and not gnew:
        notes.append("no golden replay in this run: the output check was not made")
    for cfg, g in sorted((base.get("golden") or {}).items()) if gnew else []:
        mid = f"golden.{cfg}.output"
        r = gnew.get(cfg)
        if r is None:
            # this run did not ask for that config (a partial run: `--configs zones`, or the default six against a baseline that also
            # holds PPE, fall and shelf): it is judged on what it covered. --strict asks for everything.
            rows.append(_row(mid, "correctness", "recorded", "not covered", INFO, "not covered by this run (--strict makes this a failure)"))
            continue
        if not r.get("ran"):
            rows.append(_row(mid, "correctness", "recorded", "did not run", FAIL if comparable else INFO, r.get("error", "no replay of this config")[:100]))
            continue
        if not comparable:
            rows.append(_row(mid, "correctness", g["sha256"][:10], r["output_sha256"][:10], INFO, "different machine: not judged"))
            continue
        if r["output_sha256"] == g["sha256"]:
            rows.append(_row(mid, "correctness", g["sha256"][:10], r["output_sha256"][:10], PASS, "identical"))
            continue
        before = os.path.join(base["_dir"], g["output"]) if base.get("_dir") and g.get("output") else _resolve(g.get("output"), outputs_root)
        after = _resolve(r.get("output_file"), outputs_root)
        if golden_diff and before and after and os.path.exists(before) and os.path.exists(after):
            ok, text = golden_diff(cfg, before, after)
            # any engine rebuilt in this run counts, not only this config's own: configs share engines (zones and barber-faces reuse what
            # attendance built a minute earlier), so the first config to need an engine pays for it and the others inherit it
            rebuilt = any((v.get("engine") or {}).get("rebuilt") for v in (new.get("runtime") or {}).values() if isinstance(v, dict))
            if not ok and rebuilt:
                # found the hard way (2026-11-12): two checkouts of the SAME code each built their own TensorRT engine and gave
                # two different, individually stable multicam outputs. A rebuilt engine is a different model file.
                rows.append(_row(mid, "correctness", g["sha256"][:10], r["output_sha256"][:10], WARN,
                                 f"{text} (a TensorRT engine was rebuilt in this run: a new build can change the output; run again)"))
            elif not ok and g.get("deterministic") is False:
                # the baseline commit's own replays disagreed with each other: failing on this would fail 1 run in N of main
                rows.append(_row(mid, "correctness", g["sha256"][:10], r["output_sha256"][:10], WARN,
                                 f"{text} (this config is flaky: identical runs of the baseline commit differed too)"))
            else:
                rows.append(_row(mid, "correctness", g["sha256"][:10], r["output_sha256"][:10], PASS if ok else FAIL, text))
        else:
            rows.append(_row(mid, "correctness", g["sha256"][:10], r["output_sha256"][:10], WARN,
                             "output differs and the baseline's output file is not here to diff against (the golden outputs are in DVC: `just data-pull golden`)"))
    for cfg in sorted(set(gnew) - set(base.get("golden") or {})):
        rows.append(_row(f"golden.{cfg}.output", "correctness", "-", "recorded" if gnew[cfg].get("ran") else "did not run", INFO,
                         "no baseline for this config yet"))

    if strict:
        for r in rows:
            if r["status"] == WARN or (r["status"] == INFO and r["note"].startswith("not covered")):
                r["status"], r["note"] = FAIL, (r["note"] + " (strict)").strip()
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in (PASS, IMPROVED, INFO, WARN, FAIL)}
    verdict = FAIL if counts[FAIL] else WARN if counts[WARN] else PASS
    return {"rows": rows, "notes": notes, "comparable": comparable, "counts": counts, "verdict": verdict,
            "exit": 1 if counts[FAIL] else 0}


def render(rep: dict, show_all: bool = False) -> str:
    out = []
    shown = [r for r in rep["rows"] if show_all or r["status"] != PASS]
    if shown:
        out.append(f"{'':<9}{'metric':<54}{'baseline':>14}{'this run':>14}  change")
        order = {FAIL: 0, WARN: 1, IMPROVED: 2, INFO: 3, PASS: 4}
        for r in sorted(shown, key=lambda r: (order[r["status"]], r["id"])):
            ch = _change(r["baseline"], r["new"])
            tail = "  ".join(x for x in (ch, r["note"]) if x)
            out.append(f"{r['status']:<9}{r['id']:<54}{str(r['baseline']):>14}{str(r['new']):>14}  {tail}")
    c = rep["counts"]
    out.append("")
    out += [f"note: {n}" for n in rep["notes"]]
    out.append(f"{c[PASS]} pass, {c[IMPROVED]} improved, {c[INFO]} info, {c[WARN]} warn, {c[FAIL]} FAIL  ->  {rep['verdict']}")
    return "\n".join(out)
