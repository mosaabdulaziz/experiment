"""Import the models the products load but no tracked run produced, from the catalog into MLflow.

    python -m mtmc.tracking.import_models --dry-run            # what would be imported; touches nothing
    python -m mtmc.tracking.import_models                      # import (MLFLOW_TRACKING_URI, else offline)
    python -m mtmc.tracking.import_models --only plate-ocr/v5_arabic_mobile --force

One run per catalog entry (mtmc/tracking/imported_models.yaml) in the experiment `imported-models`, through the
tracking helper (so the run carries the lineage tags of the import itself: git commit, host, versions): provenance
tags, params and metrics parsed from the existing records (Ultralytics args.yaml / results.csv, the Paddle config and
the Kaggle training report, the staff classifier's .json), the record files as artifacts, and every model file via
log_model_file, which writes `<file>.mlflow.json` beside it.

It does NOT register models: registering trained and imported models alike is the registry sub-task's job.

Idempotent: each run is tagged `import.key` = `<capability>:<sha256 of the entry's model file>`, and an entry whose key
already has a FINISHED run is skipped (`--force` imports it again and tags the older runs `import.superseded_by`).

Metric steps are 0-based epochs: per-epoch metrics (results.csv, the Kaggle training-report.md) are logged at
step = epoch - 1, because Ultralytics' MLflow callback logs at step = trainer.epoch (0-based) while results.csv and the
report number epochs from 1. Imported and newly trained runs then line up on the same step axis when compared. Every
imported run is tagged `import.step_base=0` to say so.
An entry whose files are not all present (DVC data not pulled), or that does not match a sha256 the repo pins, is
skipped with the reason; the other entries still import.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import _backend, _lineage, log_model_file, tracked_run

logger = logging.getLogger("mtmc.tracking")

CATALOG = Path(__file__).with_name("imported_models.yaml")
EXPERIMENT = "imported-models"
STEP_BASE = 0                                   # metric step of the first epoch (Ultralytics' MLflow callback: 0)
SOURCES = ("imported", "kaggle")
LOCKFILE = "tools/mtmc-research/uv.lock"       # the environment the import runs in (D7)


# -- catalog ------------------------------------------------------------------------------------------------------
@dataclass
class Entry:
    id: str
    capability: str
    source: str
    files: list[str]
    records: list[dict] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    data: dict[str, str] = field(default_factory=dict)
    expected_sha256: dict[str, str] = field(default_factory=dict)
    provenance: dict | str = "unknown"
    evidence: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def provenance_status(self) -> str:
        if self.provenance == "unknown":
            return "unknown"
        return "partial" if self.provenance.get("gaps") else "known"


class CatalogError(ValueError):
    pass


def load_catalog(path: Path = CATALOG) -> list[Entry]:
    import yaml                                   # noqa: PLC0415 (research environment dependency)

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries, seen = [], set()
    for i, e in enumerate(raw.get("entries") or []):
        where = f"{path}: entry {i} ({e.get('id', '?')})"
        missing = [k for k in ("id", "capability", "source", "files") if not e.get(k)]
        if missing:
            raise CatalogError(f"{where}: missing {missing}")
        if e["id"] in seen:
            raise CatalogError(f"{where}: duplicate id")
        seen.add(e["id"])
        if e["source"] not in SOURCES:
            raise CatalogError(f"{where}: source must be one of {SOURCES}, got {e['source']!r}")
        prov = e.get("provenance", "unknown")
        if prov != "unknown" and not isinstance(prov, dict):
            raise CatalogError(f"{where}: provenance must be a mapping or `unknown`")
        for r in e.get("records") or []:
            if r.get("kind") not in PARSERS or not r.get("file"):
                raise CatalogError(f"{where}: record {r} needs kind (one of {sorted(PARSERS)}) and file")
        unknown_keys = set(e) - {f for f in Entry.__dataclass_fields__}
        if unknown_keys:
            raise CatalogError(f"{where}: unknown keys {sorted(unknown_keys)}")
        entries.append(Entry(**{k: v for k, v in e.items() if v is not None}))
    return entries


# -- record parsers: (params, {step: metrics}) ---------------------------------------------------------------------
Parsed = tuple[dict[str, str], dict[int | None, dict[str, float]]]


def _scalar(v: Any) -> str:
    return json.dumps(v) if isinstance(v, (list, dict)) else str(v)


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _slug(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", s.lower()).strip("_")


def parse_ultralytics_args(path: Path) -> Parsed:
    import yaml                                   # noqa: PLC0415
    args = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {k: _scalar(v) for k, v in args.items()}, {}


def parse_ultralytics_results(path: Path) -> Parsed:
    """results.csv: one row per epoch; step = the csv's `epoch` - 1 (the csv is 1-based, Ultralytics' MLflow callback
    logs 0-based steps; see STEP_BASE). Metric names are
    sanitized the way Ultralytics' MLflow callback does (parentheses dropped: `metrics/mAP50-95(B)` -> `...B`)."""
    steps: dict[int | None, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
            epoch = int(float(row.pop("epoch")))
            steps[epoch - 1 + STEP_BASE] = {k.replace("(", "").replace(")", ""): float(v) for k, v in row.items() if v}
    return {}, steps


def parse_paddle_config(path: Path) -> Parsed:
    import yaml                                   # noqa: PLC0415
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {k: _scalar(v) for k, v in _flatten(cfg).items()}, {}


_EMPTY = {"", "—", "-", "n/a"}


def parse_ocr_training_report(path: Path) -> Parsed:
    """training-report.md: the header line (`GPUs: ... | batch per GPU: 8 | samples: ...`) becomes `report.*` params,
    the per-epoch table becomes metrics with step = Epoch - 1 (1-based in the report, 0-based steps like Ultralytics'
    MLflow callback; see STEP_BASE) (cells `—` = not logged that epoch)."""
    params: dict[str, str] = {}
    steps: dict[int | None, dict[str, float]] = {}
    header: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not header and not s.startswith("|") and "|" in s and ":" in s:
            for part in s.split("|"):
                if ":" in part:
                    k, v = part.split(":", 1)
                    params[f"report.{_slug(k)}"] = v.strip()
            continue
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if header is None:
            if cells and cells[0].lower() == "epoch":
                header = [_slug(c) for c in cells]
            continue
        if all(set(c) <= set("-: ") for c in cells):           # the |---:| separator row
            continue
        row = dict(zip(header, cells))
        epoch = int(row.pop("epoch"))
        steps[epoch - 1 + STEP_BASE] = {k: float(v) for k, v in row.items() if v not in _EMPTY}
    if header is None:
        raise ValueError(f"{path}: no per-epoch table (a header row starting with `| Epoch`)")
    return params, steps


def parse_staff_json(path: Path) -> Parsed:
    """The staff classifier's metadata: scalar fields -> params; palm_test and val_points -> metrics (no step)."""
    meta = json.loads(path.read_text(encoding="utf-8"))
    params, metrics = {}, {}
    for k, v in meta.items():
        if k == "val_points":
            for p in v:
                t = p.get("threshold")
                for mk, mv in p.items():
                    if mk != "threshold" and isinstance(mv, (int, float)):
                        metrics[f"val/{mk}_at_{t}"] = float(mv)
        elif isinstance(v, dict):
            for fk, fv in _flatten(v).items():
                if isinstance(fv, (int, float)) and not isinstance(fv, bool):
                    metrics[f"{k}/{fk.replace('.', '/')}"] = float(fv)
                else:
                    params[f"{k}.{fk}"] = _scalar(fv)
        else:
            params[k] = _scalar(v)
    return params, ({None: metrics} if metrics else {})


PARSERS = {
    "ultralytics_args": parse_ultralytics_args,
    "ultralytics_results": parse_ultralytics_results,
    "paddle_config": parse_paddle_config,
    "ocr_training_report": parse_ocr_training_report,
    "staff_json": parse_staff_json,
}


# -- planning (no MLflow) -----------------------------------------------------------------------------------------
@dataclass
class Plan:
    entry: Entry
    paths: dict[str, Path]                       # catalog path -> resolved path, for every file the entry needs
    sha256: dict[str, str] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def key(self) -> str | None:
        first = self.entry.files[0]
        return f"{self.entry.capability}:{self.sha256[first]}" if first in self.sha256 else None


def resolve(p: str, root: Path) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else root / path


def plan_entry(entry: Entry, root: Path) -> Plan:
    """Check that everything the entry needs is here and is what the repo says it is. Hashes the model files."""
    needed = [*entry.files, *(r["file"] for r in entry.records), *entry.artifacts]
    plan = Plan(entry, {p: resolve(p, root) for p in needed})
    missing = [p for p, path in plan.paths.items() if not path.is_file()]
    if missing:
        hint = [p for p in missing if resolve(p, root).with_name(resolve(p, root).name + ".dvc").is_file()]
        plan.problems.append(f"not present locally: {', '.join(missing)}"
                             + (f" (DVC pointers exist: `dvc pull` {', '.join(h + '.dvc' for h in hint)})"
                                if hint else ""))
        return plan
    for p in entry.files:
        path = plan.paths[p]
        plan.sha256[p] = _lineage.sha256_file(path)
        pinned = entry.expected_sha256.get(p)
        if pinned and pinned != plan.sha256[p]:
            plan.problems.append(f"{p}: sha256 {plan.sha256[p][:12]}... does not match the pinned {pinned[:12]}...")
        pointer = path.with_name(path.name + ".dvc")
        if pointer.is_file():
            md5 = _lineage.parse_dvc_pointer(pointer)["md5"]
            if not md5.endswith(".dir") and md5 != _md5(path):
                plan.problems.append(f"{p}: the local file is not the version its .dvc pointer names "
                                     f"(md5 {md5}); run `dvc checkout {pointer.relative_to(root) if pointer.is_relative_to(root) else pointer}`")
    for r in entry.records:
        if r["kind"] == "staff_json":                 # the .json names the sha256 of the model it describes
            declared = json.loads(plan.paths[r["file"]].read_text(encoding="utf-8")).get("sha256")
            first = entry.files[0]
            if declared and declared != plan.sha256.get(first):
                plan.problems.append(f"{first}: sha256 does not match the one its {r['file']} declares")
    return plan


def _md5(path: Path) -> str:
    h = hashlib.md5()                                 # noqa: S324 (DVC's content hash, not security)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_records(plan: Plan) -> Parsed:
    params: dict[str, str] = {}
    steps: dict[int | None, dict[str, float]] = {}
    for r in plan.entry.records:
        p, s = PARSERS[r["kind"]](plan.paths[r["file"]])
        params.update(p)
        for step, metrics in s.items():
            steps.setdefault(step, {}).update(metrics)
    return params, steps


def provenance_tags(entry: Entry) -> dict[str, str]:
    tags = {"source": entry.source, "capability": entry.capability, "import.catalog_entry": entry.id,
            "provenance.status": entry.provenance_status}
    prov = entry.provenance if isinstance(entry.provenance, dict) else {}
    tags["origin"] = str(prov.get("origin", "unknown"))
    if prov.get("notebook"):
        tags["notebook"] = str(prov["notebook"])
    for k, v in prov.items():
        if k not in ("origin", "notebook"):
            tags[f"provenance.{k}"] = "; ".join(map(str, v)) if isinstance(v, list) else str(v)
    if entry.evidence:
        tags["provenance.evidence"] = "; ".join(entry.evidence)
    if entry.notes:
        tags["provenance.notes"] = entry.notes
    return tags


def hidden_sidecars(entries: list[Entry], root: Path) -> list[str]:
    """Sidecars (<file>.mlflow.json) that .gitignore would hide, so `git add` would not pick them up."""
    paths = [resolve(f, root) for e in entries for f in e.files]
    rel = [str(p.relative_to(root)) + ".mlflow.json" for p in paths if p.is_relative_to(root)]
    if not rel:
        return []
    out = subprocess.run(["git", "check-ignore", "--stdin"], cwd=root, input="\n".join(rel), capture_output=True,
                         text=True)
    return out.stdout.split()


# -- import -------------------------------------------------------------------------------------------------------
@dataclass
class Result:
    id: str
    status: str               # imported | exists | skipped | would-import
    run_id: str | None = None
    detail: str = ""


def existing_runs(client, key: str) -> list:
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        return []
    return client.search_runs([exp.experiment_id],
                              filter_string=f"tags.`import.key` = '{key}' and attributes.status = 'FINISHED'")


def import_entry(plan: Plan, root: Path, tracking_uri: str, client, force: bool = False) -> Result:
    entry = plan.entry
    previous = existing_runs(client, plan.key)
    if previous and not force:
        return Result(entry.id, "exists", previous[0].info.run_id, f"already imported ({plan.key[:40]}...)")
    params, steps = parse_records(plan)
    tags = {**provenance_tags(entry), "import.key": plan.key, "import.forced": str(bool(previous)).lower(),
            "model.sha256": plan.sha256[entry.files[0]], "import.step_base": str(STEP_BASE)}
    with tracked_run(EXPERIMENT, entry.id, params=params, data=entry.data or None, lockfile=LOCKFILE, tags=tags,
                     tracking_uri=tracking_uri, root=root) as run:
        if not run.active:
            raise RuntimeError(f"tracking is not available (mode {run.mode}); nothing imported")
        for step in sorted(steps, key=lambda s: -1 if s is None else s):
            run.log_metrics(steps[step], step=step)
        for r in entry.records:
            run.log_artifact(plan.paths[r["file"]], "records")
        for a in entry.artifacts:
            run.log_artifact(plan.paths[a], "records")
        run.log_text(json.dumps({k: v for k, v in entry.__dict__.items()}, indent=2, default=str),
                     "import/catalog_entry.json")
        for f in entry.files:
            log_model_file(plan.paths[f], artifact_path="model", run=run)
        run_id = run.run_id
    for old in previous:
        client.set_tag(old.info.run_id, "import.superseded_by", run_id)
    return Result(entry.id, "imported", run_id, f"{len(params)} params, {len(steps)} metric steps")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", type=Path, default=CATALOG)
    ap.add_argument("--only", nargs="*", default=None, help="entry ids to import (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; no MLflow call, no file written")
    ap.add_argument("--force", action="store_true", help="import again entries that already have a run")
    ap.add_argument("--tracking-uri", default=None, help="default: MLFLOW_TRACKING_URI, else the offline store")
    ap.add_argument("--root", type=Path, default=None, help="repo root the catalog paths are relative to")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    root = (args.root or _lineage.repo_root()).resolve()
    entries = load_catalog(args.catalog)
    if args.only:
        unknown = set(args.only) - {e.id for e in entries}
        if unknown:
            print(f"unknown entry ids: {sorted(unknown)}", file=sys.stderr)
            return 2
        entries = [e for e in entries if e.id in args.only]

    results: list[Result] = []
    plans = [plan_entry(e, root) for e in entries]
    if args.dry_run:
        for p in plans:
            results.append(Result(p.entry.id, "would-import" if p.ok else "skipped", None,
                                  f"key {p.key}" if p.ok else "; ".join(p.problems)))
    else:
        try:
            from mlflow.tracking import MlflowClient  # noqa: PLC0415
        except ImportError:
            print("mlflow is not installed: nothing to import into (use --dry-run to see the plan)", file=sys.stderr)
            return 2
        backend = _backend.resolve(root, args.tracking_uri)
        if backend.mode == "offline":
            print(f"WARNING: importing into the OFFLINE store {backend.tracking_uri}; sync it to the server later",
                  file=sys.stderr)
        client = MlflowClient(tracking_uri=backend.tracking_uri)
        for p in plans:
            if not p.ok:
                results.append(Result(p.entry.id, "skipped", None, "; ".join(p.problems)))
                continue
            try:
                results.append(import_entry(p, root, backend.tracking_uri, client, args.force))
            except Exception as exc:                  # one bad entry must not stop the others
                results.append(Result(p.entry.id, "failed", None, f"{type(exc).__name__}: {exc}"))

    width = max(len(r.id) for r in results) if results else 10
    for r in results:
        print(f"{r.status:<13} {r.id:<{width}}  {r.run_id or '-':<32}  {r.detail}")
    hidden = hidden_sidecars(entries, root)
    if hidden:
        print(f"\n{len(hidden)} sidecar(s) would be hidden by .gitignore (for the registry task):")
        for h in hidden:
            print(f"  {h}")
    counts = {s: sum(r.status == s for r in results) for s in ("imported", "exists", "would-import", "skipped",
                                                                 "failed")}
    print("\n" + ", ".join(f"{v} {k}" for k, v in counts.items() if v))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
