"""Register model files as MLflow model versions and point product aliases at them (sub-task 7 of the MLflow story).

    python -m mtmc.tracking.registry register --model perception/models/staff/staff_dinov2_s.onnx   # run from sidecar
    python -m mtmc.tracking.registry register --run-id <run> --name vehicle-detector [--file best.onnx]
    python -m mtmc.tracking.registry promote staff-classifier 3 --alias production-barber-bekky        # rollback: older v
    python -m mtmc.tracking.registry baseline [--dry-run]                                           # what main loads

The store is --tracking-uri, else MLFLOW_TRACKING_URI. Unlike the training helper, the registry never falls back to the
offline store: a version registered there would be invisible to the team, so an unreachable server is an error.

Conventions (registered names, products and sites, alias format, what counts as an evaluation metric, the baseline)
live in registry_conventions.yaml beside this file.

How a plain file becomes a version: models here are .onnx/.pth files logged by `log_model_file`, not MLflow "models"
with an MLmodel file. In mlflow 3.18 `mlflow.register_model("runs:/<run>/<path>", name)` refuses those (it looks for an
MLmodel file, then for an MLflow 3 LoggedModel of that run, and raises NOT_FOUND, after creating the registered model;
mlflow/tracking/_model_registry/fluent.py `_register_model`). `MlflowClient.create_model_version(name,
source="runs:/<run>/<artifact path>/<file>", run_id=<run>)` takes any artifact, so that is what `register` calls, after
its own checks. The version's source is the file itself (not its directory), so it is unambiguous when a run logged
several files; fetch the bytes with `mlflow.artifacts.download_artifacts(version.source)`. (`models:/<name>/<v>` URIs
download a *directory* and come back empty for a file source; nothing here relies on them, and the pipeline keeps
loading files from disk.)

register: refuses unless the run is FINISHED, carries its lineage (git.commit, env.lockfile*, data.* for trained runs)
and the model file's sha256 tag (`model_file.<file>.sha256`, written by log_model_file). source=trained runs (no
`source` tag = trained) also need evaluation metrics. Imported and Kaggle runs may lack both: the version is then tagged
`metrics=none` / `data=none` instead (a third-party model has no dataset in this repo). The version's tags copy the run's
lineage, provenance and a metrics summary. Idempotent on (name, sha256): a version with the same file is returned, not
duplicated. With a local model file, its `<file>.mlflow.json` sidecar gets `registered_model` and `version`.

promote: sets an alias on a version and prints the version it pointed to before. No history is kept here.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import SIDECAR_SUFFIX, _backend, _lineage, read_model_sidecar
from ._lineage import sha256_file

logger = logging.getLogger("mtmc.tracking.registry")

CONVENTIONS = Path(__file__).with_name("registry_conventions.yaml")
SOURCES = ("trained", "imported", "kaggle")
CANDIDATE = "candidate"
PRODUCTION = "production"
_MLFLOW_ALIAS = re.compile(r"^[\w\-]{1,255}$")          # mlflow/utils/validation.py _validate_model_alias_name
_MODEL_FILE_TAG = re.compile(r"^model_file\.(.+)\.sha256$")
_COPIED_TAG_PREFIXES = ("data.", "provenance.", "model_file.", "env.lockfile")
_COPIED_TAGS = ("source", "origin", "notebook", "git.commit", "git.branch", "git.dirty", "import.catalog_entry",
                "import.key", "import.step_base", "capability")


class RegistryError(Exception):
    """A refusal: the message says what is missing or wrong."""


# -- conventions --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Conventions:
    products: dict[str, tuple[str, ...]]          # product -> its sites (empty: no per-site aliases)
    capabilities: dict[str, tuple[str, ...]]      # registered name -> products that load it
    evaluation_metrics: re.Pattern
    baseline: dict[str, tuple[str, ...]]          # import catalog entry id -> canonical aliases

    def is_eval_metric(self, key: str) -> bool:
        return bool(self.evaluation_metrics.search(key))


def _norm(s: str) -> str:
    return s.strip().lower().replace("_", "-")


def load_conventions(path: Path = CONVENTIONS) -> Conventions:
    import yaml                                   # noqa: PLC0415 (research environment dependency)

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    products = {p: tuple((v or {}).get("sites") or ()) for p, v in (raw.get("products") or {}).items()}
    for p, sites in products.items():
        for name in (p, *sites):
            if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
                raise RegistryError(f"{path}: {name!r} is not a valid product/site name (lowercase, digits, _ and -)")
    normed = [_norm(p) for p in products]
    if len(set(normed)) != len(normed):
        raise RegistryError(f"{path}: two products differ only by case or _/-: {sorted(products)}")
    caps = {c: tuple((v or {}).get("loaded_by") or ()) for c, v in (raw.get("capabilities") or {}).items()}
    for c, users in caps.items():
        if unknown := set(users) - set(products):
            raise RegistryError(f"{path}: capability {c} is loaded_by unknown products {sorted(unknown)}")
    conv = Conventions(products, caps, re.compile(raw.get("evaluation_metrics") or r"$^"), {})
    baseline = {}
    for entry_id, aliases in (raw.get("baseline") or {}).items():
        baseline[entry_id] = tuple(parse_alias(a, conv) for a in aliases or ())
    return Conventions(products, caps, conv.evaluation_metrics, baseline)


def parse_alias(text: str, conv: Conventions) -> str:
    """The canonical alias for `text`: `candidate`, `production-<product>` or `production-<product>-<site>`.

    Case and `_`/`-` are normalised to the conventions' spelling (`Production-Fall-Detection` ->
    `production-fall_detection`); an unknown product, an unknown site, or a site on a product without sites is refused.
    """
    if not _MLFLOW_ALIAS.fullmatch(text.strip()):
        raise RegistryError(f"alias {text!r}: MLflow aliases may only contain letters, digits, _ and - (max 255)")
    t = _norm(text)
    if t == CANDIDATE:
        return CANDIDATE
    if not t.startswith(PRODUCTION + "-"):
        raise RegistryError(f"alias {text!r}: must be `{CANDIDATE}` or `{PRODUCTION}-<product>[-<site>]`")
    rest = t[len(PRODUCTION) + 1:]
    for product in sorted(conv.products, key=len, reverse=True):          # longest first: no prefix ambiguity
        p = _norm(product)
        if rest == p:
            alias = f"{PRODUCTION}-{product}"
            break
        if rest.startswith(p + "-"):
            site = rest[len(p) + 1:]
            sites = {_norm(s): s for s in conv.products[product]}
            if site not in sites:
                known = ", ".join(conv.products[product]) or "none: it has no per-site aliases"
                raise RegistryError(f"alias {text!r}: unknown site {site!r} for {product} (sites: {known})")
            alias = f"{PRODUCTION}-{product}-{sites[site]}"
            break
    else:
        raise RegistryError(f"alias {text!r}: unknown product (products: {', '.join(sorted(conv.products))})")
    if not _MLFLOW_ALIAS.fullmatch(alias):                                 # cannot happen with validated names
        raise RegistryError(f"alias {alias!r}: not a valid MLflow alias")
    return alias


def alias_product(alias: str, conv: Conventions) -> str | None:
    """The product a canonical production alias belongs to; None for `candidate`."""
    if alias == CANDIDATE:
        return None
    rest = alias[len(PRODUCTION) + 1:]
    return next(p for p in sorted(conv.products, key=len, reverse=True) if rest == p or rest.startswith(p + "-"))


def check_name(name: str, conv: Conventions) -> None:
    if name not in conv.capabilities:
        raise RegistryError(f"{name!r} is not a registered model name (D2): {', '.join(conv.capabilities)}")


# -- the store ----------------------------------------------------------------------------------------------------
def client_for(tracking_uri: str | None = None):
    """An MlflowClient on the given store. No offline fallback: a server that does not answer is an error."""
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RegistryError("no tracking store: pass --tracking-uri or set MLFLOW_TRACKING_URI "
                            "(the registry never registers into the offline store)")
    if uri.startswith(("http://", "https://")):
        for k, v in _backend.HTTP_ENV_DEFAULTS.items():
            os.environ.setdefault(k, v)
        if why := _backend.probe(uri):
            raise RegistryError(f"MLflow server not available: {why}")
    try:
        from mlflow.tracking import MlflowClient  # noqa: PLC0415
    except ImportError as exc:
        raise RegistryError(f"mlflow is not installed ({exc})") from exc
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def _mlflow_error():
    from mlflow.exceptions import MlflowException  # noqa: PLC0415
    return MlflowException


# -- register -----------------------------------------------------------------------------------------------------
@dataclass
class Candidate:
    """Everything `register` decided before writing anything."""
    name: str
    run_id: str
    file: str
    artifact: str                                 # path of the file inside the run's artifacts
    sha256: str
    source: str
    tags: dict[str, str]                          # the version's tags
    eval_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def source_uri(self) -> str:
        return f"runs:/{self.run_id}/{self.artifact}"


@dataclass
class Registered:
    name: str
    version: str                                  # MLflow returns an int or a str depending on the store
    created: bool
    sha256: str
    run_id: str
    sidecar: Path | None = None
    note: str = ""
    companions: list[Path] = field(default_factory=list)   # sidecars of the version's other files, updated too


def _model_file(tags: dict[str, str], run_id: str, file: str | None) -> tuple[str, str]:
    """(file name, sha256) of the model file in the run, from log_model_file's `model_file.<file>.sha256` tags."""
    files = {m.group(1): v for k, v in tags.items() if (m := _MODEL_FILE_TAG.match(k))}
    if not files:
        raise RegistryError(f"run {run_id} has no model file sha256 (no `model_file.<file>.sha256` tag: "
                            "the model was not logged with log_model_file)")
    if file:
        if file not in files:
            raise RegistryError(f"run {run_id} logged no model file {file!r} (it logged: {', '.join(sorted(files))})")
        return file, files[file]
    if len(files) == 1:
        return next(iter(files.items()))
    main = tags.get("model.sha256")                       # the import run names its main file
    picked = [f for f, h in files.items() if h == main]
    if len(picked) == 1:
        return picked[0], main
    raise RegistryError(f"run {run_id} logged several model files ({', '.join(sorted(files))}); pass --file")


def _find_artifact(client, run_id: str, file: str, path: str | None = None, depth: int = 4) -> str | None:
    for a in client.list_artifacts(run_id, path):
        if a.is_dir:
            if depth and (found := _find_artifact(client, run_id, file, a.path, depth - 1)):
                return found
        elif a.path.rsplit("/", 1)[-1] == file:
            return a.path
    return None


def _latest_eval_metrics(run, conv: Conventions) -> dict[str, float]:
    return {k: v for k, v in sorted(run.data.metrics.items()) if conv.is_eval_metric(k)}


def metrics_summary(metrics: dict[str, float]) -> str:
    return "; ".join(f"{k}={v:.6g}" for k, v in metrics.items()) if metrics else "none"


def plan_registration(client, conv: Conventions, name: str, run_id: str, file: str | None = None) -> Candidate:
    """Check a run can become a version of `name` and compute the version's tags. Writes nothing."""
    check_name(name, conv)
    try:
        run = client.get_run(run_id)
    except _mlflow_error() as exc:
        raise RegistryError(f"run {run_id} not found in this store ({exc})") from exc
    tags = dict(run.data.tags)
    if run.info.status != "FINISHED":
        raise RegistryError(f"run {run_id} is {run.info.status}, not FINISHED")
    if tags.get("capability") and tags["capability"] != name:
        raise RegistryError(f"run {run_id} is a {tags['capability']} run, not {name}")
    source = tags.get("source", "trained")
    if source not in SOURCES:
        raise RegistryError(f"run {run_id}: source tag {source!r} is not one of {SOURCES}")
    file, sha = _model_file(tags, run_id, file)

    missing = []
    if tags.get("git.commit", "unknown") == "unknown":
        missing.append("git.commit")
    # A trained version must be reproducible from a locked environment (D7): its run names the lockfile it ran from
    # (tools/mtmc-research/uv.lock for the finetune and export scripts). An imported or Kaggle run only has to carry
    # the tag: its model was not produced in this repo's environments, so `missing`/`none` is the honest answer there.
    lockfile = tags.get("env.lockfile")
    if lockfile is None:
        missing.append("env.lockfile")
    elif source == "trained" and lockfile in ("missing", "none"):
        missing.append(f"env.lockfile (it is {lockfile!r}: a trained model must come from a locked environment)")
    has_data = any(k.startswith("data.") for k in tags)
    if not has_data and source == "trained":
        missing.append("data.* (the dataset's .dvc pointer md5)")
    if missing:
        raise RegistryError(f"run {run_id} lacks lineage: {', '.join(missing)}")
    metrics = _latest_eval_metrics(run, conv)
    if not metrics and source == "trained":
        raise RegistryError(f"run {run_id} (source=trained) has no evaluation metrics "
                            f"(none match {conv.evaluation_metrics.pattern!r}); evaluate it before registering")
    artifact = _find_artifact(client, run_id, file)
    if artifact is None:
        raise RegistryError(f"run {run_id} has a sha256 tag for {file} but no such artifact")

    vtags = {k: v for k, v in tags.items() if k in _COPIED_TAGS or k.startswith(_COPIED_TAG_PREFIXES)}
    vtags.update({"run_id": run_id, "source": source, "sha256": sha, "model.file": file, "model.artifact": artifact,
                  "metrics": metrics_summary(metrics)})
    if not has_data:
        vtags["data"] = "none"
    return Candidate(name, run_id, file, artifact, sha, source, vtags, metrics)


def find_version(client, name: str, sha256: str):
    """The oldest version of `name` whose file has this sha256, or None."""
    try:
        versions = client.search_model_versions(f"name = '{name}'")
    except _mlflow_error():
        return None
    same = [v for v in versions if v.tags.get("sha256") == sha256]
    return min(same, key=lambda v: int(v.version)) if same else None


def update_sidecar(model_path: Path, name: str, version: str, sha256: str) -> Path | None:
    """Name the version in the file's sidecar. None when it already names ANOTHER version: one sidecar per file, so
    byte-identical files shared by several versions keep the first registered one (MLFLOW.md, known gaps)."""
    sidecar = model_path.with_name(model_path.name + SIDECAR_SUFFIX)
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    if meta.get("sha256") != sha256:
        raise RegistryError(f"{sidecar} describes sha256 {str(meta.get('sha256'))[:12]}..., not the registered "
                            f"{sha256[:12]}...; not updated")
    if meta.get("registered_model") and (meta["registered_model"], str(meta.get("version"))) != (name, str(version)):
        logger.warning("registry: %s already names %s v%s; kept, not repointed to %s v%s: the file is byte-identical "
                       "and shared by several versions (MLFLOW.md, known gaps)", sidecar, meta["registered_model"],
                       meta.get("version"), name, version)
        return None
    meta["registered_model"], meta["version"] = name, str(version)
    tmp = sidecar.with_name(sidecar.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    tmp.replace(sidecar)
    return sidecar


def register(client, conv: Conventions, name: str, run_id: str, file: str | None = None,
             model_path: Path | None = None) -> Registered:
    """Create (or find) the version of `name` for the run's model file; update the file's sidecar if given.

    `model_path`, when given, must be the same file (sha256) as the run's, with a sidecar beside it: it is checked
    before anything is written.
    """
    cand = plan_registration(client, conv, name, run_id, file)
    if model_path is not None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise RegistryError(f"{model_path}: no such file")
        if (local := sha256_file(model_path)) != cand.sha256:
            raise RegistryError(f"{model_path} (sha256 {local[:12]}...) is not the file run {run_id} logged "
                                f"({cand.sha256[:12]}...)")
        if not model_path.with_name(model_path.name + SIDECAR_SUFFIX).is_file():
            raise RegistryError(f"{model_path} has no {SIDECAR_SUFFIX} sidecar to update")

    existing = find_version(client, name, cand.sha256)
    if existing is not None:
        note = "" if existing.run_id == run_id else f"registered earlier from run {existing.run_id}"
        result = Registered(name, str(existing.version), False, cand.sha256, existing.run_id, note=note)
    else:
        try:
            client.create_registered_model(name, tags={"capability": name,
                                                       "loaded_by": ",".join(conv.capabilities[name])})
        except _mlflow_error() as exc:
            if exc.error_code not in ("RESOURCE_ALREADY_EXISTS", "ALREADY_EXISTS"):
                raise
        v = client.create_model_version(name, cand.source_uri, run_id=run_id, tags=cand.tags,
                                        description=f"{cand.file} from run {run_id} ({cand.source})")
        result = Registered(name, str(v.version), True, cand.sha256, run_id)
    if model_path is not None:
        result.sidecar = update_sidecar(model_path, name, result.version, cand.sha256)
        result.companions = update_companion_sidecars(client, model_path, name, result, run_id)
    return result


def update_companion_sidecars(client, model_path: Path, name: str, result: Registered, run_id: str) -> list[Path]:
    """The version's other files (labels, dicts, a second stream: the run's other `model_file.<file>.sha256` tags)
    that sit beside `model_path` with a sidecar of the SAME run and the same sha256 get the registered name and
    version too, so a manifest can name them without the import catalog (a retrained export's labels.txt).
    Anything else is left alone."""
    tags = client.get_run(result.run_id).data.tags
    updated = []
    for key, sha in sorted(tags.items()):
        m = _MODEL_FILE_TAG.match(key)
        if not m or m.group(1) == model_path.name:
            continue
        other = model_path.with_name(m.group(1))
        sidecar = other.with_name(other.name + SIDECAR_SUFFIX)
        if not (other.is_file() and sidecar.is_file()):
            continue
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if meta.get("run_id") != result.run_id or meta.get("sha256") != sha or sha256_file(other) != sha:
            continue
        if (done := update_sidecar(other, name, result.version, sha)) is not None:
            updated.append(done)
    return updated


# -- promote ------------------------------------------------------------------------------------------------------
def promote(client, conv: Conventions, name: str, version: str | int, alias: str) -> tuple[str, str | None]:
    """Point `alias` at `version` of `name`. Returns (canonical alias, the version it pointed to before or None)."""
    check_name(name, conv)
    alias = parse_alias(alias, conv)
    product = alias_product(alias, conv)
    if product and product not in conv.capabilities[name]:
        raise RegistryError(f"{alias}: {product} does not load {name} (loaded by: "
                            f"{', '.join(conv.capabilities[name])}; see registry_conventions.yaml)")
    try:
        v = client.get_model_version(name, str(version))
    except _mlflow_error() as exc:
        raise RegistryError(f"{name} has no version {version}") from exc
    if v.status != "READY":
        raise RegistryError(f"{name} v{version} is {v.status}, not READY")
    try:
        previous = str(client.get_model_version_by_alias(name, alias).version)
    except _mlflow_error():
        previous = None
    client.set_registered_model_alias(name, alias, str(version))
    return alias, previous


# -- baseline -----------------------------------------------------------------------------------------------------
@dataclass
class BaselineRow:
    entry: str
    status: str                  # registered | exists | would-register | missing | refused | failed
    name: str
    version: str | None = None
    aliases: list[str] = field(default_factory=list)
    detail: str = ""


def import_run(client, entry_id: str):
    """The current import run of a catalog entry: the newest FINISHED one not superseded by a --force import."""
    from .import_models import EXPERIMENT       # noqa: PLC0415
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        return None
    runs = client.search_runs([exp.experiment_id],
                              filter_string=f"tags.`import.catalog_entry` = '{entry_id}' "
                                            "and attributes.status = 'FINISHED'",
                              order_by=["attributes.start_time DESC"])
    current = [r for r in runs if "import.superseded_by" not in r.data.tags]
    return current[0] if current else None


def missing_reason(entry, root: Path) -> str:
    """Why a catalog entry has no import run: which of its files are not here, and whether `dvc pull` can fetch them
    (a .dvc pointer exists) or not (no pointer: the file lives outside DVC, e.g. a file under the home directory)."""
    from .import_models import resolve           # noqa: PLC0415
    absent = []
    for f in dict.fromkeys([*entry.files, *(r["file"] for r in entry.records), *entry.artifacts]):
        path = resolve(f, root)
        if path.is_file():
            continue
        pointer = _lineage.find_dvc_pointer(path)
        if pointer is None:
            absent.append(f"{f} (no DVC pointer)")
        else:
            shown = pointer.relative_to(root) if pointer.is_relative_to(root) else pointer
            absent.append(f"{f} (`dvc pull {shown}`)")
    if absent:
        return "not present locally: " + ", ".join(absent)
    return "files present but not imported yet (run `python -m mtmc.tracking.import_models`)"


def baseline(client, conv: Conventions, catalog: Path, root: Path, dry_run: bool = False) -> list[BaselineRow]:
    """Register every import catalog entry that has an import run and set the conventions' baseline aliases.

    Never stops at an entry: a missing, refused or failing entry gets its row and reason, and the rest go on."""
    from .import_models import load_catalog      # noqa: PLC0415
    entries = load_catalog(catalog)
    if unlisted := [e.id for e in entries if e.id not in conv.baseline]:
        raise RegistryError(f"catalog entries without a baseline line in registry_conventions.yaml: {unlisted}")
    rows = []
    for e in entries:
        aliases = list(conv.baseline[e.id])
        row = BaselineRow(e.id, "", e.capability, aliases=aliases)
        rows.append(row)
        try:
            _baseline_entry(client, conv, e, row, root, dry_run)
        except Exception as exc:                          # one bad entry must not stop the others
            row.status, row.detail = "failed", f"{type(exc).__name__}: {exc}"
    return rows


def _baseline_entry(client, conv: Conventions, e, row: BaselineRow, root: Path, dry_run: bool) -> None:
    from .import_models import resolve           # noqa: PLC0415
    aliases = row.aliases
    for a in aliases:
        p = alias_product(a, conv)
        if p and p not in conv.capabilities.get(e.capability, ()):
            row.status, row.detail = "refused", f"{a}: {p} does not load {e.capability}"
    if row.status:
        return
    run = import_run(client, e.id)
    if run is None:
        row.status, row.detail = "missing", missing_reason(e, root)
        return
    main_file = Path(e.files[0]).name
    local = resolve(e.files[0], root)
    sidecar_ok = local.is_file() and (meta := read_model_sidecar(local)) is not None \
        and meta.get("run_id") == run.info.run_id
    try:
        if dry_run:
            cand = plan_registration(client, conv, e.capability, run.info.run_id, main_file)
            existing = find_version(client, e.capability, cand.sha256)
            row.status = "exists" if existing else "would-register"
            row.version = str(existing.version) if existing else None
            row.detail = f"run {run.info.run_id}, metrics={cand.tags['metrics'][:60]}"
            return
        reg = register(client, conv, e.capability, run.info.run_id, main_file,
                       model_path=local if sidecar_ok else None)
    except RegistryError as exc:
        row.status, row.detail = "refused", str(exc)
        return
    row.status, row.version = ("registered" if reg.created else "exists"), reg.version
    notes = [] if reg.sidecar else [f"sidecar not updated ({e.files[0]} or its sidecar for this run is not here)"]
    for a in aliases:
        try:
            _, prev = promote(client, conv, e.capability, reg.version, a)
        except RegistryError as exc:
            row.status = "refused"
            notes.append(f"{a}: {exc}")
            continue
        if prev is not None and prev != reg.version:
            notes.append(f"{a} was v{prev}")
    row.detail = "; ".join(notes)


# -- CLI ----------------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracking-uri", default=None, help="default: MLFLOW_TRACKING_URI (no offline fallback)")
    ap.add_argument("--conventions", type=Path, default=CONVENTIONS)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register", help="create (or find) a model version from a run's logged model file")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", type=Path, help="a model file with a .mlflow.json sidecar (gives the run and file)")
    src.add_argument("--run-id")
    r.add_argument("--name", help="registered model name (default: the run's capability tag)")
    r.add_argument("--file", help="which logged model file, when the run logged several")
    r.add_argument("--model-path", type=Path, help="with --run-id: the local file whose sidecar to update")

    p = sub.add_parser("promote", help="point an alias at a version (rollback = promote an older version)")
    p.add_argument("name")
    p.add_argument("version")
    p.add_argument("--alias", required=True, help="candidate | production-<product>[-<site>]")

    b = sub.add_parser("baseline", help="register the import catalog's models and set the production aliases")
    b.add_argument("--catalog", type=Path, default=None)
    b.add_argument("--root", type=Path, default=None, help="repo root the catalog paths are relative to")
    b.add_argument("--dry-run", action="store_true", help="check and print; register nothing, set no alias")

    args = ap.parse_args(argv)
    try:
        conv = load_conventions(args.conventions)
        client = client_for(args.tracking_uri)
        if args.cmd == "register":
            return _cli_register(client, conv, args)
        if args.cmd == "promote":
            alias, prev = promote(client, conv, args.name, args.version, args.alias)
            was = f"v{prev}" if prev else "nothing"
            print(f"{alias}: {args.name} v{args.version} (previously {was})"
                  + (" (unchanged)" if prev == str(args.version) else ""))
            return 0
        from .import_models import CATALOG                # noqa: PLC0415
        root = (args.root or _lineage.repo_root()).resolve()
        rows = baseline(client, conv, args.catalog or CATALOG, root, args.dry_run)
    except RegistryError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    width = max(len(r.entry) for r in rows)
    for r in rows:
        print(f"{r.status:<15} {r.entry:<{width}}  v{r.version or '-':<4} {','.join(r.aliases) or '-'}"
              + (f"  {r.detail}" if r.detail else ""))
    counts = {s: sum(r.status == s for r in rows) for s in
              ("registered", "exists", "would-register", "missing", "refused", "failed")}
    print("\n" + ", ".join(f"{v} {k}" for k, v in counts.items() if v))
    problems = [r for r in rows if r.status in ("missing", "refused", "failed")]
    if problems:
        print(f"\nNot registered ({len(problems)}), everything else was:")
        for r in problems:
            print(f"  {r.status:<8} {r.entry}: {r.detail}")
    return 1 if problems else 0


def _cli_register(client, conv: Conventions, args) -> int:
    model_path, run_id, file = args.model_path, args.run_id, args.file
    if args.model:
        meta = read_model_sidecar(args.model)             # verifies the file's sha256 against the sidecar
        if meta is None:
            raise RegistryError(f"{args.model}: no {SIDECAR_SUFFIX} sidecar, or the file does not match it")
        model_path, run_id, file = args.model, meta["run_id"], file or meta["file"]
        if not run_id:
            raise RegistryError(f"{args.model}{SIDECAR_SUFFIX} names no run")
    name = args.name
    if not name:
        try:
            name = client.get_run(run_id).data.tags.get("capability")
        except _mlflow_error() as exc:
            raise RegistryError(f"run {run_id} not found in this store") from exc
        if not name:
            raise RegistryError(f"run {run_id} has no capability tag; pass --name")
    reg = register(client, conv, name, run_id, file, model_path)
    print(f"{'registered' if reg.created else 'exists'}: {reg.name} v{reg.version} (run {reg.run_id}, "
          f"sha256 {reg.sha256[:12]}...)" + (f"; {reg.note}" if reg.note else ""))
    for c in reg.companions:
        print(f"updated {c}")
    if reg.sidecar:
        print(f"updated {reg.sidecar}")
    elif model_path is None:
        print("no local model file given: no sidecar updated (pass --model or --model-path)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
