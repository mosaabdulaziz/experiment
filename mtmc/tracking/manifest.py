"""Generate a deployment's model manifest from the registry's production aliases (sub-task 8 of the MLflow story).

    python -m mtmc.tracking.manifest generate --product barber --site bekky      # writes products/barber/models.manifest.bekky.json
    python -m mtmc.tracking.manifest generate --product multicam --out /tmp/m.json
    python -m mtmc.tracking.manifest files --product barber --site bekky         # what it would cover (no MLflow)

A deployment (registry_conventions.yaml `deployments:`) is one sidecar yaml plus, where the product has one, its
service's yaml. The manifest covers every model file it can load: each role of the sidecar yaml, each setting switch
(Module.model_files with every_variant: barber's staff_model classifier/clip, a tracker ReID net that is off), and the
service's own models (products/<product>/model_files.py). The startup verifier (perception/engine/model_manifest.py)
then checks the files the running configuration actually loads.

For each file: its registered model is the one its `<file>.mlflow.json` sidecar names, else the import catalog entry
that lists it; the version is the one production-<product>-<site> points at, else production-<product>; the file must
be part of that version (its `model_file.<name>.sha256` tag), and the LOCAL file must have exactly that sha256, so a
manifest is only written from a checkout that has the registered bytes (the message says which dvc checkout/pull is
needed). Any problem refuses the whole manifest: none is written half-filled.

The store is --tracking-uri, else MLFLOW_TRACKING_URI, with no offline fallback (as for the registry).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import SIDECAR_SUFFIX, _lineage
from .import_models import CATALOG, load_catalog
from .registry import CONVENTIONS, Conventions, RegistryError, client_for, load_conventions, parse_alias

SCHEMA = 1
MODEL_FILE_TAG = "model_file.{}.sha256"


class ManifestRefused(RegistryError):
    """The manifest cannot be generated; the message lists every problem."""


@dataclass(frozen=True)
class Deployment:
    product: str
    site: str | None
    sidecar: str
    service: str | None
    manifest: str

    @property
    def label(self) -> str:
        return f"{self.product}-{self.site}" if self.site else self.product


def load_deployments(path: Path = CONVENTIONS) -> list[Deployment]:
    import yaml                                   # noqa: PLC0415
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [Deployment(d["product"], d.get("site"), d["sidecar"], d.get("service"), d["manifest"])
            for d in raw.get("deployments") or []]


def find_deployment(product: str, site: str | None, path: Path = CONVENTIONS) -> Deployment:
    deps = load_deployments(path)
    for d in deps:
        if d.product == product and d.site == site:
            return d
    known = ", ".join(f"--product {d.product}" + (f" --site {d.site}" if d.site else "") for d in deps)
    raise ManifestRefused(f"no deployment {product}{'/' + site if site else ''} in {path.name} (known: {known})")


# -- what a deployment loads (no MLflow) --------------------------------------------------------------------------
def _engine(root: Path):
    """The engine's pure-Python pieces (no gi/pyds): config loading and the model reference collection."""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from perception.engine import model_manifest  # noqa: PLC0415
    from perception.engine.model_check import engine_model_refs  # noqa: PLC0415
    from perception.engine.sources import apply_role_overrides, load_sidecar_config  # noqa: PLC0415
    return model_manifest, engine_model_refs, apply_role_overrides, load_sidecar_config


def _product_helper(root: Path, product: str):
    path = root / "products" / product / "model_files.py"
    spec = importlib.util.spec_from_file_location(f"_model_files_{product}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def deployment_files(dep: Deployment, root: Path) -> tuple[dict[str, str], list[str]]:
    """({manifest key: what loads it}, problems) over every role and setting switch of the sidecar yaml, plus the
    service's own models. A model-pack directory that is not on this machine stays as `<dir>/*.onnx`."""
    mm, engine_model_refs, apply_role_overrides, load_sidecar_config = _engine(root)
    cfg = load_sidecar_config(str(root / dep.sidecar))
    files: dict[str, str] = {}
    problems: list[str] = []
    for role in [None, *sorted(cfg.get("role_overrides") or {})]:
        f, p = mm.expand(engine_model_refs(apply_role_overrides(cfg, role) if role else cfg, every_variant=True), root)
        for k, v in f.items():
            files.setdefault(k, v)
        problems += [x for x in p if x not in problems]
    if dep.service:
        for path, via in _product_helper(root, dep.product).deployment_model_files(root / dep.service):
            files.setdefault(mm.manifest_key(path, root), via)
    return files, problems


def named_manifests(dep: Deployment, root: Path) -> dict[str, str | None]:
    """What the deployment's yamls name in `model_manifest:` (they must all be dep.manifest)."""
    import yaml                                   # noqa: PLC0415
    _, _, _, load_sidecar_config = _engine(root)
    out = {dep.sidecar: load_sidecar_config(str(root / dep.sidecar)).get("model_manifest")}
    if dep.service:
        def read(path: Path) -> dict:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            base = raw.pop("extends", None)
            return {**read(path.parent / base), **raw} if base else raw
        out[dep.service] = read(root / dep.service).get("model_manifest")
    return out


# -- generating (MLflow) ------------------------------------------------------------------------------------------
def _capabilities_by_file(catalog: Path) -> dict[str, str]:
    out = {}
    for e in load_catalog(catalog):
        for f in e.files:
            out.setdefault(Path(f).as_posix(), e.capability)
    return out


def _registered_name(key: str, real: Path, by_catalog: dict[str, str]) -> str | None:
    sidecar = real.with_name(real.name + SIDECAR_SUFFIX)
    if sidecar.is_file():
        try:
            name = json.loads(sidecar.read_text(encoding="utf-8")).get("registered_model")
        except (OSError, ValueError):
            name = None
        if name:
            return name
    return by_catalog.get(key)


def _resolve_version(client, conv: Conventions, name: str, dep: Deployment):
    """(version, alias) via production-<product>-<site>, else production-<product>; None when neither is set."""
    from mlflow.exceptions import MlflowException  # noqa: PLC0415
    aliases = ([parse_alias(f"production-{dep.product}-{dep.site}", conv)] if dep.site else []) \
        + [parse_alias(f"production-{dep.product}", conv)]
    for alias in aliases:
        try:
            return client.get_model_version_by_alias(name, alias), alias
        except MlflowException:
            continue
    return None, " or ".join(aliases)


def generate(client, dep: Deployment, root: Path, conv: Conventions, catalog: Path = CATALOG,
             tracking_uri: str = "") -> dict:
    """The manifest dict for `dep`, or ManifestRefused listing every problem."""
    mm, *_ = _engine(root)
    files, problems = deployment_files(dep, root)
    by_catalog = _capabilities_by_file(catalog)
    for key in [k for k in files if k.endswith("/*.onnx")]:      # a pack not on this machine: the catalog's files
        prefix = key[:-len("*.onnx")]
        listed = sorted(k for k in by_catalog if k.startswith(prefix))
        if listed:
            via = files.pop(key)
            for k in listed:
                files.setdefault(k, via)
    models = []
    for key, via in sorted(files.items()):
        real = mm.resolve(key, root)
        name = _registered_name(key, real, by_catalog)
        if name is None:
            problems.append(f"{key} ({via}) is not a registered model: no {SIDECAR_SUFFIX} names one and the import "
                            "catalog does not list it; register it first")
            continue
        version, alias = _resolve_version(client, conv, name, dep)
        if version is None:
            problems.append(f"{key} ({via}): {name} has no version with alias {alias}; promote one "
                            f"(`python -m mtmc.tracking.registry promote {name} <version> --alias ...`)")
            continue
        expected = version.tags.get(MODEL_FILE_TAG.format(real.name))
        if expected is None:
            problems.append(f"{key} ({via}): {name} v{version.version} ({alias}) does not include {real.name} "
                            f"(its files: {', '.join(sorted(t.split('.', 1)[1].rsplit('.', 1)[0] for t in version.tags if t.startswith('model_file.')))})")
            continue
        pointer = f"{key}.dvc"
        if not real.is_file():
            problems.append(f"{key} ({via}) is not here; the manifest is only written from the registered bytes: run "
                            f"`dvc pull {pointer}`" + ("" if mm.resolve(pointer, root).is_file() else
                                                       " (it has no DVC pointer yet: restore it from its release)"))
            continue
        local = mm.sha256_file(real)
        if local != expected:
            problems.append(f"{key} ({via}) is not {name} v{version.version} ({alias}): local sha256 {local[:12]}..., "
                            f"registered {expected[:12]}...; run `dvc checkout {pointer}` (or `dvc pull {pointer}`) "
                            "for the registered version, or register and promote this file")
            continue
        models.append({"path": key, "registered_model": name, "version": str(version.version), "alias": alias,
                       "run_id": version.run_id, "sha256": expected, "size_bytes": real.stat().st_size,
                       "loaded_by": via})
    if problems:
        raise ManifestRefused(f"model manifest for {dep.label} not generated:\n  - " + "\n  - ".join(problems))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    return {"schema": SCHEMA, "product": dep.product, "site": dep.site,
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "generated_from_commit": commit or "unknown", "tracking_uri": tracking_uri, "models": models}


def write(manifest: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("generate", "files"))
    ap.add_argument("--product", required=True)
    ap.add_argument("--site", default=None)
    ap.add_argument("--out", type=Path, default=None, help="default: the deployment's manifest path")
    ap.add_argument("--tracking-uri", default=None, help="default: MLFLOW_TRACKING_URI (no offline fallback)")
    ap.add_argument("--conventions", type=Path, default=CONVENTIONS)
    ap.add_argument("--catalog", type=Path, default=CATALOG)
    ap.add_argument("--root", type=Path, default=None, help="repo root (default: this checkout)")
    args = ap.parse_args(argv)
    root = (args.root or _lineage.repo_root()).resolve()
    try:
        dep = find_deployment(args.product, args.site, args.conventions)
        if args.cmd == "files":
            files, problems = deployment_files(dep, root)
            for k, v in sorted(files.items()):
                print(f"{k}  <- {v}")
            for p in problems:
                print(f"problem: {p}", file=sys.stderr)
            return 1 if problems else 0
        conv = load_conventions(args.conventions)
        client = client_for(args.tracking_uri)
        uri = args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "")
        manifest = generate(client, dep, root, conv, args.catalog, uri)
    except RegistryError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    out = args.out or root / dep.manifest
    write(manifest, out)
    print(f"wrote {out} ({len(manifest['models'])} model files)")
    for m in manifest["models"]:
        print(f"  {m['registered_model']} v{m['version']:<3} {m['alias']:<30} {m['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
