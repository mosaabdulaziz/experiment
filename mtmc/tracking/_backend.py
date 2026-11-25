"""Which tracking store a run uses: the server, the offline store, or none (mlflow not installed).

Training must never wait on, or fail because of, tracking (D10, AC6):

- An http(s) tracking URI is probed with a short GET <uri>/health before anything else talks to it. Unreachable or
  unhealthy: the run goes to the offline store instead.
- MLflow's own HTTP client retries for minutes by default (MLFLOW_HTTP_REQUEST_MAX_RETRIES=7 with backoff,
  MLFLOW_HTTP_REQUEST_TIMEOUT=120 s in mlflow 3.18, mlflow/environment_variables.py). For a server that dies mid-run,
  those are lowered for this process unless the caller set them, so a failing call costs seconds, not minutes.
- The offline store is sqlite at <repo>/mlruns-offline/mlflow.db with artifacts beside it (the D10 path). MLflow 3.18
  refuses the plain `file:` store unless MLFLOW_ALLOW_FILE_STORE=true ("maintenance mode"), so sqlite it is: still a
  local file, and the backend MLflow supports and can export from for a later sync.
"""
from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("mtmc.tracking")

OFFLINE_DIR_ENV = "IRIS_MLFLOW_OFFLINE_DIR"      # override the offline store's directory (tests, other checkouts)
PROBE_TIMEOUT_S = 3.0

# Applied with setdefault: a value the caller exported wins.
HTTP_ENV_DEFAULTS = {
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "2",
    "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR": "1",
    "MLFLOW_HTTP_REQUEST_TIMEOUT": "10",
}


@dataclass(frozen=True)
class Backend:
    mode: str                     # "server" | "offline" | "local" (an explicit non-http URI, e.g. sqlite in tests)
    tracking_uri: str
    artifact_root: str | None     # where new experiments put artifacts; None = the server decides
    reason: str = ""


def offline_dir(root: Path) -> Path:
    return Path(os.environ.get(OFFLINE_DIR_ENV) or root / "mlruns-offline")


def sqlite_backend(directory: Path, mode: str, reason: str = "") -> Backend:
    directory.mkdir(parents=True, exist_ok=True)
    return Backend(mode, f"sqlite:///{(directory / 'mlflow.db').resolve()}",
                   (directory / "artifacts").resolve().as_uri(), reason)


def probe(uri: str, timeout: float = PROBE_TIMEOUT_S) -> str | None:
    """None when `uri` answers GET /health with 200 within `timeout`; otherwise why not."""
    url = uri.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:   # noqa: S310 (http(s) only, checked by caller)
            return None if resp.status == 200 else f"{url} answered HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        return f"{url} answered HTTP {exc.code}"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return f"{url} unreachable: {getattr(exc, 'reason', exc)}"


def resolve(root: Path, tracking_uri: str | None = None, timeout: float = PROBE_TIMEOUT_S) -> Backend:
    """Pick the store for a run, before any MLflow call is made."""
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or ""
    if not uri:
        directory = offline_dir(root)
        logger.warning("tracking: MLFLOW_TRACKING_URI is not set; logging OFFLINE to %s "
                       "(set it to the team server, or the dev server from docker-compose.mlflow.yml)", directory)
        return sqlite_backend(directory, "offline", "MLFLOW_TRACKING_URI not set")
    if uri.startswith(("http://", "https://")):
        for k, v in HTTP_ENV_DEFAULTS.items():
            os.environ.setdefault(k, v)
        why = probe(uri, timeout)
        if why is None:
            return Backend("server", uri, None)
        directory = offline_dir(root)
        logger.warning("tracking: MLflow server %s is not reachable (%s). Logging OFFLINE to %s; training continues. "
                       "Sync the run to the server later.", uri, why, directory)
        return sqlite_backend(directory, "offline", why)
    if uri.startswith("sqlite:///"):
        db = Path(uri[len("sqlite:///"):])
        # A child process (e.g. the export run from `--export`) inherits the offline store's URI: still offline.
        mode = "offline" if db.parent.resolve() == offline_dir(root).resolve() else "local"
        return Backend(mode, uri, (db.parent / "artifacts").resolve().as_uri(),
                       "inherited the offline store" if mode == "offline" else "")
    return Backend("local", uri, None)
