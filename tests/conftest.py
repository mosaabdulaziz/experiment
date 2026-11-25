"""Shared pytest configuration for tests/."""
import os
import sqlite3
from pathlib import Path

import pytest


def pytest_configure(config):
    # tests/test_mlflow_e2e.py: the whole MLflow chain, off by default (IRIS_MLFLOW_E2E=1 runs it; MLFLOW.md)
    config.addinivalue_line("markers", "e2e: end-to-end MLflow chain, run only with IRIS_MLFLOW_E2E=1")
    config.addinivalue_line("markers", "mlflow_store_absent: the `mlflow_store` fixture's file is not created (a test "
                                       "asserting that something creates no store)")


# -- the MLflow tests' tracking store -----------------------------------------------------------------------------
# A new sqlite store runs every MLflow schema migration on first open: ~4.5 s, paid by each of ~75 tests (measured
# 2026-11-15: 338 s for the MLflow files). The store is migrated once per session and each test gets its own copy:
# still an empty store of its own per test, opened by the same MLflow, in 0.02 s.

@pytest.fixture(scope="session")
def _migrated_mlflow_db(tmp_path_factory) -> Path | None:
    """An empty, fully migrated sqlite tracking store; None without mlflow (the tests that need it skip anyway)."""
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return None
    where = tmp_path_factory.mktemp("mlflow-template")
    db = where / "mlflow.db"
    here = os.getcwd()
    os.chdir(where)                    # the Default experiment's artifact root is cwd/mlruns: never the repo
    try:
        MlflowClient(tracking_uri=f"sqlite:///{db}").search_experiments()
    finally:
        os.chdir(here)
    return db


@pytest.fixture
def mlflow_store(request, tmp_path, _migrated_mlflow_db) -> str:
    """sqlite:///<tmp_path>/store/mlflow.db: this test's own empty store, a copy of the migrated one.

    Marked `mlflow_store_absent`: only the URI, no file, for a test asserting that nothing creates the store."""
    db = tmp_path / "store" / "mlflow.db"
    if _migrated_mlflow_db is not None and request.node.get_closest_marker("mlflow_store_absent") is None:
        db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(_migrated_mlflow_db) as src, sqlite3.connect(db) as dst:
            src.backup(dst)
            # as a store created here would have it, not the template's directory
            dst.execute("UPDATE experiments SET artifact_location = ? WHERE experiment_id = 0",
                        (str(tmp_path / "mlruns" / "0"),))
        dst.close()
        src.close()
    return f"sqlite:///{db}"
