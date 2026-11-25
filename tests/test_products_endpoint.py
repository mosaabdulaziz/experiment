"""GET /api/products: which products this installation shows.

The dashboard is one build for every customer; what a given install shows is
decided at run time from IRIS_PRODUCT (MULTI_PRODUCT_PLAN.md decision 10, and
decision 14: it accepts a list). Unset means "everything this build has" --
the dashboard fails open, because an install that cannot be told what it sells
is better showing its products than showing an empty sidebar. It is a
convenience for the menu, never access control.
"""
from fastapi.testclient import TestClient

from app.server import create_app


def _cfg(tmp_path):
    return {
        "perception": {"backend": "local"},
        "api": {"backend": "cloud", "api_key": "x", "model": "m"},
        "monitoring": {"mode": "proactive"}, "sampling": {},
        "events": {"db_path": str(tmp_path / "e.db")},
        "chat": {"tools_enabled": False}, "cv": {"enabled": False}, "camera": {"source": 0},
    }


def _products(tmp_path, value, monkeypatch):
    if value is None:
        monkeypatch.delenv("IRIS_PRODUCT", raising=False)
    else:
        monkeypatch.setenv("IRIS_PRODUCT", value)
    with TestClient(create_app(_cfg(tmp_path))) as client:
        r = client.get("/api/products")
        assert r.status_code == 200
        return r.json()["products"]


def test_a_list_is_returned_in_order(tmp_path, monkeypatch):
    assert _products(tmp_path, "attendance,barber", monkeypatch) == ["attendance", "barber"]


def test_one_product(tmp_path, monkeypatch):
    assert _products(tmp_path, "barber", monkeypatch) == ["barber"]


def test_unset_means_everything_the_build_has(tmp_path, monkeypatch):
    # An empty list is the shell's signal to show what it was built with.
    assert _products(tmp_path, None, monkeypatch) == []


def test_spacing_and_empty_entries_are_tolerated(tmp_path, monkeypatch):
    # People edit .env by hand; "attendance, barber," must not become a product
    # named " barber" or an empty one that matches nothing in the registry.
    assert _products(tmp_path, " attendance , barber , ", monkeypatch) == ["attendance", "barber"]
