"""Generic hot-reloading JSON file store.

Forked from products/attendance/jsonstore.py @ 1f5b27b — changes: persist_default (2026-10-15).
"""
from __future__ import annotations

import json
from pathlib import Path


class JsonFileStore:
    """Small hot-reloading JSON file (policy / zones): re-read when its mtime
    changes, so hand edits AND API writes both take effect within a cycle."""

    def __init__(self, path: Path, default, persist_default: bool = True):
        """persist_default: a missing file is written with the default on the first load (zones). False: the
        default is returned and nothing is written until save() (the policy: no file = nothing set)."""
        self.path = path
        self._default = default
        self._persist_default = persist_default
        self._mtime = None
        self._data = None

    def load(self):
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            if not self._persist_default:
                self._mtime, self._data = None, None
                return self._default() if callable(self._default) else self._default
            self.save(self._default() if callable(self._default) else self._default)
            mtime = self.path.stat().st_mtime
        if mtime != self._mtime or self._data is None:
            with open(self.path) as f:
                self._data = json.load(f)
            self._mtime = mtime
        return self._data

    def save(self, data) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)
        self._mtime = None  # force reload
