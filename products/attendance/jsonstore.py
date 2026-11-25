"""Generic hot-reloading JSON file store (2026-09-02, extracted from
productivity.py). Owned by neither feature -- productivity.py (desk/zone
monitoring) and sessions.py (attendance schedule/gap settings) each keep
their own JSON file and their own DEFAULT_* dict, but share this one small
utility class rather than one importing the other's module for it.
"""
from __future__ import annotations

import json
from pathlib import Path


class JsonFileStore:
    """Small hot-reloading JSON file (policy / zones): re-read when its mtime
    changes, so hand edits AND API writes both take effect within a cycle."""

    def __init__(self, path: Path, default):
        self.path = path
        self._default = default
        self._mtime = None
        self._data = None

    def load(self):
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
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
