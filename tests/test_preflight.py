# =============================================================================
# tests/test_preflight.py
# Tests for preflight failure messaging.
# =============================================================================

from __future__ import annotations

import importlib

import pytest

from src.evaluation.preflight import ensure_dependency


def test_ensure_dependency_raises_actionable_error(monkeypatch):
    real_import_module = importlib.import_module

    def _fake_import(name, package=None):
        if name == "httpx":
            raise ModuleNotFoundError(name="httpx")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import)

    with pytest.raises(RuntimeError) as excinfo:
        ensure_dependency("httpx", "pip install httpx")

    assert "pip install httpx" in str(excinfo.value)
