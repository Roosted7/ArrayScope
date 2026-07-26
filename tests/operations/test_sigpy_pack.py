"""The demoted SigPy pack keeps a lazy probe but registers no operations."""

from __future__ import annotations

import builtins
import sys

import pytest

from arrayscope.operations import plugins, registry
from arrayscope.operations.packs import sigpy_pack


@pytest.fixture(autouse=True)
def _clean_pack_state():
    registry._reset_operation_packs()
    plugins._reset_plugin_cache()
    yield
    registry._reset_operation_packs()
    plugins._reset_plugin_cache()


def test_sigpy_pack_has_no_specs_or_registered_operations(monkeypatch):
    monkeypatch.setattr(sigpy_pack, "sigpy_available", lambda: True)

    assert sigpy_pack.pack_specs() == ()
    assert sigpy_pack.register() is False
    assert not any(entry.id.startswith("sigpy:") for entry in registry.all_operations())


def test_sigpy_availability_probe_never_imports_sigpy(monkeypatch):
    sys.modules.pop("sigpy", None)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sigpy" or name.startswith("sigpy."):
            raise AssertionError("availability probe imported sigpy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert isinstance(sigpy_pack.sigpy_available(), bool)
    assert "sigpy" not in sys.modules
