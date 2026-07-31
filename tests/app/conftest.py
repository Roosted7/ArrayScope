"""Shared fixtures for the harness's own tests."""

from __future__ import annotations

import pytest

from tests import conftest as harness


@pytest.fixture
def as_the_controller(monkeypatch):
    """Run controller-only reporting code from a test that lives on a worker.

    ``_report_map_erosion``, ``_report_function_coverage``,
    ``_report_sweep_cadence`` and ``_refuse_since_on_unattributed_map`` all
    return immediately on an xdist worker, because one process must own the
    sidecar and the terminal. Tests of those paths execute *on* a worker, so
    without this they assert against a function that declined to act — passing
    under ``-n 0`` and failing under ``-n auto``, which is exactly the shape
    the suite's own docs warn about. Production never hits it: every one of
    those callers really is controller-only.
    """

    monkeypatch.setattr(harness, "_XDIST_WORKER", None)
