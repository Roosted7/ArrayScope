"""What ``make_backend_window`` does when the surface it asked for is not the
surface it got.

This is a guard on the suite's own gate, and it exists because the previous
answer was wrong in the direction that hides itself. The helper skipped, with a
message the run had already disproved three lines earlier, and a call-phase skip
in a suite with seventy of them is invisible -- so the same fallback happened on
three different tests across eleven full passes without anyone learning why once.

Both halves are asserted here, because either one alone leaves the class where
it was: that it *fails* rather than skips, and that the failure carries the
factory's recorded cause. A loud failure nobody can act on would just relocate
the mystery.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore

from arrayscope.app.settings_state import ImageRenderingBackendChoice
from tests.ui import helpers


@pytest.fixture
def wgpu_pinned():
    """QSettings pinned to the explicit wgpu backend, restored afterwards."""

    helpers.clear_arrayscope_settings()
    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.WGPU.value)
    settings.sync()
    yield
    helpers.clear_arrayscope_settings()


@pytest.fixture
def broken_wgpu_surface(monkeypatch):
    """Make the wgpu surface build fail the way a real device loss does."""

    from arrayscope.display.backends import wgpu as wgpu_backend

    class BrokenWgpuSurface:
        def __init__(self, **_kwargs):
            raise RuntimeError("injected device loss")

    monkeypatch.setattr(wgpu_backend, "WgpuSurface", BrokenWgpuSurface)


def test_a_degraded_wgpu_window_fails_and_names_the_cause(
    qtbot, monkeypatch, wgpu_pinned, broken_wgpu_surface
):
    """Past the adapter gate, a PyQtGraph surface is a defect, not an absence.

    The gate is stubbed rather than relied on, and that is the whole point of
    the test: it pins the branch that runs *after* ``require_wgpu_adapter``
    has said yes, which is the branch that used to skip. Stubbing it also means
    this test asserts the same thing on a machine with no Vulkan at all, where
    the real gate would take the run away before reaching the question.
    """

    monkeypatch.setattr(helpers, "require_wgpu_adapter", lambda: None)
    data = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)

    # Deliberately not ``pytest.raises``. The regression this guards against is
    # the helper going back to *skipping*, and a skip escaping a ``raises``
    # block skips this test too -- so the one guard on the disease would catch
    # the disease and then exhibit it. Every non-failure outcome is turned into
    # a failure here instead, which is the only shape a run reports.
    try:
        helpers.make_backend_window(qtbot, data, backend="wgpu")
    except pytest.fail.Exception as failure:
        message = str(failure)
    except BaseException as other:
        raise AssertionError(
            f"a degraded wgpu window must fail; got {type(other).__name__}: {other}"
        ) from None
    else:
        raise AssertionError("a degraded wgpu window must fail; the helper returned it")

    assert "requested backend 'wgpu'" in message
    assert "'pyqtgraph'" in message
    # The cause, carried all the way from the factory's ``except`` clause. This
    # is the sentence whose absence made eleven passes' worth of observations
    # unusable.
    assert "injected device loss" in message


def test_the_no_adapter_case_still_skips(qtbot, monkeypatch, wgpu_pinned, broken_wgpu_surface):
    """The genuine environment gate keeps its skip, and keeps owning it.

    Making the fallback loud must not make a machine without a GPU red -- that
    case has an owner already, it is upstream, and it answers before any window
    is built. Same broken surface as above; only the gate differs.
    """

    def _no_adapter():
        pytest.skip("no WGPU Vulkan adapter: stubbed")

    monkeypatch.setattr(helpers, "require_wgpu_adapter", _no_adapter)
    data = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)

    with pytest.raises(pytest.skip.Exception) as excinfo:
        helpers.make_backend_window(qtbot, data, backend="wgpu")

    assert "no WGPU Vulkan adapter" in str(excinfo.value)


def test_a_pyqtgraph_window_is_unaffected(qtbot):
    """The PyQtGraph request cannot be degraded, so nothing here can redden it.

    Four call sites ask for this backend and none of them go near the adapter
    gate; the change must be invisible to them.
    """

    helpers.use_pyqtgraph_backend()
    try:
        data = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
        win = helpers.make_backend_window(qtbot, data, backend="pyqtgraph")
        assert win.img_view is not None
    finally:
        helpers.clear_arrayscope_settings()
