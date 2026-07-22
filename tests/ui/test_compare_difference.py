"""Compare v1c: difference as a derived source (queue item 7).

``open_difference_window`` opens a THIRD window over ``A - B`` backed by a
``CompositeArraySource`` (wrapped in ``LazySourceArray`` like every other lazy
source). It is linked into the SAME compare group as A and B (dims + camera +
levels), reads region-by-region through the unchanged pipeline (so a
progressive/streaming A still streams), and reports the exact ``A - B`` value
under the linked cursor -- checked against a plain NumPy oracle.

Two ownership hazards are pinned here:

- **Lifecycle:** the composite is built with ``own_inputs=False`` so closing
  the A - B window never tears down A's or B's still-live sources. The
  lifecycle test closes the difference window and asserts A and B still render.

- **Disposal:** every window is fully disposed (close -> drop the app-global
  ``_arrayscope_live_windows`` retention refs -> ``deleteLater`` -> wait on
  ``destroyed``). A half-lived window's deferred deletions crash a LATER
  test's nested event loop, so this discipline is mandatory and matches
  ``tests/ui/test_compare_launcher.py``.
"""

import uuid

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.core.array_source import (
    CompositeArraySource,
    LazySourceArray,
    NdArraySource,
)
from arrayscope.io.progressive import ProgressiveArraySource
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from arrayscope.window.compare_launcher import read_source_value
from tests.ui.helpers import (
    clear_arrayscope_settings,
    committed_value,
    frame_session_settled,
    process_events,
)

pytest.importorskip("pytestqt")

_SETTLE_TIMEOUT_MS = min(4000, INTERACTION_SETTLE_HARD_LIMIT_MS)
_DISPOSE_TIMEOUT_MS = min(5000, INTERACTION_SETTLE_HARD_LIMIT_MS)


def _clear_settings():
    clear_arrayscope_settings()
    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", "pyqtgraph")
    settings.sync()


@pytest.fixture
def isolated_sync(monkeypatch):
    monkeypatch.setenv("ARRAYSCOPE_SYNC_NAME", f"arrayscope-diff-uitest-{uuid.uuid4().hex[:12]}")


def _dispose(qtbot, windows):
    """Fully dispose a set of windows (mirrors the compare_launcher fixture)."""

    app = QtWidgets.QApplication.instance()
    windows = [w for w in windows if w is not None]
    destroyed = {id(win): False for win in windows}
    for win in windows:
        win_key = id(win)
        win.destroyed.connect(lambda *_a, key=win_key: destroyed.__setitem__(key, True))
    for win in windows:
        win.close()
    if app is not None:
        app.setProperty("_arrayscope_live_windows", None)
    for win in windows:
        win.deleteLater()
    qtbot.waitUntil(lambda: all(destroyed.values()), timeout=_DISPOSE_TIMEOUT_MS)


@pytest.fixture
def compare_windows(qtbot):
    """Hand out ArrayScopeWindows and guarantee full teardown after the test."""

    tracked = []

    def track(win):
        assert win is not None
        tracked.append(win)
        return win

    yield track

    # Some tests dispose a window themselves; skip anything already destroyed.
    _dispose(qtbot, [w for w in tracked if not _is_destroyed(w)])


def _is_destroyed(win):
    try:
        win.objectName()
    except RuntimeError:
        return True
    return False


def _settle(qtbot, *windows):
    process_events(qtbot, count=20)
    qtbot.waitUntil(
        lambda: (
            all(frame_session_settled(win) for win in windows)
            and all(win.renderer.display_geometry is not None for win in windows)
        ),
        timeout=_SETTLE_TIMEOUT_MS,
    )


def _hover(win, view_col, view_row):
    scene_pos = win.img_view.getView().mapViewToScene(
        QtCore.QPointF(float(view_col) + 0.25, float(view_row) + 0.25)
    )
    win.getPixel(scene_pos)
    return getattr(win, "_last_compare_array_index", None)


class _ClosableNdSource(NdArraySource):
    """NdArraySource that records whether ``close`` was called."""

    def __init__(self, array, *, label="closable"):
        self.closed = False
        super().__init__(array, label=label, close=self._mark)

    def _mark(self):
        self.closed = True


class _SpyProgressive(ProgressiveArraySource):
    """Record every region spec read, to prove region-only streaming."""

    def __init__(self, array, *, label="A-stream"):
        super().__init__(array, label=label)
        self.read_specs = []

    def read_region(self, index_spec, *, cancellation_token=None):
        self.read_specs.append(tuple(index_spec))
        return super().read_region(index_spec, cancellation_token=cancellation_token)


def test_difference_window_is_linked_and_reports_exact_a_minus_b(
    qtbot, isolated_sync, compare_windows
):
    _clear_settings()
    from arrayscope.window import ArrayScopeWindow

    arr_a = np.arange(6 * 8, dtype=float).reshape(6, 8)
    arr_b = arr_a * -1.0 - 1000.0  # distinct so A - B is discriminating

    win_a = compare_windows(ArrayScopeWindow(arr_a))
    win_b = compare_windows(win_a.open_compare_window(data=arr_b))
    diff = compare_windows(win_a.open_difference_window())
    assert diff is not None

    # Linked into the SAME group: A, B, C, all facets enabled on all three.
    assert (win_a.compare_label, win_b.compare_label, diff.compare_label) == ("A", "B", "C")
    for win in (win_a, win_b, diff):
        for facet in ("dims", "camera", "levels"):
            assert win.sync_controller.facet_enabled(facet), (win.compare_label, facet)

    # base_data is a CompositeArraySource (wrapped in the lazy proxy).
    assert isinstance(diff.base_data, LazySourceArray)
    composite = diff.base_data.source
    assert isinstance(composite, CompositeArraySource)
    assert composite.op == "subtract"
    assert tuple(diff.base_data.shape) == arr_a.shape

    _settle(qtbot, win_a, win_b, diff)

    # Hover in the difference window: source index (2, 3) at view (col=3, row=2).
    idx = _hover(diff, view_col=3, view_row=2)
    assert idx == (2, 3), idx
    oracle = arr_a[idx] - arr_b[idx]

    # The linked cursor reports each member's own value, and C == A - B exactly.
    values = diff._last_compare_values
    assert set(values) == {"A", "B", "C"}, values
    assert values["A"] == pytest.approx(arr_a[idx])
    assert values["B"] == pytest.approx(arr_b[idx])
    assert values["C"] == pytest.approx(oracle)

    # The committed/rendered frame carries A - B too (framebuffer value path).
    assert committed_value(diff, 3, 2) == pytest.approx(oracle)

    # Oracle discriminates: a transposed coordinate is a different value.
    assert values["C"] != (arr_a[(3, 2)] - arr_b[(3, 2)])

    # A hover in A mirrors the C = A - B readout onto all windows.
    idx2 = _hover(win_a, view_col=5, view_row=1)
    assert idx2 == (1, 5)
    assert diff._last_compare_values["C"] == pytest.approx(arr_a[idx2] - arr_b[idx2])


def test_progressive_input_streams_region_reads_into_the_difference(
    qtbot, isolated_sync, compare_windows
):
    _clear_settings()
    from arrayscope.window import ArrayScopeWindow

    arr_a = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
    arr_b = (np.arange(6 * 8, dtype=np.float32).reshape(6, 8) * 0.25) + 7.0

    # A is a streaming source: reads flow through read_region, never a whole
    # array materialization.
    backing = np.zeros((6, 8), dtype=np.float32)
    spy = _SpyProgressive(backing)
    with spy.write_transaction() as arr:
        arr[:] = arr_a
    a_src = LazySourceArray(spy)

    win_a = compare_windows(ArrayScopeWindow(a_src))
    win_b = compare_windows(win_a.open_compare_window(data=arr_b))
    diff = compare_windows(win_a.open_difference_window())
    assert diff is not None

    _settle(qtbot, win_a, win_b, diff)

    # Region-only proof: the difference window pulled A through read_region
    # (each recorded spec is a full per-axis index spec, one item per axis),
    # never a whole-array asarray bypass.
    assert spy.read_specs, "difference render never read the streaming source"
    for spec in spy.read_specs:
        assert len(spec) == arr_a.ndim

    idx = _hover(diff, view_col=4, view_row=3)
    assert idx == (3, 4), idx
    oracle = arr_a[idx] - arr_b[idx]
    assert diff._last_compare_values["C"] == pytest.approx(oracle)
    assert committed_value(diff, 4, 3) == pytest.approx(oracle)


def test_mismatched_shapes_are_refused_without_crashing(qtbot, isolated_sync, compare_windows):
    _clear_settings()
    from arrayscope.window import ArrayScopeWindow

    arr_a = np.arange(6 * 8, dtype=float).reshape(6, 8)
    arr_b = np.arange(6 * 9, dtype=float).reshape(6, 9)  # different shape

    win_a = compare_windows(ArrayScopeWindow(arr_a))
    win_b = compare_windows(win_a.open_compare_window(data=arr_b))
    assert win_b is not None

    # No crash, no third window; the launcher declines the mismatch.
    diff = win_a.open_difference_window()
    assert diff is None


def test_closing_difference_leaves_a_and_b_functional(qtbot, isolated_sync, compare_windows):
    _clear_settings()
    from arrayscope.window import ArrayScopeWindow

    arr_a = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
    arr_b = arr_a * -1.0 - 1000.0

    # A and B over closable sources so we can assert they are NOT torn down.
    src_a = _ClosableNdSource(arr_a, label="A")
    src_b = _ClosableNdSource(arr_b, label="B")
    win_a = compare_windows(ArrayScopeWindow(LazySourceArray(src_a)))
    win_b = compare_windows(win_a.open_compare_window(data=LazySourceArray(src_b)))
    diff = win_a.open_difference_window()  # disposed manually below, NOT tracked
    assert diff is not None

    _settle(qtbot, win_a, win_b, diff)
    idx = (2, 3)
    oracle = arr_a[idx] - arr_b[idx]
    assert committed_value(diff, 3, 2) == pytest.approx(oracle)

    # The composite must not own the shared inputs.
    composite = diff.base_data.source
    assert composite._own_inputs is False

    # Fully dispose ONLY the difference window.
    _dispose(qtbot, [diff])

    # A and B sources were NOT closed and stay readable.
    assert not src_a.closed
    assert not src_b.closed
    assert read_source_value(win_a.base_data, idx) == pytest.approx(arr_a[idx])
    assert read_source_value(win_b.base_data, idx) == pytest.approx(arr_b[idx])

    # A and B still render after the difference window is gone.
    _settle(qtbot, win_a, win_b)
    assert committed_value(win_a, 3, 2) == pytest.approx(arr_a[idx])
    assert committed_value(win_b, 3, 2) == pytest.approx(arr_b[idx])
