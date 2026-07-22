"""Compare launcher + linked complex cursor (queue item 6).

Two in-process windows, pre-linked on dims/camera/levels, with a shared value
cursor: a hover in either window makes BOTH windows report the exact source
value of window A AND window B at the shared source index. Exactness is
checked against a direct NumPy index (the oracle), and the oracle is shown to
discriminate: a wrong (transposed / neighbour) coordinate, or a dropped
sibling value, goes red.
"""

import uuid

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore

from tests.ui.helpers import (
    clear_arrayscope_settings,
    frame_session_settled,
    process_events,
)

pytest.importorskip("pytestqt")


def _clear_settings():
    from pyqtgraph.Qt import QtCore as _QtCore

    clear_arrayscope_settings()
    settings = _QtCore.QSettings()
    settings.setValue("image_rendering_backend", "pyqtgraph")
    settings.sync()


@pytest.fixture
def isolated_sync(monkeypatch):
    monkeypatch.setenv("ARRAYSCOPE_SYNC_NAME", f"arrayscope-compare-uitest-{uuid.uuid4().hex[:12]}")


def _settle(qtbot, *windows):
    process_events(qtbot, count=20)
    qtbot.waitUntil(
        lambda: all(frame_session_settled(win) for win in windows)
        and all(win.renderer.display_geometry is not None for win in windows),
        timeout=4000,
    )


def _hover(win, view_col, view_row):
    """Hover the pointer over view coordinate (col, row) and return the
    resolved full source index the compare cursor used."""

    scene_pos = win.img_view.getView().mapViewToScene(
        QtCore.QPointF(float(view_col) + 0.25, float(view_row) + 0.25)
    )
    win.getPixel(scene_pos)
    return getattr(win, "_last_compare_array_index", None)


def test_compare_launcher_links_facets_and_reports_both_real_values(qtbot, isolated_sync):
    _clear_settings()
    from arrayscope.window import ArrayScopeWindow

    # Distinct per-cell values so any wrong coordinate is detectable; window B
    # holds a DIFFERENT array so each window must read its OWN source.
    arr_a = np.arange(6 * 8, dtype=float).reshape(6, 8)
    arr_b = arr_a * -1.0 - 1000.0

    win_a = ArrayScopeWindow(arr_a)
    qtbot.addWidget(win_a)
    win_b = win_a.open_compare_window(data=arr_b)
    qtbot.addWidget(win_b)
    try:
        assert win_b is not None
        # Pre-linked: dims + camera + levels enabled on BOTH controllers.
        for win in (win_a, win_b):
            for facet in ("dims", "camera", "levels"):
                assert win.sync_controller.facet_enabled(facet), (win.compare_label, facet)
        assert (win_a.compare_label, win_b.compare_label) == ("A", "B")

        _settle(qtbot, win_a, win_b)

        # Hover in window A at view (col=3, row=2) -> source index (2, 3).
        idx = _hover(win_a, view_col=3, view_row=2)
        assert idx == (2, 3), idx

        # Oracle: exact direct NumPy index, each window on its OWN array.
        oracle_a = arr_a[idx]
        oracle_b = arr_b[idx]
        assert oracle_a != oracle_b  # windows hold different data

        for win in (win_a, win_b):
            values = win._last_compare_values
            assert set(values) == {"A", "B"}, values
            assert values["A"] == pytest.approx(oracle_a)
            assert values["B"] == pytest.approx(oracle_b)
            # A dropped sibling value would leave "B" missing -> red above.
            assert "A" in win._last_compare_hud_text
            assert "B" in win._last_compare_hud_text

        # Oracle discriminates: a transposed / neighbour coordinate is a
        # DIFFERENT value, so a wrong lookup would go red.
        assert win_a._last_compare_values["A"] != arr_a[(3, 2)]
        assert win_a._last_compare_values["A"] != arr_a[(2, 4)]

        # Hovering elsewhere updates the shared readout on both windows.
        idx2 = _hover(win_a, view_col=5, view_row=1)
        assert idx2 == (1, 5)
        assert win_b._last_compare_values["A"] == pytest.approx(arr_a[idx2])
        assert win_b._last_compare_values["B"] == pytest.approx(arr_b[idx2])

        # A hover originating in window B is mirrored onto window A too.
        idx3 = _hover(win_b, view_col=0, view_row=4)
        assert idx3 == (4, 0)
        assert win_a._last_compare_values["A"] == pytest.approx(arr_a[idx3])
        assert win_a._last_compare_values["B"] == pytest.approx(arr_b[idx3])
    finally:
        win_b.close()
        win_a.close()


def test_compare_cursor_reports_complex_magnitude_and_phase(qtbot, isolated_sync):
    _clear_settings()
    from arrayscope.window import ArrayScopeWindow

    rng = np.random.default_rng(4)
    arr_a = (rng.standard_normal((6, 8)) + 1j * rng.standard_normal((6, 8))).astype(np.complex64)
    arr_b = (arr_a * (0.5 - 0.5j)).astype(np.complex64)

    win_a = ArrayScopeWindow(arr_a)
    qtbot.addWidget(win_a)
    win_b = win_a.open_compare_window(data=arr_b)
    qtbot.addWidget(win_b)
    try:
        _settle(qtbot, win_a, win_b)

        idx = _hover(win_a, view_col=4, view_row=3)
        assert idx == (3, 4), idx

        oracle_a = arr_a[idx]
        oracle_b = arr_b[idx]

        for win in (win_a, win_b):
            values = win._last_compare_values
            # Exact complex value read from each window's own source array.
            assert values["A"] == oracle_a
            assert values["B"] == oracle_b
            # Magnitude and phase both surface in the HUD text.
            text = win._last_compare_hud_text
            assert "|" in text
            assert "∠" in text

        # The reported magnitude/phase match the NumPy oracle exactly.
        val_a = win_a._last_compare_values["A"]
        assert np.abs(val_a) == pytest.approx(np.abs(oracle_a))
        assert np.angle(val_a) == pytest.approx(np.angle(oracle_a))

        # Oracle discriminates: transposed coordinate is a different complex
        # value, so a wrong lookup would go red.
        assert win_a._last_compare_values["A"] != arr_a[(4, 3)]
    finally:
        win_b.close()
        win_a.close()
