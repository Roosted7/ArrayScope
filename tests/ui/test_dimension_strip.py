import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_slice_range_text_shift_preserves_step(qt_app):
    from arrayscope.ui.dimension_strip import _shift_slice_text

    assert _shift_slice_text("0:2:10", 1, 20) == "2:2:12"
    assert _shift_slice_text("2:2:10", -1, 20) == "0:2:8"
    assert _shift_slice_text("2::3", 1, 20) == "3:4"


def test_image_axes_show_full_range_colon(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.ui.dimension_strip import DimensionStrip

    strip = DimensionStrip(3)
    state = ViewState.from_shape((4, 5, 6))

    strip.update_state((4, 5, 6), state, profile_axes=(2,))

    assert strip.chip(0).slice_edit.text() == ":"
    assert strip.chip(1).slice_edit.text() == ":"
    assert strip.chip(2).slice_edit.text() == "0"
    strip.close()
