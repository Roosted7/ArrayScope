import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_slice_range_text_shift_preserves_step(qt_app):
    from arrayscope.ui.dimension_strip import _shift_slice_text

    assert _shift_slice_text("0:2:10", 1, 20) == "2:2:12"
    assert _shift_slice_text("2:2:10", -1, 20) == "0:2:8"
    assert _shift_slice_text("2:6", 1, 20) == "3:7"


def test_image_axes_show_full_range_colon(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.ui.dimension_strip import DimensionStrip

    strip = DimensionStrip(3)
    state = ViewState.from_shape((4, 5, 6))

    strip.update_state((4, 5, 6), state, profile_axes=(2,))

    assert strip.chip(0).slice_edit.text() == ":"
    assert strip.chip(1).slice_edit.text() == ":"
    assert strip.chip(2).slice_edit.text() == "3"
    strip.close()


def test_dimension_chip_update_state_skips_unchanged_icon_work(qt_app, monkeypatch):
    from arrayscope.core.view_state import ViewState
    import arrayscope.ui.dimension_strip as dimension_strip

    calls = []
    original = dimension_strip.set_button_icon

    def record_icon(button, name, **kwargs):
        calls.append((button, name, kwargs.get("tooltip")))
        return original(button, name, **kwargs)

    monkeypatch.setattr(dimension_strip, "set_button_icon", record_icon)
    strip = dimension_strip.DimensionStrip(3)
    state = ViewState.from_shape((4, 5, 6))
    try:
        strip.update_state((4, 5, 6), state, profile_axes=(2,))
        first_count = len(calls)

        strip.update_state((4, 5, 6), state, profile_axes=(2,))

        assert first_count > 0
        assert len(calls) == first_count
    finally:
        strip.close()


def test_dimension_strip_wraps_to_allocated_width(qt_app):
    from arrayscope.ui.dimension_strip import DimensionStrip

    strip = DimensionStrip(6)
    strip.resize(520, 120)
    strip.show()
    qt_app.processEvents()
    strip._relayout()

    assert strip._columns == 2
    assert strip.maximumWidth() <= 484
    assert max(strip.chip(axis).geometry().right() for axis in range(6)) <= strip.contentsRect().right()
    assert strip.chip(2).geometry().top() > strip.chip(0).geometry().top()
    strip.close()


def test_chip_labels_show_axis_metadata_when_available(qt_app):
    from arrayscope.core.axis_info import AxisInfo, default_axes
    from arrayscope.core.view_state import ViewState
    from arrayscope.ui.dimension_strip import DimensionStrip

    strip = DimensionStrip(3)
    state = ViewState.from_shape((4, 5, 6))
    axes = (
        AxisInfo("readout", "Readout", 4, unit="mm", spacing=1.5),
        AxisInfo("axis-1", "Dim 1", 5),
        AxisInfo("slice", "Slice", 6, unit="mm", spacing=3.0, origin=-9.0),
    )

    strip.update_state((4, 5, 6), state, profile_axes=(2,), axes=axes)

    assert strip.chip(0).axis_label.text() == "Readout [4]"
    assert strip.chip(1).axis_label.text() == "1 [5]"
    assert strip.chip(2).axis_label.text() == "Slice [6]"
    assert "spacing: 1.5 mm" in strip.chip(0).axis_label.toolTip()
    assert "origin: -9 mm" in strip.chip(2).axis_label.toolTip()

    # Default axes keep the compact positional label.
    strip.update_state((4, 5, 6), state, profile_axes=(2,), axes=default_axes((4, 5, 6)))
    assert strip.chip(0).axis_label.text() == "0 [4]"

    # Without metadata, behavior is unchanged.
    strip.update_state((4, 5, 6), state, profile_axes=(2,))
    assert strip.chip(0).axis_label.text() == "0 [4]"
    strip.close()


def test_chip_ignores_axes_that_do_not_match_shape(qt_app):
    from arrayscope.core.axis_info import AxisInfo
    from arrayscope.core.view_state import ViewState
    from arrayscope.ui.dimension_strip import DimensionStrip

    strip = DimensionStrip(2)
    state = ViewState.from_shape((4, 5))

    strip.update_state((4, 5), state, axes=(AxisInfo("readout", "Readout", 4),))

    assert strip.chip(0).axis_label.text() == "0 [4]"
    strip.close()


def test_long_axis_labels_are_elided(qt_app):
    from arrayscope.ui.dimension_strip import _elide

    assert _elide("Slice") == "Slice"
    assert _elide("A Very Long Axis Label") == "A Very Long…"


def test_slice_selection_validator_rejects_unsupported_characters(qt_app):
    from PySide6 import QtGui

    from arrayscope.ui.dimension_strip import SliceIndexEdit

    edit = SliceIndexEdit()
    validator = edit.lineEdit().validator()

    assert validator.validate("0:100:2", 0)[0] == QtGui.QValidator.State.Acceptable
    assert validator.validate("0 5,8;9", 0)[0] == QtGui.QValidator.State.Acceptable
    assert validator.validate("abc", 0)[0] == QtGui.QValidator.State.Invalid
    assert validator.validate("0#4", 0)[0] == QtGui.QValidator.State.Invalid
    edit.close()
