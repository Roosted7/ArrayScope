import time

import numpy as np
import pytest

from tests.ui.helpers import (
    assert_panel_invariants as _assert_panel_invariants,
    assert_size_close as _assert_size_close,
    clear_arrayscope_settings as _clear_arrayscope_settings,
    panel_body as _panel_body,
    process_events as _process_events,
    view_action as _view_action,
    wait_for_panel_preserve as _wait_for_panel_preserve,
)


def test_tiled_dimension_x_y_buttons_promote_range_to_image_crop(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, indices=(0, 1), text=":"))
        win.render(reason="test")
        _process_events(qtbot)

        chip = win.dimension_strip.chip(2)
        assert chip.x_button.isEnabled()
        assert chip.y_button.isEnabled()
        win.set_dimension_role("x", 2)

        assert win.view_state.image_axes == (0, 2)
        assert win.view_state.montage_axis is None
        assert win.view_state.axis_range_indices[2] == (0, 1)
        assert win.view_state.axis_range_text[2] == ":"
        assert win.view_state.slice_indices[1] == 1
    finally:
        win.close()


def test_demoting_cropped_image_axis_preserves_it_as_montage(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 5 * 4, dtype=float).reshape(2, 5, 4))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_axis_range(1, indices=(0, 2, 4), text="0:5:2"))

        win.set_dimension_role("x", 2)

        assert win.view_state.image_axes == (0, 2)
        assert win.view_state.montage_axis == 1
        assert win.view_state.montage_indices == (0, 2, 4)
        assert win.view_state.montage_text == "0:5:2"
        assert win.view_state.axis_range_indices[1] is None
    finally:
        win.close()


def test_demoting_full_image_axis_centers_scalar_slice(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(6 * 5 * 4, dtype=float).reshape(6, 5, 4))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_slice(0, 0))

        win.set_dimension_role("y", 2)

        assert win.view_state.image_axes == (2, 1)
        assert win.view_state.slice_indices[0] == 3
    finally:
        win.close()


def test_swapping_cropped_x_axis_to_y_keeps_existing_montage_axis(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(6 * 5 * 4, dtype=float).reshape(6, 5, 4))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        state = (
            win.view_state.with_image_axes(1, 2)
            .with_montage_axis(0, indices=(0, 2, 4), text="0:6:2")
            .with_axis_range(2, indices=(1, 2, 3), text="1:4")
        )
        win._set_view_state(state)

        win.set_dimension_role("y", 2)

        assert win.view_state.image_axes == (2, 1)
        assert win.view_state.montage_axis == 0
        assert win.view_state.montage_indices == (0, 2, 4)
        assert win.view_state.axis_range_indices[2] == (1, 2, 3)
    finally:
        win.close()


def test_empty_tiled_slice_text_clears_to_midpoint_scalar(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 5, dtype=float).reshape(2, 3, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, indices=(0, 1, 2), text=":"))
        win._on_slice_text_changed(2, "")
        _process_events(qtbot)

        assert win.view_state.montage_axis is None
        assert win.view_state.axis_range_indices[2] is None
        assert win.view_state.slice_indices[2] == 2
    finally:
        win.close()


def test_invalid_slice_text_restores_state_and_shows_status(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow
    import arrayscope.window.state_sync as state_sync

    messages = []
    monkeypatch.setattr(state_sync, "show_status_message", lambda _window, message, **_kwargs: messages.append(str(message)))

    win = ArrayScopeWindow(np.arange(2 * 3 * 5, dtype=float).reshape(2, 3, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        before = win.view_state

        win._on_slice_text_changed(2, "abc")
        _process_events(qtbot)

        assert win.view_state == before
        assert messages
        assert "Could not understand" in messages[-1]
        assert win.dimension_strip.chip(2).slice_edit.text() == "2"
    finally:
        win.close()


def test_raw_index_list_text_creates_montage_selection(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 5, dtype=float).reshape(2, 3, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)

        win._on_slice_text_changed(2, "0,2;4")
        _process_events(qtbot)

        assert win.view_state.montage_axis == 2
        assert win.view_state.montage_indices == (0, 2, 4)
        assert win.view_state.montage_text == "0 2 4"
    finally:
        win.close()


def test_raw_index_list_montage_renders_each_selected_source(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((4, 5, 60), dtype=float)
    for index in range(data.shape[2]):
        data[:, :, index] = float(index)
    selected = (7, 8, 9, 11, 14, 56)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)

        win._on_slice_text_changed(2, "7 8 9 11 14 56")
        for _ in range(120):
            _process_events(qtbot, count=2)
            session = getattr(win, "_montage_session", None)
            if session is not None and len(getattr(session, "rendered_tiles", {})) == len(selected):
                break

        session = win._montage_session
        rendered = [session.rendered_tiles[tile] for tile in sorted(session.rendered_tiles)]
        assert tuple(tile.tile.source_index for tile in rendered) == selected
        for rendered_tile in rendered:
            assert np.nanmin(rendered_tile.image) == float(rendered_tile.tile.source_index)
            assert np.nanmax(rendered_tile.image) == float(rendered_tile.tile.source_index)
    finally:
        win.close()


def test_live_profile_from_axis_sets_exactly_one_profile_axis(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.profile_axes = (0, 1)
        win._enable_live_profile_for_axis(2)
        _process_events(qtbot)

        assert win.profile_axes == (2,)
        assert win.view_state.line_axis == 2
        assert win.profile_dock.isVisible()
    finally:
        win.close()
