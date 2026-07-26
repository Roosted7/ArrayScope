"""Discovered user-op shapes must remain truthful through montage tile planning."""

from __future__ import annotations

import numpy as np

from arrayscope.operations import library, registry
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.ui.helpers import (
    frame_session_settled,
    process_events,
    use_pyqtgraph_backend,
)


def test_decimating_user_operation_settles_truthful_montage_tiles(qtbot, tmp_path, monkeypatch):
    settings = use_pyqtgraph_backend()
    operations_dir = tmp_path / "operations"
    monkeypatch.setattr(library, "user_operations_directory", lambda: str(operations_dir))
    source = tmp_path / "decimate.py"
    source.write_text("def decimate(data):\n    return data[..., ::2]\n")
    operation_id = library.import_custom_operation(str(source), "decimate", changes_shape=True)

    from arrayscope.window import ArrayScopeWindow

    data = np.empty((12, 10, 12), dtype=np.float32)
    for index in range(data.shape[2]):
        data[:, :, index] = index
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        win.show()
        process_events(qtbot)
        win.operation_coordinator.load_operations((registry.create_operation(operation_id),))
        win._set_document(win.operation_coordinator.document)
        assert win.document.current_shape == (12, 10, 6)

        indices = tuple(range(6))
        win._set_view_state(
            win.view_state.with_montage_axis(2, columns=3, indices=indices, text=":")
        )
        win.render(reason="test-discovered-shape-montage")
        qtbot.waitUntil(
            lambda: (
                frame_session_settled(win)
                and win.renderer._frame_session.plan.geometry.indices == indices
            ),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )

        session = win.renderer._frame_session
        assert tuple(tile.montage_index for tile in session.visible_tiles) == indices
        frame = win._committed_display_frame
        payloads = frame.value_source.payloads
        for index in indices:
            payload = payloads[index]
            np.testing.assert_array_equal(
                np.asarray(payload.image),
                np.full((12, 10), index * 2, dtype=np.float32),
            )
        diagnostics = win.operation_evaluator.stage_cache_diagnostics()
        assert diagnostics.entries >= 1
        assert diagnostics.candidates_seen >= 1
    finally:
        win.close()
        library.remove_user_operation(operation_id)
        settings.clear()
        settings.sync()
