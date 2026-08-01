"""Live PyQtGraph coverage for the large-montage preview handoff."""

from __future__ import annotations

import numpy as np
import pytest

from tests.settle import settle_window
from tests.ui.helpers import (
    frame_session_settled,
    make_backend_window,
    restore_default_backend,
    use_pyqtgraph_backend,
)

# ``complex-fft`` used to be xfail(raises=QtBotTimeoutError): the transition
# does complete, it just takes longer than the 5 s interaction cap, so the wait
# expired and the expired wait was declared the expected outcome.  That made the
# xfail a latency assertion wearing a correctness test's clothes, and it cost
# the correctness claim entirely — nothing in this file has ever verified that
# the complex path presents the 272-tile physical union, because the run never
# got that far.  The wait now bounds a wedge rather than a duration, so the
# parameter is an ordinary one and the assertions below finally apply to it.
#
# The latency itself is unchanged and still red: R5 in docs/queue.md, measured
# by ``arrayscope.tools.profile_montage_workflow``, whose FFT-over-dim-2 montage
# stage records per-step settlement time on a quiet machine. A pytest worker
# sharing a box with fifteen others cannot adjudicate a 5 s bar.


@pytest.mark.parametrize(
    "complex_fft",
    [
        pytest.param(False, id="scalar"),
        pytest.param(True, id="complex-fft"),
    ],
)
def test_single_slice_to_full_montage_presents_around_retained_exact_tile(
    qtbot,
    complex_fft,
):
    """A retained single slice is the physical complement of the preview atlas.

    The field reproduction enters a 272-tile montage from an already-settled
    central slice.  That exact predecessor remains drawable, so the preview
    producer emits only the other 271 payloads.  The aggregate transaction
    must accept that 271+1 physical union instead of waiting forever for a
    preview payload that is intentionally unnecessary.
    """

    settings = use_pyqtgraph_backend(extra_settings={"montage_quality_policy": "resident"})
    data = np.arange(336 * 336 * 272, dtype=np.float32).reshape(336, 336, 272)
    win = make_backend_window(qtbot, data, backend="pyqtgraph")
    win.resize(950, 950)
    try:
        win.show()
        settle_window(win, lambda: frame_session_settled(win), label="single slice settles")
        previous_session_id = int(win.renderer._frame_session.session_id)

        win._set_view_state(
            win.view_state.with_montage_axis(
                2,
                columns=17,
                indices=tuple(range(272)),
                text=":",
            )
        )
        win.update_image_view()

        def preview_is_physically_complete() -> bool:
            session = win.renderer._frame_session
            if (
                session is None
                or int(session.session_id) == previous_session_id
                or len(session.plan.tiles) != 272
            ):
                return False
            return bool(
                session.required_first_pixels_presented()
                and len(win.img_view.tileTruthPhysicalRows()) == 272
            )

        settle_window(win, preview_is_physically_complete, label="montage preview completes")
        if complex_fft:
            previous_session_id = int(win.renderer._frame_session.session_id)
            assert win.request_operation("centered_fft", 2)
            settle_window(win, preview_is_physically_complete, label="fft preview completes")

        rows = win.img_view.tileTruthPhysicalRows()
        storage_modes = {str(row.get("physical_storage_mode", "")) for row in rows.values()}
        assert set(rows) == set(range(272))
        assert "compact_preview_atlas" in storage_modes
        if not complex_fft:
            assert "image_item" in storage_modes
        atlas_rows = {
            tile: row
            for tile, row in rows.items()
            if row.get("physical_storage_mode") == "compact_preview_atlas"
        }
        assert atlas_rows
        assert {str(row.get("physical_texture_kind", "")) for row in atlas_rows.values()} == (
            {"complex_rg32f"} if complex_fft else {"scalar_r32f"}
        )
        if complex_fft:
            assert {str(row.get("physical_mapping_mode", "")) for row in atlas_rows.values()} == {
                "cpu_rgb_from_complex_atlas"
            }
        assert int(getattr(win.renderer, "_montage_stall_assertions", 0) or 0) == 0
    finally:
        win.close()
        restore_default_backend(settings)
