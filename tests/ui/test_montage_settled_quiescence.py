"""A montage that has finished stops owing itself work.

The subject is one claim: after a data change, a settled montage's presentation
bookkeeping drains to empty, and the montage prefetch ``_busy`` predicate — the
gate that decides whether speculative residency work may run at all — goes
false.  It did not.  Thirteen tiles in the visible MARGIN but outside the
required scope held a display payload from before the change; every commit
rebound the same superseded wrapper, re-armed the flush and reconciled them
again, so ``dirty_payloads`` never emptied and ``_busy`` stayed true for the
life of the session.

The regime is chosen against the failure mode this repo keeps hitting -- a
green result resting on a test that cannot fail:

* 272 tiles against a nine-tile viewport, because the standing count was
  measured at field scale.  A montage small enough to have no margin ring
  outside the required scope has no defect to observe, so the fixture asserts
  that ring is non-empty before it asserts anything about draining.
* the drain assertions are preceded by their own mutant: the data change must
  actually have dirtied the bookkeeping it is then required to drain.  A
  fixture that stops producing a data change would otherwise pass by
  measuring nothing.
* both backends.  The defect was identical on ``cpu_item`` and ``gpu_atlas``
  because it lives above the backend seam, in the commit's reconcile pass.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from arrayscope.window.montage_prefetch import _busy
from tests.ui.helpers import (
    frame_session_settled,
    make_backend_window,
    restore_default_backend,
    use_pyqtgraph_backend,
    use_wgpu_backend,
)

TILE = 32
# The field montage the standing count was measured on.
SOURCES, COLUMNS, ROWS = (272, 17, 16)


def _montage_data(sources: int) -> np.ndarray:
    """Three slices whose per-tile values differ, at one global range.

    Identical range so an auto-window change cannot be mistaken for a content
    change; different per-tile values so a payload carried over from the
    previous slice is a real staleness rather than a relabelling.
    """

    yy = np.arange(TILE, dtype=np.float32)[:, None, None]
    xx = np.arange(TILE, dtype=np.float32)[None, :, None]
    zz = np.arange(sources, dtype=np.float32)[None, None, :]
    base = np.ascontiguousarray(yy * 3.0 + xx * 1.0 + zz * 5000.0)
    reversed_base = np.ascontiguousarray(base[:, :, ::-1])
    return np.ascontiguousarray(np.stack([reversed_base, base, base], axis=-1))


def _settle(win, qtbot) -> None:
    with contextlib.suppress(Exception):
        qtbot.waitUntil(
            lambda: frame_session_settled(win), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS
        )


def _owed(session) -> dict[str, int]:
    """Everything a commit would still call unfinished business."""

    return {
        "pending_removals": len(session.pending_removals),
        "deferred_out_of_view": len(session.deferred_out_of_view_invalidations),
        "dirty_payloads": len(session.dirty_payloads),
        "pending_payload_upserts": len(session.pending_payload_upserts),
    }


@pytest.mark.parametrize("backend", ["pyqtgraph", "wgpu"])
def test_settled_montage_after_a_data_change_owes_no_bookkeeping(qtbot, backend):
    settings = use_pyqtgraph_backend() if backend == "pyqtgraph" else use_wgpu_backend()
    win = make_backend_window(
        qtbot, _montage_data(SOURCES), backend=backend, require_gpu_atlas=backend == "wgpu"
    )
    try:
        win.resize(900, 700)
        win.show()
        win._set_view_state(
            win.view_state.with_montage_axis(
                2, columns=COLUMNS, indices=tuple(range(SOURCES)), text=":"
            )
        )
        win.update_image_view()
        _settle(win, qtbot)

        # Nine tiles visible out of 272, so the margin ring outside the required
        # scope is a real population rather than a rounding error.
        height, width = (int(v) for v in win.renderer._frame_session.plan.display_shape[:2])
        tile_w, tile_h = width / COLUMNS, height / ROWS
        win.img_view.getView().setRange(
            xRange=(0.0, 3 * tile_w), yRange=(0.0, 3 * tile_h), padding=0
        )
        win.update_image_view()
        _settle(win, qtbot)

        session = win.renderer._frame_session
        margin_only = set(session.visible_tile_numbers) - set(session.required_tile_numbers())
        assert margin_only, (
            "no tile is visible-but-not-required, so this viewport has no margin "
            "ring and the standing-payload regime cannot occur here"
        )

        # The first slice step of a session pays a one-off pipeline warm-up that
        # is not the subject; the measured change is the second.  Both target
        # slices the montage has never displayed, so every visible tile is a
        # display-cache miss -- the regime where superseded payloads exist at all.
        win._set_view_state(win.view_state.with_slice(3, 2))
        win.render(reason="test-pipeline-warmup")
        _settle(win, qtbot)

        win._set_view_state(win.view_state.with_slice(3, 0))
        win.render(reason="test-data-change")

        # The mutant: the change must actually have dirtied the bookkeeping that
        # the assertions below require it to drain.
        session = win.renderer._frame_session
        owed_at_change = _owed(session)
        assert sum(owed_at_change.values()) > 0, (
            f"the data change left nothing owed ({owed_at_change}), so the drain "
            "assertions below would pass without draining anything"
        )

        _settle(win, qtbot)

        # Both halves of the claim, waited for together and asserted at the same
        # instant.  ``_busy`` is deliberately not asserted at a later arbitrary
        # moment: once the gate opens, prefetch starts speculative work and
        # honestly reports busy again, which is the machine being used rather
        # than the bookkeeping refusing to let go.
        def quiet() -> bool:
            live = win.renderer._frame_session
            return not any(_owed(live).values()) and not _busy(win.renderer, live)

        qtbot.waitUntil(quiet, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

        session = win.renderer._frame_session
        assert _owed(session) == {
            "pending_removals": 0,
            "deferred_out_of_view": 0,
            "dirty_payloads": 0,
            "pending_payload_upserts": 0,
        }
        assert not _busy(win.renderer, session)
    finally:
        restore_default_backend(settings)
        win.close()
