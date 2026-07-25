"""Resident crop-window scrub short-circuits to a rebind (no producer work).

Scrubbing the displayed-axis crop window of an already-resident montage plane
re-samples the same canonical GPU pages at a shifted origin.  The demand/planning
layer otherwise never consults physical residency, so every shifted window is a
fresh typed target and each tile is re-evaluated (one display-cache miss and one
producer per tile per step) even though the pixels are already resident.  With
the opt-in ``resident_crop_rebind`` capability the planner rebinds the resident
pages before the ladder plans, scheduling ZERO producers for the resident tiles.

Residency is not a gift a cropped view receives: a view cropped from its first
frame onward never presents a whole source plane, so under a crop-local upload
nothing would ever be resident to rebind.  The same capability therefore widens
a cropped tile's evaluation to the whole canonical plane its anchor's content
key already names, so the first fill warms the window-invariant pages once and
every later crop step is a pure source-origin rebind.

The capability is gated OFF by default: the rebind reuses the predecessor
window's auto-level evidence (the maturity contract) instead of re-anchoring, so
it is only pixel-exact against the CPU oracle while the level window is stable
(verified here with statistically uniform data).  A crop whose pages are NOT
resident, any pixel-affecting identity change, or a montage of planes too large
for the residency byte policy falls through to the ordinary evaluation.
A rebound window re-anchors its own auto levels: the evidence it carries
describes the PREDECESSOR window, so it is demoted to preview quality and the
semantic level-evidence owner re-samples the new window off the display lane.
The capability is still gated OFF by default because that owner is not admitted
on an operation-pipeline montage, which the last test here pins.  A crop whose
pages are NOT resident, or any pixel-affecting identity change, falls through to
the ordinary evaluation.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.display.model.montage_levels import LevelEvidenceQuality
from arrayscope.tools.framebuffer_reference import assert_wgpu_frame_matches_cpu_reference
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from arrayscope.window.frame_effects import canonical_plane_memo_bytes
from tests.ui.helpers import (
    frame_session_settled,
    make_backend_window,
    restore_default_backend,
    use_wgpu_backend,
)

_MONTAGE_INDICES = tuple(range(30, 230, 2))


def _preparation_completed(win) -> int:
    lanes = win.kernel.diagnostics().lanes
    return int((lanes.get("display_preparation") or {}).get("completed", 0) or 0)


def _busy_pump_until(predicate, budget_s, label) -> None:
    # A busy pump, not qtbot.waitUntil: a real event loop never idles, so the
    # low-priority planning continuations only run under sustained queue
    # pressure.  The idle qtbot loop would dispatch them instantly and hide the
    # field scheduling entirely.
    app = QtWidgets.QApplication.instance()
    deadline = perf_counter() + budget_s
    while not predicate():
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
        assert perf_counter() < deadline, f"{label} failed to settle within {budget_s:.1f}s"


def _uniform_source() -> np.ndarray:
    # Statistically uniform across every crop window so the auto-level window is
    # stable and a level-reusing rebind stays pixel-exact.  A per-tile constant
    # gradient would shift the window and is deliberately out of scope for the
    # gated rebind.
    return np.random.default_rng(20260724).standard_normal((336, 336, 272), dtype=np.float32)


def _cropped_state(win, start: int):
    state = win.view_state
    state = state.with_axis_range(
        0, indices=tuple(range(start, start + 200)), text=f"{start}:{start + 200}"
    )
    state = state.with_axis_range(1, indices=tuple(range(66, 266)), text="66:266")
    return state.with_montage_axis(2, columns=10, indices=_MONTAGE_INDICES, text="30:2:230")


def _crop_settled(win, start: int) -> bool:
    session = getattr(win.renderer, "_frame_session", None)
    if session is None or session.plan is None:
        return False
    ranges = tuple(getattr(session.view_state, "axis_range_indices", None) or ())
    indices = tuple(ranges[0] or ()) if ranges else ()
    if not indices or int(indices[0]) != int(start):
        return False
    return frame_session_settled(win)


def _fill_full_plane_then_crop(win, start: int) -> None:
    full = win.view_state.with_montage_axis(
        2, columns=10, indices=_MONTAGE_INDICES, text="30:2:230"
    )
    win._set_view_state(full)
    win.update_image_view()
    _busy_pump_until(
        lambda: frame_session_settled(win),
        INTERACTION_SETTLE_HARD_LIMIT_MS * 4 / 1000.0,
        "full-plane fill",
    )
    win._apply_slice_state(
        0,
        _cropped_state(win, start),
        reason="slice-range",
        interactive=True,
        immediate_axis_only=False,
    )
    _busy_pump_until(
        lambda: _crop_settled(win, start),
        INTERACTION_SETTLE_HARD_LIMIT_MS * 4 / 1000.0,
        "cold cropped fill",
    )


def test_resident_crop_scrub_schedules_no_producers(qtbot):
    settings = use_wgpu_backend(
        extra_settings={"montage_quality_policy": "resident", "resident_crop_rebind": True}
    )
    win = make_backend_window(qtbot, _uniform_source(), backend="wgpu", require_gpu_atlas=True)
    win.resize(780, 760)
    try:
        win.show()
        _fill_full_plane_then_crop(win, 94)

        # Every scrub step shifts a fully-resident window: zero producers.
        for step in range(4):
            start = 96 + step
            before = _preparation_completed(win)
            win._on_slice_text_changed(0, f"{start}:{start + 200}")
            _busy_pump_until(
                lambda start=start: _crop_settled(win, start),
                INTERACTION_SETTLE_HARD_LIMIT_MS / 1000.0,
                f"resident scrub {start}",
            )
            assert _preparation_completed(win) - before == 0, (
                "a resident crop scrub must schedule no display-preparation producers"
            )

        assert int(getattr(win.renderer, "_montage_stall_assertions", 0) or 0) == 0
        assert_wgpu_frame_matches_cpu_reference(win)
    finally:
        win.close()
        restore_default_backend(settings)


def test_crop_scrub_without_the_capability_still_evaluates(qtbot):
    """With the capability OFF nothing changes: no warm, no rebind, producers.

    Widening a cropped evaluation to its whole canonical plane exists only to
    feed the rebind, so it is gated by the same capability.  Default-OFF must
    therefore be bit-for-bit the pre-warm behaviour: crop-local payloads
    (``native_residency_data`` describes the WINDOW, not the plane), no
    canonical page ever uploaded, and one producer per tile per scrub step.
    """

    # Spelled out rather than relied upon: another window in this process can
    # flush its own QSettings after the fixture clears them, and the point of
    # this gate is the OFF path, not which default produced it.
    settings = use_wgpu_backend(
        extra_settings={"montage_quality_policy": "resident", "resident_crop_rebind": False}
    )
    win = make_backend_window(qtbot, _uniform_source(), backend="wgpu", require_gpu_atlas=True)
    win.resize(780, 760)
    try:
        win.show()
        assert bool(getattr(win.app_settings, "resident_crop_rebind", False)) is False
        win._set_view_state(_cropped_state(win, 94))
        win.update_image_view()
        _busy_pump_until(
            lambda: _crop_settled(win, 94),
            INTERACTION_SETTLE_HARD_LIMIT_MS * 4 / 1000.0,
            "cold cropped fill",
        )
        session = win.renderer._frame_session
        payload = session.display_tile_payloads[0]
        assert payload.source_anchor.plane_shape == (336, 336)
        native = payload.native_residency_data
        assert native is None or tuple(np.shape(native)[:2]) != (336, 336), (
            "the capability is off, so no evaluation may be widened to the plane"
        )
        assert not any(
            key.document_generation[0] == "wgpu-source-plane"
            for key in win.img_view._wgpu_executor.page_table.resident_keys()
        ), "a crop-local commit must not warm canonical source-plane pages when off"

        before = _preparation_completed(win)
        win._on_slice_text_changed(0, "96:296")
        _busy_pump_until(
            lambda: _crop_settled(win, 96),
            INTERACTION_SETTLE_HARD_LIMIT_MS * 2 / 1000.0,
            "cold scrub",
        )
        assert _preparation_completed(win) - before > 0, (
            "a non-resident crop window must still schedule its producers"
        )
        totals = dict(getattr(win.renderer, "resident_crop_rebind_totals", None) or {})
        assert totals.get("gate:disabled", 0) > 0
        assert totals.get("rebound", 0) == 0
        assert int(getattr(win.renderer, "_montage_stall_assertions", 0) or 0) == 0
    finally:
        win.close()
        restore_default_backend(settings)


def test_slice_change_does_not_reuse_residency(qtbot):
    """A pixel-affecting identity change (slice index) never rebinds residency.

    The rebind is only legal for a pure window shift under an unchanged content
    key.  Advancing the non-displayed slice index changes the content key, so
    the residency short-circuit must decline and the planner must evaluate.
    """

    settings = use_wgpu_backend(
        extra_settings={"montage_quality_policy": "resident", "resident_crop_rebind": True}
    )
    data = _uniform_source()
    win = make_backend_window(qtbot, data, backend="wgpu", require_gpu_atlas=True)
    win.resize(780, 760)
    try:
        win.show()
        _fill_full_plane_then_crop(win, 94)
        # Re-slice the montage-adjacent axis via a fresh montage window (new
        # source indices): a genuine content change, not a window shift.
        before = _preparation_completed(win)
        shifted = tuple(index + 1 for index in _MONTAGE_INDICES)
        state = win.view_state.with_montage_axis(2, columns=10, indices=shifted, text="31:2:231")
        win._set_view_state(state)
        win.update_image_view()

        def montage_settled() -> bool:
            session = getattr(win.renderer, "_frame_session", None)
            if session is None or session.plan is None:
                return False
            plan_sources = tuple(int(tile.source_index) for tile in session.plan.tiles)
            return bool(plan_sources == shifted and frame_session_settled(win))

        _busy_pump_until(
            montage_settled, INTERACTION_SETTLE_HARD_LIMIT_MS * 2 / 1000.0, "content change"
        )
        assert _preparation_completed(win) - before > 0, (
            "a content-identity change must not be served by a resident rebind"
        )
    finally:
        win.close()
        restore_default_backend(settings)


# --- Single-slice (non-montage) shape --------------------------------------
# The same canonical source-plane pages back a single-slice view: its payload
# is crop-local exactly like a cropped montage tile's, and its source anchor
# resolves through the same ``source_anchoring_for_view`` content key.  A
# resident crop scrub there must short-circuit identically.


def _single_slice_cropped_state(win, start: int):
    state = win.view_state.with_montage_axis(None)
    state = state.with_axis_range(
        0, indices=tuple(range(start, start + 200)), text=f"{start}:{start + 200}"
    )
    return state.with_axis_range(1, indices=tuple(range(66, 266)), text="66:266")


def _single_slice_fill_full_plane_then_crop(win, start: int) -> None:
    """Fill the uncropped plane first so the canonical pages are complete.

    A source-anchored binding only reuses canonical pages that a
    ``supplies_complete_pages`` payload uploaded; the uncropped plane is what
    makes them complete.  Starting already cropped leaves the window on
    crop-local pages, which is the cold case below.
    """

    win._set_view_state(win.view_state.with_montage_axis(None))
    win.update_image_view()
    _busy_pump_until(
        lambda: frame_session_settled(win),
        INTERACTION_SETTLE_HARD_LIMIT_MS * 4 / 1000.0,
        "single-slice full-plane fill",
    )
    win._apply_slice_state(
        0,
        _single_slice_cropped_state(win, start),
        reason="slice-range",
        interactive=True,
        immediate_axis_only=False,
    )
    _busy_pump_until(
        lambda: _crop_settled(win, start),
        INTERACTION_SETTLE_HARD_LIMIT_MS * 4 / 1000.0,
        "single-slice cold cropped fill",
    )


def test_single_slice_resident_crop_scrub_schedules_no_producers(qtbot):
    """A non-montage resident crop scrub schedules zero producers, pixel-exact.

    The montage gate excluded this shape on the rationale that its
    source-anchored composition diverges and that it is "already 1
    producer/step".  Measured, the composition is the same canonical
    source-plane binding, and the step still costs a full evaluation.
    """

    settings = use_wgpu_backend(
        extra_settings={"montage_quality_policy": "resident", "resident_crop_rebind": True}
    )
    win = make_backend_window(qtbot, _uniform_source(), backend="wgpu", require_gpu_atlas=True)
    win.resize(780, 760)
    try:
        win.show()
        assert win.view_state.montage_axis is None
        _single_slice_fill_full_plane_then_crop(win, 94)

        for step in range(4):
            start = 96 + step
            before = _preparation_completed(win)
            win._on_slice_text_changed(0, f"{start}:{start + 200}")
            _busy_pump_until(
                lambda start=start: _crop_settled(win, start),
                INTERACTION_SETTLE_HARD_LIMIT_MS / 1000.0,
                f"single-slice resident scrub {start}",
            )
            assert _preparation_completed(win) - before == 0, (
                "a resident single-slice crop scrub must schedule no display-preparation producers"
            )

        assert int(getattr(win.renderer, "_montage_stall_assertions", 0) or 0) == 0
        assert_wgpu_frame_matches_cpu_reference(win)
    finally:
        win.close()
        restore_default_backend(settings)


def test_single_slice_rebind_carries_the_new_window_semantics(qtbot):
    """A rebound single-slice payload carries the NEW window's exact values.

    ``TiledValueSource`` indexes a payload's semantic plane by window-LOCAL
    coordinates, so a rebind that shifted only ``source_anchor`` would draw the
    new window on the GPU while every CPU value read (ROI, profile, export, and
    the physical-truth oracle) silently answered from the predecessor window's
    plane.  The rebind re-slices those planes out of the whole plane, so the
    payload's semantics move with its anchor.
    """

    settings = use_wgpu_backend(
        extra_settings={"montage_quality_policy": "resident", "resident_crop_rebind": True}
    )
    data = _uniform_source()
    win = make_backend_window(qtbot, data, backend="wgpu", require_gpu_atlas=True)
    win.resize(780, 760)
    try:
        win.show()
        _single_slice_fill_full_plane_then_crop(win, 94)
        slice_index = int(win.view_state.slice_indices[2])

        for start in (96, 99, 103):
            before = _preparation_completed(win)
            win._on_slice_text_changed(0, f"{start}:{start + 200}")
            _busy_pump_until(
                lambda start=start: _crop_settled(win, start),
                INTERACTION_SETTLE_HARD_LIMIT_MS / 1000.0,
                f"semantics scrub {start}",
            )
            assert _preparation_completed(win) - before == 0, (
                f"window {start} was evaluated, so it does not exercise the rebind"
            )
            payload = win.renderer._frame_session.display_tile_payloads[0]
            assert payload.source_anchor.source_rect == (start, start + 200, 66, 266)
            np.testing.assert_allclose(
                np.asarray(payload.semantic_data),
                data[start : start + 200, 66:266, slice_index],
                atol=1e-6,
                err_msg=(
                    f"window {start}: the rebound payload's semantic plane does not "
                    "describe its own anchor (stale predecessor-window values)"
                ),
            )
    finally:
        win.close()
        restore_default_backend(settings)


def _hover_text(win, view_x: float, view_y: float) -> str:
    """Hover the real pointer entry point and return the readout label."""

    scene_pos = win.img_view.getView().mapViewToScene(
        QtCore.QPointF(float(view_x) + 0.25, float(view_y) + 0.25)
    )
    win.getPixel(scene_pos)
    return str(win.widgets["labels"]["pixelValue"].text())


def test_single_slice_rebind_hover_reads_the_new_window_value(qtbot):
    """Hovering a rebound tile reports its value, not "tile loading...".

    A rebind presents pixels with no evaluation, so the slot has no
    ``RenderedTile``.  ``mark_presented`` used that as its only admission test
    and dropped the tile, leaving the session's tile-state mirror at UNLOADED
    while the lifecycle already held the tile as presented — so
    ``view_point_to_array_index``'s ``require_loaded`` gate refused every hover
    over a tile that was drawn, current, and carrying the new window's own
    re-sliced CPU planes.  Scrubbing with the capability on therefore blanked
    the readout for the whole scrub.

    The value is checked against a full evaluation of the SAME window (the
    capability toggled off mid-test), which is the only comparison that proves
    the rebound plane is read with the right window-LOCAL coordinates: an
    off-by-the-shift read would still produce a plausible number.
    """

    settings = use_wgpu_backend(
        extra_settings={"montage_quality_policy": "resident", "resident_crop_rebind": True}
    )
    data = _uniform_source()
    win = make_backend_window(qtbot, data, backend="wgpu", require_gpu_atlas=True)
    win.resize(780, 760)
    try:
        win.show()
        _single_slice_fill_full_plane_then_crop(win, 94)
        slice_index = int(win.view_state.slice_indices[2])
        hover_x, hover_y = 37, 61

        # Baseline: this window WAS fully evaluated, and hover reads it.
        evaluated_94 = _hover_text(win, hover_x, hover_y)
        assert "loading" not in evaluated_94.lower()
        np.testing.assert_allclose(
            win.renderer._hover_value_from_display(
                win.renderer.display_geometry.context_for_view_point(
                    hover_x + 0.25, hover_y + 0.25
                ).mapping
            ),
            data[94 + hover_y, 66 + hover_x, slice_index],
            atol=1e-6,
        )

        before = _preparation_completed(win)
        win._on_slice_text_changed(0, "96:296")
        _busy_pump_until(
            lambda: _crop_settled(win, 96),
            INTERACTION_SETTLE_HARD_LIMIT_MS / 1000.0,
            "hover scrub 96",
        )
        assert _preparation_completed(win) - before == 0, (
            "window 96 was evaluated, so it does not exercise the rebind"
        )
        rebound_96 = _hover_text(win, hover_x, hover_y)
        assert "loading" not in rebound_96.lower(), (
            f"a rebound tile must answer hover, got {rebound_96!r}"
        )

        # The same window, produced the ordinary way: toggling the capability
        # off makes the next scrub evaluate, so the readout below is the
        # evaluated truth for exactly the window the rebind served.
        win._set_resident_crop_rebind_enabled(False)
        for start in (98, 96):
            before = _preparation_completed(win)
            win._on_slice_text_changed(0, f"{start}:{start + 200}")
            _busy_pump_until(
                lambda start=start: _crop_settled(win, start),
                INTERACTION_SETTLE_HARD_LIMIT_MS * 2 / 1000.0,
                f"evaluated scrub {start}",
            )
        assert _preparation_completed(win) - before > 0, (
            "the capability is off, so the return to window 96 must evaluate"
        )
        evaluated_96 = _hover_text(win, hover_x, hover_y)
        assert rebound_96 == evaluated_96, (
            "the rebound readout must equal the evaluated readout for the same window"
        )
    finally:
        win.close()
        restore_default_backend(settings)


def test_single_slice_cold_crop_scrub_falls_back_to_evaluation(qtbot):
    """A non-resident single-slice crop window keeps the ordinary evaluation."""

    settings = use_wgpu_backend(
        extra_settings={"montage_quality_policy": "resident", "resident_crop_rebind": True}
    )
    win = make_backend_window(qtbot, _uniform_source(), backend="wgpu", require_gpu_atlas=True)
    win.resize(780, 760)
    try:
        win.show()
        win._set_view_state(_single_slice_cropped_state(win, 94))
        win.update_image_view()
        _busy_pump_until(
            lambda: _crop_settled(win, 94),
            INTERACTION_SETTLE_HARD_LIMIT_MS * 4 / 1000.0,
            "single-slice cold cropped fill",
        )
        before = _preparation_completed(win)
        win._on_slice_text_changed(0, "96:296")
        _busy_pump_until(
            lambda: _crop_settled(win, 96),
            INTERACTION_SETTLE_HARD_LIMIT_MS * 2 / 1000.0,
            "single-slice cold scrub",
        )
        assert _preparation_completed(win) - before > 0, (
            "a non-resident single-slice crop window must still schedule its producers"
        )
        assert int(getattr(win.renderer, "_montage_stall_assertions", 0) or 0) == 0
    finally:
        win.close()
        restore_default_backend(settings)


def test_single_slice_slice_change_does_not_reuse_residency(qtbot):
    """Advancing the slider index is a content change, never a resident rebind.

    The non-montage anchor's content key folds the non-displayed axis slice
    index (``source_anchoring_for_view`` zeroes only the DISPLAYED axes'
    windows), so a new plane cannot be served by the predecessor's pixels.
    """

    settings = use_wgpu_backend(
        extra_settings={"montage_quality_policy": "resident", "resident_crop_rebind": True}
    )
    win = make_backend_window(qtbot, _uniform_source(), backend="wgpu", require_gpu_atlas=True)
    win.resize(780, 760)
    try:
        win.show()
        _single_slice_fill_full_plane_then_crop(win, 94)
        before = _preparation_completed(win)
        target = int(win.view_state.slice_indices[2]) + 5
        win._set_view_state(win.view_state.with_slice(2, target))
        win.update_image_view()

        def slice_settled() -> bool:
            session = getattr(win.renderer, "_frame_session", None)
            if session is None or session.plan is None:
                return False
            if int(session.view_state.slice_indices[2]) != target:
                return False
            return frame_session_settled(win)

        _busy_pump_until(
            slice_settled, INTERACTION_SETTLE_HARD_LIMIT_MS * 2 / 1000.0, "single-slice re-slice"
        )
        assert _preparation_completed(win) - before > 0, (
            "a content-identity change must not be served by a resident rebind"
        )
    finally:
        win.close()
        restore_default_backend(settings)


# --- Canonical-plane warming (what makes a born-cropped view rebindable) ----


def _anchored_state():
    from arrayscope.core.view_state import ViewState

    return ViewState(
        ndim=3,
        shape=(336, 336, 272),
        image_axes=(0, 1),
        line_axis=None,
        slice_indices=(0, 0, 30),
        axis_flipped=(False, False, False),
        axis_fftshifted=(False, False, False),
    )


def _anchoring(*, starts):
    from arrayscope.display.source_anchoring import SourceAnchoring

    return SourceAnchoring(anchored_starts=starts, content_key=("key",))


def test_canonical_plane_view_state_drops_only_anchored_windows():
    """Widening is legal exactly where the content key already dropped a window.

    An anchored axis' window is stripped from the content key, so the key is a
    promise about the whole plane and evaluating it redeems that promise.  A
    NON-anchored axis keeps its window folded into the key (the chain does not
    commute with slicing there), so widening it would name a plane no key ever
    described — and would silently hand the backend foreign pixels.
    """

    from arrayscope.display.source_anchoring import canonical_plane_view_state

    state = _anchored_state()
    state = state.with_axis_range(0, indices=tuple(range(38, 238)), text="38:238")
    state = state.with_axis_range(1, indices=tuple(range(66, 266)), text="66:266")

    both = canonical_plane_view_state(state, _anchoring(starts=(38, 66)))
    assert both is not None
    assert both.axis_range_indices[0] is None
    assert both.axis_range_indices[1] is None

    # Y anchored, X not: only Y widens.
    partial = canonical_plane_view_state(state, _anchoring(starts=(38, None)))
    assert partial is not None
    assert partial.axis_range_indices[0] is None
    assert tuple(partial.axis_range_indices[1]) == tuple(range(66, 266))

    # No anchored axis is cropped: the ordinary evaluation already covers the
    # plane, so there is nothing to widen and no second evaluation to pay for.
    assert canonical_plane_view_state(_anchored_state(), _anchoring(starts=(0, 0))) is None
    assert canonical_plane_view_state(state, _anchoring(starts=(None, None))) is None


def test_canonical_plane_warm_declines_a_montage_larger_than_the_budget():
    """A montage of whole planes that cannot be held keeps its crop-local upload.

    The warm plane REPLACES the crop-local page upload, so the budget question
    is whether the montage's whole physical working set fits the tile-residency
    byte policy — not whether one extra plane does.  Above the share, widening
    would only churn the page pool, so the crop stays local and the rebind stays
    inert there by design (it declines with ``pages_not_resident``).
    """

    import types

    from arrayscope.render.effects import canonical_plane_residency_source

    state = _anchored_state().with_axis_range(0, indices=tuple(range(38, 238)), text="38:238")
    tile = types.SimpleNamespace(view_state=state, source_index=30)
    plane_bytes = 336 * 336 * 4

    def session(*, tiles, budget):
        return types.SimpleNamespace(
            source_anchoring=_anchoring(starts=(38, 0)),
            tile_residency_budget_bytes=int(budget),
            output_dtype=np.dtype(np.float32),
            plan=types.SimpleNamespace(tiles=tuple(range(tiles))),
            document=None,
            colormap_lut=None,
            canonical_orientation=False,
        )

    # Two planes at exactly the share: admitted (it would reach the evaluator).
    fits = session(tiles=2, budget=int(plane_bytes * 2 / 0.5))
    # One more tile than the share holds: declined before any evaluation, so a
    # ``document=None`` session cannot even be asked to evaluate.
    over = session(tiles=3, budget=int(plane_bytes * 2 / 0.5))
    assert canonical_plane_residency_source(over, tile, shader_display=True) is None
    assert (
        canonical_plane_residency_source(session(tiles=2, budget=0), tile, shader_display=True)
        is None
    )
    with pytest.raises(AttributeError):
        # Proves the budget, not an unrelated guard, is what declined ``over``.
        canonical_plane_residency_source(fits, tile, shader_display=True)


def test_resident_crop_rebind_flag_reads_settings_object():
    """The pipeline reads the first-class setting, not a raw QSettings key."""

    import types

    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window.frame_effects import FramePipelineEffects

    win = types.SimpleNamespace(app_settings=AppSettingsState(resident_crop_rebind=True))
    effects = FramePipelineEffects(types.SimpleNamespace(win=win), session=None)
    assert effects._resident_crop_rebind_enabled() is True

    win.app_settings = AppSettingsState()  # default OFF
    other = FramePipelineEffects(types.SimpleNamespace(win=win), session=None)
    assert other._resident_crop_rebind_enabled() is False


def test_resident_crop_rebind_flag_live_toggles_without_restart():
    """A menu toggle flips the live pipeline's behavior via cache invalidation.

    The flag is snapshotted once per session (read on every retarget), so a
    menu toggle must drop that snapshot for the change to take effect on the
    next scrub without an app restart or a new session.
    """

    import dataclasses
    import types

    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window.frame_effects import FramePipelineEffects

    win = types.SimpleNamespace(app_settings=AppSettingsState())  # default OFF
    effects = FramePipelineEffects(types.SimpleNamespace(win=win), session=None)
    assert effects._resident_crop_rebind_enabled() is False

    # The menu setter replaces the settings object; the per-session snapshot
    # still reports the old value until it is invalidated.
    win.app_settings = dataclasses.replace(win.app_settings, resident_crop_rebind=True)
    assert effects._resident_crop_rebind_enabled() is False

    # Invalidation (what the menu toggle triggers on the live effects) makes the
    # next read reflect the new value — live, no restart.
    effects.invalidate_resident_crop_rebind_flag()
    assert effects._resident_crop_rebind_enabled() is True

    # And back off, again live.
    win.app_settings = dataclasses.replace(win.app_settings, resident_crop_rebind=False)
    effects.invalidate_resident_crop_rebind_flag()
    assert effects._resident_crop_rebind_enabled() is False


_EXACT_TILE_INDICES = tuple(range(50))


def _exact_source() -> np.ndarray:
    # Small enough that a ten-column montage presents every tile at native
    # resolution: the payloads are exact (quality="exact") and carry the whole
    # source plane as CPU semantics, which is the shape the canonical-plane
    # memo must serve.  Statistically uniform, so the reused level window stays
    # pixel-exact against the CPU oracle (see the module docstring).
    return np.random.default_rng(20260725).standard_normal((64, 64, 60), dtype=np.float32)


def _exact_cropped_state(win, start: int):
    state = win.view_state
    state = state.with_axis_range(
        0, indices=tuple(range(start, start + 40)), text=f"{start}:{start + 40}"
    )
    state = state.with_axis_range(1, indices=tuple(range(12, 52)), text="12:52")
    return state.with_montage_axis(2, columns=10, indices=_EXACT_TILE_INDICES, text="0:50")


def _exact_crop_settled(win, start: int) -> bool:
    session = getattr(win.renderer, "_frame_session", None)
    if session is None or session.plan is None:
        return False
    ranges = tuple(getattr(session.view_state, "axis_range_indices", None) or ())
    indices = tuple(ranges[0] or ()) if ranges else ()
    if not indices or int(indices[0]) != int(start):
        return False
    return frame_session_settled(win)


def _fill_exact_montage_then_crop(win, start: int) -> None:
    # The whole plane must be presented once: it is what makes the canonical
    # pages physically resident AND what the memo re-slices from.
    full = win.view_state.with_montage_axis(2, columns=10, indices=_EXACT_TILE_INDICES, text="0:50")
    win._set_view_state(full)
    win.update_image_view()
    _busy_pump_until(
        lambda: frame_session_settled(win),
        INTERACTION_SETTLE_HARD_LIMIT_MS * 4 / 1000.0,
        "full-plane fill",
    )
    win._apply_slice_state(
        0,
        _exact_cropped_state(win, start),
        reason="slice-range",
        interactive=True,
        immediate_axis_only=False,
    )
    _busy_pump_until(
        lambda: _exact_crop_settled(win, start),
        INTERACTION_SETTLE_HARD_LIMIT_MS * 4 / 1000.0,
        "cropped fill",
    )


def test_exact_montage_crop_scrub_rebinds_every_visible_tile(qtbot):
    """A 50-tile exact montage rebinds ALL its tiles, not the first four.

    Exact (unreduced) tile payloads carry window-local CPU semantics, so a
    rebind must re-slice them out of the memoized whole plane.  That memo is the
    only thing standing between a 50-tile montage and 50 producers per scrub
    step, and a montage needs one whole plane PER TILE — capping it by entry
    count instead of by total bytes served four tiles and left the other 46
    re-evaluating on every step (measured: 46 producers and 367-474 ms per step,
    against 0 producers and 86-110 ms once the memo is budgeted by size).
    """

    settings = use_wgpu_backend(
        extra_settings={"montage_quality_policy": "resident", "resident_crop_rebind": True}
    )
    win = make_backend_window(qtbot, _exact_source(), backend="wgpu", require_gpu_atlas=True)
    win.resize(790, 780)
    try:
        win.show()
        _fill_exact_montage_then_crop(win, 10)
        assert len(getattr(win.renderer, "_resident_crop_canonical_planes", {})) == len(
            _EXACT_TILE_INDICES
        ), "every exact tile's whole plane must be memoized, not an arbitrary prefix"

        for step in range(3):
            start = 11 + step
            before = _preparation_completed(win)
            win._on_slice_text_changed(0, f"{start}:{start + 40}")
            _busy_pump_until(
                lambda start=start: _exact_crop_settled(win, start),
                INTERACTION_SETTLE_HARD_LIMIT_MS / 1000.0,
                f"exact resident scrub {start}",
            )
            assert _preparation_completed(win) - before == 0, (
                "every tile of an exact resident crop scrub must rebind, not re-evaluate"
            )

        assert int(getattr(win.renderer, "_montage_stall_assertions", 0) or 0) == 0
        assert_wgpu_frame_matches_cpu_reference(win)
    finally:
        win.close()
        restore_default_backend(settings)


def _pinned_canonical_planes(win) -> tuple[int, int]:
    memo = getattr(win.renderer, "_resident_crop_canonical_planes", None) or {}
    return len(memo), canonical_plane_memo_bytes(memo)


def test_document_change_releases_the_pinned_canonical_planes(qtbot):
    """A new document generation unpins the memoized whole planes.

    The memo keeps STRONG references (a weak one would die the moment the crop
    replaces the whole-plane payload, which is exactly when the scrub needs it)
    and is keyed by tile number alone, so nothing about a document change made
    the retired planes go away.  Correctness never depended on it — the
    content-key check refuses to re-slice a stale plane — but up to the whole
    memo budget stayed resident for the rest of the session.
    """

    settings = use_wgpu_backend(
        extra_settings={"montage_quality_policy": "resident", "resident_crop_rebind": True}
    )
    win = make_backend_window(qtbot, _exact_source(), backend="wgpu", require_gpu_atlas=True)
    win.resize(790, 780)
    try:
        win.show()
        _fill_exact_montage_then_crop(win, 10)
        entries, pinned_bytes = _pinned_canonical_planes(win)
        assert entries == len(_EXACT_TILE_INDICES)
        assert pinned_bytes > 0

        win.notify_data_changed()
        _busy_pump_until(
            lambda: frame_session_settled(win),
            INTERACTION_SETTLE_HARD_LIMIT_MS * 4 / 1000.0,
            "settle after document change",
        )
        assert _pinned_canonical_planes(win) == (0, 0), (
            "planes of the retired document generation stayed pinned"
        )
        assert win.renderer._resident_crop_canonical_planes_release_reason == "document-changed"
        assert win.renderer._resident_crop_canonical_planes_released_bytes == pinned_bytes
    finally:
        win.close()
        restore_default_backend(settings)


def test_operation_change_releases_the_pinned_canonical_planes(qtbot):
    """An operation edit retires the pinned planes as surely as a data reload.

    ``_document_key`` folds the operation steps, so the memo's identity stamp
    moves when the pipeline is edited — the pinned planes describe pixels the
    document no longer produces.
    """

    settings = use_wgpu_backend(
        extra_settings={"montage_quality_policy": "resident", "resident_crop_rebind": True}
    )
    win = make_backend_window(qtbot, _exact_source(), backend="wgpu", require_gpu_atlas=True)
    win.resize(790, 780)
    try:
        win.show()
        _fill_exact_montage_then_crop(win, 10)
        entries, pinned_bytes = _pinned_canonical_planes(win)
        assert entries == len(_EXACT_TILE_INDICES)
        assert pinned_bytes > 0

        win.request_operation("centered_fft", 2)
        _busy_pump_until(
            lambda: frame_session_settled(win),
            INTERACTION_SETTLE_HARD_LIMIT_MS * 6 / 1000.0,
            "settle after operation change",
        )
        assert _pinned_canonical_planes(win) == (0, 0), (
            "planes of the pre-edit operation chain stayed pinned"
        )
        assert win.renderer._resident_crop_canonical_planes_released_bytes == pinned_bytes
    finally:
        win.close()
        restore_default_backend(settings)


def test_disabling_the_capability_releases_the_pinned_canonical_planes(qtbot):
    """Turning the toggle off hands the pinned memory back.

    The memo exists only to serve rebinds; with the capability off it can never
    be read again, so the menu toggle is also how a user reclaims the bytes it
    holds.
    """

    settings = use_wgpu_backend(
        extra_settings={"montage_quality_policy": "resident", "resident_crop_rebind": True}
    )
    win = make_backend_window(qtbot, _exact_source(), backend="wgpu", require_gpu_atlas=True)
    win.resize(790, 780)
    try:
        win.show()
        _fill_exact_montage_then_crop(win, 10)
        assert _pinned_canonical_planes(win)[0] == len(_EXACT_TILE_INDICES)

        win._set_resident_crop_rebind_enabled(False)
        win._on_slice_text_changed(0, "11:51")
        _busy_pump_until(
            lambda: _exact_crop_settled(win, 11),
            INTERACTION_SETTLE_HARD_LIMIT_MS * 2 / 1000.0,
            "scrub with the capability off",
        )
        assert _pinned_canonical_planes(win) == (0, 0)
        assert win.renderer._resident_crop_canonical_planes_release_reason == "capability-disabled"
    finally:
        win.close()
        restore_default_backend(settings)


def test_born_cropped_montage_warms_canonical_planes_and_rebinds(qtbot):
    """A montage that is NEVER uncropped still rebinds: it warms its own planes.

    This is the field shape (2026-07-25 diagnostics: 336x336x272 float32, raw
    scalar, crop windows on BOTH displayed axes from the first snapshot onward,
    presented at LOD level 1).  It used to be the one shape the rebind could not
    serve: nothing ever presented a whole source plane, so no canonical
    ``("wgpu-source-plane", content_key)`` page was ever uploaded, every window
    bound crop-local under an identity folding its own source rect, and the
    residency probe declined every shifted window with ``pages_not_resident``.

    The evaluation is now widened to the plane the anchor's content key already
    names (``canonical_plane_residency_source``), so the first fill uploads
    whole canonical planes and every later crop step is a pure origin rebind.
    Measured on this fixture (100 tiles, offscreen Vulkan), scrubbing the
    displayed dimension one row at a time:

    ==========================  ==========================  ================
    capability                  producers per step          ms per step
    ==========================  ==========================  ================
    off (crop-local, before)    100                         2460-6548
    on (canonical warm, after)  0                           300-365
    ==========================  ==========================  ================

    The cold born-cropped fill itself is unchanged (3707 ms off / 3698 ms on):
    widening replaces the crop-local evaluation instead of adding to it, so
    first-pixel latency does not pay for the scrub's speed.
    """

    settings = use_wgpu_backend(
        extra_settings={"montage_quality_policy": "resident", "resident_crop_rebind": True}
    )
    win = make_backend_window(qtbot, _uniform_source(), backend="wgpu", require_gpu_atlas=True)
    win.resize(780, 760)
    try:
        win.show()
        win._set_view_state(_cropped_state(win, 38))
        win.update_image_view()
        _busy_pump_until(
            lambda: _crop_settled(win, 38),
            INTERACTION_SETTLE_HARD_LIMIT_MS * 4 / 1000.0,
            "born-cropped fill",
        )
        session = win.renderer._frame_session
        payload = session.display_tile_payloads[0]
        assert payload.source_anchor.source_rect == (38, 238, 66, 266)
        assert tuple(np.shape(payload.native_residency_data)[:2]) == (336, 336), (
            "a cropped evaluation must carry the whole canonical plane for warming"
        )
        assert any(
            key.document_generation[0] == "wgpu-source-plane"
            for key in win.img_view._wgpu_executor.page_table.resident_keys()
        ), "the cropped commit must upload canonical source-plane pages"

        for step in range(3):
            start = 39 + step
            before = _preparation_completed(win)
            win._on_slice_text_changed(0, f"{start}:{start + 200}")
            _busy_pump_until(
                lambda start=start: _crop_settled(win, start),
                INTERACTION_SETTLE_HARD_LIMIT_MS / 1000.0,
                f"born-cropped scrub {start}",
            )
            assert _preparation_completed(win) - before == 0, (
                "a born-cropped scrub over warmed canonical planes must rebind, not evaluate"
            )

        totals = dict(getattr(win.renderer, "resident_crop_rebind_totals", None) or {})
        assert totals.get("gate:attempted", 0) > 0, "the seed must run, not be gated out"
        assert totals.get("rebound", 0) > 0
        assert totals.get("pages_not_resident", 0) == 0, (
            "the residency decline this workflow used to record must be gone"
        )

        # The same counters must reach the diagnostics snapshot the field
        # records: making them measurable there is the point.
        from arrayscope.window.diagnostics_snapshot import collect_runtime_diagnostics_snapshot

        snapshot = collect_runtime_diagnostics_snapshot(win)
        assert snapshot.montage.resident_crop_rebind_last_gate == "attempted"
        assert snapshot.montage.resident_crop_rebind_totals.get("rebound", 0) > 0

        assert int(getattr(win.renderer, "_montage_stall_assertions", 0) or 0) == 0
        assert_wgpu_frame_matches_cpu_reference(win)
    finally:
        win.close()
        restore_default_backend(settings)


# --- Level re-anchoring on a rebind-only scrub ------------------------------
# A rebound window is presented without any evaluation sampling it, so nothing
# in the payload path describes its value range.  The semantic level-evidence
# owner re-samples the source instead: window-exact, kernel-side, and off the
# display lane.  Gradient data is the discriminating fixture — every crop window
# has a different range, so a reused predecessor window is visible in the levels
# while statistically uniform data would hide it.

_GRADIENT_TILE_INDICES = tuple(range(50))


def _gradient_source() -> np.ndarray:
    # A row gradient shared by every tile: shifting the row window by one moves
    # the exact auto-level window by ten, so stale evidence cannot be mistaken
    # for sampling noise.
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    planes = tuple(yy * 10.0 + xx + float(index) for index in range(60))
    return np.stack(planes, axis=2).astype(np.float32)


def _gradient_window_bounds(start: int) -> tuple[float, float]:
    window = _gradient_source()[start : start + 40, 12:52, :][:, :, list(_GRADIENT_TILE_INDICES)]
    return float(window.min()), float(window.max())


def _gradient_cropped_state(win, start: int):
    state = win.view_state
    state = state.with_axis_range(
        0, indices=tuple(range(start, start + 40)), text=f"{start}:{start + 40}"
    )
    state = state.with_axis_range(1, indices=tuple(range(12, 52)), text="12:52")
    return state.with_montage_axis(2, columns=10, indices=_GRADIENT_TILE_INDICES, text="0:50")


def _gradient_crop_settled(win, start: int) -> bool:
    session = getattr(win.renderer, "_frame_session", None)
    if session is None or session.plan is None:
        return False
    ranges = tuple(getattr(session.view_state, "axis_range_indices", None) or ())
    indices = tuple(ranges[0] or ()) if ranges else ()
    if not indices or int(indices[0]) != int(start):
        return False
    return frame_session_settled(win)


def _gradient_levels_anchored(win, start: int) -> bool:
    """Settled, carrying this window's mature evidence, AND showing it.

    The tracker reaching full refined coverage and the display publishing that
    window are separate turns, so a gate on the tracker alone would sample the
    predecessor's levels and read as a re-anchor failure.
    """

    if not _gradient_crop_settled(win, start):
        return False
    session = win.renderer._frame_session
    summary = win.renderer._montage_level_tracker().summary_for(session.level_key)
    if (
        summary is None
        or int(summary.evidence_quality) != int(LevelEvidenceQuality.REFINED)
        or len(summary.source_indices) != len(_GRADIENT_TILE_INDICES)
        or summary.bounds is None
    ):
        return False
    displayed = tuple(round(float(value), 4) for value in win.img_view.getLevels())
    return displayed == tuple(round(float(value), 4) for value in summary.bounds)


def _gradient_scrub(win, starts, *, deadline_ms) -> list[dict]:
    """Fill the whole plane, crop, scrub, and report each settled step."""

    full = win.view_state.with_montage_axis(
        2, columns=10, indices=_GRADIENT_TILE_INDICES, text="0:50"
    )
    win._set_view_state(full)
    win.update_image_view()
    _busy_pump_until(
        lambda: frame_session_settled(win),
        INTERACTION_SETTLE_HARD_LIMIT_MS * 4 / 1000.0,
        "full-plane fill",
    )
    first = int(starts[0])
    win._apply_slice_state(
        0,
        _gradient_cropped_state(win, first),
        reason="slice-range",
        interactive=True,
        immediate_axis_only=False,
    )
    _busy_pump_until(
        lambda: _gradient_levels_anchored(win, first),
        INTERACTION_SETTLE_HARD_LIMIT_MS * 4 / 1000.0,
        "cropped fill",
    )

    steps = []
    for start in tuple(starts)[1:]:
        start = int(start)
        before = _preparation_completed(win)
        started = perf_counter()
        win._on_slice_text_changed(0, f"{start}:{start + 40}")
        _busy_pump_until(
            lambda start=start: _gradient_levels_anchored(win, start),
            deadline_ms / 1000.0,
            f"gradient scrub {start}",
        )
        steps.append(
            {
                "start": start,
                "producers": _preparation_completed(win) - before,
                "levels": tuple(round(float(value), 4) for value in win.img_view.getLevels()),
                "anchor_ms": (perf_counter() - started) * 1000.0,
            }
        )
    return steps


def test_rebound_crop_window_reanchors_its_own_auto_levels(qtbot):
    """A rebind-only scrub settles on the NEW window's exact auto levels.

    This is the caveat the capability was gated on.  A rebound wrapper carries
    the level evidence of the window it was sampled in, and a scrub clones
    forward from one ancestor, so admitting that evidence republished the FIRST
    window's bounds under every later window's ``level_key`` — measured on this
    fixture as 10:50's ``(112, 590)`` surviving 11:51, 12:52 and 13:53 unchanged.

    The rebound payloads are therefore evidence-blind by construction, and the
    semantic level-evidence owner re-samples the window from the source.  It is
    a statistics lane, not a display lane, so the whole point of the rebind
    survives: still ZERO display-preparation producers per step.
    """

    settings = use_wgpu_backend(
        extra_settings={"montage_quality_policy": "resident", "resident_crop_rebind": True}
    )
    win = make_backend_window(qtbot, _gradient_source(), backend="wgpu", require_gpu_atlas=True)
    win.resize(790, 780)
    try:
        win.show()
        steps = _gradient_scrub(win, (10, 11, 12, 13), deadline_ms=INTERACTION_SETTLE_HARD_LIMIT_MS)
        for step in steps:
            assert step["producers"] == 0, (
                f"window {step['start']}: re-anchoring must not schedule display producers "
                f"({step['producers']} scheduled)"
            )
            assert step["levels"] == pytest.approx(_gradient_window_bounds(step["start"])), (
                f"window {step['start']}: auto levels describe a different window "
                f"(a stale predecessor summary)"
            )

        session = win.renderer._frame_session
        assert session.level_evidence_reanchor is True
        assert session.semantic_level_evidence_diagnostics()["blocking_reason"] == "ready"
        assert all(
            payload.level_evidence_window_stale
            for payload in session.display_tile_payloads.values()
        ), "every rebound wrapper must declare that its evidence predates its window"
    finally:
        win.close()
        restore_default_backend(settings)


def test_rebind_and_evaluation_paths_settle_identical_levels(qtbot):
    """Path independence: the fast path may not change what the levels ARE.

    The rebind and the ordinary evaluation reach the same window by different
    routes — resident pages versus fresh per-tile producers — and produce their
    level evidence from different owners.  Only the schedule may differ.
    """

    def settled_levels(*, rebind: bool) -> list[tuple[float, float]]:
        settings = use_wgpu_backend(
            extra_settings={"montage_quality_policy": "resident", "resident_crop_rebind": rebind}
        )
        win = make_backend_window(qtbot, _gradient_source(), backend="wgpu", require_gpu_atlas=True)
        win.resize(790, 780)
        try:
            win.show()
            steps = _gradient_scrub(
                win,
                (10, 11, 12),
                # The evaluation path re-runs every tile, so it needs the cold
                # budget; the rebind path is measured for speed separately.
                deadline_ms=INTERACTION_SETTLE_HARD_LIMIT_MS * 4,
            )
            return [step["levels"] for step in steps]
        finally:
            win.close()
            restore_default_backend(settings)

    rebound = settled_levels(rebind=True)
    evaluated = settled_levels(rebind=False)
    assert rebound == evaluated
    assert rebound == [
        pytest.approx(_gradient_window_bounds(11)),
        pytest.approx(_gradient_window_bounds(12)),
    ]


def test_operation_pipeline_montage_keeps_the_predecessor_window_levels(qtbot):
    """The re-anchor does NOT reach operation-pipeline montages — pinned, not fixed.

    Levels for a rebound window come from the semantic evidence owner, and that
    owner is unreachable on a montage with an operation pipeline: its refinement
    work is never admitted (measured under ``CenteredFFT(axis=2)`` — armed target,
    ``verdict``/``side-work`` refusals, zero completed batches, with or without
    the rebind).  So a rebound crop window here can only be as good as its
    demoted placeholder, which is precisely why the capability stays default OFF.

    What must hold is that this degrades honestly: the placeholder keeps the
    successor population COMPLETE but IMMATURE, so the display holds the
    predecessor's window instead of republishing the ancestor window's bounds
    under this window's key, and the histogram still has a source.  Dropping the
    placeholder outright left this montage with no evidence at all.
    """

    from arrayscope.operations.pipeline import CenteredFFT

    settings = use_wgpu_backend(
        extra_settings={"montage_quality_policy": "resident", "resident_crop_rebind": True}
    )
    win = make_backend_window(qtbot, _gradient_source(), backend="wgpu", require_gpu_atlas=True)
    win.resize(790, 780)
    try:
        win.show()
        win.operation_coordinator.load_operations((CenteredFFT(axis=2),))
        win._set_document(win.operation_coordinator.document)
        win._coerce_channel_for_current_dtype()

        full = win.view_state.with_montage_axis(
            2, columns=10, indices=_GRADIENT_TILE_INDICES, text="0:50"
        )
        win._set_view_state(full)
        win.update_image_view()
        _busy_pump_until(
            lambda: frame_session_settled(win),
            INTERACTION_SETTLE_HARD_LIMIT_MS * 4 / 1000.0,
            "fft full-plane fill",
        )
        win._apply_slice_state(
            0,
            _gradient_cropped_state(win, 10),
            reason="slice-range",
            interactive=True,
            immediate_axis_only=False,
        )
        _busy_pump_until(
            lambda: _gradient_crop_settled(win, 10),
            INTERACTION_SETTLE_HARD_LIMIT_MS * 4 / 1000.0,
            "fft cropped fill",
        )
        held = tuple(round(float(value), 4) for value in win.img_view.getLevels())

        for start in (11, 12):
            win._on_slice_text_changed(0, f"{start}:{start + 40}")
            _busy_pump_until(
                lambda start=start: _gradient_crop_settled(win, start),
                INTERACTION_SETTLE_HARD_LIMIT_MS * 2 / 1000.0,
                f"fft scrub {start}",
            )
            assert tuple(round(float(value), 4) for value in win.img_view.getLevels()) == held, (
                "an unreachable re-anchor must hold the predecessor window, not remap"
            )

        session = win.renderer._frame_session
        assert session.level_evidence_reanchor is True, "the rebind must still arm the owner"
        summary = win.renderer._montage_level_tracker().summary_for(session.level_key)
        assert summary is not None, "the demoted placeholder must remain as the histogram source"
        assert int(summary.evidence_quality) == int(LevelEvidenceQuality.ROUGH_PREVIEW), (
            "a rebound window's carried evidence must never claim target quality"
        )
        assert len(summary.source_indices) == len(_GRADIENT_TILE_INDICES)
    finally:
        win.close()
        restore_default_backend(settings)
