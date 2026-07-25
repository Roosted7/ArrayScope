"""Resident crop-window scrub short-circuits to a rebind (no producer work).

Scrubbing the displayed-axis crop window of an already-resident montage plane
re-samples the same canonical GPU pages at a shifted origin.  The demand/planning
layer otherwise never consults physical residency, so every shifted window is a
fresh typed target and each tile is re-evaluated (one display-cache miss and one
producer per tile per step) even though the pixels are already resident.  With
the opt-in ``resident_crop_rebind`` capability the planner rebinds the resident
pages before the ladder plans, scheduling ZERO producers for the resident tiles.

The capability is gated OFF by default: the rebind reuses the predecessor
window's auto-level evidence (the maturity contract) instead of re-anchoring, so
it is only pixel-exact against the CPU oracle while the level window is stable
(verified here with statistically uniform data).  A crop whose pages are NOT
resident, or any pixel-affecting identity change, falls through to the ordinary
evaluation.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.tools.framebuffer_reference import assert_wgpu_frame_matches_cpu_reference
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
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


def test_cold_crop_scrub_falls_back_to_evaluation(qtbot):
    """A crop window whose pages are NOT resident keeps the ordinary evaluation.

    Starting already cropped never builds the canonical full-plane pages, so
    each shifted window is a cold local identity.  The residency probe withholds
    the rebind and the planner schedules the missing producers, proving the
    short-circuit is strictly residency-gated (partial residency => only the
    missing work runs; here nothing is resident, so all of it does).
    """

    settings = use_wgpu_backend(
        extra_settings={"montage_quality_policy": "resident", "resident_crop_rebind": True}
    )
    win = make_backend_window(qtbot, _uniform_source(), backend="wgpu", require_gpu_atlas=True)
    win.resize(780, 760)
    try:
        win.show()
        win._set_view_state(_cropped_state(win, 94))
        win.update_image_view()
        _busy_pump_until(
            lambda: _crop_settled(win, 94),
            INTERACTION_SETTLE_HARD_LIMIT_MS * 4 / 1000.0,
            "cold cropped fill",
        )
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
        # The whole plane must be presented once: it is what makes the canonical
        # pages physically resident AND what the memo re-slices from.
        full = win.view_state.with_montage_axis(
            2, columns=10, indices=_EXACT_TILE_INDICES, text="0:50"
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
            _exact_cropped_state(win, 10),
            reason="slice-range",
            interactive=True,
            immediate_axis_only=False,
        )
        _busy_pump_until(
            lambda: _exact_crop_settled(win, 10),
            INTERACTION_SETTLE_HARD_LIMIT_MS * 4 / 1000.0,
            "cropped fill",
        )
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


def test_born_cropped_montage_records_why_it_cannot_rebind(qtbot):
    """A montage that is never uncropped records the residency decline.

    This is the field shape (2026-07-25 diagnostics: 336x336x272 float32, raw
    scalar, 50 tiles, crop windows on BOTH displayed axes from the first
    snapshot onward, presented at LOD level 1).  Nothing ever presents the whole
    source plane, so no canonical ``("wgpu-source-plane", content_key)`` page is
    ever uploaded — every window binds crop-local under an identity that folds
    its own source rect — and the residency probe correctly refuses every
    shifted window.  The rebind is therefore inert for that workflow, and the
    only way to tell it apart from "the feature never ran" is the decline
    histogram this pins: ``pages_not_resident``, not a closed gate.
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
        before = _preparation_completed(win)
        win._on_slice_text_changed(0, "39:239")
        _busy_pump_until(
            lambda: _crop_settled(win, 39),
            INTERACTION_SETTLE_HARD_LIMIT_MS * 2 / 1000.0,
            "born-cropped scrub",
        )
        assert _preparation_completed(win) - before > 0

        totals = dict(getattr(win.renderer, "resident_crop_rebind_totals", None) or {})
        assert totals.get("gate:attempted", 0) > 0, "the seed must run, not be gated out"
        assert totals.get("rebound", 0) == 0
        assert totals.get("pages_not_resident", 0) > 0, (
            "the decline must name physical residency, so a field JSONL says why"
        )

        # The same counters must reach the diagnostics snapshot the field
        # records: making them measurable there is the point.
        from arrayscope.window.diagnostics_snapshot import collect_runtime_diagnostics_snapshot

        snapshot = collect_runtime_diagnostics_snapshot(win)
        assert snapshot.montage.resident_crop_rebind_last_gate == "attempted"
        assert snapshot.montage.resident_crop_rebind_totals.get("pages_not_resident", 0) > 0
    finally:
        win.close()
        restore_default_backend(settings)
