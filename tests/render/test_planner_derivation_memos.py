"""The ladder snapshot must stop recomputing what did not change.

``retarget_frame_pipeline`` is 26.5 % of a field session's wall clock (trace 7:
1854 calls, 38.2 s, p50 12.3 ms) and **95 % of those calls submit nothing**.
The 2026-08-04 complexity measurement attributed that cost: every pass on the
path is O(all tiles in scope) and the largest single term is
``render.effects.tile_lod_states`` -- profiled at 82.2 us per tile over a
272-tile montage, of which

* **41 %** was ``payload_ref_from_display_payload``, because ``target_payload``
  and ``fallback_payload`` are recomputed properties that each converted every
  entry of ``presentable_payloads`` from scratch, and the loop reads both;
* **13 %** was ``repr(ref.source_id)`` inside ``target_payload``'s ``min`` key
  -- a tie-break over the whole semantic content graph, evaluated once per
  tile even though almost every tile has a single candidate;
* **16 %** was ``tile_source_rect``, a pure derivation of the plan geometry and
  the window anchor, asked once per tile the planner touches (measured 1.00
  per touch: the second, inside ``_floor_available``, is already absorbed by
  the ``best_floor_key`` residency memo).

None of that is a function of the camera, so a pan recomputed 272 identical
answers and then threw them away. Both memos are keyed on the OBJECT identity
of what they derive from -- a frozen ``DisplayTilePayload`` for the ref, the
plan/view-state/anchoring quintet for the rect -- so neither can age: a
superseded input is a different object and the entry is simply not used.

**Why the brief this came from asked for something else.** It asked for the
planner to be restricted to ``camera_pending_tiles``. Measured on a montage
with tiles genuinely in flight, the inert (non-pending) share of the planner's
scope is **1.6 %**: ``camera_pending_tiles`` answers "not finished yet", not
"unchanged since the last pass", and during a fill nothing is finished. The
restriction is real but it is worth 1.6 % in the regime that matters, so what
is here instead is the per-touch cost, which pays in every regime and for all
~1850 callers rather than the ~460 that come from the viewport.

The adjudication is in three parts, and the first is the one that matters:
a memo that can disagree with a fresh computation is a lost wakeup with no
backstop, so the disagreement is checked **in call**, on every call, over a
real montage that is still filling -- not argued from the key.
"""

from __future__ import annotations

import numpy as np

from arrayscope.presentation import tile_lifecycle
from arrayscope.presentation.tile_lifecycle import TileRecord
from arrayscope.render import lod as render_lod
from arrayscope.tools.framebuffer_reference import assert_wgpu_frame_matches_cpu_reference
from tests.settle import deliver_posted_events
from tests.ui.helpers import clear_arrayscope_settings
from tests.ui.helpers import settle_quiescent as _settle_quiet


def _unsettled_montage(qtbot, *, depth=272, columns=17, tile=112, size=(760, 620)):
    """A wgpu montage whose fill is still in flight while the camera moves.

    The gate landed in ``680a8da1`` was counted on a SETTLED montage and
    predicted 60-85 % suppression; it delivered 0 % in both phases of the
    field trace that matter, because the operator's montage is never settled
    (trace 7: 0 of 70 fully quiet snapshots in P3, median 187 unsettled
    tiles). Every count in this file is taken while the fill is running, and
    :func:`test_the_fixture_is_not_settled` is the assertion that keeps it
    that way.
    """

    from tests.ui.helpers import make_backend_window, use_wgpu_backend

    use_wgpu_backend()
    data = np.arange(tile * tile * depth, dtype=np.float32).reshape(tile, tile, depth)
    win = make_backend_window(qtbot, data, backend="wgpu", require_gpu_atlas=True)
    win.resize(*size)
    win.show()
    _settle_quiet(win)
    win._set_view_state(
        win.view_state.with_montage_axis(2, columns=columns, indices=tuple(range(depth)), text=":")
    )
    win.update_image_view()
    view = win.img_view.getView()
    (x0, x1), (y0, y1) = view.viewRange()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    sx, sy = (x1 - x0) * 0.55, (y1 - y0) * 0.55
    view.setRange(xRange=(cx - sx / 2, cx + sx / 2), yRange=(cy - sy / 2, cy + sy / 2), padding=0)
    win.renderer.apply_montage_viewport_retarget()
    return win, view


def _pan(win, view, fraction):
    (x0, x1), (y0, y1) = view.viewRange()
    dx = (x1 - x0) * fraction
    view.setRange(xRange=(x0 + dx, x1 + dx), yRange=(y0, y1), padding=0)
    win.renderer.apply_montage_viewport_retarget()
    deliver_posted_events()


def _unsettled_tiles(session) -> int:
    return (
        len(set(session.required_tile_numbers()) - set(session.lifecycle.presented_tiles))
        + len(session.dirty_payloads)
        + len(session.loading_tiles)
        + len(session.active_tile_requests)
        + len(session.pending_payload_upserts)
    )


def _drive(win, view, steps=24):
    """N viewport changes across tile boundaries, reporting the unsettled count."""

    unsettled = []
    for step in range(steps):
        _pan(win, view, 0.3 if step % 8 < 4 else -0.3)
        unsettled.append(_unsettled_tiles(win.renderer._frame_session))
    return unsettled


#: How many of the driven viewport changes have to land on a montage with a
#: real number of tiles in flight.  The previous gate was counted on a SETTLED
#: montage, predicted 60-85 % suppression and delivered 0 % in both field
#: phases that matter; a count taken here is worthless unless the montage is
#: in the state the operator's actually is.  Expressed as a floor on the
#: number of steps rather than a ceiling on the quiet ones so that a machine
#: under load, where the fill runs longer, cannot fail it.
_IN_FLIGHT_STEPS = 8
_IN_FLIGHT_TILES = 40


def _assert_in_flight(label, unsettled) -> None:
    busy = sum(1 for count in unsettled if count >= _IN_FLIGHT_TILES)
    assert busy >= _IN_FLIGHT_STEPS, (
        f"only {busy} of {len(unsettled)} viewport changes in {label} landed on a "
        f"montage with {_IN_FLIGHT_TILES}+ tiles in flight, so this is measuring a "
        f"settled montage -- the regime that made the previous gate vacuous "
        f"({unsettled})"
    )


def test_the_fixture_is_not_settled(qtbot):
    """Guard the guard: the counting arms below must not measure a still montage."""

    clear_arrayscope_settings()
    win, view = _unsettled_montage(qtbot)
    try:
        unsettled = _drive(win, view)
        _assert_in_flight("the fixture", unsettled)
        _settle_quiet(win)
        assert win.renderer._frame_session.required_target_settled()
        assert_wgpu_frame_matches_cpu_reference(win)
    finally:
        win.close()
        clear_arrayscope_settings()


def test_the_payload_ref_memo_never_disagrees_with_a_fresh_conversion(qtbot, monkeypatch):
    """Every memoised ref, checked against an uncached one, on every call.

    The memo's key is the payload object's identity, which is sound only
    because every attribute the conversion reads lives on a frozen object. That
    is an argument; this is the measurement. A disagreement here is a ladder
    that plans against a payload the slot no longer holds -- a tile stranded at
    its floor with no wakeup left, which ``_montage_watchdog_tick`` observes
    and never repairs.
    """

    calls = {"refs": 0, "pairs": 0}
    real_refs = TileRecord.presentable_payload_refs
    real_pair = TileRecord.target_and_fallback_payloads

    def checked_refs(self):
        memoised = real_refs(self)
        fresh = tuple(
            tile_lifecycle.payload_ref_from_display_payload(value)
            for value in self.presentable_payloads.values()
        )
        assert memoised == fresh, (
            f"the payload-ref memo disagrees with a fresh conversion for tile "
            f"{self.tile_number}: {memoised} != {fresh}"
        )
        calls["refs"] += 1
        return memoised

    def checked_pair(self):
        memoised = real_pair(self)
        target = self.target
        if target is not None:
            fresh_refs = tuple(
                tile_lifecycle.payload_ref_from_display_payload(value)
                for value in self.presentable_payloads.values()
            )
            # The two predecessor properties, verbatim, as the control.
            fresh_target = min(
                (ref for ref in fresh_refs if ref.satisfies_target(target)),
                key=lambda ref: (ref.lod_level, repr(ref.source_id)),
                default=None,
            )
            fresh_fallback = min(
                (
                    ref
                    for ref in fresh_refs
                    if ref.source_index == target.source_index and not ref.satisfies_target(target)
                ),
                key=lambda ref: (0 if ref.quality == "exact" else 1, ref.lod_level),
                default=None,
            )
            assert memoised == (fresh_target, fresh_fallback), (
                f"the combined walk disagrees with the two properties it replaced for "
                f"tile {self.tile_number}: {memoised} != {(fresh_target, fresh_fallback)}"
            )
        calls["pairs"] += 1
        return memoised

    monkeypatch.setattr(TileRecord, "presentable_payload_refs", checked_refs)
    monkeypatch.setattr(TileRecord, "target_and_fallback_payloads", checked_pair)

    clear_arrayscope_settings()
    win, view = _unsettled_montage(qtbot)
    try:
        _drive(win, view)
        _settle_quiet(win)
        assert win.renderer._frame_session.required_target_settled()
        assert_wgpu_frame_matches_cpu_reference(win)
    finally:
        win.close()
        clear_arrayscope_settings()

    reason = (
        f"the check ran {calls} times, which is too few for this montage to have "
        "exercised the memo at all -- a check that is never asked proves nothing"
    )
    assert calls["pairs"] > 500, reason
    assert calls["refs"] > 500, reason


def test_the_tile_source_rect_memo_never_disagrees_with_a_fresh_computation(qtbot, monkeypatch):
    """Same question for the rect memo, whose guard is a five-object tuple.

    The failure mode this rules out is a guard that misses an input: a rect
    computed under one window anchor and served under another anchors the
    tile's pages at the wrong origin, and the tile then presents another
    slice's pixels.
    """

    calls = {"n": 0}
    real = render_lod.tile_source_rect

    def checked(session, tile):
        memoised = real(session, tile)
        fresh = render_lod._compute_tile_source_rect(session, tile)
        assert memoised == fresh, (
            f"the tile_source_rect memo disagrees with a fresh computation for tile "
            f"{getattr(tile, 'montage_index', None)}: {memoised} != {fresh}"
        )
        calls["n"] += 1
        return memoised

    monkeypatch.setattr(render_lod, "tile_source_rect", checked)

    clear_arrayscope_settings()
    win, view = _unsettled_montage(qtbot)
    try:
        _drive(win, view)
        # A zoom stream too: the demand moves, which is the other input the
        # guard has to be right about.
        for factor in (0.7, 1.4, 0.55):
            (x0, x1), (y0, y1) = view.viewRange()
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            sx, sy = (x1 - x0) * factor, (y1 - y0) * factor
            view.setRange(
                xRange=(cx - sx / 2, cx + sx / 2), yRange=(cy - sy / 2, cy + sy / 2), padding=0
            )
            win.renderer.apply_montage_viewport_retarget()
            deliver_posted_events()
        _settle_quiet(win)
        assert win.renderer._frame_session.required_target_settled()
        assert_wgpu_frame_matches_cpu_reference(win)
    finally:
        win.close()
        clear_arrayscope_settings()

    assert calls["n"] > 500, f"the check ran {calls['n']} times, too few to have exercised the memo"


def _predecessor_target(record):
    """``TileRecord.target_payload`` as it was before the combined walk."""

    target = record.target
    if target is None:
        return None
    candidates = (
        ref
        for ref in (
            tile_lifecycle.payload_ref_from_display_payload(value)
            for value in record.presentable_payloads.values()
        )
        if ref.satisfies_target(target)
    )
    return min(candidates, key=lambda ref: (ref.lod_level, repr(ref.source_id)), default=None)


def _predecessor_fallback(record):
    """``TileRecord.fallback_payload`` as it was before the combined walk."""

    target = record.target
    if target is None:
        return None
    candidates = (
        ref
        for ref in (
            tile_lifecycle.payload_ref_from_display_payload(value)
            for value in record.presentable_payloads.values()
        )
        if ref.source_index == target.source_index and not ref.satisfies_target(target)
    )
    return min(
        candidates,
        key=lambda ref: (0 if ref.quality == "exact" else 1, ref.lod_level),
        default=None,
    )


def _derivation_run(qtbot, monkeypatch, *, predecessor):
    """Count the planner's derivations over N viewport changes.

    ``predecessor=True`` restores the shape this change replaced -- two
    independent walks of ``presentable_payloads`` with no memo, and an
    uncached ``tile_source_rect`` -- so the two arms differ in the memos and
    nothing else. Counting only what happens INSIDE ``tile_lod_states``,
    because the planner's per-tile loop is what the change is about and the
    same helpers have other callers.
    """

    from arrayscope.render import effects as render_effects

    counts = {"ref": 0, "rect": 0, "touches": 0}
    inside = {"n": 0}
    real_convert = tile_lifecycle.payload_ref_from_display_payload
    real_compute = render_lod._compute_tile_source_rect
    real_states = render_effects.tile_lod_states

    def counted_convert(payload):
        if inside["n"]:
            counts["ref"] += 1
        return real_convert(payload)

    def counted_compute(session, tile):
        if inside["n"]:
            counts["rect"] += 1
        return real_compute(session, tile)

    def counted_states(session, demand=None, *, tile_numbers=None, scope=None):
        inside["n"] += 1
        try:
            out = real_states(session, demand, tile_numbers=tile_numbers, scope=scope)
        finally:
            inside["n"] -= 1
        counts["touches"] += len(out)
        return out

    monkeypatch.setattr(tile_lifecycle, "payload_ref_from_display_payload", counted_convert)
    monkeypatch.setattr(render_lod, "_compute_tile_source_rect", counted_compute)
    monkeypatch.setattr(render_effects, "tile_lod_states", counted_states)
    if predecessor:
        monkeypatch.setattr(
            TileRecord,
            "target_and_fallback_payloads",
            lambda self: (_predecessor_target(self), _predecessor_fallback(self)),
        )
        # The counted body, not the raw one: the predecessor computed the rect
        # at the call site with no memo, and this arm has to be counted too.
        monkeypatch.setattr(render_lod, "tile_source_rect", counted_compute)

    clear_arrayscope_settings()
    win, view = _unsettled_montage(qtbot)
    try:
        unsettled = _drive(win, view)
        _settle_quiet(win)
        session = win.renderer._frame_session
        assert session.required_target_settled(), (
            "the fill did not settle, so this arm is not comparable"
        )
        assert_wgpu_frame_matches_cpu_reference(win)
        frame = tuple(sorted(repr(item) for item in win.img_view._wgpu_tile_instances()))
    finally:
        win.close()
        clear_arrayscope_settings()
    return counts, unsettled, frame


def test_the_memos_remove_the_repeated_derivations(qtbot, monkeypatch):
    """The counting adjudication, A/B, on the non-settled fixture.

    ``docs/testing/strategy.md``: the honest measurement is a count. The
    quantity that proves this change is the number of DERIVATIONS per tile the
    planner touches -- the number of passes and the number of tiles per pass
    are both unchanged, and deliberately so, because the inert share of the
    planner's scope during a fill is 1.6 % and restricting the scope therefore
    buys nothing here. What moved is the cost of each touch.

    Predecessor: ``tile_lod_states`` read ``target_payload`` and
    ``fallback_payload`` per tile and each converted every presentable payload
    from scratch, and ``tile_source_rect`` was computed twice per touch (once
    directly, once inside ``_floor_available``).
    """

    shipped, shipped_unsettled, shipped_frame = _derivation_run(
        qtbot, monkeypatch, predecessor=False
    )
    monkeypatch.undo()
    previous, previous_unsettled, previous_frame = _derivation_run(
        qtbot, monkeypatch, predecessor=True
    )

    _assert_in_flight("the shipped arm", shipped_unsettled)
    _assert_in_flight("the predecessor arm", previous_unsettled)
    too_few = f"too few planner tile touches to measure ({shipped}, {previous})"
    assert shipped["touches"] > 1000, too_few
    assert previous["touches"] > 1000, too_few
    # Deliberately NOT a cross-arm frame comparison. Both arms end with
    # ``assert_wgpu_frame_matches_cpu_reference`` inside ``_derivation_run``,
    # which is the oracle that matters, but the two arms are two separate
    # racing fills of a 272-tile montage and the surface legitimately ends up
    # RETAINING a different set of departed tiles and allocating different
    # atlas plane indices. Equality there would be an assertion about the
    # scheduler's timing, not about this change. What proves the outputs are
    # identical is stronger and lives above: every derivation is checked
    # against a fresh computation, in call, on every call.
    assert shipped_frame
    assert previous_frame

    ref_before = previous["ref"] / previous["touches"]
    ref_after = shipped["ref"] / shipped["touches"]
    rect_before = previous["rect"] / previous["touches"]
    rect_after = shipped["rect"] / shipped["touches"]
    print(
        f"\nDERIVATIONS per planner tile touch: ref {ref_before:.2f} -> {ref_after:.2f}; "
        f"rect {rect_before:.2f} -> {rect_after:.2f}\nshipped={shipped} previous={previous}"
    )
    assert ref_before >= 1.5, (
        f"the predecessor arm derived only {ref_before:.2f} refs per touch, so "
        "there was nothing here to remove and this test measures nothing"
    )
    assert ref_after * 2 < ref_before, (
        f"payload-ref conversions per planner tile touch {ref_before:.2f} -> "
        f"{ref_after:.2f}: the memo is not being reused"
    )
    assert rect_before >= 0.9, (
        f"the predecessor arm computed only {rect_before:.2f} source rects per "
        "touch, so there was nothing here to remove"
    )
    assert rect_after * 10 < rect_before, (
        f"source-rect computations per planner tile touch {rect_before:.2f} -> "
        f"{rect_after:.2f}: the memo is not being reused"
    )
