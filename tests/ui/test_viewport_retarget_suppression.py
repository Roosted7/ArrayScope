"""A camera move that cannot produce work must not replan the montage.

Trace 5 phase 4 measured ``montage_retarget`` at 47.9 % of wall clock -- 573
calls in 28 s, p50 20 ms -- with **95 % of all recorded retargets submitting
nothing**. The sub-tile pan half of that stream was already gated by
``_viewport_converged_key``; the residual was the *zoom*, because the
convergence key's zoom term was ``round(span, 4)`` and a span is continuous,
so every wheel notch invalidated it and bought a full replan of every tile
for zero submissions.

``viewport_invariant_key`` now keys the zoom on the LOD decision the camera
selects instead of on the span itself. This file is the count adjudication
for that (``docs/testing/strategy.md``: the honest measurement is a count,
never a stopwatch), and it is written as an A/B against the *previous* key so
the number has a control:

* the **arms differ only in the key**, so a difference in planner passes is
  the key and not the scheduler;
* the **final settled frame must be identical** between them -- the gate is
  allowed to change how much work was done, never what ended up on screen;
* a pan that genuinely changes the required scope must still be planned,
  because a gate that suppresses everything would pass the counting test
  perfectly;
* and the demand term is mutation-checked directly: blank it and two cameras
  at different LOD levels compare equal, which is the lost wakeup the change
  would be if the term were not there.

Trace 6 then measured that key at **18 suppressed / 451 planned, and 0 of 112
in the 4K phase**, because its tile term was the visible set's IDENTITY and
the operator's visible set swings 30 <-> 272. The term is now
``camera_pending_tiles``: of the tiles the camera reaches, the ones a retarget
could still find work for.

The arms for that live below the pyqtgraph ones and are **wgpu**, because
out-of-view retention is ``gpu_atlas``-only -- on a ``cpu_item`` surface a
tile that leaves the viewport stops being drawn, so a pan onto it is real work
and the pyqtgraph arms above measure the zoom half only. There are three, and
they are three because the gate makes two separate claims:

* **the planner would do nothing** -- every tile the camera reaches is already
  drawn at the demanded level with nothing outstanding (the counting arm);
* **the pass is not owed for anything else** -- ``retarget_viewport`` is the
  sole publisher of the session's viewport scope, so skipping it must never
  leave that scope short of the camera (the containment arm);
* and the negative arm, which pans somewhere the surface is not drawing and
  requires the retarget to run.
"""

from __future__ import annotations

import numpy as np
import pytest

from arrayscope.render.pipeline import FramePipeline
from arrayscope.tools.framebuffer_reference import assert_wgpu_frame_matches_cpu_reference
from arrayscope.window.frame_session import FrameSession
from tests.ui.helpers import clear_arrayscope_settings
from tests.ui.helpers import settle_quiescent as _settle_quiet

#: Captured at import, before any arm patches the attributes. A variant that
#: reached the live attribute would recurse into itself, and a second arm that
#: wrapped the first arm's counter would keep incrementing it.
_SHIPPED_KEY = FrameSession.viewport_invariant_key
_SHIPPED_RETARGET = FramePipeline.retarget


def _build(qtbot, depth, columns, *, size=(520, 420)):
    from pyqtgraph.Qt import QtCore

    from arrayscope.window import ArrayScopeWindow

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", "pyqtgraph")
    settings.sync()
    win = ArrayScopeWindow(np.arange(8 * 8 * depth, dtype=np.float32).reshape(8, 8, depth))
    qtbot.addWidget(win)
    win.resize(*size)
    win.show()
    _settle_quiet(win)
    win._set_view_state(
        win.view_state.with_montage_axis(2, columns=columns, indices=tuple(range(depth)), text=":")
    )
    win.update_image_view()
    _settle_quiet(win)
    return win


def _blank_demand(key):
    """The shipped key with every LOD-demand term erased."""

    return key._replace(
        desired_level=0,
        desired_factor=0,
        desired_factor_xy=(),
        acceptable_levels=(),
        demand_reason="",
        retention_level=0,
        preview_floor=0,
    )


def _span_key(self, plan, view_range, viewport_shape):
    """The predecessor: the same key with the camera SPAN as its zoom term.

    A field substitution on the shipped key rather than a copy of the old
    body, so the arms differ in exactly one thing.
    """

    key = _SHIPPED_KEY(self, plan, view_range, viewport_shape)
    if key is None:
        return None
    (x0, x1), (y0, y1) = view_range
    return _blank_demand(key)._replace(
        desired_level=round(float(x1) - float(x0), 4),
        desired_factor=round(float(y1) - float(y0), 4),
    )


def _zoom(win, factor):
    view = win.img_view.getView()
    (x0, x1), (y0, y1) = view.viewRange()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    sx, sy = (x1 - x0) * factor, (y1 - y0) * factor
    view.setRange(xRange=(cx - sx / 2, cx + sx / 2), yRange=(cy - sy / 2, cy + sy / 2), padding=0)
    win.renderer.apply_montage_viewport_retarget()
    _settle_quiet(win)


def _nudge(win):
    view = win.img_view.getView()
    (x0, x1), (y0, y1) = view.viewRange()
    dx = (x1 - x0) * 0.003
    view.setRange(xRange=(x0 + dx, x1 + dx), yRange=(y0, y1), padding=0)
    win.renderer.apply_montage_viewport_retarget()
    _settle_quiet(win)


def _presented(win):
    """What the tile layer is actually showing, per drawn tile."""

    rows = {}
    for tile_number, state in sorted(win.img_view._montage_tile_layer.states.items()):
        if not state.visible or not state.item.isVisible():
            continue
        image = getattr(state.item, "image", None)
        rows[int(tile_number)] = (
            int(state.source_index),
            None if image is None else float(np.asarray(image).sum()),
            tuple(round(float(value), 6) for value in state.levels),
            tuple(state.world_rect),
        )
    return rows


def _counted(monkeypatch):
    """Count full planner passes, always wrapping the SHIPPED implementation."""

    passes = {"n": 0, "submitted": 0}

    def counting(self, intent, demand, scope):
        passes["n"] += 1
        submitted = _SHIPPED_RETARGET(self, intent, demand, scope)
        passes["submitted"] += int(submitted)
        return submitted

    monkeypatch.setattr(FramePipeline, "retarget", counting)
    return passes


def _run(qtbot, monkeypatch, *, key):
    if key is not None:
        monkeypatch.setattr(FrameSession, "viewport_invariant_key", key)
    win = _build(qtbot, 16, 4)
    passes = _counted(monkeypatch)
    try:
        # 12 sub-tile pans and 12 sub-level zooms over a settled, fully
        # resident montage. Neither can uncover a tile and neither crosses an
        # LOD level, so every one of them is a retarget with nothing to do.
        for _ in range(12):
            _nudge(win)
        for _ in range(12):
            _zoom(win, 0.995)
        # Read every counter BEFORE the close: teardown replans, and folding
        # that into the measurement prices the gate on a window nobody is
        # looking at.
        result = (
            dict(passes),
            _presented(win),
            int(getattr(win.renderer, "_montage_viewport_retargets_suppressed", 0) or 0),
            int(getattr(win.renderer, "_montage_viewport_retargets_planned", 0) or 0),
        )
    finally:
        win.close()
    return result


def test_a_camera_move_that_cannot_produce_work_stops_replanning_the_montage(qtbot, monkeypatch):
    clear_arrayscope_settings()
    try:
        shipped, shipped_frame, suppressed, planned = _run(qtbot, monkeypatch, key=None)
        previous, previous_frame, _, _ = _run(qtbot, monkeypatch, key=_span_key)
    finally:
        clear_arrayscope_settings()

    assert previous["submitted"] == 0, (
        "the script is meant to be 24 camera moves with no work behind them; "
        f"the previous-key arm found some ({previous})"
    )
    assert shipped["submitted"] == 0, (
        "the script is meant to be 24 camera moves with no work behind them; "
        f"the shipped arm found some ({shipped})"
    )
    assert previous["n"] > 8, (
        f"the previous key ran only {previous['n']} planner passes over 24 camera "
        "moves, so there was nothing here to remove and this test measures nothing"
    )
    assert shipped["n"] * 3 < previous["n"], (
        f"planner passes {previous['n']} -> {shipped['n']}: the key change did not "
        "remove the replans it exists to remove"
    )

    # The gate is allowed to change how much work was done, never what ended
    # up on screen.
    assert shipped_frame == previous_frame

    assert suppressed > planned, (
        "a suppressed retarget must be countable in a stall dump; "
        f"the diagnostics saw {suppressed} suppressed against {planned} planned"
    )


@pytest.mark.parametrize(("depth", "columns"), [(64, 8)])
def test_a_pan_onto_tiles_that_were_never_on_screen_is_not_suppressed(
    qtbot, monkeypatch, depth, columns
):
    """The other half: the gate must not swallow a move that changes the scope.

    Without this, "planner passes went to zero" is also satisfied by a gate
    that suppresses everything, and the counting test above cannot tell the
    difference.

    The assertion is deliberately *not* a kernel submission count. On a
    persistent-residency backend the build evaluates the whole array, so tiles
    that scroll into view are already resident and a pan onto them
    legitimately admits nothing -- an assertion on submissions would be
    pinning the fixture, not the gate. What must hold is that the retarget RAN
    and that every newly required tile is on the screen afterwards.
    """

    clear_arrayscope_settings()
    win = _build(qtbot, depth, columns, size=(360, 300))
    try:
        view = win.img_view.getView()
        # Zoom in far enough that most of the montage is off screen, then
        # settle and nudge so the latch is armed on the tiles that remain.
        (x0, x1), (y0, y1) = view.viewRange()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        sx, sy = (x1 - x0) * 0.12, (y1 - y0) * 0.12
        view.setRange(
            xRange=(cx - sx / 2, cx + sx / 2), yRange=(cy - sy / 2, cy + sy / 2), padding=0
        )
        win.renderer.apply_montage_viewport_retarget()
        _settle_quiet(win)
        for _ in range(3):
            _nudge(win)
        session = win.renderer._frame_session
        assert session._viewport_converged_key is not None, (
            "the latch never engaged, so the pan below is not being asked to "
            "escape a latch and this test proves nothing"
        )
        before = set(session.required_tile_numbers())

        passes = _counted(monkeypatch)

        # Pan up the montage: rows that were never on screen are now required.
        (x0, x1), (y0, y1) = view.viewRange()
        height = y1 - y0
        view.setRange(xRange=(x0, x1), yRange=(y0 - height * 2.0, y1 - height * 2.0), padding=0)
        win.renderer.apply_montage_viewport_retarget()
        _settle_quiet(win)
        session = win.renderer._frame_session
        after = set(session.required_tile_numbers())

        uncovered = after - before
        assert uncovered, (
            "the pan reached no new tiles, so it never asked the gate the "
            f"question this test exists to ask (before={sorted(before)}, "
            f"after={sorted(after)})"
        )
        assert passes["n"] > 0, "a pan onto uncovered tiles was suppressed"
        presented = set(_presented(win))
        assert not uncovered - presented, (
            f"uncovered tiles {sorted(uncovered - presented)} never reached the screen"
        )
        assert session.required_target_settled()
    finally:
        win.close()
        clear_arrayscope_settings()


def test_the_demand_term_is_what_catches_a_zoom_across_an_lod_level():
    """Mutation check on the key change itself.

    Replacing the span with the demand is only safe because the demand is
    still THERE. Two cameras that show the same tiles at different LOD levels
    must give different keys; with the demand terms blanked they compare
    equal, and a retarget with a level change behind it is skipped with
    nothing left to come back for it.

    Qt-free on purpose: this is a statement about the key function, and
    driving it through a window would make it a statement about whichever LOD
    policy that window happened to select.
    """

    from arrayscope.core.view_state import ViewState
    from arrayscope.display.lod import LodPolicyDecision, select_lod_demand
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.presentation import TileLifecycle
    from tests.ui.helpers import viewport_invariant_key_stand_in

    state = ViewState.from_shape((64, 64, 16)).with_montage_axis(
        2, columns=4, indices=tuple(range(16)), text="0:16"
    )
    plan = make_montage_plan(
        state, axis=2, indices=tuple(range(16)), tile_shape=(64, 64), columns=4
    )
    viewport_shape = (256, 256)
    session = viewport_invariant_key_stand_in(
        lod_policy_decision=LodPolicyDecision(
            demand=select_lod_demand(None, viewport_shape, plan.tile_shape)
        ),
        lifecycle=TileLifecycle(),
        lod_preview_min_level=0,
        lod_preview_level=0,
    )

    def key_for(view_range):
        return _SHIPPED_KEY(session, plan, view_range, viewport_shape)

    # Two cameras over the SAME tile set -- the whole montage, framed tightly
    # and framed from far enough back that the demanded level is coarser.
    width = float(plan.tiles[-1].x0 + plan.tiles[-1].width)
    height = float(plan.tiles[-1].y0 + plan.tiles[-1].height)
    near = ((0.0, width), (0.0, height))
    far = ((-3.5 * width, 4.5 * width), (-3.5 * height, 4.5 * height))

    near_key, far_key = key_for(near), key_for(far)
    assert near_key is not None
    assert far_key is not None
    assert near_key.camera_pending_tiles == far_key.camera_pending_tiles, (
        "the two cameras must agree on the tile set, or the tile-set term "
        "catches the difference and the demand term is never asked"
    )
    assert near_key.desired_level != far_key.desired_level, (
        "the two cameras must differ in demanded LOD level, or neither key "
        "is being asked the question this test exists to ask"
    )

    assert near_key != far_key, "the shipped key missed an LOD level change"
    assert _blank_demand(near_key) == _blank_demand(far_key), (
        "blanking the demand terms left the keys different, so this arm is "
        "not the mutant it claims to be and the assertion above proves nothing"
    )


# --------------------------------------------------------------------------
# The case the field actually shows: a pan across tile boundaries on a settled,
# fully resident montage. WGPU, because that is where a tile that leaves the
# viewport keeps being drawn (``_retains_out_of_view_payloads`` is
# ``gpu_atlas``-only) and therefore where a newly visible tile can be work-free.
# --------------------------------------------------------------------------


#: The predecessor term, as a field substitution on the shipped key so the arms
#: differ in exactly one thing: the IDENTITY of the visible ring, which is what
#: ``25e4d750`` keyed and what trace 6 measured at 0 suppressed / 112 planned.
def _visible_identity_key(self, plan, view_range, viewport_shape):
    from arrayscope.display.montage import VISIBLE_MARGIN_TILES, viewport_tiles

    key = _SHIPPED_KEY(self, plan, view_range, viewport_shape)
    if key is None:
        return None
    return key._replace(
        camera_pending_tiles=frozenset(
            int(tile.montage_index)
            for tile in viewport_tiles(
                plan, view_range=view_range, margin_tiles=VISIBLE_MARGIN_TILES
            )
        )
    )


def _wgpu_montage(qtbot, *, depth=64, columns=8, tile=48, size=(700, 560)):
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
    _settle_quiet(win)
    return win


def _frame(view):
    """Camera plus every drawn instance: what the surface will put on screen."""

    (x0, x1), (y0, y1) = view.getView().viewRange()
    instances = view._wgpu_tile_instances()
    return (
        tuple(round(float(value), 6) for value in (x0, x1, y0, y1)),
        tuple(sorted(repr(instance) for instance in instances)),
    )


def _visible_ring(session, view):
    """The tiles the CAMERA covers, whatever the session last recorded.

    Read off the plan rather than off ``session.visible_tiles``, because a
    suppressed retarget deliberately leaves that field naming the last camera
    a full pass ran for -- so asserting on it would say the pans never moved.
    """

    from arrayscope.display.montage import VISIBLE_MARGIN_TILES, viewport_tiles

    return {
        int(tile.montage_index)
        for tile in viewport_tiles(
            session.plan, view_range=view.viewRange(), margin_tiles=VISIBLE_MARGIN_TILES
        )
    }


def _pan(win, view, fraction):
    (x0, x1), (y0, y1) = view.viewRange()
    dx = (x1 - x0) * fraction
    view.setRange(xRange=(x0 + dx, x1 + dx), yRange=(y0, y1), padding=0)
    win.renderer.apply_montage_viewport_retarget()
    _settle_quiet(win)


def _pan_run(qtbot, monkeypatch, *, key):
    if key is not None:
        monkeypatch.setattr(FrameSession, "viewport_invariant_key", key)
    win = _wgpu_montage(qtbot)
    try:
        view = win.img_view.getView()
        # Frame a fraction of the montage so a pan genuinely moves the visible
        # ring, and settle so every tile the pan can reach is already resident
        # and already drawn.
        (x0, x1), (y0, y1) = view.viewRange()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        sx, sy = (x1 - x0) * 0.22, (y1 - y0) * 0.22
        view.setRange(
            xRange=(cx - sx / 2, cx + sx / 2), yRange=(cy - sy / 2, cy + sy / 2), padding=0
        )
        win.renderer.apply_montage_viewport_retarget()
        _settle_quiet(win)

        session = win.renderer._frame_session
        reachable = set(session.required_tile_numbers())
        assert session.required_target_settled(), (
            "the montage is meant to be settled before the pans start; it is not, "
            "so the pans below are measuring a fill"
        )
        first_ring = _visible_ring(session, view)

        passes = _counted(monkeypatch)
        rings = []
        for step in range(16):
            _pan(win, view, 0.3 if step < 12 else -0.3)
            rings.append(frozenset(_visible_ring(session, view)))
        assert set().union(*rings) - first_ring, (
            "the pans never reached a tile the first camera did not, so the "
            f"gate was never asked the question (first={sorted(first_ring)}, "
            f"reached={sorted(set().union(*rings))})"
        )
        assert len(set(rings)) >= 6, (
            f"16 pans produced only {len(set(rings))} distinct visible rings, so "
            "this is measuring sub-tile pans, which the previous key already gated"
        )
        crossed = set(win.renderer._frame_session.required_tile_numbers()) | reachable
        result = (
            dict(passes),
            _frame(win.img_view),
            int(getattr(win.renderer, "_montage_viewport_retargets_suppressed", 0) or 0),
            int(getattr(win.renderer, "_montage_viewport_retargets_planned", 0) or 0),
            crossed,
        )
        assert_wgpu_frame_matches_cpu_reference(win)
    finally:
        win.close()
    return result


def test_a_pan_over_a_resident_montage_stops_replanning_it(qtbot, monkeypatch):
    """The counting adjudication for the ``camera_pending_tiles`` term.

    16 pans of 0.3 viewport widths each across tile boundaries, over a montage
    that is settled and whose tiles all stay drawn. The two assertions on the
    rings are the regime: the run has to reach tiles the first camera did not
    and to produce many distinct visible rings, or it is measuring the sub-tile
    pans the previous key already gated. None of the pans can produce work,
    which is exactly the stream trace 6 measured at 0 suppressed / 112 planned
    in the 4K phase.

    The A/B control is the predecessor term -- the same key with the visible
    ring's identity in place of the pending set -- so a difference in planner
    passes is that substitution and not the scheduler. Both arms end with the
    framebuffer checked against the CPU reference inside ``_pan_run``, which
    is the assertion that matters: a gate that suppressed a pan with work
    behind it leaves the screen stale, and a stale screen diverges there.
    """

    clear_arrayscope_settings()
    try:
        shipped, shipped_frame, suppressed, planned, shipped_reach = _pan_run(
            qtbot, monkeypatch, key=None
        )
        previous, previous_frame, _, _, previous_reach = _pan_run(
            qtbot, monkeypatch, key=_visible_identity_key
        )
    finally:
        clear_arrayscope_settings()

    assert previous["submitted"] == 0, (
        "the script is meant to be 16 pans with no work behind them; the "
        f"previous-key arm found some ({previous})"
    )
    assert shipped["submitted"] == 0, (
        "the script is meant to be 16 pans with no work behind them; the "
        f"shipped arm found some ({shipped})"
    )
    assert shipped_reach <= previous_reach, (
        "the arms must pan over the same montage; the shipped arm reached "
        f"{sorted(shipped_reach - previous_reach)} the control never did"
    )
    assert previous["n"] >= 12, (
        f"the previous key ran only {previous['n']} planner passes over 16 pans, "
        "so there was nothing here to remove and this test measures nothing"
    )
    assert shipped["n"] * 3 < previous["n"], (
        f"planner passes {previous['n']} -> {shipped['n']}: keying the camera on "
        "the tiles work is possible for did not remove the replans it exists to remove"
    )
    assert shipped_frame == previous_frame, (
        "the gate is allowed to change how much work was done, never what ended up on screen"
    )
    assert suppressed > 3 * planned, (
        "the field ratio this change exists to move is suppressed/planned; "
        f"the diagnostics saw {suppressed} suppressed against {planned} planned"
    )


def test_a_pan_onto_tiles_the_surface_is_not_drawing_still_admits_them(qtbot, monkeypatch):
    """The negative arm, on the backend the positive one runs on.

    ``camera_pending_tiles`` is only allowed to call a tile inert because the
    surface is already drawing it. Zoom in far enough that most of the montage
    is neither drawn nor resident, settle so the latch is armed on what is
    left, then pan somewhere the montage has never been shown: the retarget
    must run, and every newly required tile must reach the screen.

    Without this, "planner passes went to zero" is also satisfied by a gate
    that suppresses everything, and the counting test above cannot tell the
    difference.
    """

    clear_arrayscope_settings()
    win = _wgpu_montage(qtbot, depth=144, columns=12, tile=32, size=(420, 340))
    try:
        view = win.img_view.getView()
        (x0, x1), (y0, y1) = view.viewRange()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        sx, sy = (x1 - x0) * 0.08, (y1 - y0) * 0.08
        view.setRange(
            xRange=(cx - sx / 2, cx + sx / 2), yRange=(cy - sy / 2, cy + sy / 2), padding=0
        )
        win.renderer.apply_montage_viewport_retarget()
        _settle_quiet(win)
        for _ in range(3):
            _pan(win, view, 0.003)
        session = win.renderer._frame_session
        before = set(session.required_tile_numbers())

        passes = _counted(monkeypatch)

        # Pan across the montage: rows that were never on screen are required.
        (x0, x1), (y0, y1) = view.viewRange()
        height = y1 - y0
        view.setRange(xRange=(x0, x1), yRange=(y0 - height * 6.0, y1 - height * 6.0), padding=0)
        win.renderer.apply_montage_viewport_retarget()
        _settle_quiet(win)
        session = win.renderer._frame_session
        after = set(session.required_tile_numbers())

        uncovered = after - before
        assert uncovered, (
            "the pan reached no new tiles, so it never asked the gate the "
            f"question this test exists to ask (before={sorted(before)}, "
            f"after={sorted(after)})"
        )
        assert passes["n"] > 0, "a pan onto tiles the surface was not drawing was suppressed"
        assert session.required_target_settled()
        assert_wgpu_frame_matches_cpu_reference(win)
    finally:
        win.close()
        clear_arrayscope_settings()


def test_a_suppressed_pan_never_leaves_the_published_scope_behind_the_camera(qtbot):
    """The containment that makes skipping the pass safe rather than merely cheap.

    ``retarget_viewport`` is the sole publisher of the session's notion of
    what the user is looking at -- ``visible_tile_numbers``, the lifecycle
    targets, the level-presentation scope, and through the frame plan it
    precedes, ``required_tile_numbers``. Other actors replan over those. A
    scope carrying EXTRA tiles is conservative and harmless; a scope missing a
    tile the camera shows is a tile an actor would leave out with nothing
    coming back for it, and the camera sitting still means nothing calls the
    viewport path again.

    So the invariant is one-sided and it is checked after every pan, including
    the suppressed ones: what was published still covers what the camera
    reaches. Measured without the guard on this script, the published ring
    stayed on the FIRST camera through six pans of 0.4 viewport widths while
    the camera crossed most of the montage.
    """

    from arrayscope.display.montage import ACTIVE_MARGIN_TILES, VISIBLE_MARGIN_TILES, viewport_tiles

    clear_arrayscope_settings()
    win = _wgpu_montage(qtbot)
    try:
        view = win.img_view.getView()
        (x0, x1), (y0, y1) = view.viewRange()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        sx, sy = (x1 - x0) * 0.22, (y1 - y0) * 0.22
        view.setRange(
            xRange=(cx - sx / 2, cx + sx / 2), yRange=(cy - sy / 2, cy + sy / 2), padding=0
        )
        win.renderer.apply_montage_viewport_retarget()
        _settle_quiet(win)

        for step in range(16):
            _pan(win, view, 0.3 if step < 12 else -0.3)
            session = win.renderer._frame_session
            for margin, published, name in (
                (VISIBLE_MARGIN_TILES, set(session.visible_tile_numbers), "visible_tile_numbers"),
                (
                    ACTIVE_MARGIN_TILES,
                    set(session.required_tile_numbers()),
                    "required_tile_numbers",
                ),
            ):
                camera = {
                    int(tile.montage_index)
                    for tile in viewport_tiles(
                        session.plan, view_range=view.viewRange(), margin_tiles=margin
                    )
                }
                assert not camera - published, (
                    f"after pan {step} the camera reaches {sorted(camera - published)}, "
                    f"which {name} does not carry -- an actor replanning over that "
                    "scope would leave those tiles out and nothing would come back"
                )
        assert int(getattr(win.renderer, "_montage_viewport_retargets_suppressed", 0) or 0) > 0, (
            "no pan was suppressed, so the invariant above held for the trivial "
            "reason that every pan ran the full pass"
        )
    finally:
        win.close()
        clear_arrayscope_settings()
