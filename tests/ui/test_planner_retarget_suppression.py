"""A replan that cannot decide anything must not walk the montage.

``_try_update_montage_viewport_only`` gates the VIEWPORT caller, and the field
traces say that is a minority of the stream: 93 % of ``montage_retarget``
calls carry ``submitted=0`` while ``viewport_retargets_suppressed`` reads
48 / 341. Attributed by caller over a 272-tile wgpu workflow (fill + scroll +
zoom/pan; 558 recorded calls, 92.5 % of them ``submitted=0``, 3108 ms), the
callers that reach the planner are

    request_montage_replan's fire()              207 calls  1406 ms wasted
    _maybe_retarget_frame_session                 83         457
    _try_update_montage_viewport_only            160         757
    frame_effects.retarget_on_gui                 95         387
    replan_deferred_interactive_native_quality     7          85
    _retarget_montage_resize_payloads              6          15

-- 70 % of the calls and 76 % of the wasted milliseconds through a door the
viewport gate does not stand in. So the containment is lifted to the planner,
and this file is the count adjudication for it (``docs/testing/strategy.md``:
the honest measurement is a count, never a stopwatch).

Both halves are here, because "planner passes went to zero" is also satisfied
perfectly by a gate that suppresses everything:

* the counting arm -- a settled montage asked to replan repeatedly plans once,
  not once per request, and the pixels are identical to the ungated arm;
* the containment arm -- a camera that reaches tiles nobody has planned takes
  the full path, and those tiles reach the screen;
* and mutation checks on the two terms that carry the gate's own risk: the
  interaction flag, which is the only planner input that is not session state,
  and the out-of-camera work set, which is the one shape ``camera_pending_tiles``
  cannot see.
"""

from __future__ import annotations

import numpy as np

from arrayscope.render.pipeline import FramePipeline
from arrayscope.window.frame_controller import FrameControllerMixin
from tests.ui.helpers import clear_arrayscope_settings
from tests.ui.helpers import settle_quiescent as _settle_quiet

#: Captured at import, before any arm patches the attribute. An arm that
#: reached the live attribute would wrap whichever arm ran before it.
_SHIPPED_RETARGET = FramePipeline.retarget
_SHIPPED_GATE = FrameControllerMixin.montage_planner_pass_is_idle


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


def _ungated(monkeypatch):
    """The control arm: the same code with the gate answering "plan"."""

    monkeypatch.setattr(
        FrameControllerMixin,
        "montage_planner_pass_is_idle",
        lambda self, session, *, force_commit: None,
    )


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


def _run_replans(qtbot, monkeypatch, *, gated):
    win = _build(qtbot, 16, 4)
    if not gated:
        _ungated(monkeypatch)
    passes = _counted(monkeypatch)
    try:
        # The completion-driven caller, driven directly. On a settled montage
        # every one of these is a replan with nothing behind it -- which is
        # exactly the population the traces measure at 93 % ``submitted=0``.
        for _ in range(24):
            win.renderer.request_montage_replan(win.renderer._frame_session)
            _settle_quiet(win)
        result = (
            dict(passes),
            _presented(win),
            int(getattr(win.renderer, "_montage_planner_retargets_suppressed", 0) or 0),
            int(getattr(win.renderer, "_montage_planner_retargets_planned", 0) or 0),
        )
    finally:
        # Read every counter BEFORE the close: teardown replans, and folding
        # that into the measurement prices the gate on a window nobody is
        # looking at.
        win.close()
    return result


def test_replanning_a_settled_montage_stops_walking_every_tile(qtbot, monkeypatch):
    clear_arrayscope_settings()
    try:
        gated, gated_frame, suppressed, planned = _run_replans(qtbot, monkeypatch, gated=True)
        ungated, ungated_frame, _, _ = _run_replans(qtbot, monkeypatch, gated=False)
    finally:
        clear_arrayscope_settings()

    assert ungated["submitted"] == 0, (
        "the script is meant to be 24 replans with no work behind them; "
        f"the control arm found some ({ungated})"
    )
    assert gated["submitted"] == 0, (
        "the script is meant to be 24 replans with no work behind them; "
        f"the gated arm found some ({gated})"
    )
    assert ungated["n"] > 8, (
        f"the control arm ran only {ungated['n']} planner passes over 24 replans, "
        "so there was nothing here to remove and this test measures nothing"
    )
    assert gated["n"] * 3 < ungated["n"], (
        f"planner passes {ungated['n']} -> {gated['n']}: the gate did not remove "
        "the replans it exists to remove"
    )

    # The gate is allowed to change how much work was done, never what ended
    # up on screen.
    assert gated_frame == ungated_frame

    assert suppressed > planned, (
        "a suppressed planner pass must be countable in a stall dump; "
        f"the diagnostics saw {suppressed} suppressed against {planned} planned"
    )


def test_a_camera_that_reaches_unplanned_tiles_still_takes_the_full_path(qtbot, monkeypatch):
    """The containment half: the gate must not swallow a real scope change.

    The assertion is deliberately not a kernel submission count -- on a
    persistent-residency backend a pan onto already-resident tiles
    legitimately admits nothing, so a submission count would pin the fixture
    rather than the gate. What must hold is that the planner RAN and that
    every newly required tile is on the screen afterwards.
    """

    clear_arrayscope_settings()
    win = _build(qtbot, 64, 8, size=(360, 300))
    try:
        view = win.img_view.getView()
        (x0, x1), (y0, y1) = view.viewRange()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        sx, sy = (x1 - x0) * 0.12, (y1 - y0) * 0.12
        view.setRange(
            xRange=(cx - sx / 2, cx + sx / 2), yRange=(cy - sy / 2, cy + sy / 2), padding=0
        )
        win.renderer.apply_montage_viewport_retarget()
        _settle_quiet(win)
        for _ in range(3):
            win.renderer.request_montage_replan(win.renderer._frame_session)
            _settle_quiet(win)
        assert getattr(win.renderer, "_montage_planner_converged", None) is not None, (
            "the planner latch never engaged, so the pan below is not being "
            "asked to escape a latch and this test proves nothing"
        )
        session = win.renderer._frame_session
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
        assert passes["n"] > 0, "a pan onto tiles nobody had planned was suppressed"
        presented = set(_presented(win))
        assert not uncovered - presented, (
            f"uncovered tiles {sorted(uncovered - presented)} never reached the screen"
        )
        assert session.required_target_settled()
    finally:
        win.close()
        clear_arrayscope_settings()


def test_work_in_flight_outside_the_camera_refuses_the_gate(qtbot):
    """Mutation check on ``_montage_planner_outstanding_is_camera_local``.

    ``camera_pending_tiles`` answers for the tiles the camera reaches. The
    planner's scope is wider -- ``_lod_admission_scope`` folds ``loading_tiles``
    and ``active_tile_requests`` into coverage wherever they are -- so a slot
    in flight somewhere the camera is not looking is the one shape the tile
    statement cannot see. Remove the clause and it is a lost wakeup.
    """

    clear_arrayscope_settings()
    win = _build(qtbot, 16, 4)
    try:
        renderer = win.renderer
        session = renderer._frame_session
        for _ in range(3):
            renderer.request_montage_replan(session)
            _settle_quiet(win)
        assert getattr(renderer, "_montage_planner_converged", None) is not None
        assert renderer._montage_planner_outstanding_is_camera_local(session)

        session.loading_tiles.add(9_999)
        assert not renderer._montage_planner_outstanding_is_camera_local(session)
        # The epoch arm may legitimately still hold -- "no session write since
        # the last pass" says nothing about the tile statement, and the write
        # above is one. What must not survive is the CAMERA arm, so make the
        # latched epoch stale and leave only that arm to answer.
        latched = renderer._montage_planner_converged
        renderer._montage_planner_converged = (
            latched[0],
            int(latched[1]) - 1,
            latched[2],
            latched[3],
        )
        assert renderer.montage_planner_pass_is_idle(session, force_commit=False) is None
        session.loading_tiles.discard(9_999)
    finally:
        win.close()
        clear_arrayscope_settings()


def test_the_interaction_flag_is_what_lets_a_gesture_end_replan(qtbot):
    """Mutation check on the one planner input that is not session state.

    Native-quality rungs are deferred while an interaction is in flight
    (``FramePipeline._defer_native_quality_during_interaction``), and the edge
    that admits them is the gesture ENDING --
    ``replan_deferred_interactive_native_quality``. Nothing writes to the
    session at that edge, so without the flag in the guard the gate would
    suppress precisely the pass that exists to un-defer the work.
    """

    clear_arrayscope_settings()
    win = _build(qtbot, 16, 4)
    try:
        renderer = win.renderer
        session = renderer._frame_session
        for _ in range(3):
            renderer.request_montage_replan(session)
            _settle_quiet(win)
        assert renderer.montage_planner_pass_is_idle(session, force_commit=False) is not None

        win._viewport_interaction_active = True
        try:
            assert renderer.montage_planner_pass_is_idle(session, force_commit=False) is None, (
                "an interaction starting left the latch holding, so the pass "
                "that un-defers native quality when it ends can be suppressed"
            )
        finally:
            win._viewport_interaction_active = False
    finally:
        win.close()
        clear_arrayscope_settings()


def test_a_tile_the_camera_reaches_and_still_needs_refuses_the_camera_arm(qtbot):
    """Mutation check on the tile term itself.

    Blinding ``camera_pending_tiles`` is not a hypothetical: driven through a
    real fill it strands the montage -- one tile short of its target, nothing
    loading, no flush pending, and the stall watchdog the only thing that
    notices (``tests/ui/test_viewport_convergence_latch.py`` documents that
    run). This is the same statement in one assertion, so a regression names
    the term instead of arriving as a settle timeout somewhere else.
    """

    clear_arrayscope_settings()
    win = _build(qtbot, 16, 4)
    try:
        renderer = win.renderer
        session = renderer._frame_session
        for _ in range(3):
            renderer.request_montage_replan(session)
            _settle_quiet(win)
        latched = renderer._montage_planner_converged
        assert latched is not None
        # Make the epoch stale so only the camera arm can answer, then put one
        # tile the camera reaches back into the pending set.
        renderer._montage_planner_converged = (
            latched[0],
            int(latched[1]) - 1,
            latched[2],
            latched[3],
        )
        assert renderer.montage_planner_pass_is_idle(session, force_commit=False) is not None

        shipped_key = type(session).viewport_invariant_key

        def one_pending(self, plan, view_range, viewport_shape):
            key = shipped_key(self, plan, view_range, viewport_shape)
            return None if key is None else key._replace(camera_pending_tiles=frozenset({0}))

        try:
            type(session).viewport_invariant_key = one_pending
            assert renderer.montage_planner_pass_is_idle(session, force_commit=False) is None, (
                "a tile the camera reaches that still needs work was called idle"
            )
        finally:
            type(session).viewport_invariant_key = shipped_key
    finally:
        win.close()
        clear_arrayscope_settings()


def test_the_gate_is_restored_between_arms():
    """The control arm patches a class attribute; prove it is the shipped one."""

    assert FrameControllerMixin.montage_planner_pass_is_idle is _SHIPPED_GATE
    assert FramePipeline.retarget is _SHIPPED_RETARGET
