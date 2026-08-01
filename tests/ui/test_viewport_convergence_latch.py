"""The converged-viewport early-out must never change what PyQtGraph shows.

``viewport_invariant_key`` (``frame_session.py``) lets a camera move that
crosses no tile boundary and no zoom threshold skip the whole retarget. The
key captures plan identity, the visible tile set and the camera span, and
nothing else -- so the question this ring answers is whether a state change the
key does NOT capture can be silently skipped on a backend that maps levels on
the CPU.

The load-bearing test is the differential one: run one scripted interaction
twice, once with the latch and once without, and compare what the tile layer
actually has on screen -- baked pixels, LUT, levels, LOD scale and geometry per
visible tile. Asserting only that the latch fires when nothing changed proves
the cheap half.

Three things keep this from being a test that cannot fail:

* a **control** arm (latch vs latch) that must agree, because a step whose
  round is still mid-flight is legitimately non-deterministic and a differential
  claim there measures the scheduler, not the latch;
* a **mutant** arm whose key forgets the visible tile set, which must be
  caught -- if the oracle cannot see a montage going stale under a pan, it
  cannot see anything;
* an assertion that the latch actually **engaged**, so the comparison is not
  between two identical unlatched runs.

Physical evidence for the same script at 272 tiles, on a real Weston output:
the montage area is byte-identical between the arms, and the only differing
pixels (1093, in the 13-row status strip) carry a diff mask byte-identical to
the latch-vs-latch control's. Recipe in
``docs/redesign/render-eagerness-2026-07-30.md`` §2.6.
"""

from __future__ import annotations

import hashlib
from time import monotonic

import numpy as np
import pytest

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_S
from tests.ui.helpers import clear_arrayscope_settings
from tests.ui.helpers import (
    settle_quiescent as _settle_quiet,
)

# The per-step wait keeps a capped owner ON PURPOSE, which is why it is a
# stopwatch and the build-time fill is not. A step that outlives the cap is not
# widened into green -- it is recorded as unsettled and excluded from the
# differential, and the one-sided settle assertion still fails if the LATCH is
# what made it outlive the cap. `_drain` therefore must return a verdict rather
# than raise, so it cannot be denominated in progress the way `_build` is.
STEP_SETTLE_BUDGET_S = INTERACTION_SETTLE_HARD_LIMIT_S


def _digest(array):
    if array is None:
        return None
    return hashlib.blake2b(
        np.ascontiguousarray(np.asarray(array)).tobytes(), digest_size=8
    ).hexdigest()


def _fingerprint(win):
    """What PyQtGraph has on screen, per visible tile."""

    rows = {}
    for tile_number, state in sorted(win.img_view._montage_tile_layer.states.items()):
        if not state.visible or not state.item.isVisible():
            continue
        rows[int(tile_number)] = (
            int(state.source_index),
            _digest(getattr(state.item, "image", None)),
            _digest(getattr(state.item, "lut", None)),
            tuple(round(float(value), 6) for value in state.levels),
            tuple(round(float(value), 6) for value in state.lod_scale),
            tuple(state.world_rect),
        )
    return rows


def _round_settled(win):
    session = win.renderer._frame_session
    if session is None:
        return False
    return bool(
        session.required_target_settled()
        and not session.has_pending_level_update()
        and not session.dirty_payloads
        and not session.pending_payload_upserts
        and not session.pending_removals
        and not win.montage_tile_evaluation_controller.is_busy()
    )


# Every wait in this file is `settle_quiescent`, and it is the clearest case in
# the suite for it: this is an A/B that compares two arms' committed frames, so
# a gesture that leaves a retarget still armed when the frame is sampled makes
# the arms differ for a reason that has nothing to do with the latch. A settled
# session is not enough -- it says the work that HAS run is done, not that none
# is queued behind it.
def _drain(win, qtbot):
    """Let the round finish. False when it never does."""

    deadline = monotonic() + STEP_SETTLE_BUDGET_S
    while monotonic() < deadline:
        if _round_settled(win):
            _settle_quiet(win)
            if _round_settled(win):
                return True
        _settle_quiet(win)
    return _round_settled(win)


def _build(qtbot, depth, columns):
    from pyqtgraph.Qt import QtCore

    from arrayscope.window import ArrayScopeWindow

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", "pyqtgraph")
    settings.sync()
    win = ArrayScopeWindow(np.arange(8 * 8 * depth, dtype=np.float32).reshape(8, 8, depth))
    qtbot.addWidget(win)
    win.resize(520, 420)
    win.show()
    _settle_quiet(win)
    win._set_view_state(
        win.view_state.with_montage_axis(2, columns=columns, indices=tuple(range(depth)), text=":")
    )
    win.update_image_view()
    # The cold fill is not a gesture, so it gets no stopwatch: `_settle_quiet`
    # already requires `required_target_settled()` (via `frame_session_settled`),
    # so the loop that used to sit here re-asked a question its own body had
    # just answered, and the assertion after it could only ever pass. A fill
    # that never completes now fails as a SettleTimeout naming this line.
    _settle_quiet(win)
    return win


def _nudge(win, qtbot):
    """A sub-tile pan: crosses no tile boundary, so the key must not change."""

    view = win.img_view.getView()
    (x0, x1), (y0, y1) = view.viewRange()
    dx = (x1 - x0) * 0.003
    view.setRange(xRange=(x0 + dx, x1 + dx), yRange=(y0, y1), padding=0)
    win.renderer.apply_montage_viewport_retarget()
    _settle_quiet(win)


def _set_levels(win, qtbot, scale):
    low, high = tuple(win.img_view._displayLevels or (0.0, 1.0))
    win.img_view.setLevels(float(low), float(low) + (float(high) - float(low)) * scale)
    win.renderer.apply_montage_viewport_retarget()
    _settle_quiet(win)


def _set_colormap(win, qtbot, name):
    win.renderer._set_display_colormap(name, user_selected=True, request_render=True)
    win.renderer.apply_montage_viewport_retarget()
    _settle_quiet(win)


def _auto_window(win, qtbot):
    win.auto_window_levels()
    win.renderer.apply_montage_viewport_retarget()
    _settle_quiet(win)


def _zoom_in(win, qtbot):
    view = win.img_view.getView()
    (x0, x1), (y0, y1) = view.viewRange()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    sx, sy = (x1 - x0) * 0.16, (y1 - y0) * 0.16
    view.setRange(xRange=(cx - sx / 2, cx + sx / 2), yRange=(cy - sy / 2, cy + sy / 2), padding=0)
    win.renderer.apply_montage_viewport_retarget()
    _settle_quiet(win)


def _pan_a_whole_tile(win, qtbot):
    """A pan across a tile boundary: the visible set MUST change."""

    view = win.img_view.getView()
    (x0, x1), (y0, y1) = view.viewRange()
    dx = x1 - x0
    view.setRange(xRange=(x0 + dx, x1 + dx), yRange=(y0, y1), padding=0)
    win.renderer.apply_montage_viewport_retarget()
    _settle_quiet(win)


# Every entry that is not a nudge changes something the invariant key does not
# capture: baked levels, the colormap LUT, or the automatic window.
SCRIPT = (
    ("nudge", _nudge),
    ("nudge", _nudge),
    ("nudge", _nudge),
    ("levels-0.5", lambda w, q: _set_levels(w, q, 0.5)),
    ("nudge", _nudge),
    ("nudge", _nudge),
    ("colormap-viridis", lambda w, q: _set_colormap(w, q, "viridis")),
    ("nudge", _nudge),
    ("levels-0.25", lambda w, q: _set_levels(w, q, 0.25)),
    ("nudge", _nudge),
    ("nudge", _nudge),
    ("auto-window", _auto_window),
    ("nudge", _nudge),
    ("nudge", _nudge),
    ("colormap-gray", lambda w, q: _set_colormap(w, q, "gray")),
    ("nudge", _nudge),
    ("nudge", _nudge),
    ("zoom-in", _zoom_in),
    ("nudge", _nudge),
    ("nudge", _nudge),
    ("pan-one-tile", _pan_a_whole_tile),
    ("nudge", _nudge),
    ("pan-one-tile", _pan_a_whole_tile),
    ("nudge", _nudge),
    ("levels-0.7", lambda w, q: _set_levels(w, q, 0.7)),
    ("nudge", _nudge),
)


def _blind_key(self, plan, view_range, viewport_shape):
    """The mutant: an invariant key that forgets which tiles are on screen."""

    if view_range is None or plan is None:
        return None
    (x0, x1), (y0, y1) = view_range
    span_x = float(x1) - float(x0)
    span_y = float(y1) - float(y0)
    if not (span_x > 0.0 and span_y > 0.0):
        return None
    return (id(plan), round(span_x, 4), round(span_y, 4))


def _run(qtbot, monkeypatch, *, latch, depth, columns, mutate):
    from arrayscope.window import frame_effects
    from arrayscope.window.frame_session import FrameSession

    monkeypatch.setattr(
        frame_effects,
        "viewport_convergence_latch_backend",
        lambda *_args, **_kwargs: bool(latch),
    )
    if mutate:
        monkeypatch.setattr(FrameSession, "viewport_invariant_key", _blind_key)
    win = _build(qtbot, depth, columns)
    engaged = 0
    frames = []
    unsettled = set()
    try:
        for index, (label, step) in enumerate(SCRIPT):
            step(win, qtbot)
            if not _drain(win, qtbot):
                unsettled.add(index)
            if win.renderer._frame_session._viewport_converged_key is not None:
                engaged += 1
            frames.append((label, _fingerprint(win)))
    finally:
        win.close()
    return engaged, frames, unsettled


@pytest.mark.parametrize(("depth", "columns"), [(16, 4)])
@pytest.mark.parametrize(
    ("control", "mutate", "expect_identical"),
    [
        pytest.param(False, False, True, id="latch-vs-unlatched"),
        pytest.param(True, False, True, id="control-latch-vs-latch"),
        pytest.param(False, True, False, id="mutant-key-forgets-tile-set"),
    ],
)
def test_convergence_latch_does_not_change_the_pyqtgraph_frame(
    qtbot, monkeypatch, depth, columns, control, mutate, expect_identical
):
    clear_arrayscope_settings()
    try:
        latched_engaged, latched, latched_unsettled = _run(
            qtbot, monkeypatch, latch=True, depth=depth, columns=columns, mutate=mutate
        )
        other_engaged, other, other_unsettled = _run(
            qtbot,
            monkeypatch,
            latch=bool(control),
            depth=depth,
            columns=columns,
            mutate=mutate,
        )
    finally:
        clear_arrayscope_settings()

    assert latched_engaged >= len(SCRIPT) // 2, (
        f"the latch engaged after only {latched_engaged}/{len(SCRIPT)} steps, so this "
        "compares two effectively unlatched runs and can never fail"
    )
    if not control:
        assert other_engaged == 0, "the unlatched arm must never latch"

    # The latch may never PREVENT a round from settling. Settling more often
    # than the unlatched build is not a defect -- the unlatched retarget
    # re-queues PyQtGraph's refined-level pass on every camera nudge.
    starved = sorted(latched_unsettled - other_unsettled)
    assert not starved, f"the latch starved these steps of their settle: {starved}"

    # A step whose round never settled is legitimately mid-flight; the control
    # arm diverges there too, so no differential claim is possible.
    mid_flight = latched_unsettled | other_unsettled
    divergences = []
    for index, ((label, latched_frame), (_other_label, other_frame)) in enumerate(
        zip(latched, other, strict=True)
    ):
        if index in mid_flight or latched_frame == other_frame:
            continue
        differing = sorted(
            tile
            for tile in set(latched_frame) | set(other_frame)
            if latched_frame.get(tile) != other_frame.get(tile)
        )
        divergences.append(
            f"step {index} after {label!r}: {len(differing)} tiles differ, "
            f"first={differing[0]} "
            f"latched={latched_frame.get(differing[0])!r} "
            f"other={other_frame.get(differing[0])!r}"
        )

    if expect_identical:
        assert not divergences, "the latch changed the frame:\n" + "\n".join(divergences)
    else:
        assert divergences, (
            "the mutant key forgot the visible tile set and the montage still "
            "matched -- this oracle cannot see a stale frame and proves nothing"
        )
