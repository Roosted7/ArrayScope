"""The GUI half of ADR 0065 stage 1: lock spans and owner-thread affinity.

Three claims live here, one per thing that can silently not happen:

1. The GUI-thread entry funnels actually take the *GUI* side of the
   session-order lock. Counted, because a decorator that resolved to a null
   context would look exactly like a working one from the outside.
2. ``request_presentation`` called off the GUI thread posts its gate instead of
   arming a ``QTimer``. A timer started on the presentation owner lives on a
   thread with no event loop: it does not fire late, it never fires, and the
   session's backlog has no other wakeup. The pin is the recorded arm action
   plus the gate actually firing — never a sleep.
3. The residency check inside ``_prepare_backend_uploads`` reads the
   lifecycle's backend-identity mirror rather than probing the live backend.
   Both directions are pinned: an identity in the mirror is skipped as
   resident, an absent one is offered to the planner.

Claims 1 and 2 need a whole window (the owner thread is the window's).
Claim 3 is ring 1: the mirror comparison is session state and a stub view is
enough to see what was offered.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.core.trace import TRACE
from arrayscope.core.view_state import ViewState
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.lod import LodInfo
from arrayscope.display.model.frame import DisplayTilePayload
from arrayscope.display.model.tile_identity import TileLodIdentity, tile_ack_identity
from arrayscope.display.montage import make_montage_plan
from arrayscope.display.shader_mapping import TexturePlaneKind
from arrayscope.presentation.owner import PresentationOwner
from arrayscope.presentation.prepared_uploads import PreparedUploadMailbox
from arrayscope.window.frame_effects import (
    FramePipelineEffects,
    session_order_lock_section,
)
from arrayscope.window.frame_session import FrameSession
from tests.settle import deliver_posted_events, settle_window

pytest.importorskip("pytestqt")

TILE_SHAPE = (4, 4)
SOURCES = (0, 1, 2, 3)


# --- claim 1: the funnels take the GUI side ----------------------------------


def test_a_section_charges_the_gui_side_and_a_missing_owner_charges_nothing():
    """The decorator itself, away from any window.

    Two arms because the null-context case is the one that cannot be seen
    later: a renderer built without a MainWindow (most of this suite) has no
    presentation owner, and the section must still run its body.
    """

    owner = PresentationOwner(None)

    class _Renderer:
        def __init__(self, win) -> None:
            self.win = win
            self.ran = 0

        @session_order_lock_section
        def section(self) -> str:
            self.ran += 1
            return "ran"

    owned = _Renderer(SimpleNamespace(presentation_owner=owner))
    bare = _Renderer(SimpleNamespace())

    assert owned.section() == "ran"
    assert owner.gui_lock_acquisitions == 1
    # The owner side is untouched: this is a GUI section and the counter that
    # adjudicates the migration must never be fed from the wrong door.
    assert owner.owner_lock_acquisitions == 0

    assert bare.section() == "ran"
    assert bare.ran == 1
    assert owner.gui_lock_acquisitions == 1


def test_the_commit_body_runs_holding_the_order_lock(montage_window, monkeypatch):
    """The invariant, observed from inside the commit rather than around it.

    A delta on ``gui_lock_acquisitions`` across the call would be satisfied by
    *any* span on the path, including one belonging to something else the
    commit happens to invite. What stage 1 actually promises is that the body
    building the presentation runs with the lock held, so that is what is read
    — from inside, on the GUI thread, with the counter delta beside it to say
    the acquisition was charged to the GUI side and not inherited.
    """

    win = montage_window
    owner = win.presentation_owner
    session = win.renderer._frame_session
    held: list[bool] = []
    build = FramePipelineEffects.direct_tile_layer_presentation

    def spy(effects):
        held.append(owner.order_lock._is_owned())
        return build(effects)

    monkeypatch.setattr(FramePipelineEffects, "direct_tile_layer_presentation", spy)
    before = owner.gui_lock_acquisitions

    win.renderer.commit_frame_session_presentation(session)

    assert held, "the commit did not build a presentation, so nothing was observed"
    assert all(held)
    assert owner.gui_lock_acquisitions > before


# --- claim 2: request_presentation off the GUI thread ------------------------


def _trace_marker() -> int:
    events = TRACE.snapshot()
    return int(events[-1]["sequence"]) if events else 0


def _presentation_gate_actions(since: int) -> list[str]:
    return [
        str(event.get("action", ""))
        for event in TRACE.snapshot()
        if int(event["sequence"]) > since and event.get("kind") == "presentation_gate"
    ]


def test_presentation_gate_observer_survives_a_full_trace_ring(monkeypatch):
    from arrayscope.core.trace import TraceBus

    trace = TraceBus()
    trace.configure(ring_events=3)
    monkeypatch.setitem(globals(), "TRACE", trace)
    for _ in range(3):
        trace.emit("older-event")
    marker = _trace_marker()
    trace.emit("presentation_gate", action="armed-post")

    assert _presentation_gate_actions(marker) == ["armed-post"]


def _disarm_gate(renderer) -> None:
    renderer._montage_presentation_gate_owner = None
    renderer._montage_presentation_gate_armed = False


def test_request_presentation_off_the_gui_thread_posts_the_gate(montage_window):
    win = montage_window
    renderer = win.renderer
    session = renderer._frame_session
    effects = session.pipeline.effects
    assert TRACE.enabled, "the live montage arms the ring; without it there is nothing to read"

    # Posted, not armed on a timer this thread owns no event loop for.
    # Membership, not list equality, and a bounded retry: the fixture window
    # is live, so between the disarm and our call its own completion traffic
    # can re-arm the gate first — our request then coalesces and emits
    # nothing, which under a loaded suite made a single attempt observe an
    # empty window (`'armed-post' in []`). An attempt only counts when it
    # produced an arm; the claim is that no off-GUI caller ever took the
    # timer branch, which every attempt checks unconditionally.
    for _ in range(50):
        _disarm_gate(renderer)
        marker = _trace_marker()
        thread = threading.Thread(target=effects.request_presentation, name="not-the-gui-thread")
        thread.start()
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        actions = _presentation_gate_actions(marker)
        assert "armed-timer" not in actions
        if "armed-post" in actions:
            break
    else:
        raise AssertionError("50 off-GUI requests all coalesced behind background arms")
    assert renderer._montage_presentation_gate_owner is not None

    settle_window(win, lambda: renderer._montage_presentation_gate_owner is None)


def test_the_gui_thread_still_coalesces_a_cpu_windowed_commit_on_its_timer(montage_window):
    """The other side of the branch, so the fix above is not a deletion.

    Skipped on a shader-windowing backend rather than asserted around: that
    backend never had the timer branch, so there is no behaviour to preserve.
    """

    win = montage_window
    renderer = win.renderer
    session = renderer._frame_session
    if bool(image_view_backend_capabilities(win.img_view).shader_windowing):
        pytest.skip("shader-windowing backend: the CPU-windowed timer branch does not exist")
    if not bool(getattr(session, "display_committed", False)):
        pytest.skip("nothing committed yet: the timer branch is a rebuild coalescer")

    # Same bounded retry as the posted-gate test, for the same live-window
    # coalescing race; the claim is that the GUI thread kept its timer
    # branch, proven by any attempt whose arm was ours.
    for _ in range(50):
        _disarm_gate(renderer)
        marker = _trace_marker()
        session.pipeline.effects.request_presentation()
        if "armed-timer" in _presentation_gate_actions(marker):
            break
    else:
        raise AssertionError("50 GUI-thread requests all coalesced behind background arms")
    settle_window(win, lambda: renderer._montage_presentation_gate_owner is None)


def test_request_montage_replan_off_the_gui_thread_still_disarms_its_gate(montage_window):
    """The replan arm crosses back, and the coalescing flag is not stranded.

    ``_montage_replan_gate_armed`` is set by whichever thread asks and cleared
    only when the timer fires. If the crossing were lost, the flag would latch
    True and every later replan request would coalesce into a wakeup that
    never comes — the lost-wakeup class, and invisible from the outside except
    as a montage that stops converging.
    """

    win = montage_window
    renderer = win.renderer
    session = renderer._frame_session
    renderer._montage_replan_gate_armed = False

    thread = threading.Thread(
        target=lambda: renderer.request_montage_replan(session), name="not-the-gui-thread"
    )
    thread.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert renderer._montage_replan_gate_armed is True

    settle_window(win, lambda: renderer._montage_replan_gate_armed is False)


# --- claim 2b: the atomic warm continuation is a locked section --------------


def _foreign_thread_can_take(lock) -> bool:
    """Whether a thread that is not this one could take ``lock`` right now.

    ``RLock.acquire(False)`` answers True to its own owner, so asking from
    here would say "free" no matter what. The probe has to run somewhere else.
    """

    answer: list[bool] = []

    def probe() -> None:
        got = lock.acquire(False)
        answer.append(got)
        if got:
            lock.release()

    thread = threading.Thread(target=probe, name="foreign-lock-probe")
    thread.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    return answer[0]


def _atomic_warm_fixture(owner):
    """The seams ``_warm_atomic_successor_residency`` actually reaches.

    Deliberately not a window: the claim is about the callback's own span, and
    a live montage would put a dozen other lock sections around it.
    """

    payload = SimpleNamespace(source_id=("tile", 0, "exact"))
    warm_calls: list[dict] = []
    state = {"resident": False}

    def warm_tiled_residency(**kwargs):
        warm_calls.append(
            {
                "held": owner.order_lock._is_owned(),
                "foreign_blocked": not _foreign_thread_can_take(owner.order_lock),
            }
        )
        state["resident"] = True

    view = SimpleNamespace(
        warmTiledResidency=warm_tiled_residency,
        tiledPayloadResident=lambda _payload: state["resident"],
    )
    replans: list[bool] = []
    renderer = SimpleNamespace(
        win=SimpleNamespace(img_view=view, presentation_owner=owner),
        _frame_session_is_current=lambda _session: True,
        _memory_policy=lambda: SimpleNamespace(
            visible_render_budget_bytes=1 << 30,
            display_cache_budget_bytes=1 << 30,
            user_render_cap_bytes=1 << 30,
        ),
        request_montage_replan=lambda _session: replans.append(owner.order_lock._is_owned()),
    )
    session = SimpleNamespace(session_id=1, key="key", viewport_revision=0)
    tile_delta = SimpleNamespace(upserts={0: payload}, atomic_handoff=True)
    return renderer, session, tile_delta, payload, warm_calls, replans


def test_the_atomic_warm_continuation_writes_session_state_under_the_order_lock(monkeypatch):
    """The posted continuation, not the funnel that posted it.

    ``continue_warm`` writes ``_atomic_warm_job``, ``final_commit_pending``
    and ``flush_pending`` from a bare receiver callback. The presentation
    owner writes session state under the order lock, and an unlocked writer
    racing a locked one is not ordered by that lock — so the callback itself
    has to be a section. Read from inside the backend call the callback makes
    and from beside the writes, with a foreign-thread probe because an RLock
    answers True to its own owner.
    """

    import arrayscope.window.frame_effects as frame_effects

    owner = PresentationOwner(None)
    renderer, session, tile_delta, payload, warm_calls, replans = _atomic_warm_fixture(owner)

    posted: list = []
    monkeypatch.setattr(
        frame_effects,
        "_post_visible_path_callback",
        lambda _renderer, callback: posted.append(callback),
    )

    warm_complete = frame_effects._warm_atomic_successor_residency(
        renderer,
        session,
        geometry=None,
        tile_delta=tile_delta,
        levels=(0.0, 1.0),
        rgb_already_windowed=False,
        payloads={0: payload},
        batch_size=1,
    )

    assert warm_complete is False, "a cold payload should have armed a continuation"
    assert len(posted) == 1
    job = session._atomic_warm_job
    assert job is not None

    before = owner.gui_lock_acquisitions

    posted[0]()

    # The three session writes happen after the loop, so sample the lock where
    # they land: the replan call that immediately follows them, plus the
    # backend call inside the loop. Both must see the section held.
    assert warm_calls, "the continuation never reached the backend, so nothing was observed"
    assert all(call["held"] for call in warm_calls)
    assert all(call["foreign_blocked"] for call in warm_calls)
    assert replans == [True]
    assert session._atomic_warm_job is None
    assert session.final_commit_pending is True
    assert session.flush_pending is True
    assert owner.gui_lock_acquisitions > before


def test_the_atomic_warm_continuation_keeps_its_job_identity_guard(monkeypatch):
    """The mutation guard for the fix above.

    ``_atomic_warm_job`` is an ``is``-identity latch: a superseding job must
    make the queued callback a no-op. Wrapping the callback in a lock section
    must not change that, and containerising the job (tried and reverted) is
    exactly what would.
    """

    import arrayscope.window.frame_effects as frame_effects

    owner = PresentationOwner(None)
    renderer, session, tile_delta, payload, warm_calls, _replans = _atomic_warm_fixture(owner)

    posted: list = []
    monkeypatch.setattr(
        frame_effects,
        "_post_visible_path_callback",
        lambda _renderer, callback: posted.append(callback),
    )
    frame_effects._warm_atomic_successor_residency(
        renderer,
        session,
        geometry=None,
        tile_delta=tile_delta,
        levels=(0.0, 1.0),
        rgb_already_windowed=False,
        payloads={0: payload},
        batch_size=1,
    )

    superseding = {"signature": ("other",), "pending": [], "payloads": {}}
    session._atomic_warm_job = superseding

    posted[0]()

    assert warm_calls == [], "a superseded continuation still warmed"
    assert session._atomic_warm_job is superseding


# --- claim 3: session-side residency ----------------------------------------


class _StubView:
    """Just the two seams ``_prepare_backend_uploads`` reaches through."""

    def __init__(self) -> None:
        self.preparedTiledUploads = PreparedUploadMailbox()
        self.offered: list[dict] = []

    def tiledUploadPreparations(self, payloads, *, levels, rgb_already_windowed=False):
        del levels, rgb_already_windowed
        self.offered.append(dict(payloads))
        return ()


def _residency_effects():
    state = ViewState.from_shape((4, 4, len(SOURCES))).with_montage_axis(
        2, indices=SOURCES, text=":"
    )
    plan = make_montage_plan(state, axis=2, indices=SOURCES, tile_shape=TILE_SHAPE, columns=2)
    session = FrameSession(
        session_id=1,
        key="key",
        render_generation=1,
        level_key="levels",
        level_expected_indices=SOURCES,
        plan=plan,
        view_state=state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(16, 16),
        view_range=None,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        shader_display=True,
    )
    # The level gate downstream of the residency check refuses to prepare
    # against unsettled or previous-round levels; settle them so a payload
    # that is *not* resident reaches the planner and the two arms differ.
    session.level_generation.target_levels = (0.0, 1.0)
    session.level_generation.semantic_key = session.level_key
    view = _StubView()
    renderer = SimpleNamespace(win=SimpleNamespace(img_view=view, kernel=SimpleNamespace()))
    return FramePipelineEffects(renderer, session), session, view


def _payload(session, tile_number: int) -> DisplayTilePayload:
    tile = session.plan.tiles[int(tile_number)]
    image = np.full(TILE_SHAPE, float(tile.source_index), dtype=np.float32)
    identity = session._tile_identity(
        tile,
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        complex_mapping=None,
        lod=TileLodIdentity(),
        quality="exact",
    )
    return DisplayTilePayload(
        tile_number=int(tile_number),
        source_index=int(tile.source_index),
        image=image,
        histogram_data=image,
        source_id=("tile", int(tile.source_index), "exact"),
        lod=LodInfo(level=0, factor=1, source_shape=TILE_SHAPE, texture_shape=TILE_SHAPE),
        tile_identity=identity,
    )


def test_a_payload_the_mirror_already_holds_is_skipped_as_resident():
    effects, session, view = _residency_effects()
    payload = _payload(session, 0)
    session.display_tile_payloads[0] = payload
    session.lifecycle.backend_presented_snapshot({0: tile_ack_identity(payload)})

    effects._prepare_backend_uploads(((SimpleNamespace(tile_number=0), payload),))

    assert view.preparedTiledUploads.counters().skipped_resident == 1
    assert view.offered == []


def test_a_payload_the_mirror_does_not_hold_is_offered_to_the_planner():
    """The mutation check: same call, mirror flipped to another identity."""

    effects, session, view = _residency_effects()
    payload = _payload(session, 0)
    session.display_tile_payloads[0] = payload
    session.lifecycle.backend_presented_snapshot({0: ("some", "other", "identity")})

    effects._prepare_backend_uploads(((SimpleNamespace(tile_number=0), payload),))

    assert view.preparedTiledUploads.counters().skipped_resident == 0
    assert view.offered == [{0: payload}]


@pytest.fixture
def montage_window(qtbot):
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 8 * 4, dtype=np.float32).reshape(8, 8, 4))
    qtbot.addWidget(win)
    win._set_view_state(win.view_state.with_montage_axis(2, columns=2, indices=SOURCES, text=":"))
    win.render(reason="test-session-order-lock")
    settle_window(win, lambda: getattr(win.renderer, "_frame_session", None) is not None)
    session = win.renderer._frame_session
    settle_window(win, lambda: session.visible_plan_complete())
    deliver_posted_events()
    try:
        yield win
    finally:
        win.close()
