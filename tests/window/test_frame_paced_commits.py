"""The owed draw wins the event-loop race against the next commit batch.

The paced draw (``WgpuScreenCanvas.request_draw`` -> ``QTimer.singleShot``)
and the commit continuation (a ``LowEventPriority`` posted gate event) are
both plain event-loop wakeups with no ordering guarantee between them, so a
30 ms commit routinely runs while the display is still owed a frame.  These
tests pin the standing-down rule in counting form: no clocks, only the
recorded interleaving of present-acks and commit batches.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from arrayscope.window.display_presenter import (
    DISPLAY_REFRESH_FALLBACK_HZ,
    DisplayPresentationMixin,
)
from tests.ui.helpers import (
    frame_session_settled,
    make_backend_window,
    restore_default_backend,
    use_wgpu_backend,
)


class _StubView(QtWidgets.QWidget):
    presentationDrawn = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.draw_pending = False
        self.refresh_hz = 0.0

    def presentationDrawPending(self) -> bool:
        return bool(self.draw_pending)

    def presentationRefreshRate(self) -> float:
        return float(self.refresh_hz)

    def acknowledge_draw(self) -> None:
        self.draw_pending = False
        self.presentationDrawn.emit()


class _StubWindow:
    def __init__(self, view) -> None:
        self.img_view = view
        self._closing = False


class _PacingHost(DisplayPresentationMixin, QtCore.QObject):
    def __init__(self, win) -> None:
        super().__init__()
        self.win = win


@pytest.fixture
def pacing_host(qtbot):
    view = _StubView()
    qtbot.addWidget(view)
    view.show()
    qtbot.waitUntil(view.isVisible)
    return _PacingHost(_StubWindow(view)), view


def test_an_owed_draw_stands_the_next_commit_down(pacing_host):
    host, view = pacing_host
    order: list[str] = []
    view.presentationDrawn.connect(lambda: order.append("draw"))

    def commit() -> None:
        order.append("commit")

    view.draw_pending = True
    assert host.defer_commit_behind_presentation_draw(commit, interactive=True) is True
    assert order == [], "the commit must not run in front of the draw it owes"

    view.acknowledge_draw()

    assert order == ["draw", "commit"]
    assert host.presentation_pacing_stats()["commit_deferrals"] == 1
    assert host.presentation_pacing_stats()["commit_rearms_draw_ack"] == 1


def test_a_commit_yields_once_per_ack_so_a_silent_surface_strands_nothing(pacing_host):
    """One yield per completed draw, so no ack is required for progress.

    A hidden surface, a headless run, or a compositor that dropped the frame
    can leave the debt armed forever.  The second continuation must commit
    anyway: the deferral is an ordering preference, never a gate.
    """

    host, view = pacing_host
    calls: list[str] = []
    view.draw_pending = True

    assert (
        host.defer_commit_behind_presentation_draw(lambda: calls.append("first"), interactive=True)
        is True
    )
    assert (
        host.defer_commit_behind_presentation_draw(lambda: calls.append("second"), interactive=True)
        is False
    )
    assert calls == []

    view.acknowledge_draw()
    view.draw_pending = True

    assert (
        host.defer_commit_behind_presentation_draw(lambda: calls.append("third"), interactive=True)
        is True
    )
    assert calls == ["first"]


def test_the_deferral_leaves_a_wakeup_armed_when_no_ack_arrives(pacing_host, qtbot):
    host, view = pacing_host
    calls: list[str] = []
    view.draw_pending = True

    assert (
        host.defer_commit_behind_presentation_draw(lambda: calls.append("commit"), interactive=True)
        is True
    )

    qtbot.waitUntil(lambda: calls == ["commit"], timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
    assert host.presentation_pacing_stats()["commit_rearms_backstop"] == 1


def test_a_fill_nobody_is_touching_is_never_stood_down(pacing_host):
    """The measured refusal: an idle fill keeps every commit it can run.

    Standing an idle cold fill down for its own owed draws was measured on the
    272-tile WGPU montage (Intel TGL iGPU, headless Weston, 2026-08-02) and
    refused: 79 tiles presented in the profiler's 5 s window against 266 with
    the deferral off, and R1's ``physical_uploads_stay_on_round_floors`` went
    from clean to two failures per run because the slower fill gave the
    unimplemented R7 level-0 warm room to land inside the round.
    """

    host, view = pacing_host
    view.draw_pending = True

    assert host.defer_commit_behind_presentation_draw(lambda: None, interactive=False) is False
    assert host.presentation_pacing_stats()["commit_deferrals"] == 0


def test_a_hidden_surface_has_no_frame_to_win(pacing_host):
    host, view = pacing_host
    view.hide()
    view.draw_pending = True

    assert host.defer_commit_behind_presentation_draw(lambda: None, interactive=True) is False


def test_the_pacing_rate_is_the_view_s_own_and_falls_back_to_60(pacing_host):
    host, view = pacing_host

    assert host.presentation_refresh_hz() == DISPLAY_REFRESH_FALLBACK_HZ
    view.refresh_hz = 144.0
    assert host.presentation_refresh_hz() == 144.0


def test_a_montage_fill_interleaves_its_draws_with_its_commit_batches(qtbot):
    """Live shape: a scripted fill, recorded as one sequence of both events.

    Counting form, no clocks.  Every commit batch and every present-ack append
    to one list; the property is that the fill does not run batch after batch
    while a draw sits owed.

    WGPU on purpose: its draw debt outlives a commit (the canvas paces itself
    and clears the debt from the draw ack), which is the ordering this is
    about.  The QGraphics path acknowledges inside the same event turn, so it
    has nothing to stand down for -- measured on this fill, its gate saw an
    owed draw zero times out of sixteen against WGPU's seven out of nineteen.

    The fill runs inside a scripted camera gesture, because that is the only
    scope the deferral has: an untouched fill is measured to lose more in tiles
    than it gains in frames (see the refusal test above).
    """

    settings = use_wgpu_backend(extra_settings={"montage_quality_policy": "resident"})
    rng = np.random.default_rng(20260802)
    data = np.ascontiguousarray(rng.standard_normal((128, 128, 40), dtype=np.float32))
    win = make_backend_window(qtbot, data, backend="wgpu", require_gpu_atlas=True)
    win.resize(900, 700)
    try:
        win.show()
        renderer = win.renderer
        order: list[str] = []
        win.img_view.presentationDrawn.connect(lambda: order.append("draw"))
        committer = renderer._display_committer()
        original = committer.commit_tiled_delta

        def recording_commit(presentation):
            order.append("commit")
            return original(presentation)

        committer.commit_tiled_delta = recording_commit
        win._viewport_interaction_active = True

        indices = tuple(range(40))
        state = win.view_state.with_montage_axis(2, columns=8, indices=indices, text="0:40")
        win._set_view_state(state)
        win.update_image_view()
        qtbot.waitUntil(
            lambda: frame_session_settled(win), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS
        )

        stats = renderer.presentation_pacing_stats()
        assert order.count("commit") > 1, "the fill must have committed in several batches"
        assert stats["commit_deferrals"] >= 1, "no commit ever stood down for an owed draw"
        # Every deferral was redeemed, by the ack it waited for or by its own
        # backstop -- a deferral that redeemed neither would have stranded the
        # fill, which the settle above would already have failed on.
        assert (
            stats["commit_rearms_draw_ack"] + stats["commit_rearms_backstop"]
            >= stats["commit_deferrals"] - 1
        )
        longest_unbroken_commits = 0
        run = 0
        for event in order:
            run = run + 1 if event == "commit" else 0
            longest_unbroken_commits = max(longest_unbroken_commits, run)
        assert longest_unbroken_commits < order.count("commit"), (
            "the fill committed every batch without ever letting a draw through"
        )
    finally:
        win._viewport_interaction_active = False
        win.close()
        restore_default_backend(settings)
