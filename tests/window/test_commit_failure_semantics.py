"""A commit that raises must report as that exception, never as a stall.

Contract: dossier ``docs/redesign/wgpu-pool-layer-leak-2026-07-26.md`` §5a,
amending ADR 0051. A throw on the presentation-commit path used to be the one
commit exit that named neither an outcome nor a wakeup: it left
``commit_outcome='started'``, armed nothing, and surfaced four seconds later as
the profiler's STALL GUARD talking about a lost wakeup. A real page-pool
exhaustion and a typo'd ``AttributeError`` produced byte-identical dumps.

These are fault-injection tests: they raise from the commit path and assert the
failure is reported as that exception.
"""

from __future__ import annotations

import types

import pytest

from arrayscope.core.trace import emit_trace  # noqa: F401  (trace bus import parity)
from arrayscope.window.frame_effects import FramePipelineEffects


class _InjectedCommitFailure(RuntimeError):
    """Sentinel distinguishable from any incidental error in the commit path."""


def _effects(session_id: int = 7, key: str = "semantic-key"):
    renderer = types.SimpleNamespace(
        _montage_presentation_gate_owner=None,
        _montage_presentation_gate_armed=False,
    )
    session = types.SimpleNamespace(session_id=session_id, key=key)
    return FramePipelineEffects(renderer, session), renderer, session


def test_commit_throw_is_recorded_as_a_named_terminal_bail():
    """The throw names an outcome and an explicitly absent wakeup."""

    effects, renderer, _session = _effects()
    bails = []
    effects._note_commit_bail = lambda outcome, *, wakeup, **details: bails.append(
        (outcome, wakeup, details)
    )

    try:
        raise _InjectedCommitFailure("page pool exhausted")
    except _InjectedCommitFailure as exc:
        effects._note_commit_raised(exc, dirty_tiles=(0, 3, 9))

    outcome, wakeup, details = bails[0]
    assert outcome == "raised", "a throw must not keep reporting commit_outcome='started'"
    # The absence of a wakeup is the contract, stated rather than implied:
    # nothing retries a commit that raised.
    assert wakeup == "none (terminal)"
    assert details["exception_type"] == "_InjectedCommitFailure"
    assert details["committing_tile_count"] == 3

    recorded = renderer._last_montage_commit_exception
    assert recorded["type"] == "_InjectedCommitFailure"
    assert recorded["message"] == "page pool exhausted"
    # Attribution: which session, and which tiles it was committing.
    assert recorded["session_id"] == 7
    assert recorded["committing_tiles"] == (0, 3, 9)
    assert "_InjectedCommitFailure" in recorded["traceback"]


def test_presentation_gate_reports_the_throw_and_does_not_re_arm(caplog):
    """Default policy: loud and terminal — logged, and no silent retry.

    Re-arming would replay a delta that is about to throw again at full flush
    rate, which is the rescue ADR 0051 forbids.
    """

    effects, renderer, session = _effects()
    owner = (int(session.session_id), id(session))
    renderer._montage_presentation_gate_owner = owner
    renderer._montage_presentation_gate_armed = True
    effects._session_is_current = lambda: True

    calls = []

    def _raise():
        calls.append("commit")
        raise _InjectedCommitFailure("boom")

    effects.commit_pending_session = _raise

    with caplog.at_level("ERROR"):
        # Must NOT propagate by default: the app stays alive, as it does for
        # every other Qt callback.
        effects._on_presentation_gate(owner)

    assert calls == ["commit"], "the commit ran exactly once — no retry loop"
    assert "montage presentation commit" in caplog.text
    assert "_InjectedCommitFailure" in caplog.text

    # The gate is left disarmed and unowned, so a later session can still arm
    # it. That is why no repair is needed — and why re-arming this one would be
    # a retry, not a rescue of corrupted state.
    assert renderer._montage_presentation_gate_armed is False
    assert renderer._montage_presentation_gate_owner is None


def test_strict_ui_makes_a_commit_throw_fatal(monkeypatch, caplog):
    """Under ARRAYSCOPE_STRICT_UI the throw stays fatal — and is still logged.

    Guards this change rather than the original defect: routing the commit
    through ``handle_ui_exception`` must not soften strict mode. The log
    assertion is what distinguishes "went through the policy" from "escaped
    raw", which is all the old code did.
    """

    monkeypatch.setenv("ARRAYSCOPE_STRICT_UI", "1")
    effects, renderer, session = _effects()
    owner = (int(session.session_id), id(session))
    renderer._montage_presentation_gate_owner = owner
    renderer._montage_presentation_gate_armed = True
    effects._session_is_current = lambda: True

    def _raise():
        raise _InjectedCommitFailure("boom")

    effects.commit_pending_session = _raise

    with caplog.at_level("ERROR"), pytest.raises(_InjectedCommitFailure, match="boom"):
        effects._on_presentation_gate(owner)

    assert "montage presentation commit" in caplog.text


def test_live_montage_commit_throw_is_named_not_silently_stranded(qtbot, monkeypatch):
    """End-to-end: inject a throw into a real montage commit.

    This is the shape of the original defect — the exception arises deep under
    ``_commit_tile_layer`` (there, inside the wgpu executor's page admission) —
    so it exercises the real ``except`` arm and the real gate, not a stub.
    """

    import numpy as np

    from tests.ui.helpers import clear_arrayscope_settings

    clear_arrayscope_settings()
    from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
    from arrayscope.window import ArrayScopeWindow

    injected = []
    real_present = FramePipelineEffects._present_tile_delta

    def _boom(self, *args, **kwargs):
        # Let the montage commit normally until it has real tiles to carry,
        # then fail the way a backend does: from inside the delta application.
        if not injected and getattr(self.session, "pending_payload_upserts", None):
            injected.append(True)
            raise _InjectedCommitFailure("injected backend failure")
        return real_present(self, *args, **kwargs)

    monkeypatch.setattr(FramePipelineEffects, "_present_tile_delta", _boom)

    data = np.arange(8 * 8 * 4, dtype=np.float32).reshape(8, 8, 4)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        win._set_view_state(
            win.view_state.with_montage_axis(2, columns=2, indices=(0, 1, 2, 3), text=":")
        )
        win.render(reason="test-commit-throw")
        qtbot.waitUntil(
            lambda: getattr(win.renderer, "_last_montage_commit_exception", None) is not None,
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )

        recorded = win.renderer._last_montage_commit_exception
        assert recorded["type"] == "_InjectedCommitFailure"
        assert recorded["message"] == "injected backend failure"
        # Attributed to the session it was committing, not left anonymous.
        assert isinstance(recorded["session_id"], int)
        assert "_InjectedCommitFailure" in recorded["traceback"]
        # The outcome is the throw, not the misleading 'started' it used to
        # leave behind for the stall dump to report.
        assert win.renderer._last_montage_commit_outcome == "raised"

        # And the profiler would name it rather than blame a lost wakeup.
        from arrayscope.tools.profile_montage_workflow import _commit_raised_failure

        message = _commit_raised_failure(win)
        assert message is not None
        assert message.startswith("COMMIT RAISED")
        assert "_InjectedCommitFailure" in message
    finally:
        win.close()


def test_profiler_reports_a_commit_throw_instead_of_a_stall():
    """The dump names the exception, not a lost wakeup."""

    from arrayscope.tools.profile_montage_workflow import _commit_raised_failure

    # A run whose commit never raised still reports as a stall — this guard
    # must not swallow genuine no-progress freezes.
    assert _commit_raised_failure(types.SimpleNamespace(renderer=types.SimpleNamespace())) is None

    win = types.SimpleNamespace(
        renderer=types.SimpleNamespace(
            _last_montage_commit_exception={
                "type": "RuntimeError",
                "message": "page pool 'complex_rg32f' exhausted",
                "traceback": "Traceback (most recent call last):\n  ...\n",
                "session_id": 9,
                "semantic_key": "'fft-key'",
                "committing_tiles": (0,),
                "committing_tile_count": 1,
            }
        )
    )
    message = _commit_raised_failure(win)
    assert message is not None
    assert message.startswith("COMMIT RAISED")
    # Everything an operator needs to skip the wrong hypothesis entirely.
    assert "RuntimeError" in message
    assert "page pool 'complex_rg32f' exhausted" in message
    assert "session: 9" in message
    assert "committing 1 tile(s), first: (0,)" in message
    assert "Traceback" in message
    # And it must not describe a lost wakeup.
    assert "STALL GUARD" not in message
