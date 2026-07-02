"""The render staleness vocabulary has one owner (roadmap Y1)."""

from types import SimpleNamespace

from arrayscope.window.render_contract import (
    RenderGeneration,
    generation_is_current,
    montage_work_token,
    montage_work_token_is_current,
    session_is_current,
    session_token_is_current,
)


def _session(session_id=1, key=("doc", "view"), **extra):
    return SimpleNamespace(session_id=session_id, key=key, render_generation=7, **extra)


def test_render_generation_orders_visible_output():
    generation = RenderGeneration()
    captured = generation.capture()
    assert generation.is_current(captured)
    advanced = generation.advance("slice")
    assert advanced == captured + 1
    assert generation.last_reason == "slice"
    assert not generation.is_current(captured)
    assert generation.is_current(advanced)


def test_missing_generation_guard_admits_everything():
    assert generation_is_current(None, 123)
    guard = RenderGeneration(current=3)
    assert generation_is_current(guard, 3)
    assert not generation_is_current(guard, 2)


def test_session_predicates_agree_on_identity():
    current = _session()
    assert session_is_current(current, current)
    assert session_token_is_current(current, 1, ("doc", "view"))
    assert not session_token_is_current(current, 2, ("doc", "view"))
    assert not session_token_is_current(current, 1, ("doc", "other"))
    assert not session_is_current(current, _session(session_id=2))
    assert not session_is_current(None, current)
    assert not session_is_current(current, None)


def test_work_token_carries_session_identity_and_generation():
    session = _session()
    token = montage_work_token(session, "stage_wait")
    assert token == ("stage_wait", 1, ("doc", "view"), 7)
    assert montage_work_token_is_current(session, token, "stage_wait")
    assert montage_work_token_is_current(session, None, "stage_wait")
    session.session_id = 2
    assert not montage_work_token_is_current(session, token, "stage_wait")


def test_commit_token_folds_payload_and_level_revisions():
    session = _session(payload_revision=4, level_revision=9)
    token = montage_work_token(session, "commit")
    assert montage_work_token_is_current(session, token, "commit")
    session.payload_revision = 5
    assert not montage_work_token_is_current(session, token, "commit")
    session.payload_revision = 4
    session.level_revision = 10
    assert not montage_work_token_is_current(session, token, "commit")


def test_viewport_tokens_fold_viewport_revision():
    session = _session(viewport_revision=2)
    for reason in ("priority_retarget", "viewport_update"):
        token = montage_work_token(session, reason)
        assert montage_work_token_is_current(session, token, reason)
        session.viewport_revision += 1
        assert not montage_work_token_is_current(session, token, reason)


def test_orchestrator_predicates_delegate_to_the_contract():
    from arrayscope.window.render import RenderOrchestrator

    orchestrator = RenderOrchestrator.__new__(RenderOrchestrator)
    orchestrator.win = SimpleNamespace(_closing=False)
    orchestrator._render_generation = RenderGeneration(current=5)
    assert orchestrator._is_current_render_generation(5)
    assert not orchestrator._is_current_render_generation(4)
    orchestrator.win._closing = True
    assert not orchestrator._is_current_render_generation(5)

    orchestrator.win._closing = False
    session = _session()
    orchestrator._montage_session = session
    assert orchestrator._montage_session_is_current(session)
    assert orchestrator._is_current_montage_session(1, ("doc", "view"))
    orchestrator._montage_session = _session(session_id=2)
    assert not orchestrator._montage_session_is_current(session)


def test_orchestrator_owns_the_render_generation():
    """The generation guard lives on the orchestrator, not the window."""

    from arrayscope.window.render import RenderOrchestrator

    orchestrator = RenderOrchestrator.__new__(RenderOrchestrator)
    orchestrator.win = SimpleNamespace(_closing=False)
    orchestrator._render_generation = RenderGeneration()
    first = orchestrator._advance_render_generation("test")
    assert orchestrator._capture_render_generation() == first
    assert orchestrator._render_generation.last_reason == "test"


def test_orchestrator_exposes_window_work_graph():
    from arrayscope.core.work_graph import WorkGraph
    from arrayscope.window.render import RenderOrchestrator

    graph = WorkGraph()
    orchestrator = RenderOrchestrator.__new__(RenderOrchestrator)
    orchestrator.win = SimpleNamespace(work_graph=graph)
    assert orchestrator.work_graph is graph
