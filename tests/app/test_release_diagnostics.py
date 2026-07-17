import json

import pytest

from arrayscope.tools.interaction_budget import bounded_interaction_settle_timeout_s


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize(
    ("backend", "previous_backend"),
    (("pyqtgraph", "vispy"), ("vispy", "pyqtgraph")),
)
def test_release_diagnostics_writes_trace_and_preserves_image_renderer_choice(
    qt_app,
    tmp_path,
    backend,
    previous_backend,
):
    from pyqtgraph.Qt import QtCore

    from arrayscope.core.diagnostics_trace import summarize_diagnostics_trace
    from arrayscope.tools.release_diagnostics import capture_release_diagnostics

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", previous_backend)
    settings.sync()

    path = tmp_path / f"release-diagnostics-{backend}.jsonl"
    written = capture_release_diagnostics(path, backend=backend)

    assert written == path
    assert settings.value("image_rendering_backend") == previous_backend

    records = _read_jsonl(path)
    assert records[0]["event"] == "start"
    assert records[0]["app_version"] == "0.8.0"
    assert records[0]["config"]["image_rendering_backend_selected"] == backend
    assert [record["event"] for record in records[1:]] == ["snapshot", "snapshot", "snapshot"]
    assert records[-1]["diagnostics"]["montage"]["session_id"] is not None
    snapshots = [record["diagnostics"]["montage"] for record in records[1:]]
    assert [snapshot["visible_tiles"] for snapshot in snapshots] == [1, 1, 6]
    assert len({snapshot["session_id"] for snapshot in snapshots}) == 3
    target_signatures = [
        tuple(row["target_identity"] for row in snapshot["tile_identity_probe"])
        for snapshot in snapshots
    ]
    assert len(set(target_signatures)) == 3
    assert {row["target_source"] for row in snapshots[-1]["tile_identity_probe"]} == set(range(6))
    for montage in snapshots:
        assert montage["visible_tiles"] > 0
        assert montage["presented_tiles"] == montage["visible_tiles"]
        assert montage["tile_presentation_draw_pending"] is False

    summary = summarize_diagnostics_trace(path)
    assert summary.backend == backend
    assert summary.snapshot_count == 3


def test_release_diagnostics_rejects_unknown_backend(tmp_path):
    from arrayscope.tools.release_diagnostics import capture_release_diagnostics

    with pytest.raises(ValueError, match="unsupported backend"):
        capture_release_diagnostics(tmp_path / "diagnostics.jsonl", backend="unknown")


def test_capture_completion_rejects_geometry_only_and_incomplete_physical_truth():
    from arrayscope.tools.presentation_settlement import presentation_is_settled

    identity = object()

    class Session:
        session_id = 7
        render_generation = 3
        viewport_revision = 2
        key = ("target", 7)
        flush_pending = False
        final_commit_pending = False
        dirty_payloads = {}
        pending_payload_upserts = {}
        pending_removals = set()
        atomic_successor_pending = False
        stage_planning_deferred = False
        pending_rung_materializations = ()
        lod_page_cache = type("PageCache", (), {"pending_count": 0})()

        def __init__(self):
            self.complete = False
            self.lifecycle = type(
                "Lifecycle",
                (),
                {"backend_presented_identities": {0: identity}},
            )()

        def visible_plan_complete(self):
            return self.complete

        def required_target_settled(self):
            return self.complete

        def required_tile_numbers(self):
            return (0,)

        def required_target_unsettled_tiles(self):
            return () if self.complete else (0,)

        def has_pending_level_update(self):
            return False

        def is_complete(self):
            return self.complete

    class ImageView:
        draw_pending = False
        rows = {}
        rendering_capabilities = type("Capabilities", (), {"name": "vispy"})()

        def presentationDrawPending(self):
            return self.draw_pending

        def tileTruthPhysicalRows(self):
            return self.rows

    session = Session()
    image_view = ImageView()
    win = type(
        "Window",
        (),
        {
            "_current_montage_geometry": object(),
            "_frame_session": session,
            "img_view": image_view,
        },
    )()

    assert win._current_montage_geometry is not None
    assert not presentation_is_settled(win)

    session.complete = True
    assert not presentation_is_settled(win)

    image_view.rows = {
        0: {
            "physical_acknowledged_identity": identity,
            "physical_draw_bounds_match_layout": True,
        }
    }
    image_view.draw_pending = True
    assert not presentation_is_settled(win)

    image_view.draw_pending = False
    assert presentation_is_settled(win)

    for name, pending, cleared in (
        ("stage_planning_deferred", True, False),
        ("flush_pending", True, False),
        ("final_commit_pending", True, False),
        ("dirty_payloads", {0: None}, {}),
        ("pending_payload_upserts", {0: None}, {}),
        ("pending_removals", {0}, set()),
        ("atomic_successor_pending", True, False),
    ):
        setattr(session, name, pending)
        assert not presentation_is_settled(win)
        setattr(session, name, cleared)

    image_view.rows[0].pop("physical_draw_bounds_match_layout")
    assert not presentation_is_settled(win)
    image_view.rows[0]["physical_draw_bounds_match_layout"] = False
    assert not presentation_is_settled(win)
    image_view.rows[0]["physical_draw_bounds_match_layout"] = True

    image_view.rows[0]["physical_acknowledged_identity"] = object()
    assert not presentation_is_settled(win)
    image_view.rows[0]["physical_acknowledged_identity"] = identity

    image_view.rows[1] = {
        "physical_acknowledged_identity": object(),
        "physical_draw_bounds_match_layout": True,
    }
    assert not presentation_is_settled(win)
    image_view.rows.pop(1)

    image_view.rendering_capabilities = type(
        "Capabilities",
        (),
        {"name": "pyqtgraph"},
    )()
    image_view.rows[0] = {
        "physical_acknowledged_identity": identity,
        "physical_storage_mode": "image_item",
    }
    assert presentation_is_settled(win)
    assert presentation_is_settled(win, require_quiescent=True)
    session.pending_tiles = (object(),)
    assert not presentation_is_settled(win, require_quiescent=True)
    session.pending_tiles = ()
    session.lod_page_cache.pending_count = 1
    assert not presentation_is_settled(win, require_quiescent=True)
    session.lod_page_cache.pending_count = 0
    image_view.rows[0].pop("physical_storage_mode")
    assert not presentation_is_settled(win)


def test_wait_until_raises_loudly_on_timeout(qt_app):
    from pyqtgraph.Qt import QtCore

    from arrayscope.tools.release_diagnostics import _wait_until

    with pytest.raises(TimeoutError, match="fault-injected presentation did not settle"):
        _wait_until(
            qt_app,
            QtCore,
            lambda: False,
            timeout_s=bounded_interaction_settle_timeout_s(0.01),
            description="fault-injected presentation",
        )


def test_capture_propagates_incomplete_presentation_timeout(qt_app, tmp_path, monkeypatch):
    import arrayscope.tools.release_diagnostics as release_diagnostics

    monkeypatch.setattr(release_diagnostics, "presentation_is_settled", lambda _win, **_kwargs: False)
    monkeypatch.setattr(
        release_diagnostics,
        "bounded_interaction_settle_timeout_s",
        lambda _requested=None: 0.01,
    )

    with pytest.raises(TimeoutError, match="initial image physical presentation did not settle"):
        release_diagnostics.capture_release_diagnostics(
            tmp_path / "incomplete-release-diagnostics.jsonl",
            backend="pyqtgraph",
        )
