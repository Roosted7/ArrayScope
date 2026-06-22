import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_profile_montage_workflow_py_spy_command_mentions_external_sampler():
    from arrayscope.tools.profile_montage_workflow import py_spy_command

    command = py_spy_command(("--backend", "all", "--jsonl", "out.jsonl"))

    assert command.startswith("py-spy record")
    assert "arrayscope.tools.profile_montage_workflow" in command
    assert "--backend all" in command


def test_profile_montage_completion_waits_for_fully_visible_vispy_draw():
    from arrayscope.tools.profile_montage_workflow import _wait_for_montage_complete

    class FakeQtCore:
        class QEventLoop:
            class ProcessEventsFlag:
                AllEvents = object()

    class FakeNative:
        def isVisible(self):
            return True

    class FakeImageView:
        def __init__(self):
            self._vispy_canvas_native = FakeNative()
            self.diagnostics = {
                "draw_count": 0,
                "tile_presentation_request_count": 4,
                "tile_presentation_draw_count": 3,
                "tile_presentation_draw_pending": True,
                "presented_tiles": (0, 1),
                "presented_tile_count": 2,
                "tile_visual_visible_pages": 1,
                "tile_visual_min_order": 10,
                "overlay_count": 1,
                "overlay_visual_visible_items": 1,
                "overlay_visual_max_order": 5,
                "overlays_above_tiles": False,
            }

        def montageDisplayMode(self):
            return "vispy_tile_layer"

        def montageTileOverlayCount(self):
            return 1

        def vispyPresentationDiagnostics(self):
            return dict(self.diagnostics)

    class FakeApp:
        def __init__(self, image_view):
            self.image_view = image_view
            self.calls = 0

        def processEvents(self, *_args):
            self.calls += 1
            self.image_view.diagnostics["draw_count"] = self.calls
            if self.calls >= 2:
                self.image_view.diagnostics["tile_presentation_draw_count"] = 4
                self.image_view.diagnostics["tile_presentation_draw_pending"] = False

    image_view = FakeImageView()
    session = SimpleNamespace(
        visible_tiles=(SimpleNamespace(montage_index=0), SimpleNamespace(montage_index=1)),
        skipped_tiles=set(),
        presented_tiles={0, 1},
        display_committed=True,
        pending_tiles=(),
        loading_tiles=set(),
        pending_completed_tiles=(),
        active_tile_requests=set(),
        active_stage_requests=set(),
        attached_stage_requests=set(),
        stage_waiting_tiles={},
        final_commit_pending=False,
        flush_pending=False,
        deferred_display_tiles=(),
        dirty_payloads={},
        pending_removals=set(),
        is_complete=lambda: True,
    )
    win = SimpleNamespace(img_view=image_view, _montage_session=session)
    app = FakeApp(image_view)

    result = _wait_for_montage_complete(
        app,
        FakeQtCore,
        win,
        timeout_s=0.5,
        start=0.0,
        draw_start=0,
    )

    assert app.calls >= 2
    assert result["active_presented_tile_count"] == 2
    assert result["active_planned_tile_count"] == 2
    assert result["deferred_display_tile_count"] == 0
    assert result["fully_visible_ms"] is not None
    assert result["vispy_tile_presentation_draw_count"] == 4


@pytest.mark.skipif(
    os.environ.get("ARRAYSCOPE_RUN_PROFILE_WORKFLOW") != "1",
    reason="opt-in realistic GUI profiling workflow; set ARRAYSCOPE_RUN_PROFILE_WORKFLOW=1",
)
def test_profile_montage_workflow_realistic_dataset_optional(tmp_path):
    from arrayscope.tools.profile_montage_workflow import DEFAULT_DATA_PATH, run_profile_montage_workflow

    data_path = Path(os.environ.get("ARRAYSCOPE_PROFILE_DATA", DEFAULT_DATA_PATH))
    if not data_path.exists():
        pytest.skip(f"profile dataset not found: {data_path}")

    backends = tuple(
        backend.strip()
        for backend in os.environ.get("ARRAYSCOPE_PROFILE_BACKENDS", "pyqtgraph,vispy").split(",")
        if backend.strip()
    )
    if "vispy" in backends:
        pytest.importorskip("vispy")

    timeout_s = float(os.environ.get("ARRAYSCOPE_PROFILE_TIMEOUT_S", "180"))
    max_tiles_raw = int(os.environ.get("ARRAYSCOPE_PROFILE_MAX_TILES", "0"))
    max_tiles = None if max_tiles_raw <= 0 else max_tiles_raw
    jsonl = tmp_path / "profile-workflow.jsonl"
    all_records = []

    for backend in backends:
        records = run_profile_montage_workflow(
            data_path=data_path,
            backend=backend,
            jsonl=jsonl,
            timeout_s=timeout_s,
            max_tiles=max_tiles,
            show_window=os.environ.get("ARRAYSCOPE_PROFILE_HIDE_WINDOW") != "1",
        )
        all_records.extend(records)

    phases = {(record["backend"], record["phase"]) for record in all_records}
    for backend in backends:
        assert (backend, "load_data") in phases
        assert (backend, "raw_full_tiled_montage") in phases
        assert (backend, "fft_full_tiled_montage") in phases
    assert jsonl.exists()
