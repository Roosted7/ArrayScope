import json
import os
import shlex
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from arrayscope.core.runtime_diagnostics import (
    MontageRuntimeDiagnostics,
    MontageTimingDiagnostics,
    RenderTimingDiagnostics,
)
from arrayscope.tools.interaction_budget import bounded_interaction_settle_timeout_s


def test_tile_presentation_draw_wait_fails_loudly_when_request_is_not_drawn(monkeypatch):
    import arrayscope.tools.profile_montage_workflow as workflow

    monkeypatch.setattr(workflow, "_process_events", lambda *_args, **_kwargs: None)
    win = SimpleNamespace(
        img_view=SimpleNamespace(
            presentation_diagnostics=lambda: {
                "tile_presentation_request_count": 2,
                "tile_presentation_draw_count": 1,
            },
        )
    )

    with pytest.raises(TimeoutError, match=r"requested=2 drawn=1"):
        workflow._wait_for_tile_presentation_draw(
            win,
            object(),
            object(),
            timeout_s=bounded_interaction_settle_timeout_s(0.01),
        )


def test_phase_record_excludes_montage_counters_without_live_owners(monkeypatch):
    import arrayscope.tools.profile_montage_workflow as workflow

    snapshot = SimpleNamespace(
        image_rendering_backend_actual="wgpu",
        montage=MontageRuntimeDiagnostics(active=False),
        montage_timing=MontageTimingDiagnostics(),
        render_timing=RenderTimingDiagnostics(),
        resource_governor=None,
    )
    win = SimpleNamespace(
        collect_runtime_diagnostics=lambda: snapshot,
        renderer=SimpleNamespace(),
    )
    monkeypatch.setattr(workflow, "_window_geometry_state", lambda _win: {})
    monkeypatch.setattr(
        workflow,
        "_backend_presentation_diagnostics",
        lambda _win: {
            "draw_count": 7,
            "tile_presentation_request_count": 6,
            "tile_presentation_draw_count": 5,
            "presented_tile_count": 4,
            "presented_tiles": (0, 1, 2, 3),
            "wgpu_uploads_total": 11,
            "wgpu_upload_bytes_total": 4096,
            "wgpu_binding_fast_path_commits": 3,
        },
    )
    monkeypatch.setattr(workflow, "_wgpu_frame_cadence", lambda _win: {})
    monkeypatch.setattr(workflow, "_montage_overlay_count", lambda _win: 0)
    monkeypatch.setattr(
        workflow,
        "_montage_level_presentation_state",
        lambda _win: {
            "revision": 0,
            "target_levels": None,
            "stale_tiles": 0,
            "pending_tiles": 0,
            "settled": True,
            "active_tile_count": 0,
            "active_presented_tile_count": 0,
        },
    )

    record = workflow._phase_record(
        win,
        phase="unit",
        elapsed_ms=1.0,
        event_loop_p95_gap_ms=None,
        event_loop_p99_gap_ms=None,
        event_loop_max_gap_ms=0.0,
    )

    assert "montage_quality_ingest_reductions" not in record
    assert "montage_quality_stage_hits_serving_derivations" not in record
    assert "montage_quality_preview_reduced_scheduled" in record
    assert record["presentation_draw_count"] == 7
    assert record["tile_presentation_request_count"] == 6
    assert record["tile_presentation_draw_count"] == 5
    assert record["presented_tile_count"] == 4
    assert record["presented_tiles"] == [0, 1, 2, 3]
    assert record["wgpu_uploads_total"] == 11
    assert record["wgpu_upload_bytes_total"] == 4096
    assert record["wgpu_binding_fast_path_commits"] == 3


def _journey_gesture_win(pending_fn, capture_log):
    return SimpleNamespace(
        img_view=SimpleNamespace(presentationDrawPending=pending_fn),
        _frame_session=None,
        _arrayscope_active_gesture_id="zoom_out-1",
        _arrayscope_active_journey="zoom_out",
        _arrayscope_active_gesture_started_ns=1,
        _arrayscope_visual_timeline_probe=SimpleNamespace(
            capture=lambda reason: capture_log.append(reason)
        ),
    )


def test_journey_end_sample_waits_for_pending_presentation_draw(monkeypatch):
    """Matrix v6/v7 zoom_out red (2026-07-18): a descriptor-only gesture's
    single repaint ran one scheduler tick after the last camera step, so the
    journey-end screenshot recorded the stale predecessor frame and the
    freshness oracle saw no pixel change ever. The end sample must key on
    presentation-draw acks and the production quiet edge: capture only after
    pending draws execute and the resulting COVERAGE pass closes."""

    import arrayscope.tools.profile_montage_workflow as workflow

    pending = {"draws": 1, "quiet": 3}
    events = []
    monkeypatch.setattr(
        workflow, "emit_trace", lambda kind, **payload: events.append((kind, payload))
    )
    captures = []
    win = _journey_gesture_win(lambda: pending["draws"] > 0, captures)
    win._interaction_active_now = lambda: pending["quiet"] > 0
    win._frame_session = SimpleNamespace(
        scheduling_policy=SimpleNamespace(
            verdict=SimpleNamespace(coverage_open=True),
        ),
    )

    def process_events(*_args, **_kwargs):
        pending.update(
            draws=max(0, pending["draws"] - 1),
            quiet=max(0, pending["quiet"] - 1),
        )
        if pending["quiet"] == 0:
            win._frame_session.scheduling_policy.verdict.coverage_open = False

    monkeypatch.setattr(workflow, "_process_events", process_events)

    workflow._finish_journey_gesture(win, "zoom_out-1", app=object(), QtCore=object())

    assert captures == ["journey-end"]
    assert pending == {"draws": 0, "quiet": 0}
    assert events[-1][1]["presentation_drained"] is True


def test_journey_end_drain_is_bounded_when_redraw_never_comes(monkeypatch):
    import arrayscope.tools.profile_montage_workflow as workflow

    pumps = []
    monkeypatch.setattr(workflow, "_process_events", lambda *_args, **_kwargs: pumps.append(1))
    win = SimpleNamespace(img_view=SimpleNamespace(presentationDrawPending=lambda: True))

    settled = workflow._drain_presentation_draw_for_journey_sample(
        win,
        object(),
        object(),
        timeout_s=bounded_interaction_settle_timeout_s(0.05),
    )

    assert settled is False  # gave up within the bound instead of raising
    assert pumps  # it did try to run the dispatcher


def test_finish_journey_gesture_still_samples_when_redraw_is_missed(monkeypatch):
    """Injected missed redraw: the pending flag never clears. The journey-end
    sample must still be captured (its stale pixels keep the freshness oracle
    red) and the give-up must be recorded as evidence, not raised."""

    import arrayscope.tools.profile_montage_workflow as workflow

    monkeypatch.setattr(workflow, "_process_events", lambda *_args, **_kwargs: None)
    events = []
    monkeypatch.setattr(
        workflow, "emit_trace", lambda kind, **payload: events.append((kind, payload))
    )
    captures = []
    win = _journey_gesture_win(lambda: True, captures)

    workflow._finish_journey_gesture(win, "zoom_out-1", app=object(), QtCore=object())

    assert captures == ["journey-end"]
    assert events[-1][1]["presentation_drained"] is False


def test_tile_presentation_draw_wait_fails_loudly_when_camera_draw_is_pending(monkeypatch):
    """A descriptor-only camera redraw is physical work even with no commit edge."""

    import arrayscope.tools.profile_montage_workflow as workflow

    monkeypatch.setattr(workflow, "_process_events", lambda *_args, **_kwargs: None)
    win = SimpleNamespace(
        img_view=SimpleNamespace(
            presentationDrawPending=lambda: True,
            wgpuPresentationDiagnostics=lambda: {
                "draw_count": 8,
                "tile_presentation_request_count": 3,
                "tile_presentation_draw_count": 3,
            },
        )
    )

    with pytest.raises(TimeoutError, match=r"draw_pending=True"):
        workflow._wait_for_tile_presentation_draw(
            win,
            object(),
            object(),
            timeout_s=bounded_interaction_settle_timeout_s(0.01),
        )


def test_tile_presentation_draw_wait_accepts_qgraphics_draw_edge_during_active_fill(
    monkeypatch, qt_app
):
    """A cold fill may re-arm paint debt in the same dispatcher turn.

    The screenshot gate needs proof that pixels painted after its milestone;
    it must not require the whole producer stream to become idle first.
    """

    from pyqtgraph.Qt import QtCore

    import arrayscope.tools.profile_montage_workflow as workflow

    class DrawEmitter(QtCore.QObject):
        drawn = QtCore.Signal()

    emitter = DrawEmitter()
    pumps = {"count": 0}

    def process_events(*_args, **_kwargs):
        pumps["count"] += 1
        if pumps["count"] == 1:
            emitter.drawn.emit()

    monkeypatch.setattr(workflow, "_process_events", process_events)
    win = SimpleNamespace(
        img_view=SimpleNamespace(
            _paints_qgraphics_scene=lambda: True,
            presentationDrawn=emitter.drawn,
            presentationDrawPending=lambda: True,
        )
    )

    workflow._wait_for_tile_presentation_draw(
        win,
        qt_app,
        QtCore,
        timeout_s=0.02,
        target_s=0.01,
    )

    assert pumps["count"] == 1


def test_coverage_pass_wait_fails_loudly_when_evidence_never_closes(monkeypatch):
    import arrayscope.tools.profile_montage_workflow as workflow

    monkeypatch.setattr(workflow, "_process_events", lambda *_args, **_kwargs: None)
    win = SimpleNamespace(
        _frame_session=SimpleNamespace(
            scheduling_policy=SimpleNamespace(
                verdict=SimpleNamespace(coverage_open=True),
            )
        )
    )

    with pytest.raises(TimeoutError, match=r"coverage_open=True"):
        workflow._wait_for_coverage_pass_close(
            win,
            object(),
            object(),
            timeout_s=bounded_interaction_settle_timeout_s(0.01),
        )


def test_physical_quiet_wait_fails_loudly_while_draw_is_pending(monkeypatch):
    import arrayscope.tools.profile_montage_workflow as workflow

    monkeypatch.setattr(workflow, "_process_events", lambda *_args, **_kwargs: None)
    win = SimpleNamespace(
        img_view=SimpleNamespace(
            presentationDrawPending=lambda: True,
            presentation_diagnostics=lambda: {"draw_count": 7},
        )
    )

    with pytest.raises(TimeoutError, match=r"draw_count=7 draw_pending=True"):
        workflow._wait_for_physical_presentation_quiet(
            win,
            object(),
            object(),
            timeout_s=bounded_interaction_settle_timeout_s(0.01),
        )


def test_physical_quiet_wait_ignores_draw_churn_after_presentation_ack(monkeypatch):
    import arrayscope.tools.profile_montage_workflow as workflow

    draw_count = {"value": 7}

    def process_events(*_args, **_kwargs):
        draw_count["value"] += 1

    monkeypatch.setattr(workflow, "_process_events", process_events)
    win = SimpleNamespace(
        img_view=SimpleNamespace(
            presentationDrawPending=lambda: False,
            presentation_diagnostics=lambda: {
                "draw_count": draw_count["value"],
            },
        )
    )

    workflow._wait_for_physical_presentation_quiet(
        win,
        object(),
        object(),
        timeout_s=bounded_interaction_settle_timeout_s(0.2),
    )

    assert draw_count["value"] > 7


def test_profile_view_range_uses_canonical_pointer_interaction_reason():
    from arrayscope.tools.profile_montage_workflow import _apply_view_range

    reasons = []
    ranges = []
    retargets = []
    image_view = SimpleNamespace()

    def set_range(**kwargs):
        ranges.append(
            {
                **kwargs,
                "wheel_identity": image_view._viewport_wheel_range_pending,
            }
        )

    view = SimpleNamespace(setRange=set_range)
    image_view.getView = lambda: view
    win = SimpleNamespace(
        _note_viewport_interaction=lambda reason: reasons.append(reason),
        img_view=image_view,
        retarget_montage_viewport=lambda: retargets.append(True),
    )

    _apply_view_range(win, (1.0, 2.0), (3.0, 4.0))

    assert reasons == ["range-pointer"]
    assert ranges == [
        {
            "xRange": (1.0, 2.0),
            "yRange": (3.0, 4.0),
            "padding": 0,
            "wheel_identity": True,
        }
    ]
    assert image_view._viewport_wheel_range_pending is False
    assert retargets == [True]


def test_visual_sampler_captures_active_presentation_draw_ack():
    from arrayscope.tools.profile_montage_workflow import _VisualTimelineProbe

    reasons = []
    probe = object.__new__(_VisualTimelineProbe)
    probe._win = SimpleNamespace(_arrayscope_active_gesture_id="zoom_out-1")
    probe._interval_ms = 100
    probe._last_sample_ns = 0
    probe.capture = lambda reason: reasons.append(reason)

    probe._capture_presentation_draw_ack()
    probe._win._arrayscope_active_gesture_id = ""
    probe._capture_presentation_draw_ack()

    assert reasons == ["presentation-draw-ack"]


def test_visual_sampler_uses_draw_acks_only_during_wgpu_gesture():
    from arrayscope.tools.profile_montage_workflow import _VisualTimelineProbe

    reasons = []
    probe = object.__new__(_VisualTimelineProbe)
    probe._win = SimpleNamespace(_arrayscope_active_gesture_id="index_scroll-1")
    probe._backend = "wgpu"
    probe._interval_ms = 100
    probe._last_sample_ns = 0
    probe.capture = lambda reason: reasons.append(reason)

    probe._capture_interval()
    probe._capture_presentation_draw_ack()
    probe._backend = "pyqtgraph"
    probe._capture_interval()
    probe._win._arrayscope_active_gesture_id = ""
    probe._backend = "wgpu"
    probe._capture_interval()

    assert reasons == ["presentation-draw-ack", "interval", "interval"]


def test_visual_sampler_throttles_wgpu_draw_ack_screenshots(monkeypatch):
    from arrayscope.tools.profile_montage_workflow import _VisualTimelineProbe

    reasons = []
    probe = object.__new__(_VisualTimelineProbe)
    probe._win = SimpleNamespace(_arrayscope_active_gesture_id="index_scroll-1")
    probe._interval_ms = 500
    probe._last_sample_ns = 1_000_000_000
    probe.capture = lambda reason: reasons.append(reason)

    monkeypatch.setattr(
        "arrayscope.tools.profile_montage_workflow.time.monotonic_ns",
        lambda: 1_499_000_000,
    )
    probe._capture_presentation_draw_ack()
    assert reasons == []

    monkeypatch.setattr(
        "arrayscope.tools.profile_montage_workflow.time.monotonic_ns",
        lambda: 1_500_000_000,
    )
    probe._capture_presentation_draw_ack()
    assert reasons == ["presentation-draw-ack"]


def test_visual_sampler_never_grabs_recursively_inside_paint(qt_app, tmp_path):
    from pyqtgraph.Qt import QtCore, QtWidgets

    from arrayscope.tools.profile_montage_workflow import _VisualTimelineProbe

    class DrawEmitter(QtCore.QObject):
        drawn = QtCore.Signal()

    emitter = DrawEmitter()
    win = QtWidgets.QWidget()
    win.img_view = SimpleNamespace(presentationDrawn=emitter.drawn)
    win._arrayscope_active_gesture_id = "index_scroll-1"
    probe = _VisualTimelineProbe(
        QtCore,
        None,
        win,
        backend="pyqtgraph",
        directory=tmp_path,
        interval_s=1.0,
    )
    reasons = []
    probe.capture = lambda reason: reasons.append(reason)
    probe.start()
    probe._last_sample_ns = 0

    emitter.drawn.emit()

    assert reasons == ["start"], "capture must not re-enter the emitting paint stack"
    qt_app.processEvents()
    assert reasons == ["start", "presentation-draw-ack"]
    probe.stop()


def test_preview_floor_physical_rows_preserve_page_shader_evidence():
    from arrayscope.tools.profile_montage_workflow import _preview_floor_physical_rows

    physical = {
        7: {
            "physical_page": 2,
            "physical_slot": 4,
            "physical_texture_kind": "complex_rg32f",
            "physical_storage_mode": "complex",
            "physical_texture_dtype": "float32",
            "physical_texture_shape": (84, 84, 2),
            "physical_mapping_mode": 4.0,
            "physical_component_mode": 2.0,
            "physical_levels": (0.0, 8.0),
            "physical_shader_mapping_key": "phase",
            "unbounded_identity": object(),
        }
    }
    win = SimpleNamespace(
        img_view=SimpleNamespace(tileTruthPhysicalRows=lambda: physical),
    )

    assert _preview_floor_physical_rows(win) == [
        {
            "tile": 7,
            "physical_page": 2,
            "physical_slot": 4,
            "physical_texture_kind": "complex_rg32f",
            "physical_storage_mode": "complex",
            "physical_texture_dtype": "float32",
            "physical_texture_shape": (84, 84, 2),
            "physical_mapping_mode": 4.0,
            "physical_component_mode": 2.0,
            "physical_levels": (0.0, 8.0),
            "physical_shader_mapping_key": "phase",
        }
    ]


def test_wgpu_continuity_count_uses_allocation_light_physical_query():
    from arrayscope.tools.profile_montage_workflow import _backend_visible_tile_count

    view = SimpleNamespace(
        montageDisplayMode=lambda: "wgpu_tile_layer",
        physicalVisibleTileCount=lambda: 37,
        wgpuPresentationDiagnostics=lambda: pytest.fail(
            "continuity sampling rebuilt detailed WGPU diagnostics"
        ),
    )

    assert _backend_visible_tile_count(SimpleNamespace(img_view=view)) == 37


def test_verbose_physical_row_preserves_atlas_and_identity_evidence():
    from arrayscope.tools.profile_montage_workflow import _verbose_physical_row

    identity = ("document", 41, "level", 2)
    row = _verbose_physical_row(
        {
            "physical_page": 3,
            "physical_slot": 8,
            "physical_texture_kind": "complex_rg32f",
            "physical_storage_mode": "complex",
            "physical_texture_dtype": "float32",
            "physical_texture_shape": (84, 84, 2),
            "physical_real_plane_identity": {"pointer": 1234},
            "physical_imag_plane_identity": {"pointer": 1238},
            "physical_mapping_mode": 4.0,
            "physical_component_mode": 2.0,
            "physical_levels": (0.0, 8.0),
            "physical_shader_mapping_key": "phase",
            "physical_acknowledged_identity": identity,
        }
    )

    assert row["physical_page"] == 3
    assert row["physical_slot"] == 8
    assert row["physical_real_plane_identity"] == {"pointer": 1234}
    assert row["physical_imag_plane_identity"] == {"pointer": 1238}
    assert row["physical_acknowledged_identity"] == repr(identity)


def test_profile_montage_workflow_py_spy_command_mentions_external_sampler():
    from arrayscope.tools.profile_montage_workflow import py_spy_command

    command = py_spy_command(("--backend", "all", "--jsonl", "out.jsonl"))

    assert command.startswith("py-spy record")
    assert "arrayscope.tools.profile_montage_workflow" in command
    assert "--backend all" in command
    assert "--native" not in command
    assert "--rate 25" in command
    assert "--gil" in command
    assert "--nonblocking" in command


def test_visual_timeline_groups_payload_lod_and_quality():
    from types import SimpleNamespace

    from arrayscope.tools.profile_montage_workflow import _visual_lod_level_counts

    payloads = {
        0: SimpleNamespace(lod=SimpleNamespace(level=0), quality="exact"),
        1: SimpleNamespace(lod=SimpleNamespace(level=2), quality="preview"),
        2: SimpleNamespace(lod=SimpleNamespace(level=2), quality="preview"),
        3: SimpleNamespace(lod=SimpleNamespace(level=1), quality="exact"),
    }

    assert _visual_lod_level_counts(payloads, {0, 1, 2}) == {
        "exact:L0": 1,
        "preview:L2": 2,
    }


def test_visual_timeline_preserves_physical_draw_geometry():
    from arrayscope.tools.profile_montage_workflow import _visual_physical_draw_rows

    rows = _visual_physical_draw_rows(
        {
            3: {
                "physical_draw_world_rects": ((1.0, 2.0, 5.0, 6.0),),
                "physical_draw_uv_rects": ((0.1, 0.2, 0.5, 0.6),),
                "physical_draw_world_bounds": (1.0, 2.0, 5.0, 6.0),
                "physical_expected_world_rect": (1.0, 2.0, 5.0, 6.0),
                "physical_draw_bounds_match_layout": True,
                "physical_texture_kind": "scalar_r32f",
                "physical_storage_mode": "scalar",
                "physical_texture_dtype": "float32",
                "physical_texture_shape": (4, 4),
                "physical_mapping_mode": 0.0,
                "physical_component_mode": 0.0,
                "physical_levels": (2.0, 6.0),
                "physical_shader_mapping_key": "linear-real",
            },
            9: {"physical_draw_bounds_match_layout": False},
        },
        {3},
    )

    assert rows == {
        "3": {
            "texture_kind": "scalar_r32f",
            "storage_mode": "scalar",
            "texture_dtype": "float32",
            "texture_shape": (4, 4),
            "mapping_mode": 0.0,
            "component_mode": 0.0,
            "levels": (2.0, 6.0),
            "shader_mapping_key": "linear-real",
            "draw_world_rects": ((1.0, 2.0, 5.0, 6.0),),
            "draw_world_bounds": (1.0, 2.0, 5.0, 6.0),
            "draw_uv_rects": ((0.1, 0.2, 0.5, 0.6),),
            "expected_world_rect": (1.0, 2.0, 5.0, 6.0),
            "bounds_match_layout": True,
            "page_bindings": (),
        }
    }


def test_headless_capture_must_return_exact_window(qt_app, tmp_path, monkeypatch):
    import numpy as np
    from pyqtgraph.Qt import QtCore, QtGui

    import arrayscope.tools.profile_montage_workflow as workflow

    monkeypatch.setenv("ARRAYSCOPE_HEADLESS_DISPLAY", "arrayscope-headless-test")

    def capture_managed(path):
        image = QtGui.QImage(40, 30, QtGui.QImage.Format.Format_RGBA8888)
        image.fill(QtGui.QColor("magenta"))
        assert image.save(str(path))
        return path

    monkeypatch.setattr(workflow, "capture_output", capture_managed)
    geometry = SimpleNamespace(
        size=lambda: QtCore.QSize(40, 30),
        width=lambda: 40,
        height=lambda: 30,
    )
    win = SimpleNamespace(
        img_view=SimpleNamespace(wgpuPresentMethod=lambda: "screen"),
        size=lambda: geometry.size(),
        frameGeometry=lambda: geometry,
    )
    path = tmp_path / "window.png"

    assert workflow._save_view_screenshot(win, path) is True
    assert win._arrayscope_last_screenshot_capture_kind == "managed-weston-window"
    assert win._arrayscope_last_screenshot_capture_error == ""
    assert QtGui.QImage(str(path)).size() == geometry.size()
    frame = np.full((12, 18, 4), 127, dtype=np.uint8)
    win.img_view.grabPresentedFramebuffer = lambda: frame
    monkeypatch.setattr(
        workflow,
        "capture_output",
        lambda *_args, **_kwargs: pytest.fail("timeline capture called managed Weston"),
    )
    timeline_path = tmp_path / "timeline.png"

    assert workflow._save_view_screenshot(win, timeline_path, full_window=False) is True
    assert win._arrayscope_last_screenshot_capture_kind == "wgpu-offscreen-replay"


def test_visual_geometry_summary_projects_physical_bounds_through_live_camera():
    from arrayscope.tools.profile_montage_workflow import _visual_geometry_summary

    summary = _visual_geometry_summary(
        {
            "3": {
                "draw_world_bounds": (0.0, 0.0, 336.0, 336.0),
                "bounds_match_layout": True,
            },
            "4": {
                "draw_world_bounds": (336.0, 0.0, 672.0, 336.0),
                "bounds_match_layout": False,
            },
        },
        view_range=((0.0, 672.0), (0.0, 336.0)),
        viewport_shape=(500, 1000),
    )

    assert summary == {
        "world_size_classes": ((336.0, 336.0),),
        "projected_pixel_size_classes": ((500.0, 500.0),),
        "mixed_world_sizes": False,
        "mixed_projected_pixel_sizes": False,
        "bounds_mismatch_tiles": (4,),
    }


def test_visual_scene_presented_tiles_does_not_treat_residency_as_drawn():
    from arrayscope.tools.profile_montage_workflow import _visual_scene_presented_tiles

    assert _visual_scene_presented_tiles(
        "wgpu",
        presentation_diagnostics={"page_table_resident_count": 60},
        physical_rows={40: {}, 41: {}, 50: {}, 51: {}},
    ) == frozenset({40, 41, 50, 51})
    assert _visual_scene_presented_tiles(
        "pyqtgraph",
        presentation_diagnostics={},
        physical_rows={4: {}, 8: {}},
    ) == frozenset({4, 8})


def test_visual_camera_state_reports_session_and_live_range_drift():
    from arrayscope.tools.profile_montage_workflow import _visual_camera_state

    state = _visual_camera_state(
        SimpleNamespace(),
        session=SimpleNamespace(view_range=((0.0, 10.0), (0.0, 20.0))),
        live_view_range=((1.0, 11.0), (2.0, 22.0)),
    )

    assert state["session_matches_live"] is False
    assert state["session_view_range"] == ((0.0, 10.0), (0.0, 20.0))
    assert state["live_view_range"] == ((1.0, 11.0), (2.0, 22.0))


def test_view_intersection_distinguishes_off_content_camera_and_visible_tile():
    from arrayscope.tools.profile_montage_workflow import (
        _view_range_intersects_world_bounds,
        _view_ranges_intersect,
    )

    content = ((0.0, 100.0), (0.0, 80.0))
    assert _view_ranges_intersect(((20.0, 40.0), (10.0, 30.0)), content)
    assert not _view_ranges_intersect(((-90.0, -10.0), (100.0, 180.0)), content)
    assert _view_range_intersects_world_bounds(
        ((20.0, 40.0), (10.0, 30.0)),
        (0.0, 0.0, 30.0, 20.0),
    )


def test_synthetic_geometry_scene_is_indexed_and_spatially_recognizable():
    import numpy as np

    from arrayscope.tools.profile_montage_workflow import _synthetic_profile_data

    data = _synthetic_profile_data("geometry", (48, 64, 8))

    assert data.shape == (48, 64, 8)
    assert data.dtype == np.float32
    assert data.flags.c_contiguous
    assert float(data.min()) >= 0.0
    assert float(data.max()) <= 1.0
    assert not np.array_equal(data[..., 0], data[..., 1])
    assert not np.array_equal(data[0, :, :], data[-1, :, :])


def test_synthetic_complex_scene_has_amplitude_phase_and_zero_fiducials():
    import numpy as np

    from arrayscope.tools.profile_montage_workflow import _synthetic_profile_data

    data = _synthetic_profile_data("complex-phase", (48, 64, 8))

    assert data.dtype == np.complex64
    assert data.flags.c_contiguous
    assert np.count_nonzero(data == 0) > 0
    assert np.ptp(np.abs(data)) > 0.5
    assert np.unique(np.round(np.angle(data[data != 0]), 2)).size > 20


def test_profile_suite_commands_cover_required_profilers(tmp_path):
    from arrayscope.tools.profile_montage_workflow import profiler_suite_commands

    commands = profiler_suite_commands(
        ("--backend", "wgpu", "--profile-suite", str(tmp_path)), tmp_path
    )

    assert {item["profiler_type"] for item in commands} == {
        "plain",
        "py-spy-raw-low-impact",
        "py-spy-raw-full",
        "perf-record",
    }
    by_type = {item["profiler_type"]: item for item in commands}
    assert "py-spy record" in by_type["py-spy-raw-low-impact"]["command"]
    assert "--format raw" in by_type["py-spy-raw-low-impact"]["command"]
    assert "--native" not in by_type["py-spy-raw-low-impact"]["command"]
    assert "--rate 25" in by_type["py-spy-raw-low-impact"]["command"]
    assert "--gil" in by_type["py-spy-raw-low-impact"]["command"]
    assert "--nonblocking" in by_type["py-spy-raw-low-impact"]["command"]
    assert "--format raw" in by_type["py-spy-raw-full"]["command"]
    assert "--duration 30" in by_type["py-spy-raw-full"]["command"]
    assert "--rate 50" in by_type["py-spy-raw-full"]["command"]
    assert "started + duration + margin" in by_type["py-spy-raw-full"]["command"]
    assert "--gil" not in by_type["py-spy-raw-full"]["command"]
    assert "--nonblocking" in by_type["py-spy-raw-full"]["command"]
    assert "perf record" in by_type["perf-record"]["command"]
    assert "-F 99" in by_type["perf-record"]["command"]
    assert "--profile-suite" not in by_type["plain"]["command"]
    for item in commands:
        assert item["jsonl"].endswith(".jsonl")
        assert item["artifact_paths"]


def test_profile_suite_can_opt_into_cprofile_without_passing_flag_to_child(tmp_path):
    from arrayscope.tools.profile_montage_workflow import profiler_suite_commands

    commands = profiler_suite_commands(
        ("--backend", "wgpu", "--profile-suite", str(tmp_path), "--include-cprofile"),
        tmp_path,
    )
    by_type = {item["profiler_type"]: item for item in commands}

    assert "cprofile" in by_type
    assert "cProfile" in by_type["cprofile"]["command"]
    assert "--include-cprofile" not in by_type["plain"]["command"]


def test_profile_parser_stages_resolve_and_deconflict():
    from arrayscope.tools.profile_montage_workflow import (
        _parse_stage_flags,
        _resolve_profile_stages,
    )

    stages = _resolve_profile_stages(
        include_stages=_parse_stage_flags(
            (
                "raw_full_tiled_montage,montage_zoompan_fft",
                "montage_zoompan_fft",
                "montage_scroll_scalar",
            )
        ),
        skip_stages=_parse_stage_flags(("montage_zoompan_fft",)),
    )
    assert stages == ("raw_full_tiled_montage", "montage_scroll_scalar")


def test_profile_stage_resolve_defaults_to_all():
    from arrayscope.tools.profile_montage_workflow import (
        PROFILE_MONTAGE_STAGES,
        _resolve_profile_stages,
    )

    assert _resolve_profile_stages() == tuple(PROFILE_MONTAGE_STAGES)
    assert "display_x_axis_slice" in PROFILE_MONTAGE_STAGES
    assert "display_y_axis_slice" in PROFILE_MONTAGE_STAGES


def test_display_axis_crop_scenarios_cover_geometry_and_boundary_classes():
    from arrayscope.tools.profile_montage_workflow import _display_axis_crop_scenarios

    scenarios = _display_axis_crop_scenarios(
        shape=(336, 336, 272),
        image_axes=(1, 0),
        primary_role="x",
    )
    by_name = {scenario.name: scenario for scenario in scenarios}

    assert tuple(by_name) == (
        "primary-only-centered",
        "both-centered",
        "primary-minus-one",
        "secondary-plus-one",
        "both-diagonal",
        "both-return",
        "primary-page-edge",
        "primary-page-cross",
        "primary-page-return",
        "both-odd",
        "both-odd-primary-plus-one",
    )
    assert by_name["primary-only-centered"].cropped_axis_count == 1
    assert all(
        scenario.cropped_axis_count == 2
        for scenario in scenarios
        if scenario.name != "primary-only-centered"
    )
    assert by_name["both-centered"].axis_ranges != by_name["both-diagonal"].axis_ranges
    assert by_name["both-diagonal"].axis_ranges != by_name["both-return"].axis_ranges
    assert by_name["primary-page-edge"].crosses_page_boundary is False
    assert by_name["primary-page-cross"].crosses_page_boundary is True
    assert by_name["primary-page-return"].axis_ranges == by_name["primary-page-edge"].axis_ranges
    odd_lengths = tuple(len(indices) for _, indices, _ in by_name["both-odd"].axis_ranges)
    assert sorted(odd_lengths) == [99, 101]


def test_display_axis_crop_scenarios_exercise_both_roles_without_special_shapes():
    from arrayscope.tools.profile_montage_workflow import _display_axis_crop_scenarios

    x_scenarios = _display_axis_crop_scenarios(
        shape=(336, 336, 272),
        image_axes=(1, 0),
        primary_role="x",
    )
    y_scenarios = _display_axis_crop_scenarios(
        shape=(336, 336, 272),
        image_axes=(1, 0),
        primary_role="y",
    )

    assert tuple(scenario.name for scenario in x_scenarios) == tuple(
        scenario.name for scenario in y_scenarios
    )
    assert x_scenarios[0].axis_ranges[0][0] == 0
    assert y_scenarios[0].axis_ranges[0][0] == 1
    assert x_scenarios[-1].axis_ranges != y_scenarios[-1].axis_ranges


def test_physical_pixel_sampling_is_bounded_replayable_and_seeded():
    import numpy as np

    from arrayscope.tools.framebuffer_reference import _bounded_sample_positions

    select = np.ones(4096, dtype=bool)
    first = _bounded_sample_positions(
        select,
        max_samples=64,
        sample_seed=17,
        tile_number=3,
    )
    replay = _bounded_sample_positions(
        select,
        max_samples=64,
        sample_seed=17,
        tile_number=3,
    )
    varied = _bounded_sample_positions(
        select,
        max_samples=64,
        sample_seed=18,
        tile_number=3,
    )

    assert first.size == 64
    assert np.array_equal(first, replay)
    assert not np.array_equal(first, varied)
    assert np.all(select[first])


def test_profile_axis_window_shift_covers_display_montage_and_slice_dimensions():
    from arrayscope.core.view_state import ViewState
    from arrayscope.tools.profile_montage_workflow import _shift_profile_axis_window

    state = (
        ViewState.from_shape((20, 30, 40, 50))
        .with_image_axes(0, 1)
        .with_axis_range(0, indices=tuple(range(3, 13)), text="3:13")
        .with_axis_range(1, indices=tuple(range(7, 19)), text="7:19")
        .with_slice(3, 25)
        .with_montage_axis(2, columns=4, indices=tuple(range(10, 22)), text="10:22")
    )

    shifted_y = _shift_profile_axis_window(state, 0, 2)
    shifted_x = _shift_profile_axis_window(state, 1, -3)
    shifted_montage = _shift_profile_axis_window(state, 2, 3)
    shifted_slice = _shift_profile_axis_window(state, 3, -4)

    assert shifted_y.axis_range_indices[0] == tuple(range(5, 15))
    assert shifted_x.axis_range_indices[1] == tuple(range(4, 16))
    assert shifted_montage.montage_indices == tuple(range(13, 25))
    assert shifted_slice.slice_indices[3] == 21
    assert shifted_y.axis_range_indices[1] == state.axis_range_indices[1]
    assert shifted_montage.axis_range_indices == state.axis_range_indices


@given(
    role=st.sampled_from(("display-y", "display-x", "montage", "slice")),
    sign=st.sampled_from((-1, 1)),
    cropped=st.booleans(),
    base_size=st.integers(min_value=8, max_value=64),
    extent_seed=st.integers(min_value=0, max_value=255),
    start_seed=st.integers(min_value=0, max_value=255),
    index_seed=st.integers(min_value=0, max_value=255),
    magnitude=st.integers(min_value=1, max_value=96),
)
@settings(max_examples=96, deadline=None)
def test_profile_axis_window_shift_preserves_canonical_axis_ownership(
    role,
    sign,
    cropped,
    base_size,
    extent_seed,
    start_seed,
    index_seed,
    magnitude,
):
    """Display roles must shift their canonical axis and no neighbouring one."""

    from arrayscope.core.view_state import ViewState
    from arrayscope.tools.profile_montage_workflow import _shift_profile_axis_window

    shape = (base_size, base_size + 7, base_size + 13, base_size + 19)
    axis_by_role = {
        "display-y": 1,
        "display-x": 0,
        "montage": 2,
        "slice": 3,
    }
    axis = axis_by_role[role]
    axis_size = shape[axis]
    extent = 1 + extent_seed % max(1, axis_size - 1)
    start = start_seed % (axis_size - extent + 1)
    indices = tuple(range(start, start + extent))
    index = index_seed % axis_size

    # Reversed image axes make display role deliberately differ from
    # canonical axis order, the distinction this property protects.
    state = ViewState.from_shape(shape).with_image_axes(1, 0)
    montage_size = shape[2]
    montage_indices = (
        indices
        if role == "montage" and cropped
        else (None if role == "montage" else tuple(range(2, montage_size - 2)))
    )
    state = state.with_montage_axis(
        2,
        columns=4,
        indices=montage_indices,
        text=None if montage_indices is None else "selection",
    )
    if role != "montage":
        if cropped:
            state = state.with_axis_range(axis, indices=indices, text="selection")
        else:
            state = state.with_slice(axis, index)

    delta = int(sign) * int(magnitude)
    shifted = _shift_profile_axis_window(state, axis, delta)

    if role == "montage":
        original = (
            tuple(range(axis_size))
            if state.montage_indices is None
            else tuple(state.montage_indices)
        )
        expected_start = max(0, min(original[0] + delta, axis_size - len(original)))
        expected = tuple(range(expected_start, expected_start + len(original)))
        assert shifted.montage_indices == (None if len(expected) == axis_size else expected)
        assert shifted.axis_range_indices == state.axis_range_indices
        assert shifted.slice_indices == state.slice_indices
    elif cropped:
        expected_start = max(0, min(start + delta, axis_size - extent))
        assert shifted.axis_range_indices[axis] == tuple(
            range(expected_start, expected_start + extent)
        )
        assert shifted.slice_indices == state.slice_indices
        assert shifted.montage_indices == state.montage_indices
    else:
        assert shifted.slice_indices[axis] == max(0, min(index + delta, axis_size - 1))
        assert shifted.axis_range_indices == state.axis_range_indices
        assert shifted.montage_indices == state.montage_indices

    for other_axis in range(state.ndim):
        if other_axis == axis:
            continue
        assert shifted.axis_range_indices[other_axis] == state.axis_range_indices[other_axis]
        assert shifted.slice_indices[other_axis] == state.slice_indices[other_axis]
    assert shifted.image_axes == (1, 0)
    assert shifted.shape == shape


def test_profile_parser_unknown_stage_is_rejected():
    from arrayscope.tools.profile_montage_workflow import _resolve_profile_stages

    with pytest.raises(ValueError, match="unknown montage workflow stage"):
        _resolve_profile_stages(include_stages=("not-a-phase",))


def test_profile_suite_commands_preserve_stage_filter_flags(tmp_path):
    from arrayscope.tools.profile_montage_workflow import profiler_suite_commands

    commands = profiler_suite_commands(
        (
            "--backend",
            "wgpu",
            "--profile-suite",
            str(tmp_path),
            "--stages",
            "raw_full_tiled_montage,montage_zoompan_fft",
            "--skip-stages",
            "montage_scroll_scalar",
        ),
        tmp_path,
    )
    for item in commands:
        split = shlex.split(item["command"])
        assert "--stages" in split
        assert "raw_full_tiled_montage,montage_zoompan_fft" in split
        assert "--skip-stages" in split
        assert "montage_scroll_scalar" in split


def test_profile_parser_default_scroll_window_and_custom_value():
    from arrayscope.tools.profile_montage_workflow import DEFAULT_SESSION_FIXTURE, _build_parser

    parser = _build_parser()
    default_args = parser.parse_args(["--backend", "wgpu"])
    custom_args = parser.parse_args(
        [
            "--backend",
            "wgpu",
            "--scroll-max-tiles",
            "84",
            "--verbose-tile-trace",
            "--disable-coarse-rung",
        ]
    )
    wgpu_args = parser.parse_args(["--backend", "wgpu"])
    default_backend_args = parser.parse_args([])

    assert default_args.scroll_max_tiles == 60
    assert custom_args.scroll_max_tiles == 84
    assert default_args.verbose_tile_trace is False
    assert custom_args.verbose_tile_trace is True
    assert default_args.enable_coarse_rung is False
    assert default_args.disable_coarse_rung is False
    assert custom_args.disable_coarse_rung is True
    assert default_args.physical_sample_seed is None
    assert wgpu_args.backend == "wgpu"
    assert default_backend_args.backend == "all"
    assert Path(default_args.session_fixture) == DEFAULT_SESSION_FIXTURE


def test_profile_main_default_dispatches_every_stage_to_both_backends(monkeypatch):
    import arrayscope.tools.profile_montage_workflow as workflow

    calls = []

    def fake_run_profile_montage_workflow(**kwargs):
        calls.append((kwargs["backend"], tuple(kwargs["stages"])))
        return ()

    monkeypatch.setattr(
        workflow,
        "run_profile_montage_workflow",
        fake_run_profile_montage_workflow,
    )
    monkeypatch.setattr(workflow, "_workflow_timing_summary", lambda records: "")

    assert workflow.main(("--session-fixture", "")) == 0
    assert calls == [
        (backend, workflow.PROFILE_MONTAGE_STAGES) for backend in workflow.PROFILE_DEFAULT_BACKENDS
    ]


def test_profile_session_fixture_is_a_portable_production_session():
    from arrayscope.core.view_session import loads_session
    from arrayscope.tools.profile_montage_workflow import DEFAULT_SESSION_FIXTURE

    session = loads_session(DEFAULT_SESSION_FIXTURE.read_text(encoding="utf-8"), (336, 336, 272))

    assert session.metadata == {}
    assert session.viewport is not None
    assert session.viewport.viewport_shape == (739, 1247)
    assert session.viewport.viewport_shape[1] / session.viewport.viewport_shape[0] > 1.6
    assert session.viewport.montage_columns == 8
    assert session.recipe.view_state.montage_indices == tuple(range(106, 166))
    assert len(session.rois) == 3


def test_profile_suite_commands_preserve_scroll_max_tiles(tmp_path):
    from arrayscope.tools.profile_montage_workflow import profiler_suite_commands

    commands = profiler_suite_commands(
        (
            "--backend",
            "wgpu",
            "--profile-suite",
            str(tmp_path),
            "--scroll-max-tiles",
            "84",
        ),
        tmp_path,
    )
    for item in commands:
        split = shlex.split(item["command"])
        assert "--scroll-max-tiles" in split
        assert "84" in split


def test_profile_suite_splits_attribution_artifacts_for_all_backends(tmp_path):
    from arrayscope.tools.profile_montage_workflow import profiler_suite_commands

    commands = profiler_suite_commands(
        ("--backend", "all", "--profile-suite", str(tmp_path), "--include-cprofile"), tmp_path
    )

    by_step = {item["step_id"]: item for item in commands}
    for backend in ("wgpu", "pyqtgraph"):
        assert by_step[f"cprofile:{backend}"]["backend"] == backend
        assert by_step[f"py-spy-raw-low-impact:{backend}"]["backend"] == backend
        assert by_step[f"py-spy-raw-full:{backend}"]["backend"] == backend
        assert by_step[f"perf-record:{backend}"]["backend"] == backend
        assert f"--backend {backend}" in by_step[f"py-spy-raw-low-impact:{backend}"]["command"]
        assert f".{backend}." in by_step[f"py-spy-raw-full:{backend}"]["jsonl"]
    assert by_step["plain"]["backend"] == "all"


def test_profile_suite_omitted_backend_uses_default_backend_matrix(tmp_path):
    from arrayscope.tools.profile_montage_workflow import (
        PROFILE_DEFAULT_BACKENDS,
        _suite_profiler_backends,
        profiler_suite_commands,
    )

    assert (
        _suite_profiler_backends(())
        == PROFILE_DEFAULT_BACKENDS
        == (
            "wgpu",
            "pyqtgraph",
        )
    )
    commands = profiler_suite_commands(("--profile-suite", str(tmp_path)), tmp_path)
    attribution_backends = {
        item["backend"] for item in commands if item["profiler_type"] != "plain"
    }
    assert attribution_backends == set(PROFILE_DEFAULT_BACKENDS)


def test_profile_suite_can_opt_into_native_py_spy_without_passing_suite_flag_to_child(tmp_path):
    from arrayscope.tools.profile_montage_workflow import profiler_suite_commands

    commands = profiler_suite_commands(
        ("--backend", "wgpu", "--profile-suite", str(tmp_path), "--py-spy-native"),
        tmp_path,
    )
    by_type = {item["profiler_type"]: item for item in commands}

    assert "py-spy-raw-low-impact-native" in by_type
    assert "py-spy-raw-full-native" in by_type
    assert "--native" in by_type["py-spy-raw-low-impact-native"]["command"]
    assert "--native" in by_type["py-spy-raw-full-native"]["command"]
    assert "--profile-suite" not in by_type["plain"]["command"]


def test_profile_workflow_preserves_theme_while_forcing_backend_and_resident_policy():
    from arrayscope.app.settings_state import (
        AppSettingsState,
        ImageRenderingBackendChoice,
        MontageQualityPolicyChoice,
    )
    from arrayscope.app.theme import ThemeChoice
    from arrayscope.tools.profile_montage_workflow import _replace_settings

    pyqtgraph = _replace_settings(
        AppSettingsState(),
        backend="pyqtgraph",
        image_choice=ImageRenderingBackendChoice,
    )
    wgpu = _replace_settings(
        AppSettingsState(),
        backend="wgpu",
        image_choice=ImageRenderingBackendChoice,
    )

    assert pyqtgraph.theme == ThemeChoice.SYSTEM
    assert pyqtgraph.montage_quality_policy == MontageQualityPolicyChoice.RESIDENT
    assert wgpu.theme == ThemeChoice.SYSTEM
    assert wgpu.image_rendering_backend == ImageRenderingBackendChoice.WGPU
    assert wgpu.montage_quality_policy == MontageQualityPolicyChoice.RESIDENT


def test_profile_transform_pipeline_uses_fft_shift_ifft_sequence():
    from arrayscope.operations.pipeline import CenteredFFT, CenteredIFFT, FFTShift
    from arrayscope.tools.profile_montage_workflow import _profile_transform_operations

    operations = _profile_transform_operations(
        2,
        centered_fft=CenteredFFT,
        fftshift=FFTShift,
        centered_ifft=CenteredIFFT,
    )

    assert tuple(type(operation) for operation in operations) == (
        CenteredFFT,
        FFTShift,
        CenteredIFFT,
    )
    assert tuple(operation.axis for operation in operations) == (2, 2, 2)


def test_profile_fit_stretch_pulse_uses_window_fit_command_and_reports_cost():
    from arrayscope.tools.profile_montage_workflow import _pulse_fit_stretch

    calls = []
    retarget_calls = []
    metrics = {}
    win = SimpleNamespace(
        fit_image_to_view=lambda enabled: calls.append(bool(enabled)),
        retarget_montage_viewport=lambda: retarget_calls.append(True),
    )

    assert _pulse_fit_stretch(win, metrics=metrics) is True

    assert calls == [True, False]
    assert retarget_calls == [True]
    assert metrics["fit_stretch_total_ms"] >= 0.0
    assert metrics["fit_stretch_enable_call_ms"] >= 0.0
    assert metrics["fit_stretch_disable_call_ms"] >= 0.0
    assert metrics["fit_stretch_retarget_call_ms"] >= 0.0
    assert metrics["fit_stretch_retarget_delivery_ms"] >= 0.0


def test_profile_montage_build_holds_intermediate_fit_range_signal():
    from arrayscope.tools.profile_montage_workflow import _hold_fit_for_montage_build

    class View:
        def __init__(self):
            self.blocked = False
            self.transitions = []

        def blockSignals(self, blocked):
            previous = self.blocked
            self.blocked = bool(blocked)
            self.transitions.append(self.blocked)
            return previous

    view = View()
    observed = []
    win = SimpleNamespace(
        img_view=SimpleNamespace(getView=lambda: view),
        fit_image_to_view=lambda enabled: observed.append((bool(enabled), view.blocked)),
    )
    metrics = {}

    assert _hold_fit_for_montage_build(win, metrics=metrics) is True
    assert observed == [(True, True)]
    assert view.transitions == [True, False]
    assert view.blocked is False
    assert metrics["fit_stretch_compound_signal_hold"] is True


def _passing_r8_phase_record(*, backend="wgpu"):
    evidence_quality = 1 if backend == "wgpu" else 2
    record = {
        "phase": "raw_full_tiled_montage",
        "backend": backend,
        "profiler_type": "plain",
        "pacing_evidence": True,
        "complete": True,
        "requested_grid_fully_visible": True,
        "requested_tile_count": 2,
        "active_planned_tile_count": 2,
        "active_presented_tile_count": 2,
        "presentation_settled": True,
        "stale_level_tiles": 0,
        "pending_level_tiles": 0,
        "display_levels": [-2.0, 8.0],
        "levels_look_default": False,
        "histogram_data_bounds": [-2.0, 8.0],
        "histogram_empty": False,
        "first_visible_display_levels": [-1.0, 7.0],
        "first_visible_levels_default": False,
        "first_visible_histogram_data_bounds": [-1.0, 7.0],
        "first_visible_histogram_empty": False,
        "first_visible_level_evidence_quality": evidence_quality,
        "window_level_flicker_free": True,
        "histogram_emptied_after_successor_visible": False,
        "levels_defaulted_after_successor_visible": False,
        "level_transient_span_dip_ratio": 1.0,
        "level_center_excursion_fraction": 0.0,
        "level_source_count_regressed": False,
        "histogram_timeline_transition_count": 2,
        "histogram_timeline_truncated": False,
        "presentation_continuity_ok": True,
        "presentation_continuity_expected": True,
        "presentation_blackout_observed": False,
        "presentation_predecessor_tile_count": 100,
        "presentation_minimum_retained_tile_count": 100,
        "presentation_extent_changed_before_commit": False,
        "session_viewport_shape_matches": True,
        "viewport_shape": [753, 1245],
        "session_viewport_shape_target": [753, 1245],
        "window_size": [1400, 948],
        "session_window_size_target": [1400, 940],
        "session_window_size_chrome_delta": [0, 8],
        "session_axis_orientation_matches": True,
        "image_axes": [0, 1],
        "axis_flipped": [False, True, False],
        "session_image_axes_target": [0, 1],
        "session_axis_flipped_target": [False, True, False],
        "montage_repeated_expensive_stage_per_tile": False,
        "fit_stretch_pulsed": True,
        "fit_disable_viewport_mode": "auto_untouched",
        "fit_disable_view_range": [[0.0, 1.0], [0.0, 1.0]],
        "grid_kind": "full",
        "grid_tile_count": 2,
        "full_tile_count": 2,
        "tile_cap_applied": False,
        "phase_recent_ui_work_observations": [{"elapsed_ms": 7.0}],
        "phase_recent_ui_work_observations_truncated": False,
        "action_render_call_ms": 4.0,
        "event_loop_max_gap_ms": 12.0,
        "physical_draw_after_complete_ms": 20.0,
        "waited_for_pyqtgraph_draw_after_complete": backend == "pyqtgraph",
        "pyqtgraph_draw_pending_after_complete": False,
        "coarse_target_trace_window_complete": True,
        "coarse_rung_enabled": True,
        "coarse_target_preview_required": True,
        "coarse_target_preview_exemption": None,
        "coarse_target_order_applicable": True,
        "coarse_target_order_status": "ordered",
        "coarse_target_ack_ordered": True,
        "coarse_target_execution_ordered": True,
        "coarse_target_t1_ms": 800.0,
        "coarse_target_t2_ms": 1500.0,
        "coarse_target_first_target_ack_ms": 1000.0,
        "coarse_target_last_preview_task_finish_ms": 700.0,
        "coarse_target_first_target_task_start_ms": 900.0,
        "coarse_target_preview_ack_tiles": 2,
        "coarse_target_target_ack_tiles": 2,
        "coarse_target_preview_task_finishes": 2,
        "coarse_target_target_task_starts": 2,
    }
    from arrayscope.tools.profile_montage_workflow import (
        _progressive_invariant_certification,
    )

    events = []
    task_seq = 1
    for _purpose, rung, level in (("preview", 0, 4), ("target", 2, 2)):
        for tile in (0, 1):
            events.extend(
                (
                    {
                        "kind": "kernel_start",
                        "session_id": 7,
                        "scheduling_generation": 3,
                        "task_seq": task_seq,
                        "tile_number": tile,
                        "rung": rung,
                        "level": level,
                    },
                    {
                        "kind": "kernel_finish",
                        "session_id": 7,
                        "scheduling_generation": 3,
                        "task_seq": task_seq,
                        "tile_number": tile,
                        "rung": rung,
                        "level": level,
                        "outcome": "completed",
                    },
                )
            )
            task_seq += 1

    def payload(tile, *, level, quality, bounds):
        return {
            "tile": tile,
            "acknowledged": True,
            "level": level,
            "quality": quality,
            "source_id": f"tile-{tile}",
            "value_bounds": bounds,
            "baked_levels": (-2.0, 8.0) if backend == "pyqtgraph" else None,
        }

    commits = (
        {
            "session_id": 7,
            "presented_tiles": (0, 1),
            "delta_qualities": ((0, "preview", 4), (1, "preview", 4)),
            "presented_payloads": (
                payload(0, level=4, quality="preview", bounds=(-1.0, 7.0)),
                payload(1, level=4, quality="preview", bounds=(-1.0, 7.0)),
            ),
            "physical_levels": (-2.0, 8.0),
            "elapsed_ms": 7.0,
            "max_upserts": 2,
            "uploads_by_level": ((2, 0), (4, 2)),
            "target_settled_after": False,
        },
        {
            "session_id": 7,
            "presented_tiles": (0, 1),
            "delta_qualities": ((0, "exact", 2), (1, "exact", 2)),
            "presented_payloads": (
                payload(0, level=2, quality="exact", bounds=(-2.0, 8.0)),
                payload(1, level=2, quality="exact", bounds=(-2.0, 8.0)),
            ),
            "physical_levels": (-2.0, 8.0),
            "elapsed_ms": 8.0,
            "max_upserts": 2,
            "uploads_by_level": ((2, 2), (4, 2)),
            "target_settled_after": True,
        },
    )
    evidence = {
        "events": tuple(events),
        "rounds": (
            {
                "round_id": "round-7",
                "session_id": 7,
                "generation": 3,
                "started_ns": 1,
                "settled_ns": 2,
                "required_tiles": (0, 1),
                "target_floor": 2,
                "preview_floor": 4,
                "baseline": (),
                "resident_query_available": True,
                "uploads_by_level_available": backend == "wgpu",
                "uploads_by_level_start": {2: 0, 4: 0},
                "settled_at_start": False,
                "commits": commits,
            },
        ),
        "event_truncated": False,
        "presented_refs_truncated": False,
    }
    record.update(_progressive_invariant_certification(evidence, record))
    return record


def test_r8_certification_passes_complete_semantic_and_responsive_phase():
    from arrayscope.tools.profile_montage_workflow import _r8_certification

    for backend in ("wgpu", "pyqtgraph"):
        result = _r8_certification(_passing_r8_phase_record(backend=backend))

        assert result["r8_gate_applicable"] is True
        assert result["r8_performance_evidence"] is True
        assert result["r8_gate_passed"] is True
        assert result["r8_gate_failures"] == []


def _passing_contract_evidence(*, backend="wgpu"):
    baked = (-2.0, 8.0) if backend == "pyqtgraph" else None
    events = (
        {
            "kind": "kernel_start",
            "session_id": 11,
            "scheduling_generation": 4,
            "task_seq": 1,
            "tile_number": 0,
            "rung": 0,
            "level": 4,
        },
        {
            "kind": "kernel_finish",
            "session_id": 11,
            "scheduling_generation": 4,
            "task_seq": 1,
            "tile_number": 0,
            "rung": 0,
            "level": 4,
            "outcome": "completed",
        },
        {
            "kind": "kernel_start",
            "session_id": 11,
            "scheduling_generation": 4,
            "task_seq": 2,
            "tile_number": 0,
            "rung": 2,
            "level": 2,
        },
        {
            "kind": "kernel_finish",
            "session_id": 11,
            "scheduling_generation": 4,
            "task_seq": 2,
            "tile_number": 0,
            "rung": 2,
            "level": 2,
            "outcome": "completed",
        },
    )
    commits = (
        {
            "presented_tiles": (0,),
            "delta_qualities": ((0, "preview", 4),),
            "presented_payloads": (
                {
                    "tile": 0,
                    "acknowledged": True,
                    "level": 4,
                    "quality": "preview",
                    "source_id": "source-0",
                    "value_bounds": (-1.0, 7.0),
                    "baked_levels": baked,
                },
            ),
            "physical_levels": (-2.0, 8.0),
            "elapsed_ms": 8.0,
            "max_upserts": 1,
            "uploads_by_level": ((2, 0), (4, 1)),
        },
        {
            "presented_tiles": (0,),
            "delta_qualities": ((0, "exact", 2),),
            "presented_payloads": (
                {
                    "tile": 0,
                    "acknowledged": True,
                    "level": 2,
                    "quality": "exact",
                    "source_id": "source-0",
                    "value_bounds": (-2.0, 8.0),
                    "baked_levels": baked,
                },
            ),
            "physical_levels": (-2.0, 8.0),
            "elapsed_ms": 9.0,
            "max_upserts": 1,
            "uploads_by_level": ((2, 1), (4, 1)),
        },
    )
    return {
        "events": events,
        "rounds": (
            {
                "round_id": "round-11",
                "session_id": 11,
                "generation": 4,
                "started_ns": 10,
                "settled_ns": 100,
                "required_tiles": (0,),
                "target_floor": 2,
                "preview_floor": 4,
                "baseline": (),
                "resident_query_available": True,
                "uploads_by_level_available": backend == "wgpu",
                "uploads_by_level_start": {2: 0, 4: 0},
                "settled_at_start": False,
                "commits": commits,
            },
        ),
        "event_truncated": False,
        "presented_refs_truncated": False,
    }


def _contract_verdict(evidence, *, backend="wgpu", **record_updates):
    from arrayscope.tools.profile_montage_workflow import (
        _progressive_invariant_certification,
    )

    record = {
        "phase": "raw_full_tiled_montage",
        "backend": backend,
        "coarse_rung_enabled": True,
        "phase_recent_ui_work_observations": ({"elapsed_ms": 5.0},),
        "phase_recent_ui_work_observations_truncated": False,
        "action_render_call_ms": 4.0,
    }
    record.update(record_updates)
    return _progressive_invariant_certification(evidence, record)


def _failed_invariant_gates(result):
    return {failure["gate"] for failure in result["invariant_gate_failures"]}


def test_progressive_invariant_fixture_carries_a_real_contract_proof():
    result = _contract_verdict(_passing_contract_evidence())

    assert result["invariant_gate_passed"] is True
    assert result["invariant_gate_round_count"] == 1
    assert result["invariant_gate_rule_failures"] == {
        "R1": 0,
        "R2": 0,
        "R2b": 0,
        "R3": 0,
        "R4": 0,
        "R5": 0,
        "R7": 0,
    }
    assert set(result["invariant_gate_unverifiable_rules"]) == {"R6"}


def test_progressive_invariant_gate_fails_without_authoritative_round_identity():
    evidence = deepcopy(_passing_contract_evidence())
    evidence["rounds"][0]["round_id"] = None

    result = _contract_verdict(evidence)

    assert result["invariant_gate_passed"] is False
    assert "authoritative_round_identity_present" in _failed_invariant_gates(result)


def test_progressive_invariant_gate_proves_round_floors_across_pipeline_plans():
    evidence = deepcopy(_passing_contract_evidence())
    evidence["events"] = (
        *evidence["events"],
        {
            "kind": "pipeline_plan",
            "render_round_id": "round-11",
            "round_preview_level": 5,
            "round_target_level": 2,
        },
    )

    result = _contract_verdict(evidence)

    assert result["invariant_gate_passed"] is False
    assert "one_floor_pair_per_round" in _failed_invariant_gates(result)


def test_r8_certification_fails_closed_without_in_process_invariant_verdict():
    from arrayscope.tools.profile_montage_workflow import _r8_certification

    record = _passing_r8_phase_record()
    for key in tuple(record):
        if key.startswith("invariant_gate_"):
            record.pop(key)

    result = _r8_certification(record)

    assert result["r8_gate_passed"] is False
    assert "progressive_render_invariants" in {
        failure["gate"] for failure in result["r8_gate_failures"]
    }


def test_workflow_exit_code_requires_invariant_verdict_without_artifact():
    from arrayscope.tools.profile_montage_workflow import _workflow_exit_code

    assert (
        _workflow_exit_code(
            (
                {
                    "r8_gate_applicable": True,
                    "r8_gate_passed": True,
                    "invariant_gate_applicable": True,
                    "invariant_gate_passed": True,
                },
            )
        )
        == 0
    )
    assert (
        _workflow_exit_code(
            (
                {
                    "r8_gate_applicable": True,
                    "r8_gate_passed": True,
                },
            )
        )
        == 1
    )
    assert (
        _workflow_exit_code(
            (
                {
                    "r8_gate_applicable": True,
                    "r8_gate_passed": True,
                    "invariant_gate_applicable": True,
                    "invariant_gate_passed": False,
                },
            )
        )
        == 1
    )


def test_progressive_invariant_gate_rejects_duplicate_target_production():
    evidence = deepcopy(_passing_contract_evidence())
    events = list(evidence["events"])
    events.extend(
        (
            {
                "kind": "kernel_start",
                "session_id": 11,
                "scheduling_generation": 4,
                "task_seq": 3,
                "tile_number": 0,
                "rung": 2,
                "level": 2,
            },
            {
                "kind": "kernel_finish",
                "session_id": 11,
                "scheduling_generation": 4,
                "task_seq": 3,
                "tile_number": 0,
                "rung": 2,
                "level": 2,
                "outcome": "completed",
            },
        )
    )
    evidence["events"] = tuple(events)

    result = _contract_verdict(evidence)

    assert "at_most_one_production_per_pass_per_tile" in _failed_invariant_gates(result)


def test_progressive_invariant_gate_rejects_wgpu_upload_outside_round_floors():
    evidence = deepcopy(_passing_contract_evidence())
    evidence["rounds"][0]["commits"][0]["uploads_by_level"] = (
        (0, 3),
        (2, 0),
        (4, 1),
    )

    result = _contract_verdict(evidence)

    assert "physical_uploads_stay_on_round_floors" in _failed_invariant_gates(result)


def test_progressive_invariant_gate_rejects_reproduction_above_reuse_floor():
    evidence = deepcopy(_passing_contract_evidence())
    evidence["rounds"][0]["baseline"] = (
        {
            "tile": 0,
            "level": 2,
            "quality": "exact",
            "resident": True,
            "acknowledged": True,
        },
    )

    result = _contract_verdict(evidence)

    assert "satisfied_floor_is_not_reproduced" in _failed_invariant_gates(result)


def test_progressive_invariant_gate_rejects_clipped_presented_tile():
    evidence = deepcopy(_passing_contract_evidence())
    evidence["rounds"][0]["commits"][0]["physical_levels"] = (0.0, 1.0)

    result = _contract_verdict(evidence)

    assert "commit_levels_contain_presented_tile" in _failed_invariant_gates(result)


def test_r3_rejects_only_stale_rebind_evidence_and_uses_current_plane():
    """A re-sliced CPU plane remains current when carried stats do not."""

    import numpy as np

    from arrayscope.display.model.montage_levels import TileLevelStats
    from arrayscope.tools.profile_montage_workflow import _presented_payload_value_bounds

    payload = SimpleNamespace(
        level_evidence_window_stale=True,
        level_stats=TileLevelStats(
            source_index=7,
            bounds=(0.0, 10.0),
            sample=np.asarray([0.0, 10.0], dtype=np.float32),
        ),
        level_data=np.asarray([0.0, 10.0], dtype=np.float32),
        semantic_data=np.asarray([[3.0, 4.0]], dtype=np.float32),
        semantic_histogram_data=None,
        histogram_data=None,
        image=np.asarray([[3.0, 4.0]], dtype=np.float32),
        page_backing=None,
    )
    assert _presented_payload_value_bounds(payload) == (
        (3.0, 4.0),
        "stale-stats-rejected-image-used",
    )

    page_backed = SimpleNamespace(
        level_evidence_window_stale=True,
        level_stats=payload.level_stats,
        level_data=payload.level_data,
        semantic_data=np.asarray([[3.0, 4.0]], dtype=np.float32),
        semantic_histogram_data=np.asarray([3.0, 4.0], dtype=np.float32),
        histogram_data=np.asarray([3.0, 4.0], dtype=np.float32),
        image=np.asarray([[3.0, 4.0]], dtype=np.float32),
        page_backing=object(),
        native_residency_data=None,
        shader_mapping=None,
    )
    assert _presented_payload_value_bounds(page_backed) == (
        None,
        "page-backed-rebind-no-current-plane",
    )

    page_backed.native_residency_data = np.asarray(
        [[-5.0, 4.0], [7.0, 20.0]],
        dtype=np.float32,
    )
    assert _presented_payload_value_bounds(page_backed) == (
        (-5.0, 20.0),
        "page-backed-rebind-full-plane-superset",
    )


@pytest.mark.parametrize(
    ("raw", "levels", "mapped_bounds", "raw_would_clip"),
    [
        ([10.0, 100.0], (0.99, 2.01), (1.0, 2.0), True),
        ([0.01, 0.1], (0.0, 1.0), (-2.0, -1.0), False),
    ],
)
def test_r3_oracle_compares_nonlinear_payloads_in_display_space(
    raw,
    levels,
    mapped_bounds,
    raw_would_clip,
):
    """Raw values can produce both false-red and false-green R3 verdicts."""

    import numpy as np

    from arrayscope.display.shader_mapping import (
        ShaderComponent,
        ShaderMapping,
        ShaderScale,
    )
    from arrayscope.tools.profile_montage_workflow import _presented_payload_value_bounds

    raw_plane = np.asarray(raw, dtype=np.float32)
    payload = SimpleNamespace(
        level_evidence_window_stale=True,
        level_stats=None,
        level_data=None,
        semantic_histogram_data=None,
        histogram_data=None,
        semantic_data=raw_plane,
        image=raw_plane,
        shader_mapping=ShaderMapping(
            component=ShaderComponent.REAL,
            scale=ShaderScale.LOG,
        ),
        page_backing=None,
        # The oracle must derive evidence independently of the clamp's claim.
        rebind_current_value_bounds=(-1000.0, 1000.0),
    )

    bounds, source = _presented_payload_value_bounds(payload)
    assert bounds == pytest.approx(mapped_bounds)
    assert source == "stale-stats-rejected-semantic_data-mapped-used"
    raw_bounds = (float(np.min(raw_plane)), float(np.max(raw_plane)))
    assert (raw_bounds[0] < levels[0] or raw_bounds[1] > levels[1]) is raw_would_clip
    assert (bounds[0] < levels[0] or bounds[1] > levels[1]) is (not raw_would_clip)


@pytest.mark.parametrize(
    ("semantic", "image", "expected", "expected_source"),
    [
        (
            [100.0, 200.0],
            [3.0 + 4.0j, 5.0 + 12.0j],
            [5.0, 13.0],
            "image-magnitude",
        ),
        ([100.0, 200.0], [3.0, 7.0], [3.0, 7.0], "image"),
    ],
)
def test_payload_value_interpretation_matches_at_all_three_call_sites(
    semantic,
    image,
    expected,
    expected_source,
):
    """Complex and unmapped payloads use one explicit fallback rule."""

    import numpy as np

    from arrayscope.render.effects import montage_refined_level_values
    from arrayscope.tools.profile_montage_workflow import _presented_payload_value_bounds
    from arrayscope.window.frame_session import _rebind_current_plane_value_bounds

    semantic_plane = np.asarray(semantic)
    image_plane = np.asarray(image)
    payload = SimpleNamespace(
        level_evidence_window_stale=False,
        level_stats=None,
        level_data=None,
        semantic_histogram_data=None,
        histogram_data=None,
        semantic_data=semantic_plane,
        image=image_plane,
        shader_mapping=None,
        page_backing=None,
    )

    refined = montage_refined_level_values(payload)
    rebind = _rebind_current_plane_value_bounds(
        payload,
        {
            "semantic_data": semantic_plane,
            "image": image_plane,
        },
    )
    oracle, source = _presented_payload_value_bounds(payload)

    np.testing.assert_allclose(refined, np.asarray(expected))
    assert rebind == pytest.approx((min(expected), max(expected)))
    assert oracle == pytest.approx((min(expected), max(expected)))
    assert source == expected_source


def test_progressive_invariant_gate_requires_pyqtgraph_tile_value_and_bake_evidence():
    evidence = deepcopy(_passing_contract_evidence(backend="pyqtgraph"))
    payload = evidence["rounds"][0]["commits"][0]["presented_payloads"][0]
    payload["value_bounds"] = None
    payload["baked_levels"] = None

    result = _contract_verdict(evidence, backend="pyqtgraph")

    assert {
        "presented_tile_value_bounds_recorded",
        "pyqtgraph_baked_levels_recorded",
    }.issubset(_failed_invariant_gates(result))


def test_progressive_invariant_gate_rejects_missing_complex_preview_without_exemption():
    evidence = deepcopy(_passing_contract_evidence(backend="pyqtgraph"))
    evidence["events"] = tuple(
        event for event in evidence["events"] if int(event.get("rung", -1)) != 0
    )
    evidence["rounds"][0]["commits"] = (evidence["rounds"][0]["commits"][1],)

    result = _contract_verdict(
        evidence,
        backend="pyqtgraph",
        phase="fft_full_tiled_montage",
    )

    assert "preview_pass_exists_for_backend_dtype" in _failed_invariant_gates(result)


def test_progressive_invariant_gate_rejects_unbounded_over_budget_commit():
    evidence = deepcopy(_passing_contract_evidence())
    evidence["rounds"][0]["commits"][0].update(elapsed_ms=55.0, max_upserts=0)

    result = _contract_verdict(evidence)

    assert {
        "commit_chunk_governed_or_sub_budget",
        "commit_callback_below_50ms",
    }.issubset(_failed_invariant_gates(result))


def test_progressive_invariant_gate_rejects_speculative_residency_inside_fill():
    evidence = deepcopy(_passing_contract_evidence())
    evidence["events"] = (
        *evidence["events"],
        {
            "kind": "kernel_submit",
            "session_id": 11,
            "lane": "Lane.SPECULATIVE_RESIDENCY",
            "task_seq": 99,
            "observed_ns": 50,
        },
    )

    result = _contract_verdict(evidence)

    assert "no_speculative_residency_during_fill" in _failed_invariant_gates(result)


def test_coarse_target_trace_metrics_rejects_ack_and_worker_overlap():
    from arrayscope.tools.profile_montage_workflow import _coarse_target_trace_metrics

    events = (
        {
            "kind": "input",
            "action": "phase_start",
            "phase": "raw_full_tiled_montage",
            "backend": "wgpu",
            "sequence": 10,
            "ts_ns": 1_000_000_000,
        },
        {
            "kind": "scheduling_phase",
            "event": "scope_started",
            "generation": 1,
            "required_tiles": 1,
            "required_tile_numbers": (0,),
            "sequence": 11,
            "ts_ns": 1_005_000_000,
        },
        {
            "kind": "kernel_start",
            "rung": 0,
            "task_seq": 41,
            "scheduling_generation": 1,
            "sequence": 12,
            "ts_ns": 1_010_000_000,
        },
        {"kind": "kernel_start", "rung": 2, "sequence": 13, "ts_ns": 1_020_000_000},
        {
            "kind": "backend_ack",
            "accepted": True,
            "quality": "exact",
            "tile": 0,
            "sequence": 14,
            "ts_ns": 1_030_000_000,
        },
        {
            "kind": "kernel_finish",
            "rung": 0,
            "task_seq": 41,
            "scheduling_generation": 1,
            "sequence": 15,
            "ts_ns": 1_040_000_000,
        },
        {
            "kind": "backend_ack",
            "accepted": True,
            "quality": "preview",
            "tile": 0,
            "sequence": 16,
            "ts_ns": 1_050_000_000,
        },
        {
            "kind": "input",
            "action": "phase_complete",
            "phase": "raw_full_tiled_montage",
            "backend": "wgpu",
            "sequence": 17,
            "ts_ns": 1_060_000_000,
        },
    )

    result = _coarse_target_trace_metrics(
        events,
        phase="raw_full_tiled_montage",
        backend="wgpu",
        phase_start_sequence=10,
        requested_tiles=1,
    )

    assert result["coarse_target_order_status"] == "overlap"
    assert result["coarse_target_ack_ordered"] is False
    assert result["coarse_target_execution_ordered"] is False
    assert result["coarse_target_t1_ms"] == 50.0
    assert result["coarse_target_t2_ms"] == 30.0


def test_coarse_target_trace_metrics_withholds_t1_t2_for_partial_coverage():
    from arrayscope.tools.profile_montage_workflow import _coarse_target_trace_metrics

    events = (
        {
            "kind": "input",
            "action": "phase_start",
            "phase": "raw_full_tiled_montage",
            "backend": "pyqtgraph",
            "sequence": 20,
            "ts_ns": 2_000_000_000,
        },
        {
            "kind": "scheduling_phase",
            "event": "scope_started",
            "generation": 1,
            "required_tiles": 2,
            "required_tile_numbers": (0, 1),
            "sequence": 21,
            "ts_ns": 2_005_000_000,
        },
        {
            "kind": "backend_ack",
            "accepted": True,
            "quality": "preview",
            "tile": 0,
            "sequence": 22,
            "ts_ns": 2_010_000_000,
        },
        {
            "kind": "backend_ack",
            "accepted": True,
            "quality": "exact",
            "tile": 0,
            "sequence": 23,
            "ts_ns": 2_020_000_000,
        },
        {
            "kind": "input",
            "action": "phase_complete",
            "phase": "raw_full_tiled_montage",
            "backend": "pyqtgraph",
            "sequence": 24,
            "ts_ns": 2_030_000_000,
        },
    )

    result = _coarse_target_trace_metrics(
        events,
        phase="raw_full_tiled_montage",
        backend="pyqtgraph",
        phase_start_sequence=20,
        requested_tiles=2,
    )

    assert result["coarse_target_t1_ms"] is None
    assert result["coarse_target_t2_ms"] is None
    assert result["coarse_target_preview_ack_last_ms"] == 10.0
    assert result["coarse_target_target_ack_last_ms"] == 20.0


def test_coarse_target_ack_oracle_does_not_wait_for_target_fill_completion():
    from arrayscope.tools.profile_montage_workflow import _coarse_target_trace_metrics

    events = (
        {
            "kind": "input",
            "action": "phase_start",
            "phase": "fft_full_tiled_montage",
            "backend": "wgpu",
            "sequence": 30,
            "ts_ns": 3_000_000_000,
        },
        {
            "kind": "scheduling_phase",
            "event": "scope_started",
            "generation": 1,
            "required_tiles": 2,
            "required_tile_numbers": (0, 1),
            "sequence": 31,
            "ts_ns": 3_005_000_000,
        },
        {
            "kind": "kernel_start",
            "rung": 0,
            "task_seq": 5,
            "scheduling_generation": 1,
            "sequence": 32,
            "ts_ns": 3_010_000_000,
        },
        {
            "kind": "kernel_finish",
            "rung": 0,
            "task_seq": 5,
            "scheduling_generation": 1,
            "sequence": 33,
            "ts_ns": 3_020_000_000,
        },
        {
            "kind": "backend_ack",
            "accepted": True,
            "quality": "preview",
            "tile": 0,
            "sequence": 34,
            "ts_ns": 3_030_000_000,
        },
        {
            "kind": "backend_ack",
            "accepted": True,
            "quality": "preview",
            "tile": 1,
            "sequence": 35,
            "ts_ns": 3_040_000_000,
        },
        {
            "kind": "kernel_start",
            "rung": 2,
            "sequence": 36,
            "ts_ns": 3_050_000_000,
        },
        {
            "kind": "backend_ack",
            "accepted": True,
            "quality": "exact",
            "tile": 0,
            "sequence": 37,
            "ts_ns": 3_060_000_000,
        },
        {
            "kind": "input",
            "action": "phase_complete",
            "phase": "fft_full_tiled_montage",
            "backend": "wgpu",
            "sequence": 38,
            "ts_ns": 3_070_000_000,
        },
    )

    result = _coarse_target_trace_metrics(
        events,
        phase="fft_full_tiled_montage",
        backend="wgpu",
        phase_start_sequence=30,
        requested_tiles=2,
    )

    assert result["coarse_target_ack_ordered"] is True
    assert result["coarse_target_execution_ordered"] is True
    assert result["coarse_target_order_status"] == "ordered"
    assert result["coarse_target_t1_ms"] == 40.0
    assert result["coarse_target_t2_ms"] is None
    assert result["coarse_target_target_ack_tiles"] == 1


def test_coarse_target_trace_metrics_rejects_target_work_from_predecessor_scope():
    from arrayscope.tools.profile_montage_workflow import _coarse_target_trace_metrics

    events = (
        {
            "kind": "input",
            "action": "phase_start",
            "phase": "raw_full_tiled_montage",
            "backend": "pyqtgraph",
            "sequence": 30,
            "ts_ns": 3_000_000_000,
        },
        {
            "kind": "scheduling_phase",
            "event": "scope_started",
            "generation": 4,
            "required_tiles": 1,
            "required_tile_numbers": (0,),
            "sequence": 31,
            "ts_ns": 3_010_000_000,
        },
        {"kind": "kernel_start", "rung": 2, "sequence": 32, "ts_ns": 3_020_000_000},
        {
            "kind": "scheduling_phase",
            "event": "scope_started",
            "generation": 5,
            "required_tiles": 2,
            "required_tile_numbers": (0, 1),
            "sequence": 33,
            "ts_ns": 3_030_000_000,
        },
        {
            "kind": "kernel_start",
            "rung": 0,
            "task_seq": 99,
            "scheduling_generation": 5,
            "sequence": 34,
            "ts_ns": 3_035_000_000,
        },
        {
            "kind": "kernel_finish",
            "rung": 0,
            "task_seq": 99,
            "scheduling_generation": 5,
            "sequence": 35,
            "ts_ns": 3_040_000_000,
        },
        {
            "kind": "backend_ack",
            "accepted": True,
            "quality": "preview",
            "tile": 0,
            "sequence": 36,
            "ts_ns": 3_050_000_000,
        },
        {
            "kind": "backend_ack",
            "accepted": True,
            "quality": "preview",
            "tile": 1,
            "sequence": 37,
            "ts_ns": 3_060_000_000,
        },
        {"kind": "kernel_start", "rung": 2, "sequence": 38, "ts_ns": 3_070_000_000},
        {
            "kind": "backend_ack",
            "accepted": True,
            "quality": "exact",
            "tile": 0,
            "sequence": 39,
            "ts_ns": 3_080_000_000,
        },
        {
            "kind": "backend_ack",
            "accepted": True,
            "quality": "exact",
            "tile": 1,
            "sequence": 40,
            "ts_ns": 3_090_000_000,
        },
        {
            "kind": "input",
            "action": "phase_complete",
            "phase": "raw_full_tiled_montage",
            "backend": "pyqtgraph",
            "sequence": 41,
            "ts_ns": 3_100_000_000,
        },
    )

    result = _coarse_target_trace_metrics(
        events,
        phase="raw_full_tiled_montage",
        backend="pyqtgraph",
        phase_start_sequence=30,
        requested_tiles=2,
    )

    assert result["coarse_target_order_status"] == "overlap"
    assert result["coarse_target_scheduling_generation"] == 5
    assert result["coarse_target_first_target_task_start_ms"] == 20.0


def test_coarse_target_trace_metrics_does_not_borrow_preview_from_old_scope():
    from arrayscope.tools.profile_montage_workflow import _coarse_target_trace_metrics

    events = (
        {
            "kind": "input",
            "action": "phase_start",
            "phase": "raw_full_tiled_montage",
            "backend": "wgpu",
            "sequence": 10,
            "ts_ns": 1_000_000_000,
        },
        {
            "kind": "scheduling_phase",
            "event": "scope_started",
            "generation": 1,
            "required_tiles": 2,
            "required_tile_numbers": (0, 1),
            "sequence": 11,
            "ts_ns": 1_010_000_000,
        },
        {"kind": "kernel_finish", "rung": 0, "sequence": 12, "ts_ns": 1_020_000_000},
        {
            "kind": "backend_ack",
            "accepted": True,
            "quality": "preview",
            "tile": 0,
            "sequence": 13,
            "ts_ns": 1_030_000_000,
        },
        {
            "kind": "backend_ack",
            "accepted": True,
            "quality": "preview",
            "tile": 1,
            "sequence": 14,
            "ts_ns": 1_040_000_000,
        },
        {
            "kind": "scheduling_phase",
            "event": "scope_started",
            "generation": 2,
            "required_tiles": 2,
            "required_tile_numbers": (0, 1),
            "sequence": 15,
            "ts_ns": 1_050_000_000,
        },
        {
            "kind": "scheduling_phase",
            "event": "coverage_closed",
            "generation": 2,
            "sequence": 16,
            "ts_ns": 1_060_000_000,
        },
        {
            "kind": "kernel_start",
            "rung": 2,
            "scheduling_generation": 2,
            "sequence": 17,
            "ts_ns": 1_070_000_000,
        },
        {
            "kind": "backend_ack",
            "accepted": True,
            "quality": "exact",
            "tile": 0,
            "sequence": 18,
            "ts_ns": 1_080_000_000,
        },
        {
            "kind": "backend_ack",
            "accepted": True,
            "quality": "exact",
            "tile": 1,
            "sequence": 19,
            "ts_ns": 1_090_000_000,
        },
        {
            "kind": "input",
            "action": "phase_complete",
            "phase": "raw_full_tiled_montage",
            "backend": "wgpu",
            "sequence": 20,
            "ts_ns": 1_100_000_000,
        },
    )

    result = _coarse_target_trace_metrics(
        events,
        phase="raw_full_tiled_montage",
        backend="wgpu",
        phase_start_sequence=10,
        requested_tiles=2,
    )

    assert result["coarse_target_scheduling_generation"] == 2
    assert result["coarse_target_order_status"] == "no-preview-pass"
    assert result["coarse_target_order_applicable"] is False
    assert result["coarse_target_preview_ack_tiles"] == 0
    assert result["coarse_target_preview_task_finishes"] == 0


def test_coarse_target_trace_metrics_hard_fails_when_exact_scope_is_missing():
    from arrayscope.tools.profile_montage_workflow import _coarse_target_trace_metrics

    events = (
        {
            "kind": "input",
            "action": "phase_start",
            "phase": "raw_full_tiled_montage",
            "backend": "wgpu",
            "sequence": 1,
            "ts_ns": 1_000_000_000,
        },
        {
            "kind": "backend_ack",
            "accepted": True,
            "quality": "preview",
            "tile": 0,
            "sequence": 2,
            "ts_ns": 1_010_000_000,
        },
        {
            "kind": "input",
            "action": "phase_complete",
            "phase": "raw_full_tiled_montage",
            "backend": "wgpu",
            "sequence": 3,
            "ts_ns": 1_020_000_000,
        },
    )

    result = _coarse_target_trace_metrics(
        events,
        phase="raw_full_tiled_montage",
        backend="wgpu",
        phase_start_sequence=1,
        requested_tiles=1,
    )

    assert result["coarse_target_trace_window_complete"] is False
    assert result["coarse_target_order_status"] == "required-scope-missing"


def test_coarse_target_trace_metrics_rejects_equal_count_wrong_preview_tiles():
    from arrayscope.tools.profile_montage_workflow import _coarse_target_trace_metrics

    events = (
        {
            "kind": "input",
            "action": "phase_start",
            "phase": "raw_full_tiled_montage",
            "backend": "wgpu",
            "sequence": 1,
            "ts_ns": 1_000_000_000,
        },
        {
            "kind": "scheduling_phase",
            "event": "scope_started",
            "generation": 4,
            "required_tiles": 2,
            "required_tile_numbers": (0, 1),
            "session_id": 7,
            "sequence": 2,
            "ts_ns": 1_010_000_000,
        },
        {
            "kind": "kernel_start",
            "rung": 0,
            "task_seq": 77,
            "scheduling_generation": 4,
            "session_id": 7,
            "sequence": 3,
            "ts_ns": 1_015_000_000,
        },
        {
            "kind": "kernel_finish",
            "rung": 0,
            "task_seq": 77,
            "scheduling_generation": 4,
            "session_id": 7,
            "sequence": 3,
            "ts_ns": 1_020_000_000,
        },
        *(
            {
                "kind": "backend_ack",
                "accepted": True,
                "quality": "preview",
                "tile": tile,
                "session_id": 7,
                "sequence": 4 + offset,
                "ts_ns": 1_030_000_000 + offset * 1_000_000,
            }
            for offset, tile in enumerate((0, 2))
        ),
        {
            "kind": "kernel_start",
            "rung": 2,
            "sequence": 6,
            "ts_ns": 1_040_000_000,
        },
        *(
            {
                "kind": "backend_ack",
                "accepted": True,
                "quality": "exact",
                "tile": tile,
                "session_id": 7,
                "sequence": 7 + offset,
                "ts_ns": 1_050_000_000 + offset * 1_000_000,
            }
            for offset, tile in enumerate((0, 1))
        ),
        {
            "kind": "input",
            "action": "phase_complete",
            "phase": "raw_full_tiled_montage",
            "backend": "wgpu",
            "sequence": 9,
            "ts_ns": 1_060_000_000,
        },
    )

    result = _coarse_target_trace_metrics(
        events,
        phase="raw_full_tiled_montage",
        backend="wgpu",
        phase_start_sequence=1,
        requested_tiles=2,
    )

    assert result["coarse_target_preview_ack_tiles"] == 2
    assert result["coarse_target_t1_ms"] is None
    assert result["coarse_target_ack_ordered"] is False


def test_coarse_target_trace_metrics_requires_every_preview_task_to_finish():
    from arrayscope.tools.profile_montage_workflow import _coarse_target_trace_metrics

    events = (
        {
            "kind": "input",
            "action": "phase_start",
            "phase": "raw_full_tiled_montage",
            "backend": "wgpu",
            "sequence": 1,
            "ts_ns": 1_000_000_000,
        },
        {
            "kind": "scheduling_phase",
            "event": "scope_started",
            "generation": 2,
            "required_tiles": 1,
            "required_tile_numbers": (0,),
            "session_id": 5,
            "sequence": 2,
            "ts_ns": 1_010_000_000,
        },
        {
            "kind": "kernel_start",
            "rung": 0,
            "task_seq": 10,
            "scheduling_generation": 2,
            "session_id": 5,
            "sequence": 3,
            "ts_ns": 1_020_000_000,
        },
        {
            "kind": "kernel_start",
            "rung": 0,
            "task_seq": 11,
            "scheduling_generation": 2,
            "session_id": 5,
            "sequence": 4,
            "ts_ns": 1_021_000_000,
        },
        {
            "kind": "kernel_finish",
            "rung": 0,
            "task_seq": 10,
            "scheduling_generation": 2,
            "session_id": 5,
            "sequence": 5,
            "ts_ns": 1_030_000_000,
        },
        {
            "kind": "backend_ack",
            "accepted": True,
            "quality": "preview",
            "tile": 0,
            "session_id": 5,
            "sequence": 6,
            "ts_ns": 1_040_000_000,
        },
        {
            "kind": "kernel_start",
            "rung": 2,
            "sequence": 7,
            "ts_ns": 1_050_000_000,
        },
        {
            "kind": "backend_ack",
            "accepted": True,
            "quality": "exact",
            "tile": 0,
            "session_id": 5,
            "sequence": 8,
            "ts_ns": 1_060_000_000,
        },
        {
            "kind": "input",
            "action": "phase_complete",
            "phase": "raw_full_tiled_montage",
            "backend": "wgpu",
            "sequence": 9,
            "ts_ns": 1_070_000_000,
        },
    )

    result = _coarse_target_trace_metrics(
        events,
        phase="raw_full_tiled_montage",
        backend="wgpu",
        phase_start_sequence=1,
        requested_tiles=1,
    )

    assert result["coarse_target_preview_tasks_complete"] is False
    assert result["coarse_target_execution_ordered"] is False


def test_coarse_target_trace_metrics_does_not_build_t2_from_predecessor_session():
    from arrayscope.tools.profile_montage_workflow import _coarse_target_trace_metrics

    events = (
        {
            "kind": "input",
            "action": "phase_start",
            "phase": "raw_full_tiled_montage",
            "backend": "wgpu",
            "sequence": 1,
            "ts_ns": 1_000_000_000,
        },
        {
            "kind": "scheduling_phase",
            "event": "scope_started",
            "generation": 2,
            "required_tiles": 2,
            "required_tile_numbers": (0, 1),
            "session_id": 5,
            "sequence": 2,
            "ts_ns": 1_010_000_000,
        },
        {
            "kind": "kernel_start",
            "rung": 0,
            "task_seq": 10,
            "scheduling_generation": 2,
            "session_id": 5,
            "sequence": 3,
            "ts_ns": 1_020_000_000,
        },
        {
            "kind": "kernel_finish",
            "rung": 0,
            "task_seq": 10,
            "scheduling_generation": 2,
            "session_id": 5,
            "sequence": 4,
            "ts_ns": 1_030_000_000,
        },
        *(
            {
                "kind": "backend_ack",
                "accepted": True,
                "quality": "preview",
                "tile": tile,
                "session_id": 5,
                "sequence": 5 + offset,
                "ts_ns": 1_040_000_000 + offset * 1_000_000,
            }
            for offset, tile in enumerate((0, 1))
        ),
        *(
            {
                "kind": "backend_ack",
                "accepted": True,
                "quality": "exact",
                "tile": tile,
                "session_id": 4,
                "sequence": 7 + offset,
                "ts_ns": 1_050_000_000 + offset * 1_000_000,
            }
            for offset, tile in enumerate((0, 1))
        ),
        {
            "kind": "input",
            "action": "phase_complete",
            "phase": "raw_full_tiled_montage",
            "backend": "wgpu",
            "sequence": 9,
            "ts_ns": 1_070_000_000,
        },
    )

    result = _coarse_target_trace_metrics(
        events,
        phase="raw_full_tiled_montage",
        backend="wgpu",
        phase_start_sequence=1,
        requested_tiles=2,
    )

    assert result["coarse_target_target_ack_tiles"] == 0
    assert result["coarse_target_t2_ms"] is None
    assert result["coarse_target_ack_ordered"] is False


def test_r8_certification_reports_coarse_target_order_without_gating():
    from arrayscope.tools.profile_montage_workflow import _r8_certification

    record = _passing_r8_phase_record(backend="wgpu")
    record.update(
        coarse_target_order_status="overlap",
        coarse_target_ack_ordered=False,
        coarse_target_execution_ordered=False,
    )

    result = _r8_certification(record)

    failures = {failure["gate"] for failure in result["r8_gate_failures"]}
    assert result["r8_gate_passed"] is True
    assert "coarse_ack_pass_precedes_target_ack" not in failures
    assert "coarse_tasks_finish_before_target_starts" not in failures


def test_r8_certification_uses_invariant_preview_verdict_not_order_diagnostic():
    from arrayscope.tools.profile_montage_workflow import _r8_certification

    record = _passing_r8_phase_record(backend="wgpu")
    record.update(
        coarse_target_order_applicable=False,
        coarse_target_order_status="no-preview-pass",
        coarse_target_ack_ordered=None,
        coarse_target_execution_ordered=None,
        coarse_target_t1_ms=None,
        coarse_target_preview_ack_tiles=0,
        coarse_target_preview_task_finishes=0,
    )

    result = _r8_certification(record)

    failures = {failure["gate"] for failure in result["r8_gate_failures"]}
    assert result["r8_gate_passed"] is True
    assert "coarse_preview_pass_present" not in failures


def test_r8_certification_has_no_pyqtgraph_complex_preview_exemption():
    from arrayscope.tools.profile_montage_workflow import _r8_certification

    record = _passing_r8_phase_record(backend="pyqtgraph")
    record.update(
        phase="fft_full_tiled_montage",
        coarse_target_preview_required=True,
        coarse_target_preview_exemption=None,
        coarse_target_order_applicable=False,
        coarse_target_order_status="no-preview-pass",
        coarse_target_ack_ordered=None,
        coarse_target_execution_ordered=None,
        coarse_target_t1_ms=None,
        coarse_target_preview_ack_tiles=0,
        coarse_target_preview_task_finishes=0,
    )

    result = _r8_certification(record)

    failures = {failure["gate"] for failure in result["r8_gate_failures"]}
    assert record["coarse_target_preview_exemption"] is None
    assert "coarse_preview_pass_present" not in failures
    assert "coarse_ack_pass_precedes_target_ack" not in failures
    assert "coarse_tasks_finish_before_target_starts" not in failures


def test_r8_certification_reports_preview_latency_without_gating():
    from arrayscope.tools.profile_montage_workflow import _r8_certification

    record = _passing_r8_phase_record(backend="wgpu")
    record["coarse_target_t1_ms"] = 1_001.0

    result = _r8_certification(record)

    failures = {failure["gate"] for failure in result["r8_gate_failures"]}
    assert result["r8_gate_passed"] is True
    assert "preview_first_t1_target" not in failures


def test_r8_display_axis_wgpu_gate_requires_source_page_reuse():
    from arrayscope.tools.profile_montage_workflow import _r8_certification

    record = _passing_r8_phase_record(backend="wgpu")
    record.update(
        phase="display_x_axis_slice",
        requested_tile_count=50,
        active_planned_tile_count=50,
        active_presented_tile_count=50,
        display_axis_min_physical_tile_count=50,
        display_axis_physical_tile_sample_count=12,
        display_axis_slice_scroll_steps=3,
        display_axis_crop_scenario_count=11,
        display_axis_crop_scenarios_settled=True,
        display_axis_crop_scenarios_committed_current=True,
        display_axis_all_dimension_scroll_axis_count=3,
        display_axis_all_dimension_scroll_expected_axis_count=3,
        display_axis_all_dimension_scrolls_settled=True,
        display_axis_all_dimension_scrolls_committed_current=True,
        display_axis_all_dimension_scroll_results=(
            {
                "axis": 0,
                "role": "display",
                "fast_input_steps": 3,
                "slow_input_steps": 2,
                "wgpu_upload_delta": 0,
            },
            {
                "axis": 1,
                "role": "display",
                "fast_input_steps": 3,
                "slow_input_steps": 2,
                "wgpu_upload_delta": 0,
            },
            {
                "axis": 2,
                "role": "montage",
                "fast_input_steps": 3,
                "slow_input_steps": 2,
                "wgpu_upload_delta": 6,
            },
        ),
        display_axis_all_dimension_scroll_wgpu_upload_delta=6,
        display_axis_all_dimension_display_roles_wgpu_upload_delta=0,
        display_axis_all_dimension_montage_role_wgpu_upload_delta=6,
        display_axis_all_dimension_slice_roles_wgpu_upload_delta=None,
        display_axis_physical_reference_check_count=12,
        display_axis_physical_reference_passed=True,
        display_axis_physical_reference_failures=(),
        display_axis_roi_placement_check_count=12,
        display_axis_roi_placement_applicable=True,
        display_axis_roi_placement_passed=True,
        display_axis_roi_placement_failures=(),
        display_axis_wgpu_source_truth_check_count=12,
        display_axis_wgpu_source_truth_passed=True,
        display_axis_wgpu_source_truth_failures=(),
        display_axis_crop_scenario_names=(
            "primary-only-centered",
            "both-centered",
            "primary-minus-one",
            "secondary-plus-one",
            "both-diagonal",
            "both-return",
            "primary-page-edge",
            "primary-page-cross",
            "primary-page-return",
            "both-odd",
            "both-odd-primary-plus-one",
        ),
        display_axis_both_crop_scenario_count=10,
        display_axis_page_boundary_scenario_count=3,
        display_axis_crop_matrix_wgpu_upload_delta=0,
        display_axis_crop_wgpu_upload_delta=0,
        display_axis_scroll_wgpu_upload_delta=0,
        display_axis_xy_swap_settled=True,
        display_axis_xy_swap_steps=2,
        display_axis_xy_swap_wgpu_upload_delta=0,
        display_axis_single_slice_settled=True,
        display_axis_single_slice_committed_current=True,
        display_axis_single_slice_steps=3,
        display_axis_single_slice_wgpu_upload_delta=0,
        display_axis_montage_restore_settled=True,
        display_axis_montage_restore_committed_current=True,
        display_axis_wgpu_cold_binding_multiwindow_tiles=50,
        display_axis_wgpu_cold_binding_aliases=(),
        display_axis_wgpu_cold_binding_identity_unique=True,
        display_axis_wgpu_pool_exhaustion="",
        wgpu_page_pools=[{"representation": "scalar_r32f", "raw_resident": 100}],
        grid_kind="display_axis",
        grid_tile_count=50,
        full_tile_count=272,
        tile_cap_applied=True,
    )

    passed = _r8_certification(record)
    assert passed["r8_gate_passed"] is True

    record["display_axis_scroll_wgpu_upload_delta"] = 50
    failed = _r8_certification(record)
    failures = {failure["gate"] for failure in failed["r8_gate_failures"]}
    assert "display_axis_source_pages_reused" in failures

    record["display_axis_scroll_wgpu_upload_delta"] = 0
    record["display_axis_all_dimension_display_roles_wgpu_upload_delta"] = 1
    failed = _r8_certification(record)
    failures = {failure["gate"] for failure in failed["r8_gate_failures"]}
    assert "display_axis_source_pages_reused" in failures

    record["display_axis_all_dimension_display_roles_wgpu_upload_delta"] = 0
    record["display_axis_wgpu_source_truth_passed"] = False
    record["display_axis_wgpu_source_truth_failures"] = (
        {"actual_start_yx": (100, 100), "expected_start_yx": (101, 100)},
    )
    failed = _r8_certification(record)
    failures = {failure["gate"] for failure in failed["r8_gate_failures"]}
    assert "display_axis_wgpu_source_window_truth" in failures

    record["display_axis_wgpu_source_truth_passed"] = True
    record["display_axis_wgpu_source_truth_failures"] = ()
    record["display_axis_physical_reference_passed"] = False
    record["display_axis_physical_reference_failures"] = (
        {"tile_number": 17, "mismatched": 40, "samples": 40},
    )
    failed = _r8_certification(record)
    failures = {failure["gate"] for failure in failed["r8_gate_failures"]}
    assert "display_axis_all_dimension_pixels_match_cpu_reference" in failures

    record["display_axis_physical_reference_passed"] = True
    record["display_axis_physical_reference_failures"] = ()
    record["display_axis_xy_swap_settled"] = False
    failed = _r8_certification(record)
    failures = {failure["gate"] for failure in failed["r8_gate_failures"]}
    assert "display_axis_xy_swap_settles" in failures

    record["display_axis_xy_swap_settled"] = True
    record["display_axis_crop_scenario_count"] = 10
    failed = _r8_certification(record)
    failures = {failure["gate"] for failure in failed["r8_gate_failures"]}
    assert "display_axis_crop_matrix_complete" in failures

    record["display_axis_crop_scenario_count"] = 11
    record["display_axis_wgpu_cold_binding_identity_unique"] = False
    record["display_axis_wgpu_cold_binding_aliases"] = (
        {"tile": 7, "source_windows": 2, "plane_identities": 1},
    )
    failed = _r8_certification(record)
    failures = {failure["gate"] for failure in failed["r8_gate_failures"]}
    assert "display_axis_cold_crop_bindings_do_not_alias" in failures


def test_r8_display_axis_wgpu_gate_surfaces_pool_exhaustion():
    from arrayscope.tools.profile_montage_workflow import _r8_certification

    record = _passing_r8_phase_record(backend="wgpu")
    record.update(
        phase="display_y_axis_slice",
        requested_tile_count=50,
        active_planned_tile_count=50,
        active_presented_tile_count=50,
        display_axis_min_physical_tile_count=50,
        display_axis_physical_tile_sample_count=8,
        display_axis_slice_scroll_steps=3,
        display_axis_crop_scenario_count=11,
        display_axis_crop_scenarios_settled=True,
        display_axis_crop_scenarios_committed_current=True,
        display_axis_all_dimension_scroll_axis_count=3,
        display_axis_all_dimension_scroll_expected_axis_count=3,
        display_axis_all_dimension_scrolls_settled=True,
        display_axis_all_dimension_scrolls_committed_current=True,
        display_axis_all_dimension_scroll_results=(
            {
                "axis": 0,
                "role": "display",
                "fast_input_steps": 3,
                "slow_input_steps": 2,
                "wgpu_upload_delta": 0,
            },
            {
                "axis": 1,
                "role": "display",
                "fast_input_steps": 3,
                "slow_input_steps": 2,
                "wgpu_upload_delta": 0,
            },
            {
                "axis": 2,
                "role": "montage",
                "fast_input_steps": 3,
                "slow_input_steps": 2,
                "wgpu_upload_delta": 0,
            },
        ),
        display_axis_all_dimension_scroll_wgpu_upload_delta=0,
        display_axis_all_dimension_display_roles_wgpu_upload_delta=0,
        display_axis_all_dimension_montage_role_wgpu_upload_delta=0,
        display_axis_all_dimension_slice_roles_wgpu_upload_delta=None,
        display_axis_physical_reference_check_count=12,
        display_axis_physical_reference_passed=True,
        display_axis_physical_reference_failures=(),
        display_axis_roi_placement_check_count=12,
        display_axis_roi_placement_applicable=True,
        display_axis_roi_placement_passed=True,
        display_axis_roi_placement_failures=(),
        display_axis_wgpu_source_truth_check_count=12,
        display_axis_wgpu_source_truth_passed=True,
        display_axis_wgpu_source_truth_failures=(),
        display_axis_crop_scenario_names=(
            "primary-only-centered",
            "both-centered",
            "primary-minus-one",
            "secondary-plus-one",
            "both-diagonal",
            "both-return",
            "primary-page-edge",
            "primary-page-cross",
            "primary-page-return",
            "both-odd",
            "both-odd-primary-plus-one",
        ),
        display_axis_both_crop_scenario_count=10,
        display_axis_page_boundary_scenario_count=3,
        display_axis_crop_matrix_wgpu_upload_delta=0,
        display_axis_crop_wgpu_upload_delta=0,
        display_axis_scroll_wgpu_upload_delta=0,
        display_axis_xy_swap_settled=True,
        display_axis_xy_swap_steps=2,
        display_axis_xy_swap_wgpu_upload_delta=0,
        display_axis_single_slice_settled=True,
        display_axis_single_slice_committed_current=True,
        display_axis_single_slice_steps=3,
        display_axis_single_slice_wgpu_upload_delta=0,
        display_axis_montage_restore_settled=True,
        display_axis_montage_restore_committed_current=True,
        display_axis_wgpu_cold_binding_multiwindow_tiles=50,
        display_axis_wgpu_cold_binding_aliases=(),
        display_axis_wgpu_cold_binding_identity_unique=True,
        display_axis_wgpu_pool_exhaustion="page pool 'scalar_r32f' exhausted",
        grid_kind="display_axis",
        grid_tile_count=50,
        full_tile_count=272,
        tile_cap_applied=True,
    )

    result = _r8_certification(record)
    failures = {failure["gate"] for failure in result["r8_gate_failures"]}
    assert "display_axis_page_pool_has_headroom" in failures


def test_r8_certification_ignores_outer_window_size_but_still_gates_the_viewport():
    """Window size is viewport + chrome, so it is an outcome, not a promise.

    Pinning it made every profile run fail the moment the menu/tool/status
    bars grew 8 px: the fixture's 739-row viewport restored EXACTLY and the
    window that holds it went 940 -> 948 (reproduced on a live Wayland
    session, so never a headless artifact).  The viewport is what decides
    aspect, montage layout, and LOD, and a restore that did not happen fails
    it — so that is the gate, and the window sizes stay in the record as
    diagnostics.
    """

    from arrayscope.tools.profile_montage_workflow import _r8_certification

    record = _passing_r8_phase_record(backend="wgpu")
    record.update(window_size=[1400, 948], session_window_size_chrome_delta=[0, 8])
    assert _r8_certification(record)["r8_gate_passed"] is True

    record.update(session_viewport_shape_matches=False, viewport_shape=[600, 1245])
    result = _r8_certification(record)
    assert result["r8_gate_passed"] is False
    assert "session_viewport_geometry_stable" in {
        failure["gate"] for failure in result["r8_gate_failures"]
    }


def test_r8_certification_gates_first_pixel_truth_but_reports_latency():
    from arrayscope.tools.profile_montage_workflow import _r8_certification

    record = _passing_r8_phase_record(backend="pyqtgraph")
    record.update(
        phase="montage_zoompan_scalar",
        first_visible_levels_default=True,
        first_visible_histogram_empty=True,
        first_visible_level_evidence_quality=1,
        presentation_blackout_observed=True,
        presentation_continuity_ok=False,
        action_render_call_ms=71.0,
        event_loop_max_gap_ms=90.0,
    )

    result = _r8_certification(record)
    failed = {failure["gate"] for failure in result["r8_gate_failures"]}

    assert result["r8_gate_passed"] is False
    assert {
        "first_visible_levels_semantic",
        "first_visible_histogram_populated",
        "first_visible_level_evidence_quality",
        "presentation_continuity",
    }.issubset(failed)
    assert "gui_callbacks_below_50ms" not in failed
    assert "event_loop_heartbeat" not in failed
    assert result["r8_direct_ui_call_ms"]["action_render_call_ms"] == 71.0
    assert result["r8_heartbeat_max_gap_ms"] == 90.0


def test_r8_certification_reports_but_does_not_gate_cold_fill_heartbeat():
    from arrayscope.tools.profile_montage_workflow import _r8_certification

    record = _passing_r8_phase_record()
    record["event_loop_max_gap_ms"] = 90.0

    result = _r8_certification(record)

    assert result["r8_heartbeat_gate_applicable"] is False
    assert result["r8_heartbeat_max_gap_ms"] == 90.0
    assert result["r8_gate_passed"] is True


def test_r8_certification_gates_deep_zoom_far_scroll_convergence():
    from arrayscope.tools.profile_montage_workflow import _r8_certification

    record = _passing_r8_phase_record(backend="wgpu")
    record.update(
        phase="montage_zoompan_scalar",
        deep_zoom_far_scroll_available=True,
        deep_zoom_far_scroll_precondition_reached_target_lod=True,
        deep_zoom_far_scroll_reached_target_lod=False,
        deep_zoom_far_scroll_target_evidence={"atomic_successor_pending": True},
    )

    result = _r8_certification(record)
    failed = {failure["gate"] for failure in result["r8_gate_failures"]}

    assert "deep_zoom_far_scroll_reaches_target_lod" in failed


def test_progressive_invariant_gate_does_not_skip_r5_for_profiled_runs():
    result = _contract_verdict(
        _passing_contract_evidence(),
        profiler_type="cprofile",
        action_render_call_ms=500.0,
        event_loop_max_gap_ms=500.0,
    )

    assert result["invariant_gate_passed"] is False
    assert "all_gui_callbacks_below_50ms" in _failed_invariant_gates(result)


def test_profile_suite_manifest_records_success_and_summary(tmp_path, monkeypatch):
    import arrayscope.tools.profile_montage_workflow as workflow

    artifact = tmp_path / "py-spy.raw"
    jsonl = tmp_path / "py-spy.jsonl"
    monkeypatch.setattr(
        workflow,
        "profiler_suite_commands",
        lambda _argv, _suite_dir: (
            {
                "profiler_type": "py-spy-raw",
                "required": True,
                "jsonl": str(jsonl),
                "artifact_paths": (str(artifact), str(jsonl)),
                "command": "py-spy record -- fake-success",
            },
        ),
    )
    monkeypatch.setattr(
        workflow, "_suite_tool_versions", lambda: {"python": "test", "py-spy": "py-spy 0.4.2"}
    )
    monkeypatch.setattr(
        workflow,
        "_repository_state",
        lambda: {"repository_revision": "abc123", "repository_dirty": False},
    )
    monkeypatch.setattr(workflow.shutil, "which", lambda exe: f"/usr/bin/{exe}")

    def fake_run(_command, **kwargs):
        artifact.write_text("samples", encoding="utf-8")
        jsonl.write_text('{"phase":"load"}\n', encoding="utf-8")
        kwargs["stdout"].write(
            "py-spy> Sampling process 50 times a second. Press Control-C to exit.\n"
            "py-spy> Wrote raw flamegraph data. Samples: 12 Errors: 0\n"
        )
        kwargs["stderr"].write("stderr\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(workflow.subprocess, "run", fake_run)

    rc = workflow.run_profile_suite(("--backend", "pyqtgraph"), tmp_path)

    records = [
        json.loads(line)
        for line in (tmp_path / "suite-manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    step, summary = records
    assert rc == 0
    assert step["record_type"] == "suite_step"
    assert step["status"] == "completed"
    assert step["valid"] is True
    assert step["complete"] is True
    assert step["command"] == "py-spy record -- fake-success"
    assert step["returncode"] == 0
    assert Path(step["stdout_path"]).exists()
    assert Path(step["stderr_path"]).exists()
    assert step["tool_versions"]["py-spy"] == "py-spy 0.4.2"
    assert step["repository_revision"] == "abc123"
    assert step["run_temperature"] == "cold"
    assert summary["record_type"] == "suite_summary"
    assert summary["overall_valid"] is True
    assert summary["overall_status"] == "completed"
    assert summary["run_temperature"] == "cold"
    report = Path(summary["interpretation_path"])
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "Profile Suite Summary" in text
    assert "Timing Evidence" in text
    assert "py-spy" in text


def test_profile_suite_summary_marks_multiple_child_processes_mixed(tmp_path, monkeypatch):
    import arrayscope.tools.profile_montage_workflow as workflow

    artifacts = [tmp_path / "plain.jsonl", tmp_path / "py-spy.raw"]
    monkeypatch.setattr(
        workflow,
        "profiler_suite_commands",
        lambda _argv, _suite_dir: (
            {
                "profiler_type": "plain",
                "required": True,
                "jsonl": str(artifacts[0]),
                "artifact_paths": (str(artifacts[0]),),
                "command": "python -m arrayscope.tools.profile_montage_workflow",
            },
            {
                "profiler_type": "py-spy-raw",
                "required": True,
                "jsonl": str(tmp_path / "py-spy.jsonl"),
                "artifact_paths": (str(artifacts[1]),),
                "command": "py-spy record -- fake-success",
            },
        ),
    )
    monkeypatch.setattr(workflow, "_suite_tool_versions", lambda: {"python": "test"})
    monkeypatch.setattr(
        workflow,
        "_repository_state",
        lambda: {"repository_revision": "abc123", "repository_dirty": False},
    )
    monkeypatch.setattr(workflow.shutil, "which", lambda exe: f"/usr/bin/{exe}")

    def fake_run(command, **kwargs):
        if command[0] == "py-spy":
            artifacts[1].write_text("samples", encoding="utf-8")
            kwargs["stdout"].write(
                "py-spy> Sampling process 50 times a second. Press Control-C to exit.\n"
                "py-spy> Wrote raw flamegraph data. Samples: 12 Errors: 0\n"
            )
        else:
            artifacts[0].write_text('{"phase":"load"}\n', encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(workflow.subprocess, "run", fake_run)

    assert workflow.run_profile_suite(("--backend", "pyqtgraph"), tmp_path) == 0

    records = [
        json.loads(line)
        for line in (tmp_path / "suite-manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["run_temperature"] for record in records[:2]] == ["cold", "warm"]
    assert records[-1]["run_temperature"] == "mixed"


def test_profile_suite_py_spy_nonzero_is_failed_even_with_artifacts(tmp_path, monkeypatch):
    import arrayscope.tools.profile_montage_workflow as workflow

    artifact = tmp_path / "py-spy.raw"
    jsonl = tmp_path / "py-spy.jsonl"
    monkeypatch.setattr(
        workflow,
        "profiler_suite_commands",
        lambda _argv, _suite_dir: (
            {
                "profiler_type": "py-spy-raw",
                "required": True,
                "jsonl": str(jsonl),
                "artifact_paths": (str(artifact), str(jsonl)),
                "command": "py-spy record -- fake-failure",
            },
        ),
    )
    monkeypatch.setattr(
        workflow, "_suite_tool_versions", lambda: {"python": "test", "py-spy": "py-spy 0.4.2"}
    )
    monkeypatch.setattr(
        workflow,
        "_repository_state",
        lambda: {"repository_revision": "abc123", "repository_dirty": False},
    )
    monkeypatch.setattr(workflow.shutil, "which", lambda exe: f"/usr/bin/{exe}")

    def fake_run(_command, **kwargs):
        artifact.write_text("samples", encoding="utf-8")
        jsonl.write_text('{"phase":"load"}\n', encoding="utf-8")
        kwargs["stderr"].write("py-spy failed\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(workflow.subprocess, "run", fake_run)

    rc = workflow.run_profile_suite(("--backend", "pyqtgraph"), tmp_path)

    records = [
        json.loads(line)
        for line in (tmp_path / "suite-manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    step, summary = records
    assert rc == 1
    assert step["status"] == "failed"
    assert step["valid"] is False
    assert step["complete"] is False
    assert step["returncode"] == 1
    assert step["missing_artifacts"] == []
    assert "nonzero_returncode_ignored" not in step
    assert summary["overall_valid"] is False
    assert summary["overall_status"] == "failed"


def test_py_spy_full_profile_tolerates_one_missed_stack(tmp_path):
    from arrayscope.tools.profile_montage_workflow import (
        _profiler_log_diagnostics,
        _profiler_sample_issue,
    )

    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_text(
        "py-spy> Sampling process 50 times a second. Press Control-C to exit.\n"
        "py-spy> Wrote raw flamegraph data. Samples: 20 Errors: 1\n",
        encoding="utf-8",
    )
    stderr.write_text("[WARN  py_spy] Failed to get stack trace from 123\n", encoding="utf-8")

    diagnostics = _profiler_log_diagnostics("py-spy-raw-full", stdout, stderr)

    assert diagnostics["sample_rate_hz"] == 50
    assert diagnostics["sample_count"] == 20
    assert diagnostics["error_count"] == 1
    assert diagnostics["missed_stack_count"] == 1
    assert diagnostics["sampling_complete"] is True
    assert _profiler_sample_issue("py-spy-raw-full", diagnostics) == ""


def test_py_spy_full_profile_rejects_multiple_missed_stacks(tmp_path):
    from arrayscope.tools.profile_montage_workflow import (
        _profiler_log_diagnostics,
        _profiler_sample_issue,
    )

    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_text(
        "py-spy> Sampling process 50 times a second. Press Control-C to exit.\n"
        "py-spy> Wrote raw flamegraph data. Samples: 20 Errors: 2\n",
        encoding="utf-8",
    )
    stderr.write_text(
        "[WARN  py_spy] Failed to get stack trace from 123\n[WARN  py_spy] Failed to get stack trace from 456\n",
        encoding="utf-8",
    )

    diagnostics = _profiler_log_diagnostics("py-spy-raw-full", stdout, stderr)

    assert diagnostics["sampling_complete"] is False
    assert (
        _profiler_sample_issue("py-spy-raw-full", diagnostics)
        == "py-spy full profile missed more than 1 stack sample(s)"
    )


def test_profile_base_record_marks_offscreen_or_capped_runs_as_smoke(monkeypatch):
    import numpy as np

    from arrayscope.tools.profile_montage_workflow import _base_record

    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    visible = _base_record(
        run_id="run",
        backend="wgpu",
        data_path=Path("data.nii"),
        data=np.zeros((2, 3, 4), dtype=np.float32),
        load_mode="native",
        montage_axis=2,
        indices=(0, 1, 2, 3),
        full_tile_count=4,
        columns=2,
        max_tiles=None,
        profiler_type="plain",
        profiler_artifact_paths=(),
        qt_platform="xcb",
    )
    photographed = _base_record(
        run_id="run",
        backend="wgpu",
        data_path=Path("data.nii"),
        data=np.zeros((2, 3, 4), dtype=np.float32),
        load_mode="native",
        montage_axis=2,
        indices=(0, 1, 2, 3),
        full_tile_count=4,
        columns=2,
        max_tiles=None,
        profiler_type="plain",
        profiler_artifact_paths=(),
        qt_platform="wayland",
        screenshot_timing_perturbed=True,
    )
    hidden = {
        **visible,
        **_base_record(
            run_id="run",
            backend="wgpu",
            data_path=Path("data.nii"),
            data=np.zeros((2, 3, 4), dtype=np.float32),
            load_mode="native",
            montage_axis=2,
            indices=(0, 1),
            full_tile_count=4,
            columns=2,
            max_tiles=2,
            profiler_type="perf-record",
            profiler_artifact_paths=("perf.data",),
            qt_platform="offscreen",
        ),
    }

    assert visible["smoke_only"] is False
    assert visible["pacing_evidence"] is True
    assert visible["screenshot_timing_perturbed"] is False
    assert photographed["smoke_only"] is False
    assert photographed["screenshot_timing_perturbed"] is True
    assert photographed["pacing_evidence"] is False
    assert visible["xdg_session_type"] == "wayland"
    assert hidden["smoke_only"] is True
    assert hidden["pacing_evidence"] is False
    assert hidden["tile_cap_applied"] is True
    assert hidden["profiler_artifact_paths"] == ["perf.data"]


def test_profile_base_record_exposes_intentional_scroll_grid_as_pacing_evidence(monkeypatch):
    import numpy as np

    from arrayscope.tools.profile_montage_workflow import _base_record

    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    record = _base_record(
        run_id="run",
        backend="wgpu",
        data_path=Path("data.nii"),
        data=np.zeros((2, 3, 272), dtype=np.float32),
        load_mode="native",
        montage_axis=2,
        indices=tuple(range(60)),
        full_tile_count=272,
        columns=8,
        max_tiles=60,
        profiler_type="plain",
        profiler_artifact_paths=(),
        qt_platform="wayland",
        grid_kind="scroll",
        source_index_count=272,
    )

    assert record["grid_kind"] == "scroll"
    assert record["grid_tile_count"] == 60
    assert record["source_index_count"] == 272
    assert record["tile_cap_applied"] is False
    assert record["smoke_only"] is False
    assert record["pacing_evidence"] is True


def test_centered_indices_selects_middle_of_full_axis():
    from arrayscope.tools.profile_montage_workflow import _centered_indices

    assert _centered_indices(10, 4) == (3, 4, 5, 6)
    assert _centered_indices(10, 12) == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert _centered_indices(11, 5) == (3, 4, 5, 6, 7)


def test_histogram_continuity_metrics_accepts_monotonic_refinement():
    from arrayscope.tools.profile_montage_workflow import _histogram_continuity_metrics

    rows = (
        {
            "successor_visible": False,
            "histogram_data_bounds": [0.0, 100.0],
            "histogram_empty": False,
            "levels_look_default": False,
            "level_source_count": 60,
        },
        {
            "successor_visible": True,
            "histogram_data_bounds": [-2.0, 102.0],
            "histogram_empty": False,
            "levels_look_default": False,
            "level_source_count": 59,
        },
        {
            "successor_visible": True,
            "histogram_data_bounds": [-3.0, 104.0],
            "histogram_empty": False,
            "levels_look_default": False,
            "level_source_count": 60,
        },
    )

    metrics = _histogram_continuity_metrics(rows)

    assert metrics["window_level_flicker_free"] is True
    assert metrics["level_transient_span_dip_ratio"] == 1.0
    assert metrics["level_source_count_regressed"] is False


def test_histogram_continuity_metrics_rejects_transient_range_dip_and_empty_state():
    from arrayscope.tools.profile_montage_workflow import _histogram_continuity_metrics

    rows = (
        {
            "successor_visible": True,
            "histogram_data_bounds": [0.0, 100.0],
            "histogram_empty": False,
            "levels_look_default": False,
            "level_source_count": 60,
        },
        {
            "successor_visible": True,
            "histogram_data_bounds": None,
            "histogram_empty": True,
            "levels_look_default": True,
            "level_source_count": 4,
        },
        {
            "successor_visible": True,
            "histogram_data_bounds": [40.0, 60.0],
            "histogram_empty": False,
            "levels_look_default": False,
            "level_source_count": 16,
        },
        {
            "successor_visible": True,
            "histogram_data_bounds": [-5.0, 105.0],
            "histogram_empty": False,
            "levels_look_default": False,
            "level_source_count": 60,
        },
    )

    metrics = _histogram_continuity_metrics(rows)

    assert metrics["window_level_flicker_free"] is False
    assert metrics["histogram_emptied_after_successor_visible"] is True
    assert metrics["levels_defaulted_after_successor_visible"] is True
    assert metrics["level_transient_span_dip_ratio"] < 0.25
    assert metrics["level_source_count_regressed"] is True


def test_histogram_continuity_metrics_allows_one_atomic_range_change_per_semantic_action():
    from arrayscope.tools.profile_montage_workflow import _histogram_continuity_metrics

    rows = (
        {
            "successor_visible": True,
            "histogram_data_bounds": [0.0, 100.0],
            "histogram_empty": False,
            "levels_look_default": False,
            "level_source_count": 50,
            "level_semantic_key": "crop-a",
        },
        {
            "successor_visible": True,
            "histogram_data_bounds": [40.0, 60.0],
            "histogram_empty": False,
            "levels_look_default": False,
            "level_source_count": 50,
            "level_semantic_key": "crop-b",
        },
        {
            "successor_visible": True,
            "histogram_data_bounds": [-20.0, 120.0],
            "histogram_empty": False,
            "levels_look_default": False,
            "level_source_count": 50,
            "level_semantic_key": "crop-c",
        },
    )

    metrics = _histogram_continuity_metrics(rows)

    assert metrics["window_level_flicker_free"] is True
    assert metrics["level_transient_span_dip_ratio"] == 1.0
    assert metrics["level_source_count_regressed"] is False


def test_presentation_continuity_probe_detects_retained_frame_blackout_and_camera_drift():
    from arrayscope.tools.profile_montage_workflow import _PresentationContinuityProbe

    class FakeTimer:
        def __init__(self, _parent):
            self.callback = None

        def setInterval(self, _interval):
            return None

        @property
        def timeout(self):
            return SimpleNamespace(connect=lambda callback: setattr(self, "callback", callback))

        def start(self):
            return None

        def stop(self):
            return None

    predecessor = SimpleNamespace(key=SimpleNamespace(semantic_key="old"))
    visible_state = SimpleNamespace(visible=True)
    image_view = SimpleNamespace(
        montageDisplayMode=lambda: "tile_layer",
        _montage_tile_layer=SimpleNamespace(states={0: visible_state}),
        _viewport_content_extent=(20, 40),
    )
    win = SimpleNamespace(img_view=image_view, _committed_display_frame=predecessor)
    probe = _PresentationContinuityProbe(SimpleNamespace(QTimer=FakeTimer), win)

    probe.start()
    visible_state.visible = False
    image_view._viewport_content_extent = (40, 20)
    probe._sample()
    probe.stop()

    record = probe.record()
    assert record["presentation_continuity_expected"] is True
    assert record["presentation_blackout_observed"] is True
    assert record["presentation_extent_changed_before_commit"] is True
    assert record["presentation_continuity_ok"] is False


def test_presentation_continuity_probe_times_topology_successor_without_committed_frame():
    from arrayscope.tools.profile_montage_workflow import _PresentationContinuityProbe

    class FakeTimer:
        def __init__(self, _parent):
            pass

        def setInterval(self, _interval):
            return None

        @property
        def timeout(self):
            return SimpleNamespace(connect=lambda _callback: None)

        def start(self):
            return None

        def stop(self):
            return None

    visible_state = SimpleNamespace(visible=True, source_array_id="old")
    image_view = SimpleNamespace(
        montageDisplayMode=lambda: "tile_layer",
        _montage_tile_layer=SimpleNamespace(states={0: visible_state}),
        _viewport_content_extent=(20, 40),
        getLevels=lambda: (-2.0, 8.0),
        getHistogramDataBounds=lambda: (-2.0, 8.0),
    )
    level_source = SimpleNamespace(rank=2, source_count=4, evidence_quality=1)
    session = SimpleNamespace(
        semantic_key="old",
        applied_level_source=level_source,
        plan=SimpleNamespace(tile_shape=(4, 4), columns=1, rows=1, gap=1, tiles=(0,)),
    )
    win = SimpleNamespace(
        img_view=image_view,
        _committed_display_frame=None,
        _frame_session=session,
        renderer=SimpleNamespace(
            _last_montage_level_decision=None, _montage_refined_level_applied_count=0
        ),
    )
    probe = _PresentationContinuityProbe(SimpleNamespace(QTimer=FakeTimer), win)

    probe.start()
    visible_state.visible = False
    session.plan = SimpleNamespace(tile_shape=(4, 4), columns=2, rows=1, gap=1, tiles=(0, 1))
    probe._sample()
    visible_state.visible = True
    visible_state.source_array_id = "new"
    probe.stop()

    record = probe.record()
    assert record["presentation_continuity_expected"] is False
    assert record["presentation_topology_changed"] is True
    assert record["presentation_blackout_observed"] is True
    assert record["presentation_successor_observed"] is True
    assert record["first_visible_tile_ms"] >= 0.0
    assert record["first_visible_display_levels"] == [-2.0, 8.0]
    assert record["first_visible_histogram_data_bounds"] == [-2.0, 8.0]
    assert record["first_visible_level_evidence_quality"] == 1
    assert record["presentation_continuity_ok"] is True


def test_montage_scroll_pattern_targets_selected_center_band(monkeypatch):
    import arrayscope.tools.profile_montage_workflow as workflow

    fake_calls = {"fast": {}, "slow": {}}

    def fake_fast(win, **kwargs):
        fake_calls["fast"] = kwargs
        return {"fast_scroll_end_start": 2}

    def fake_slow(win, **kwargs):
        fake_calls["slow"] = kwargs
        return {"slow_scroll_steps": 0}

    class FakeViewState:
        def with_image_axes(self, *args, **kwargs):
            return self

        def with_montage_axis(self, *args, **kwargs):
            return ("state", tuple(kwargs["indices"]))

    win = type(
        "W",
        (),
        {
            "view_state": FakeViewState(),
            "_set_view_state": lambda self, state: None,
            "render": lambda self, *args, **kwargs: None,
        },
    )()

    def fake_lod_state(_win):
        return {}

    monkeypatch.setattr(workflow, "_fast_scroll_60fps", fake_fast)
    monkeypatch.setattr(workflow, "_slow_scroll_lod_paced", fake_slow)
    monkeypatch.setattr(workflow, "_lod_state_record", fake_lod_state)

    record = workflow._apply_montage_scroll_pattern(
        win,
        montage_axis=2,
        columns=3,
        indices=tuple(range(20)),
        window_size=8,
        probe=None,
        app=None,
        QtCore=None,
    )
    assert record["scroll_window_size"] == fake_calls["fast"]["size"]
    assert record["scroll_center_band"] == [fake_calls["fast"]["low"], fake_calls["fast"]["high"]]
    selected_count = len(range(20))
    window_size = 8
    assert (
        0 <= fake_calls["fast"]["low"] <= fake_calls["fast"]["high"] <= selected_count - window_size
    )
    assert fake_calls["slow"].get("indices") == tuple(range(20))


def test_apply_montage_zoom_pan_targets_bounded_factors(monkeypatch):
    import arrayscope.tools.profile_montage_workflow as workflow

    def fake_montage_view_range(_win):
        return ((-10.0, 10.0), (-20.0, 20.0))

    def fake_glide(win, app, QtCore, probe, target_range, *, frames, fps=60.0, frame_action=None):
        for frame in range(1, int(frames) + 1):
            if callable(frame_action):
                frame_action(frame, int(frames))
        return {"frames": int(frames), "glide_frames": int(frames), "glide_fps": fps}

    monkeypatch.setattr(workflow, "_montage_view_range", fake_montage_view_range)
    monkeypatch.setattr(workflow, "_glide_view_range", fake_glide)
    monkeypatch.setattr(
        workflow,
        "_maximum_zoomout_view_range",
        lambda *_args: ((-200.0, 200.0), (-400.0, 400.0)),
    )
    monkeypatch.setattr(
        workflow,
        "_full_montage_view_range",
        lambda *_args: ((0.0, 80.0), (0.0, 60.0)),
    )
    monkeypatch.setattr(
        workflow,
        "_few_tile_view_range",
        lambda *_args, **_kwargs: ((20.0, 30.0), (20.0, 30.0)),
    )
    monkeypatch.setattr(
        workflow,
        "_wait_for_visible_target_then_observe_near",
        lambda *_args: {
            "visible_target_reached": True,
            "active_tiles": tuple(range(60)),
            "near_tiles": (),
            "near_new_before_visible": (),
            "near_new_before_visible_count": 0,
            "resident_query_available": True,
        },
    )
    monkeypatch.setattr(workflow, "_wait_for_target_lod", lambda *args, **kwargs: (True, 0.0))
    monkeypatch.setattr(workflow, "_wait_for_tile_presentation_draw", lambda *args, **kwargs: True)
    scroll_calls = []
    monkeypatch.setattr(
        workflow, "_scroll_montage_window", lambda *args, **kwargs: scroll_calls.append(kwargs)
    )

    record = workflow._apply_montage_zoom_pan_stress(
        SimpleNamespace(),
        probe=None,
        app=object(),
        QtCore=object(),
        mid_toggle=None,
        montage_axis=2,
        columns=8,
        indices=tuple(range(272)),
        window_size=60,
    )
    assert record["zoompan_input_fps"] >= 120.0
    assert record["zoompan_max_out_request_scale"] > 1000.0
    assert tuple(record["zoompan_max_out_range"][0]) == (-200.0, 200.0)
    assert 0.1 < record["zoompan_zoomin_span_scale"] <= 1.0
    assert 0.3 <= record["zoompan_pan_right_dx_frac"] <= 0.5
    assert 0.0 < record["zoompan_pan_down_dy_frac"] <= 0.5
    assert 0.01 < record["zoompan_deep_zoom_span_scale"] < 0.2
    assert record["erratic_zoomout_frames"] == 3
    assert record["opposite_pan_frames"] == 2
    assert record["combined_zoom_scroll_available"] is True
    assert record["combined_scroll_window_size"] == 60
    assert record["lod_full_grid_active_count"] == 60
    assert record["full_grid_zoomin_frames"] == 4
    assert record["combined_full_grid_scroll_pause_frames"] == 3
    assert record["combined_zoom_scroll_pause_frames"] == 3
    assert record["combined_pan_scroll_pause_frames"] == 3
    assert record["deep_zoom_far_scroll_available"] is True
    assert record["deep_zoom_far_scroll_precondition_reached_target_lod"] is True
    assert record["deep_zoom_far_scroll_reached_target_lod"] is True
    assert record["deep_zoom_far_scroll_index_distance"] > 0
    assert scroll_calls
    assert all(call["size"] == 60 and call["interactive"] is True for call in scroll_calls)
    assert all(0 < call["window_start"] < 212 for call in scroll_calls)


def test_profile_montage_completion_waits_for_level_generation_by_default():
    from arrayscope.display.model.presentation_generation import PresentationGenerationTracker
    from arrayscope.operations.stage_fanin import StageFanInState
    from arrayscope.tools.profile_montage_workflow import _wait_for_montage_complete

    class FakeQtCore:
        class QEventLoop:
            class ProcessEventsFlag:
                AllEvents = object()

        class Qt:
            class TimerType:
                PreciseTimer = object()

        class QDeadlineTimer:
            def __init__(self, *_args):
                pass

    class FakeImageView:
        def montageDisplayMode(self):
            return "tile_layer"

        def montageTileOverlayCount(self):
            return 0

    level_generation = PresentationGenerationTracker()
    level_generation.begin_target((2.0, 4.0), active_tiles=(0,))
    level_generation.tile_values[0] = (0.0, 1.0)
    level_generation.tile_revisions[0] = 0
    level_generation.set_active_tiles((0,))
    session = SimpleNamespace(
        visible_tiles=(SimpleNamespace(montage_index=0),),
        skipped_tiles=set(),
        lifecycle=SimpleNamespace(presented_tiles=frozenset({0})),
        display_committed=True,
        required_target_unsettled_tiles=lambda: (),
        loading_tiles=set(),
        active_tile_requests=set(),
        stage_fan_in=StageFanInState(),
        final_commit_pending=False,
        flush_pending=False,
        dirty_payloads={},
        pending_removals=set(),
        level_generation=level_generation,
        is_complete=lambda: False,
        required_target_settled=lambda: True,
    )
    session.level_presentation_snapshot = lambda: session.level_generation.snapshot()
    session.has_pending_level_update = lambda: not session.level_presentation_snapshot().settled
    win = SimpleNamespace(img_view=FakeImageView(), _frame_session=session)

    class FakeApp:
        def __init__(self):
            self.calls = 0

        def processEvents(self, *_args):
            self.calls += 1
            if self.calls >= 3:
                session.level_generation.acknowledge_upserts(
                    session.level_generation.revision,
                    (0,),
                    levels=(2.0, 4.0),
                )

    app = FakeApp()
    result = _wait_for_montage_complete(
        app,
        FakeQtCore,
        win,
        timeout_s=bounded_interaction_settle_timeout_s(0.5),
        start=0.0,
        draw_start=0,
    )

    assert app.calls >= 3
    assert result["presentation_settled"] is True
    assert result["stale_level_tiles"] == 0
    assert result["pending_level_tiles"] == 0
    assert result["level_revision"] == 1
    assert result["active_level_value_count"] == 1


def test_displayed_axis_profile_waits_for_the_requested_successor(monkeypatch):
    import arrayscope.tools.profile_montage_workflow as workflow

    old_session = SimpleNamespace(session_id=7)
    new_session = SimpleNamespace(session_id=8)
    win = SimpleNamespace(_frame_session=old_session)
    settled_sessions = []

    class FakeQtCore:
        class QEventLoop:
            class ProcessEventsFlag:
                AllEvents = object()

    class FakeApp:
        def __init__(self):
            self.calls = 0

        def processEvents(self, *_args):
            self.calls += 1
            if self.calls == 3:
                win._frame_session = new_session

    monkeypatch.setattr(
        workflow,
        "_wait_for_montage_complete_soft",
        lambda **_kwargs: settled_sessions.append(win._frame_session.session_id) or True,
    )
    app = FakeApp()

    assert (
        workflow._wait_for_montage_successor_settled(
            win=win,
            app=app,
            QtCore=FakeQtCore,
            predecessor_session_id=7,
            budget_s=bounded_interaction_settle_timeout_s(0.5),
        )
        is True
    )
    assert app.calls == 3
    assert settled_sessions == [8]


def test_profile_montage_level_state_uses_session_snapshot():
    from arrayscope.display.model.presentation_generation import PresentationGenerationTracker
    from arrayscope.tools.profile_montage_workflow import _montage_level_presentation_state

    snapshot = SimpleNamespace(
        revision=11,
        target_levels=(2.0, 8.0),
        stale_count=3,
        pending_count=4,
        settled=False,
        active_tile_count=7,
        active_presented_tile_count=5,
    )
    level_generation = PresentationGenerationTracker()
    level_generation.tile_values = {0: (0.0, 1.0), 1: (2.0, 8.0), 2: (2.0, 8.0)}
    level_generation.set_active_tiles((0, 1, 2))
    session = SimpleNamespace(
        level_generation=level_generation, level_presentation_snapshot=lambda: snapshot
    )
    win = SimpleNamespace(_frame_session=session)

    state = _montage_level_presentation_state(win)

    assert state["settled"] is False
    assert state["pending"] is True
    assert state["revision"] == 11
    assert state["target_levels"] == [2.0, 8.0]
    assert state["stale_tiles"] == 3
    assert state["pending_tiles"] == 4
    assert state["active_level_value_count"] == 2
    assert state["active_tile_count"] == 7
    assert state["active_presented_tile_count"] == 5


def test_profile_timing_detects_immediate_level_work():
    from arrayscope.tools.profile_montage_workflow import _timing_has_level_work

    assert _timing_has_level_work(SimpleNamespace(tile_layer_shader_uniform_updates=2)) is True
    assert _timing_has_level_work(SimpleNamespace(tile_layer_level_updates=1)) is True
    assert _timing_has_level_work(SimpleNamespace(tile_layer_texture_uploads=3)) is False
    assert _timing_has_level_work(None) is False


def test_profile_montage_completion_drains_final_scheduled_wgpu_draw(monkeypatch):
    import arrayscope.tools.profile_montage_workflow as workflow
    from arrayscope.display.model.presentation_generation import PresentationGenerationTracker
    from arrayscope.operations.stage_fanin import StageFanInState

    class FakeQtCore:
        class QEventLoop:
            class ProcessEventsFlag:
                AllEvents = object()

        class Qt:
            class TimerType:
                PreciseTimer = object()

        class QDeadlineTimer:
            def __init__(self, *_args):
                pass

    class FakeImageView:
        def __init__(self):
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
            return "wgpu_tile_layer"

        def montageTileOverlayCount(self):
            return 1

        def presentation_diagnostics(self):
            return dict(self.diagnostics)

    target_state = {"settled": True}

    class FakeApp:
        def __init__(self, image_view):
            self.image_view = image_view
            self.calls = 0

        def processEvents(self, *_args):
            self.calls += 1
            self.image_view.diagnostics["draw_count"] = self.calls

    image_view = FakeImageView()
    session = SimpleNamespace(
        visible_tiles=(SimpleNamespace(montage_index=0), SimpleNamespace(montage_index=1)),
        skipped_tiles=set(),
        lifecycle=SimpleNamespace(presented_tiles=frozenset({0, 1})),
        display_committed=True,
        required_target_unsettled_tiles=lambda: (),
        loading_tiles=set(),
        active_tile_requests=set(),
        stage_fan_in=StageFanInState(),
        final_commit_pending=False,
        flush_pending=False,
        dirty_payloads={},
        pending_removals=set(),
        level_generation=PresentationGenerationTracker(),
        is_complete=lambda: True,
        required_target_settled=lambda: target_state["settled"],
    )
    session.level_presentation_snapshot = lambda: session.level_generation.snapshot()
    session.has_pending_level_update = lambda: False
    win = SimpleNamespace(img_view=image_view, _frame_session=session)
    app = FakeApp(image_view)
    drain_calls = []

    def drain_final_draw(*_args, **kwargs):
        drain_calls.append(kwargs)
        image_view.diagnostics["tile_presentation_draw_count"] = 4
        image_view.diagnostics["tile_presentation_draw_pending"] = False

    monkeypatch.setattr(workflow, "_wait_for_tile_presentation_draw", drain_final_draw)

    result = workflow._wait_for_montage_complete(
        app,
        FakeQtCore,
        win,
        timeout_s=bounded_interaction_settle_timeout_s(0.5),
        start=0.0,
        draw_start=0,
    )

    assert drain_calls == [{"timeout_s": 0.5, "target_s": 0.5}]
    assert result["active_presented_tile_count"] == 2
    assert result["active_planned_tile_count"] == 2
    assert result["fully_visible_ms"] is not None
    assert result["tile_presentation_draw_count"] == 4
    assert result["required_target_settled"] is True


def test_profile_montage_visibility_ignores_offscreen_unsettled_targets():
    from arrayscope.operations.stage_fanin import StageFanInState
    from arrayscope.tools.profile_montage_workflow import _montage_visibility_state

    session = SimpleNamespace(
        visible_tiles=(SimpleNamespace(montage_index=0), SimpleNamespace(montage_index=1)),
        skipped_tiles=set(),
        lifecycle=SimpleNamespace(presented_tiles=frozenset({0, 1})),
        display_committed=True,
        level_expected_indices=(10, 11),
        required_target_unsettled_tiles=lambda: (5,),
        loading_tiles=set(),
        active_tile_requests=set(),
        stage_fan_in=StageFanInState(),
        final_commit_pending=False,
        flush_pending=False,
        dirty_payloads={},
        pending_payload_upserts={},
        pending_removals=set(),
    )

    class FakeImageView:
        def montageTileOverlayCount(self):
            return 0

        def presentation_diagnostics(self):
            return {}

    win = SimpleNamespace(img_view=FakeImageView(), _frame_session=session)

    state = _montage_visibility_state(win, mode="wgpu_tile_layer")

    assert state["fully_visible"] is True
    assert state["visible_target_unsettled_tiles"] == 0
    assert state["active_presented_tile_count"] == 2

    session.required_target_unsettled_tiles = lambda: (1,)
    state = _montage_visibility_state(win, mode="wgpu_tile_layer")

    assert state["fully_visible"] is False
    assert state["visible_target_unsettled_tiles"] == 1

    session.required_target_unsettled_tiles = lambda: (5,)
    session.atomic_successor_pending = True
    state = _montage_visibility_state(win, mode="wgpu_tile_layer")

    assert state["fully_visible"] is False
    assert state["atomic_successor_pending"] is True


def test_profile_montage_visibility_is_viewport_scoped_when_selection_is_larger():
    from arrayscope.operations.stage_fanin import StageFanInState
    from arrayscope.tools.profile_montage_workflow import _montage_visibility_state

    session = SimpleNamespace(
        visible_tiles=tuple(SimpleNamespace(montage_index=index) for index in range(44)),
        skipped_tiles=set(),
        lifecycle=SimpleNamespace(presented_tiles=frozenset(range(44))),
        display_committed=True,
        level_expected_indices=tuple(range(60)),
        required_target_unsettled_tiles=lambda: (58,),
        loading_tiles=set(),
        active_tile_requests=set(),
        stage_fan_in=StageFanInState(),
        final_commit_pending=False,
        flush_pending=False,
        dirty_payloads={},
        pending_payload_upserts={},
        pending_removals=set(),
    )

    class FakeImageView:
        def montageTileOverlayCount(self):
            return 0

        def presentation_diagnostics(self):
            return {}

    win = SimpleNamespace(img_view=FakeImageView(), _frame_session=session)

    state = _montage_visibility_state(win, mode="wgpu_tile_layer")

    assert state["fully_visible"] is True
    assert state["active_presented_tile_count"] == 44
    assert state["active_planned_tile_count"] == 44
    assert state["requested_tile_count"] == 60
    assert state["visible_target_unsettled_tiles"] == 0


@pytest.mark.skipif(
    os.environ.get("ARRAYSCOPE_RUN_PY_SPY_SMOKE") != "1",
    reason="opt-in real py-spy workflow smoke; set ARRAYSCOPE_RUN_PY_SPY_SMOKE=1",
)
def test_py_spy_smoke_profile_workflow_exits_cleanly(tmp_path):
    from arrayscope.tools.profile_montage_workflow import DEFAULT_DATA_PATH

    py_spy = shutil.which("py-spy")
    if py_spy is None:
        pytest.skip("py-spy executable not found")
    if not DEFAULT_DATA_PATH.exists():
        pytest.skip(f"profile dataset not found: {DEFAULT_DATA_PATH}")
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        pytest.skip("onscreen py-spy smoke requires a real display")
    if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
        pytest.skip("onscreen py-spy smoke cannot run with QT_QPA_PLATFORM=offscreen")

    raw = tmp_path / "workflow.raw"
    jsonl = tmp_path / "workflow.jsonl"
    completed = subprocess.run(
        (
            py_spy,
            "record",
            "--format",
            "raw",
            "--rate",
            "50",
            "--nonblocking",
            "--gil",
            "-o",
            str(raw),
            "--",
            sys.executable,
            "-m",
            "arrayscope.tools.profile_montage_workflow",
            "--backend",
            "pyqtgraph",
            "--timeout-s",
            "5",
            "--jsonl",
            str(jsonl),
            "--profiler-type",
            "py-spy-raw",
            "--profiler-artifact",
            str(raw),
        ),
        check=False,
        text=True,
        capture_output=True,
        # Whole profiler-child deadlock guard.  The workflow itself hard-fails
        # each user-visible step after the shared five-second limit.
        timeout=45,
    )

    assert completed.returncode == 0, completed.stderr
    assert raw.exists()
    assert raw.stat().st_size > 0
    assert jsonl.exists()
    assert jsonl.stat().st_size > 0
    records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    by_phase = {record["phase"]: record for record in records}
    assert (
        by_phase["raw_full_tiled_montage"]["tile_count"]
        == by_phase["raw_full_tiled_montage"]["full_tile_count"]
    )
    assert by_phase["raw_full_tiled_montage"]["tile_cap_applied"] is False
    assert (
        by_phase["raw_full_tiled_montage"]["active_presented_tile_count"]
        == by_phase["raw_full_tiled_montage"]["requested_tile_count"]
    )
    assert (
        by_phase["fft_full_tiled_montage"]["tile_count"]
        == by_phase["fft_full_tiled_montage"]["full_tile_count"]
    )
    assert by_phase["fft_full_tiled_montage"]["tile_cap_applied"] is False
    assert (
        by_phase["fft_full_tiled_montage"]["active_presented_tile_count"]
        == by_phase["fft_full_tiled_montage"]["requested_tile_count"]
    )
    assert "Signal source has been deleted" not in completed.stderr


@pytest.mark.skipif(
    os.environ.get("ARRAYSCOPE_RUN_PROFILE_WORKFLOW") != "1",
    reason="opt-in realistic GUI profiling workflow; set ARRAYSCOPE_RUN_PROFILE_WORKFLOW=1",
)
def test_profile_montage_workflow_realistic_dataset_optional(tmp_path):
    from arrayscope.tools.profile_montage_workflow import (
        DEFAULT_DATA_PATH,
        PROFILE_MONTAGE_STAGES,
        run_profile_montage_workflow,
    )

    data_path = Path(os.environ.get("ARRAYSCOPE_PROFILE_DATA", DEFAULT_DATA_PATH))
    if not data_path.exists():
        pytest.skip(f"profile dataset not found: {data_path}")

    backends = tuple(
        backend.strip()
        for backend in os.environ.get("ARRAYSCOPE_PROFILE_BACKENDS", "wgpu,pyqtgraph").split(",")
        if backend.strip()
    )
    timeout_s = bounded_interaction_settle_timeout_s(
        float(os.environ.get("ARRAYSCOPE_PROFILE_TIMEOUT_S", "5"))
    )
    max_tiles_raw = int(os.environ.get("ARRAYSCOPE_PROFILE_MAX_TILES", "0"))
    max_tiles = None if max_tiles_raw <= 0 else max_tiles_raw
    jsonl = tmp_path / "profile-workflow.jsonl"
    all_records = []

    for backend in backends:
        records = run_profile_montage_workflow(
            data_path=data_path,
            backend=backend,
            jsonl=jsonl,
            timeout_s=bounded_interaction_settle_timeout_s(timeout_s),
            max_tiles=max_tiles,
        )
        all_records.extend(records)

    phases = {(record["backend"], record["phase"]) for record in all_records}
    for backend in backends:
        for phase in PROFILE_MONTAGE_STAGES:
            assert (backend, phase) in phases
    assert jsonl.exists()


def test_session_fixture_shape_mismatch_raises_actionable_error(tmp_path):
    import numpy as np

    from arrayscope.core.view_session import (
        loads_session,
        metadata_for_file,
        save_session_file,
        settings_key_for_metadata,
    )
    from arrayscope.tools.profile_montage_workflow import (
        DEFAULT_SESSION_FIXTURE,
        _install_profile_session_fixture,
    )

    class _Settings:
        def fileName(self):
            return str(tmp_path / "settings" / "profile.conf")

        def setValue(self, key, value):
            pass

        def sync(self):
            pass

    with pytest.raises(ValueError) as excinfo:
        _install_profile_session_fixture(
            None,
            data_path=Path("small.npy"),
            data=np.zeros((64, 64, 12), dtype=np.float32),
            session_fixture=DEFAULT_SESSION_FIXTURE,
            settings=_Settings(),
            loads_session=loads_session,
            metadata_for_file=metadata_for_file,
            save_session_file=save_session_file,
            settings_key_for_metadata=settings_key_for_metadata,
        )

    message = str(excinfo.value)
    assert "does not fit dataset shape" in message
    assert "--session-fixture ''" in message


def test_profile_session_fixture_uses_file_view_session_directory_owner(
    qt_app, tmp_path, monkeypatch
):
    from dataclasses import dataclass

    from pyqtgraph.Qt import QtCore

    from arrayscope.tools.profile_montage_workflow import _install_profile_session_fixture

    @dataclass(frozen=True)
    class _Session:
        metadata: object

    fixture = tmp_path / "profile-session.json"
    fixture.write_text("{}", encoding="utf-8")
    owned_directory = tmp_path / "owned-file-view-sessions"
    written_directories = []

    def save_session_file(config_dir, _session):
        written_directories.append(Path(config_dir))
        return Path(config_dir) / "stored-session.json"

    assert qt_app.applicationName() == "ArrayScopeTests"
    monkeypatch.setattr(
        "arrayscope.window.file_view_session._file_view_session_config_dir",
        lambda: owned_directory,
    )
    settings = QtCore.QSettings()

    _install_profile_session_fixture(
        QtCore,
        data_path=tmp_path / "data.npy",
        data=object(),
        session_fixture=fixture,
        settings=settings,
        loads_session=lambda _text, _shape: _Session(metadata=None),
        metadata_for_file=lambda _path, **_kwargs: "metadata",
        save_session_file=save_session_file,
        settings_key_for_metadata=lambda _metadata: "profile-session-key",
    )

    assert written_directories == [owned_directory]


def test_montage_work_in_flight_counts_semantic_evidence_owner():
    from arrayscope.tools.profile_montage_workflow import _montage_work_in_flight

    progress = SimpleNamespace(inflight_generation=("levels", 7))
    session = SimpleNamespace(
        stage_fan_in=None,
        active_tile_requests=set(),
        pending_rung_materializations=(),
        level_evidence_inflight=False,
        semantic_level_evidence_progress=progress,
        histogram_aggregate_inflight=False,
    )

    assert _montage_work_in_flight(session) is True
    progress.inflight_generation = None
    assert _montage_work_in_flight(session) is False


def test_post_visible_gate_blockers_names_stuck_completion_gates():
    from arrayscope.tools.profile_montage_workflow import _post_visible_gate_blockers

    # Progress states never report blockers.
    assert not _post_visible_gate_blockers(
        fully_visible=False,
        requested_grid_visible=False,
        physical_drawn=False,
        presentation_ready=False,
        target_settled=False,
        work_in_flight=False,
        dirty_payloads=False,
    )
    assert not _post_visible_gate_blockers(
        fully_visible=True,
        requested_grid_visible=True,
        physical_drawn=False,
        presentation_ready=True,
        target_settled=False,
        work_in_flight=True,
        dirty_payloads=False,
    )
    assert not _post_visible_gate_blockers(
        fully_visible=True,
        requested_grid_visible=True,
        physical_drawn=False,
        presentation_ready=True,
        target_settled=False,
        work_in_flight=False,
        dirty_payloads=True,
    )

    # A visible frame with nothing in flight names exactly the stuck gates.
    assert _post_visible_gate_blockers(
        fully_visible=True,
        requested_grid_visible=True,
        physical_drawn=False,
        presentation_ready=False,
        target_settled=False,
        work_in_flight=False,
        dirty_payloads=False,
    ) == ("physical_drawn", "presentation_settled", "required_target_settled")

    # A fully unblocked frame reports nothing (the loop returns success).
    assert not _post_visible_gate_blockers(
        fully_visible=True,
        requested_grid_visible=True,
        physical_drawn=True,
        presentation_ready=True,
        target_settled=True,
        work_in_flight=False,
        dirty_payloads=False,
    )


def test_physical_tile_timeline_reports_draw_rate_without_settlement_gates():
    from arrayscope.tools.profile_montage_workflow import _physical_tile_timeline_metrics

    result = _physical_tile_timeline_metrics(
        (
            {
                "timestamp_ns": 1_050_000_000,
                "tile_count": 100,
                "presentation_identity": ((0, "old"),),
            },
            {
                "timestamp_ns": 1_100_000_000,
                "tile_count": 100,
                "presentation_identity": ((0, "new"),),
                "lod_counts": {"4": 1},
            },
            {
                "timestamp_ns": 1_300_000_000,
                "tile_count": 100,
                "presentation_identity": tuple((tile, "new") for tile in range(50)),
                "lod_counts": {"4": 50},
            },
            {
                "timestamp_ns": 1_500_000_000,
                "tile_count": 100,
                "presentation_identity": tuple((tile, "new") for tile in range(100)),
                "lod_counts": {"4": 100},
            },
        ),
        phase_start_s=1.0,
        requested_tiles=100,
        target_presentation_identity=tuple((tile, "new") for tile in range(100)),
    )

    assert result["physical_tile_first_ms"] == 100.0
    assert result["physical_tile_full_ms"] == 500.0
    assert result["physical_tile_rate_after_first_per_s"] == 247.5
    assert result["physical_tile_milestone_ms"] == {
        "25": 300.0,
        "50": 300.0,
        "75": 500.0,
        "100": 500.0,
    }
    assert "semantic evidence" in result["physical_tile_timeline_scope"]


def test_repeat_spread_summary_reports_every_run_and_the_median():
    """A `--repeat` batch must show the spread, not one run as "the number".

    The reference machine's raw montage stage covers 4.0-4.9 s, so a single
    elapsed value cannot support or refute a sub-0.5 s change.
    """

    from arrayscope.tools.profile_montage_workflow import _workflow_repeat_spread_summary

    records = tuple(
        {
            "backend": "wgpu",
            "phase": "raw_full_tiled_montage",
            "repeat_index": index,
            "elapsed_ms": elapsed,
        }
        for index, elapsed in enumerate((4412.5, 4216.1, 4436.1))
    )

    summary = _workflow_repeat_spread_summary(records)

    assert "Repeat spread over 3 runs" in summary
    assert "4412.5, 4216.1, 4436.1" in summary  # per run, in run order
    assert "| 3 | 4412.5 | 4216.1 | 4436.1 |" in summary  # runs, median, min, max


def test_repeat_spread_summary_is_silent_for_a_single_pass():
    from arrayscope.tools.profile_montage_workflow import _workflow_repeat_spread_summary

    records = ({"backend": "wgpu", "phase": "raw", "repeat_index": 0, "elapsed_ms": 1.0},)
    assert _workflow_repeat_spread_summary(records) == ""
    assert _workflow_repeat_spread_summary(()) == ""


def test_repeat_spread_summary_groups_backends_and_phases_separately():
    from arrayscope.tools.profile_montage_workflow import _workflow_repeat_spread_summary

    records = tuple(
        {
            "backend": backend,
            "phase": phase,
            "repeat_index": index,
            "elapsed_ms": 100.0 * (index + 1),
        }
        for index in (0, 1)
        for backend in ("wgpu", "pyqtgraph")
        for phase in ("load_data", "raw_full_tiled_montage")
    )

    lines = [line for line in _workflow_repeat_spread_summary(records).splitlines() if "`" in line]

    assert len(lines) == 4
    assert all("| 2 | 150.0 | 100.0 | 200.0 |" in line for line in lines)
