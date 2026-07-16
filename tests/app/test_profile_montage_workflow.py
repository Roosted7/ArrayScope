import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import shlex
from types import SimpleNamespace

import pytest


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
                "physical_draw_world_bounds": (1.0, 2.0, 5.0, 6.0),
                "physical_expected_world_rect": (1.0, 2.0, 5.0, 6.0),
                "physical_draw_bounds_match_layout": True,
            },
            9: {"physical_draw_bounds_match_layout": False},
        },
        {3},
    )

    assert rows == {
        "3": {
            "draw_world_rects": ((1.0, 2.0, 5.0, 6.0),),
            "draw_world_bounds": (1.0, 2.0, 5.0, 6.0),
            "expected_world_rect": (1.0, 2.0, 5.0, 6.0),
            "bounds_match_layout": True,
        }
    }


def test_synthetic_geometry_scene_is_indexed_and_spatially_recognizable():
    import numpy as np

    from arrayscope.tools.profile_montage_workflow import _synthetic_profile_data

    data = _synthetic_profile_data("geometry", (48, 64, 8))

    assert data.shape == (48, 64, 8)
    assert data.dtype == np.float32
    assert data.flags.c_contiguous
    assert float(data.min()) >= 0.0 and float(data.max()) <= 1.0
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

    commands = profiler_suite_commands(("--backend", "vispy", "--profile-suite", str(tmp_path)), tmp_path)

    assert {item["profiler_type"] for item in commands} == {"plain", "py-spy-raw-low-impact", "py-spy-raw-full", "perf-record"}
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
        ("--backend", "vispy", "--profile-suite", str(tmp_path), "--include-cprofile"),
        tmp_path,
    )
    by_type = {item["profiler_type"]: item for item in commands}

    assert "cprofile" in by_type
    assert "cProfile" in by_type["cprofile"]["command"]
    assert "--include-cprofile" not in by_type["plain"]["command"]


def test_profile_parser_stages_resolve_and_deconflict():
    from arrayscope.tools.profile_montage_workflow import _parse_stage_flags, _resolve_profile_stages

    stages = _resolve_profile_stages(
        include_stages=_parse_stage_flags(("raw_full_tiled_montage,montage_zoompan_fft", "montage_zoompan_fft", "montage_scroll_scalar")),
        skip_stages=_parse_stage_flags(("montage_zoompan_fft",)),
    )
    assert stages == ("raw_full_tiled_montage", "montage_scroll_scalar")


def test_profile_stage_resolve_defaults_to_all():
    from arrayscope.tools.profile_montage_workflow import _resolve_profile_stages, PROFILE_MONTAGE_STAGES

    assert _resolve_profile_stages() == tuple(PROFILE_MONTAGE_STAGES)


def test_profile_parser_unknown_stage_is_rejected():
    from arrayscope.tools.profile_montage_workflow import _resolve_profile_stages

    with pytest.raises(ValueError, match="unknown montage workflow stage"):
        _resolve_profile_stages(include_stages=("not-a-phase",))


def test_profile_suite_commands_preserve_stage_filter_flags(tmp_path):
    from arrayscope.tools.profile_montage_workflow import profiler_suite_commands

    commands = profiler_suite_commands(
        ("--backend", "vispy", "--profile-suite", str(tmp_path), "--stages", "raw_full_tiled_montage,montage_zoompan_fft", "--skip-stages", "montage_scroll_scalar"),
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
    default_args = parser.parse_args(["--backend", "vispy"])
    custom_args = parser.parse_args(
        ["--backend", "vispy", "--scroll-max-tiles", "84", "--verbose-tile-trace"]
    )

    assert default_args.scroll_max_tiles == 60
    assert custom_args.scroll_max_tiles == 84
    assert default_args.verbose_tile_trace is False
    assert custom_args.verbose_tile_trace is True
    assert Path(default_args.session_fixture) == DEFAULT_SESSION_FIXTURE


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
            "vispy",
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

    commands = profiler_suite_commands(("--backend", "all", "--profile-suite", str(tmp_path), "--include-cprofile"), tmp_path)

    by_step = {item["step_id"]: item for item in commands}
    for backend in ("pyqtgraph", "vispy"):
        assert by_step[f"cprofile:{backend}"]["backend"] == backend
        assert by_step[f"py-spy-raw-low-impact:{backend}"]["backend"] == backend
        assert by_step[f"py-spy-raw-full:{backend}"]["backend"] == backend
        assert by_step[f"perf-record:{backend}"]["backend"] == backend
        assert f"--backend {backend}" in by_step[f"py-spy-raw-low-impact:{backend}"]["command"]
        assert f".{backend}." in by_step[f"py-spy-raw-full:{backend}"]["jsonl"]
    assert by_step["plain"]["backend"] == "all"


def test_profile_suite_can_opt_into_native_py_spy_without_passing_suite_flag_to_child(tmp_path):
    from arrayscope.tools.profile_montage_workflow import profiler_suite_commands

    commands = profiler_suite_commands(
        ("--backend", "vispy", "--profile-suite", str(tmp_path), "--py-spy-native"),
        tmp_path,
    )
    by_type = {item["profiler_type"]: item for item in commands}

    assert "py-spy-raw-low-impact-native" in by_type
    assert "py-spy-raw-full-native" in by_type
    assert "--native" in by_type["py-spy-raw-low-impact-native"]["command"]
    assert "--native" in by_type["py-spy-raw-full-native"]["command"]
    assert "--profile-suite" not in by_type["plain"]["command"]


def test_profile_workflow_preserves_theme_while_forcing_backend_and_resident_policy():
    from arrayscope.app.settings_state import AppSettingsState, ImageRenderingBackendChoice, MontageQualityPolicyChoice
    from arrayscope.app.theme import ThemeChoice
    from arrayscope.tools.profile_montage_workflow import _replace_settings

    pyqtgraph = _replace_settings(
        AppSettingsState(),
        backend="pyqtgraph",
        image_choice=ImageRenderingBackendChoice,
    )
    vispy = _replace_settings(
        AppSettingsState(),
        backend="vispy",
        image_choice=ImageRenderingBackendChoice,
    )

    assert pyqtgraph.theme == ThemeChoice.SYSTEM
    assert pyqtgraph.montage_quality_policy == MontageQualityPolicyChoice.RESIDENT
    assert vispy.theme == ThemeChoice.SYSTEM
    assert vispy.montage_quality_policy == MontageQualityPolicyChoice.RESIDENT


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
    metrics = {}
    win = SimpleNamespace(fit_image_to_view=lambda enabled: calls.append(bool(enabled)))

    assert _pulse_fit_stretch(win, metrics=metrics) is True

    assert calls == [True, False]
    assert metrics["fit_stretch_total_ms"] >= 0.0
    assert metrics["fit_stretch_enable_call_ms"] >= 0.0
    assert metrics["fit_stretch_disable_call_ms"] >= 0.0


def _passing_r8_phase_record(*, backend="vispy"):
    evidence_quality = 1 if backend == "vispy" else 3
    return {
        "phase": "raw_full_tiled_montage",
        "backend": backend,
        "profiler_type": "plain",
        "pacing_evidence": True,
        "complete": True,
        "requested_grid_fully_visible": True,
        "requested_tile_count": 272,
        "active_planned_tile_count": 272,
        "active_presented_tile_count": 272,
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
        "presentation_continuity_ok": True,
        "presentation_blackout_observed": False,
        "presentation_minimum_retained_tile_count": 100,
        "presentation_extent_changed_before_commit": False,
        "session_viewport_shape_matches": True,
        "viewport_shape": [753, 1245],
        "session_viewport_shape_target": [753, 1245],
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
        "grid_tile_count": 272,
        "full_tile_count": 272,
        "tile_cap_applied": False,
        "phase_recent_ui_work_observations": [{"elapsed_ms": 7.0}],
        "phase_recent_ui_work_observations_truncated": False,
        "action_render_call_ms": 4.0,
        "event_loop_max_gap_ms": 12.0,
        "physical_draw_after_complete_ms": 20.0,
        "waited_for_pyqtgraph_draw_after_complete": backend == "pyqtgraph",
        "pyqtgraph_draw_pending_after_complete": False,
    }


def test_r8_certification_passes_complete_semantic_and_responsive_phase():
    from arrayscope.tools.profile_montage_workflow import _r8_certification

    result = _r8_certification(_passing_r8_phase_record())

    assert result["r8_gate_applicable"] is True
    assert result["r8_performance_evidence"] is True
    assert result["r8_gate_passed"] is True
    assert result["r8_gate_failures"] == []


def test_r8_certification_names_first_pixel_and_latency_failures():
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
        "gui_callbacks_below_50ms",
        "event_loop_heartbeat",
    }.issubset(failed)


def test_r8_certification_reports_but_does_not_gate_cold_fill_heartbeat():
    from arrayscope.tools.profile_montage_workflow import _r8_certification

    record = _passing_r8_phase_record()
    record["event_loop_max_gap_ms"] = 90.0

    result = _r8_certification(record)

    assert result["r8_heartbeat_gate_applicable"] is False
    assert result["r8_heartbeat_max_gap_ms"] == 90.0
    assert result["r8_gate_passed"] is True


def test_r8_certification_skips_timing_only_for_profiled_or_smoke_runs():
    from arrayscope.tools.profile_montage_workflow import _r8_certification

    record = _passing_r8_phase_record()
    record.update(profiler_type="cprofile", action_render_call_ms=500.0, event_loop_max_gap_ms=500.0)

    result = _r8_certification(record)

    assert result["r8_performance_evidence"] is False
    assert result["r8_gate_passed"] is True


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
    monkeypatch.setattr(workflow, "_suite_tool_versions", lambda: {"python": "test", "py-spy": "py-spy 0.4.2"})
    monkeypatch.setattr(workflow, "_repository_state", lambda: {"repository_revision": "abc123", "repository_dirty": False})
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

    records = [json.loads(line) for line in (tmp_path / "suite-manifest.jsonl").read_text(encoding="utf-8").splitlines()]
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
    monkeypatch.setattr(workflow, "_repository_state", lambda: {"repository_revision": "abc123", "repository_dirty": False})
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

    records = [json.loads(line) for line in (tmp_path / "suite-manifest.jsonl").read_text(encoding="utf-8").splitlines()]
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
    monkeypatch.setattr(workflow, "_suite_tool_versions", lambda: {"python": "test", "py-spy": "py-spy 0.4.2"})
    monkeypatch.setattr(workflow, "_repository_state", lambda: {"repository_revision": "abc123", "repository_dirty": False})
    monkeypatch.setattr(workflow.shutil, "which", lambda exe: f"/usr/bin/{exe}")

    def fake_run(_command, **kwargs):
        artifact.write_text("samples", encoding="utf-8")
        jsonl.write_text('{"phase":"load"}\n', encoding="utf-8")
        kwargs["stderr"].write("py-spy failed\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(workflow.subprocess, "run", fake_run)

    rc = workflow.run_profile_suite(("--backend", "pyqtgraph"), tmp_path)

    records = [json.loads(line) for line in (tmp_path / "suite-manifest.jsonl").read_text(encoding="utf-8").splitlines()]
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
    from arrayscope.tools.profile_montage_workflow import _profiler_log_diagnostics, _profiler_sample_issue

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
    from arrayscope.tools.profile_montage_workflow import _profiler_log_diagnostics, _profiler_sample_issue

    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_text(
        "py-spy> Sampling process 50 times a second. Press Control-C to exit.\n"
        "py-spy> Wrote raw flamegraph data. Samples: 20 Errors: 2\n",
        encoding="utf-8",
    )
    stderr.write_text("[WARN  py_spy] Failed to get stack trace from 123\n[WARN  py_spy] Failed to get stack trace from 456\n", encoding="utf-8")

    diagnostics = _profiler_log_diagnostics("py-spy-raw-full", stdout, stderr)

    assert diagnostics["sampling_complete"] is False
    assert _profiler_sample_issue("py-spy-raw-full", diagnostics) == "py-spy full profile missed more than 1 stack sample(s)"


def test_profile_base_record_marks_offscreen_or_capped_runs_as_smoke(monkeypatch):
    import numpy as np
    from arrayscope.tools.profile_montage_workflow import _base_record

    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    visible = _base_record(
        run_id="run",
        backend="vispy",
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
    hidden = {**visible, **_base_record(
        run_id="run",
        backend="vispy",
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
    )}

    assert visible["smoke_only"] is False
    assert visible["pacing_evidence"] is True
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
        backend="vispy",
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
    win._committed_display_frame = SimpleNamespace(key=SimpleNamespace(semantic_key="new"))
    probe.stop()

    record = probe.record()
    assert record["presentation_continuity_expected"] is True
    assert record["presentation_blackout_observed"] is True
    assert record["presentation_extent_changed_before_commit"] is True
    assert record["presentation_continuity_ok"] is False


def test_presentation_continuity_probe_accepts_atomic_successor_commit():
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

    predecessor = SimpleNamespace(key=SimpleNamespace(semantic_key="old"))
    image_view = SimpleNamespace(
        montageDisplayMode=lambda: "tile_layer",
        _montage_tile_layer=SimpleNamespace(states={0: SimpleNamespace(visible=True)}),
        _viewport_content_extent=(20, 40),
        getLevels=lambda: (-2.0, 8.0),
        getHistogramDataBounds=lambda: (-2.0, 8.0),
    )
    level_source = SimpleNamespace(rank=2, source_count=4, evidence_quality=1)
    win = SimpleNamespace(
        img_view=image_view,
        _committed_display_frame=predecessor,
        _frame_session=SimpleNamespace(applied_level_source=level_source),
        renderer=SimpleNamespace(_last_montage_level_decision=None, _montage_refined_level_applied_count=0),
    )
    probe = _PresentationContinuityProbe(SimpleNamespace(QTimer=FakeTimer), win)

    probe.start()
    win._committed_display_frame = SimpleNamespace(key=SimpleNamespace(semantic_key="new"))
    probe.stop()

    record = probe.record()
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
    assert 0 <= fake_calls["fast"]["low"] <= fake_calls["fast"]["high"] <= selected_count - window_size
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
    scroll_calls = []
    monkeypatch.setattr(workflow, "_scroll_montage_window", lambda *args, **kwargs: scroll_calls.append(kwargs))

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
    assert scroll_calls
    assert all(call["size"] == 60 and call["interactive"] is True for call in scroll_calls)
    assert all(0 < call["window_start"] < 212 for call in scroll_calls)


def test_profile_montage_completion_waits_for_level_generation_when_requested():
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
        pending_tiles=(),
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
        timeout_s=0.5,
        start=0.0,
        draw_start=0,
        require_presentation_settled=True,
    )

    assert app.calls >= 3
    assert result["presentation_settled"] is True
    assert result["stale_level_tiles"] == 0
    assert result["pending_level_tiles"] == 0
    assert result["level_revision"] == 1
    assert result["active_level_value_count"] == 1


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
    session = SimpleNamespace(level_generation=level_generation, level_presentation_snapshot=lambda: snapshot)
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


def test_profile_montage_completion_waits_for_fully_visible_vispy_draw():
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

    target_state = {"settled": False}

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
            if self.calls >= 3:
                target_state["settled"] = True

    image_view = FakeImageView()
    session = SimpleNamespace(
        visible_tiles=(SimpleNamespace(montage_index=0), SimpleNamespace(montage_index=1)),
        skipped_tiles=set(),
        lifecycle=SimpleNamespace(presented_tiles=frozenset({0, 1})),
        display_committed=True,
        pending_tiles=(),
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

    result = _wait_for_montage_complete(
        app,
        FakeQtCore,
        win,
        timeout_s=0.5,
        start=0.0,
        draw_start=0,
    )

    assert app.calls >= 3
    assert result["active_presented_tile_count"] == 2
    assert result["active_planned_tile_count"] == 2
    assert result["fully_visible_ms"] is not None
    assert result["vispy_tile_presentation_draw_count"] == 4
    assert result["required_target_settled"] is True


def test_profile_montage_visibility_ignores_offscreen_pending_tiles():
    from arrayscope.operations.stage_fanin import StageFanInState
    from arrayscope.tools.profile_montage_workflow import _montage_visibility_state

    session = SimpleNamespace(
        visible_tiles=(SimpleNamespace(montage_index=0), SimpleNamespace(montage_index=1)),
        skipped_tiles=set(),
        lifecycle=SimpleNamespace(presented_tiles=frozenset({0, 1})),
        display_committed=True,
        level_expected_indices=(10, 11),
        pending_tiles=(SimpleNamespace(montage_index=5),),
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

        def vispyPresentationDiagnostics(self):
            return {}

    win = SimpleNamespace(img_view=FakeImageView(), _frame_session=session)

    state = _montage_visibility_state(win, mode="vispy_tile_layer")

    assert state["fully_visible"] is True
    assert state["visible_pending_tiles"] == 0
    assert state["active_presented_tile_count"] == 2


def test_profile_montage_visibility_is_viewport_scoped_when_selection_is_larger():
    from arrayscope.operations.stage_fanin import StageFanInState
    from arrayscope.tools.profile_montage_workflow import _montage_visibility_state

    session = SimpleNamespace(
        visible_tiles=tuple(SimpleNamespace(montage_index=index) for index in range(44)),
        skipped_tiles=set(),
        lifecycle=SimpleNamespace(presented_tiles=frozenset(range(44))),
        display_committed=True,
        level_expected_indices=tuple(range(60)),
        pending_tiles=(SimpleNamespace(montage_index=58),),
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

        def vispyPresentationDiagnostics(self):
            return {}

    win = SimpleNamespace(img_view=FakeImageView(), _frame_session=session)

    state = _montage_visibility_state(win, mode="vispy_tile_layer")

    assert state["fully_visible"] is True
    assert state["active_presented_tile_count"] == 44
    assert state["active_planned_tile_count"] == 44
    assert state["requested_tile_count"] == 60
    assert state["visible_pending_tiles"] == 0


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
    assert raw.exists() and raw.stat().st_size > 0
    assert jsonl.exists() and jsonl.stat().st_size > 0
    records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    by_phase = {record["phase"]: record for record in records}
    assert by_phase["raw_full_tiled_montage"]["tile_count"] == by_phase["raw_full_tiled_montage"]["full_tile_count"]
    assert by_phase["raw_full_tiled_montage"]["tile_cap_applied"] is False
    assert by_phase["raw_full_tiled_montage"]["active_presented_tile_count"] == by_phase["raw_full_tiled_montage"]["requested_tile_count"]
    assert by_phase["fft_full_tiled_montage"]["tile_count"] == by_phase["fft_full_tiled_montage"]["full_tile_count"]
    assert by_phase["fft_full_tiled_montage"]["tile_cap_applied"] is False
    assert by_phase["fft_full_tiled_montage"]["active_presented_tile_count"] == by_phase["fft_full_tiled_montage"]["requested_tile_count"]
    assert "Signal source has been deleted" not in completed.stderr


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

    timeout_s = float(os.environ.get("ARRAYSCOPE_PROFILE_TIMEOUT_S", "5"))
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
        )
        all_records.extend(records)

    phases = {(record["backend"], record["phase"]) for record in all_records}
    for backend in backends:
        assert (backend, "load_data") in phases
        assert (backend, "raw_full_tiled_montage") in phases
        assert (backend, "fft_full_tiled_montage") in phases
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
