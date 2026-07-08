import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


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


def test_profile_workflow_forces_backend_specific_themes():
    from arrayscope.app.settings_state import AppSettingsState, ImageRenderingBackendChoice
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

    assert pyqtgraph.theme == ThemeChoice.LIGHT
    assert vispy.theme == ThemeChoice.DARK


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


def test_profile_fit_stretch_pulse_uses_window_fit_command():
    from arrayscope.tools.profile_montage_workflow import _pulse_fit_stretch

    calls = []
    win = SimpleNamespace(fit_image_to_view=lambda enabled: calls.append(bool(enabled)))

    assert _pulse_fit_stretch(win) is True

    assert calls == [True, False]


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


def test_profile_montage_completion_waits_for_level_generation_when_requested():
    from arrayscope.display.model.presentation_generation import PresentationGenerationTracker
    from arrayscope.operations.stage_fanin import StageFanInState
    from arrayscope.tools.profile_montage_workflow import _wait_for_montage_complete

    class FakeQtCore:
        class QEventLoop:
            class ProcessEventsFlag:
                AllEvents = object()

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
    )
    session.level_presentation_snapshot = lambda: session.level_generation.snapshot()
    session.has_pending_level_update = lambda: not session.level_presentation_snapshot().settled
    win = SimpleNamespace(img_view=FakeImageView(), _montage_session=session)

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
    win = SimpleNamespace(_montage_session=session)

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
    )
    session.level_presentation_snapshot = lambda: session.level_generation.snapshot()
    session.has_pending_level_update = lambda: False
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
    assert result["fully_visible_ms"] is not None
    assert result["vispy_tile_presentation_draw_count"] == 4


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
            "180",
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
        )
        all_records.extend(records)

    phases = {(record["backend"], record["phase"]) for record in all_records}
    for backend in backends:
        assert (backend, "load_data") in phases
        assert (backend, "raw_full_tiled_montage") in phases
        assert (backend, "fft_full_tiled_montage") in phases
    assert jsonl.exists()
