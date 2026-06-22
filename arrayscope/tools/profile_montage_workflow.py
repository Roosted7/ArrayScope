"""Profile a realistic full-montage workflow in a real ArrayScope window."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from time import perf_counter
from uuid import uuid4

import numpy as np


DEFAULT_DATA_PATH = Path("data/_WIPDelRec-tT2_20260223150234_14.nii")


def run_profile_montage_workflow(
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    backend: str = "pyqtgraph",
    jsonl: str | Path | None = None,
    timeout_s: float = 180.0,
    max_tiles: int | None = None,
    columns: int | None = None,
    load_mode: str = "app",
    show_window: bool = True,
    profiler_type: str = "plain",
    profiler_artifact_paths: tuple[str | Path, ...] = (),
) -> tuple[dict[str, object], ...]:
    """Run raw full montage, then FFT-over-montage-axis full montage.

    The function is intentionally suitable for wrapping with an external
    sampling profiler such as ``py-spy``.  Returned and JSONL records contain
    enough app diagnostics to correlate profiler stacks with UI-visible phases.
    """

    from arrayscope.app.qt_binding import prefer_pyside6

    prefer_pyside6()
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore

    from arrayscope.app.settings_state import (
        ImageRenderingBackendChoice,
        MontageDisplayBackendChoice,
    )
    from arrayscope.operations.pipeline import CenteredFFT
    from arrayscope.window import ArrayScopeWindow

    backend = _normalize_backend(backend)
    data_path = Path(data_path)
    run_id = uuid4().hex
    records: list[dict[str, object]] = []

    app = pg.mkQApp()
    settings = QtCore.QSettings()
    previous_image_backend = settings.value("image_rendering_backend", None)
    previous_montage_backend = settings.value("montage_display_backend", None)
    settings.setValue(
        "image_rendering_backend",
        ImageRenderingBackendChoice.VISPY.value if backend == "vispy" else ImageRenderingBackendChoice.PYQTGRAPH.value,
    )
    settings.setValue("montage_display_backend", MontageDisplayBackendChoice.TILE_LAYER.value)
    settings.sync()

    win = None
    try:
        load_start = perf_counter()
        data = _load_dataset(data_path, mode=load_mode)
        load_elapsed_ms = (perf_counter() - load_start) * 1000.0
        if np.ndim(data) < 3:
            raise ValueError(f"profile workflow requires at least 3 dimensions, got shape {np.shape(data)}")
        montage_axis = 2
        tile_count = int(np.shape(data)[montage_axis])
        indices = tuple(range(tile_count if max_tiles is None else min(tile_count, max(1, int(max_tiles)))))
        columns = _default_columns(len(indices)) if columns is None else max(1, int(columns))
        base = _base_record(
            run_id=run_id,
            backend=backend,
            data_path=data_path,
            data=data,
            load_mode=load_mode,
            montage_axis=montage_axis,
            indices=indices,
            full_tile_count=tile_count,
            columns=columns,
            show_window=show_window,
            max_tiles=max_tiles,
            profiler_type=profiler_type,
            profiler_artifact_paths=profiler_artifact_paths,
            qt_platform=str(app.platformName()),
        )
        _append_record(
            records,
            jsonl,
            {
                **base,
                "phase": "load_data",
                "elapsed_ms": load_elapsed_ms,
                "complete": True,
            },
        )

        win = ArrayScopeWindow(data)
        win.app_settings = _replace_settings(
            win.app_settings,
            backend=backend,
            image_choice=ImageRenderingBackendChoice,
            montage_choice=MontageDisplayBackendChoice,
        )
        win.resize(1400, 900)
        if show_window:
            win.show()
        _process_events(app, QtCore, count=20)
        probe = _EventLoopProbe(QtCore)
        probe.start()

        raw_state = win.view_state.with_image_axes(0, 1).with_montage_axis(
            montage_axis,
            columns=columns,
            indices=indices,
            text=":",
        )
        raw_record = _run_phase(
            app,
            QtCore,
            win,
            probe,
            phase="raw_full_tiled_montage",
            timeout_s=timeout_s,
            action=lambda: (win._set_view_state(raw_state), win.render(reason="profile-raw-full-montage")),
        )
        _append_record(records, jsonl, {**base, **raw_record})

        def apply_fft() -> None:
            win.operation_coordinator.load_operations((CenteredFFT(axis=montage_axis),))
            win._set_document(win.operation_coordinator.document)
            win._coerce_channel_for_current_dtype()
            fft_state = win.view_state.with_image_axes(0, 1).with_montage_axis(
                montage_axis,
                columns=columns,
                indices=indices,
                text=":",
            )
            win._set_view_state(fft_state)
            win.render(reason="profile-fft-full-montage")

        fft_record = _run_phase(
            app,
            QtCore,
            win,
            probe,
            phase="fft_full_tiled_montage",
            timeout_s=timeout_s,
            action=apply_fft,
        )
        _append_record(records, jsonl, {**base, **fft_record})
        return tuple(records)
    finally:
        if win is not None:
            win.close()
            _process_events(app, QtCore, count=10)
        _restore_setting(settings, "image_rendering_backend", previous_image_backend)
        _restore_setting(settings, "montage_display_backend", previous_montage_backend)
        settings.sync()


class _EventLoopProbe:
    def __init__(self, QtCore):
        self._timer = QtCore.QTimer()
        self._timer.setInterval(1)
        self._timer.timeout.connect(self._tick)
        self._last = perf_counter()
        self.max_gap_ms = 0.0
        self.tick_count = 0

    def start(self) -> None:
        self.reset()
        self._timer.start()

    def reset(self) -> None:
        self._last = perf_counter()
        self.max_gap_ms = 0.0
        self.tick_count = 0

    def _tick(self) -> None:
        now = perf_counter()
        self.max_gap_ms = max(self.max_gap_ms, (now - self._last) * 1000.0)
        self._last = now
        self.tick_count += 1


def _run_phase(app, QtCore, win, probe: _EventLoopProbe, *, phase: str, timeout_s: float, action) -> dict[str, object]:
    probe.reset()
    start = perf_counter()
    draw_start = _vispy_draw_count(win)
    action()
    milestones = _wait_for_montage_complete(
        app,
        QtCore,
        win,
        timeout_s=timeout_s,
        start=start,
        draw_start=draw_start,
    )
    elapsed_ms = (perf_counter() - start) * 1000.0
    _process_events(app, QtCore, count=5)
    record = _phase_record(win, phase=phase, elapsed_ms=elapsed_ms, event_loop_max_gap_ms=probe.max_gap_ms)
    record.update(milestones)
    record["event_loop_ticks"] = int(probe.tick_count)
    return record


def _wait_for_montage_complete(app, QtCore, win, *, timeout_s: float, start: float, draw_start: int) -> dict[str, float | int | bool | None]:
    deadline = time.monotonic() + float(timeout_s)
    first_loaded_ms = None
    first_display_committed_ms = None
    first_overlay_clear_ms = None
    saw_overlays = _montage_overlay_count(win) > 0
    first_logical_complete_ms = None
    draw_after_complete_ms = None
    fully_visible_ms = None
    fully_visible_tile_request_count = None
    final_visibility_state: dict[str, object] = {}
    while time.monotonic() < deadline:
        _process_events(app, QtCore, count=2)
        session = getattr(win, "_montage_session", None)
        mode = getattr(win.img_view, "montageDisplayMode", lambda: "")()
        vispy_tiled = str(mode) == "vispy_tile_layer" and _vispy_canvas_visible(win)
        if session is not None:
            if first_loaded_ms is None and bool(getattr(session, "presented_tiles", ())):
                first_loaded_ms = (perf_counter() - start) * 1000.0
            if first_display_committed_ms is None and bool(getattr(session, "display_committed", False)):
                first_display_committed_ms = (perf_counter() - start) * 1000.0
        overlay_count = _montage_overlay_count(win)
        saw_overlays = bool(saw_overlays or overlay_count > 0)
        if saw_overlays and first_overlay_clear_ms is None and overlay_count == 0:
            first_overlay_clear_ms = (perf_counter() - start) * 1000.0
        logical_complete = (
            session is not None
            and bool(getattr(session, "display_committed", False))
            and session.is_complete()
            and mode in {"tile_layer", "vispy_tile_layer", "canvas"}
        )
        if logical_complete and first_logical_complete_ms is None:
            first_logical_complete_ms = (perf_counter() - start) * 1000.0
        visibility_state = _montage_visibility_state(win, mode=str(mode))
        final_visibility_state = visibility_state
        fully_visible = bool(visibility_state["fully_visible"])
        if fully_visible and fully_visible_ms is None:
            fully_visible_ms = (perf_counter() - start) * 1000.0
            fully_visible_tile_request_count = _vispy_tile_presentation_request_count(win)
        current_request_count = _vispy_tile_presentation_request_count(win)
        final_drawn = _vispy_tile_presentation_draw_count(win) >= int(current_request_count)
        if fully_visible and (not vispy_tiled or final_drawn):
            if vispy_tiled:
                draw_after_complete_ms = (perf_counter() - start) * 1000.0
            return {
                "first_loaded_tile_ms": first_loaded_ms,
                "first_display_committed_ms": first_display_committed_ms,
                "first_overlay_clear_ms": first_overlay_clear_ms,
                "logical_complete_ms": first_logical_complete_ms,
                "draw_after_complete_ms": draw_after_complete_ms,
                "fully_visible_ms": fully_visible_ms,
                "active_presented_tile_count": int(visibility_state["active_presented_tile_count"]),
                "active_planned_tile_count": int(visibility_state["active_planned_tile_count"]),
                "requested_tile_count": int(visibility_state["requested_tile_count"]),
                "deferred_display_tile_count": int(visibility_state["deferred_display_tile_count"]),
                "vispy_draw_count_start": int(draw_start),
                "vispy_draw_count_complete": _vispy_draw_count(win),
                "vispy_tile_presentation_request_count": _vispy_tile_presentation_request_count(win),
                "vispy_tile_presentation_draw_count": _vispy_tile_presentation_draw_count(win),
                "waited_for_vispy_draw_after_complete": bool(vispy_tiled),
            }
        time.sleep(0.005)
    snapshot = win.collect_runtime_diagnostics()
    session = getattr(win, "_montage_session", None)
    raise TimeoutError(
        "timed out waiting for montage completion: "
        f"loaded={snapshot.montage.loaded_tiles} pending={snapshot.montage.pending_tiles} "
        f"loading={snapshot.montage.loading_tiles} deferred={snapshot.montage.deferred_display_tiles} "
        f"active={0 if session is None else len(getattr(session, 'active_tile_requests', ()) or ())} "
        f"completed={0 if session is None else len(getattr(session, 'pending_completed_tiles', ()) or ())} "
        f"stage_waiting={0 if session is None else sum(len(tiles) for tiles in getattr(session, 'stage_waiting_tiles', {}).values())} "
        f"lead_warmups={0 if session is None else len(getattr(session, 'lead_stage_warmups', {}) or {})} "
        f"active_presented={final_visibility_state.get('active_presented_tile_count', 0)}/"
        f"{final_visibility_state.get('active_planned_tile_count', 0)} "
        f"overlays={_montage_overlay_count(win)} vispy_draws={_vispy_draw_count(win)} "
        f"tile_draw={_vispy_tile_presentation_draw_count(win)}/{_vispy_tile_presentation_request_count(win)}"
    )


def _phase_record(win, *, phase: str, elapsed_ms: float, event_loop_max_gap_ms: float) -> dict[str, object]:
    snapshot = win.collect_runtime_diagnostics()
    timing = snapshot.montage_timing
    montage = snapshot.montage
    resource = snapshot.resource_governor
    recent_callbacks = () if resource is None else tuple(resource.recent_over_warning_callbacks)
    feedback_channels = () if resource is None else tuple(resource.feedback_channels)
    ui_decisions = () if resource is None else tuple(resource.ui_decisions)
    vispy = _vispy_presentation_diagnostics(win)
    return {
        "phase": phase,
        "elapsed_ms": float(elapsed_ms),
        "event_loop_max_gap_ms": float(event_loop_max_gap_ms),
        "complete": True,
        "image_backend_actual": str(snapshot.image_rendering_backend_actual),
        "montage_display_mode": str(montage.display_mode),
        "montage_backend_chosen": str(montage.backend_chosen),
        "montage_loaded_tiles": int(montage.loaded_tiles),
        "montage_loading_tiles": int(montage.loading_tiles),
        "montage_pending_tiles": int(montage.pending_tiles),
        "montage_deferred_display_tiles": int(montage.deferred_display_tiles),
        "montage_tile_compute_cache_hits": int(montage.tile_compute_cache_hits),
        "montage_tile_compute_stage_backed": int(montage.tile_compute_stage_backed),
        "montage_tile_compute_direct": int(montage.tile_compute_direct),
        "montage_tile_compute_waiting_for_stage": int(montage.tile_compute_waiting_for_stage),
        "montage_lead_direct_tiles": int(montage.lead_direct_tiles),
        "montage_stage_backed_tiles_pending": int(montage.stage_backed_tiles_pending),
        "montage_retained_stage_index": montage.retained_stage_index,
        "montage_retained_stage_decision": str(montage.retained_stage_decision),
        "montage_repeated_expensive_stage_per_tile": bool(montage.repeated_expensive_stage_per_tile),
        "last_render_sync_ms": _optional_float(snapshot.render_timing.last_render_sync_ms),
        "last_display_commit_ms": _optional_float(snapshot.render_timing.last_display_commit_ms),
        "last_canvas_commit_ms": _optional_float(timing.last_canvas_commit_ms),
        "last_tile_layer_upload_ms": _optional_float(timing.last_tile_layer_upload_ms),
        "last_tile_layer_rgb_window_ms": _optional_float(timing.last_tile_layer_rgb_window_ms),
        "last_overlay_update_ms": _optional_float(timing.last_overlay_update_ms),
        "tile_layer_visible_items": int(timing.tile_layer_visible_items),
        "tile_layer_items_updated": int(timing.tile_layer_items_updated),
        "tile_layer_items_skipped": int(timing.tile_layer_items_skipped),
        "tile_layer_texture_uploads": int(timing.tile_layer_texture_uploads),
        "tile_layer_texture_upload_bytes": int(timing.tile_layer_texture_upload_bytes),
        "tile_layer_vertex_uploads": int(timing.tile_layer_vertex_uploads),
        "tile_layer_level_updates": int(timing.tile_layer_level_updates),
        "tile_layer_estimated_gpu_bytes": int(timing.tile_layer_estimated_gpu_bytes),
        "tile_layer_page_count": int(timing.tile_layer_page_count),
        "tile_layer_active_pages": int(timing.tile_layer_active_pages),
        "montage_overlay_count": _montage_overlay_count(win),
        "vispy_draw_count": int(vispy.get("draw_count", 0)),
        "vispy_tile_presentation_request_count": int(vispy.get("tile_presentation_request_count", 0)),
        "vispy_tile_presentation_draw_count": int(vispy.get("tile_presentation_draw_count", 0)),
        "vispy_tile_presentation_draw_pending": bool(vispy.get("tile_presentation_draw_pending", False)),
        "vispy_presented_tile_count": int(vispy.get("presented_tile_count", 0)),
        "vispy_presented_tiles": list(vispy.get("presented_tiles", ()) or ()),
        "vispy_tile_visual_visible_pages": int(vispy.get("tile_visual_visible_pages", 0)),
        "vispy_tile_visual_min_order": vispy.get("tile_visual_min_order"),
        "vispy_overlay_visual_visible_items": int(vispy.get("overlay_visual_visible_items", 0)),
        "vispy_overlay_visual_max_order": vispy.get("overlay_visual_max_order"),
        "vispy_overlays_above_tiles": bool(vispy.get("overlays_above_tiles", False)),
        "resource_feedback_channels": [asdict(channel) for channel in feedback_channels],
        "resource_ui_decisions": [asdict(decision) for decision in ui_decisions],
        "recent_over_warning_callbacks": [asdict(callback) for callback in recent_callbacks],
    }


def _vispy_presentation_diagnostics(win) -> dict[str, object]:
    getter = getattr(getattr(win, "img_view", None), "vispyPresentationDiagnostics", None)
    if callable(getter):
        try:
            return dict(getter())
        except Exception:
            return {}
    return {}


def _vispy_draw_count(win) -> int:
    diagnostics = _vispy_presentation_diagnostics(win)
    return int(diagnostics.get("draw_count", 0) or 0)


def _vispy_tile_presentation_request_count(win) -> int:
    diagnostics = _vispy_presentation_diagnostics(win)
    return int(diagnostics.get("tile_presentation_request_count", 0) or 0)


def _vispy_tile_presentation_draw_count(win) -> int:
    diagnostics = _vispy_presentation_diagnostics(win)
    return int(diagnostics.get("tile_presentation_draw_count", 0) or 0)


def _montage_visibility_state(win, *, mode: str | None = None) -> dict[str, object]:
    session = getattr(win, "_montage_session", None)
    if mode is None:
        mode = str(getattr(win.img_view, "montageDisplayMode", lambda: "")())
    if session is None:
        return {
            "fully_visible": False,
            "active_presented_tile_count": 0,
            "active_planned_tile_count": 0,
            "deferred_display_tile_count": 0,
        }
    active = set(_active_planned_montage_tiles(session))
    expected = set(_expected_requested_montage_tiles(session))
    if not expected:
        expected = set(active)
    presented = {int(tile) for tile in tuple(getattr(session, "presented_tiles", ()) or ())}
    deferred = tuple(int(tile) for tile in tuple(getattr(session, "deferred_display_tiles", ()) or ()))
    vispy = _vispy_presentation_diagnostics(win)
    overlay_count = _montage_overlay_count(win)
    overlays_above_tiles = bool(vispy.get("overlays_above_tiles", False))
    overlay_nonblocking = (
        overlay_count == 0
        or (
            str(mode) == "vispy_tile_layer"
            and not overlays_above_tiles
            and active
            and active.issubset(presented)
        )
    )
    has_backlog = bool(
        getattr(session, "pending_tiles", ())
        or getattr(session, "loading_tiles", ())
        or getattr(session, "pending_completed_tiles", ())
        or getattr(session, "active_tile_requests", ())
        or getattr(session, "active_stage_requests", ())
        or getattr(session, "attached_stage_requests", ())
        or getattr(session, "stage_waiting_tiles", ())
        or getattr(session, "final_commit_pending", False)
        or getattr(session, "final_display_drain_pending", False)
        or getattr(session, "flush_pending", False)
        or deferred
        or getattr(session, "dirty_payloads", ())
        or getattr(session, "pending_removals", ())
    )
    active_presented = active.intersection(presented)
    fully_visible = bool(
        str(mode) in {"tile_layer", "vispy_tile_layer", "canvas"}
        and getattr(session, "display_committed", False)
        and not has_backlog
        and expected.issubset(active)
        and expected.issubset(presented)
        and overlay_nonblocking
    )
    return {
        "fully_visible": fully_visible,
        "active_presented_tile_count": len(active_presented),
        "active_planned_tile_count": len(active),
        "requested_tile_count": len(expected),
        "deferred_display_tile_count": len(deferred),
    }


def _active_planned_montage_tiles(session) -> tuple[int, ...]:
    skipped = {int(tile) for tile in tuple(getattr(session, "skipped_tiles", ()) or ())}
    visible = tuple(getattr(session, "visible_tiles", ()) or ())
    active = []
    for tile in visible:
        try:
            index = int(getattr(tile, "montage_index"))
        except Exception:
            continue
        if index not in skipped:
            active.append(index)
    return tuple(dict.fromkeys(active))


def _expected_requested_montage_tiles(session) -> tuple[int, ...]:
    skipped = {int(tile) for tile in tuple(getattr(session, "skipped_tiles", ()) or ())}
    indices = tuple(getattr(session, "level_expected_indices", ()) or ())
    if not indices:
        geometry = getattr(getattr(session, "plan", None), "geometry", None)
        indices = tuple(getattr(geometry, "indices", ()) or ())
    if not indices:
        return ()
    # level_expected_indices stores source indices for the montage request.  For
    # ordinary full-workflow montages the semantic tile numbers are positional.
    # When a custom subset is used, still require every requested tile slot.
    return tuple(index for index in range(len(indices)) if int(index) not in skipped)


def _vispy_canvas_visible(win) -> bool:
    native = getattr(getattr(win, "img_view", None), "_vispy_canvas_native", None)
    if native is None:
        return False
    try:
        return bool(native.isVisible())
    except Exception:
        return False


def _montage_overlay_count(win) -> int:
    getter = getattr(getattr(win, "img_view", None), "montageTileOverlayCount", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:
            return 0
    return 0


def _base_record(
    *,
    run_id: str,
    backend: str,
    data_path: Path,
    data,
    load_mode: str,
    montage_axis: int,
    indices: tuple[int, ...],
    columns: int,
    full_tile_count: int,
    show_window: bool,
    max_tiles: int | None,
    profiler_type: str,
    profiler_artifact_paths: tuple[str | Path, ...],
    qt_platform: str,
) -> dict[str, object]:
    capped = max_tiles is not None and len(indices) < int(full_tile_count)
    smoke_only = bool((not show_window) or str(qt_platform).lower() == "offscreen" or capped)
    return {
        "run_id": run_id,
        "backend": backend,
        "data_path": str(data_path),
        "load_mode": str(load_mode),
        "profiler_type": str(profiler_type),
        "profiler_artifact_paths": [str(path) for path in tuple(profiler_artifact_paths or ())],
        "qt_platform": str(qt_platform),
        "xdg_session_type": os.environ.get("XDG_SESSION_TYPE", ""),
        "wayland_display": os.environ.get("WAYLAND_DISPLAY", ""),
        "display": os.environ.get("DISPLAY", ""),
        "show_window": bool(show_window),
        "max_tiles": None if max_tiles is None else int(max_tiles),
        "tile_cap_applied": bool(capped),
        "smoke_only": bool(smoke_only),
        "pacing_evidence": bool(not smoke_only),
        "data_shape": tuple(int(value) for value in np.shape(data)),
        "data_dtype": str(getattr(getattr(data, "dtype", None), "str", getattr(data, "dtype", ""))),
        "montage_axis": int(montage_axis),
        "tile_count": len(indices),
        "full_tile_count": int(full_tile_count),
        "columns": int(columns),
    }


def _load_dataset(path: Path, *, mode: str):
    mode = str(mode)
    if mode == "app":
        from arrayscope.io.file_interpreters import load_path

        return load_path(path).data
    if mode == "native":
        if _is_nifti(path):
            import nibabel as nib

            return np.asanyarray(nib.load(str(path)).dataobj)
        if path.suffix.lower() == ".npy":
            return np.load(path)
    raise ValueError(f"unsupported load mode {mode!r}; expected 'app' or 'native'")


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _replace_settings(settings, *, backend: str, image_choice, montage_choice):
    from dataclasses import replace

    return replace(
        settings,
        image_rendering_backend=image_choice.VISPY if backend == "vispy" else image_choice.PYQTGRAPH,
        montage_display_backend=montage_choice.TILE_LAYER,
    )


def _append_record(records: list[dict[str, object]], jsonl: str | Path | None, record: dict[str, object]) -> None:
    records.append(record)
    line = json.dumps(record, sort_keys=True)
    if jsonl is None:
        return
    path = Path(jsonl)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _process_events(app, QtCore, *, count: int) -> None:
    flags = QtCore.QEventLoop.ProcessEventsFlag.AllEvents
    for _ in range(max(1, int(count))):
        app.processEvents(flags, 50)


def _restore_setting(settings, key: str, previous) -> None:
    if previous is None:
        settings.remove(key)
    else:
        settings.setValue(key, previous)


def _default_columns(tile_count: int) -> int:
    return max(1, int(math.ceil(math.sqrt(max(1, int(tile_count))))))


def _normalize_backend(backend: str) -> str:
    backend = str(backend).strip().lower()
    if backend not in {"pyqtgraph", "vispy"}:
        raise ValueError(f"unsupported backend {backend!r}; expected 'pyqtgraph' or 'vispy'")
    return backend


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def py_spy_command(argv: tuple[str, ...] | None = None) -> str:
    args = tuple(sys.argv[1:] if argv is None else argv)
    return shlex.join(("py-spy", "record", "--native", "-o", "arrayscope-montage-workflow.svg", "--", sys.executable, "-m", "arrayscope.tools.profile_montage_workflow", *args))


def cprofile_command(argv: tuple[str, ...], output: str | Path) -> str:
    return shlex.join((sys.executable, "-m", "cProfile", "-o", str(output), "-m", "arrayscope.tools.profile_montage_workflow", *tuple(argv)))


def py_spy_raw_command(argv: tuple[str, ...], output: str | Path) -> str:
    return shlex.join(("py-spy", "record", "--native", "--format", "raw", "-o", str(output), "--", sys.executable, "-m", "arrayscope.tools.profile_montage_workflow", *tuple(argv)))


def perf_record_command(argv: tuple[str, ...], output: str | Path) -> str:
    return shlex.join(("perf", "record", "-g", "-o", str(output), "--", sys.executable, "-m", "arrayscope.tools.profile_montage_workflow", *tuple(argv)))


def profiler_suite_commands(argv: tuple[str, ...], suite_dir: str | Path) -> tuple[dict[str, object], ...]:
    suite_dir = Path(suite_dir)
    plain_jsonl = suite_dir / "plain.jsonl"
    cprofile_artifact = suite_dir / "montage.cprofile"
    py_spy_artifact = suite_dir / "montage.pyspy.raw"
    perf_artifact = suite_dir / "montage.perf.data"
    base = _suite_child_args(argv)
    return (
        {
            "profiler_type": "plain",
            "jsonl": str(plain_jsonl),
            "artifact_paths": (str(plain_jsonl),),
            "command": shlex.join((sys.executable, "-m", "arrayscope.tools.profile_montage_workflow", *base, "--jsonl", str(plain_jsonl), "--profiler-type", "plain", "--profiler-artifact", str(plain_jsonl))),
        },
        {
            "profiler_type": "cprofile",
            "jsonl": str(suite_dir / "cprofile.jsonl"),
            "artifact_paths": (str(cprofile_artifact), str(suite_dir / "cprofile.jsonl")),
            "command": cprofile_command((*base, "--jsonl", str(suite_dir / "cprofile.jsonl"), "--profiler-type", "cprofile", "--profiler-artifact", str(cprofile_artifact)), cprofile_artifact),
        },
        {
            "profiler_type": "py-spy-raw",
            "jsonl": str(suite_dir / "py-spy.jsonl"),
            "artifact_paths": (str(py_spy_artifact), str(suite_dir / "py-spy.jsonl")),
            "command": py_spy_raw_command((*base, "--jsonl", str(suite_dir / "py-spy.jsonl"), "--profiler-type", "py-spy-raw", "--profiler-artifact", str(py_spy_artifact)), py_spy_artifact),
        },
        {
            "profiler_type": "perf-record",
            "jsonl": str(suite_dir / "perf.jsonl"),
            "artifact_paths": (str(perf_artifact), str(suite_dir / "perf.jsonl")),
            "command": perf_record_command((*base, "--jsonl", str(suite_dir / "perf.jsonl"), "--profiler-type", "perf-record", "--profiler-artifact", str(perf_artifact)), perf_artifact),
        },
    )


def run_profile_suite(argv: tuple[str, ...], suite_dir: str | Path) -> int:
    suite_dir = Path(suite_dir)
    suite_dir.mkdir(parents=True, exist_ok=True)
    commands = profiler_suite_commands(argv, suite_dir)
    manifest_path = suite_dir / "suite-manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()
    for item in commands:
        profiler = str(item["profiler_type"])
        command = str(item["command"])
        executable = shlex.split(command)[0]
        if profiler.startswith("py-spy") and shutil.which("py-spy") is None:
            return _write_suite_failure(manifest_path, item, "py-spy executable not found")
        if profiler.startswith("perf") and shutil.which("perf") is None:
            return _write_suite_failure(manifest_path, item, "perf executable not found")
        started = perf_counter()
        completed = subprocess.run(shlex.split(command), cwd=Path.cwd(), check=False)
        elapsed_ms = (perf_counter() - started) * 1000.0
        artifacts = tuple(str(path) for path in tuple(item.get("artifact_paths", ()) or ()))
        missing = [path for path in artifacts if not Path(path).exists() or Path(path).stat().st_size <= 0]
        record = {
            **item,
            "command_executable": executable,
            "returncode": int(completed.returncode),
            "elapsed_ms": float(elapsed_ms),
            "missing_artifacts": missing,
            "complete": completed.returncode == 0 and not missing,
        }
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if completed.returncode != 0 or missing:
            return completed.returncode or 2
    return 0


def _write_suite_failure(path: Path, item: dict[str, object], reason: str) -> int:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**item, "complete": False, "missing_reason": str(reason)}, sort_keys=True) + "\n")
    return 127


def _suite_child_args(argv: tuple[str, ...]) -> tuple[str, ...]:
    blocked = {"--profile-suite", "--print-py-spy-command"}
    result: list[str] = []
    skip_next = False
    for index, arg in enumerate(tuple(argv)):
        if skip_next:
            skip_next = False
            continue
        if arg in blocked:
            skip_next = arg == "--profile-suite"
            continue
        if arg.startswith("--profile-suite="):
            continue
        if arg in {"--jsonl", "--profiler-type", "--profiler-artifact"}:
            skip_next = True
            continue
        if arg.startswith("--jsonl=") or arg.startswith("--profiler-type=") or arg.startswith("--profiler-artifact="):
            continue
        result.append(arg)
    return tuple(result)


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the realistic tiled montage + FFT profiling workflow")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Dataset path; defaults to the bundled realistic NIfTI")
    parser.add_argument("--backend", choices=("pyqtgraph", "vispy", "all"), default="pyqtgraph")
    parser.add_argument("--jsonl", default=None, help="Optional JSONL metrics output")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-tiles", type=int, default=0, help="Optional tile cap for local smoke runs; 0 means full dim 2")
    parser.add_argument("--columns", type=int, default=0, help="Montage columns; 0 chooses a near-square layout")
    parser.add_argument("--load-mode", choices=("app", "native"), default="app")
    parser.add_argument("--hide-window", action="store_true", help="Do not show the window; useful with QT_QPA_PLATFORM=offscreen")
    parser.add_argument("--print-py-spy-command", action="store_true", help="Print an external py-spy command for this invocation and exit")
    parser.add_argument("--profile-suite", default=None, help="Run plain JSONL, cProfile, py-spy raw, and perf record into this directory")
    parser.add_argument("--profiler-type", default="plain", help=argparse.SUPPRESS)
    parser.add_argument("--profiler-artifact", action="append", default=[], help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.print_py_spy_command:
        filtered = tuple(arg for arg in (argv if argv is not None else sys.argv[1:]) if arg != "--print-py-spy-command")
        print(py_spy_command(filtered))
        return 0
    if args.profile_suite:
        source_argv = tuple(argv if argv is not None else sys.argv[1:])
        return run_profile_suite(source_argv, args.profile_suite)

    jsonl = None if args.jsonl is None else Path(args.jsonl)
    if jsonl is not None and jsonl.exists():
        jsonl.unlink()
    records = []
    for backend in (("pyqtgraph", "vispy") if args.backend == "all" else (args.backend,)):
        records.extend(
            run_profile_montage_workflow(
                data_path=args.data,
                backend=backend,
                jsonl=jsonl,
                timeout_s=args.timeout_s,
                max_tiles=None if args.max_tiles <= 0 else args.max_tiles,
                columns=None if args.columns <= 0 else args.columns,
                load_mode=args.load_mode,
                show_window=not args.hide_window,
                profiler_type=args.profiler_type,
                profiler_artifact_paths=tuple(args.profiler_artifact or ()),
            )
        )
    for record in records:
        print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
