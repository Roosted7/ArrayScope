"""Profile a realistic full-montage workflow in a real ArrayScope window."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib import metadata
import io
import json
import math
import os
import pstats
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from time import perf_counter
from uuid import uuid4

import numpy as np


DEFAULT_DATA_PATH = Path("data/_WIPDelRec-tT2_20260223150234_14.nii")
PY_SPY_LOW_IMPACT_SAMPLE_RATE_HZ = 50
PY_SPY_FULL_SAMPLE_RATE_HZ = 80
PY_SPY_FULL_DURATION_S = 16
PY_SPY_FULL_DETACH_MARGIN_S = 1
PY_SPY_FULL_ALLOWED_MISSED_STACKS = 1


def run_profile_montage_workflow(
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    backend: str = "pyqtgraph",
    jsonl: str | Path | None = None,
    timeout_s: float = 180.0,
    max_tiles: int | None = None,
    columns: int | None = None,
    load_mode: str = "app",
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
            max_tiles=max_tiles,
            profiler_type=profiler_type,
            profiler_artifact_paths=profiler_artifact_paths,
            run_temperature=_workflow_run_temperature(),
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
                "run_temperature": "cold",
            },
        )

        win = ArrayScopeWindow(data)
        win.app_settings = _replace_settings(
            win.app_settings,
            backend=backend,
            image_choice=ImageRenderingBackendChoice,
            montage_choice=MontageDisplayBackendChoice,
        )
        apply_theme = getattr(win, "_apply_theme_choice", None)
        if callable(apply_theme):
            apply_theme(win.app_settings.theme, persist=False)
        win.resize(1400, 900)
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
        _append_record(records, jsonl, {**base, **raw_record, "run_temperature": "cold"})

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
        _append_record(records, jsonl, {**base, **fft_record, "run_temperature": "mixed"})
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
        self.gaps_ms: list[float] = []

    def start(self) -> None:
        self.reset()
        self._timer.start()

    def reset(self) -> None:
        self._last = perf_counter()
        self.max_gap_ms = 0.0
        self.tick_count = 0
        self.gaps_ms = []

    def _tick(self) -> None:
        now = perf_counter()
        gap_ms = (now - self._last) * 1000.0
        self.gaps_ms.append(float(gap_ms))
        self.max_gap_ms = max(self.max_gap_ms, gap_ms)
        self._last = now
        self.tick_count += 1

    def percentile_ms(self, percentile: float) -> float | None:
        if not self.gaps_ms:
            return None
        return _percentile(tuple(self.gaps_ms), percentile)


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
    record = _phase_record(
        win,
        phase=phase,
        elapsed_ms=elapsed_ms,
        event_loop_p95_gap_ms=probe.percentile_ms(95),
        event_loop_p99_gap_ms=probe.percentile_ms(99),
        event_loop_max_gap_ms=probe.max_gap_ms,
    )
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
        f"loading={snapshot.montage.loading_tiles} "
        f"active={0 if session is None else len(getattr(session, 'active_tile_requests', ()) or ())} "
        f"completed={0 if session is None else len(getattr(session, 'pending_completed_tiles', ()) or ())} "
        f"stage_waiting={0 if session is None else sum(len(tiles) for tiles in getattr(session, 'stage_waiting_tiles', {}).values())} "
        f"lead_warmups={0 if session is None else len(getattr(session, 'lead_stage_warmups', {}) or {})} "
        f"active_presented={final_visibility_state.get('active_presented_tile_count', 0)}/"
        f"{final_visibility_state.get('active_planned_tile_count', 0)} "
        f"overlays={_montage_overlay_count(win)} vispy_draws={_vispy_draw_count(win)} "
        f"tile_draw={_vispy_tile_presentation_draw_count(win)}/{_vispy_tile_presentation_request_count(win)}"
    )


def _phase_record(
    win,
    *,
    phase: str,
    elapsed_ms: float,
    event_loop_p95_gap_ms: float | None,
    event_loop_p99_gap_ms: float | None,
    event_loop_max_gap_ms: float,
) -> dict[str, object]:
    snapshot = win.collect_runtime_diagnostics()
    timing = snapshot.montage_timing
    montage = snapshot.montage
    resource = snapshot.resource_governor
    recent_callbacks = () if resource is None else tuple(resource.recent_over_warning_callbacks)
    recent_ui_work = () if resource is None else tuple(resource.recent_ui_work_observations)
    feedback_channels = () if resource is None else tuple(resource.feedback_channels)
    ui_decisions = () if resource is None else tuple(resource.ui_decisions)
    lane_decisions = () if resource is None else tuple(resource.lane_decisions)
    vispy = _vispy_presentation_diagnostics(win)
    return {
        "phase": phase,
        "elapsed_ms": float(elapsed_ms),
        "event_loop_p95_gap_ms": _optional_float(event_loop_p95_gap_ms),
        "event_loop_p99_gap_ms": _optional_float(event_loop_p99_gap_ms),
        "event_loop_max_gap_ms": float(event_loop_max_gap_ms),
        "complete": True,
        "image_backend_actual": str(snapshot.image_rendering_backend_actual),
        "montage_display_mode": str(montage.display_mode),
        "montage_backend_chosen": str(montage.backend_chosen),
        "montage_loaded_tiles": int(montage.loaded_tiles),
        "montage_loading_tiles": int(montage.loading_tiles),
        "montage_pending_tiles": int(montage.pending_tiles),
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
        "vispy_fast_drain_last_enabled": bool(getattr(win, "_vispy_tile_layer_fast_drain_last_enabled", False)),
        "vispy_fast_drain_enabled_count": int(getattr(win, "_vispy_tile_layer_fast_drain_enabled_count", 0) or 0),
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
        "resource_lane_decisions": [
            {
                **asdict(decision),
                "lane": str(getattr(getattr(decision, "lane", ""), "value", getattr(decision, "lane", ""))),
            }
            for decision in lane_decisions
        ],
        "resource_ui_decisions": [asdict(decision) for decision in ui_decisions],
        "recent_ui_work_observations": [asdict(observation) for observation in recent_ui_work],
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
        }
    active = set(_active_planned_montage_tiles(session))
    expected = set(_expected_requested_montage_tiles(session))
    if not expected:
        expected = set(active)
    presented = {int(tile) for tile in tuple(getattr(session, "presented_tiles", ()) or ())}
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
        or getattr(session, "flush_pending", False)
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
    max_tiles: int | None,
    profiler_type: str,
    profiler_artifact_paths: tuple[str | Path, ...],
    run_temperature: str = "mixed",
    qt_platform: str,
) -> dict[str, object]:
    capped = max_tiles is not None and len(indices) < int(full_tile_count)
    smoke_only = bool(str(qt_platform).lower() == "offscreen" or capped)
    return {
        "run_id": run_id,
        "backend": backend,
        "data_path": str(data_path),
        "load_mode": str(load_mode),
        "profiler_type": str(profiler_type),
        "profiler_artifact_paths": [str(path) for path in tuple(profiler_artifact_paths or ())],
        "run_temperature": str(run_temperature),
        "qt_platform": str(qt_platform),
        "xdg_session_type": os.environ.get("XDG_SESSION_TYPE", ""),
        "wayland_display": os.environ.get("WAYLAND_DISPLAY", ""),
        "display": os.environ.get("DISPLAY", ""),
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

    from arrayscope.app.theme import ThemeChoice

    return replace(
        settings,
        theme=ThemeChoice.DARK if backend == "vispy" else ThemeChoice.LIGHT,
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
    native = _py_spy_native_requested(args)
    args = _py_spy_filtered_args(args)
    return shlex.join(
        (
            "py-spy",
            "record",
            *(("--native",) if native else ()),
            "--rate",
            str(PY_SPY_LOW_IMPACT_SAMPLE_RATE_HZ),
            "--gil",
            "--nonblocking",
            "-o",
            "arrayscope-montage-workflow.svg",
            "--",
            sys.executable,
            "-m",
            "arrayscope.tools.profile_montage_workflow",
            *args,
        )
    )


def cprofile_command(argv: tuple[str, ...], output: str | Path) -> str:
    return shlex.join((sys.executable, "-m", "cProfile", "-o", str(output), "-m", "arrayscope.tools.profile_montage_workflow", *tuple(argv)))


def py_spy_raw_command(
    argv: tuple[str, ...],
    output: str | Path,
    *,
    rate_hz: int,
    nonblocking: bool = False,
    gil: bool = False,
    duration_s: int | None = None,
    detach_margin_s: int = 0,
) -> str:
    native = _py_spy_native_requested(tuple(argv))
    target = (
        (
            sys.executable,
            "-c",
            "import sys, time; from arrayscope.tools.profile_montage_workflow import main; "
            "started = time.monotonic(); duration = float(sys.argv[-2]); margin = float(sys.argv[-1]); "
            "rc = main(tuple(sys.argv[1:-2])); "
            "time.sleep(max(0.0, started + duration + margin - time.monotonic())); raise SystemExit(rc)",
            *tuple(argv),
            str(int(duration_s or 0)),
            str(int(detach_margin_s)),
        )
        if duration_s is not None
        else (sys.executable, "-m", "arrayscope.tools.profile_montage_workflow", *tuple(argv))
    )
    return shlex.join(
        (
            "py-spy",
            "record",
            *(("--native",) if native else ()),
            *(("--duration", str(int(duration_s))) if duration_s is not None else ()),
            "--rate",
            str(int(rate_hz)),
            *(("--nonblocking",) if nonblocking else ()),
            *(("--gil",) if gil else ()),
            "--format",
            "raw",
            "-o",
            str(output),
            "--",
            *target,
        )
    )


def perf_record_command(argv: tuple[str, ...], output: str | Path) -> str:
    return shlex.join(("perf", "record", "-F", "99", "-g", "-o", str(output), "--", sys.executable, "-m", "arrayscope.tools.profile_montage_workflow", *tuple(argv)))


def profiler_suite_commands(argv: tuple[str, ...], suite_dir: str | Path) -> tuple[dict[str, object], ...]:
    suite_dir = Path(suite_dir)
    plain_jsonl = suite_dir / "plain.jsonl"
    base = _suite_child_args(argv)
    include_cprofile = _include_cprofile_requested(argv)
    py_spy_native = _py_spy_native_requested(argv)
    py_spy_low_type = "py-spy-raw-low-impact-native" if py_spy_native else "py-spy-raw-low-impact"
    py_spy_full_type = "py-spy-raw-full-native" if py_spy_native else "py-spy-raw-full"
    backends = _suite_profiler_backends(base)
    split_backend_artifacts = len(backends) > 1
    commands: list[dict[str, object]] = [
        {
            "step_id": "plain",
            "profiler_type": "plain",
            "backend": "all" if split_backend_artifacts else backends[0],
            "required": True,
            "jsonl": str(plain_jsonl),
            "artifact_paths": (str(plain_jsonl),),
            "command": shlex.join((sys.executable, "-m", "arrayscope.tools.profile_montage_workflow", *base, "--jsonl", str(plain_jsonl), "--profiler-type", "plain", "--profiler-artifact", str(plain_jsonl))),
        },
    ]
    if include_cprofile:
        for backend in backends:
            backend_base = _suite_args_for_backend(base, backend)
            suffix = f".{backend}" if split_backend_artifacts else ""
            cprofile_artifact = suite_dir / f"montage{suffix}.cprofile"
            cprofile_jsonl = suite_dir / f"cprofile{suffix}.jsonl"
            commands.append(
                {
                    "step_id": f"cprofile:{backend}",
                    "profiler_type": "cprofile",
                    "backend": backend,
                    "required": True,
                    "jsonl": str(cprofile_jsonl),
                    "artifact_paths": (str(cprofile_artifact), str(cprofile_jsonl)),
                    "command": cprofile_command((*backend_base, "--jsonl", str(cprofile_jsonl), "--profiler-type", "cprofile", "--profiler-artifact", str(cprofile_artifact)), cprofile_artifact),
                }
            )
    for backend in backends:
        backend_base = _suite_args_for_backend(base, backend)
        py_spy_base = (*backend_base, "--py-spy-native") if py_spy_native else backend_base
        suffix = f".{backend}" if split_backend_artifacts else ""
        py_spy_low_artifact = suite_dir / f"montage{suffix}.pyspy.low-impact.raw"
        py_spy_low_jsonl = suite_dir / f"py-spy-low-impact{suffix}.jsonl"
        py_spy_full_artifact = suite_dir / f"montage{suffix}.pyspy.full.raw"
        py_spy_full_jsonl = suite_dir / f"py-spy-full{suffix}.jsonl"
        perf_artifact = suite_dir / f"montage{suffix}.perf.data"
        perf_jsonl = suite_dir / f"perf{suffix}.jsonl"
        commands.extend(
            (
                {
                    "step_id": f"{py_spy_low_type}:{backend}",
                    "profiler_type": py_spy_low_type,
                    "backend": backend,
                    "required": True,
                    "jsonl": str(py_spy_low_jsonl),
                    "artifact_paths": (str(py_spy_low_artifact), str(py_spy_low_jsonl)),
                    "command": py_spy_raw_command(
                        (*py_spy_base, "--jsonl", str(py_spy_low_jsonl), "--profiler-type", py_spy_low_type, "--profiler-artifact", str(py_spy_low_artifact)),
                        py_spy_low_artifact,
                        rate_hz=PY_SPY_LOW_IMPACT_SAMPLE_RATE_HZ,
                        nonblocking=True,
                        gil=True,
                    ),
                },
                {
                    "step_id": f"{py_spy_full_type}:{backend}",
                    "profiler_type": py_spy_full_type,
                    "backend": backend,
                    "required": True,
                    "jsonl": str(py_spy_full_jsonl),
                    "artifact_paths": (str(py_spy_full_artifact), str(py_spy_full_jsonl)),
                    "command": py_spy_raw_command(
                        (*py_spy_base, "--jsonl", str(py_spy_full_jsonl), "--profiler-type", py_spy_full_type, "--profiler-artifact", str(py_spy_full_artifact)),
                        py_spy_full_artifact,
                        rate_hz=PY_SPY_FULL_SAMPLE_RATE_HZ,
                        duration_s=PY_SPY_FULL_DURATION_S,
                        detach_margin_s=PY_SPY_FULL_DETACH_MARGIN_S,
                    ),
                },
                {
                    "step_id": f"perf-record:{backend}",
                    "profiler_type": "perf-record",
                    "backend": backend,
                    "required": False,
                    "jsonl": str(perf_jsonl),
                    "artifact_paths": (str(perf_artifact), str(perf_jsonl)),
                    "command": perf_record_command((*backend_base, "--jsonl", str(perf_jsonl), "--profiler-type", "perf-record", "--profiler-artifact", str(perf_artifact)), perf_artifact),
                },
            )
        )
    return tuple(commands)


def run_profile_suite(argv: tuple[str, ...], suite_dir: str | Path) -> int:
    suite_dir = Path(suite_dir)
    suite_dir.mkdir(parents=True, exist_ok=True)
    commands = profiler_suite_commands(argv, suite_dir)
    manifest_path = suite_dir / "suite-manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()
    tool_versions = _suite_tool_versions()
    repository = _repository_state()
    step_records: list[dict[str, object]] = []
    for item in commands:
        profiler = str(item["profiler_type"])
        step_temperature = _suite_step_temperature(step_records)
        command = str(item["command"])
        executable = shlex.split(command)[0]
        log_stem = _artifact_stem(str(item.get("step_id", profiler)))
        stdout_path = suite_dir / f"{log_stem}.stdout.log"
        stderr_path = suite_dir / f"{log_stem}.stderr.log"
        required = bool(item.get("required", False))
        if profiler.startswith("py-spy") and shutil.which("py-spy") is None:
            status = "failed" if required else "skipped"
            reason_key = "failure_reason" if status == "failed" else "skip_reason"
            record = _suite_step_record(
                item,
                command_executable=executable,
                returncode=127,
                elapsed_ms=0.0,
                missing_artifacts=tuple(item.get("artifact_paths", ()) or ()),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                status=status,
                reason_key=reason_key,
                reason="py-spy executable not found",
                tool_versions=tool_versions,
                repository=repository,
                run_temperature=step_temperature,
            )
            _write_manifest_record(manifest_path, record)
            step_records.append(record)
            continue
        if profiler.startswith("perf") and shutil.which("perf") is None:
            record = _suite_step_record(
                item,
                command_executable=executable,
                returncode=127,
                elapsed_ms=0.0,
                missing_artifacts=tuple(item.get("artifact_paths", ()) or ()),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                status="skipped",
                reason_key="skip_reason",
                reason="perf executable not found",
                tool_versions=tool_versions,
                repository=repository,
                run_temperature=step_temperature,
            )
            _write_manifest_record(manifest_path, record)
            step_records.append(record)
            continue
        started = perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(shlex.split(command), cwd=Path.cwd(), check=False, stdout=stdout, stderr=stderr)
        elapsed_ms = (perf_counter() - started) * 1000.0
        artifacts = tuple(str(path) for path in tuple(item.get("artifact_paths", ()) or ()))
        missing = [path for path in artifacts if not Path(path).exists() or Path(path).stat().st_size <= 0]
        profiler_diagnostics = _profiler_log_diagnostics(profiler, stdout_path, stderr_path)
        sample_issue = _profiler_sample_issue(profiler, profiler_diagnostics)
        complete = int(completed.returncode) == 0 and not missing and sample_issue == ""
        if complete:
            status = "completed"
            reason_key = None
            reason = ""
        else:
            status = "failed" if required else "degraded"
            reason_key = "failure_reason" if status == "failed" else "degraded_reason"
            if int(completed.returncode) != 0:
                reason = f"command exited with {int(completed.returncode)}"
            elif sample_issue:
                reason = sample_issue
            else:
                reason = "missing or empty expected artifacts"
        record = _suite_step_record(
            item,
            command_executable=executable,
            returncode=int(completed.returncode),
            elapsed_ms=elapsed_ms,
            missing_artifacts=missing,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            profiler_diagnostics=profiler_diagnostics,
            status=status,
            reason_key=reason_key,
            reason=reason,
            tool_versions=tool_versions,
            repository=repository,
            run_temperature=step_temperature,
        )
        _write_manifest_record(manifest_path, record)
        step_records.append(record)
    summary = _suite_summary_record(step_records, tool_versions=tool_versions, repository=repository)
    interpretation_path = suite_dir / "suite-summary.md"
    summary["interpretation_path"] = str(interpretation_path)
    _write_suite_interpretation(interpretation_path, step_records, summary)
    print(interpretation_path.read_text(encoding="utf-8"), end="")
    _write_manifest_record(manifest_path, summary)
    return 0 if bool(summary["overall_valid"]) else _suite_exit_code(step_records)


def _write_manifest_record(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _suite_step_record(
    item: dict[str, object],
    *,
    command_executable: str,
    returncode: int,
    elapsed_ms: float,
    missing_artifacts,
    stdout_path: Path,
    stderr_path: Path,
    profiler_diagnostics: dict[str, object] | None = None,
    status: str,
    reason_key: str | None = None,
    reason: str = "",
    tool_versions: dict[str, str],
    repository: dict[str, object],
    run_temperature: str,
) -> dict[str, object]:
    complete = str(status) == "completed"
    record = {
        **item,
        "record_type": "suite_step",
        "required": bool(item.get("required", False)),
        "status": str(status),
        "valid": bool(complete),
        "complete": bool(complete),
        "command_executable": str(command_executable),
        "returncode": int(returncode),
        "elapsed_ms": float(elapsed_ms),
        "missing_artifacts": [str(path) for path in tuple(missing_artifacts or ())],
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "profiler_diagnostics": dict(profiler_diagnostics or {}),
        "tool_versions": dict(tool_versions),
        "repository_revision": str(repository["repository_revision"]),
        "repository_dirty": bool(repository["repository_dirty"]),
        "run_temperature": str(run_temperature),
    }
    if reason_key is not None and reason:
        record[str(reason_key)] = str(reason)
    return record


def _suite_summary_record(
    records: list[dict[str, object]],
    *,
    tool_versions: dict[str, str],
    repository: dict[str, object],
) -> dict[str, object]:
    statuses = {str(record.get("step_id", record["profiler_type"])): str(record["status"]) for record in records}
    overall_valid = bool(records) and all(bool(record.get("valid", False)) for record in records)
    if overall_valid:
        overall_status = "completed"
    elif any(str(record.get("status")) == "failed" for record in records):
        overall_status = "failed"
    else:
        overall_status = "degraded"
    return {
        "record_type": "suite_summary",
        "overall_valid": bool(overall_valid),
        "overall_status": overall_status,
        "step_statuses": statuses,
        "step_count": len(records),
        "tool_versions": dict(tool_versions),
        "repository_revision": str(repository["repository_revision"]),
        "repository_dirty": bool(repository["repository_dirty"]),
        "run_temperature": _aggregate_run_temperature(tuple(str(record.get("run_temperature", "")) for record in records)),
    }


def _suite_exit_code(records: list[dict[str, object]]) -> int:
    for record in records:
        if str(record.get("status")) == "failed":
            return int(record.get("returncode") or 1)
    for record in records:
        if str(record.get("status")) in {"degraded", "skipped"}:
            return int(record.get("returncode") or 2)
    return 2


def _artifact_stem(profiler: str) -> str:
    return str(profiler).replace("/", "-").replace(" ", "-")


def _profiler_log_diagnostics(profiler: str, stdout_path: Path, stderr_path: Path) -> dict[str, object]:
    stdout = _read_text_if_exists(stdout_path)
    stderr = _read_text_if_exists(stderr_path)
    diagnostics: dict[str, object] = {
        "warning_count": stderr.count("WARN  "),
    }
    if str(profiler).startswith("py-spy"):
        rate_match = re.search(r"Sampling process\s+(\d+)\s+times a second", stdout)
        match = re.search(r"Samples:\s*(\d+)\s+Errors:\s*(\d+)", stdout)
        diagnostics["sample_rate_hz"] = int(rate_match.group(1)) if rate_match else 0
        diagnostics["sample_count"] = int(match.group(1)) if match else 0
        diagnostics["error_count"] = int(match.group(2)) if match else 0
        diagnostics["missed_stack_count"] = stderr.count("Failed to get stack trace")
        diagnostics["scope"] = "low_impact_python_gil_holders" if "low-impact" in str(profiler) else "complete_sampling_all_python_threads"
        diagnostics["sampling_mode"] = "nonblocking_gil_samples" if "low-impact" in str(profiler) else "blocking_all_python_thread_samples"
        diagnostics["allowed_missed_stack_count"] = 0 if "low-impact" in str(profiler) else PY_SPY_FULL_ALLOWED_MISSED_STACKS
        diagnostics["sampling_complete"] = bool(
            int(diagnostics["sample_count"]) > 0
            and (
                ("low-impact" in str(profiler))
                or (
                    int(diagnostics["error_count"]) <= PY_SPY_FULL_ALLOWED_MISSED_STACKS
                    and int(diagnostics["missed_stack_count"]) <= PY_SPY_FULL_ALLOWED_MISSED_STACKS
                )
            )
        )
    elif str(profiler).startswith("perf"):
        diagnostics["scope"] = "native_and_python_process_samples"
    elif str(profiler) == "cprofile":
        diagnostics["scope"] = "deterministic_python_calls"
    else:
        diagnostics["scope"] = "workflow_timing_jsonl"
    return diagnostics


def _profiler_sample_issue(profiler: str, diagnostics: dict[str, object]) -> str:
    if str(profiler).startswith("py-spy") and int(diagnostics.get("sample_count", 0) or 0) <= 0:
        return "py-spy produced no samples"
    if str(profiler).startswith("py-spy") and "full" in str(profiler) and not bool(diagnostics.get("sampling_complete", False)):
        return f"py-spy full profile missed more than {PY_SPY_FULL_ALLOWED_MISSED_STACKS} stack sample(s)"
    return ""


def _read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _write_suite_interpretation(path: Path, records: list[dict[str, object]], summary: dict[str, object]) -> None:
    lines = [
        "# Profile Suite Summary",
        "",
        f"- Overall status: `{summary.get('overall_status')}`",
        f"- Overall valid: `{summary.get('overall_valid')}`",
        f"- Run temperature: `{summary.get('run_temperature')}`",
        "",
        "## Run Metadata",
        "",
        *_run_metadata_summary(summary),
        "",
        "## Tool Status",
        "",
        *_tool_status_summary(records),
        "",
        "## Timing Evidence",
        "",
    ]
    plain = _find_step(records, "plain")
    if plain is None:
        lines.append("Plain JSONL timing evidence was not produced.")
    else:
        lines.extend(_plain_timing_summary(Path(str(plain.get("jsonl", "")))))
        lines.extend(["", "## Tooling Slowdown", ""])
        lines.extend(_tooling_slowdown_summary(plain, records))
    lines.extend(["", "## Python Attribution", ""])
    backend_names = _summary_backend_names(records)
    py_spy_count = 0
    for backend in backend_names:
        backend_lines: list[str] = []
        low = _find_step(records, "py-spy-raw-low-impact", backend=backend)
        full = _find_step(records, "py-spy-raw-full", backend=backend)
        generic = _find_step(records, "py-spy-raw", backend=backend) if low is None and full is None else None
        if low is not None:
            backend_lines.extend(_py_spy_summary(low, title="Low-impact py-spy", heading="####"))
        if full is not None:
            backend_lines.extend(_py_spy_summary(full, title="Full sampled py-spy", heading="####"))
        if generic is not None:
            backend_lines.extend(_py_spy_summary(generic, title="py-spy", heading="####"))
        if backend_lines:
            lines.extend([f"### {backend}", "", *backend_lines])
            py_spy_count += 1
    if py_spy_count == 0:
        lines.append("No py-spy artifacts were produced.")
    lines.extend(["", "## Deterministic Python Calls", ""])
    cprofile_count = 0
    for backend in backend_names:
        cprofile = _find_step(records, "cprofile", backend=backend)
        if cprofile is not None:
            lines.extend([f"### {backend}", ""])
            lines.extend(_cprofile_summary(cprofile))
            cprofile_count += 1
    if cprofile_count == 0:
        lines.append("cProfile was not run. Use `--include-cprofile` when deterministic Python call counts are worth the slowdown.")
    lines.extend(["", "## Native Attribution", ""])
    perf_count = 0
    for backend in backend_names:
        perf = _find_step(records, "perf-record", backend=backend)
        if perf is not None:
            lines.extend([f"### {backend}", ""])
            lines.extend(_perf_summary(perf))
            perf_count += 1
    if perf_count == 0:
        lines.append("perf was not run.")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _run_metadata_summary(summary: dict[str, object]) -> list[str]:
    versions = dict(summary.get("tool_versions", {}) or {})
    return [
        f"- Revision: `{summary.get('repository_revision', 'unknown')}`",
        f"- Dirty worktree: `{summary.get('repository_dirty')}`",
        "- Tools: "
        + ", ".join(
            f"`{name} {versions.get(name, 'unknown')}`"
            for name in ("python", "PySide6", "pyqtgraph", "vispy", "py-spy", "perf")
        ),
    ]


def _tool_status_summary(records: list[dict[str, object]]) -> list[str]:
    rows = [
        "| Step | backend | required | status | rc | artifacts | profiler health |",
        "|---|---|---:|---|---:|---|---|",
    ]
    for record in records:
        rows.append(
            "| "
            + " | ".join(
                (
                    f"`{record.get('step_id', record.get('profiler_type', ''))}`",
                    f"`{record.get('backend', '')}`",
                    str(bool(record.get("required", False))),
                    str(record.get("status", "")),
                    str(record.get("returncode", "")),
                    _artifact_status(record),
                    _profiler_health(record),
                )
            )
            + " |"
        )
    return rows


def _artifact_status(record: dict[str, object]) -> str:
    artifacts = tuple(record.get("artifact_paths", ()) or ())
    missing = tuple(record.get("missing_artifacts", ()) or ())
    if not artifacts:
        return "none"
    if missing:
        return f"{len(artifacts) - len(missing)}/{len(artifacts)} present"
    return f"{len(artifacts)}/{len(artifacts)} present"


def _profiler_health(record: dict[str, object]) -> str:
    diagnostics = dict(record.get("profiler_diagnostics", {}) or {})
    profiler = str(record.get("profiler_type", ""))
    if profiler.startswith("py-spy"):
        return (
            f"samples {diagnostics.get('sample_count', 0)}, "
            f"errors {diagnostics.get('error_count', 0)}, "
            f"missed {diagnostics.get('missed_stack_count', 0)}"
        )
    warning_count = diagnostics.get("warning_count")
    if warning_count is not None:
        return f"warnings {warning_count}"
    return str(diagnostics.get("scope", "n/a"))


def _find_step(records: list[dict[str, object]], profiler_type: str, *, backend: str | None = None) -> dict[str, object] | None:
    for record in records:
        record_backend = str(record.get("backend", "all") or "all")
        if backend is not None and record_backend != str(backend):
            continue
        if str(record.get("profiler_type", "")).startswith(profiler_type):
            return record
    return None


def _summary_backend_names(records: list[dict[str, object]]) -> tuple[str, ...]:
    names = sorted({str(record.get("backend", "")) for record in records if str(record.get("backend", "")) not in {"", "all"}})
    return tuple(names) if names else ("all",)


def _plain_timing_summary(path: Path) -> list[str]:
    records = _read_jsonl(path)
    if not records:
        return [f"No readable timing records found at `{path}`."]
    lines = [
        f"Source: `{path}`",
        "",
        "Headline only; detailed counters remain in JSONL.",
        "",
        "| Backend | phase | temp | pacing ms | event-loop gap ms | tiles | work |",
        "|---|---|---|---|---|---:|---|",
    ]
    for record in records:
        backend = str(record.get("backend", ""))
        phase = str(record.get("phase", ""))
        temperature = str(record.get("run_temperature", ""))
        pacing = _pacing_summary(record)
        gaps = _event_loop_summary(record)
        tiles = _tile_summary(record)
        work = _work_summary(record)
        lines.append(f"| `{backend}` | `{phase}` | `{temperature}` | {pacing} | {gaps} | {tiles} | {work} |")
    return lines


def _py_spy_summary(record: dict[str, object], *, title: str, heading: str = "###") -> list[str]:
    diagnostics = dict(record.get("profiler_diagnostics", {}) or {})
    lines = [
        f"{heading} {title}",
        "",
        f"- Status: `{record.get('status')}`",
        f"- Mode: `{diagnostics.get('sampling_mode', 'unknown')}`",
        f"- Samples/errors: `{diagnostics.get('sample_count', 0)}` / `{diagnostics.get('error_count', 0)}`",
        f"- Missed stacks: `{diagnostics.get('missed_stack_count', 0)}`",
    ]
    artifacts = tuple(record.get("artifact_paths", ()) or ())
    raw_paths = [Path(str(path)) for path in artifacts if str(path).endswith(".raw")]
    if raw_paths:
        top = _top_py_spy_stacks(raw_paths[0])
        if top:
            lines.extend(["", "Top sampled stacks (cropped to leaf context):"])
            for stack, count in top:
                lines.append(f"- `{count}` {stack}")
        else:
            lines.append(f"- No readable raw stack samples found in `{raw_paths[0]}`.")
    return lines


def _tooling_slowdown_summary(plain: dict[str, object], records: list[dict[str, object]]) -> list[str]:
    plain_phases = _backend_phase_elapsed_map(Path(str(plain.get("jsonl", ""))))
    if not plain_phases:
        return ["No readable plain timing records were available for slowdown comparisons."]
    rows: list[str] = []
    for record in records:
        profiler_type = str(record.get("profiler_type", ""))
        if profiler_type == "plain":
            continue
        phases = _backend_phase_elapsed_map(Path(str(record.get("jsonl", ""))))
        record_backend = str(record.get("backend", "") or "")
        compared_backends = (record_backend,) if record_backend and record_backend != "all" else tuple(sorted(plain_phases))
        for backend in compared_backends:
            if backend not in plain_phases:
                continue
            baseline = plain_phases.get(backend, {})
            values = phases.get(backend, {})
            rows.append(
                "| "
                + " | ".join(
                    (
                        f"`{profiler_type}`",
                        f"`{backend}`",
                        str(record.get("status", "unknown")),
                        _delta_cell(values.get("raw_full_tiled_montage"), baseline.get("raw_full_tiled_montage")),
                        _delta_cell(values.get("fft_full_tiled_montage"), baseline.get("fft_full_tiled_montage")),
                        _delta_cell(_combined_elapsed_ms(values), _combined_elapsed_ms(baseline)),
                    )
                )
                + " |"
            )
    if not rows:
        return ["No tooled runs were available for slowdown comparisons."]
    return [
        "Compared with the non-tooled `plain` workflow. Negative values mean the tooled run happened to finish faster.",
        "",
        "| Tool | backend | status | normal delta | FFT delta | combined delta |",
        "|---|---|---|---:|---:|---:|",
        *rows,
    ]


def _cprofile_summary(record: dict[str, object]) -> list[str]:
    artifacts = tuple(record.get("artifact_paths", ()) or ())
    profiles = [Path(str(path)) for path in artifacts if str(path).endswith(".cprofile")]
    if not profiles or not profiles[0].exists():
        return ["cProfile was requested, but no readable `.cprofile` artifact was found."]
    stream = io.StringIO()
    try:
        stats = pstats.Stats(str(profiles[0]), stream=stream).strip_dirs().sort_stats("cumulative")
        stats.print_stats(8)
    except Exception as exc:
        return [f"Could not read cProfile artifact `{profiles[0]}`: {exc}"]
    lines = [f"Source: `{profiles[0]}`", "", "Top cumulative Python call entries (cropped):", "```text"]
    lines.extend(stream.getvalue().strip().splitlines()[:14])
    lines.append("```")
    return lines


def _perf_summary(record: dict[str, object]) -> list[str]:
    artifacts = tuple(record.get("artifact_paths", ()) or ())
    perf_paths = [Path(str(path)) for path in artifacts if str(path).endswith(".data")]
    if not perf_paths or not perf_paths[0].exists():
        return ["perf was requested, but no readable `perf.data` artifact was found."]
    completed = None
    if shutil.which("perf") is not None:
        try:
            completed = subprocess.run(
                ("perf", "report", "--stdio", "-i", str(perf_paths[0]), "--no-children", "--sort", "comm,dso,symbol"),
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            completed = None
    lines = [f"Source: `{perf_paths[0]}`"]
    if completed is None or completed.returncode != 0:
        lines.append("Run `perf report -i <artifact>` locally for native attribution.")
        return lines
    report_lines = _interesting_perf_lines(completed.stdout.splitlines())[:12]
    lines.extend(["", "Top perf report lines (cropped):", "```text", *report_lines, "```"])
    return lines


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []


def _backend_phase_elapsed_map(path: Path) -> dict[str, dict[str, float]]:
    phases: dict[str, dict[str, float]] = {}
    for record in _read_jsonl(path):
        backend = str(record.get("backend", "unknown"))
        phase = str(record.get("phase", ""))
        try:
            phases.setdefault(backend, {})[phase] = float(record["elapsed_ms"])
        except Exception:
            continue
    return phases


def _combined_elapsed_ms(phases: dict[str, float]) -> float | None:
    raw = phases.get("raw_full_tiled_montage")
    fft = phases.get("fft_full_tiled_montage")
    if raw is None or fft is None:
        return None
    return raw + fft


def _delta_cell(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None or baseline == 0:
        return "n/a"
    delta = value - baseline
    percent = (delta / baseline) * 100.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f} ms ({sign}{percent:.1f}%)"


def _top_py_spy_stacks(path: Path, *, limit: int = 8) -> list[tuple[str, int]]:
    stacks: list[tuple[str, int]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        stack, sep, raw_count = line.rpartition(" ")
        if not sep:
            continue
        try:
            count = int(raw_count)
        except ValueError:
            continue
        frames = [frame.strip() for frame in stack.split(";") if frame.strip()]
        if not frames:
            continue
        interesting = " -> ".join(frames[-4:]) if frames else stack
        stacks.append((interesting, count))
    stacks.sort(key=lambda item: item[1], reverse=True)
    return stacks[:limit]


def _interesting_perf_lines(lines: list[str]) -> list[str]:
    result = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "%" not in stripped and "Overhead" not in stripped:
            continue
        result.append(line[:180])
    return result


def _format_ms(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}"
    except Exception:
        return "n/a"


def _percentile(values: tuple[float, ...], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (float(percentile) / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _tile_summary(record: dict[str, object]) -> str:
    tile_count = record.get("tile_count")
    full = record.get("full_tile_count")
    requested = record.get("requested_tile_count")
    presented = record.get("active_presented_tile_count")
    if requested is not None and presented is not None:
        return f"{presented}/{requested}"
    if tile_count is not None and full is not None:
        return f"{tile_count}/{full}"
    return "n/a"


def _pacing_summary(record: dict[str, object]) -> str:
    first = record.get("first_loaded_tile_ms") or record.get("first_display_committed_ms")
    exact_visible = record.get("fully_visible_ms")
    draw = record.get("draw_after_complete_ms")
    full = draw if draw is not None else record.get("elapsed_ms")
    return " / ".join(
        (
            f"first {_format_ms(first)}",
            f"visible {_format_ms(exact_visible)}",
            f"full {_format_ms(full)}",
        )
    )


def _event_loop_summary(record: dict[str, object]) -> str:
    p95 = _format_ms(record.get("event_loop_p95_gap_ms"))
    p99 = _format_ms(record.get("event_loop_p99_gap_ms"))
    max_gap = _format_ms(record.get("event_loop_max_gap_ms"))
    return f"p95 {p95} / p99 {p99} / max {max_gap}"


def _work_summary(record: dict[str, object]) -> str:
    parts = []
    compute = _compute_summary(record)
    if compute != "n/a":
        parts.append(compute)
    upload_bytes = _format_bytes(record.get("tile_layer_texture_upload_bytes"))
    uploads = record.get("tile_layer_texture_uploads")
    if uploads is not None:
        parts.append(f"upload {uploads} / {upload_bytes}")
    updated = record.get("tile_layer_items_updated")
    skipped = record.get("tile_layer_items_skipped")
    if updated is not None and skipped is not None:
        parts.append(f"items up {updated}, skip {skipped}")
    level_updates = record.get("tile_layer_level_updates")
    vertex_uploads = record.get("tile_layer_vertex_uploads")
    if level_updates is not None or vertex_uploads is not None:
        parts.append(f"rebind level {level_updates or 0}, vertex {vertex_uploads or 0}")
    pages = record.get("tile_layer_active_pages")
    gpu_bytes = record.get("tile_layer_estimated_gpu_bytes")
    if pages is not None and gpu_bytes is not None:
        parts.append(f"resident {pages} pages / {_format_bytes(gpu_bytes)}")
    return "; ".join(parts) if parts else "n/a"


def _compute_summary(record: dict[str, object]) -> str:
    direct = record.get("montage_tile_compute_direct")
    stage = record.get("montage_tile_compute_stage_backed")
    cache = record.get("montage_tile_compute_cache_hits")
    parts = []
    if direct is not None:
        parts.append(f"direct {direct}")
    if stage is not None:
        parts.append(f"stage {stage}")
    if cache is not None:
        parts.append(f"cache {cache}")
    return ", ".join(parts) if parts else "n/a"


def _format_bytes(value) -> str:
    if value is None:
        return "n/a"
    try:
        amount = float(value)
    except Exception:
        return "n/a"
    units = ("B", "KiB", "MiB", "GiB")
    unit = units[0]
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    if unit == "B":
        return f"{int(amount)} {unit}"
    return f"{amount:.1f} {unit}"


def _suite_step_temperature(previous_records: list[dict[str, object]]) -> str:
    return "cold" if not previous_records else "warm"


def _aggregate_run_temperature(temperatures: tuple[str, ...]) -> str:
    observed = {temperature for temperature in temperatures if temperature in {"cold", "warm", "mixed"}}
    if not observed:
        return "mixed"
    if len(observed) == 1:
        return next(iter(observed))
    return "mixed"


def _workflow_run_temperature() -> str:
    return "mixed"


def _repository_state() -> dict[str, object]:
    revision = _run_text_command(("git", "rev-parse", "--verify", "HEAD"))
    dirty = bool(_run_text_command(("git", "status", "--short")))
    return {
        "repository_revision": revision or "unknown",
        "repository_dirty": dirty,
    }


def _suite_tool_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "arrayscope": _package_version("ArrayScope"),
        "numpy": getattr(np, "__version__", ""),
        "py-spy": _run_text_command(("py-spy", "--version")) if shutil.which("py-spy") else "unavailable",
        "perf": _run_text_command(("perf", "--version")) if shutil.which("perf") else "unavailable",
    }
    for package in ("PySide6", "pyqtgraph", "vispy", "nibabel"):
        versions[package] = _package_version(package)
    return versions


def _package_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _run_text_command(command: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5)
    except Exception:
        return ""
    output = (completed.stdout or completed.stderr or "").strip()
    return output.splitlines()[0] if output else ""


def _suite_child_args(argv: tuple[str, ...]) -> tuple[str, ...]:
    blocked = {"--profile-suite", "--print-py-spy-command", "--py-spy-native", "--include-cprofile"}
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


def _suite_profiler_backends(argv: tuple[str, ...]) -> tuple[str, ...]:
    backend = "pyqtgraph"
    args = tuple(argv)
    for index, arg in enumerate(args):
        if arg == "--backend" and index + 1 < len(args):
            backend = str(args[index + 1])
            break
        if str(arg).startswith("--backend="):
            backend = str(arg).split("=", 1)[1]
            break
    if backend == "all":
        return ("pyqtgraph", "vispy")
    return (backend,)


def _suite_args_for_backend(argv: tuple[str, ...], backend: str) -> tuple[str, ...]:
    args = tuple(argv)
    result: list[str] = []
    replaced = False
    skip_next = False
    for index, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--backend":
            result.extend(("--backend", str(backend)))
            replaced = True
            skip_next = index + 1 < len(args)
            continue
        if str(arg).startswith("--backend="):
            result.append(f"--backend={backend}")
            replaced = True
            continue
        result.append(str(arg))
    if not replaced:
        result.extend(("--backend", str(backend)))
    return tuple(result)


def _py_spy_native_requested(argv: tuple[str, ...]) -> bool:
    return any(str(arg) == "--py-spy-native" for arg in tuple(argv))


def _include_cprofile_requested(argv: tuple[str, ...]) -> bool:
    return any(str(arg) == "--include-cprofile" for arg in tuple(argv))


def _py_spy_filtered_args(argv: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(arg for arg in tuple(argv) if str(arg) != "--py-spy-native")


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the realistic tiled montage + FFT profiling workflow")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Dataset path; defaults to the bundled realistic NIfTI")
    parser.add_argument("--backend", choices=("pyqtgraph", "vispy", "all"), default="pyqtgraph")
    parser.add_argument("--jsonl", default=None, help="Optional JSONL metrics output")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-tiles", type=int, default=0, help="Optional tile cap for local smoke runs; 0 means full dim 2")
    parser.add_argument("--columns", type=int, default=0, help="Montage columns; 0 chooses a near-square layout")
    parser.add_argument("--load-mode", choices=("app", "native"), default="app")
    parser.add_argument("--print-py-spy-command", action="store_true", help="Print an external py-spy command for this invocation and exit")
    parser.add_argument("--profile-suite", default=None, help="Run plain JSONL, py-spy raw, and perf record into this directory")
    parser.add_argument("--include-cprofile", action="store_true", help="Include cProfile call-count attribution; slower and not timing evidence")
    parser.add_argument(
        "--py-spy-native",
        action="store_true",
        help="Include native stacks in py-spy artifacts; useful diagnostically but not pacing evidence",
    )
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
    for backend in (("pyqtgraph", "vispy") if args.backend == "all" else (args.backend,)):
        run_profile_montage_workflow(
            data_path=args.data,
            backend=backend,
            jsonl=jsonl,
            timeout_s=args.timeout_s,
            max_tiles=None if args.max_tiles <= 0 else args.max_tiles,
            columns=None if args.columns <= 0 else args.columns,
            load_mode=args.load_mode,
            profiler_type=args.profiler_type,
            profiler_artifact_paths=tuple(args.profiler_artifact or ()),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
