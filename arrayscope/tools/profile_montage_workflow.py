"""Profile a realistic full-montage workflow in a real ArrayScope window."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shlex
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
            columns=columns,
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
    logical_complete_tile_request_count = None
    draw_after_complete_ms = None
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
            logical_complete_tile_request_count = _vispy_tile_presentation_request_count(win)
        tile_drawn = _vispy_tile_presentation_draw_count(win) >= int(logical_complete_tile_request_count or 0)
        if logical_complete and (not vispy_tiled or tile_drawn):
            if vispy_tiled:
                draw_after_complete_ms = (perf_counter() - start) * 1000.0
            return {
                "first_loaded_tile_ms": first_loaded_ms,
                "first_display_committed_ms": first_display_committed_ms,
                "first_overlay_clear_ms": first_overlay_clear_ms,
                "logical_complete_ms": first_logical_complete_ms,
                "draw_after_complete_ms": draw_after_complete_ms,
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
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "backend": backend,
        "data_path": str(data_path),
        "load_mode": str(load_mode),
        "data_shape": tuple(int(value) for value in np.shape(data)),
        "data_dtype": str(getattr(getattr(data, "dtype", None), "str", getattr(data, "dtype", ""))),
        "montage_axis": int(montage_axis),
        "tile_count": len(indices),
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
    args = parser.parse_args(argv)

    if args.print_py_spy_command:
        filtered = tuple(arg for arg in (argv if argv is not None else sys.argv[1:]) if arg != "--print-py-spy-command")
        print(py_spy_command(filtered))
        return 0

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
            )
        )
    for record in records:
        print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
