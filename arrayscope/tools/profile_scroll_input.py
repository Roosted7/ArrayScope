"""Profile high-frequency dimension scrolling against a real ArrayScope window."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from arrayscope.tools.interaction_budget import (
    INTERACTION_SETTLE_HARD_LIMIT_S,
    bounded_interaction_settle_timeout_s,
)

DEFAULT_DATA_PATH = Path("data/_WIPDelRec-tT2_20260223150234_14.nii")


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile 60Hz dimension scrolling in a real ArrayScope window"
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--backend", choices=("pyqtgraph", "vispy"), default="vispy")
    parser.add_argument("--axis", type=int, default=2)
    parser.add_argument("--image-y", type=int, default=1)
    parser.add_argument("--image-x", type=int, default=0)
    parser.add_argument("--start", type=int, default=120)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--ticks", type=int, default=180)
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument(
        "--ops",
        choices=("none", "fft", "fftshift", "fft-fftshift", "fft-ifft"),
        default="fft-fftshift",
    )
    parser.add_argument("--load-mode", choices=("app", "native"), default="app")
    parser.add_argument("--artifact-dir", default="tests/artifacts/scroll-input")
    parser.add_argument("--jsonl", default=None)
    args = parser.parse_args(argv)

    from arrayscope.app.qt_binding import prefer_pyside6

    prefer_pyside6()
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore

    from arrayscope.app.settings_state import ImageRenderingBackendChoice
    from arrayscope.window import ArrayScopeWindow

    app = pg.mkQApp()
    data_path = Path(args.data)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(args.jsonl) if args.jsonl else artifact_dir / "scroll-input.jsonl"
    cprofile_path = artifact_dir / "scroll-input.cprofile"
    summary_path = artifact_dir / "scroll-input-summary.txt"

    settings = QtCore.QSettings()
    previous_backend = settings.value("image_rendering_backend", None)
    settings.setValue(
        "image_rendering_backend",
        ImageRenderingBackendChoice.VISPY.value
        if args.backend == "vispy"
        else ImageRenderingBackendChoice.PYQTGRAPH.value,
    )
    settings.sync()

    win = None
    try:
        load_start = perf_counter()
        data = _load_dataset(data_path, mode=args.load_mode)
        load_ms = (perf_counter() - load_start) * 1000.0
        shape = tuple(int(value) for value in np.shape(data))
        axis = int(args.axis)
        if axis < 0 or axis >= len(shape):
            raise ValueError(f"axis {axis} is outside shape {shape}")
        stop = min(shape[axis], int(args.start) + max(1, int(args.count)))
        indices = tuple(range(max(0, int(args.start)), stop))
        if not indices:
            raise ValueError("empty scroll index range")

        win = ArrayScopeWindow(data, filepath=data_path)
        win.app_settings = _replace_backend(
            win.app_settings, args.backend, ImageRenderingBackendChoice
        )
        apply_theme = getattr(win, "_apply_theme_choice", None)
        if callable(apply_theme):
            apply_theme(win.app_settings.theme, persist=False)
        win.resize(1400, 900)
        win.show()
        _process_events(app, QtCore, count=30)

        operations = _operations_for(args.ops, axis)
        if operations:
            win.operation_coordinator.load_operations(operations)
            win._set_document(win.operation_coordinator.document)
            win._coerce_channel_for_current_dtype()

        state = (
            win.view_state.with_image_axes(int(args.image_y), int(args.image_x))
            .with_montage_axis(None)
            .with_slice(axis, indices[0])
        )
        win._set_view_state(state)
        win.render(reason="scroll-profile-initial")
        _wait_idle(app, QtCore, win, timeout_s=INTERACTION_SETTLE_HARD_LIMIT_S)

        warm_stats = _warm_cache(app, QtCore, win, axis=axis, indices=indices)
        _wait_idle(app, QtCore, win, timeout_s=INTERACTION_SETTLE_HARD_LIMIT_S)

        render_records: list[dict[str, object]] = []
        original_render = win.render

        def profiled_render(**kwargs):
            start = perf_counter()
            try:
                return original_render(**kwargs)
            finally:
                render_records.append(
                    {
                        "reason": str(kwargs.get("reason", "")),
                        "elapsed_ms": (perf_counter() - start) * 1000.0,
                        "slice_index": int(win.view_state.slice_indices[axis]),
                        "pending_draw": _presentation_pending(win),
                    }
                )

        win.render = profiled_render
        before = _coordinator_snapshot(win)
        tick_records: list[dict[str, object]] = []
        profiler = cProfile.Profile()
        # Timer category: UI cosmetic. Probe heartbeat drives synthetic input
        # and measures latency; it is outside production scheduling.
        timer = QtCore.QTimer()
        timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        interval_ms = max(1, round(1000.0 / max(1.0, float(args.hz))))
        timer.setInterval(interval_ms)
        state_box = {"tick": 0, "last": None, "start": 0.0}

        def on_tick() -> None:
            now = perf_counter()
            tick = int(state_box["tick"])
            previous = state_box["last"]
            index = int(indices[tick % len(indices)])
            state_box["last"] = now
            state_box["tick"] = tick + 1
            pending_before = _presentation_pending(win)
            request_before = _coordinator_snapshot(win)
            win._on_slice_index_changed(axis, index)
            request_after = _coordinator_snapshot(win)
            tick_records.append(
                {
                    "tick": tick,
                    "index": index,
                    "dt_ms": 0.0 if previous is None else (now - float(previous)) * 1000.0,
                    "handler_ms": (perf_counter() - now) * 1000.0,
                    "pending_before": pending_before,
                    "pending_after": _presentation_pending(win),
                    "requested_delta": int(request_after["requested"])
                    - int(request_before["requested"]),
                    "flushed_delta": int(request_after["flushed"]) - int(request_before["flushed"]),
                    "backpressure_delta": int(request_after["presentation_backpressure_skips"])
                    - int(request_before["presentation_backpressure_skips"]),
                    "coalesced_delta": int(request_after["coalesced"])
                    - int(request_before["coalesced"]),
                }
            )
            if state_box["tick"] >= int(args.ticks):
                timer.stop()
                # Timer category: UI cosmetic. Probe shutdown grace period.
                QtCore.QTimer.singleShot(500, app, app.quit)

        timer.timeout.connect(on_tick)
        state_box["start"] = perf_counter()
        profiler.enable()
        timer.start()
        app.exec()
        profiler.disable()
        _process_events(app, QtCore, count=20)
        profiler.dump_stats(str(cprofile_path))
        after = _coordinator_snapshot(win)
        summary = _summary(
            args=args,
            data_path=data_path,
            shape=shape,
            load_ms=load_ms,
            indices=indices,
            warm_stats=warm_stats,
            before=before,
            after=after,
            tick_records=tick_records,
            render_records=render_records,
            cprofile_path=cprofile_path,
        )
        _write_jsonl(jsonl_path, summary, tick_records, render_records)
        _write_profile_summary(cprofile_path, summary_path)
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(summary_path.read_text(encoding="utf-8"))
        return 0
    finally:
        if win is not None:
            win.close()
            _process_events(app, QtCore, count=10)
        _restore_setting(settings, "image_rendering_backend", previous_backend)


def _load_dataset(path: Path, *, mode: str):
    if mode == "app":
        from arrayscope.io.file_interpreters import load_path

        return load_path(path).data
    if _is_nifti(path):
        import nibabel as nib

        return np.asanyarray(nib.load(str(path)).dataobj)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    raise ValueError(f"unsupported data path {path}")


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".nii", ".nii.gz"))


def _operations_for(name: str, axis: int):
    from arrayscope.operations.pipeline import CenteredFFT, CenteredIFFT, FFTShift

    if name == "none":
        return ()
    if name == "fft":
        return (CenteredFFT(axis=axis),)
    if name == "fftshift":
        return (FFTShift(axis=axis),)
    if name == "fft-fftshift":
        return (CenteredFFT(axis=axis), FFTShift(axis=axis))
    if name == "fft-ifft":
        return (CenteredFFT(axis=axis), CenteredIFFT(axis=axis))
    raise ValueError(name)


def _replace_backend(settings, backend: str, image_choice):
    from dataclasses import replace

    return replace(
        settings,
        image_rendering_backend=image_choice.VISPY
        if backend == "vispy"
        else image_choice.PYQTGRAPH,
    )


def _process_events(app, QtCore, *, count: int) -> None:
    flags = QtCore.QEventLoop.ProcessEventsFlag.AllEvents
    for _ in range(max(1, int(count))):
        app.processEvents(flags, 50)


def _warm_cache(app, QtCore, win, *, axis: int, indices: tuple[int, ...]) -> dict[str, object]:
    start = perf_counter()
    for index in indices:
        win._set_view_state(win.view_state.with_slice(axis, int(index)))
        win.render(reason="scroll-profile-warm")
        _wait_idle(app, QtCore, win, timeout_s=INTERACTION_SETTLE_HARD_LIMIT_S)
    return {"count": len(indices), "elapsed_ms": (perf_counter() - start) * 1000.0}


def _wait_idle(app, QtCore, win, *, timeout_s: float) -> None:
    timeout_s = bounded_interaction_settle_timeout_s(timeout_s)
    start = perf_counter()
    while perf_counter() - start < timeout_s:
        _process_events(app, QtCore, count=3)
        if not _presentation_pending(win) and not _controller_busy(win):
            return
    raise TimeoutError("window did not become idle")


def _presentation_pending(win) -> bool:
    predicate = getattr(getattr(win, "img_view", None), "presentationDrawPending", None)
    return bool(callable(predicate) and predicate())


def _controller_busy(win) -> bool:
    for name in (
        "visible_evaluation_controller",
        "montage_tile_evaluation_controller",
        "stage_evaluation_controller",
        "histogram_evaluation_controller",
    ):
        controller = getattr(win, name, None)
        if controller is not None and controller.is_busy():
            return True
    coordinator = getattr(win, "render_coordinator", None)
    return bool(coordinator is not None and getattr(coordinator, "has_pending_render", False))


def _coordinator_snapshot(win) -> dict[str, int]:
    coordinator = getattr(win, "render_coordinator", None)
    if coordinator is None:
        return {}
    return {
        "requested": int(getattr(coordinator, "requested", 0)),
        "flushed": int(getattr(coordinator, "flushed", 0)),
        "coalesced": int(getattr(coordinator, "coalesced", 0)),
        "immediate_cache_flushes": int(getattr(coordinator, "immediate_cache_flushes", 0)),
        "presentation_backpressure_skips": int(
            getattr(coordinator, "presentation_backpressure_skips", 0)
        ),
    }


def _summary(
    *,
    args,
    data_path: Path,
    shape: tuple[int, ...],
    load_ms: float,
    indices: tuple[int, ...],
    warm_stats: dict[str, object],
    before: dict[str, int],
    after: dict[str, int],
    tick_records: list[dict[str, object]],
    render_records: list[dict[str, object]],
    cprofile_path: Path,
) -> dict[str, object]:
    dts = [float(row["dt_ms"]) for row in tick_records if float(row["dt_ms"]) > 0.0]
    handler = [float(row["handler_ms"]) for row in tick_records]
    renders = [float(row["elapsed_ms"]) for row in render_records]
    delta = {
        key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in set(before) | set(after)
    }
    return {
        "data": str(data_path),
        "backend": str(args.backend),
        "ops": str(args.ops),
        "shape": list(shape),
        "axis": int(args.axis),
        "indices": [int(indices[0]), int(indices[-1]), len(indices)],
        "target_hz": float(args.hz),
        "target_ticks": int(args.ticks),
        "ticks": len(tick_records),
        "load_ms": float(load_ms),
        "warm": warm_stats,
        "timer_dt_mean_ms": _mean(dts),
        "timer_dt_p95_ms": _percentile(dts, 95),
        "timer_dt_max_ms": max(dts) if dts else 0.0,
        "handler_mean_ms": _mean(handler),
        "handler_p95_ms": _percentile(handler, 95),
        "handler_max_ms": max(handler) if handler else 0.0,
        "render_count": len(render_records),
        "render_mean_ms": _mean(renders),
        "render_p95_ms": _percentile(renders, 95),
        "render_max_ms": max(renders) if renders else 0.0,
        "coordinator_delta": delta,
        "tick_backpressure_events": sum(
            1 for row in tick_records if int(row["backpressure_delta"]) > 0
        ),
        "tick_flushed_events": sum(1 for row in tick_records if int(row["flushed_delta"]) > 0),
        "cprofile": str(cprofile_path),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((float(percentile) / 100.0) * (len(ordered) - 1))))
    return float(ordered[index])


def _write_jsonl(path: Path, summary: dict[str, object], ticks, renders) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "summary", **summary}, sort_keys=True) + "\n")
        for row in ticks:
            handle.write(json.dumps({"event": "tick", **row}, sort_keys=True) + "\n")
        for row in renders:
            handle.write(json.dumps({"event": "render", **row}, sort_keys=True) + "\n")


def _write_profile_summary(profile_path: Path, summary_path: Path) -> None:
    with summary_path.open("w", encoding="utf-8") as handle:
        stats = pstats.Stats(str(profile_path), stream=handle).strip_dirs().sort_stats("cumulative")
        stats.print_stats(35)


def _restore_setting(settings, key: str, value) -> None:
    if value is None:
        settings.remove(key)
    else:
        settings.setValue(key, value)
    settings.sync()


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))
