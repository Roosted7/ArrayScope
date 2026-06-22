"""Generate deterministic release-candidate diagnostics artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import numpy as np


def capture_release_diagnostics(path, *, backend: str = "pyqtgraph", interval_ms: int = 500) -> Path:
    """Capture a small real-window diagnostics JSONL trace for RC evidence."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from arrayscope import __version__
    from arrayscope.app.qt_binding import prefer_pyside6

    prefer_pyside6()
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore

    from arrayscope.ui.diagnostics_logging import DiagnosticsJsonlLogger
    from arrayscope.window import ArrayScopeWindow

    app = pg.mkQApp()
    backend = _normalize_backend(backend)
    settings = QtCore.QSettings()
    previous_backend = settings.value("image_rendering_backend", None)
    settings.setValue("image_rendering_backend", backend)
    settings.sync()

    logger = DiagnosticsJsonlLogger(path)
    win = None
    try:
        data = _release_dataset()
        win = ArrayScopeWindow(data)
        _process_events(app, QtCore, count=20)

        logger.start(win.collect_runtime_diagnostics(), app_version=__version__, interval_ms=interval_ms)

        win.operation_evaluator.image(win.view_state)
        win.render(reason="release-diagnostics-image")
        _process_events(app, QtCore, count=30)
        logger.write_snapshot(win.collect_runtime_diagnostics())

        state = win.view_state.with_montage_axis(2, columns=3, indices=tuple(range(data.shape[2])), text=":")
        win._set_view_state(state)
        win.render(reason="release-diagnostics-montage")
        _wait_until(app, QtCore, lambda: getattr(win, "_current_montage_geometry", None) is not None, timeout_s=3.0)
        logger.write_snapshot(win.collect_runtime_diagnostics())
    finally:
        logger.close()
        if win is not None:
            win.close()
            _process_events(app, QtCore, count=10)
        if previous_backend is None:
            settings.remove("image_rendering_backend")
        else:
            settings.setValue("image_rendering_backend", previous_backend)
        settings.sync()

    return logger.path


def _release_dataset() -> np.ndarray:
    y = np.linspace(-1.0, 1.0, 12, dtype=np.float32)[:, None, None]
    x = np.linspace(0.0, 2.0, 14, dtype=np.float32)[None, :, None]
    z = np.arange(6, dtype=np.float32)[None, None, :]
    return (np.sin(x + z * 0.2) + np.cos(y - z * 0.1)).astype(np.float32, copy=False)


def _normalize_backend(backend: str) -> str:
    backend = str(backend).strip().lower()
    if backend not in {"pyqtgraph", "vispy"}:
        raise ValueError(f"unsupported backend {backend!r}; expected 'pyqtgraph' or 'vispy'")
    return backend


def _process_events(app, QtCore, *, count: int) -> None:
    flags = QtCore.QEventLoop.ProcessEventsFlag.AllEvents
    for _ in range(max(1, int(count))):
        app.processEvents(flags, 50)


def _wait_until(app, QtCore, predicate, *, timeout_s: float) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        _process_events(app, QtCore, count=2)
        if predicate():
            return
        time.sleep(0.01)


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture deterministic ArrayScope RC diagnostics JSONL")
    parser.add_argument("--jsonl", required=True, help="Path for the diagnostics JSONL artifact")
    parser.add_argument("--backend", default="pyqtgraph", choices=("pyqtgraph", "vispy"))
    parser.add_argument("--interval-ms", type=int, default=500)
    args = parser.parse_args(argv)

    path = capture_release_diagnostics(args.jsonl, backend=args.backend, interval_ms=args.interval_ms)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
