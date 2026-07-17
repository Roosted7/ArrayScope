"""Generate deterministic release-candidate diagnostics artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import numpy as np

from arrayscope.tools.interaction_budget import bounded_interaction_settle_timeout_s
from arrayscope.tools.presentation_settlement import (
    PresentationTargetToken,
    presentation_is_settled,
    presentation_settlement_diagnostic,
    presentation_target_token,
)


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
        # Physical draw completion is a capture prerequisite; a hidden widget
        # can acknowledge payloads without ever receiving the paint that
        # clears ``presentationDrawPending``.
        win.show()
        _process_events(app, QtCore, count=20)

        # A geometry object only proves that layout planning ran.  Release
        # evidence starts after the current target has complete acknowledged
        # backend coverage and the corresponding draw has reached the screen.
        current_target = _wait_for_capture_presentation(
            app,
            QtCore,
            win,
            phase="initial image",
        )

        logger.start(win.collect_runtime_diagnostics(), app_version=__version__, interval_ms=interval_ms)

        # Exercise a real image-target change.  Re-requesting the identical
        # cached target can legitimately coalesce to no new pixels and is not
        # useful release evidence.
        image_index = (int(win.view_state.slice_indices[2]) + 1) % int(data.shape[2])
        win._set_view_state(win.view_state.with_slice(2, image_index))
        win.render(reason="release-diagnostics-image")
        current_target = _wait_for_capture_presentation(
            app,
            QtCore,
            win,
            phase="image render",
            previous_target=current_target,
        )
        logger.write_snapshot(win.collect_runtime_diagnostics())

        state = win.view_state.with_montage_axis(2, columns=3, indices=tuple(range(data.shape[2])), text=":")
        win._set_view_state(state)
        win.render(reason="release-diagnostics-montage")
        _wait_for_capture_presentation(
            app,
            QtCore,
            win,
            phase="montage render",
            previous_target=current_target,
        )
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


def _wait_for_capture_presentation(
    app,
    QtCore,
    win,
    *,
    phase: str,
    expected_target: PresentationTargetToken | None = None,
    previous_target: PresentationTargetToken | None = None,
) -> PresentationTargetToken:
    observed_target: list[PresentationTargetToken | None] = [None]

    def current_target_is_settled() -> bool:
        target = presentation_target_token(win)
        if target is None:
            return False
        if previous_target is not None and target == previous_target:
            return False
        if expected_target is not None and target != expected_target:
            return False
        observed_target[0] = target
        return presentation_is_settled(win, expected_target=target)

    try:
        _wait_until(
            app,
            QtCore,
            current_target_is_settled,
            timeout_s=bounded_interaction_settle_timeout_s(),
            description=f"{phase} physical presentation",
        )
    except TimeoutError as exc:
        raise TimeoutError(
            f"{exc}; "
            f"previous_target={previous_target!r}; "
            f"{presentation_settlement_diagnostic(win, expected_target=expected_target)}"
        ) from exc
    target = observed_target[0]
    if target is None:
        raise RuntimeError(f"{phase} settled without a current presentation target")
    return target


def _wait_until(app, QtCore, predicate, *, timeout_s: float, description: str = "condition") -> None:
    timeout_s = bounded_interaction_settle_timeout_s(timeout_s)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _process_events(app, QtCore, count=2)
        if predicate():
            return
        time.sleep(0.01)
    _process_events(app, QtCore, count=2)
    if predicate():
        return
    raise TimeoutError(f"{description} did not settle within {timeout_s:.3f} s")


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
