"""Asynchronous file-open flow: never block the GUI on file I/O.

The contract, in event order:

1. ``open_path_async`` shows a :class:`LoadingWindow` immediately — before
   any file I/O — and starts a reader thread.
2. Streaming-capable formats (.npy, .cfl, .rec) report a
   :class:`StreamingProbe` as soon as the destination array is allocated;
   the viewer window opens right then, showing the array as it fills, with
   a status-bar readout of how much of the file is available.
3. Progress observations update whichever window is current; the viewer is
   periodically refreshed (throttled) so already-loaded regions appear.
4. On completion the viewer gets a final cache-invalidating refresh; on
   error a message is shown; cancelling keeps whatever loaded so far.

Everything GUI-side runs on the Qt main thread; the reader thread only
talks through queued signal emissions.
"""

from __future__ import annotations

import contextlib
import threading
import time
import traceback
from pathlib import Path

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.io.file_interpreters import consume_handoff_file, data_file_suffix, load_path

# Formats load_path handles directly; multi-dataset formats go through
# dataset selectors (arrayscope.io.selectors).
SINGLE_DATASET_SUFFIXES = (".npy", ".rec", ".cfl", ".dcm", ".nii", ".nii.gz", ".txt")
SELECTOR_SUFFIXES = (".h5", ".hdf5", ".npz", ".mat")
SUPPORTED_SUFFIXES = SINGLE_DATASET_SUFFIXES + SELECTOR_SUFFIXES

# Throttle for mid-stream viewer refreshes: each refresh invalidates the
# evaluation caches and re-renders, so keep them coarse.
_STREAM_REFRESH_MIN_INTERVAL_S = 1.0
_STREAM_REFRESH_MIN_FRACTION_STEP = 0.15

_ACTIVE_SESSIONS = []


def file_dialog_name_filter():
    globs = " ".join(f"*{suffix}" for suffix in SUPPORTED_SUFFIXES)
    return f"Array files ({globs});;All files (*)"


def is_supported_path(path):
    path = Path(path)
    return path.is_dir() or data_file_suffix(path) in SUPPORTED_SUFFIXES


class _LoaderBridge(QtCore.QObject):
    """Thread → GUI marshalling for one load."""

    probed = QtCore.Signal(object)  # StreamingProbe
    progressed = QtCore.Signal(object)  # LoadProgress
    finished = QtCore.Signal(object)  # LoadedPath
    cancelled = QtCore.Signal()
    failed = QtCore.Signal(str, str)  # summary, traceback text


class FileOpenSession(QtCore.QObject):
    """One file being opened asynchronously.

    Emits ``done()`` once terminal (viewer open and load finished, load
    failed, or cancelled with no viewer).
    """

    done = QtCore.Signal()

    def __init__(
        self,
        filepath,
        *,
        title=None,
        mmap=False,
        consume=False,
        show_loading_window=True,
        parent=None,
    ):
        super().__init__(parent)
        self.filepath = Path(filepath)
        self.window = None
        self.loading_window = None
        self._title_override = title
        self._mmap = mmap
        self._consume = consume
        self._show_loading_window = show_loading_window
        self._cancel = threading.Event()
        self._probe = None
        self._status_widget = None
        self._last_refresh_time = 0.0
        self._last_refresh_fraction = 0.0
        self._latest_fraction = 0.0
        self._terminal = False

        self._bridge = _LoaderBridge(self)
        self._bridge.probed.connect(self._on_probed)
        self._bridge.progressed.connect(self._on_progress)
        self._bridge.finished.connect(self._on_finished)
        self._bridge.cancelled.connect(self._on_cancelled)
        self._bridge.failed.connect(self._on_failed)

        self._thread = threading.Thread(
            target=self._worker,
            name=f"arrayscope-load:{self.filepath.name}",
            daemon=True,
        )

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        if self._show_loading_window:
            from arrayscope.ui.loading_window import LoadingWindow

            self.loading_window = LoadingWindow(self.filepath)
            self.loading_window.cancel_requested.connect(self.cancel)
            self.loading_window.show()
        _ACTIVE_SESSIONS.append(self)
        self._thread.start()
        return self

    def cancel(self):
        self._cancel.set()

    def wait(self, timeout=None):
        """Join the reader thread (tests only; the GUI never blocks on this)."""
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _finish(self):
        if self._terminal:
            return
        self._terminal = True
        with contextlib.suppress(ValueError):
            _ACTIVE_SESSIONS.remove(self)
        self.done.emit()

    # -- reader thread -----------------------------------------------------

    def _worker(self):
        from arrayscope.io.progressive import LoadCancelled

        try:
            loaded = load_path(
                self.filepath,
                mmap=self._mmap,
                progress=self._bridge.progressed.emit,
                cancel=self._cancel,
                on_streaming_probe=self._bridge.probed.emit,
            )
        except LoadCancelled:
            self._bridge.cancelled.emit()
            return
        except BaseException as exc:  # report any load failure to the GUI
            self._bridge.failed.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())
            return
        if self._consume:
            consume_handoff_file(self.filepath)
        self._bridge.finished.emit(loaded)

    # -- GUI-thread handlers ----------------------------------------------

    def _display_title(self, metadata, *, loading):
        if self._title_override is not None:
            return self._title_override
        title = self.filepath.name or str(self.filepath)
        detected_format = (metadata or {}).get("detected_format")
        if detected_format:
            qualifiers = [detected_format]
            if (metadata or {}).get("lazy"):
                qualifiers.append("lazy")
            if loading:
                qualifiers.append("loading…")
            title = f"{title} [{', '.join(qualifiers)}]"
        return title

    def _open_viewer(self, data, *, axes, metadata, loading):
        from arrayscope.app.launch import _create_window, _retain_window_reference

        app, win = _create_window(
            data,
            title=self._display_title(metadata, loading=loading),
            filepath=self.filepath,
            axes=axes,
        )
        _retain_window_reference(app, win)
        self.window = win
        if self.loading_window is not None:
            self.loading_window.close_quietly()
            self.loading_window = None
        return win

    def _on_probed(self, probe):
        if self._cancel.is_set() or self._terminal or self.window is not None:
            return
        self._probe = probe
        win = self._open_viewer(
            probe.data,
            axes=probe.axes,
            metadata=probe.metadata,
            loading=True,
        )
        from arrayscope.ui.load_status import LoadStatusWidget

        self._status_widget = LoadStatusWidget()
        self._status_widget.cancel_requested.connect(self.cancel)
        win.statusBar().addPermanentWidget(self._status_widget)
        self._last_refresh_time = time.monotonic()

    def _on_progress(self, event):
        if self._terminal:
            return
        if event.fraction is not None:
            self._latest_fraction = event.fraction
        if self.loading_window is not None:
            self.loading_window.apply_progress(event)
        if self._status_widget is not None:
            self._status_widget.apply_progress(event)
        self._maybe_refresh_streaming_viewer(event)

    def _maybe_refresh_streaming_viewer(self, event):
        if self.window is None or event.stage != "reading" or event.fraction is None:
            return
        now = time.monotonic()
        if (
            now - self._last_refresh_time < _STREAM_REFRESH_MIN_INTERVAL_S
            or event.fraction - self._last_refresh_fraction < _STREAM_REFRESH_MIN_FRACTION_STEP
        ):
            return
        self._last_refresh_time = now
        self._last_refresh_fraction = event.fraction
        self._refresh_viewer_data()

    def _refresh_viewer_data(self):
        win = self.window
        if win is None:
            return
        try:
            win.notify_data_changed(force_autolevel=True)
        except Exception:
            traceback.print_exc()

    def _remove_status_widget(self):
        if self._status_widget is None:
            return
        widget = self._status_widget
        self._status_widget = None
        with contextlib.suppress(Exception):
            self.window.statusBar().removeWidget(widget)
        widget.deleteLater()

    def _on_finished(self, loaded):
        if self._terminal:
            return
        if self.window is None:
            try:
                self._open_viewer(
                    loaded.data,
                    axes=getattr(loaded, "axes", None),
                    metadata=loaded.metadata,
                    loading=False,
                )
            except BaseException as exc:
                self._on_failed(f"{type(exc).__name__}: {exc}", traceback.format_exc())
                return
        else:
            self._remove_status_widget()
            self.window.setWindowTitle(self._display_title(loaded.metadata, loading=False))
            self._refresh_viewer_data()
            from arrayscope.ui.toasts import show_status_message

            show_status_message(self.window, "File fully loaded")
        self._finish()

    def _on_cancelled(self):
        if self._terminal:
            return
        if self.window is not None:
            self._remove_status_widget()
            self._refresh_viewer_data()
            from arrayscope.ui.toasts import show_status_message

            show_status_message(
                self.window,
                f"Loading stopped — showing the {self._latest_fraction:.0%} read so far",
                timeout=8000,
            )
        elif self.loading_window is not None:
            self.loading_window.close_quietly()
            self.loading_window = None
        self._finish()

    def _on_failed(self, summary, traceback_text):
        if self._terminal:
            return
        import sys

        print(f"Error loading {self.filepath}: {summary}", file=sys.stderr)
        print(traceback_text, file=sys.stderr)
        if self.loading_window is not None:
            self.loading_window.show_error(summary)
        elif self.window is not None:
            self._remove_status_widget()
            QtWidgets.QMessageBox.warning(
                self.window,
                "Load Error",
                f"Failed to load {self.filepath.name}:\n{summary}",
            )
        self._finish()


def open_path_async(filepath, *, title=None, mmap=False, consume=False, show_loading_window=True):
    """Open one single-dataset file (or DICOM directory) asynchronously."""
    ensure_open_app()
    session = FileOpenSession(
        filepath,
        title=title,
        mmap=mmap,
        consume=consume,
        show_loading_window=show_loading_window,
    )
    return session.start()


def open_any_path(filepath, *, title=None, mmap=False, consume=False):
    """Open any supported path: async load, or a dataset selector for
    multi-dataset containers. Returns truthy when something opened."""
    filepath = Path(filepath)
    if not filepath.exists():
        import sys

        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return False
    suffix = data_file_suffix(filepath)
    if filepath.is_dir() or suffix in SINGLE_DATASET_SUFFIXES:
        return open_path_async(filepath, title=title, mmap=mmap, consume=consume)
    if suffix in SELECTOR_SUFFIXES:
        from arrayscope.io.selectors import (
            H5DatasetSelector,
            MatDatasetSelector,
            NpzDatasetSelector,
        )

        selector_types = {
            ".h5": H5DatasetSelector,
            ".hdf5": H5DatasetSelector,
            ".npz": NpzDatasetSelector,
            ".mat": MatDatasetSelector,
        }
        selector = selector_types[suffix](filepath)
        if not selector.requires_gui():
            result = selector.get_single_data()
            selector.close()
            if not result:
                import sys

                print(f"No compatible datasets found in {filepath}", file=sys.stderr)
                return False
            name, data = result
            from arrayscope.app.launch import _create_window, _retain_window_reference

            app, win = _create_window(
                data,
                title=title or f"{filepath.name} - {name}",
                filepath=filepath,
                dataset_path=name,
                selector_class_name=selector.__class__.__name__,
            )
            _retain_window_reference(app, win)
            return win
        if not selector.view(block=False):
            import sys

            print(f"No compatible datasets found in {filepath}", file=sys.stderr)
            return False
        return True
    import sys

    print(
        f"Unsupported file type: {suffix}. Supported types: directories with DICOM "
        f".dcm files, {', '.join(SUPPORTED_SUFFIXES)}",
        file=sys.stderr,
    )
    return False


class _FileOpenEventFilter(QtCore.QObject):
    """Handle QFileOpenEvent (macOS Finder / Dock document opens)."""

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.FileOpen:
            path = event.file()
            if path:
                open_any_path(Path(path))
                return True
        return super().eventFilter(obj, event)


def ensure_open_app():
    """Create/fetch the QApplication configured for desktop file opens."""
    from arrayscope.app.launch import _prepare_qt_environment

    _prepare_qt_environment()
    app = pg.mkQApp()
    app.setOrganizationName("ArrayScope")
    app.setApplicationName("ArrayScope")
    app.setStyle("Fusion")
    if app.property("_arrayscope_file_open_filter") is None:
        open_filter = _FileOpenEventFilter(app)
        app.installEventFilter(open_filter)
        app.setProperty("_arrayscope_file_open_filter", open_filter)
    _apply_app_icon(app)
    return app


def _apply_app_icon(app):
    if app.property("_arrayscope_icon_applied"):
        return
    from arrayscope.desktop.assets import application_icon_path

    icon_path = application_icon_path()
    if icon_path is not None:
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))
    app.setProperty("_arrayscope_icon_applied", True)


def show_launcher_window():
    """Open the empty-state launcher (no file arguments)."""
    from arrayscope.ui.launcher_window import LauncherWindow

    app = ensure_open_app()
    win = LauncherWindow(
        open_any_path,
        supported_suffixes=SUPPORTED_SUFFIXES,
        name_filter=file_dialog_name_filter(),
    )
    win.show()
    from arrayscope.app.launch import _retain_window_reference

    _retain_window_reference(app, win)
    return win
