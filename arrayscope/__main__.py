#!/usr/bin/env python3
"""
Command-line interface for arrayscope.
"""

import argparse
import atexit
import sys
from pathlib import Path

from arrayscope.core.trace import close_trace, configure_trace


def _run_cli_event_loop():
    import pyqtgraph as pg

    return pg.mkQApp().exec()


def _open_file_async(
    filepath: Path, *, mmap: bool = False, consume: bool = False, title: str | None = None
):
    """Open one path (data file, DICOM dir, or dataset container) without
    blocking: a loading window appears immediately, the viewer as soon as
    data starts being available."""
    from arrayscope.app.open_flow import open_any_path

    return open_any_path(filepath, title=title, mmap=mmap, consume=consume)


def _show_launcher():
    from arrayscope.app.open_flow import show_launcher_window

    return show_launcher_window()


def _run_desktop_integration(args) -> int:
    from arrayscope.desktop import (
        install_desktop_integration,
        uninstall_desktop_integration,
    )

    try:
        if args.install_desktop:
            report = install_desktop_integration()
        else:
            report = uninstall_desktop_integration()
    except Exception as exc:
        print(f"Desktop integration failed: {exc}", file=sys.stderr)
        return 1
    for line in report.lines:
        print(line)
    return 0 if report.ok else 1


def main():
    parser = argparse.ArgumentParser(
        prog="arrayscope",
        description="Interactive N-dimensional array viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  arrayscope                               # Launcher window (open files via dialog or drag-and-drop)
  arrayscope data.npy                      # View single file
  arrayscope data.h5 data2.npy data3.npz   # View multiple files
  arrayscope scan.REC                      # View Philips REC/XML pair
  arrayscope ref.cfl                       # View BART CFL/HDR pair
  arrayscope dicomdir/                     # Convert DICOM directory via dcm2niix, then view
  arrayscope scan.dcm                      # View DICOM file
  arrayscope scan.nii                      # View NIfTI file
  arrayscope data.txt                      # View text file with numeric data
  arrayscope --mmap --consume handoff.npy  # Language-wrapper handoff (Julia/MATLAB)
  arrayscope --install-desktop             # Register with the desktop shell (menu entry, icons, file types)

Files open asynchronously: a loading window shows progress, and for formats
that support it the viewer opens while the file is still being read.
For files with multiple datasets (HDF5, NPZ, MAT), a GUI selector will automatically appear.
        """,
    )
    parser.add_argument(
        "files",
        type=str,
        nargs="*",
        help="Path(s) to data files or DICOM directories (omit to open the launcher window)",
    )
    parser.add_argument(
        "--title", type=str, default=None, help="Window title override for single-dataset files"
    )
    parser.add_argument(
        "--mmap",
        action="store_true",
        help="Memory-map .npy files (copy-on-write) instead of an eager read",
    )
    parser.add_argument(
        "--consume",
        action="store_true",
        help="Delete the input file once loaded (for temporary handoff files "
        "written by the Julia/MATLAB wrappers; best effort)",
    )
    parser.add_argument(
        "--trace", default=None, help="Write structured render/kernel/presentation events to JSONL"
    )
    parser.add_argument(
        "--install-desktop",
        action="store_true",
        help="Integrate ArrayScope with the desktop shell (application menu "
        "entry, icons, and file-type associations), then exit",
    )
    parser.add_argument(
        "--uninstall-desktop",
        action="store_true",
        help="Remove the desktop shell integration, then exit",
    )

    args = parser.parse_args()

    # Desktop (un)registration is a plain filesystem operation: run it before
    # the supervisors below so no GUI child process is spawned.
    if args.install_desktop or args.uninstall_desktop:
        raise SystemExit(_run_desktop_integration(args))

    # Free-threading policy (free-threaded builds only), after argparse for
    # the same reason as below and OUTERMOST so the display-server retry
    # inside the child resolves wayland crashes before an early exit can be
    # blamed on free threading: by default this relaunches the CLI as a
    # supervised child with PYTHON_GIL=0 and, if it dies abnormally within
    # the grace period, persists auto_disabled and retries once with the
    # GIL enabled; when (force/auto) disabled it re-execs with PYTHON_GIL=1.
    from arrayscope.app.free_threading import supervise_free_threading_if_needed

    free_threading_rc = supervise_free_threading_if_needed()
    if free_threading_rc is not None:
        raise SystemExit(free_threading_rc)

    # Display-server policy (Linux/Wayland only), after argparse so --help
    # and usage errors exit here without spawning anything: in "auto" this
    # relaunches the CLI as a supervised child on wayland and retries once
    # on xcb if it dies abnormally within the grace period; in forced modes
    # it just exports QT_QPA_PLATFORM before any QApplication exists.  The
    # supervised child re-enters main() with identical argv and owns trace
    # configuration and all real work itself.
    from arrayscope.app.qt_platform import supervise_cli_if_needed

    supervised_rc = supervise_cli_if_needed()
    if supervised_rc is not None:
        raise SystemExit(supervised_rc)

    if args.trace:
        configure_trace(args.trace)
        atexit.register(close_trace)

    needs_event_loop = False

    if not args.files:
        _show_launcher()
        needs_event_loop = True

    for file_arg in args.files:
        filepath = Path(file_arg)

        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            continue

        try:
            needs_event_loop = (
                bool(
                    _open_file_async(
                        filepath,
                        mmap=args.mmap,
                        consume=args.consume,
                        title=args.title,
                    )
                )
                or needs_event_loop
            )
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if needs_event_loop:
        _run_cli_event_loop()


if __name__ == "__main__":
    main()
