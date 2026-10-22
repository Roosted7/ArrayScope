"""Argument and terminal adapters for synchronous, Qt-free batch export."""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import threading
from contextlib import contextmanager
from pathlib import Path


def _parse_range(value):
    match = re.fullmatch(r"([0-9]+)=([0-9]+):([0-9]+)", value.strip())
    if match is None:
        raise argparse.ArgumentTypeError(
            "range must be AXIS=START:STOP with nonnegative integers (for example 0=10:20)"
        )
    axis, start, stop = (int(part) for part in match.groups())
    if stop < start:
        raise argparse.ArgumentTypeError("range STOP must be at least START")
    return axis, start, stop


def add_batch_arguments(parser):
    group = parser.add_argument_group(
        "Headless NumPy export",
        "One .npy input or an explicitly selected HDF5 dataset; no display required.",
    )
    group.add_argument(
        "--export", metavar="OUTPUT", help="Export exact scientific values to a .npy or .npz file"
    )
    group.add_argument(
        "--recipe", metavar="RECIPE.json", help="Apply a built-in operation recipe before export"
    )
    group.add_argument(
        "--range",
        dest="export_ranges",
        metavar="AXIS=START:STOP",
        type=_parse_range,
        action="append",
        default=[],
        help="Select a half-open range in a derived output axis (repeat for other axes)",
    )
    group.add_argument(
        "--no-squeeze",
        action="store_true",
        help="Preserve singleton axes (by default they are removed)",
    )
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a JSON plan with estimated reads and costs; read no source values and write no output",
    )
    group.add_argument(
        "--allow-expensive",
        action="store_true",
        help="Accept supported operation cost warnings; source read budgets still apply",
    )


class _CancellationToken:
    def __init__(self):
        self._event = threading.Event()

    @property
    def cancelled(self):
        return self._event.is_set()

    def cancel(self, _signal=None, _frame=None):
        self._event.set()


@contextmanager
def _cancel_on_interrupt(token):
    previous = signal.signal(signal.SIGINT, token.cancel)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


class _ProgressReporter:
    """A bounded terminal report: one phase plus at most ten write updates."""

    def __init__(self):
        self._phase = None
        self._percent = -1

    def __call__(self, progress):
        phase = progress.phase
        percent = None
        if phase == "writing" and progress.total_bytes:
            percent = min(100, max(0, int(progress.completed_bytes * 100 // progress.total_bytes)))
            percent = percent // 10 * 10
        if phase == self._phase and (percent is None or percent <= self._percent):
            return
        self._phase = phase
        self._percent = -1 if percent is None else percent
        suffix = "" if percent is None else f" {percent}%"
        print(f"{progress.message}{suffix}", file=sys.stderr)


def run_batch_cli(args, parser):
    """Return an exit status for --export, or None for the ordinary viewer."""

    batch_options = (
        args.recipe is not None
        or args.export_ranges
        or args.no_squeeze
        or args.dry_run
        or args.allow_expensive
    )
    if args.export is None:
        if batch_options:
            parser.error(
                "--recipe, --range, --no-squeeze, --dry-run and --allow-expensive require --export"
            )
        return None
    conflicts = [
        name
        for name, value in (
            ("--title", args.title),
            ("--mmap", args.mmap),
            ("--consume", args.consume),
            ("--trace", args.trace),
            ("--install-desktop", args.install_desktop),
            ("--uninstall-desktop", args.uninstall_desktop),
        )
        if value is not None and value is not False
    ]
    if conflicts:
        parser.error(f"--export cannot be combined with {', '.join(conflicts)}")
    if len(args.files) != 1:
        parser.error("--export requires exactly one .npy or HDF5 input file")
    suffix = Path(args.files[0]).suffix.lower()
    dataset_path = getattr(args, "dataset", None)
    if suffix == ".npy":
        if dataset_path is not None:
            parser.error("--dataset is only supported for .h5 or .hdf5 input files")
    elif suffix in {".h5", ".hdf5"}:
        if not dataset_path:
            parser.error("headless HDF5 export requires an explicit --dataset path")
    else:
        parser.error("headless export requires a .npy input or an HDF5 input with --dataset")
    if Path(args.export).suffix.lower() not in {".npy", ".npz"}:
        parser.error("--export output must end in .npy or .npz")
    axes = tuple(axis for axis, _start, _stop in args.export_ranges)
    if len(set(axes)) != len(axes):
        parser.error("each --range axis may be specified only once")

    from arrayscope.core.array_source import SourceReadRefused
    from arrayscope.io.recipe_replay import (
        BatchExportNeedsConfirmation,
        BatchExportRequest,
        run_batch_export,
    )
    from arrayscope.operations.cancellation import EvaluationCancelled

    token = _CancellationToken()
    result = None
    try:
        request = BatchExportRequest(
            input_path=Path(args.files[0]),
            output_path=Path(args.export),
            recipe_path=None if args.recipe is None else Path(args.recipe),
            ranges=tuple(args.export_ranges),
            squeeze=not args.no_squeeze,
            allow_expensive=args.allow_expensive,
            dataset_path=dataset_path,
        )
        with _cancel_on_interrupt(token):
            result = run_batch_export(
                request,
                dry_run=args.dry_run,
                cancellation_token=token,
                progress_callback=None if args.dry_run else _ProgressReporter(),
            )
            # Successful publication wins a cancellation arriving after commit.
            # Keep the cooperative handler until its success report is emitted.
            print(json.dumps(result.summary, sort_keys=True, allow_nan=False))
    except (EvaluationCancelled, KeyboardInterrupt):
        # Restoring the caller's SIGINT handler can deliver a pending interrupt
        # after the completed result and JSON report. Publication stays final.
        if result is not None:
            return 0
        print("Export cancelled.", file=sys.stderr)
        return 130
    except BatchExportNeedsConfirmation as exc:
        print(f"Export refused: {exc}", file=sys.stderr)
        return 3
    except SourceReadRefused as exc:
        print(
            f"Export refused: {exc}\nChoose a smaller selection or reduce the required recipe stage; "
            "--allow-expensive cannot override this budget.",
            file=sys.stderr,
        )
        return 3
    except ValueError as exc:
        print(f"Invalid export: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    return 0
