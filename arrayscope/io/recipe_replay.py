"""Headless recipe replay over an exclusively owned, read-only array source.

The first replay contract supports ArrayScope's built-in operations. Plugin
lookup can load code, consult Qt settings, and execute characterization probes;
rejecting those IDs before lookup keeps dry-run and import policy explicit.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, replace
from math import prod
from operator import index
from pathlib import Path
from stat import S_ISREG

import numpy as np

from arrayscope.core.array_source import LazySourceArray, NdArraySource
from arrayscope.io.region_export import RegionExportRequest, export_region, plan_region_export
from arrayscope.io.streaming_export import export_streaming_region, plan_streaming_export
from arrayscope.operations import recipes
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.pipeline import ArrayDocument
from arrayscope.operations.registry import operation_entries


@dataclass(frozen=True)
class BatchExportRequest:
    input_path: Path
    output_path: Path
    recipe_path: Path | None = None
    ranges: tuple[tuple[int, int, int], ...] = ()
    squeeze: bool = True
    allow_expensive: bool = False
    dataset_path: str | None = None
    streaming: bool = False

    def __post_init__(self):
        object.__setattr__(self, "input_path", Path(self.input_path).expanduser().absolute())
        object.__setattr__(self, "output_path", Path(self.output_path).expanduser().absolute())
        if self.recipe_path is not None:
            object.__setattr__(self, "recipe_path", Path(self.recipe_path).expanduser().absolute())
        object.__setattr__(self, "ranges", tuple(tuple(bounds) for bounds in self.ranges))
        if self.dataset_path is not None:
            object.__setattr__(self, "dataset_path", str(self.dataset_path))


@dataclass(frozen=True)
class BatchExportResult:
    input_path: Path
    output_path: Path
    input_shape: tuple[int, ...]
    derived_shape: tuple[int, ...]
    ranges: tuple[tuple[int, int], ...]
    squeeze: bool
    output_shape: tuple[int, ...]
    dtype: str | None
    output_bytes: int | None
    source_read_bytes: int | None
    estimated_peak_bytes: int | None
    warnings: tuple[str, ...]
    needs_confirmation: bool
    paths: tuple[Path, ...]
    dry_run: bool
    dataset_path: str | None = None
    streaming: bool = False
    chunk_shape: tuple[int, ...] | None = None
    chunk_count: int | None = None
    estimated_chunk_peak_bytes: int | None = None

    @property
    def summary(self):
        """Return detached JSON values, never source arrays, documents, or plans."""
        summary = {
            "schema_version": 1,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "input_shape": list(self.input_shape),
            "derived_shape": list(self.derived_shape),
            "ranges": [list(bounds) for bounds in self.ranges],
            "squeeze": self.squeeze,
            "output_shape": list(self.output_shape),
            "dtype": self.dtype,
            "output_bytes": self.output_bytes,
            "source_read_bytes": self.source_read_bytes,
            "estimated_peak_bytes": self.estimated_peak_bytes,
            "warnings": list(self.warnings),
            "needs_confirmation": self.needs_confirmation,
            "paths": [str(path) for path in self.paths],
            "dry_run": self.dry_run,
        }
        if self.dataset_path is not None:
            summary["dataset_path"] = self.dataset_path
        if self.streaming:
            summary.update(
                execution_mode="streaming",
                chunk_shape=list(self.chunk_shape),
                chunk_count=self.chunk_count,
                estimated_chunk_peak_bytes=self.estimated_chunk_peak_bytes,
            )
        return summary


class BatchExportNeedsConfirmation(RuntimeError):
    """Execution was refused until the caller accepts the prepared cost."""

    def __init__(self, summary):
        self.summary = summary
        super().__init__(
            "Export requires explicit consent for its processing cost. "
            "Inspect --dry-run, then use --allow-expensive to proceed."
        )


def _check_cancelled(token):
    if token is not None and getattr(token, "cancelled", False):
        raise EvaluationCancelled()


@contextmanager
def _owned_npy_source(path):
    # This mapping is opened exclusively for this call. Generic viewer sources
    # can be shared by windows and retain their existing close semantics.
    mapped = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(mapped, np.memmap):
        close = getattr(mapped, "close", None)
        if close is not None:
            close()
        raise ValueError("batch input must be a memory-mappable NumPy array")
    source = LazySourceArray(
        NdArraySource(mapped, label=f"batch-npy:{path.name}", close=mapped._mmap.close)
    )
    try:
        if source.dtype.kind not in "biufc":
            raise ValueError(f"unsupported batch input dtype: {source.dtype}")
        yield source
    finally:
        source.close()


@contextmanager
def _owned_replay_source(request):
    if request.input_path.suffix.lower() == ".npy":
        with _owned_npy_source(request.input_path) as source:
            yield source
        return
    from arrayscope.io.hdf5_source import open_hdf5_source

    source = open_hdf5_source(request.input_path, request.dataset_path)
    try:
        yield source
    finally:
        source.close()


def _builtin_policy(operation_id):
    if operation_id not in {entry.id for entry in operation_entries()}:
        raise ValueError(
            f"headless replay supports built-in operations only; {operation_id!r} "
            "requires a plugin, user definition, or command runtime and will not be loaded"
        )


def _selected_ranges(shape, requested):
    ranges = [(0, size) for size in shape]
    seen = set()
    for bounds in requested:
        if len(bounds) != 3:
            raise ValueError("each range must contain an axis, start, and stop")
        try:
            axis, start, stop = (index(value) for value in bounds)
        except TypeError as exc:
            raise ValueError("range axes and bounds must be integers") from exc
        if not 0 <= axis < len(shape):
            raise ValueError(f"range axis {axis} is outside the derived array rank {len(shape)}")
        if axis in seen:
            raise ValueError(f"range axis {axis} was specified more than once")
        seen.add(axis)
        if not 0 <= start <= stop <= shape[axis]:
            raise ValueError(
                f"range for axis {axis} must satisfy 0 <= start <= stop <= {shape[axis]}"
            )
        ranges[axis] = (start, stop)
    return tuple(ranges)


def run_batch_export(
    request: BatchExportRequest,
    *,
    dry_run=False,
    cancellation_token=None,
    progress_callback=None,
):
    """Replay supported recipe steps and export exact selected scientific data.

    Dry-run validates and plans using metadata only, returning before any
    source read or output write. Execution uses the same region plan and atomic
    publication owner as the GUI. Every return/exception closes this call's
    unique mapping; a successful result wins cancellation after publication.
    """
    _check_cancelled(cancellation_token)
    suffix = request.input_path.suffix.lower()
    if suffix == ".npy":
        if request.dataset_path is not None:
            raise ValueError("dataset_path is only supported for HDF5 input")
    elif suffix in {".h5", ".hdf5"}:
        if not request.dataset_path:
            raise ValueError("headless HDF5 replay requires an explicit dataset_path")
    else:
        raise ValueError("headless replay input must be .npy or selected .h5/.hdf5 data")
    if request.output_path.suffix.lower() not in {".npy", ".npz"}:
        raise ValueError("headless replay output must end with .npy or .npz")
    if request.streaming and request.output_path.suffix.lower() != ".npy":
        raise ValueError("streaming export requires .npy output; omit streaming for .npz")
    if not S_ISREG(request.input_path.stat().st_mode):
        raise ValueError("headless replay input must be a regular array file")
    if request.recipe_path is not None and not S_ISREG(request.recipe_path.stat().st_mode):
        raise ValueError("the recipe must be a regular JSON file")
    for input_path in (request.input_path, request.recipe_path):
        if input_path is not None and (
            input_path.resolve() == request.output_path.resolve()
            or (request.output_path.exists() and input_path.samefile(request.output_path))
        ):
            raise ValueError("input and output refer to the same file; choose a distinct output")

    with _owned_replay_source(request) as source:
        _check_cancelled(cancellation_token)
        dataset_path = None
        if suffix in {".h5", ".hdf5"}:
            dataset_path = source.source.descriptor.key
        steps = ()
        if request.recipe_path is not None:
            steps = recipes.loads_recipe_steps(
                request.recipe_path.read_text(encoding="utf-8"),
                source.shape,
                imported=False,
                operation_policy=_builtin_policy,
                require_available=True,
            )
        document = ArrayDocument(source, steps=steps)
        ranges = _selected_ranges(document.current_shape, request.ranges)
        region_request = RegionExportRequest(
            document,
            request.output_path,
            ranges,
            squeeze=request.squeeze,
            recipe_json=recipes.dumps_recipe(steps),
            sidecar=False,
            source_metadata_json=(
                json.dumps({"source_path": str(request.input_path), "dataset_path": dataset_path})
                if dataset_path is not None
                else None
            ),
        )
        planner = plan_streaming_export if request.streaming else plan_region_export
        exporter = export_streaming_region if request.streaming else export_region
        plan = planner(region_request, cancellation_token=cancellation_token)
        result = BatchExportResult(
            input_path=request.input_path,
            output_path=request.output_path,
            input_shape=source.shape,
            derived_shape=document.current_shape,
            ranges=ranges,
            squeeze=bool(request.squeeze),
            output_shape=plan.output_shape,
            dtype=None if plan.output_dtype is None else str(plan.output_dtype),
            output_bytes=plan.output_bytes,
            source_read_bytes=(
                plan.total_source_read_bytes if request.streaming else plan.source_read_bytes
            ),
            estimated_peak_bytes=plan.estimated_peak_bytes,
            warnings=plan.warnings,
            needs_confirmation=plan.needs_confirmation,
            paths=(),
            dry_run=bool(dry_run),
            dataset_path=dataset_path,
            streaming=bool(request.streaming),
            chunk_shape=plan.chunk_shape if request.streaming else None,
            chunk_count=plan.chunk_count if request.streaming else None,
            estimated_chunk_peak_bytes=plan.estimated_peak_bytes if request.streaming else None,
        )
        _check_cancelled(cancellation_token)
        if dry_run:
            return result
        if plan.needs_confirmation and not request.allow_expensive:
            raise BatchExportNeedsConfirmation(result.summary)
        exported = exporter(
            region_request,
            prepared=plan,
            cancellation_token=cancellation_token,
            progress_callback=progress_callback,
        )
        # Do not retain the evaluated array/plan, or check cancellation after
        # publication: the writer's committed result is the final outcome.
        return replace(
            result,
            output_shape=exported.shape,
            dtype=str(exported.dtype),
            output_bytes=prod(exported.shape) * exported.dtype.itemsize,
            paths=exported.paths,
        )
