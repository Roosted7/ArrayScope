"""Exact, selection-first scientific NumPy export, independent of Qt.

Requests freeze document/selection semantics and retain the source reference;
they do not deep-copy source pixels. Callers must cancel on notified in-place
source changes and refuse still-streaming sources. Unnotified external writes
cannot provide snapshot isolation without a source-owned version or copy.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from math import prod
from operator import index
from pathlib import Path

import numpy as np

from arrayscope.core.array_source import ArraySource, CompositeArraySource, LazySourceArray
from arrayscope.core.compute_policy import ComputeLane, EvaluationContext
from arrayscope.core.memory_budget import DEFAULT_VISIBLE_RENDER_BUDGET_BYTES, format_bytes
from arrayscope.core.roi import RoiGeometry
from arrayscope.io.numpy_save import save_derived_array
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.input_slots import ResolvedSlot
from arrayscope.operations.pipeline import ArrayDocument
from arrayscope.operations.planner import RegionPlan, plan_output_region
from arrayscope.operations.plugin_conformance import _freeze, _source_identity
from arrayscope.operations.plugins import PluginOperation, load_plugin_spec
from arrayscope.operations.regions import region_from_index_spec, region_nbytes
from arrayscope.operations.slabs import evaluate_region_plan

_LOGGER = logging.getLogger(__name__)


class RegionExportChanged(RuntimeError):
    """A captured operation definition or file-backed input changed."""


@dataclass(frozen=True)
class _DefinitionSnapshot:
    operation_id: str
    spec: object = field(compare=False, repr=False)
    identity: object


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    identity: tuple


def _file_identity(path):
    stat = path.stat()
    return (str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino, stat.st_dev)


def _slot_is_snapshot(source):
    if not isinstance(source, ResolvedSlot) or not source.available:
        return False
    value = source.source
    if source.loader in {"saved-array", "file"}:
        return isinstance(value, (str, Path))
    if source.loader == "document":
        return isinstance(value, ArrayDocument)
    if source.loader in {"roi-mask", "roi-coordinates"}:
        return isinstance(value, RoiGeometry)
    if source.loader in {"array", "dimension-set"}:
        if callable(value) or any(
            base.__module__.startswith(("PyQt", "PySide")) for base in type(value).__mro__
        ):
            return False
        return isinstance(value, (np.ndarray, LazySourceArray, ArraySource))
    return False


def capture_export_definitions(document):
    """Capture lightweight definition/file identities without evaluating data.

    Linked code and file-slot identities may stat files. No operation shape
    probes, source reads, or array materialization run here.
    """

    snapshots = []
    visited = set()

    def visit(current):
        if id(current) in visited:
            return
        visited.add(id(current))
        for operation in current.enabled_operations:
            if not isinstance(operation, PluginOperation):
                continue
            try:
                spec = load_plugin_spec(operation.plugin_id)
                identity = _freeze(_source_identity(spec))
            except Exception as exc:
                raise RegionExportChanged(
                    f"Operation {operation.plugin_id!r} is unavailable; refresh its definition."
                ) from exc
            snapshots.append(_DefinitionSnapshot(operation.plugin_id, spec, identity))
            slots = dict(operation.resolved_slots)
            for slot in spec.input_slots:
                if not _slot_is_snapshot(slots.get(slot.name)):
                    raise RegionExportChanged(
                        f"Input {slot.label!r} must be a resolved array, document, file, "
                        "or frozen ROI snapshot before export."
                    )
            for _name, source in operation.resolved_slots:
                if source.loader in {"saved-array", "file"}:
                    path = Path(source.binding.path).expanduser().absolute()
                    try:
                        actual = _file_identity(path)
                    except OSError as exc:
                        raise RegionExportChanged(
                            f"Export input {path} is unavailable; refresh its binding."
                        ) from exc
                    expected = source.source_identity
                    if (
                        isinstance(expected, tuple)
                        and len(expected) == 3
                        and expected != actual[:3]
                    ):
                        raise RegionExportChanged(
                            f"Export input {path} changed; refresh its binding."
                        )
                    snapshots.append(_FileSnapshot(path, actual))
                elif source.loader == "document" and isinstance(source.source, ArrayDocument):
                    visit(source.source)

    visit(document)
    return tuple(snapshots)


def capture_export_sources(document):
    """Retain primary and auxiliary array identities without reading data.

    Source owners can compare notifications against these references with
    ``is``. The references keep captured mappings alive; external alias writes
    still need their source owner's notification or a versioned source API.
    """

    sources = []
    visited = set()

    def add(source):
        if any(source is existing for existing in sources):
            return
        sources.append(source)
        if isinstance(source, CompositeArraySource):
            add(source.a)
            add(source.b)
        elif isinstance(source, LazySourceArray):
            add(source.source)

    def visit(current):
        if id(current) in visited:
            return
        visited.add(id(current))
        add(current.base_data)
        for operation in current.enabled_operations:
            for _name, slot in getattr(operation, "resolved_slots", ()):
                if slot.loader == "document" and isinstance(slot.source, ArrayDocument):
                    visit(slot.source)
                elif slot.loader in {"array", "dimension-set"}:
                    add(slot.source)

    visit(document)
    return tuple(sources)


@dataclass(frozen=True)
class RegionExportRequest:
    document: ArrayDocument = field(compare=False, repr=False)
    path: Path
    ranges: tuple[tuple[int, int], ...]
    squeeze: bool = True
    recipe_json: str | None = None
    view_recipe_json: str | None = None
    sidecar: bool = False
    definitions: tuple | None = field(default=None, repr=False)
    source_metadata_json: str | None = None

    def __post_init__(self):
        shape = tuple(self.document.current_shape)
        ranges = tuple(tuple(index(value) for value in bounds) for bounds in self.ranges)
        if len(ranges) != len(shape):
            raise ValueError("export ranges must match the derived array rank")
        for axis, (size, bounds) in enumerate(zip(shape, ranges, strict=True)):
            if len(bounds) != 2 or not 0 <= bounds[0] <= bounds[1] <= size:
                raise ValueError(
                    f"export range for axis {axis} must satisfy 0 <= start <= stop <= {size}"
                )
        object.__setattr__(self, "path", Path(self.path).expanduser().absolute())
        object.__setattr__(self, "ranges", ranges)
        object.__setattr__(self, "squeeze", bool(self.squeeze))
        if self.source_metadata_json is not None:
            source_metadata = json.loads(self.source_metadata_json)
            if not isinstance(source_metadata, dict):
                raise ValueError("export source metadata must be a JSON object")
            object.__setattr__(self, "source_metadata_json", json.dumps(source_metadata))
        object.__setattr__(
            self,
            "definitions",
            capture_export_definitions(self.document)
            if self.definitions is None
            else tuple(self.definitions),
        )
        _validate_export_sources(self.document)

    @property
    def selected_shape(self):
        return tuple(stop - start for start, stop in self.ranges)

    @property
    def output_shape(self):
        return tuple(size for size in self.selected_shape if not self.squeeze or size != 1)


def _validate_export_sources(document):
    validated = set()
    for source in capture_export_sources(document):
        validator = getattr(source, "validate_snapshot", None)
        if callable(validator):
            key = (
                id(getattr(validator, "__self__", source)),
                id(getattr(validator, "__func__", validator)),
            )
            if key in validated:
                continue
            validated.add(key)
            try:
                validator()
            except Exception as exc:
                raise RegionExportChanged(
                    f"Export source changed or became unavailable; reload it before exporting: {exc}"
                ) from exc


def validate_region_export_snapshot(request: RegionExportRequest) -> None:
    _validate_export_sources(request.document)
    for snapshot in request.definitions:
        if isinstance(snapshot, _DefinitionSnapshot):
            try:
                current = load_plugin_spec(snapshot.operation_id)
                unchanged = (
                    current is snapshot.spec
                    and _freeze(_source_identity(current)) == snapshot.identity
                )
            except Exception:
                unchanged = False
            if not unchanged:
                raise RegionExportChanged(
                    f"Operation {snapshot.operation_id!r} changed during export; start the export again."
                )
        elif isinstance(snapshot, _FileSnapshot):
            try:
                unchanged = _file_identity(snapshot.path) == snapshot.identity
            except OSError:
                unchanged = False
            if not unchanged:
                raise RegionExportChanged(
                    f"Export input {snapshot.path} changed during export; start the export again."
                )
        else:
            raise TypeError("invalid export definition snapshot")


@dataclass(frozen=True)
class RegionExportPlan:
    request: RegionExportRequest = field(compare=False, repr=False)
    region_plan: RegionPlan | None
    output_shape: tuple[int, ...]
    output_dtype: np.dtype | None
    output_bytes: int | None
    source_read_bytes: int | None
    estimated_peak_bytes: int | None
    warnings: tuple[str, ...]
    needs_confirmation: bool


@dataclass(frozen=True)
class RegionExportProgress:
    phase: str
    message: str
    completed_bytes: int = 0
    total_bytes: int | None = None


@dataclass(frozen=True)
class RegionExportResult:
    paths: tuple[Path, ...]
    shape: tuple[int, ...]
    dtype: np.dtype


def _check_cancelled(token):
    if token is not None and getattr(token, "cancelled", False):
        raise EvaluationCancelled()


def _context(evaluation_context, token):
    if evaluation_context is None:
        return EvaluationContext(ComputeLane.STAGE, token, 1, None)
    return replace(evaluation_context, cancellation_token=token)


def plan_region_export(request, *, cancellation_token=None, evaluation_context=None):
    """Price exact work on a worker before the caller grants costly execution."""

    token = (
        cancellation_token
        if cancellation_token is not None
        else getattr(evaluation_context, "cancellation_token", None)
    )
    _check_cancelled(token)
    validate_region_export_snapshot(request)
    if any(
        getattr(operation, "resolved_slots", ())
        for operation in request.document.enabled_operations
    ):
        # Cold plugin characterization reads auxiliary inputs. Defer even that
        # planning until the caller has accepted the explicit unknown cost.
        return RegionExportPlan(
            request=request,
            region_plan=None,
            output_shape=request.output_shape,
            output_dtype=None,
            output_bytes=None,
            source_read_bytes=None,
            estimated_peak_bytes=None,
            warnings=(
                "Primary and auxiliary input costs are unknown. Planning this operation may "
                "materialize auxiliary inputs before evaluating the selected output.",
            ),
            needs_confirmation=True,
        )
    region = region_from_index_spec(
        request.document.current_shape, tuple(slice(start, stop) for start, stop in request.ranges)
    )
    plan = plan_output_region(request.document, region, request_kind="export")
    _check_cancelled(token)
    dtype = plan.stages[-1].output_dtype
    output_bytes = None if dtype is None else prod(request.output_shape) * np.dtype(dtype).itemsize
    source_bytes = region_nbytes(
        request.document.base_data.shape,
        getattr(request.document.base_data, "dtype", None),
        plan.required_input_region,
    )
    warnings = list(plan.warnings)
    peak = plan.estimated_peak_bytes
    policy = getattr(evaluation_context, "memory_policy", None)
    budget = int(
        getattr(policy, "visible_render_budget_bytes", DEFAULT_VISIBLE_RENDER_BUDGET_BYTES)
    )
    warnings = tuple(dict.fromkeys(warnings))
    return RegionExportPlan(
        request=request,
        region_plan=plan,
        output_shape=request.output_shape,
        output_dtype=None if dtype is None else np.dtype(dtype),
        output_bytes=output_bytes,
        source_read_bytes=source_bytes,
        estimated_peak_bytes=peak,
        warnings=warnings,
        needs_confirmation=bool(warnings)
        or peak is None
        or peak > budget
        or (output_bytes or 0) > budget,
    )


def _selection_json(request, data):
    selection = {
        "format": "arrayscope-region-export",
        "version": 1,
        "document_revision": request.document.revision,
        "derived_shape": request.document.current_shape,
        "ranges": request.ranges,
        "squeeze": request.squeeze,
        "selected_shape": request.selected_shape,
        "output_shape": tuple(data.shape),
        "output_axes": [
            axis
            for axis, size in enumerate(request.selected_shape)
            if not request.squeeze or size != 1
        ],
        "dtype": data.dtype.str,
    }
    if request.source_metadata_json is not None:
        selection["source"] = json.loads(request.source_metadata_json)
    return json.dumps(selection, indent=2)


def export_region(
    request,
    *,
    prepared=None,
    cancellation_token=None,
    evaluation_context=None,
    progress_callback=None,
):
    """Evaluate exact selected values and atomically save them, off the GUI thread."""

    token = (
        cancellation_token
        if cancellation_token is not None
        else getattr(evaluation_context, "cancellation_token", None)
    )
    context = _context(evaluation_context, token)

    def report(phase, message, completed=0, total=None):
        if progress_callback is not None:
            progress_callback(RegionExportProgress(phase, message, completed, total))

    _check_cancelled(token)
    validate_region_export_snapshot(request)
    if prepared is None:
        report("planning", "Planning exact selected array export…")
        prepared = plan_region_export(request, cancellation_token=token, evaluation_context=context)
    elif prepared.request is not request:
        raise ValueError("prepared export plan belongs to a different request")
    source_text = (
        "unknown"
        if prepared.source_read_bytes is None
        else format_bytes(prepared.source_read_bytes)
    )
    peak_text = (
        "unknown"
        if prepared.estimated_peak_bytes is None
        else format_bytes(prepared.estimated_peak_bytes)
    )
    report(
        "evaluating", f"Reading {source_text} from the primary source; estimated peak {peak_text}."
    )
    _check_cancelled(token)
    region_plan = prepared.region_plan
    if region_plan is None:
        region_plan = plan_output_region(
            request.document,
            region_from_index_spec(
                request.document.current_shape,
                tuple(slice(start, stop) for start, stop in request.ranges),
            ),
            request_kind="export",
        )
        _check_cancelled(token)
    output_dtype = region_plan.stages[-1].output_dtype
    if 0 in request.selected_shape and output_dtype is not None:
        data = np.empty(request.selected_shape, dtype=output_dtype)
    else:
        data = np.asarray(
            evaluate_region_plan(
                request.document,
                region_plan,
                cancellation_token=token,
                evaluation_context=context,
            )
        )
    _check_cancelled(token)
    if tuple(data.shape) != request.selected_shape:
        raise ValueError(f"export produced shape {data.shape}, expected {request.selected_shape}")
    if output_dtype is not None and data.dtype != output_dtype:
        raise ValueError(f"export produced dtype {data.dtype}, expected {output_dtype}")
    if request.squeeze:
        data = np.squeeze(data)
    report("writing", "Writing selected array…", total=int(data.nbytes))

    def before_publish():
        _check_cancelled(token)
        validate_region_export_snapshot(request)

    paths = save_derived_array(
        request.path,
        data,
        recipe_json=request.recipe_json,
        view_recipe_json=request.view_recipe_json,
        selection_json=_selection_json(request, data),
        sidecar=request.sidecar,
        cancellation_token=token,
        progress_callback=lambda count: report(
            "writing", "Writing selected array…", count, int(data.nbytes)
        ),
        before_publish=before_publish,
    )
    # Publication is a commit point. A late cancel cannot truthfully turn an
    # already-published file into a cancelled export.
    try:
        report("complete", "Selected array exported.", int(data.nbytes), int(data.nbytes))
    except Exception:
        _LOGGER.exception("Export committed, but its completion observer failed")
    return RegionExportResult(tuple(paths), tuple(data.shape), data.dtype)
