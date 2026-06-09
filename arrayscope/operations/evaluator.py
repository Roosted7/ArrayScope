"""Evaluation and display cache for operation-backed documents."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.pipeline import ArrayDocument
from arrayscope.operations.cache import BoundedArrayCache
from arrayscope.operations.slabs import (
    evaluate_slab,
    plan_slab,
    request_for_export_frame,
    request_for_image,
    request_for_line,
    request_for_scalar,
)
from arrayscope.core.cache_status import (
    CacheStatusSnapshot,
    cache_status_computing,
    cache_status_error,
    cache_status_for_hit,
    cache_status_ready,
    CacheStatus,
)
from arrayscope.display.montage import RenderedTilePayload
from arrayscope.display.slice_engine import make_image, make_image_from_slab, make_line, make_line_from_slab, make_scalar_from_slab


DEFAULT_IMAGE_CACHE_BYTES = 256 * 1024 * 1024
DEFAULT_TILE_CACHE_BYTES = 512 * 1024 * 1024
DEFAULT_PROFILE_CACHE_BYTES = 64 * 1024 * 1024
LARGE_MATERIALIZE_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class EvaluationResult:
    value: object
    eval_ms: float
    slab_shape: tuple[int, ...]
    slab_nbytes: int | None
    mode: str = "lazy"
    chunk_count: int = 1
    degraded: bool = False


@dataclass
class OperationEvaluator:
    document: ArrayDocument
    _derived_key: tuple | None = None
    _derived_data: object | None = None
    _image_key: tuple | None = None
    _image_result: object | None = None
    _line_key: tuple | None = None
    _line_result: object | None = None
    derived_evaluations: int = 0
    image_evaluations: int = 0
    line_evaluations: int = 0
    scalar_evaluations: int = 0
    prefetch_scheduled: int = 0
    prefetch_deduped: int = 0
    prefetch_limited: int = 0
    prefetch_skipped: int = 0
    prefetch_stored: int = 0
    prefetch_stale: int = 0
    degraded_evaluations: int = 0
    refused_evaluations: int = 0
    chunked_evaluations: int = 0
    cancelled_evaluations: int = 0
    display_generation: int = 0
    last_status: CacheStatusSnapshot = CacheStatusSnapshot(CacheStatus.COLD, "No evaluation yet")
    last_diagnostics: object | None = None

    def __post_init__(self):
        self._image_cache = BoundedArrayCache(DEFAULT_IMAGE_CACHE_BYTES, 96)
        self._tile_cache = BoundedArrayCache(DEFAULT_TILE_CACHE_BYTES, 512)
        self._profile_cache = BoundedArrayCache(DEFAULT_PROFILE_CACHE_BYTES, 256)

    def set_document(self, document: ArrayDocument):
        if (
            document.steps != self.document.steps
            or document.base_data is not self.document.base_data
            or document.revision != self.document.revision
        ):
            self.document = document
            self.clear_cache()
        else:
            self.document = document

    def clear_cache(self):
        self._derived_key = None
        self._derived_data = None
        self._image_key = None
        self._image_result = None
        self._line_key = None
        self._line_result = None
        if hasattr(self, "_image_cache"):
            self._image_cache.clear()
        if hasattr(self, "_tile_cache"):
            self._tile_cache.clear()
        if hasattr(self, "_profile_cache"):
            self._profile_cache.clear()
        self.display_generation += 1
        self.last_status = CacheStatusSnapshot(CacheStatus.STALE, "Cache cleared")

    def current_data(self):
        key = _document_key(self.document)
        if self._derived_key == key:
            self.last_status = cache_status_for_hit(True)
            return self._derived_data

        self.last_status = cache_status_computing("Evaluating derived array")
        try:
            self._derived_data = self.document.materialize()
            self._derived_key = key
            self.derived_evaluations += 1
            self.last_status = cache_status_ready("Derived array cached")
            return self._derived_data
        except Exception as exc:
            self.last_status = cache_status_error(exc)
            raise

    def image(self, view_state, colormap_lut=None):
        request = request_for_image(view_state)
        key = self.image_key(view_state, colormap_lut=colormap_lut)
        cached = self._image_cache.get(key)
        if cached is not None:
            self.last_status = cache_status_for_hit(True)
            self.last_diagnostics = self._image_cache.diagnostics(CacheStatus.CACHED, "Using cached image view")
            return cached

        self.last_status = cache_status_computing("Evaluating image slab")
        try:
            result = evaluate_image_snapshot(self.document, view_state, colormap_lut=colormap_lut)
            return self.store_image_result(view_state, colormap_lut, result)
        except Exception as exc:
            self.last_status = cache_status_error(exc)
            raise

    def line(self, view_state):
        request = request_for_line(view_state)
        key = self.line_key(view_state)
        cached = self._profile_cache.get(key)
        if cached is not None:
            self.last_status = cache_status_for_hit(True)
            self.last_diagnostics = self._profile_cache.diagnostics(CacheStatus.CACHED, "Using cached profile")
            return cached

        self.last_status = cache_status_computing("Evaluating profile slab")
        try:
            result = evaluate_line_snapshot(self.document, view_state)
            return self.store_line_result(view_state, result)
        except Exception as exc:
            self.last_status = cache_status_error(exc)
            raise

    def scalar(self, view_state, index):
        request = request_for_scalar(view_state, index)
        key = self.scalar_key(view_state, index)
        cached = self._profile_cache.get(key)
        if cached is not None:
            self.last_status = cache_status_for_hit(True)
            self.last_diagnostics = self._profile_cache.diagnostics(CacheStatus.CACHED, "Using cached scalar")
            return cached

        self.last_status = cache_status_computing("Evaluating pixel value")
        try:
            result = evaluate_scalar_snapshot(self.document, view_state, index)
            return self.store_scalar_result(view_state, index, result)
        except Exception as exc:
            self.last_status = cache_status_error(exc)
            raise

    def export_frame(self, view_state, frame_axis, frame_index, colormap_lut=None):
        request = request_for_export_frame(view_state, frame_axis, frame_index)
        key = self.export_frame_key(view_state, frame_axis, frame_index, colormap_lut=colormap_lut)
        cached = self._image_cache.get(key)
        if cached is not None:
            self.last_status = cache_status_for_hit(True)
            self.last_diagnostics = self._image_cache.diagnostics(CacheStatus.CACHED, "Using cached export frame")
            return cached

        self.last_status = cache_status_computing("Evaluating export frame")
        try:
            result = evaluate_export_frame_snapshot(self.document, view_state, frame_axis, frame_index, colormap_lut=colormap_lut)
            return self.store_export_frame_result(view_state, frame_axis, frame_index, colormap_lut, result)
        except Exception as exc:
            self.last_status = cache_status_error(exc)
            raise

    def image_key(self, view_state, *, colormap_lut=None, document=None):
        document = self.document if document is None else document
        return ("image", _document_key(document), _request_key(request_for_image(view_state)), _lut_key(colormap_lut))

    def line_key(self, view_state, *, document=None):
        document = self.document if document is None else document
        return ("line", _document_key(document), _request_key(request_for_line(view_state)))

    def scalar_key(self, view_state, index, *, document=None):
        document = self.document if document is None else document
        return ("scalar", _document_key(document), _request_key(request_for_scalar(view_state, index)))

    def export_frame_key(self, view_state, frame_axis, frame_index, *, colormap_lut=None, document=None):
        document = self.document if document is None else document
        return ("export_frame", _document_key(document), _request_key(request_for_export_frame(view_state, frame_axis, frame_index)), _lut_key(colormap_lut))

    def montage_tile_key(self, tile_state, *, montage_axis, source_index, colormap_lut=None, document=None):
        document = self.document if document is None else document
        return (
            "montage_tile",
            _document_key(document),
            int(montage_axis),
            int(source_index),
            _request_key(request_for_image(tile_state)),
            _lut_key(colormap_lut),
        )

    def cached_image(self, view_state, colormap_lut=None):
        cached = self._image_cache.get(self.image_key(view_state, colormap_lut=colormap_lut))
        if cached is not None:
            self.last_status = cache_status_for_hit(True)
            self.last_diagnostics = self._image_cache.diagnostics(CacheStatus.CACHED, "Using cached image view")
        return cached

    def cached_montage_tile(self, tile_state, *, montage_axis, source_index, colormap_lut=None):
        cached = self._tile_cache.get(
            self.montage_tile_key(tile_state, montage_axis=montage_axis, source_index=source_index, colormap_lut=colormap_lut)
        )
        if cached is not None:
            self.last_status = cache_status_for_hit(True)
            self.last_diagnostics = self._tile_cache.diagnostics(CacheStatus.CACHED, "Using cached montage tile")
        return cached

    def cached_line(self, view_state):
        cached = self._profile_cache.get(self.line_key(view_state))
        if cached is not None:
            self.last_status = cache_status_for_hit(True)
            self.last_diagnostics = self._profile_cache.diagnostics(CacheStatus.CACHED, "Using cached profile")
        return cached

    def cached_scalar(self, view_state, index):
        cached = self._profile_cache.get(self.scalar_key(view_state, index))
        if cached is not None:
            self.last_status = cache_status_for_hit(True)
            self.last_diagnostics = self._profile_cache.diagnostics(CacheStatus.CACHED, "Using cached scalar")
        return cached

    def store_image_result(self, view_state, colormap_lut, result: EvaluationResult):
        key = self.image_key(view_state, colormap_lut=colormap_lut)
        self._image_cache.last_eval_ms = result.eval_ms
        self._image_result = result.value
        self._image_cache.put(key, result.value)
        self.image_evaluations += 1
        if result.mode == "chunked" or result.chunk_count > 1:
            self.note_chunked_evaluation()
        self.last_status = cache_status_ready("Image view cached")
        self.last_diagnostics = self._image_cache.diagnostics(CacheStatus.READY, _request_message("Image view cached", result))
        return result.value

    def store_line_result(self, view_state, result: EvaluationResult):
        key = self.line_key(view_state)
        self._profile_cache.last_eval_ms = result.eval_ms
        self._line_result = result.value
        self._profile_cache.put(key, result.value)
        self.line_evaluations += 1
        self.last_status = cache_status_ready("Profile cached")
        self.last_diagnostics = self._profile_cache.diagnostics(CacheStatus.READY, _request_message("Profile cached", result))
        return result.value

    def store_scalar_result(self, view_state, index, result: EvaluationResult):
        key = self.scalar_key(view_state, index)
        self._profile_cache.last_eval_ms = result.eval_ms
        self._profile_cache.put(key, result.value)
        self.scalar_evaluations += 1
        self.last_status = cache_status_ready("Pixel value cached")
        self.last_diagnostics = self._profile_cache.diagnostics(CacheStatus.READY, _request_message("Pixel value cached", result))
        return result.value

    def store_export_frame_result(self, view_state, frame_axis, frame_index, colormap_lut, result: EvaluationResult):
        key = self.export_frame_key(view_state, frame_axis, frame_index, colormap_lut=colormap_lut)
        self._image_cache.last_eval_ms = result.eval_ms
        self._image_cache.put(key, result.value)
        self.image_evaluations += 1
        self.last_status = cache_status_ready("Export frame cached")
        self.last_diagnostics = self._image_cache.diagnostics(CacheStatus.READY, _request_message("Export frame cached", result))
        return result.value

    def store_montage_tile_result(self, tile, *, montage_axis, colormap_lut, result: EvaluationResult):
        key = self.montage_tile_key(tile.view_state, montage_axis=montage_axis, source_index=tile.source_index, colormap_lut=colormap_lut)
        value = RenderedTilePayload(
            image=result.value.data,
            histogram_data=result.value.histogram_data,
            eval_ms=result.eval_ms,
            slab_shape=result.slab_shape,
            slab_nbytes=result.slab_nbytes,
        )
        self._tile_cache.last_eval_ms = result.eval_ms
        self._tile_cache.put(key, value)
        self.image_evaluations += 1
        self.last_status = cache_status_ready("Montage tile cached")
        self.last_diagnostics = self._tile_cache.diagnostics(CacheStatus.READY, _request_message("Montage tile cached", result))
        return value.bind(tile)

    def prefetch_image_snapshot(self, document, view_state, colormap_lut=None):
        key = self.image_key(view_state, colormap_lut=colormap_lut, document=document)
        if self._image_cache.get(key) is not None:
            self.prefetch_skipped += 1
            return None
        if self._image_cache.bytes_used > int(self._image_cache.max_bytes * 0.8):
            self.prefetch_skipped += 1
            return None
        return evaluate_image_snapshot(document, view_state, colormap_lut=colormap_lut)

    def store_prefetch_image_result(self, document, view_state, colormap_lut, result):
        if result is None:
            return False
        if _document_key(document) != _document_key(self.document):
            self.prefetch_stale += 1
            return False
        self.store_image_result(view_state, colormap_lut, result)
        self.prefetch_stored += 1
        return True

    def prefetch_line_snapshot(self, document, view_state):
        key = self.line_key(view_state, document=document)
        if self._profile_cache.get(key) is not None:
            self.prefetch_skipped += 1
            return None
        if self._profile_cache.bytes_used > int(self._profile_cache.max_bytes * 0.8):
            self.prefetch_skipped += 1
            return None
        return evaluate_line_snapshot(document, view_state)

    def store_prefetch_line_result(self, document, view_state, result):
        if result is None:
            return False
        if _document_key(document) != _document_key(self.document):
            self.prefetch_stale += 1
            return False
        self.store_line_result(view_state, result)
        self.prefetch_stored += 1
        return True

    def note_prefetch_scheduled(self):
        self.prefetch_scheduled += 1

    def note_prefetch_deduped(self):
        self.prefetch_deduped += 1

    def note_prefetch_limited(self):
        self.prefetch_limited += 1

    def note_prefetch_stale(self):
        self.prefetch_stale += 1

    def note_prefetch_skipped(self):
        self.prefetch_skipped += 1

    def note_render_refused(self, reason: str = ""):
        self.refused_evaluations += 1
        self.last_status = CacheStatusSnapshot(CacheStatus.STALE, str(reason or "Render refused"))

    def note_render_degraded(self):
        self.degraded_evaluations += 1

    def note_render_cancelled(self):
        self.cancelled_evaluations += 1

    def note_chunked_evaluation(self):
        self.chunked_evaluations += 1

    def cache_diagnostics(self):
        if self.last_diagnostics is not None:
            return self.last_diagnostics
        return self._image_cache.diagnostics(self.last_status.status, self.last_status.message)

    def apply_memory_policy(self, policy) -> None:
        self._image_cache.resize(max_bytes=int(policy.image_cache_budget_bytes))
        self._tile_cache.resize(max_bytes=int(policy.tile_cache_budget_bytes))
        self._profile_cache.resize(max_bytes=int(policy.profile_cache_budget_bytes))

    def image_cache_diagnostics(self):
        return self._image_cache.diagnostics(self.last_status.status, self.last_status.message, **self._prefetch_diagnostics())

    def tile_cache_diagnostics(self):
        return self._tile_cache.diagnostics(self.last_status.status, self.last_status.message, **self._prefetch_diagnostics())

    def profile_cache_diagnostics(self):
        return self._profile_cache.diagnostics(self.last_status.status, self.last_status.message, **self._prefetch_diagnostics())

    def derived_estimate(self):
        dtype = _estimated_dtype(self.document)
        nbytes = int(np.prod(self.document.current_shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        return tuple(self.document.current_shape), np.dtype(dtype), nbytes

    def _prefetch_diagnostics(self):
        return {
            "prefetch_scheduled": int(self.prefetch_scheduled),
            "prefetch_deduped": int(self.prefetch_deduped),
            "prefetch_limited": int(self.prefetch_limited),
            "prefetch_skipped": int(self.prefetch_skipped),
            "prefetch_stored": int(self.prefetch_stored),
            "prefetch_stale": int(self.prefetch_stale),
            "degraded_evaluations": int(self.degraded_evaluations),
            "refused_evaluations": int(self.refused_evaluations),
            "chunked_evaluations": int(self.chunked_evaluations),
            "cancelled_evaluations": int(self.cancelled_evaluations),
        }


def _document_key(document: ArrayDocument):
    dtype = getattr(document.base_data, "dtype", None)
    dtype_key = None if dtype is None else str(np.dtype(dtype))
    return (id(document.base_data), tuple(np.shape(document.base_data)), dtype_key, int(document.revision), document.steps)


def _lut_key(colormap_lut):
    if colormap_lut is None:
        return None
    lut = np.asarray(colormap_lut)
    return (lut.shape, str(lut.dtype), lut.tobytes())


def _request_key(request):
    return (
        request.kind,
        request.view_state,
        tuple(request.keep_axes),
        tuple(request.slice_indices),
        request.frame_axis,
        request.frame_index,
    )


def evaluate_image_snapshot(document, view_state, colormap_lut=None, cancellation_token=None, *, degraded=False) -> EvaluationResult:
    request = request_for_image(view_state)
    plan = plan_slab(document, request)
    start = perf_counter()
    _check_cancelled(cancellation_token)
    slab = evaluate_slab(document, request)
    _check_cancelled(cancellation_token)
    value = make_image_from_slab(slab, request, colormap_lut=colormap_lut)
    _check_cancelled(cancellation_token)
    return EvaluationResult(
        value=value,
        eval_ms=(perf_counter() - start) * 1000.0,
        slab_shape=tuple(np.shape(slab)),
        slab_nbytes=int(getattr(slab, "nbytes", plan.estimated_nbytes or 0)),
        degraded=bool(degraded),
    )


def evaluate_line_snapshot(document, view_state) -> EvaluationResult:
    request = request_for_line(view_state)
    plan = plan_slab(document, request)
    start = perf_counter()
    slab = evaluate_slab(document, request)
    value = make_line_from_slab(slab, request)
    return EvaluationResult(
        value=value,
        eval_ms=(perf_counter() - start) * 1000.0,
        slab_shape=tuple(np.shape(slab)),
        slab_nbytes=int(getattr(slab, "nbytes", plan.estimated_nbytes or 0)),
    )


def evaluate_scalar_snapshot(document, view_state, index) -> EvaluationResult:
    request = request_for_scalar(view_state, index)
    plan = plan_slab(document, request)
    start = perf_counter()
    slab = evaluate_slab(document, request)
    value = make_scalar_from_slab(slab, request)
    return EvaluationResult(
        value=value,
        eval_ms=(perf_counter() - start) * 1000.0,
        slab_shape=tuple(np.shape(slab)),
        slab_nbytes=int(getattr(slab, "nbytes", plan.estimated_nbytes or 0)),
    )


def evaluate_export_frame_snapshot(document, view_state, frame_axis, frame_index, colormap_lut=None) -> EvaluationResult:
    request = request_for_export_frame(view_state, frame_axis, frame_index)
    plan = plan_slab(document, request)
    start = perf_counter()
    slab = evaluate_slab(document, request)
    value = make_image_from_slab(slab, request, colormap_lut=colormap_lut)
    return EvaluationResult(
        value=value,
        eval_ms=(perf_counter() - start) * 1000.0,
        slab_shape=tuple(np.shape(slab)),
        slab_nbytes=int(getattr(slab, "nbytes", plan.estimated_nbytes or 0)),
    )


def _request_message(prefix, result: EvaluationResult):
    nbytes = "unknown"
    if result.slab_nbytes is not None:
        nbytes = _format_nbytes(result.slab_nbytes)
    chunks = "" if result.chunk_count <= 1 else f", {result.chunk_count} chunks"
    degraded = ", degraded preview" if result.degraded else ""
    return f"{prefix}; last request {result.mode}{chunks}{degraded}, slab {result.slab_shape}, {nbytes}"


def _check_cancelled(token):
    if token is not None and getattr(token, "cancelled", False):
        raise EvaluationCancelled()


def _format_nbytes(nbytes):
    nbytes = int(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024 or unit == "GiB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024


def _estimated_dtype(document):
    dtype = getattr(document.base_data, "dtype", np.dtype(float))
    try:
        from arrayscope.operations.coordinator import _operation_output_dtype

        for operation in document.enabled_operations:
            dtype = _operation_output_dtype(dtype, operation)
    except Exception:
        pass
    return np.dtype(dtype)
