"""Evaluation and display cache for operation-backed documents."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from arrayscope.core.cache_status import (
    CacheStatus,
    CacheStatusSnapshot,
    cache_status_computing,
    cache_status_error,
    cache_status_for_hit,
    cache_status_ready,
)
from arrayscope.display.model.montage_levels import (
    provisional_tile_level_stats,
    sample_tile_level_stats,
)
from arrayscope.display.montage import RenderedTilePayload
from arrayscope.display.shader_mapping import apply_scale as apply_shader_scale
from arrayscope.display.shader_mapping import extract_component
from arrayscope.display.slice_engine import (
    make_image_from_slab,
    make_line_from_slab,
    make_scalar_from_slab,
    make_shader_image_from_slab,
)
from arrayscope.operations.cache import BoundedArrayCache
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.compressed_tier import (
    CompressedBackingTier,
    TwoLevelArrayCache,
    split_payload_for_tier,
)
from arrayscope.operations.cost import operation_output_dtype
from arrayscope.operations.pipeline import ArrayDocument
from arrayscope.operations.slabs import (
    evaluate_slab_from_plan,
    plan_slab,
    request_for_export_frame,
    request_for_image,
    request_for_line,
    request_for_scalar,
)
from arrayscope.operations.stage_cache import StageCache
from arrayscope.operations.stage_materialization import StageMaterializationManager

DEFAULT_DISPLAY_CACHE_BYTES = 512 * 1024 * 1024
DEFAULT_PROFILE_CACHE_BYTES = 64 * 1024 * 1024
DEFAULT_STAGE_CACHE_BYTES = 512 * 1024 * 1024
DEFAULT_STAGE_CACHE_ENTRIES = 64
LARGE_MATERIALIZE_BYTES = 512 * 1024 * 1024
COMPRESSED_TIER_RAW_FRACTION = 0.25


def _build_array_cache(max_bytes: int, max_entries: int, codec: str | None):
    """Build a display/region/profile cache, tier-backed unless codec is RAW.

    ``codec`` RAW (or empty/None) -> a plain ``BoundedArrayCache``, byte-for-byte
    identical to before this feature existed.  Otherwise a ``TwoLevelArrayCache``:
    one total budget split between a raw hot tier and compressed backing tier.
    Values evicted from the raw cache are compressed synchronously into the tier
    (best lossless codec per dtype under ``auto``) and recovered by a synchronous
    decode on a later revisit. This remains an explicit experiment: only an
    off-thread tier at an expensive miss owner could justify AUTO. The configured
    memory-policy budget remains the physical byte-accounting owner.
    """

    codec = (codec or "raw").strip().lower()
    if codec == "raw":
        return BoundedArrayCache(int(max_bytes), int(max_entries))
    raw_bytes = max(0, int(max_bytes * COMPRESSED_TIER_RAW_FRACTION))
    tier_bytes = max(0, int(max_bytes) - raw_bytes)
    tier = CompressedBackingTier(max_bytes=tier_bytes, codec_name=codec)
    return TwoLevelArrayCache.build(
        raw_max_bytes=raw_bytes,
        raw_max_entries=int(max_entries),
        tier=tier,
        store_adapter=split_payload_for_tier,
        raw_budget_fraction=COMPRESSED_TIER_RAW_FRACTION,
    )


@dataclass(frozen=True)
class EvaluationResult:
    value: object
    eval_ms: float
    slab_shape: tuple[int, ...]
    slab_nbytes: int | None
    mode: str = "lazy"
    chunk_count: int = 1
    degraded: bool = False
    region_plan: object | None = None
    compute_path: str = "direct"


@dataclass(frozen=True)
class LevelEvidenceSourceResult:
    """Statistics-only result for one semantic montage source."""

    source_index: int
    stats: object | None
    sampled_pixels: int
    slab_shape: tuple[int, ...]
    slab_nbytes: int
    region_plan: object | None = None


@dataclass(frozen=True)
class LevelEvidenceBatchResult:
    """Bounded evidence result with no display or presentation payloads."""

    sources: tuple[LevelEvidenceSourceResult, ...]
    pixel_limit: int
    elapsed_ms: float

    @property
    def source_indices(self) -> tuple[int, ...]:
        return tuple(int(source.source_index) for source in self.sources)

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def requested_pixels(self) -> int:
        return sum(int(source.sampled_pixels) for source in self.sources)

    @property
    def max_source_pixels(self) -> int:
        return max((int(source.sampled_pixels) for source in self.sources), default=0)

    @property
    def slab_nbytes(self) -> int:
        return sum(int(source.slab_nbytes) for source in self.sources)


# Shared immutable default for OperationEvaluator.last_status (frozen dataclass).
_NO_EVALUATION_STATUS = CacheStatusSnapshot(CacheStatus.COLD, "No evaluation yet")


@dataclass
class OperationEvaluator:
    document: ArrayDocument
    # G7 host-cache compression codec name ("auto"/"raw"/"zfp"/"blosc2").  RAW
    # keeps the evaluator's caches byte-identical plain BoundedArrayCaches; any
    # other value backs the big display/region/profile caches with a compressed
    # tier (see ``_build_display_cache``).  Threaded from the app settings via
    # the coordinator; defaults to raw so headless/library use is unchanged.
    chunk_transport_codec: str = "raw"
    # When True, display tiles are materialized and keyed in canonical
    # (sorted-image-axes) orientation and an X/Y axis-order swap becomes a pure
    # display transform applied by the backend rather than baked into pixels.
    # Set per-frame from the active backend's ``display_axis_transpose``
    # capability; keys sort image_axes so a canonical entry never collides with
    # a legacy display-order entry (a flag flip degrades to a cache miss, never
    # a wrong-orientation reuse).
    canonical_orientation: bool = False
    _derived_key: tuple | None = None
    _derived_data: object | None = None
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
    last_status: CacheStatusSnapshot = _NO_EVALUATION_STATUS
    last_diagnostics: object | None = None
    last_region_plan: object | None = None

    def __post_init__(self):
        self._display_cache = _build_array_cache(
            DEFAULT_DISPLAY_CACHE_BYTES, 512, self.chunk_transport_codec
        )
        self._profile_cache = _build_array_cache(
            DEFAULT_PROFILE_CACHE_BYTES, 256, self.chunk_transport_codec
        )
        # This is the exact demand-ROI cache used by TileDataProvider.  It is
        # not the GPU source-grid/page store and it must not share eviction
        # state with the display-materialization cache.
        self._region_cache = _build_array_cache(
            DEFAULT_DISPLAY_CACHE_BYTES, 512, self.chunk_transport_codec
        )
        # StageCache is intentionally NOT wired to the compressed tier here: its
        # retention scoring, in-flight-compute claims and lock-free resident
        # snapshots make it materially riskier than the BoundedArrayCache-based
        # caches above.  It stays a plain StageCache; adopting the tier is a
        # bounded follow-up (the tier is value-generic for exactly that).
        self._stage_cache = StageCache(
            max_bytes=DEFAULT_STAGE_CACHE_BYTES, max_entries=DEFAULT_STAGE_CACHE_ENTRIES
        )
        self._stage_materializer = StageMaterializationManager(self._stage_cache)

    def set_chunk_transport_codec(self, codec: str | None) -> bool:
        """Switch the host-cache compression codec live, rebuilding the caches.

        Rebuilds the display/profile/region caches with the new codec (RAW = a
        plain cache; else a compressed backing tier). The rebuilt caches start
        empty — the new codec applies to everything cached from here on — so this
        lets a Performance-menu change take effect without a restart. No-op when
        the codec is unchanged. Returns True if a rebuild happened.
        """

        normalized = (codec or "raw").strip().lower()
        if normalized == (self.chunk_transport_codec or "raw").strip().lower():
            return False
        display_bytes = int(self._display_cache.max_bytes)
        profile_bytes = int(self._profile_cache.max_bytes)
        region_bytes = int(self._region_cache.max_bytes)
        self.chunk_transport_codec = normalized
        self._display_cache = _build_array_cache(display_bytes, 512, normalized)
        self._profile_cache = _build_array_cache(profile_bytes, 256, normalized)
        self._region_cache = _build_array_cache(region_bytes, 512, normalized)
        self.last_diagnostics = None
        return True

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
        self._line_key = None
        self._line_result = None
        if hasattr(self, "_display_cache"):
            self._display_cache.clear()
        if hasattr(self, "_profile_cache"):
            self._profile_cache.clear()
        if hasattr(self, "_region_cache"):
            self._region_cache.clear()
        if hasattr(self, "_stage_cache"):
            self._stage_cache.clear()
        if hasattr(self, "_stage_materializer"):
            self._stage_materializer.clear()
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
        request_for_image(view_state)
        key = self.display_tile_key(view_state, colormap_lut=colormap_lut)
        cached = self._display_cache.get(key)
        if cached is not None:
            self.last_status = cache_status_for_hit(True)
            self.last_diagnostics = self._display_cache.diagnostics(
                CacheStatus.CACHED, "Using cached display tile"
            )
            return cached

        self.last_status = cache_status_computing("Evaluating image slab")
        try:
            result = evaluate_image_snapshot(
                self.document,
                view_state,
                colormap_lut=colormap_lut,
                stage_cache=self._stage_cache,
                stage_document_key=stage_document_key(self.document),
                canonical_orientation=self.canonical_orientation,
            )
            return self.store_display_tile_result(view_state, colormap_lut, result)
        except Exception as exc:
            self.last_status = cache_status_error(exc)
            raise

    def line(self, view_state):
        request_for_line(view_state)
        key = self.line_key(view_state)
        cached = self._profile_cache.get(key)
        if cached is not None:
            self.last_status = cache_status_for_hit(True)
            self.last_diagnostics = self._profile_cache.diagnostics(
                CacheStatus.CACHED, "Using cached profile"
            )
            return cached

        self.last_status = cache_status_computing("Evaluating profile slab")
        try:
            result = evaluate_line_snapshot(
                self.document,
                view_state,
                stage_cache=self._stage_cache,
                stage_document_key=stage_document_key(self.document),
            )
            return self.store_line_result(view_state, result)
        except Exception as exc:
            self.last_status = cache_status_error(exc)
            raise

    def scalar(self, view_state, index):
        request_for_scalar(view_state, index)
        key = self.scalar_key(view_state, index)
        cached = self._profile_cache.get(key)
        if cached is not None:
            self.last_status = cache_status_for_hit(True)
            self.last_diagnostics = self._profile_cache.diagnostics(
                CacheStatus.CACHED, "Using cached scalar"
            )
            return cached

        self.last_status = cache_status_computing("Evaluating pixel value")
        try:
            result = evaluate_scalar_snapshot(
                self.document,
                view_state,
                index,
                stage_cache=self._stage_cache,
                stage_document_key=stage_document_key(self.document),
            )
            return self.store_scalar_result(view_state, index, result)
        except Exception as exc:
            self.last_status = cache_status_error(exc)
            raise

    def export_frame(self, view_state, frame_axis, frame_index, colormap_lut=None):
        request_for_export_frame(view_state, frame_axis, frame_index)
        key = self.export_frame_key(view_state, frame_axis, frame_index, colormap_lut=colormap_lut)
        cached = self._display_cache.get(key)
        if cached is not None:
            self.last_status = cache_status_for_hit(True)
            self.last_diagnostics = self._display_cache.diagnostics(
                CacheStatus.CACHED, "Using cached export tile"
            )
            return cached

        self.last_status = cache_status_computing("Evaluating export frame")
        try:
            result = evaluate_export_frame_snapshot(
                self.document,
                view_state,
                frame_axis,
                frame_index,
                colormap_lut=colormap_lut,
                stage_cache=self._stage_cache,
                stage_document_key=stage_document_key(self.document),
            )
            return self.store_export_frame_result(
                view_state, frame_axis, frame_index, colormap_lut, result
            )
        except Exception as exc:
            self.last_status = cache_status_error(exc)
            raise

    def display_tile_key(
        self,
        view_state,
        *,
        montage_axis=None,
        source_index=None,
        tile_number: int | None = 0,
        colormap_lut=None,
        document=None,
        shader_display: bool = False,
    ):
        document = self.document if document is None else document
        return _display_tile_key_from_parts(
            _document_key(document),
            None if montage_axis is None else int(montage_axis),
            None if source_index is None else int(source_index),
            None if tile_number is None else int(tile_number),
            _request_key(
                request_for_image(view_state), canonical_orientation=self.canonical_orientation
            ),
            _lut_key(colormap_lut),
            bool(shader_display),
        )

    def line_key(self, view_state, *, document=None):
        document = self.document if document is None else document
        return ("line", _document_key(document), _request_key(request_for_line(view_state)))

    def scalar_key(self, view_state, index, *, document=None):
        document = self.document if document is None else document
        return (
            "scalar",
            _document_key(document),
            _request_key(request_for_scalar(view_state, index)),
        )

    def export_frame_key(
        self, view_state, frame_axis, frame_index, *, colormap_lut=None, document=None
    ):
        document = self.document if document is None else document
        return (
            "export_tile",
            _document_key(document),
            _request_key(request_for_export_frame(view_state, frame_axis, frame_index)),
            _lut_key(colormap_lut),
        )

    def montage_tile_key(
        self,
        tile_state,
        *,
        montage_axis,
        source_index,
        colormap_lut=None,
        document=None,
        shader_display: bool = False,
    ):
        return self.display_tile_key(
            tile_state,
            colormap_lut=colormap_lut,
            document=document,
            shader_display=shader_display,
        )

    def montage_tile_key_batch(
        self, *, colormap_lut=None, document=None, shader_display: bool = False
    ):
        """Return ``key_for(tile_state)`` with tile-invariant key work hoisted.

        A montage step derives keys for every plan tile several times (cache
        resolve, source-id table), and profiling shows the per-tile rebuild
        of the document key, LUT bytes, and view-state tuple dominating
        scrub steps (~50k genexpr calls per 21 steps).  Across one plan's
        tile states everything except ``slice_indices`` is identical
        (montage_viewport strips montage fields and varies only the montage
        axis index), so the batch computes the invariants once and swaps the
        slice tuple per tile.

        Correctness is self-checked at runtime: the first tile whose slices
        differ from the template is also keyed through the unbatched path,
        and any mismatch flips the batch into permanent slow-path fallback
        (counted in ``montage_key_batch_fallbacks``) — key-format drift
        degrades to the old cost, never to wrong keys.

        Key *layout* is owned elsewhere: the slow path is exactly
        ``display_tile_key`` (the scalar owner), and the fast path assembles
        through the same ``_display_tile_key_from_parts`` /
        ``_request_key_from_parts`` / ``_view_state_key_with_slices`` helpers
        that back the scalar derivation.  The only knowledge this closure adds
        is the hoisting bet itself — that a plan's tile states vary solely in
        ``slice_indices`` — and that bet is what the runtime check guards.
        """

        document = self.document if document is None else document
        doc_key = _document_key(document)
        lut_key = _lut_key(colormap_lut)
        shader = bool(shader_display)
        state: dict[str, object] = {"template": None, "validated": False}

        def _slow(tile_state):
            return self.display_tile_key(
                tile_state,
                colormap_lut=colormap_lut,
                document=document,
                shader_display=shader_display,
            )

        def key_for(tile_state):
            template = state["template"]
            if template is None:
                base_key = _view_state_key(
                    tile_state, canonical_orientation=self.canonical_orientation
                )
                keep_axes = _canonical_axis_tuple(
                    tile_state.image_axes or (), self.canonical_orientation
                )
                base_slices = tuple(int(index) for index in tile_state.slice_indices)
                state["template"] = (base_key, keep_axes, base_slices)
                return _slow(tile_state)
            if state.get("fallback"):
                return _slow(tile_state)
            base_key, keep_axes, base_slices = template
            slices = tuple(int(index) for index in tile_state.slice_indices)
            fast = _display_tile_key_from_parts(
                doc_key,
                None,
                None,
                0,
                _request_key_from_parts(
                    "image",
                    _view_state_key_with_slices(base_key, slices),
                    keep_axes,
                    slices,
                    None,
                    None,
                ),
                lut_key,
                shader,
            )
            if not state["validated"] and slices != base_slices:
                state["validated"] = True
                if fast != _slow(tile_state):
                    state["fallback"] = True
                    self.montage_key_batch_fallbacks = (
                        int(getattr(self, "montage_key_batch_fallbacks", 0)) + 1
                    )
                    return _slow(tile_state)
            return fast

        return key_for

    def cached_display_tile(
        self, view_state, colormap_lut=None, *, document=None, shader_display: bool = False
    ):
        cached = self._display_cache.get(
            self.display_tile_key(
                view_state,
                colormap_lut=colormap_lut,
                document=document,
                shader_display=shader_display,
            )
        )
        if cached is not None:
            self.last_status = cache_status_for_hit(True)
            self.last_diagnostics = self._display_cache.diagnostics(
                CacheStatus.CACHED, "Using cached display tile"
            )
        return cached

    def cached_montage_tile_by_key(self, tile_key):
        """Display-cache lookup for a key produced by ``montage_tile_key_batch``."""

        cached = self._display_cache.get(tile_key)
        if cached is not None:
            self.last_status = cache_status_for_hit(True)
            self.last_diagnostics = self._display_cache.diagnostics(
                CacheStatus.CACHED, "Using cached display tile"
            )
        return cached

    def cached_montage_tile(
        self,
        tile_state,
        *,
        montage_axis,
        source_index,
        colormap_lut=None,
        document=None,
        shader_display: bool = False,
    ):
        cached = self._display_cache.get(
            self.montage_tile_key(
                tile_state,
                montage_axis=montage_axis,
                source_index=source_index,
                colormap_lut=colormap_lut,
                document=document,
                shader_display=shader_display,
            )
        )
        if cached is not None:
            self.last_status = cache_status_for_hit(True)
            self.last_diagnostics = self._display_cache.diagnostics(
                CacheStatus.CACHED, "Using cached display tile"
            )
        return cached

    def cached_line(self, view_state):
        cached = self._profile_cache.get(self.line_key(view_state))
        if cached is not None:
            self.last_status = cache_status_for_hit(True)
            self.last_diagnostics = self._profile_cache.diagnostics(
                CacheStatus.CACHED, "Using cached profile"
            )
        return cached

    def cached_scalar(self, view_state, index):
        cached = self._profile_cache.get(self.scalar_key(view_state, index))
        if cached is not None:
            self.last_status = cache_status_for_hit(True)
            self.last_diagnostics = self._profile_cache.diagnostics(
                CacheStatus.CACHED, "Using cached scalar"
            )
        return cached

    def tile_region_key(self, request, *, document=None):
        document = self.document if document is None else document
        region = getattr(request, "tile_local_region", None)
        return (
            "tile_region",
            _document_key(document),
            _request_key(request_for_image(request.view_state)),
            None if request.montage_axis is None else int(request.montage_axis),
            None if request.source_index is None else int(request.source_index),
            None if request.tile_number is None else int(request.tile_number),
            _slice_key(region),
        )

    def cached_tile_region(self, request):
        cached = self._region_cache.get(self.tile_region_key(request))
        if cached is not None:
            self.last_status = cache_status_for_hit(True)
            self.last_diagnostics = self._region_cache.diagnostics(
                CacheStatus.CACHED, "Using cached tile region"
            )
        return cached

    def cached_tile_region_silent(self, request):
        return self._region_cache.get(self.tile_region_key(request))

    def cached_montage_tile_silent(
        self,
        tile_state,
        *,
        montage_axis,
        source_index,
        colormap_lut=None,
        document=None,
        shader_display: bool = False,
    ):
        return self._display_cache.get(
            self.montage_tile_key(
                tile_state,
                montage_axis=montage_axis,
                source_index=source_index,
                colormap_lut=colormap_lut,
                document=document,
                shader_display=shader_display,
            )
        )

    def store_tile_region_result(self, request, result):
        self._region_cache.put(self.tile_region_key(request), result)
        self.last_status = cache_status_ready("Tile region cached")
        self.last_diagnostics = self._region_cache.diagnostics(
            CacheStatus.READY, "Tile region cached"
        )
        return result

    def store_tile_region_result_silent(self, request, result):
        return self._region_cache.put(self.tile_region_key(request), result)

    def evaluate_image_snapshot_silent(
        self, document, view_state, colormap_lut=None, *, evaluation_context=None
    ):
        return evaluate_image_snapshot(
            document,
            view_state,
            colormap_lut=colormap_lut,
            stage_cache=self._stage_cache,
            stage_document_key=stage_document_key(document),
            evaluation_context=evaluation_context,
            canonical_orientation=self.canonical_orientation,
        )

    def level_evidence(
        self,
        view_state,
        source_indices,
        *,
        pixel_limit: int,
        cancellation_token=None,
        evaluation_context=None,
        document=None,
    ) -> LevelEvidenceBatchResult:
        document = self.document if document is None else document
        return evaluate_level_evidence_snapshot(
            document,
            view_state,
            source_indices,
            pixel_limit=int(pixel_limit),
            cancellation_token=cancellation_token,
            stage_cache=self._stage_cache,
            stage_document_key=stage_document_key(document),
            evaluation_context=evaluation_context,
        )

    def store_display_tile_result(
        self,
        view_state,
        colormap_lut,
        result: EvaluationResult,
        *,
        document=None,
        shader_display: bool = False,
    ):
        key = self.display_tile_key(
            view_state,
            colormap_lut=colormap_lut,
            document=document,
            shader_display=shader_display,
        )
        self._display_cache.last_eval_ms = result.eval_ms
        self.last_region_plan = result.region_plan
        self._display_cache.put(key, result.value)
        self.image_evaluations += 1
        if result.mode == "chunked" or result.chunk_count > 1:
            self.note_chunked_evaluation()
        self.last_status = cache_status_ready("Display tile cached")
        self.last_diagnostics = self._display_cache.diagnostics(
            CacheStatus.READY, _request_message("Display tile cached", result)
        )
        return result.value

    def store_line_result(self, view_state, result: EvaluationResult):
        key = self.line_key(view_state)
        self._profile_cache.last_eval_ms = result.eval_ms
        self._line_result = result.value
        self.last_region_plan = result.region_plan
        self._profile_cache.put(key, result.value)
        self.line_evaluations += 1
        self.last_status = cache_status_ready("Profile cached")
        self.last_diagnostics = self._profile_cache.diagnostics(
            CacheStatus.READY, _request_message("Profile cached", result)
        )
        return result.value

    def store_scalar_result(self, view_state, index, result: EvaluationResult):
        key = self.scalar_key(view_state, index)
        self._profile_cache.last_eval_ms = result.eval_ms
        self.last_region_plan = result.region_plan
        self._profile_cache.put(key, result.value)
        self.scalar_evaluations += 1
        self.last_status = cache_status_ready("Pixel value cached")
        self.last_diagnostics = self._profile_cache.diagnostics(
            CacheStatus.READY, _request_message("Pixel value cached", result)
        )
        return result.value

    def store_export_frame_result(
        self, view_state, frame_axis, frame_index, colormap_lut, result: EvaluationResult
    ):
        key = self.export_frame_key(view_state, frame_axis, frame_index, colormap_lut=colormap_lut)
        self._display_cache.last_eval_ms = result.eval_ms
        self.last_region_plan = result.region_plan
        self._display_cache.put(key, result.value)
        self.image_evaluations += 1
        self.last_status = cache_status_ready("Export tile cached")
        self.last_diagnostics = self._display_cache.diagnostics(
            CacheStatus.READY, _request_message("Export tile cached", result)
        )
        return result.value

    def store_montage_tile_result(
        self,
        tile,
        *,
        montage_axis,
        colormap_lut,
        result: EvaluationResult,
        document=None,
        shader_display: bool = False,
    ):
        key = self.montage_tile_key(
            tile.view_state,
            montage_axis=montage_axis,
            source_index=tile.source_index,
            colormap_lut=colormap_lut,
            document=document,
            shader_display=shader_display,
        )
        level_stats = getattr(result.value, "level_stats", None)
        level_data = getattr(result.value, "level_data", None)
        if level_stats is None:
            values = _display_image_level_values(result.value)
            level_stats = (
                sample_tile_level_stats(values, int(tile.source_index), refined=True)
                if not bool(shader_display)
                else provisional_tile_level_stats(values, int(tile.source_index))
            )
        value = RenderedTilePayload(
            image=result.value.data,
            histogram_data=result.value.histogram_data,
            eval_ms=result.eval_ms,
            slab_shape=result.slab_shape,
            slab_nbytes=result.slab_nbytes,
            shader_mapping=getattr(result.value, "shader_mapping", None),
            texture_kind=getattr(result.value, "texture_kind", None),
            semantic_data=getattr(result.value, "semantic_data", None),
            lod_source_data=getattr(result.value, "lod_source_data", None),
            lod=getattr(result.value, "lod", None),
            level_data=level_data,
            level_stats=level_stats,
        )
        self._display_cache.last_eval_ms = result.eval_ms
        self.last_region_plan = result.region_plan
        self._display_cache.put(key, value)
        self.image_evaluations += 1
        self.last_status = cache_status_ready("Display tile cached")
        self.last_diagnostics = self._display_cache.diagnostics(
            CacheStatus.READY, _request_message("Display tile cached", result)
        )
        return value.bind(tile)

    def prefetch_display_tile_snapshot(
        self,
        document,
        view_state,
        colormap_lut=None,
        *,
        evaluation_context=None,
        shader_display: bool = False,
    ):
        key = self.display_tile_key(
            view_state,
            colormap_lut=colormap_lut,
            document=document,
            shader_display=shader_display,
        )
        if self._display_cache.get(key) is not None:
            self.prefetch_skipped += 1
            return None
        if self._display_cache.bytes_used > int(self._display_cache.max_bytes * 0.8):
            self.prefetch_skipped += 1
            return None
        return evaluate_image_snapshot(
            document,
            view_state,
            colormap_lut=colormap_lut,
            stage_cache=self._stage_cache,
            stage_document_key=stage_document_key(document),
            evaluation_context=evaluation_context,
            shader_display=shader_display,
            canonical_orientation=self.canonical_orientation,
        )

    def store_prefetch_display_tile_result(
        self, document, view_state, colormap_lut, result, *, shader_display: bool = False
    ):
        if result is None:
            return False
        if _document_key(document) != _document_key(self.document):
            self.prefetch_stale += 1
            return False
        self.store_display_tile_result(
            view_state,
            colormap_lut,
            result,
            document=document,
            shader_display=shader_display,
        )
        self.prefetch_stored += 1
        return True

    def prefetch_line_snapshot(self, document, view_state, *, evaluation_context=None):
        key = self.line_key(view_state, document=document)
        if self._profile_cache.get(key) is not None:
            self.prefetch_skipped += 1
            return None
        if self._profile_cache.bytes_used > int(self._profile_cache.max_bytes * 0.8):
            self.prefetch_skipped += 1
            return None
        return evaluate_line_snapshot(
            document,
            view_state,
            stage_cache=self._stage_cache,
            stage_document_key=stage_document_key(document),
            evaluation_context=evaluation_context,
        )

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
        return self._display_cache.diagnostics(self.last_status.status, self.last_status.message)

    def apply_memory_policy(self, policy) -> None:
        self._display_cache.resize(max_bytes=int(policy.display_cache_budget_bytes))
        self._region_cache.resize(max_bytes=int(policy.display_cache_budget_bytes))
        self._profile_cache.resize(max_bytes=int(policy.profile_cache_budget_bytes))
        self._stage_cache.resize(max_bytes=int(policy.stage_cache_budget_bytes))

    def display_cache_diagnostics(self):
        return self._display_cache.diagnostics(
            self.last_status.status, self.last_status.message, **self._prefetch_diagnostics()
        )

    def profile_cache_diagnostics(self):
        return self._profile_cache.diagnostics(
            self.last_status.status, self.last_status.message, **self._prefetch_diagnostics()
        )

    @property
    def stage_cache(self):
        return self._stage_cache

    @property
    def stage_materializer(self):
        return self._stage_materializer

    def stage_cache_diagnostics(self):
        return self._stage_cache.diagnostics()

    def stage_materialization_diagnostics(self):
        return self._stage_materializer.diagnostics()

    def derived_estimate(self):
        dtype = _estimated_dtype(self.document)
        nbytes = (
            int(np.prod(self.document.current_shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        )
        return tuple(self.document.current_shape), np.dtype(dtype), nbytes

    def planner_diagnostics(self):
        return self.last_region_plan

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
    dtype_key = None if dtype is None else np.dtype(dtype)
    return (
        id(document.base_data),
        tuple(np.shape(document.base_data)),
        dtype_key,
        int(document.revision),
        document.steps,
    )


def stage_document_key(document: ArrayDocument):
    dtype = getattr(document.base_data, "dtype", None)
    dtype_key = None if dtype is None else np.dtype(dtype)
    return (
        id(document.base_data),
        tuple(np.shape(document.base_data)),
        dtype_key,
        int(document.revision),
    )


def _lut_key(colormap_lut):
    if colormap_lut is None:
        return None
    lut = np.asarray(colormap_lut)
    return (lut.shape, str(lut.dtype), lut.tobytes())


def _display_tile_key_from_parts(
    document_key, montage_axis, source_index, tile_number, request_key, lut_key, shader_display
):
    """Sole owner of the display-tile key layout.

    Every display/montage tile key — scalar (``display_tile_key``) and hoisted
    (``montage_tile_key_batch``) — must assemble through this function so the
    layout cannot drift between derivation paths.
    """
    return (
        "display_tile",
        document_key,
        montage_axis,
        source_index,
        tile_number,
        request_key,
        lut_key,
        shader_display,
    )


def _request_key(request, canonical_orientation: bool = False):
    return _request_key_from_parts(
        request.kind,
        _view_state_key(request.view_state, canonical_orientation=canonical_orientation),
        _canonical_axis_tuple(request.keep_axes, canonical_orientation),
        tuple(request.slice_indices),
        request.frame_axis,
        request.frame_index,
    )


def _request_key_from_parts(
    kind, view_state_key, keep_axes, slice_indices, frame_axis, frame_index
):
    """Sole owner of the request-key layout (shared with ``montage_tile_key_batch``)."""
    return (kind, view_state_key, keep_axes, slice_indices, frame_axis, frame_index)


def _canonical_axis_tuple(axes, canonical_orientation: bool):
    """Sort image/keep axes when canonical so a transpose reuses the same key.

    Canonicalizing keys means an X/Y-swapped view keys to the same slot as the
    unswapped view (its payload is materialized canonically), while a legacy
    display-order view keeps its distinct order -- so the two key spaces never
    collide.  The slab is read set-wise (planner treats keep_axes as a set) and
    always emerges in ascending-axis order, so sorting here only affects the
    key, never which pixels are read.
    """

    if axes is None:
        return None
    axes = tuple(int(axis) for axis in axes)
    return tuple(sorted(axes)) if canonical_orientation else axes


def _view_state_key(view_state, canonical_orientation: bool = False):
    if view_state is None:
        return None
    return (
        int(getattr(view_state, "ndim", 0)),
        tuple(int(size) for size in getattr(view_state, "shape", ())),
        _canonical_axis_tuple(getattr(view_state, "image_axes", None), canonical_orientation),
        None if getattr(view_state, "line_axis", None) is None else int(view_state.line_axis),
        tuple(int(index) for index in getattr(view_state, "slice_indices", ())),
        getattr(
            getattr(view_state, "channel", None), "value", getattr(view_state, "channel", None)
        ),
        getattr(getattr(view_state, "scale", None), "value", getattr(view_state, "scale", None)),
        # Axis direction is a display transform, not a pixel-evaluation input.
        # Keeping it out of cache keys lets rapid flip interactions reuse the
        # same evaluated image/tile payloads instead of restarting compute.
        tuple(bool(value) for value in getattr(view_state, "axis_fftshifted", ())),
        None if getattr(view_state, "montage_axis", None) is None else int(view_state.montage_axis),
        None
        if getattr(view_state, "montage_columns", None) is None
        else int(view_state.montage_columns),
        _optional_int_tuple(getattr(view_state, "montage_indices", None)),
        _range_key(getattr(view_state, "axis_range_indices", ())),
    )


# Position of the slice_indices tuple inside _view_state_key's layout.  Keep
# in lockstep with the tuple above; montage_tile_key_batch's runtime parity
# check (and its regression test) trips if this slot drifts.
_VIEW_STATE_KEY_SLICES_SLOT = 4


def _view_state_key_with_slices(view_state_key, slice_indices):
    """Return ``view_state_key`` with its slice_indices slot replaced."""
    return (
        *view_state_key[:_VIEW_STATE_KEY_SLICES_SLOT],
        slice_indices,
        *view_state_key[_VIEW_STATE_KEY_SLICES_SLOT + 1 :],
    )


def _optional_int_tuple(values):
    if values is None:
        return None
    return tuple(int(value) for value in values)


def _range_key(values):
    return tuple(
        None if value is None else tuple(int(index) for index in value) for value in values
    )


def _slice_key(region):
    if region is None:
        return None
    key = [
        (
            None if slc.start is None else int(slc.start),
            None if slc.stop is None else int(slc.stop),
            None if slc.step is None else int(slc.step),
        )
        for slc in region
    ]
    return tuple(key)


def evaluate_image_snapshot(
    document,
    view_state,
    colormap_lut=None,
    cancellation_token=None,
    *,
    degraded=False,
    shader_display: bool = False,
    provisional_histogram: bool = False,
    stage_cache=None,
    stage_document_key=None,
    evaluation_context=None,
    canonical_orientation: bool = False,
) -> EvaluationResult:
    request = request_for_image(view_state)
    plan = plan_slab(document, request)
    start = perf_counter()
    _check_cancelled(cancellation_token)
    slab = evaluate_slab_from_plan(
        document,
        request,
        plan,
        stage_cache=stage_cache,
        document_key=stage_document_key,
        cancellation_token=cancellation_token,
        evaluation_context=evaluation_context,
    )
    _check_cancelled(cancellation_token)
    if bool(shader_display):
        value = make_shader_image_from_slab(
            slab,
            request,
            colormap_lut=colormap_lut,
            provisional_histogram=bool(provisional_histogram),
            canonical_orientation=canonical_orientation,
        )
    else:
        value = make_image_from_slab(
            slab, request, colormap_lut=colormap_lut, canonical_orientation=canonical_orientation
        )
    _check_cancelled(cancellation_token)
    return EvaluationResult(
        value=value,
        eval_ms=(perf_counter() - start) * 1000.0,
        slab_shape=tuple(np.shape(slab)),
        slab_nbytes=int(getattr(slab, "nbytes", plan.estimated_nbytes or 0)),
        degraded=bool(degraded),
        region_plan=plan.region_plan,
    )


def evaluate_level_evidence_snapshot(
    document,
    view_state,
    source_indices,
    *,
    pixel_limit: int,
    cancellation_token=None,
    stage_cache=None,
    stage_document_key=None,
    evaluation_context=None,
) -> LevelEvidenceBatchResult:
    """Evaluate bounded semantic level evidence without display construction.

    Each source is requested through the normal slab planner.  Raw documents
    therefore read only the sparse image sample, while operation-backed
    documents retain their declared region expansion and can reuse the normal
    stage cache when an operation couples the montage axis.
    """

    montage_axis = getattr(view_state, "montage_axis", None)
    if montage_axis is None:
        raise ValueError("montage_axis must be set for semantic level evidence")
    pixel_limit = max(1, int(pixel_limit))
    sources = tuple(int(source) for source in source_indices)
    start = perf_counter()
    results = []
    for source_index in sources:
        _check_cancelled(cancellation_token)
        tile_state = view_state.tile_state_for_slice(int(montage_axis), int(source_index))
        sample_state = _bounded_level_evidence_state(tile_state, pixel_limit=pixel_limit)
        request = request_for_image(sample_state)
        plan = plan_slab(document, request)
        slab = evaluate_slab_from_plan(
            document,
            request,
            plan,
            stage_cache=stage_cache,
            document_key=stage_document_key,
            cancellation_token=cancellation_token,
            evaluation_context=evaluation_context,
        )
        _check_cancelled(cancellation_token)
        values = _semantic_level_values(np.asarray(slab), sample_state)
        stats = sample_tile_level_stats(values, int(source_index), refined=True)
        results.append(
            LevelEvidenceSourceResult(
                source_index=int(source_index),
                stats=stats,
                sampled_pixels=int(np.asarray(values).size),
                slab_shape=tuple(int(size) for size in np.shape(slab)),
                slab_nbytes=int(getattr(slab, "nbytes", plan.estimated_nbytes or 0)),
                region_plan=plan.region_plan,
            )
        )
    return LevelEvidenceBatchResult(
        sources=tuple(results),
        pixel_limit=pixel_limit,
        elapsed_ms=(perf_counter() - start) * 1000.0,
    )


def evaluate_line_snapshot(
    document,
    view_state,
    *,
    stage_cache=None,
    stage_document_key=None,
    cancellation_token=None,
    evaluation_context=None,
) -> EvaluationResult:
    request = request_for_line(view_state)
    plan = plan_slab(document, request)
    start = perf_counter()
    _check_cancelled(cancellation_token)
    slab = evaluate_slab_from_plan(
        document,
        request,
        plan,
        stage_cache=stage_cache,
        document_key=stage_document_key,
        cancellation_token=cancellation_token,
        evaluation_context=evaluation_context,
    )
    _check_cancelled(cancellation_token)
    value = make_line_from_slab(slab, request)
    return EvaluationResult(
        value=value,
        eval_ms=(perf_counter() - start) * 1000.0,
        slab_shape=tuple(np.shape(slab)),
        slab_nbytes=int(getattr(slab, "nbytes", plan.estimated_nbytes or 0)),
        region_plan=plan.region_plan,
    )


def evaluate_scalar_snapshot(
    document,
    view_state,
    index,
    *,
    stage_cache=None,
    stage_document_key=None,
    cancellation_token=None,
    evaluation_context=None,
) -> EvaluationResult:
    request = request_for_scalar(view_state, index)
    plan = plan_slab(document, request)
    start = perf_counter()
    _check_cancelled(cancellation_token)
    slab = evaluate_slab_from_plan(
        document,
        request,
        plan,
        stage_cache=stage_cache,
        document_key=stage_document_key,
        cancellation_token=cancellation_token,
        evaluation_context=evaluation_context,
    )
    _check_cancelled(cancellation_token)
    value = make_scalar_from_slab(slab, request)
    return EvaluationResult(
        value=value,
        eval_ms=(perf_counter() - start) * 1000.0,
        slab_shape=tuple(np.shape(slab)),
        slab_nbytes=int(getattr(slab, "nbytes", plan.estimated_nbytes or 0)),
        region_plan=plan.region_plan,
    )


def evaluate_export_frame_snapshot(
    document,
    view_state,
    frame_axis,
    frame_index,
    colormap_lut=None,
    *,
    stage_cache=None,
    stage_document_key=None,
    cancellation_token=None,
    evaluation_context=None,
) -> EvaluationResult:
    request = request_for_export_frame(view_state, frame_axis, frame_index)
    plan = plan_slab(document, request)
    start = perf_counter()
    _check_cancelled(cancellation_token)
    slab = evaluate_slab_from_plan(
        document,
        request,
        plan,
        stage_cache=stage_cache,
        document_key=stage_document_key,
        cancellation_token=cancellation_token,
        evaluation_context=evaluation_context,
    )
    _check_cancelled(cancellation_token)
    value = make_image_from_slab(slab, request, colormap_lut=colormap_lut)
    return EvaluationResult(
        value=value,
        eval_ms=(perf_counter() - start) * 1000.0,
        slab_shape=tuple(np.shape(slab)),
        slab_nbytes=int(getattr(slab, "nbytes", plan.estimated_nbytes or 0)),
        region_plan=plan.region_plan,
    )


def _bounded_level_evidence_state(view_state, *, pixel_limit: int):
    """Return a tile state whose image-axis cross product is pixel bounded."""

    image_axes = tuple(int(axis) for axis in (getattr(view_state, "image_axes", None) or ()))
    if len(image_axes) != 2:
        raise ValueError("semantic montage evidence requires exactly two image axes")
    first_axis, second_axis = image_axes
    first_pool = _level_evidence_axis_pool(view_state, first_axis)
    second_pool = _level_evidence_axis_pool(view_state, second_axis)
    first_count, second_count = _bounded_grid_shape(
        len(first_pool),
        len(second_pool),
        limit=max(1, int(pixel_limit)),
    )
    state = view_state.with_axis_range(
        first_axis,
        _even_pool_sample(first_pool, first_count),
        text="semantic-level-evidence",
    )
    return state.with_axis_range(
        second_axis,
        _even_pool_sample(second_pool, second_count),
        text="semantic-level-evidence",
    )


def _level_evidence_axis_pool(view_state, axis: int) -> tuple[int, ...]:
    existing = tuple(getattr(view_state, "axis_range_indices", ()) or ())
    selected = existing[int(axis)] if int(axis) < len(existing) else None
    if selected is not None:
        return tuple(int(index) for index in selected)
    shape = tuple(int(size) for size in getattr(view_state, "shape", ()))
    return tuple(range(int(shape[int(axis)])))


def _bounded_grid_shape(first_size: int, second_size: int, *, limit: int) -> tuple[int, int]:
    first_size = max(1, int(first_size))
    second_size = max(1, int(second_size))
    limit = max(1, int(limit))
    if first_size * second_size <= limit:
        return first_size, second_size
    first_count = max(
        1,
        min(first_size, int(np.sqrt(float(limit) * float(first_size) / float(second_size)))),
    )
    second_count = max(1, min(second_size, limit // first_count))
    while first_count < first_size and (first_count + 1) * second_count <= limit:
        first_count += 1
    while second_count < second_size and first_count * (second_count + 1) <= limit:
        second_count += 1
    return int(first_count), int(second_count)


def _even_pool_sample(pool: tuple[int, ...], count: int) -> tuple[int, ...]:
    if not pool:
        return ()
    count = max(1, min(int(count), len(pool)))
    if count == len(pool):
        return tuple(int(index) for index in pool)
    offsets = np.linspace(0, len(pool) - 1, count, dtype=np.int64)
    return tuple(int(pool[int(offset)]) for offset in offsets)


def _semantic_level_values(slab: np.ndarray, view_state) -> np.ndarray:
    channel = getattr(
        getattr(view_state, "channel", None), "value", getattr(view_state, "channel", "real")
    )
    component = {
        "imag": "imag",
        "angle": "angle",
        "abs": "abs",
        "complex": "abs",
    }.get(str(channel), "real")
    values = extract_component(np.asarray(slab), component)
    scale = getattr(
        getattr(view_state, "scale", None), "value", getattr(view_state, "scale", "linear")
    )
    return np.asarray(apply_shader_scale(values, scale), dtype=np.float32)


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


def _display_image_level_values(display_image) -> np.ndarray:
    level_data = getattr(display_image, "level_data", None)
    if level_data is not None:
        return np.asarray(level_data)
    histogram = getattr(display_image, "histogram_data", None)
    if histogram is not None:
        return np.asarray(histogram)
    mapping = getattr(display_image, "shader_mapping", None)
    semantic = getattr(display_image, "semantic_data", None)
    if mapping is not None and semantic is not None:
        values = extract_component(np.asarray(semantic), getattr(mapping, "component", "real"))
        return apply_shader_scale(
            values,
            getattr(mapping, "scale", "linear"),
            symlog_constant=float(getattr(mapping, "symlog_constant", 0.0) or 0.0),
        )
    data = getattr(display_image, "data", None)
    if data is None:
        return np.asarray((), dtype=np.float32)
    data = np.asarray(data)
    if np.iscomplexobj(data):
        return np.abs(data).astype(np.float32, copy=False)
    return data


def _format_nbytes(nbytes):
    nbytes = int(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024 or unit == "GiB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024


def _estimated_dtype(document):
    dtype = getattr(document.base_data, "dtype", np.dtype(float))
    for operation in document.enabled_operations:
        dtype = operation_output_dtype(dtype, operation)
    return np.dtype(dtype)
