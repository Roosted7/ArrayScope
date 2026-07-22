"""Two-level host cache: a large compressed tier backing a small raw cache (G7 A).

The problem this solves is the *expensive miss*.  ``BoundedArrayCache`` /
``StageCache`` are byte-capacity-bounded: when full they evict, and an evicted
decoded array must be recomputed (an FFT) or re-read (disk) to be seen again.
Under a fixed RAM budget only ``N`` full-fidelity arrays fit, so a working set
larger than ``N`` thrashes -- every revisit past ``N`` pays the recompute.

Phase A adds a second tier under the same RAM budget: when the small raw cache
evicts a value, its bytes are stored *compressed* in a much larger backing tier
(same RAM, ~2.6x the entries because compressed).  A raw-cache miss first checks
the compressed tier and, on a hit, *decodes* (fast, CPU-local) instead of
recomputing/re-reading (slow).  Net: with a fixed RAM budget ~2.6x more of the
working set stays resident, so the expensive-miss rate drops.  A single decode
is slower than a raw memcpy, but it is far cheaper than the recompute/re-read it
replaces -- that is the measured, real win (see ``g7_cache_benchmark``).

Design / seam
-------------
* :class:`CompressedBackingTier` -- a standalone, byte-bounded LRU of *compressed*
  entries, built on the one shared ``BoundedCache`` core (same eviction owner as
  every other cache).  It is dtype-driven and lossless: :func:`resolve_codec`
  picks the codec for each value's dtype and falls back to ``raw`` when a codec
  cannot round-trip that dtype exactly, so a value returned through the tier is
  always bit-identical to the uncompressed path.
* :class:`TwoLevelArrayCache` -- an *optional* wrapper around
  :class:`~arrayscope.operations.cache.BoundedArrayCache`.  It subscribes to the
  raw cache's eviction hook to populate the tier and consults the tier on a raw
  miss.  With no tier (the default) it is a straight pass-through: the raw path
  is byte-identical to today.

Both raw ``BoundedArrayCache`` and ``StageCache`` build on the same
``BoundedCache`` core and could share this tier; the wrapper here targets
``BoundedArrayCache`` (ndarray values) because wiring the tier *into*
``StageCache`` -- with its retention scoring, in-flight-compute claims and
lock-free resident snapshots -- is materially riskier than a standalone tier the
cache delegates to.  The tier is value-generic (``to_array``/``from_array``
adapters) so ``StageCache`` can adopt it later without changing this code.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from arrayscope.core.bounded_cache import BoundedCache
from arrayscope.core.cache_status import CacheDiagnosticsSnapshot, CacheStatus
from arrayscope.gpu.chunk_codec import get_codec, resolve_codec

__all__ = [
    "CompressedBackingTier",
    "CompressedTierDiagnostics",
    "TwoLevelArrayCache",
    "split_payload_for_tier",
]


def _default_from_array(array: np.ndarray, meta: object) -> object:
    """Reconstruct a stored value from its decoded primary array and ``meta``.

    ``meta is None`` is the plain-array path (the tier stored a bare ndarray):
    return it unchanged.  A :class:`_PayloadTemplate` meta rebuilds the original
    payload by dropping the decoded primary array back into its field -- every
    other field (auxiliary arrays and metadata objects alike) is the exact same
    object it was at store time, so the reconstruction is bit-identical.
    """

    if meta is None:
        return array
    if isinstance(meta, _PayloadTemplate):
        return meta.rebuild(array)
    return array


# Sentinel placeholder for the primary array field in a stored payload template.
# It keeps the big primary array from being retained by the template (which
# would defeat the RAM win), and is always overwritten on rebuild.
_PRIMARY_PLACEHOLDER = None


@dataclass(frozen=True)
class _PayloadTemplate:
    """A frozen-dataclass payload with its primary array field nulled out.

    Holds every non-primary field verbatim (auxiliary arrays and metadata); the
    primary array lives compressed in the tier and is restored on ``rebuild``.
    """

    stripped: object
    field: str

    def rebuild(self, array: np.ndarray) -> object:
        return dataclasses.replace(self.stripped, **{self.field: array})


def split_payload_for_tier(value: object):
    """Split a cache value into (primary_array, meta, meta_nbytes) for the tier.

    Returns ``None`` to decline compression -- the caller then lets the value
    evict as it does today (byte-identical), so an unrecognized value type is
    never mishandled.  Lossless by construction:

    * a bare ``np.ndarray`` compresses whole (``meta`` None);
    * a frozen dataclass with at least one ``np.ndarray`` field compresses its
      *largest* array field and carries every other field verbatim in the
      template.  ``meta_nbytes`` accounts for any auxiliary arrays kept verbatim
      so the tier's byte budget stays honest.

    Compressing only the dominant array (not every array) keeps this adapter
    simple and provably correct for heterogeneous payloads; the dominant display
    image is the bulk of the memory, so the RAM win lands where it matters.  A
    fuller multi-array pack is a bounded follow-up.
    """

    if isinstance(value, np.ndarray):
        return value, None, 0
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        array_fields = [
            (f.name, getattr(value, f.name))
            for f in dataclasses.fields(value)
            if isinstance(getattr(value, f.name), np.ndarray)
        ]
        if not array_fields:
            return None
        primary_name, primary = max(array_fields, key=lambda nv: int(nv[1].nbytes))
        aux_nbytes = sum(int(arr.nbytes) for (name, arr) in array_fields if name != primary_name)
        try:
            stripped = dataclasses.replace(value, **{primary_name: _PRIMARY_PLACEHOLDER})
        except Exception:
            # Non-init or otherwise non-replaceable field: decline rather than
            # risk a wrong reconstruction.
            return None
        return primary, _PayloadTemplate(stripped=stripped, field=primary_name), aux_nbytes
    return None


@dataclass(frozen=True)
class CompressedTierDiagnostics:
    entries: int
    compressed_bytes: int
    max_bytes: int
    decode_hits: int
    misses: int
    stores: int
    evictions: int
    # Sum of the uncompressed nbytes currently represented by the tier's
    # compressed entries -- the *working set retained* for the RAM budget spent.
    resident_uncompressed_bytes: int
    mean_ratio: float | None


@dataclass(frozen=True)
class _CompressedEntry:
    data: bytes
    shape: tuple[int, ...]
    dtype: np.dtype
    codec_name: str
    uncompressed_bytes: int
    meta: object = None


class CompressedBackingTier:
    """Byte-bounded LRU of compressed array bytes, decoded on fetch (lossless).

    Keyed identically to the raw cache it backs.  ``store`` encodes a value's
    ndarray with its dtype's lossless codec and keeps only the compressed bytes;
    ``load`` decodes back the exact ndarray.  Its byte budget bounds the
    *compressed* footprint, so for a 2.6x-compressible dtype it retains ~2.6x the
    working set a raw cache of the same byte budget would.
    """

    def __init__(
        self,
        *,
        max_bytes: int,
        codec_name: str = "raw",
        max_entries: int | None = None,
        to_array: Callable[[object], np.ndarray] | None = None,
        from_array: Callable[[np.ndarray, object], object] | None = None,
        codec_kwargs: dict | None = None,
    ) -> None:
        self.codec_name = str(codec_name)
        self._codec_kwargs = dict(codec_kwargs or {})
        self._to_array = to_array or (lambda value: value)
        self._from_array = from_array or _default_from_array
        self._cache = BoundedCache(max_bytes=int(max_bytes), max_entries=max_entries)
        self.stores = 0
        self.decode_hits = 0
        self.misses = 0
        self._resident_uncompressed = 0

    @property
    def max_bytes(self) -> int:
        return int(self._cache.max_bytes)

    @property
    def compressed_bytes(self) -> int:
        return int(self._cache.bytes_used)

    @property
    def evictions(self) -> int:
        return int(self._cache.evictions)

    def is_active(self) -> bool:
        """True when a real (non-identity) codec may be used for some dtype."""

        return self.codec_name != "raw"

    def _resolve_codec_for(self, dtype):
        """Concrete lossless codec for ``dtype`` under this tier's codec name.

        ``"auto"`` is the aggressive per-dtype choice: the best lossless codec
        for the dtype (zfp's transform for numeric, blosc2's byte codec for the
        rest), degrading to ``raw`` when no codec library is available.  A fixed
        name (``zfp``/``blosc2``) is honored via ``resolve_codec`` (which itself
        degrades to ``raw`` rather than lose pixels), so correctness never
        depends on codec coverage.
        """

        if self.codec_name == "auto":
            # Lazy import: keeps this operations-layer module free of an eager
            # gpu-layer dependency and preserves import health.
            from arrayscope.gpu.cache_policy import preferred_codec_for_dtype

            for name in preferred_codec_for_dtype(dtype):
                codec = resolve_codec(name, dtype, **self._codec_kwargs)
                if codec.name == name:
                    return codec
            return resolve_codec("raw", dtype, **self._codec_kwargs)
        return resolve_codec(self.codec_name, dtype, **self._codec_kwargs)

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: object) -> bool:
        return key in self._cache

    def store(
        self, key: object, value: object, *, meta: object = None, meta_nbytes: int = 0
    ) -> int:
        """Encode ``value``'s array and retain the bytes; return compressed len.

        The codec is chosen per the value's dtype and is always lossless
        (``resolve_codec`` degrades to ``raw`` rather than lose pixels).

        ``meta`` is opaque state the tier hands back to ``from_array`` on load
        (e.g. a rebuild template for a payload whose primary array was the one
        compressed).  ``meta_nbytes`` is any *uncompressed* footprint that the
        meta still retains (e.g. a payload's small auxiliary arrays carried
        verbatim); it is added to the tier's byte-budget accounting so the RAM
        budget stays honest and the tier never silently exceeds it.
        """

        array = np.asarray(self._to_array(value))
        dtype = np.dtype(array.dtype)
        codec = self._resolve_codec_for(dtype)
        data = codec.encode(array)
        meta_nbytes = int(max(0, meta_nbytes))
        uncompressed = int(array.nbytes) + meta_nbytes
        entry = _CompressedEntry(
            data=data,
            shape=tuple(array.shape),
            dtype=dtype,
            codec_name=codec.name,
            uncompressed_bytes=uncompressed,
            meta=meta,
        )
        with self._cache.lock:
            self._cache.put(key, entry, nbytes=len(data) + meta_nbytes)
            self.stores += 1
            # Reconcile the resident uncompressed footprint from the survivors:
            # this insert may have evicted older entries under the byte budget.
            self._reconcile_resident()
        return len(data)

    def load(self, key: object) -> object | None:
        """Decode and return the exact value stored under ``key`` (or None)."""

        with self._cache.lock:
            entry = self._cache.get(key)
            if not isinstance(entry, _CompressedEntry):
                self.misses += 1
                return None
            self.decode_hits += 1
        codec = get_codec(entry.codec_name, **self._codec_kwargs)
        array = codec.decode(entry.data, shape=entry.shape, dtype=entry.dtype)
        return self._from_array(array, entry.meta)

    def discard(self, key: object) -> None:
        with self._cache.lock:
            entry = self._cache.discard(key)
            if isinstance(entry, _CompressedEntry):
                self._resident_uncompressed -= entry.uncompressed_bytes

    def clear(self) -> None:
        with self._cache.lock:
            self._cache.clear()
            self._resident_uncompressed = 0

    def resize(self, *, max_bytes: int | None = None, max_entries: int | None = None) -> None:
        with self._cache.lock:
            self._cache.resize(max_bytes=max_bytes, max_entries=max_entries)
            self._reconcile_resident()

    def stored_bytes(self, key: object) -> int:
        entry = self._cache.peek(key)
        return len(entry.data) if isinstance(entry, _CompressedEntry) else 0

    def _reconcile_resident(self) -> None:
        # Recompute the resident uncompressed footprint from the surviving
        # entries.  Cheap (entry count is small: it is bounded by the tier's
        # byte budget / mean compressed size) and robust against eviction.
        # BoundedCache.items() yields (key, value) tuples (not a dict view).
        total = 0
        for _key, entry in self._cache.items():  # noqa: PERF102 - not a dict
            if isinstance(entry, _CompressedEntry):
                total += entry.uncompressed_bytes
        self._resident_uncompressed = total

    def diagnostics(self) -> CompressedTierDiagnostics:
        with self._cache.lock:
            ratio = None
            if self._cache.bytes_used > 0 and self._resident_uncompressed > 0:
                ratio = float(self._resident_uncompressed) / float(self._cache.bytes_used)
            return CompressedTierDiagnostics(
                entries=len(self._cache),
                compressed_bytes=int(self._cache.bytes_used),
                max_bytes=int(self._cache.max_bytes),
                decode_hits=int(self.decode_hits),
                misses=int(self.misses),
                stores=int(self.stores),
                evictions=int(self._cache.evictions),
                resident_uncompressed_bytes=int(self._resident_uncompressed),
                mean_ratio=ratio,
            )


class TwoLevelArrayCache:
    """A small raw cache backed by a large compressed tier (opt-in).

    ``get``/``put``/``get_or_compute`` mirror :class:`BoundedArrayCache`.  When
    ``tier`` is None this is a straight pass-through to the raw cache -- the
    default, byte-identical path.  With a tier:

    * ``put`` stores into the raw cache; the raw cache's eviction hook pushes
      each evicted value into the tier (compressed).
    * ``get`` returns a raw hit directly; on a raw miss it decodes from the tier
      (an *avoided* recompute) and promotes the value back into the raw cache.
    * ``get_or_compute`` only calls ``compute`` on a true miss (absent from both
      tiers) -- the tier hit spares the expensive recompute/re-read.

    ``recomputes`` counts true misses (``compute`` was called); ``tier_recoveries``
    counts raw-misses served by a decode instead.  Their ratio is the win.
    """

    def __init__(
        self,
        raw_cache,
        tier: CompressedBackingTier | None = None,
        *,
        promote_on_recover: bool = True,
    ) -> None:
        self._raw = raw_cache
        self._tier = tier
        self._promote = bool(promote_on_recover)
        self.recomputes = 0
        self.tier_recoveries = 0
        self.raw_hits = 0
        if tier is not None:
            # Subscribe the tier to the raw cache's eviction hook so evicted
            # values are compressed instead of dropped.  Requires a raw cache
            # constructed with this hook wired to us (see ``build``).
            pass

    @classmethod
    def build(
        cls,
        *,
        raw_max_bytes: int,
        raw_max_entries: int,
        tier: CompressedBackingTier | None = None,
        promote_on_recover: bool = True,
        store_adapter: Callable[[object], tuple | None] | None = None,
    ) -> TwoLevelArrayCache:
        """Construct a raw ``BoundedArrayCache`` wired to ``tier``'s eviction hook.

        ``store_adapter(value) -> (array, meta, meta_nbytes) | None`` splits an
        evicted value into the array to compress plus opaque rebuild state; when
        it returns None (an unrecognized value type) the value is not stored in
        the tier and simply evicts as it does today.  The default stores the
        whole value (identity) -- unchanged from the plain ndarray path.
        """

        from arrayscope.operations.cache import BoundedArrayCache

        adapter = store_adapter or (lambda value: (value, None, 0))
        on_evict = None
        if tier is not None:

            def on_evict(key, value, _nbytes, _tier=tier, _adapter=adapter):
                parts = _adapter(value)
                if parts is None:
                    return
                array, meta, meta_nbytes = parts
                _tier.store(key, array, meta=meta, meta_nbytes=meta_nbytes)

        raw = BoundedArrayCache(raw_max_bytes, raw_max_entries, on_evict=on_evict)
        return cls(raw, tier, promote_on_recover=promote_on_recover)

    @property
    def raw(self):
        return self._raw

    @property
    def tier(self) -> CompressedBackingTier | None:
        return self._tier

    # --- BoundedArrayCache drop-in surface (proxied to the raw cache) ---------

    @property
    def last_eval_ms(self):
        return self._raw.last_eval_ms

    @last_eval_ms.setter
    def last_eval_ms(self, value):
        self._raw.last_eval_ms = value

    @property
    def bytes_used(self) -> int:
        return int(self._raw.bytes_used)

    @property
    def max_bytes(self) -> int:
        return int(self._raw.max_bytes)

    @property
    def max_entries(self) -> int:
        return int(self._raw.max_entries)

    def __len__(self) -> int:
        return len(self._raw)

    def clear(self) -> None:
        self._raw.clear()
        if self._tier is not None:
            self._tier.clear()

    def resize(self, *, max_bytes: int | None = None, max_entries: int | None = None) -> None:
        self._raw.resize(max_bytes=max_bytes, max_entries=max_entries)
        # Keep the compressed tier's budget locked to the raw budget so a memory
        # policy change scales both tiers together (the tier is additional RAM
        # equal to the raw budget; see the evaluator's tier sizing).
        if self._tier is not None and max_bytes is not None:
            self._tier.resize(max_bytes=int(max_bytes))

    def get(self, key):
        value = self._raw.get(key)
        if value is not None:
            self.raw_hits += 1
            return value
        if self._tier is None:
            return None
        recovered = self._tier.load(key)
        if recovered is None:
            return None
        self.tier_recoveries += 1
        if self._promote:
            # Promote back into the raw cache; a resulting eviction re-enters the
            # tier via the hook, so no working-set truth is lost.
            self._raw.put(key, recovered)
        return recovered

    def put(self, key, value):
        return self._raw.put(key, value)

    def get_or_compute(self, key, compute):
        cached = self.get(key)
        if cached is not None:
            return cached, True
        value = compute()
        self.recomputes += 1
        self.put(key, value)
        return value, False

    def diagnostics(
        self, status=CacheStatus.READY, message="", **extra
    ) -> CacheDiagnosticsSnapshot:
        """Drop-in for ``BoundedArrayCache.diagnostics``, with tier fields folded in.

        Returns the raw cache's snapshot (byte-identical to a plain cache when the
        tier is off) enriched with the compressed-tier counters, so the
        diagnostics dialog shows the tier is engaged and how often a decode
        replaced a recompute (``tier_recoveries``).
        """

        base = self._raw.diagnostics(status, message, **extra)
        if self._tier is None:
            return base
        tier_diag = self._tier.diagnostics()
        return dataclasses.replace(
            base,
            tier_engaged=True,
            tier_codec=str(self._tier.codec_name),
            tier_entries=int(tier_diag.entries),
            tier_compressed_bytes=int(tier_diag.compressed_bytes),
            tier_resident_uncompressed_bytes=int(tier_diag.resident_uncompressed_bytes),
            tier_recoveries=int(self.tier_recoveries),
            tier_decode_hits=int(tier_diag.decode_hits),
            tier_stores=int(tier_diag.stores),
            tier_evictions=int(tier_diag.evictions),
        )
