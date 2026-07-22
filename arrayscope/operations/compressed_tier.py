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

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from arrayscope.core.bounded_cache import BoundedCache
from arrayscope.gpu.chunk_codec import get_codec, resolve_codec

__all__ = [
    "CompressedBackingTier",
    "CompressedTierDiagnostics",
    "TwoLevelArrayCache",
]


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
        self._from_array = from_array or (lambda array, meta: array)
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

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: object) -> bool:
        return key in self._cache

    def store(self, key: object, value: object, *, meta: object = None) -> int:
        """Encode ``value``'s array and retain the bytes; return compressed len.

        The codec is chosen per the value's dtype and is always lossless
        (``resolve_codec`` degrades to ``raw`` rather than lose pixels).
        """

        array = np.asarray(self._to_array(value))
        dtype = np.dtype(array.dtype)
        codec = resolve_codec(self.codec_name, dtype, **self._codec_kwargs)
        data = codec.encode(array)
        uncompressed = int(array.nbytes)
        entry = _CompressedEntry(
            data=data,
            shape=tuple(array.shape),
            dtype=dtype,
            codec_name=codec.name,
            uncompressed_bytes=uncompressed,
            meta=meta,
        )
        with self._cache.lock:
            self._cache.put(key, entry, nbytes=len(data))
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
    ) -> TwoLevelArrayCache:
        """Construct a raw ``BoundedArrayCache`` wired to ``tier``'s eviction hook."""

        from arrayscope.operations.cache import BoundedArrayCache

        on_evict = None
        if tier is not None:
            def on_evict(key, value, _nbytes, _tier=tier):
                _tier.store(key, value)

        raw = BoundedArrayCache(raw_max_bytes, raw_max_entries, on_evict=on_evict)
        return cls(raw, tier, promote_on_recover=promote_on_recover)

    @property
    def raw(self):
        return self._raw

    @property
    def tier(self) -> CompressedBackingTier | None:
        return self._tier

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

    def diagnostics(self):
        raw_diag = self._raw.diagnostics()
        tier_diag = self._tier.diagnostics() if self._tier is not None else None
        return raw_diag, tier_diag
