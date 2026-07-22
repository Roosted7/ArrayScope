"""Codec-aware chunk transport (G7): compress host chunk bytes, decode to raw.

The renderer's transport seam is *host chunk bytes -> GPU texture*.  Today a
chunk's ``payload`` ndarray is handed straight to ``write_texture`` (see
``arrayscope.gpu.wgpu_executor._ensure``).  G7 inserts an optional codec at the
*host cache* boundary: a chunk may be held in RAM compressed and decoded back to
its exact ndarray just before upload.  That saves host RAM (more chunks resident
per byte budget) and, if a real GPU-side decoder existed, could save PCIe bytes
too.

This module is the codec abstraction and registry only.  It is deliberately
free of any GPU dependency and is *never on the default path*: the production
transport is byte-for-byte unchanged unless a caller explicitly selects a
non-``raw`` codec (see ``arrayscope.app.settings_state.ChunkTransportCodecChoice``,
which defaults to ``raw``).

Contract:

* ``encode(array) -> bytes`` / ``decode(bytes, shape, dtype) -> ndarray``.
* Lossless by default: ``decode(encode(x)) == x`` exactly for every supported
  dtype.  Lossy modes (zfp tolerance/precision/rate) are opt-in only -- a wrong
  tolerance is silently-wrong pixels, so they are never the default.
* dtype-driven: :func:`resolve_codec` picks a concrete lossless codec for a
  dtype and falls back to ``raw`` when the requested codec cannot represent it
  losslessly, so correctness never depends on codec coverage.
* Optional deps: importing this module never requires ``zfpy`` or ``blosc2``.
  An unavailable codec reports ``available() is False``; ``raw`` always works.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Blosc2Codec",
    "ChunkCodec",
    "CodecError",
    "CompressedChunkCache",
    "RawCodec",
    "ZfpCodec",
    "available_codec_names",
    "get_codec",
    "resolve_codec",
]


class CodecError(RuntimeError):
    """Unknown codec, unavailable dependency, or unsupported dtype."""


@runtime_checkable
class ChunkCodec(Protocol):
    """Encode a chunk ndarray to bytes and decode it back exactly (lossless)."""

    name: str
    lossless: bool

    def available(self) -> bool:
        """Whether the backing dependency is importable on this machine."""

    def supports(self, dtype: np.dtype) -> bool:
        """Whether this codec can round-trip ``dtype`` losslessly."""

    def encode(self, array: np.ndarray) -> bytes: ...

    def decode(self, data: bytes, *, shape: tuple[int, ...], dtype) -> np.ndarray: ...


class RawCodec:
    """Identity codec: contiguous ``tobytes`` / ``frombuffer``.

    This is the default and the reference.  ``encode`` produces exactly the
    bytes ``write_texture`` would have uploaded, so a ``raw`` round-trip through
    the host cache is provably a no-op on the transport payload.
    """

    name = "raw"
    lossless = True

    def available(self) -> bool:
        return True

    def supports(self, dtype: np.dtype) -> bool:
        return True

    def encode(self, array: np.ndarray) -> bytes:
        return np.ascontiguousarray(array).tobytes()

    def decode(self, data: bytes, *, shape: tuple[int, ...], dtype) -> np.ndarray:
        return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)


# zfp (via zfpy) is a floating-point-oriented transform codec.  Its native
# scalar types are float32/float64/int32/int64.  We map the transport dtypes
# onto those losslessly: complex64 as an interleaved float32 pair, int16 as a
# widened int32.  uint8 has no lossless zfp mapping worth the transform, so the
# codec declines it (resolve_codec then keeps such chunks on ``raw``).
_ZFP_NATIVE = frozenset({np.dtype(np.float32), np.dtype(np.float64), np.dtype(np.int32), np.dtype(np.int64)})


class ZfpCodec:
    """ZFP-class codec (zfpy).  Reversible/lossless mode by default.

    ``tolerance``/``precision``/``rate`` mirror zfp's lossy fixed-accuracy,
    fixed-precision and fixed-rate modes.  They default to ``None`` -> zfp's
    reversible mode, which round-trips every supported dtype exactly.  Passing
    any of them opts into lossy compression and flips ``lossless`` to False; a
    caller must do that deliberately.
    """

    name = "zfp"

    def __init__(
        self,
        *,
        tolerance: float | None = None,
        precision: int | None = None,
        rate: float | None = None,
    ) -> None:
        self.tolerance = tolerance
        self.precision = precision
        self.rate = rate
        self.lossless = tolerance is None and precision is None and rate is None

    def available(self) -> bool:
        try:
            import zfpy  # noqa: F401
        except Exception:
            return False
        return True

    def supports(self, dtype: np.dtype) -> bool:
        if not self.available():
            return False
        dtype = np.dtype(dtype)
        if dtype in _ZFP_NATIVE:
            return True
        # Lossless dtype bridges are only valid in reversible mode; a lossy
        # transform on a widened/int-viewed array would not restore the
        # original dtype's values exactly.
        if not self.lossless:
            return dtype in _ZFP_NATIVE
        return dtype in (np.dtype(np.complex64), np.dtype(np.complex128), np.dtype(np.int16))

    def _kwargs(self) -> dict:
        kwargs: dict = {}
        if self.tolerance is not None:
            kwargs["tolerance"] = float(self.tolerance)
        if self.precision is not None:
            kwargs["precision"] = int(self.precision)
        if self.rate is not None:
            kwargs["rate"] = float(self.rate)
        return kwargs

    def encode(self, array: np.ndarray) -> bytes:
        import zfpy

        dtype = np.dtype(array.dtype)
        if not self.supports(dtype):
            raise CodecError(f"zfp cannot losslessly encode dtype {dtype}")
        arr = np.ascontiguousarray(array)
        if dtype in (np.dtype(np.complex64), np.dtype(np.complex128)):
            arr = arr.view(np.float32 if dtype == np.complex64 else np.float64)
        elif dtype == np.dtype(np.int16):
            arr = arr.astype(np.int32)
        return zfpy.compress_numpy(np.ascontiguousarray(arr), **self._kwargs())

    def decode(self, data: bytes, *, shape: tuple[int, ...], dtype) -> np.ndarray:
        import zfpy

        dtype = np.dtype(dtype)
        decoded = zfpy.decompress_numpy(data)
        if dtype in (np.dtype(np.complex64), np.dtype(np.complex128)):
            view = np.complex64 if dtype == np.dtype(np.complex64) else np.complex128
            return np.ascontiguousarray(decoded).view(view).reshape(shape)
        if dtype == np.dtype(np.int16):
            return decoded.astype(np.int16).reshape(shape)
        return decoded.astype(dtype, copy=False).reshape(shape)


class Blosc2Codec:
    """Byte-wise lossless codec (blosc2) -- dtype-agnostic, exact for all dtypes.

    blosc2 compresses the raw contiguous buffer, so every numpy dtype (including
    uint8, which zfp declines) round-trips exactly.  It is a general-purpose
    fallback and a useful benchmark contrast to the zfp transform.
    """

    name = "blosc2"
    lossless = True

    def __init__(self, *, clevel: int = 5) -> None:
        self.clevel = int(clevel)

    def available(self) -> bool:
        try:
            import blosc2  # noqa: F401
        except Exception:
            return False
        return True

    def supports(self, dtype: np.dtype) -> bool:
        return self.available()

    def encode(self, array: np.ndarray) -> bytes:
        import blosc2

        buf = np.ascontiguousarray(array).tobytes()
        return blosc2.compress2(buf, clevel=self.clevel)

    def decode(self, data: bytes, *, shape: tuple[int, ...], dtype) -> np.ndarray:
        import blosc2

        raw = blosc2.decompress2(data)
        return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape)


_BUILTIN_FACTORIES = {
    "raw": RawCodec,
    "zfp": ZfpCodec,
    "blosc2": Blosc2Codec,
}


def get_codec(name: str, **kwargs) -> ChunkCodec:
    """Instantiate a codec by name.  Unknown names raise :class:`CodecError`."""

    try:
        factory = _BUILTIN_FACTORIES[str(name)]
    except KeyError:
        known = ", ".join(sorted(_BUILTIN_FACTORIES))
        raise CodecError(f"unknown chunk codec {name!r}; known codecs: {known}") from None
    return factory(**kwargs)


def available_codec_names() -> tuple[str, ...]:
    """Codec names whose backing dependency is importable (``raw`` always is)."""

    return tuple(name for name, factory in _BUILTIN_FACTORIES.items() if factory().available())


def resolve_codec(name: str, dtype, **kwargs) -> ChunkCodec:
    """Pick a concrete *lossless* codec for ``dtype``.

    Returns the requested codec when it is available and can round-trip
    ``dtype`` exactly; otherwise falls back to :class:`RawCodec`.  This is the
    dtype/error-policy driver: correctness never depends on codec coverage, so a
    dtype the chosen codec cannot represent (e.g. uint8 under zfp) is simply
    stored raw rather than lost.
    """

    codec = get_codec(name, **kwargs)
    if codec.name == "raw":
        return codec
    if not codec.available():
        return RawCodec()
    if not getattr(codec, "lossless", False):
        # Lossy is opt-in and explicit; resolve_codec only ever returns a
        # lossless codec so callers cannot lose pixels by accident.
        return RawCodec()
    if not codec.supports(np.dtype(dtype)):
        return RawCodec()
    return codec


class CompressedChunkCache:
    """Host cache holding chunk payloads compressed, decoded on fetch.

    This is the opt-in host-cache-compression path: ``store`` encodes a chunk's
    ndarray with the dtype-appropriate codec and keeps only the bytes; ``load``
    decodes them back to the exact ndarray the transport would have uploaded.
    With the default ``raw`` codec every entry is stored via the identity codec,
    so the cache is a byte-for-byte pass-through and the production path is
    unchanged.

    The GPU upload path (``wgpu_executor``) is intentionally *not* wired to this
    cache in G7 -- correctness of the resident-texture upload is not risked.
    Callers opt in explicitly; the benchmark exercises this component directly.
    """

    def __init__(self, codec_name: str = "raw", **codec_kwargs) -> None:
        self.codec_name = str(codec_name)
        self._codec_kwargs = codec_kwargs
        self._entries: dict[object, tuple[bytes, tuple[int, ...], np.dtype, str]] = {}

    def is_active(self) -> bool:
        """True when a real (non-identity) codec may be used for some dtype."""

        return self.codec_name != "raw"

    def store(self, key: object, array: np.ndarray) -> int:
        """Encode and retain ``array``; return the stored (compressed) byte count."""

        dtype = np.dtype(array.dtype)
        codec = resolve_codec(self.codec_name, dtype, **self._codec_kwargs)
        data = codec.encode(array)
        self._entries[key] = (data, tuple(array.shape), dtype, codec.name)
        return len(data)

    def load(self, key: object) -> np.ndarray:
        """Decode the exact ndarray previously stored under ``key``."""

        data, shape, dtype, codec_name = self._entries[key]
        codec = get_codec(codec_name, **self._codec_kwargs)
        return codec.decode(data, shape=shape, dtype=dtype)

    def stored_bytes(self, key: object) -> int:
        return len(self._entries[key][0])

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)
