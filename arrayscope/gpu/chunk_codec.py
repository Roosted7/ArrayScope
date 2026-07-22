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
* GPU-decodability (G7 Phase B): the ``bitpack`` codec's decode is embarrassingly
  parallel (a per-sample bit-window extract), so it *could* be decoded in a
  compute shader off the CPU critical path.  It is retained here as the
  **lossless** narrow-integer fallback.  The headline Phase-B transfer win uses
  native block-compressed textures (BC4/BC5) instead -- see
  :mod:`arrayscope.gpu.bc_codec`: those are decoded for free by the hardware
  texture sampler (no decode pass at all) and stay compressed in VRAM, at the
  cost of being lossy (acceptable on the display path, and measured).  A codec
  advertises the parallel-decode property with the ``gpu_decodable`` flag.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "BitpackCodec",
    "BitpackPlan",
    "Blosc2Codec",
    "ChunkCodec",
    "CodecError",
    "CompressedChunkCache",
    "RawCodec",
    "ZfpCodec",
    "available_codec_names",
    "get_codec",
    "gpu_decodable_codec_names",
    "resolve_codec",
]


class CodecError(RuntimeError):
    """Unknown codec, unavailable dependency, or unsupported dtype."""


@runtime_checkable
class ChunkCodec(Protocol):
    """Encode a chunk ndarray to bytes and decode it back exactly (lossless)."""

    name: str
    lossless: bool
    gpu_decodable: bool

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
    gpu_decodable = False

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
_ZFP_NATIVE = frozenset(
    {np.dtype(np.float32), np.dtype(np.float64), np.dtype(np.int32), np.dtype(np.int64)}
)


class ZfpCodec:
    """ZFP-class codec (zfpy).  Reversible/lossless mode by default.

    ``tolerance``/``precision``/``rate`` mirror zfp's lossy fixed-accuracy,
    fixed-precision and fixed-rate modes.  They default to ``None`` -> zfp's
    reversible mode, which round-trips every supported dtype exactly.  Passing
    any of them opts into lossy compression and flips ``lossless`` to False; a
    caller must do that deliberately.
    """

    name = "zfp"
    gpu_decodable = False

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
    gpu_decodable = False

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


# ---------------------------------------------------------------------------
# bitpack -- the GPU-decodable narrow-integer codec (G7 Phase B)
# ---------------------------------------------------------------------------
#
# Narrow-integer bit-packing: for an integer chunk whose values span a range
# that fits in K <= 32 bits, subtract the per-chunk minimum (the "offset") and
# store each sample in exactly K bits.  Decode subtracts nothing on the GPU side
# beyond the offset add -- an unpack of K bits per sample -- which is
# embarrassingly parallel: invocation ``i`` reads its own bit window and writes
# one output sample, no cross-lane dependency -- so it could run in a WGSL compute
# shader.  Retained as the lossless fallback; the lossy BC-texture path
# (:mod:`arrayscope.gpu.bc_codec`) is the headline Phase-B transfer/VRAM win.
#
# Losslessness: exact for any integer dtype whose *actual* value range fits K
# bits.  When a chunk needs its dtype's full width (K would not shrink it) or a
# non-integer dtype is handed in, the codec DECLINES to pack and stores the raw
# contiguous bytes in a ``mode=raw`` blob -- never dropping a bit.  ``plan()``
# exposes the pack decision so the GPU path and the policy can tell a genuinely
# GPU-unpackable (mode=packed) blob from a declined (mode=raw) one.
#
# Supported dtypes: signed/unsigned 8/16/32/64-bit integers.  (int64/uint64 are
# accepted only when the chunk's value *range* still fits 32 bits -- MRI/quantized
# data routinely does.)  Floats are out of scope here -- that is the future-work
# float GPU codec; a float dtype is declined to ``mode=raw``.

_BITPACK_VERSION = 1
_BITPACK_MODE_PACKED = 0
_BITPACK_MODE_RAW = 1
_BITPACK_HEADER = np.dtype(
    [
        ("version", "<u1"),
        ("mode", "<u1"),
        ("bits", "<u1"),
        ("reserved", "<u1"),
        ("n_samples", "<u4"),
        ("offset", "<i8"),
    ]
)
assert _BITPACK_HEADER.itemsize == 16
_BITPACK_MAX_BITS = 32
_BITPACK_INT_KINDS = frozenset("iu")  # numpy integer kind codes


class BitpackPlan:
    """The pack decision for one chunk: bit-width ``K`` and value ``offset``.

    ``bits`` is the number of bits each sample occupies; ``offset`` is the value
    subtracted from every sample before packing (added back on decode).  A plan
    exists only when packing is lossless AND strictly narrower than the dtype;
    :meth:`BitpackCodec.plan` returns ``None`` otherwise (declined -> raw).
    """

    __slots__ = ("bits", "dtype", "n_samples", "offset")

    def __init__(self, *, bits: int, offset: int, n_samples: int, dtype: np.dtype) -> None:
        self.bits = int(bits)
        self.offset = int(offset)
        self.n_samples = int(n_samples)
        self.dtype = np.dtype(dtype)


def _bitpack_words(values_u32: np.ndarray, bits: int, n: int) -> np.ndarray:
    """Pack ``n`` non-negative values (< 2**bits) LSB-first into u32 words.

    The stream places sample ``i`` at bit range ``[i*bits, i*bits+bits)`` where
    bit ``b`` is bit ``b % 32`` of little-endian word ``b // 32``.  A sample
    spans at most two words (bits <= 32), so each value contributes a low part to
    word ``w0`` and a high part to ``w0+1`` (which is exactly 0 when the sample
    fits in one word).  Contributions are bit-disjoint, so integer ``add.at``
    equals a bitwise OR.  One guard word is appended so the GPU shader may always
    read ``words[w0+1]`` in bounds.
    """

    if n == 0:
        return np.zeros(1, dtype=np.uint32)
    bitpos = np.arange(n, dtype=np.uint64) * np.uint64(bits)
    w0 = (bitpos >> np.uint64(5)).astype(np.int64)
    s0 = (bitpos & np.uint64(31)).astype(np.uint64)
    v64 = values_u32.astype(np.uint64)
    n_words = int((n * bits + 31) // 32) + 1  # +1 guard word for the high write
    words = np.zeros(n_words, dtype=np.uint32)
    low = ((v64 << s0) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    # v64 < 2**bits <= 2**(32-s0) when the sample fits one word, so >> (32-s0) is
    # naturally 0 there; np.uint64 shift-by-32 is well defined (unlike C/WGSL).
    high = (v64 >> (np.uint64(32) - s0)).astype(np.uint32)
    np.add.at(words, w0, low)
    np.add.at(words, w0 + 1, high)
    return words


def _bitunpack_words(words: np.ndarray, bits: int, n: int) -> np.ndarray:
    """Inverse of :func:`_bitpack_words`: extract ``n`` values of ``bits`` bits."""

    if n == 0:
        return np.zeros(0, dtype=np.uint32)
    words = np.ascontiguousarray(words, dtype=np.uint32)
    bitpos = np.arange(n, dtype=np.uint64) * np.uint64(bits)
    w0 = (bitpos >> np.uint64(5)).astype(np.int64)
    s0 = (bitpos & np.uint64(31)).astype(np.uint64)
    lo = words[w0].astype(np.uint64) >> s0
    hi = words[w0 + 1].astype(np.uint64) << (np.uint64(32) - s0)
    mask = np.uint64((1 << bits) - 1)
    return ((lo | hi) & mask).astype(np.uint32)


class BitpackCodec:
    """Lossless narrow-integer codec (bit-packing); the GPU-decodable fallback.

    Encodes integer chunks by subtracting the per-chunk minimum and packing each
    sample into the minimum number of bits its range needs.  Its decode is a
    per-sample bit-window extract with no cross-lane dependency, so it *could* be
    unpacked in a compute shader off the CPU critical path.  It is retained as the
    **lossless** Phase-B fallback -- the headline lossy transfer/VRAM win uses
    native BC textures (:mod:`arrayscope.gpu.bc_codec`).  Declines (stores raw
    bytes) for floats or for integer chunks that need their dtype's full width, so
    it never loses a bit.
    """

    name = "bitpack"
    lossless = True
    gpu_decodable = True

    def available(self) -> bool:
        return True

    def supports(self, dtype: np.dtype) -> bool:
        # dtype-level capability: integer dtypes.  Whether a *given chunk* is
        # actually packable (vs declined to raw) depends on its value range and
        # is decided per-chunk in plan()/encode(); encode() never raises and is
        # always lossless regardless.
        return np.dtype(dtype).kind in _BITPACK_INT_KINDS

    def plan(self, array: np.ndarray) -> BitpackPlan | None:
        """Return the pack plan for ``array``, or ``None`` if it must stay raw.

        ``None`` means: not an integer dtype, empty, or the value range needs the
        dtype's full width (>= dtype bits, or > 32) -- i.e. packing cannot both be
        lossless and shrink it.  A returned plan is guaranteed lossless.
        """

        dtype = np.dtype(array.dtype)
        if dtype.kind not in _BITPACK_INT_KINDS:
            return None
        n = int(array.size)
        if n == 0:
            return None
        wide = array.astype(np.int64, copy=False)
        lo = int(wide.min())
        hi = int(wide.max())
        span = hi - lo  # values map to [0, span]
        bits = int(span).bit_length() if span > 0 else 1
        dtype_bits = dtype.itemsize * 8
        if bits > _BITPACK_MAX_BITS or bits >= dtype_bits:
            # Needs full width (or unrepresentable in <=32 bits): decline, don't
            # pack -- a pointless pack, or one that could not stay lossless.
            return None
        return BitpackPlan(bits=bits, offset=lo, n_samples=n, dtype=dtype)

    def encode(self, array: np.ndarray) -> bytes:
        arr = np.ascontiguousarray(array)
        plan = self.plan(arr)
        header = np.zeros(1, dtype=_BITPACK_HEADER)
        header["version"] = _BITPACK_VERSION
        if plan is None:
            header["mode"] = _BITPACK_MODE_RAW
            header["bits"] = 0
            header["n_samples"] = int(arr.size)
            header["offset"] = 0
            return header.tobytes() + arr.tobytes()
        values = (arr.astype(np.int64).ravel() - np.int64(plan.offset)).astype(np.uint32)
        words = _bitpack_words(values, plan.bits, plan.n_samples)
        header["mode"] = _BITPACK_MODE_PACKED
        header["bits"] = plan.bits
        header["n_samples"] = plan.n_samples
        header["offset"] = plan.offset
        return header.tobytes() + words.tobytes()

    def decode(self, data: bytes, *, shape: tuple[int, ...], dtype) -> np.ndarray:
        dtype = np.dtype(dtype)
        header = np.frombuffer(data, dtype=_BITPACK_HEADER, count=1)[0]
        payload = memoryview(data)[_BITPACK_HEADER.itemsize :]
        n = int(header["n_samples"])
        if int(header["mode"]) == _BITPACK_MODE_RAW:
            return np.frombuffer(payload, dtype=dtype).reshape(shape)
        bits = int(header["bits"])
        offset = int(header["offset"])
        words = np.frombuffer(payload, dtype=np.uint32)
        values = _bitunpack_words(words, bits, n).astype(np.int64) + np.int64(offset)
        return values.astype(dtype).reshape(shape)

    # -- GPU-path helpers -----------------------------------------------------

    @staticmethod
    def is_packed(data: bytes) -> bool:
        """Whether ``data`` is a genuinely bit-packed (GPU-unpackable) blob."""

        header = np.frombuffer(data, dtype=_BITPACK_HEADER, count=1)[0]
        return int(header["mode"]) == _BITPACK_MODE_PACKED

    @staticmethod
    def read_header(data: bytes) -> tuple[int, int, int, int]:
        """Return ``(mode, bits, n_samples, offset)`` from a bitpack blob."""

        h = np.frombuffer(data, dtype=_BITPACK_HEADER, count=1)[0]
        return int(h["mode"]), int(h["bits"]), int(h["n_samples"]), int(h["offset"])

    @staticmethod
    def payload_words(data: bytes) -> np.ndarray:
        """Return the packed u32 word array (the bytes uploaded to the GPU)."""

        return np.frombuffer(memoryview(data)[_BITPACK_HEADER.itemsize :], dtype=np.uint32)

    HEADER_BYTES = _BITPACK_HEADER.itemsize


_BUILTIN_FACTORIES = {
    "raw": RawCodec,
    "zfp": ZfpCodec,
    "blosc2": Blosc2Codec,
    "bitpack": BitpackCodec,
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


def gpu_decodable_codec_names() -> tuple[str, ...]:
    """Codec names whose decode can run on the GPU (``bitpack``), if available."""

    return tuple(
        name
        for name, factory in _BUILTIN_FACTORIES.items()
        if getattr(factory(), "gpu_decodable", False) and factory().available()
    )


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
