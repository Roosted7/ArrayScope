"""G7 chunk-transport codec: lossless exactness, registry, default-off proof."""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from arrayscope.app.settings_state import (
    ChunkTransportCodecChoice,
    settings_from_mapping,
)
from arrayscope.gpu import chunk_codec
from arrayscope.gpu.chunk_codec import (
    Blosc2Codec,
    CodecError,
    CompressedChunkCache,
    RawCodec,
    ZfpCodec,
    available_codec_names,
    get_codec,
    resolve_codec,
)

_SUPPORTED_DTYPES = [np.float32, np.complex64, np.int16, np.uint8]


def _sample(dtype: np.dtype, shape=(64, 64)) -> np.ndarray:
    dtype = np.dtype(dtype)
    rng = np.random.default_rng(1234)
    if dtype == np.dtype(np.complex64):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    if dtype == np.dtype(np.float32):
        return (rng.standard_normal(shape) * 1000.0).astype(np.float32)
    if dtype == np.dtype(np.int16):
        return rng.integers(-30000, 30000, shape, dtype=np.int16)
    if dtype == np.dtype(np.uint8):
        return rng.integers(0, 256, shape, dtype=np.uint8)
    raise AssertionError(dtype)


# --- raw identity -----------------------------------------------------------


@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_raw_is_exact_identity(dtype):
    codec = RawCodec()
    x = _sample(dtype)
    data = codec.encode(x)
    # The identity codec's bytes are exactly the transport payload.
    assert data == np.ascontiguousarray(x).tobytes()
    back = codec.decode(data, shape=x.shape, dtype=x.dtype)
    assert np.array_equal(back, x)
    assert back.dtype == x.dtype


# --- zfp lossless exactness -------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.complex64, np.int16])
def test_zfp_lossless_roundtrip_exact(dtype):
    codec = ZfpCodec()
    if not codec.available():
        pytest.skip("zfpy not installed")
    assert codec.lossless is True
    x = _sample(dtype)
    data = codec.encode(x)
    back = codec.decode(data, shape=x.shape, dtype=x.dtype)
    assert back.dtype == np.dtype(dtype)
    assert np.array_equal(back, x)


def test_zfp_declines_uint8_but_supports_reports_false():
    codec = ZfpCodec()
    if not codec.available():
        pytest.skip("zfpy not installed")
    assert codec.supports(np.dtype(np.uint8)) is False
    with pytest.raises(CodecError):
        codec.encode(_sample(np.uint8))


def test_zfp_lossy_is_opt_in_only():
    lossy = ZfpCodec(tolerance=1e-2)
    assert lossy.lossless is False
    # resolve_codec must never hand back a lossy codec (silent-wrong-pixels
    # guard): a lossy request degrades to raw.
    resolved = resolve_codec("zfp", np.float32, tolerance=1e-2)
    assert isinstance(resolved, RawCodec)


# --- blosc2 lossless exactness ---------------------------------------------


@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
def test_blosc2_lossless_roundtrip_exact(dtype):
    codec = Blosc2Codec()
    if not codec.available():
        pytest.skip("blosc2 not installed")
    x = _sample(dtype)
    data = codec.encode(x)
    back = codec.decode(data, shape=x.shape, dtype=x.dtype)
    assert back.dtype == np.dtype(dtype)
    assert np.array_equal(back, x)


# --- registry ---------------------------------------------------------------


def test_registry_unknown_codec_errors_clearly():
    with pytest.raises(CodecError) as excinfo:
        get_codec("does-not-exist")
    assert "unknown chunk codec" in str(excinfo.value)


def test_raw_always_available():
    assert "raw" in available_codec_names()


def test_resolve_codec_falls_back_to_raw_for_unsupported_dtype():
    # zfp cannot losslessly represent uint8 -> resolve keeps it raw.
    resolved = resolve_codec("zfp", np.uint8)
    assert isinstance(resolved, RawCodec)


# --- default-off proof ------------------------------------------------------


def test_default_setting_is_auto_and_raw_is_byte_identical():
    # G7 host-cache AUTO: the default flipped to AUTO (aggressive dogfood) so the
    # compressed tier is actually exercised in the live app.  Explicit RAW stays
    # the byte-for-byte pass-through reference.
    settings = settings_from_mapping({})
    assert settings.chunk_transport_codec is ChunkTransportCodecChoice.AUTO
    # The explicit RAW value is a byte-for-byte pass-through: for every dtype the
    # stored bytes equal the raw transport payload and the decoded chunk is exact.
    cache = CompressedChunkCache(ChunkTransportCodecChoice.RAW.value)
    assert cache.is_active() is False
    for dtype in _SUPPORTED_DTYPES:
        x = _sample(dtype)
        stored = cache.store(("k", dtype), x)
        assert stored == np.ascontiguousarray(x).tobytes().__len__()
        back = cache.load(("k", dtype))
        assert np.array_equal(back, x)
        assert back.dtype == x.dtype


@pytest.mark.parametrize("codec_name", ["raw", "zfp", "blosc2"])
def test_compressed_cache_roundtrip_matches_raw(codec_name):
    if codec_name != "raw" and not get_codec(codec_name).available():
        pytest.skip(f"{codec_name} not installed")
    cache = CompressedChunkCache(codec_name)
    for dtype in _SUPPORTED_DTYPES:
        x = _sample(dtype)
        cache.store((codec_name, dtype), x)
        back = cache.load((codec_name, dtype))
        # A codec round-trip through the host cache yields chunks identical to
        # the raw path -- this is the correctness pin for the transport seam.
        assert np.array_equal(back, x)
        assert back.dtype == x.dtype


# --- optional-dependency health --------------------------------------------


def test_zfp_absent_leaves_raw_working(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "zfpy":
            raise ImportError("simulated missing zfpy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    codec = ZfpCodec()
    assert codec.available() is False
    # With zfp unavailable, resolve degrades to raw and raw still round-trips.
    resolved = resolve_codec("zfp", np.float32)
    assert isinstance(resolved, RawCodec)
    x = _sample(np.float32)
    assert np.array_equal(resolved.decode(resolved.encode(x), shape=x.shape, dtype=x.dtype), x)


def test_module_imports_without_optional_codecs(monkeypatch):
    # Importing the codec module must never require zfpy/blosc2 (import health).
    import importlib

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("zfpy", "blosc2"):
            raise ImportError(f"simulated missing {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    importlib.reload(chunk_codec)
    assert "raw" in chunk_codec.available_codec_names()
    importlib.reload(chunk_codec)  # restore real module state
