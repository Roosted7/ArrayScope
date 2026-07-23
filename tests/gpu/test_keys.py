"""Identity laws for DataChunkKey/ViewTileKey/ChunkLod (ADR 0055 G1):
value-equal, hashable, numerically normalized keys — the vocabulary the
GPU residency engine stands on."""

import pytest

from arrayscope.gpu import ChunkLod, DataChunkKey, ViewTileKey
from arrayscope.gpu.keys import COMPLEX_RG32F, SCALAR_R32F


def make_key(**overrides):
    kwargs = {
        "document_generation": ("doc", 1),
        "operation_key": ("op", "fft"),
        "lod": ChunkLod(),
        "chunk_origin": (0, 128),
        "chunk_shape": (128, 128),
        "dtype": "float32",
        "representation": SCALAR_R32F,
    }
    kwargs.update(overrides)
    return DataChunkKey(**kwargs)


def test_chunk_key_is_hashable_and_value_equal():
    assert make_key() == make_key()
    assert hash(make_key()) == hash(make_key())
    assert make_key() != make_key(chunk_origin=(128, 128))
    assert len({make_key(), make_key(), make_key(dtype="float64")}) == 2


def test_chunk_key_normalizes_numeric_fields():
    key = make_key(chunk_origin=(0.0, 128.0), chunk_shape=(128.0, 64.0), lod=(1, 2, 0))
    assert key.chunk_origin == (0, 128)
    assert key.chunk_shape == (128, 64)
    assert key.lod == ChunkLod(level=1, factor=2)
    assert key.stop == (128, 192)
    assert key.rank == 2


def test_chunk_key_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        make_key(chunk_origin=(0,), chunk_shape=(128, 128))
    with pytest.raises(ValueError):
        make_key(chunk_origin=(-1, 0))
    with pytest.raises(ValueError):
        make_key(chunk_shape=(0, 128))


def test_chunk_key_rejects_unknown_representation():
    with pytest.raises(ValueError):
        make_key(representation="bc7")
    assert make_key(representation=COMPLEX_RG32F).representation == COMPLEX_RG32F


def test_chunk_key_hashes_large_semantic_identity_only_once():
    class CountedIdentity:
        def __init__(self):
            self.calls = 0

        def __hash__(self):
            self.calls += 1
            return 41

    identity = CountedIdentity()
    key = make_key(document_generation=identity)
    construction_calls = identity.calls

    for _ in range(100):
        hash(key)

    assert construction_calls == 1
    assert identity.calls == construction_calls


def test_chunk_lod_clamps_like_tile_lod_identity():
    lod = ChunkLod(level=-3, factor=0, gutter=-1)
    assert (lod.level, lod.factor, lod.gutter) == (0, 1, 0)


def test_view_tile_key_is_presentation_scoped():
    a = ViewTileKey(presentation_key=("frame", 7), tile_number=3)
    b = ViewTileKey(presentation_key=("frame", 7), tile_number="3")
    assert a == b
    assert a != ViewTileKey(presentation_key=("frame", 8), tile_number=3)


def test_chunk_lod_anisotropic_reduction_identity():
    from arrayscope.gpu.keys import REDUCER_MEAN_ABS, REDUCER_NATIVE

    native = ChunkLod()
    assert native.is_native
    assert native.reduction == ()
    assert native.reducer == REDUCER_NATIVE
    aniso = ChunkLod(reduction=(4, 1), reducer=REDUCER_MEAN_ABS)
    assert aniso.axis_scale(0) == 16
    assert aniso.axis_scale(1) == 2
    assert aniso.axis_scale(2) == 1  # axes beyond the vector are native
    assert not aniso.is_native
    # Reduction vector and reducer are identity: same geometry, different
    # reducer, different chunk.
    a = make_key(lod=ChunkLod(reduction=(1, 1), reducer="mean"))
    b = make_key(lod=ChunkLod(reduction=(1, 1), reducer="mean_abs"))
    assert a != b
    assert make_key(lod=ChunkLod(reduction=(1, 1), reducer="mean")) == a


def test_chunk_lod_rejects_inconsistent_reduction():
    with pytest.raises(ValueError):
        ChunkLod(reduction=(-1,), reducer="mean")
    with pytest.raises(ValueError):
        ChunkLod(reduction=(1,), reducer="native")
    with pytest.raises(ValueError):
        ChunkLod(reduction=(1,), reducer="bicubic-guess")
