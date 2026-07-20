import numpy as np

from arrayscope.display.model.tile_identity import (
    TileIdentity,
    TileLodIdentity,
    TilePresentationIdentity,
    array_plane_identities,
)
from arrayscope.display.shader_mapping import TexturePlaneKind
from arrayscope.presentation import TileLifecycle, TilePayloadRef, TileTarget


def _identity(
    *,
    mapping=("phase_color", "abs", "mapped"),
    kind=TexturePlaneKind.COMPLEX_RG32F,
    level=0,
    quality="exact",
):
    values = np.asarray([[1 + 2j]], dtype=np.complex64)
    real, imag = array_plane_identities(values)
    return TileIdentity(
        document_generation=("document", 3),
        operation_key=("fft",),
        source_index=7,
        image_axes=(0, 1),
        axis_flips=(False, True),
        channel="complex",
        complex_mapping=mapping,
        texture_kind=kind,
        semantic_generation=11,
        lod=TileLodIdentity(level=level, factor=2**level),
        quality=quality,
        real_plane=real,
        imag_plane=imag,
    )


def _ref(identity):
    return TilePayloadRef(
        source_id=("legacy-residency-key", 7),
        quality=identity.quality,
        lod_level=identity.lod.level,
        source_index=identity.source_index,
        texture_kind=identity.texture_kind.value,
        shader_mapping_key=identity.complex_mapping,
        identity=identity,
        payload=identity,
    )


def test_exact_acknowledgement_rejects_mixed_complex_mapping_and_texture_kind():
    target_identity = _identity()
    lifecycle = TileLifecycle()
    lifecycle.retarget({0: TileTarget(0, 7, ("semantic", 7), identity=target_identity)})
    target = _ref(target_identity)
    lifecycle.target_ready(0, target)
    lifecycle.commit_emitted({0: target})

    wrong_mapping = _ref(_identity(mapping=("scalar", "real", "mapped")))
    wrong_kind = _ref(_identity(kind=TexturePlaneKind.SCALAR_R32F))

    assert lifecycle.backend_ack({0: wrong_mapping}) == ()
    assert lifecycle.backend_ack({0: wrong_kind}) == ()
    assert lifecycle.backend_ack({0: target}) == (0,)


def test_only_same_semantic_coarser_lod_is_a_compatible_fallback():
    target = _identity(level=0)
    fallback = _identity(level=2, quality="fallback")
    incompatible = _identity(level=2, quality="fallback", mapping=("scalar", "imag", "mapped"))

    assert fallback.compatible_fallback_for(target)
    assert not incompatible.compatible_fallback_for(target)


def test_fallback_compatibility_does_not_recurse_through_full_identity_equality(monkeypatch):
    target = _identity(level=0)
    fallback = _identity(level=2, quality="fallback")
    monkeypatch.setattr(
        TileIdentity,
        "__eq__",
        lambda _self, _other: (_ for _ in ()).throw(
            AssertionError("full TileIdentity equality entered")
        ),
    )

    assert fallback.compatible_fallback_for(target)


def test_presentation_identity_is_separate_from_pixel_identity():
    identity = _identity()
    first = TilePresentationIdentity(4, levels=(0.0, 2.0), scale="linear", lut_identity="gray")
    second = TilePresentationIdentity(5, levels=(0.5, 1.5), scale="linear", lut_identity="gray")

    assert first != second
    assert identity.semantic_key == _identity().semantic_key
