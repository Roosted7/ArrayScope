"""An anchored crop that does not start on a reduction boundary.

A source-anchored axis reduces on the GLOBAL source grid, not on the crop's
own origin.  A crop whose native start is not a multiple of the factor
therefore straddles one extra partial leading bin: 100 native rows starting at
row 1 reduce to 51 samples at factor 2, not 50.

The WGPU geometry check derived the expected extent from the local extent
alone, so it computed 50, rejected the legitimate payload, and stranded the
work — reproducible on a 336x336x272 montage by slicing a display axis to an
odd offset.

The rule now lives in one place (:func:`arrayscope.display.lod.reduced_extent`)
and both the producer and the validating consumer are checked against it here,
because the failure mode was precisely two owners disagreeing about it.
"""

from __future__ import annotations

import numpy as np
import pytest

from arrayscope.display.lod import LodInfo, reduced_extent
from arrayscope.display.model.frame import DisplayTilePayload
from arrayscope.display.wgpu_imageview2d import _wgpu_payload_lod_geometry
from arrayscope.render.effects import _reduced_axis_length
from arrayscope.operations.regions import AxisRegion, AxisRegionKind

FACTORS = (1, 2, 4, 8)
ORIGINS = (0, 1, 2, 3, 5, 7, 8, 33, 100)
EXTENTS = (1, 2, 3, 15, 33, 50, 100, 336)


def _payload(origin_yx, source_shape, factor, texture_shape):
    texture = np.zeros(texture_shape, dtype=np.float32)
    return (
        DisplayTilePayload(
            tile_number=0,
            source_index=0,
            image=texture,
            histogram_data=None,
            source_id=("tile", 0),
            lod=LodInfo(
                level=max(0, factor.bit_length() - 1),
                factor=factor,
                source_shape=source_shape,
                texture_shape=texture_shape,
                source_origin=origin_yx,
            ),
        ),
        texture,
    )


@pytest.mark.parametrize("factor", FACTORS)
@pytest.mark.parametrize("origin", ORIGINS)
@pytest.mark.parametrize("extent", EXTENTS)
def test_the_reduction_rule_matches_the_producer(origin, extent, factor):
    """One rule, two owners: the extent emitted is the extent expected.

    ``_reduced_axis_length`` is what the preview producer actually uses to size
    a reduced read. If the backend's expectation ever diverges from it, an
    honest payload gets rejected — which is the defect this file exists for.
    """

    region = AxisRegion(AxisRegionKind.SLICE, (origin, origin + extent, 1))
    produced = _reduced_axis_length(region, origin + extent, factor, source_aligned=True)

    assert reduced_extent(origin, extent, factor) == produced, (
        f"the backend expects {reduced_extent(origin, extent, factor)} reduced samples for "
        f"origin={origin} extent={extent} factor={factor}, but the producer emits {produced}"
    )


def test_an_unanchored_axis_keeps_the_plain_ceiling():
    """Phase 0 is the unanchored case, and must not change behaviour."""

    for factor in FACTORS:
        for extent in EXTENTS:
            region = AxisRegion(AxisRegionKind.SLICE, (0, extent, 1))
            assert (
                reduced_extent(0, extent, factor)
                == _reduced_axis_length(region, extent, factor, source_aligned=False)
                == -(-extent // factor)
            )


@pytest.mark.parametrize("origin", (1, 3, 5, 7))
def test_wgpu_accepts_an_anchored_odd_origin_crop(origin):
    """The exact shape the profiler stranded on is now admitted."""

    extent, factor = 100, 2
    reduced = reduced_extent(origin, extent, factor)
    payload, texture = _payload((origin, origin), (extent, extent), factor, (reduced, reduced))

    level, source_shape = _wgpu_payload_lod_geometry(payload, texture)

    assert level == 1
    assert source_shape == (extent, extent)


def test_wgpu_admits_a_per_axis_phase_in_both_orientations():
    """Each axis carries its own phase; a transpose swaps the pair with it."""

    factor = 2
    for source_shape, origin in (((100, 60), (1, 0)), ((60, 100), (0, 1))):
        reduced = tuple(
            reduced_extent(o, extent, factor)
            for o, extent in zip(origin, source_shape, strict=True)
        )
        payload, texture = _payload(origin, source_shape, factor, reduced)

        _level, resolved = _wgpu_payload_lod_geometry(payload, texture)

        assert resolved == source_shape
        # Only the phased axis gains the extra bin.
        assert reduced == tuple(
            -(-extent // factor) + (1 if o % factor else 0)
            for o, extent in zip(origin, source_shape, strict=True)
        )


@pytest.mark.parametrize("wrong", (-1, 1))
def test_wgpu_still_rejects_a_geometry_that_is_off_by_one(wrong):
    """The check stays strict: exactly one extent is admissible per phase."""

    extent, factor, origin = 100, 2, 1
    reduced = reduced_extent(origin, extent, factor) + wrong
    payload, texture = _payload((origin, origin), (extent, extent), factor, (reduced, reduced))

    with pytest.raises(ValueError, match="does not match its native LOD ladder"):
        _wgpu_payload_lod_geometry(payload, texture)


def test_wgpu_rejects_an_odd_extent_claimed_without_its_phase():
    """A payload that omits the phase cannot smuggle the extra bin through.

    This is the pre-fix state expressed as a rule: 51 samples for 100 native
    rows is only correct BECAUSE the reduction is phased. Claim it at phase 0
    and it is simply wrong.
    """

    payload, texture = _payload((0, 0), (100, 100), 2, (51, 51))

    with pytest.raises(ValueError, match="does not match its native LOD ladder"):
        _wgpu_payload_lod_geometry(payload, texture)
