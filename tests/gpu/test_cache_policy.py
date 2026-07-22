"""G7 Phase A: the adaptive RAM-axis policy engages only when it strictly helps."""

from __future__ import annotations

import numpy as np

from arrayscope.gpu.cache_policy import (
    decide_compressed_tier,
    preferred_codec_for_dtype,
)
from arrayscope.gpu.device_topology import DeviceTopology

_INTEGRATED = DeviceTopology(kind="integrated", unified_memory=True)
_DISCRETE = DeviceTopology(kind="discrete", unified_memory=False)


def test_small_data_that_fits_stays_off():
    d = decide_compressed_tier(
        working_set_bytes=1 << 20,
        budget_bytes=8 << 20,
        dtype=np.float32,
        topology=_INTEGRATED,
    )
    assert d.engage is False
    assert d.codec_name == "raw"
    assert "fits" in d.reason


def test_large_data_under_pressure_engages_lossless_codec():
    d = decide_compressed_tier(
        working_set_bytes=64 << 20,
        budget_bytes=8 << 20,
        dtype=np.float32,
        topology=_INTEGRATED,
    )
    assert d.engage is True
    # Float/complex/int16 prefer zfp's lossless transform.
    assert d.codec_name == "zfp"
    assert d.pressure_ratio > 1.0


def test_codec_choice_is_dtype_driven():
    # uint8 is declined by zfp -> blosc2 (byte-exact) is chosen instead.
    d = decide_compressed_tier(
        working_set_bytes=64 << 20,
        budget_bytes=8 << 20,
        dtype=np.uint8,
        topology=_INTEGRATED,
    )
    assert d.engage is True
    assert d.codec_name == "blosc2"
    assert preferred_codec_for_dtype(np.uint8) == ("blosc2",)
    assert preferred_codec_for_dtype(np.float32)[0] == "zfp"


def test_topology_does_not_gate_ram_win_but_sets_phase_b_seam():
    integrated = decide_compressed_tier(
        working_set_bytes=64 << 20, budget_bytes=8 << 20, dtype=np.float32, topology=_INTEGRATED
    )
    discrete = decide_compressed_tier(
        working_set_bytes=64 << 20, budget_bytes=8 << 20, dtype=np.float32, topology=_DISCRETE
    )
    # Both engage on the RAM axis; only the discrete device flags the transfer seam.
    assert integrated.engage is True
    assert discrete.engage is True
    assert integrated.discrete_transfer_candidate is False
    assert discrete.discrete_transfer_candidate is True


def test_never_selects_a_lossy_codec():
    # The policy only ever names lossless codecs; resolve_codec is the final gate.
    for dtype in (np.float32, np.complex64, np.int16, np.uint8):
        d = decide_compressed_tier(
            working_set_bytes=64 << 20, budget_bytes=1 << 20, dtype=dtype, topology=_INTEGRATED
        )
        assert d.codec_name in ("raw", "zfp", "blosc2")
