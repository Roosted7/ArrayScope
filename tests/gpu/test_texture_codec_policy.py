"""G7 Phase B: device-feature texture-codec policy (offscreen, no GPU).

Pins the family the executor picks for a device: discrete -> BC (NVIDIA has no
ASTC), integrated -> ASTC when the device advertises it and an encoder is
installed, else BC, and OFF by default (the render path stays byte-identical
unless a caller opts in).
"""

from __future__ import annotations

from arrayscope.gpu import astc_codec
from arrayscope.gpu.keys import COMPLEX_RG32F, SCALAR_R32F
from arrayscope.gpu.wgpu_executor import (
    _ASTC_FEATURE,
    _BC_FEATURE,
    _BC_POOL_FORMATS,
    _decide_codec_family,
)

_BOTH = {_BC_FEATURE, _ASTC_FEATURE}


def test_default_is_off_and_render_path_is_byte_identical():
    for integrated in (False, True):
        assert _decide_codec_family("off", _BOTH, integrated) == "none"


def test_discrete_prefers_bc_even_where_astc_is_advertised(monkeypatch):
    monkeypatch.setattr(astc_codec, "astc_available", lambda: True)

    assert _decide_codec_family("auto", _BOTH, False) == "bc"
    assert _BC_POOL_FORMATS[SCALAR_R32F] == "bc4-r-unorm"
    assert _BC_POOL_FORMATS[COMPLEX_RG32F] == "bc5-rg-unorm"


def test_integrated_prefers_astc_when_supported(monkeypatch):
    monkeypatch.setattr(astc_codec, "astc_available", lambda: True)

    assert _decide_codec_family("auto", _BOTH, True) == "astc"


def test_integrated_falls_back_to_bc_without_an_astc_encoder(monkeypatch):
    monkeypatch.setattr(astc_codec, "astc_available", lambda: False)

    assert _decide_codec_family("auto", _BOTH, True) == "bc"


def test_device_without_a_compressed_feature_degrades_to_raw(monkeypatch):
    monkeypatch.setattr(astc_codec, "astc_available", lambda: True)

    assert _decide_codec_family("auto", set(), False) == "none"
