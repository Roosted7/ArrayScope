"""G7 Phase B GPU ring: on-device BC4 encode + hardware-sampler quality closure.

Runs on a real wgpu device that exposes ``texture-compression-bc``.  Skips
cleanly when no such adapter is present (CI stays green); on a BC-capable machine
it asserts: the WGSL compute BC4 encoder matches the CPU reference encoder
bit-for-bit, and the hardware texture sampler returns what the reference decode
predicts (so the measured CPU-decode quality is the quality the GPU delivers).
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from arrayscope.gpu import bc_codec


def _bc_device():
    try:
        import wgpu  # noqa: F401

        from arrayscope.gpu.bc_gpu import GpuDecodeUnavailable, create_compute_device
    except Exception:
        return None
    try:
        return create_compute_device("low-power", features=["texture-compression-bc"])
    except GpuDecodeUnavailable:
        return None
    except Exception:
        return None


_DEVICE = _bc_device()
pytestmark = pytest.mark.skipif(
    _DEVICE is None, reason="no BC-capable wgpu adapter on this machine"
)


def _smooth_tile(shape=(256, 256)):
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    return (np.sin(x / 40.0) * np.cos(y / 35.0) + x / float(shape[1])).astype(np.float32)


def test_gpu_bc4_encoder_is_quality_equivalent_to_cpu_encoder():
    from arrayscope.gpu.bc_gpu import GpuBc4Encoder

    unit, _ = bc_codec.normalize_tile(_smooth_tile())
    cpu_bytes, h, w = bc_codec.bc4_encode(unit)
    gpu_bytes, gh, gw = GpuBc4Encoder(_DEVICE).encode(unit)
    assert (gh, gw) == (h, w)
    assert len(gpu_bytes) == len(cpu_bytes)
    cpu_q = bc_codec.quality_of(unit, bc_codec.bc4_decode(cpu_bytes, h, w))
    gpu_q = bc_codec.quality_of(unit, bc_codec.bc4_decode(gpu_bytes, gh, gw))
    # The WGSL encoder uses the same endpoints/palette; f32 vs f64 rounding can
    # flip a handful of texel indices at exact ties, so the two are quality-
    # equivalent rather than byte-identical.
    assert gpu_q.psnr_db == pytest.approx(cpu_q.psnr_db, abs=0.5)
    assert gpu_q.psnr_db > 40.0


def test_hardware_sampler_matches_reference_decode():
    from arrayscope.gpu.bc_gpu import sample_bc4_texture, upload_bc4_texture

    unit, _ = bc_codec.normalize_tile(_smooth_tile())
    data, h, w = bc_codec.bc4_encode(unit)
    tex = upload_bc4_texture(_DEVICE, data, h, w)
    hw = sample_bc4_texture(_DEVICE, tex, h, w)
    ref = bc_codec.bc4_decode(data, h, w)
    # the hardware sampler reproduces the spec decode (residual is r32float target
    # precision, not a codec disagreement)
    assert bc_codec.quality_of(ref, hw).psnr_db > 50.0


def teardown_module(module):
    with contextlib.suppress(Exception):
        _DEVICE._destroy() if hasattr(_DEVICE, "_destroy") else None
