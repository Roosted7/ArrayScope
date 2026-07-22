"""Equivalence + lifecycle tests for the optional numba display kernels.

The numba fast path must be **bit-identical** to the NumPy reference so that
displayed pixels never depend on whether the JIT happens to be warm.  Each test
compares the warm path against the same public function with the accelerator
forced off.
"""

from __future__ import annotations

import numpy as np
import pytest

from arrayscope.display import shader_kernels
from arrayscope.display import shader_mapping as sm
from arrayscope.display.image_upload import rgb_display_for_levels
from arrayscope.display.shader_mapping import (
    ShaderComponent,
    ShaderMapping,
    ShaderScale,
    cpu_display_rgba,
    default_gray_lut,
)

_NUMBA_WARM = shader_kernels.prewarm_blocking()
requires_numba = pytest.mark.skipif(not _NUMBA_WARM, reason="numba accelerator unavailable")


@pytest.fixture
def force_numpy(monkeypatch):
    """Context in which cpu_display_rgba / rgb_display_for_levels use NumPy."""

    monkeypatch.setattr(shader_kernels, "ready", lambda: False)
    return None


def _numpy_cpu_rgba(monkeypatch, data, mapping):
    with monkeypatch.context() as m:
        m.setattr(shader_kernels, "ready", lambda: False)
        return cpu_display_rgba(data, mapping)


def _numpy_rgb(monkeypatch, base, hist, levels):
    with monkeypatch.context() as m:
        m.setattr(shader_kernels, "ready", lambda: False)
        return rgb_display_for_levels(base, hist, levels)


_EDGE = np.array(
    [[-10.0, -1.0, 0.0, 1e-6, 1.0, 10.0, 100.0, np.inf, -np.inf, np.nan]], dtype=np.float32
)
_LUTS = [
    None,
    default_gray_lut(256),
    np.array([[0, 0, 255], [255, 0, 0]], np.uint8),
    np.full((4, 3), 255, np.uint8),
    np.array([[10, 20, 30], [40, 50, 60], [200, 100, 50]], np.uint8),
]
_LEVELS = [(0.0, 1.0), (-2.0, 3.0), (0.0, 2.0), (-1.0, 1.0)]


@requires_numba
@pytest.mark.parametrize(
    "component", [ShaderComponent.REAL, ShaderComponent.ABS, ShaderComponent.IMAG]
)
@pytest.mark.parametrize("scale", [ShaderScale.LINEAR, ShaderScale.LOG, ShaderScale.SYMLOG])
def test_scalar_path_bit_identical_across_edge_values(monkeypatch, component, scale):
    data = (_EDGE + 0j).astype(np.complex64)
    for lut in _LUTS:
        for levels in _LEVELS:
            mapping = ShaderMapping(
                component=component,
                scale=scale,
                levels=levels,
                lut_data=lut,
                display_mode="scalar",
                symlog_constant=1.0,
            )
            ref = _numpy_cpu_rgba(monkeypatch, data, mapping)
            got = cpu_display_rgba(data, mapping)
            np.testing.assert_array_equal(got, ref)


@requires_numba
@pytest.mark.parametrize("seed", range(6))
def test_scalar_path_bit_identical_on_random_fft_tiles(monkeypatch, seed):
    rng = np.random.default_rng(seed)
    d = (rng.standard_normal((37, 53)) + 1j * rng.standard_normal((37, 53))).astype(np.complex64)
    d *= np.exp(rng.random((37, 53)).astype(np.float32) * 6).astype(np.float32)
    mapping = ShaderMapping(
        component=ShaderComponent.ABS,
        scale=ShaderScale.LOG,
        levels=(-2.0, 3.0),
        lut_data=default_gray_lut(256),
        display_mode="scalar",
    )
    np.testing.assert_array_equal(
        cpu_display_rgba(d, mapping), _numpy_cpu_rgba(monkeypatch, d, mapping)
    )


@requires_numba
@pytest.mark.parametrize("component", [ShaderComponent.ANGLE, ShaderComponent.COMPLEX_PHASE])
def test_phase_color_angle_branch_bit_identical(monkeypatch, component):
    data = np.array([[-1j, 1 + 0j, 1j, -1 + 0j, np.nan + 0j]], np.complex64)
    for lut in _LUTS:
        for levels in [None, (-np.pi, np.pi), (-np.pi / 2, np.pi / 2)]:
            mapping = ShaderMapping(
                component=component,
                scale=ShaderScale.LINEAR,
                levels=levels,
                lut_data=lut,
                display_mode="phase_color",
            )
            np.testing.assert_array_equal(
                cpu_display_rgba(data, mapping), _numpy_cpu_rgba(monkeypatch, data, mapping)
            )


@requires_numba
def test_phase_color_magnitude_branch_falls_back_to_numpy():
    # component=ABS in phase-color mode uses apply_phase_lut (truncating cast),
    # which we deliberately leave on NumPy. The result must still be correct.
    data = np.array([[1 + 0j, 10 + 0j, 100 + 0j]], dtype=np.complex64)
    lut = np.full((4, 3), 255, dtype=np.uint8)
    mapping = ShaderMapping(
        component=ShaderComponent.ABS,
        scale=ShaderScale.LOG,
        levels=(0.0, 2.0),
        lut_data=lut,
        display_mode="phase_color",
    )
    assert sm._numba_cpu_display_rgba(data, mapping) is None
    rgba = cpu_display_rgba(data, mapping)
    np.testing.assert_array_equal(rgba[0, :, 0], np.array([0, 127, 255], dtype=np.uint8))


@requires_numba
def test_scalar_without_levels_falls_back_to_numpy():
    data = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    mapping = ShaderMapping(component=ShaderComponent.REAL, scale=ShaderScale.LINEAR)
    assert mapping.levels is None
    assert sm._numba_cpu_display_rgba(data, mapping) is None


@requires_numba
@pytest.mark.parametrize("seed", range(8))
def test_rgb_display_for_levels_bit_identical(monkeypatch, seed):
    rng = np.random.default_rng(seed + 100)
    size = int(rng.integers(20, 300))
    base = (rng.random((size, size, 3)) * 255).astype(np.float32)
    hist = rng.random((size, size)).astype(np.float32) * 2 - 0.5
    idx = rng.integers(0, size * size, 5)
    hist.flat[idx] = [np.nan, np.inf, -np.inf, 1e30, -1e30]
    for levels in [(0.25, 1.25), (0.0, 1.0), (-0.5, 0.5)]:
        np.testing.assert_array_equal(
            rgb_display_for_levels(base, hist, levels),
            _numpy_rgb(monkeypatch, base, hist, levels),
        )


def test_existing_public_results_unchanged_without_numba(force_numpy):
    # With the accelerator forced off, the documented oracle values still hold.
    data = np.array([[0.0, 1.0, np.nan]], dtype=np.float32)
    mapping = ShaderMapping(
        component=ShaderComponent.REAL, scale=ShaderScale.LINEAR, levels=(0.0, 1.0)
    )
    rgba = cpu_display_rgba(data, mapping)
    assert rgba[0, 0, 0] == 0
    assert rgba[0, 1, 0] == 255
    assert rgba[0, 2, 3] == 0


def test_ready_is_false_before_prewarm(monkeypatch):
    # ensure_prewarming must never block the caller.
    monkeypatch.setattr(shader_kernels, "_READY", type(shader_kernels._READY)())
    monkeypatch.setattr(shader_kernels, "_WARM_STARTED", False)
    assert shader_kernels.ready() is False
    shader_kernels.ensure_prewarming()  # returns immediately (spawns a thread)
