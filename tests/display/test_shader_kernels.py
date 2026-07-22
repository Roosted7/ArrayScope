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


def test_ensure_prewarming_never_blocks_and_registers_group():
    # The display kernels register with the shared runtime under "display" and
    # ensure_prewarming must return immediately (it only spawns a daemon thread).
    from arrayscope.core import numba_runtime

    assert "display" in numba_runtime.registered_names()
    shader_kernels.ensure_prewarming()  # returns immediately even if already warm


def test_unregistered_group_is_never_ready():
    from arrayscope.core import numba_runtime

    assert numba_runtime.ready("does-not-exist") is False


def test_display_prewarm_is_gated_to_the_cpu_backend():
    # The display kernels run only on pyqtgraph; a wgpu/vispy session must not
    # bulk-compile them. The gate is a settings-only check (no numba needed).
    from arrayscope.app.settings_state import ImageRenderingBackendChoice
    from arrayscope.display.image_view_factory import cpu_display_backend_likely

    class _Settings:
        def __init__(self, choice):
            self.image_rendering_backend = choice

    assert cpu_display_backend_likely(_Settings(ImageRenderingBackendChoice.WGPU)) is False
    assert cpu_display_backend_likely(_Settings(ImageRenderingBackendChoice.VISPY)) is False
    assert cpu_display_backend_likely(_Settings(ImageRenderingBackendChoice.PYQTGRAPH)) is True

    # The registered display group defers to that predicate for bulk prewarm.
    from arrayscope.core import numba_runtime

    group = numba_runtime.get_group("display")
    assert group is not None
    monkeypatched = {"value": False}
    import arrayscope.display.shader_kernels as sk

    original = sk._cpu_display_backend_active
    try:
        sk._GROUP._should_prewarm = lambda: monkeypatched["value"]
        assert group.wanted() is False
        monkeypatched["value"] = True
        assert group.wanted() is True
    finally:
        sk._GROUP._should_prewarm = original
