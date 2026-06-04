import ast
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
PACKAGE = types.ModuleType("arrayscope")
PACKAGE.__path__ = [str(ROOT / "arrayscope")]
sys.modules.setdefault("arrayscope", PACKAGE)


def load_module(name):
    path = ROOT / "arrayscope" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"arrayscope.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


view_state = load_module("view_state")
slice_engine = load_module("slice_engine")

ChannelMode = view_state.ChannelMode
ScaleMode = view_state.ScaleMode
ViewState = view_state.ViewState
apply_channel = slice_engine.apply_channel
complex_to_rgb = slice_engine.complex_to_rgb
make_image = slice_engine.make_image
make_line = slice_engine.make_line
symlog = slice_engine.symlog


def state_for(shape, image_axes=None, line_axis=None, slices=None, channel=ChannelMode.REAL, scale=ScaleMode.LINEAR):
    return ViewState(
        ndim=len(shape),
        shape=tuple(shape),
        image_axes=image_axes,
        line_axis=line_axis,
        slice_indices=tuple(slices if slices is not None else (0,) * len(shape)),
        channel=channel,
        scale=scale,
        axis_flipped=(False,) * len(shape),
        axis_fftshifted=(False,) * len(shape),
    )


def test_apply_channel_values():
    data = np.array([1 + 2j, -3 - 4j])

    np.testing.assert_array_equal(apply_channel(data, ChannelMode.REAL), np.array([1.0, -3.0]))
    np.testing.assert_array_equal(apply_channel(data, ChannelMode.IMAG), np.array([2.0, -4.0]))
    np.testing.assert_allclose(apply_channel(data, ChannelMode.ABS), np.array([np.sqrt(5), 5.0]))
    np.testing.assert_allclose(apply_channel(data, ChannelMode.COMPLEX), np.array([np.sqrt(5), 5.0]))
    np.testing.assert_allclose(apply_channel(data, ChannelMode.ANGLE), np.angle(data))


def test_make_image_2d_real_preserves_existing_transpose_behavior():
    data = np.arange(6).reshape(2, 3)
    state = state_for(data.shape, image_axes=(0, 1), line_axis=0)

    image = make_image(data, state)

    assert image.data.shape == (3, 2)
    np.testing.assert_array_equal(image.data, data.T)
    assert image.histogram_data is None


def test_make_image_3d_slices_non_display_axis_and_applies_channel_and_scale():
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(float)
    state = state_for(
        data.shape,
        image_axes=(1, 2),
        line_axis=1,
        slices=(1, 0, 0),
        channel=ChannelMode.REAL,
        scale=ScaleMode.SYMLOG,
    )

    image = make_image(data, state)

    expected = symlog(data[1, :, :].T)
    assert image.data.shape == (4, 3)
    np.testing.assert_allclose(image.data, expected)


def test_make_image_ndslice_reversed_axes_preserves_existing_orientation():
    data = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    state = state_for(
        data.shape,
        image_axes=(3, 1),
        line_axis=3,
        slices=(1, 0, 2, 0),
        channel=ChannelMode.REAL,
    )

    image = make_image(data, state)

    expected = np.squeeze(data[1:2, :, 2:3, :])
    assert image.data.shape == (3, 5)
    np.testing.assert_array_equal(image.data, expected)


def test_make_image_angle_sets_default_levels_and_values():
    data = np.array([[1 + 0j, 0 + 1j], [-1 + 0j, 0 - 1j]])
    state = state_for(data.shape, image_axes=(0, 1), line_axis=0, channel=ChannelMode.ANGLE)

    image = make_image(data, state)

    assert image.default_levels == (-np.pi, np.pi)
    np.testing.assert_allclose(image.data, np.angle(data.T))


def test_make_image_complex_real_imag_and_abs_channels():
    data = np.array([[1 + 2j, -3 + 4j], [5 - 6j, -7 - 8j]])

    real_state = state_for(data.shape, image_axes=(0, 1), line_axis=0, channel=ChannelMode.REAL)
    imag_state = state_for(data.shape, image_axes=(0, 1), line_axis=0, channel=ChannelMode.IMAG)
    abs_state = state_for(data.shape, image_axes=(0, 1), line_axis=0, channel=ChannelMode.ABS)

    np.testing.assert_array_equal(make_image(data, real_state).data, np.real(data.T))
    np.testing.assert_array_equal(make_image(data, imag_state).data, np.imag(data.T))
    np.testing.assert_allclose(make_image(data, abs_state).data, np.abs(data.T))


def test_make_image_complex_returns_rgb_and_magnitude_histogram():
    data = np.array([[1 + 0j, -1 + 0j], [1j, -1j]])
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = np.arange(256, dtype=np.uint8)
    lut[:, 2] = 255 - lut[:, 0]
    state = state_for(data.shape, image_axes=(0, 1), line_axis=0, channel=ChannelMode.COMPLEX)

    image = make_image(data, state, colormap_lut=lut)

    assert image.data.shape == (2, 2, 3)
    np.testing.assert_array_equal(image.histogram_data, np.ones((2, 2)))
    np.testing.assert_array_equal(image.data[0, 0], lut[127])
    np.testing.assert_array_equal(image.data[1, 0], lut[255])


def test_make_image_complex_rgb_uses_phase_for_color_and_magnitude_for_histogram():
    data = np.array([[1 + 0j, 2j], [-3 + 0j, -4j]])
    lut = np.zeros((4, 3), dtype=np.uint8)
    lut[0] = [10, 0, 0]
    lut[1] = [20, 0, 0]
    lut[2] = [30, 0, 0]
    lut[3] = [40, 0, 0]
    state = state_for(data.shape, image_axes=(0, 1), line_axis=0, channel=ChannelMode.COMPLEX)

    image = make_image(data, state, colormap_lut=lut)

    assert image.data.shape == (2, 2, 3)
    np.testing.assert_array_equal(image.histogram_data, np.abs(data.T))
    np.testing.assert_array_equal(image.data[0, 0], lut[1])
    np.testing.assert_array_equal(image.data[1, 0], lut[2])


def test_complex_to_rgb_rejects_bad_lut_shape():
    with pytest.raises(ValueError, match="colormap_lut"):
        complex_to_rgb(np.array([1 + 0j]), colormap_lut=np.array([1, 2, 3]))


def test_make_line_1d_uses_channel_conversion():
    data = np.array([1 + 2j, 3 + 4j, 5 + 6j])
    state = state_for(data.shape, line_axis=0, channel=ChannelMode.IMAG)

    line = make_line(data, state)

    assert line.axis == 0
    assert line.data.shape == (3,)
    np.testing.assert_array_equal(line.data, np.array([2.0, 4.0, 6.0]))


def test_make_line_3d_slices_other_axes():
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    state = state_for(data.shape, image_axes=(1, 2), line_axis=2, slices=(1, 2, 0), channel=ChannelMode.REAL)

    line = make_line(data, state)

    np.testing.assert_array_equal(line.data, data[1, 2, :])


def test_make_image_rejects_shape_mismatch():
    data = np.arange(6).reshape(2, 3)
    state = state_for((2, 4), image_axes=(0, 1), line_axis=0)

    with pytest.raises(ValueError, match="does not match"):
        make_image(data, state)


def test_slice_engine_module_has_no_qt_or_pyqtgraph_imports():
    tree = ast.parse((ROOT / "arrayscope" / "slice_engine.py").read_text())

    imported_roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.append(node.module.split(".")[0])

    assert "pyqtgraph" not in imported_roots
    assert not any(name.startswith(("PyQt", "PySide")) for name in imported_roots)
