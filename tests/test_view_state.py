import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PACKAGE = types.ModuleType("arrayscope")
PACKAGE.__path__ = [str(ROOT / "arrayscope")]
sys.modules.setdefault("arrayscope", PACKAGE)

VIEW_STATE_PATH = ROOT / "arrayscope" / "core" / "view_state.py"
SPEC = importlib.util.spec_from_file_location("arrayscope.core.view_state", VIEW_STATE_PATH)
view_state_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = view_state_module
SPEC.loader.exec_module(view_state_module)

ChannelMode = view_state_module.ChannelMode
ScaleMode = view_state_module.ScaleMode
ViewState = view_state_module.ViewState


def test_from_shape_1d_uses_line_axis_only():
    state = ViewState.from_shape((5,))

    assert state.ndim == 1
    assert state.shape == (5,)
    assert state.image_axes is None
    assert state.line_axis == 0
    assert state.slice_indices == (0,)
    assert state.display_axes() == (0,)
    assert state.non_display_axes() == ()


def test_from_shape_2d_defaults_to_image_axes():
    state = ViewState.from_shape((4, 6))

    assert state.image_axes == (0, 1)
    assert state.line_axis == 0
    assert state.display_axes() == (0, 1)
    assert state.non_display_axes() == ()


def test_from_shape_3d_uses_first_two_non_singleton_image_axes():
    state = ViewState.from_shape((1, 5, 7))

    assert state.image_axes == (1, 2)
    assert state.line_axis == 1
    assert state.non_display_axes() == (0,)


def test_singleton_dimensions_keep_current_viewer_fallback():
    state = ViewState.from_shape((1, 5, 1))

    assert state.image_axes == (0, 1)
    assert state.line_axis == 1
    assert state.slice_indices == (0, 0, 0)


def test_with_slice_returns_new_valid_state():
    state = ViewState.from_shape((3, 4, 5))
    updated = state.with_slice(2, 4)

    assert updated is not state
    assert state.slice_indices == (0, 0, 0)
    assert updated.slice_indices == (0, 0, 4)


def test_invalid_axes_are_rejected():
    state = ViewState.from_shape((3, 4, 5))

    with pytest.raises(ValueError, match="out of bounds"):
        state.with_line_axis(3)

    with pytest.raises(ValueError, match="distinct"):
        state.with_image_axes(1, 1)

    with pytest.raises(ValueError, match="out of bounds"):
        ViewState(
            ndim=2,
            shape=(3, 4),
            image_axes=(0, 2),
            line_axis=0,
            slice_indices=(0, 0),
            axis_flipped=(False, False),
            axis_fftshifted=(False, False),
        )


def test_slice_index_bounds_are_rejected():
    state = ViewState.from_shape((3, 4))

    with pytest.raises(ValueError, match="slice index"):
        state.with_slice(0, 3)

    with pytest.raises(ValueError, match="slice index"):
        state.with_slice(1, -1)


def test_lengths_and_shape_are_validated():
    with pytest.raises(ValueError, match="shape length"):
        ViewState(
            ndim=2,
            shape=(3,),
            image_axes=None,
            line_axis=0,
            slice_indices=(0, 0),
            axis_flipped=(False, False),
            axis_fftshifted=(False, False),
        )

    with pytest.raises(ValueError, match="slice_indices length"):
        ViewState(
            ndim=2,
            shape=(3, 4),
            image_axes=(0, 1),
            line_axis=0,
            slice_indices=(0,),
            axis_flipped=(False, False),
            axis_fftshifted=(False, False),
        )

    with pytest.raises(ValueError, match="axis_flipped length"):
        ViewState(
            ndim=2,
            shape=(3, 4),
            image_axes=(0, 1),
            line_axis=0,
            slice_indices=(0, 0),
            axis_flipped=(False,),
            axis_fftshifted=(False, False),
        )


def test_channel_scale_and_axis_flags_are_validated_and_updated():
    state = ViewState.from_shape((3, 4))
    updated = (
        state.with_channel(ChannelMode.ABS)
        .with_scale(ScaleMode.SYMLOG)
        .with_axis_flipped(0, True)
        .with_axis_fftshifted(1, True)
    )

    assert updated.channel == ChannelMode.ABS
    assert updated.scale == ScaleMode.SYMLOG
    assert updated.axis_flipped == (True, False)
    assert updated.axis_fftshifted == (False, True)


def test_for_shape_preserves_surviving_flags_and_clamps_slices():
    state = (
        ViewState.from_shape((3, 4, 5))
        .with_slice(0, 2)
        .with_slice(1, 3)
        .with_axis_flipped(1, True)
        .with_axis_fftshifted(2, True)
    )

    migrated = state.for_shape((3, 2))

    assert migrated.shape == (3, 2)
    assert migrated.image_axes == (0, 1)
    assert migrated.slice_indices == (2, 1)
    assert migrated.axis_flipped == (False, True)
    assert migrated.axis_fftshifted == (False, False)


def test_for_shape_uses_line_axis_only_for_1d():
    state = ViewState.from_shape((3, 4)).with_axis_flipped(0, True)

    migrated = state.for_shape((5,))

    assert migrated.image_axes is None
    assert migrated.line_axis == 0
    assert migrated.axis_flipped == (True,)


def test_with_image_axis_keeps_axes_distinct():
    state = ViewState.from_shape((3, 4, 5))

    moved_y = state.with_image_axis("y", 1)
    moved_x = state.with_image_axis("x", 0)

    assert moved_y.image_axes == (1, 0)
    assert moved_x.image_axes == (1, 0)


def test_transposed_image_axes_swaps_axes_without_touching_flags():
    state = ViewState.from_shape((3, 4, 5)).with_axis_flipped(0, True)

    transposed = state.transposed_image_axes()

    assert transposed.image_axes == (1, 0)
    assert transposed.axis_flipped == state.axis_flipped


def test_view_state_module_has_no_qt_or_pyqtgraph_imports():
    tree = ast.parse(VIEW_STATE_PATH.read_text())

    imported_roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.append(node.module.split(".")[0])

    assert "pyqtgraph" not in imported_roots
    assert not any(name.startswith(("PyQt", "PySide")) for name in imported_roots)
