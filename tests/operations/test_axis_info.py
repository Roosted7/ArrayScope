from arrayscope.core.axis_info import AxisInfo, default_axes, output_axes_for_operation, output_axes_for_operations
from arrayscope.operations.pipeline import (
    ArrayDocument,
    CenteredFFT,
    CombineRealImagAxis,
    Crop,
    Mean,
    ReverseAxis,
    RootSumSquares,
    SplitComplexAxis,
)


def test_default_axes_have_stable_ids_labels_and_sizes():
    axes = default_axes((3, 4))

    assert tuple(axis.id for axis in axes) == ("axis-0", "axis-1")
    assert tuple(axis.label for axis in axes) == ("Dim 0", "Dim 1")
    assert tuple(axis.size for axis in axes) == (3, 4)


def test_axis_info_crop_updates_size_and_preserves_identity():
    axes = default_axes((3, 4))

    updated = output_axes_for_operation(axes, Crop(axis=1, start=1, stop=3))

    assert updated[1].id == axes[1].id
    assert updated[1].size == 2


def test_axis_info_reduction_removes_axis():
    axes = default_axes((3, 4, 5))

    updated = output_axes_for_operations(axes, (Mean(axis=1), RootSumSquares(axis=1)))

    assert tuple(axis.id for axis in updated) == ("axis-0",)


def test_axis_info_fft_and_reverse_preserve_identity():
    axes = default_axes((3, 4))

    updated = output_axes_for_operations(axes, (ReverseAxis(axis=0), CenteredFFT(axis=1)))

    assert updated == axes


def test_axis_info_combine_and_split_update_size_and_coordinate():
    axes = default_axes((3, 2))

    combined = output_axes_for_operation(axes, CombineRealImagAxis(axis=1))
    split = output_axes_for_operation(combined, SplitComplexAxis(axis=1))

    assert combined[1].size == 1
    assert combined[1].coordinate == "complex"
    assert split[1].size == 2
    assert split[1].coordinate == "real-imag"


def test_array_document_exposes_current_axes_matching_current_shape():
    axes = (AxisInfo("readout", "Readout", 3), AxisInfo("coil", "Coil", 2))
    document = ArrayDocument([[1, 2], [3, 4], [5, 6]], operations=(Mean(axis=1),), axes=axes)

    assert document.current_shape == (3,)
    assert tuple(axis.id for axis in document.current_axes) == ("readout",)
    assert tuple(axis.size for axis in document.current_axes) == document.current_shape

