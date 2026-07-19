import numpy as np
import pytest

from arrayscope.core.axis_info import default_axes
from arrayscope.io.file_interpreters import (
    _PHILIPS_REC_AXIS_LABELS,
    _axes_matching_shape,
    _labeled_axes,
    _stacked_axes,
    data_file_suffix,
    load_path,
)


def test_data_file_suffix_keeps_only_supported_compound_suffixes():
    assert data_file_suffix("subject.session.npy") == ".npy"
    assert data_file_suffix("scan.nii.gz") == ".nii.gz"


def test_load_path_accepts_dotted_numpy_filename(tmp_path):
    path = tmp_path / "subject.session.npy"
    data = np.arange(4).reshape(2, 2)
    np.save(path, data)

    loaded = load_path(path)

    np.testing.assert_array_equal(loaded.data, data)
    assert loaded.metadata["detected_format"] == "numpy"
    assert loaded.axes is None


def test_labeled_axes_sets_labels_units_and_spacings_for_leading_axes():
    axes = _labeled_axes((4, 5, 6), labels=("X", "Y"), units=("mm",), spacings=(1.5, None, 0.0))

    assert axes[0].label == "X"
    assert axes[0].unit == "mm"
    assert axes[0].spacing == 1.5
    assert axes[1].label == "Y"
    assert axes[1].spacing is None
    assert axes[2].label == "Dim 2"
    assert axes[2].spacing is None


def test_philips_rec_axis_labels_cover_the_canonical_layout():
    axes = _labeled_axes((64, 64, 20, 3), labels=_PHILIPS_REC_AXIS_LABELS)

    assert tuple(axis.label for axis in axes) == ("X", "Y", "Slice", "Echo")


def test_axes_matching_shape_trims_trailing_axes_or_discards():
    axes = default_axes((4, 5, 1, 1))

    assert _axes_matching_shape(axes, (4, 5)) == axes[:2]
    assert _axes_matching_shape(axes, (4, 6)) is None
    assert _axes_matching_shape(axes, (4, 5, 1, 1, 2)) is None
    assert _axes_matching_shape(None, (4, 5)) is None


def test_stacked_axes_appends_labeled_stacking_axis():
    base = default_axes((4, 5))

    axes = _stacked_axes(base, (4, 5, 3), "echoes")

    assert len(axes) == 3
    assert axes[:2] == base
    assert axes[2].label == "Echoes"
    assert axes[2].size == 3


def test_nifti_loader_extracts_spacing_and_units(tmp_path):
    nib = pytest.importorskip("nibabel")

    data = np.zeros((4, 5, 6), dtype=np.float32)
    affine = np.diag((2.0, 1.5, 3.0, 1.0))
    image = nib.Nifti1Image(data, affine)
    image.header.set_zooms((2.0, 1.5, 3.0))
    image.header.set_xyzt_units(xyz="mm")
    path = tmp_path / "scan.nii"
    nib.save(image, path)

    loaded = load_path(path)

    assert loaded.axes is not None
    assert tuple(axis.spacing for axis in loaded.axes) == (2.0, 1.5, 3.0)
    assert tuple(axis.unit for axis in loaded.axes) == ("mm", "mm", "mm")
    assert tuple(axis.size for axis in loaded.axes) == loaded.data.shape


def test_nifti_loader_narrows_to_float32_and_applies_scaling(tmp_path):
    nib = pytest.importorskip("nibabel")

    voxels = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    image = nib.Nifti1Image(voxels, np.eye(4))
    image.header.set_data_dtype(np.int16)
    image.header["scl_slope"] = 123.25
    image.header["scl_inter"] = 4.5
    path = tmp_path / "scaled.nii"
    nib.save(image, path)

    loaded = load_path(path)

    assert loaded.data.dtype == np.float32
    expected = voxels.astype(np.float64) * 123.25 + 4.5
    np.testing.assert_allclose(loaded.data, expected, rtol=1e-6)
