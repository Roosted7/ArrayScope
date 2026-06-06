import numpy as np

from arrayscope.io.numpy_save import default_numpy_filename, selected_numpy_data


def test_default_numpy_filename_uses_source_stem_and_nii_gz_suffix():
    assert default_numpy_filename(None) == "arrayscope.npy"
    assert default_numpy_filename("/tmp/source.npy") == "source.npy"
    assert default_numpy_filename("/tmp/scan.nii.gz") == "scan.npy"


def test_selected_numpy_data_applies_ranges_and_optional_squeeze():
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    squeezed = selected_numpy_data(data, [(1, 2), (0, 2), (1, 4)], squeeze=True)
    unsqueezed = selected_numpy_data(data, [(1, 2), (0, 2), (1, 4)], squeeze=False)

    np.testing.assert_array_equal(squeezed, data[1:2, 0:2, 1:4].squeeze())
    assert squeezed.shape == (2, 3)
    np.testing.assert_array_equal(unsqueezed, data[1:2, 0:2, 1:4])
    assert unsqueezed.shape == (1, 2, 3)
