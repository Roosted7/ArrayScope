import numpy as np

from arrayscope.io.file_interpreters import data_file_suffix, load_path


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
