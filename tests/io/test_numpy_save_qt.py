import numpy as np

from arrayscope.io import numpy_save_qt


def test_current_array_save_preserves_selected_uppercase_path_and_range(tmp_path, monkeypatch):
    path = tmp_path / "selection.NPY"
    data = np.arange(12).reshape(3, 4)

    class RangeDialog:
        def __init__(self, _parent, _shape):
            pass

        def exec(self):
            return numpy_save_qt.QtWidgets.QDialog.DialogCode.Accepted

        def get_ranges(self):
            return [(1, 2), (1, 4)]

        def should_squeeze(self):
            return True

    monkeypatch.setattr(numpy_save_qt, "SaveRangeDialog", RangeDialog)
    monkeypatch.setattr(numpy_save_qt, "get_save_file_name", lambda *_args: (str(path), ""))

    result = numpy_save_qt.save_current_numpy_file(None, data)

    assert result == (str(path), (3,))
    np.testing.assert_array_equal(np.load(path), data[1, 1:4])
    assert set(tmp_path.iterdir()) == {path}
