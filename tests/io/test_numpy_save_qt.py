import numpy as np

from arrayscope.io import numpy_save_qt
from arrayscope.operations.pipeline import ArrayDocument


def test_collecting_export_preserves_uppercase_path_and_ranges_without_writing(
    tmp_path, monkeypatch
):
    path = tmp_path / "selection.NPY"
    document = ArrayDocument(np.arange(12).reshape(3, 4))

    class RangeDialog:
        def __init__(self, _parent, shape, **kwargs):
            assert tuple(shape) == (3, 4)
            assert kwargs["action_text"] == "Next…"

        def exec(self):
            return numpy_save_qt.QtWidgets.QDialog.DialogCode.Accepted

        def get_ranges(self):
            return [(1, 2), (1, 4)]

        def should_squeeze(self):
            return True

    monkeypatch.setattr(numpy_save_qt, "SaveRangeDialog", RangeDialog)
    monkeypatch.setattr(numpy_save_qt, "get_save_file_name", lambda *_args: (str(path), ""))

    result = numpy_save_qt.collect_numpy_save_request(None, document)

    assert result.path == path
    assert result.document is document
    assert result.ranges == ((1, 2), (1, 4))
    assert result.output_shape == (3,)
    assert not tuple(tmp_path.iterdir())
