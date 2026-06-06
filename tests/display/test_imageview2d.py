import os

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")


def test_profile_marker_callback_replacement_and_programmatic_move(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    calls = []

    view.set_profile_marker_callback(lambda x, y: calls.append(("first", x, y)))
    view.set_profile_marker_callback(lambda x, y: calls.append(("second", x, y)))

    view.setProfileMarker(1, 2, visible=True)
    assert calls == []

    view._profile_vline.setValue(3)
    assert len(calls) == 1
    assert calls[0][0] == "second"

    view.clear_profile_marker_callback()
    view._profile_vline.setValue(4)
    assert len(calls) == 1
    view.close()
