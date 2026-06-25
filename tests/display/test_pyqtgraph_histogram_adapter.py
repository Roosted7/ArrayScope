import os
import re

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


def test_installed_pyqtgraph_histogram_lut_api_shape(qt_app):
    from arrayscope.display.backends.pyqtgraph.histogram_adapter import installed_histogram_lut_api_facts

    facts = installed_histogram_lut_api_facts()

    assert _version_tuple(facts.pyqtgraph_version) >= (0, 14, 0)
    assert facts.widget_has_set_image_item is True
    assert facts.item_has_image_item is True
    assert facts.item_has_lookup_table_refresh is True
    assert facts.item_has_region_changed is True
    assert facts.item_has_image_changed is True


def test_histogram_adapter_new_binding_connects_once_and_disconnects_auto_recompute(qt_app, monkeypatch):
    import pyqtgraph as pg
    from pyqtgraph.graphicsItems.ImageItem import ImageItem

    from arrayscope.display.backends.pyqtgraph.histogram_adapter import PyQtGraphHistogramAdapter

    widget = pg.HistogramLUTWidget()
    adapter = PyQtGraphHistogramAdapter(widget)
    item = ImageItem(axisOrder="row-major")
    calls = []
    original = widget.setImageItem

    def recording_set_image_item(image_item):
        calls.append(image_item)
        return original(image_item)

    monkeypatch.setattr(widget, "setImageItem", recording_set_image_item)

    facts = adapter.bind_image_item(item)

    assert calls == [item]
    assert facts.public_set_image_item_calls == 1
    assert facts.sig_image_changed_disconnects >= 1

    item.setImage(np.ones((4, 4), dtype=float), autoLevels=False)

    assert facts.sig_image_changed_disconnects >= 1


def test_histogram_adapter_rebinding_known_item_uses_private_adapter_path(qt_app, monkeypatch):
    import pyqtgraph as pg
    from pyqtgraph.graphicsItems.ImageItem import ImageItem

    from arrayscope.display.backends.pyqtgraph.histogram_adapter import PyQtGraphHistogramAdapter

    widget = pg.HistogramLUTWidget()
    adapter = PyQtGraphHistogramAdapter(widget)
    first = ImageItem(axisOrder="row-major")
    second = ImageItem(axisOrder="row-major")
    set_calls = []
    original = widget.setImageItem

    def recording_set_image_item(image_item):
        set_calls.append(image_item)
        return original(image_item)

    monkeypatch.setattr(widget, "setImageItem", recording_set_image_item)

    adapter.bind_image_item(first)
    adapter.bind_image_item(second)
    facts = adapter.bind_image_item(first)

    assert set_calls == [first, second]
    assert facts.public_set_image_item_calls == 0
    assert facts.private_rebind_calls == 1
    assert facts.lookup_table_refreshes == 1
    assert facts.region_changed_calls == 1
    assert widget.item.imageItem() is first


def test_histogram_adapter_repeated_commits_do_not_multiply_sig_image_changed_callbacks(qt_app):
    import pyqtgraph as pg
    from pyqtgraph.graphicsItems.ImageItem import ImageItem

    from arrayscope.display.backends.pyqtgraph.histogram_adapter import PyQtGraphHistogramAdapter

    widget = pg.HistogramLUTWidget()
    adapter = PyQtGraphHistogramAdapter(widget)
    item = ImageItem(axisOrder="row-major")
    calls = []

    def recording_image_changed(*args, **kwargs):
        calls.append((args, kwargs))

    widget.item.imageChanged = recording_image_changed

    probe = ImageItem(axisOrder="row-major")
    probe.sigImageChanged.connect(widget.item.imageChanged)
    probe.setImage(np.ones((4, 4), dtype=float), autoLevels=False)
    assert calls
    probe.sigImageChanged.disconnect(widget.item.imageChanged)
    calls.clear()

    for _ in range(4):
        facts = adapter.bind_image_item(item)
        calls.clear()
        item.setImage(np.ones((4, 4), dtype=float), autoLevels=False)

        if facts.item_changed:
            assert facts.sig_image_changed_disconnects >= 1
        else:
            assert facts.sig_image_changed_disconnects == 0
        assert calls == []
