"""PyQtGraph HistogramLUTItem binding mechanics.

This module is the only ArrayScope owner of PyQtGraph's HistogramLUTWidget
rebinding internals.  Image views ask for a binding and receive diagnostic facts;
they do not patch HistogramLUTItem state directly.
"""

from __future__ import annotations

import warnings
import weakref
from dataclasses import dataclass
from time import perf_counter

import pyqtgraph as pg


@dataclass(frozen=True)
class HistogramBindingFacts:
    elapsed_ms: float
    item_changed: bool
    public_set_image_item_calls: int = 0
    private_rebind_calls: int = 0
    lookup_table_refreshes: int = 0
    region_changed_calls: int = 0
    sig_image_changed_disconnects: int = 0


@dataclass(frozen=True)
class HistogramLUTApiFacts:
    pyqtgraph_version: str
    widget_has_set_image_item: bool
    item_has_image_item: bool
    item_has_lookup_table_refresh: bool
    item_has_region_changed: bool
    item_has_image_changed: bool


class PyQtGraphHistogramAdapter:
    """Bind ImageItems to a HistogramLUTWidget without duplicate recomputes."""

    def __init__(self, histogram_widget):
        self.histogram_widget = histogram_widget
        self._bound_item = None
        self._known_item_ids: set[int] = set()

    @property
    def bound_item(self):
        return self._bound_item

    def is_bound_item(self, item) -> bool:
        return item is not None and self._bound_item is item

    def bind_image_item(self, item) -> HistogramBindingFacts:
        start = perf_counter()
        if item is None or self._bound_item is item:
            return HistogramBindingFacts(
                elapsed_ms=(perf_counter() - start) * 1000.0, item_changed=False
            )

        public_calls = 0
        private_calls = 0
        lookup_calls = 0
        region_calls = 0
        item_id = id(item)
        if item_id not in self._known_item_ids:
            self.histogram_widget.setImageItem(item)
            self._known_item_ids.add(item_id)
            public_calls = 1
        else:
            hist_item = self.histogram_widget.item
            hist_item.imageItem = weakref.ref(item)
            private_calls = 1
            if hasattr(hist_item, "_setImageLookupTable"):
                hist_item._setImageLookupTable()
                lookup_calls = 1
            hist_item.regionChanged()
            region_calls = 1

        disconnects = self.disconnect_automatic_image_changed(item)
        self._bound_item = item
        return HistogramBindingFacts(
            elapsed_ms=(perf_counter() - start) * 1000.0,
            item_changed=True,
            public_set_image_item_calls=public_calls,
            private_rebind_calls=private_calls,
            lookup_table_refreshes=lookup_calls,
            region_changed_calls=region_calls,
            sig_image_changed_disconnects=disconnects,
        )

    def disconnect_automatic_image_changed(self, item) -> int:
        """Disconnect PyQtGraph's automatic histogram recompute callbacks."""

        signal = getattr(item, "sigImageChanged", None)
        if signal is None:
            return 0
        slot = self.histogram_widget.item.imageChanged
        disconnects = 0
        for _ in range(32):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message=".*Failed to disconnect.*", category=RuntimeWarning
                    )
                    signal.disconnect(slot)
            except (TypeError, RuntimeError):
                break
            else:
                disconnects += 1
        return disconnects


def installed_histogram_lut_api_facts(histogram_widget=None) -> HistogramLUTApiFacts:
    widget = histogram_widget if histogram_widget is not None else pg.HistogramLUTWidget()
    item = widget.item
    return HistogramLUTApiFacts(
        pyqtgraph_version=str(getattr(pg, "__version__", "")),
        widget_has_set_image_item=hasattr(widget, "setImageItem"),
        item_has_image_item=hasattr(item, "imageItem"),
        item_has_lookup_table_refresh=hasattr(item, "_setImageLookupTable"),
        item_has_region_changed=hasattr(item, "regionChanged"),
        item_has_image_changed=hasattr(item, "imageChanged"),
    )
