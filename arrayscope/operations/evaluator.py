"""Evaluation and display cache for operation-backed documents."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arrayscope.operations.pipeline import ArrayDocument
from arrayscope.core.cache_status import (
    CacheStatusSnapshot,
    cache_status_computing,
    cache_status_error,
    cache_status_for_hit,
    cache_status_ready,
    CacheStatus,
)
from arrayscope.display.slice_engine import make_image, make_line


@dataclass
class OperationEvaluator:
    document: ArrayDocument
    _derived_key: tuple | None = None
    _derived_data: object | None = None
    _image_key: tuple | None = None
    _image_result: object | None = None
    _line_key: tuple | None = None
    _line_result: object | None = None
    derived_evaluations: int = 0
    image_evaluations: int = 0
    line_evaluations: int = 0
    display_generation: int = 0
    last_status: CacheStatusSnapshot = CacheStatusSnapshot(CacheStatus.COLD, "No evaluation yet")

    def set_document(self, document: ArrayDocument):
        if document.operations != self.document.operations or document.base_data is not self.document.base_data:
            self.document = document
            self.clear_cache()
        else:
            self.document = document

    def clear_cache(self):
        self._derived_key = None
        self._derived_data = None
        self._image_key = None
        self._image_result = None
        self._line_key = None
        self._line_result = None
        self.display_generation += 1
        self.last_status = CacheStatusSnapshot(CacheStatus.STALE, "Cache cleared")

    def current_data(self):
        key = _document_key(self.document)
        if self._derived_key == key:
            self.last_status = cache_status_for_hit(True)
            return self._derived_data

        self.last_status = cache_status_computing("Evaluating derived array")
        try:
            self._derived_data = self.document.materialize()
            self._derived_key = key
            self.derived_evaluations += 1
            self.last_status = cache_status_ready("Derived array cached")
            return self._derived_data
        except Exception as exc:
            self.last_status = cache_status_error(exc)
            raise

    def image(self, view_state, colormap_lut=None):
        key = (_document_key(self.document), view_state, _lut_key(colormap_lut))
        if self._image_key == key:
            self.last_status = cache_status_for_hit(True)
            return self._image_result

        self.last_status = cache_status_computing("Evaluating image view")
        try:
            self._image_result = make_image(self.current_data(), view_state, colormap_lut=colormap_lut)
            self._image_key = key
            self.image_evaluations += 1
            self.last_status = cache_status_ready("Image view cached")
            return self._image_result
        except Exception as exc:
            self.last_status = cache_status_error(exc)
            raise

    def line(self, view_state):
        key = (_document_key(self.document), view_state)
        if self._line_key == key:
            self.last_status = cache_status_for_hit(True)
            return self._line_result

        self.last_status = cache_status_computing("Evaluating profile")
        try:
            self._line_result = make_line(self.current_data(), view_state)
            self._line_key = key
            self.line_evaluations += 1
            self.last_status = cache_status_ready("Profile cached")
            return self._line_result
        except Exception as exc:
            self.last_status = cache_status_error(exc)
            raise


def _document_key(document: ArrayDocument):
    return (id(document.base_data), document.operations)


def _lut_key(colormap_lut):
    if colormap_lut is None:
        return None
    lut = np.asarray(colormap_lut)
    return (lut.shape, str(lut.dtype), lut.tobytes())
