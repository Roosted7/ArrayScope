"""Evaluation and display cache for operation-backed documents."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .operation_pipeline import ArrayDocument
from .slice_engine import make_image, make_line


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

    def current_data(self):
        key = _document_key(self.document)
        if self._derived_key == key:
            return self._derived_data

        self._derived_data = self.document.materialize()
        self._derived_key = key
        self.derived_evaluations += 1
        return self._derived_data

    def image(self, view_state, colormap_lut=None):
        key = (_document_key(self.document), view_state, _lut_key(colormap_lut))
        if self._image_key == key:
            return self._image_result

        self._image_result = make_image(self.current_data(), view_state, colormap_lut=colormap_lut)
        self._image_key = key
        self.image_evaluations += 1
        return self._image_result

    def line(self, view_state):
        key = (_document_key(self.document), view_state)
        if self._line_key == key:
            return self._line_result

        self._line_result = make_line(self.current_data(), view_state)
        self._line_key = key
        self.line_evaluations += 1
        return self._line_result


def _document_key(document: ArrayDocument):
    return (id(document.base_data), document.operations)


def _lut_key(colormap_lut):
    if colormap_lut is None:
        return None
    lut = np.asarray(colormap_lut)
    return (lut.shape, str(lut.dtype), lut.tobytes())
