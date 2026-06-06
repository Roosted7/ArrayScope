"""Small pure helpers for ROI comparison layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArrayLayer:
    id: str
    label: str
    data: object
    visible: bool = True


@dataclass(frozen=True)
class CompareDocument:
    layers: tuple[ArrayLayer, ...]
    active_layer_id: str

    @classmethod
    def from_base(cls, data, label="Layer 1"):
        layer = ArrayLayer("layer-1", str(label), data)
        return cls((layer,), layer.id)

    def with_layer(self, data, label=None):
        index = len(self.layers) + 1
        layer = ArrayLayer(f"layer-{index}", str(label or f"Layer {index}"), data)
        return CompareDocument(self.layers + (layer,), self.active_layer_id)


def compatible_roi_shape(layer_data, reference_shape):
    return tuple(np.shape(layer_data)[:2]) == tuple(int(value) for value in reference_shape[:2])
