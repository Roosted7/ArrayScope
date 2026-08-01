import numpy as np

import arrayscope.core.compare as compare


def test_compare_document_adds_compatible_layers():
    document = compare.CompareDocument.from_base(np.zeros((4, 5)), label="base")
    document = document.with_layer(np.ones((4, 5)), label="other")

    assert document.layers[0].label == "base"
    assert document.layers[1].label == "other"
    assert document.active_layer_id == "layer-1"
    assert np.shape(document.layers[1].data) == (4, 5)
