"""Qt-free semantic identity of the tiled histogram stream (ADR 0050).

Histogram/level work is driven by SEMANTIC tile content, never by texture or
display-LOD identity.  ``payload.image`` may be a reduced display plane;
``payload.semantic_data`` and the semantic histogram source are retained
unchanged across display-LOD level swaps, so a level swap leaves this identity
— and therefore the histogram stream — untouched, while a real content change
(new evaluation, slice change) replaces the arrays and refreshes it.
"""

from __future__ import annotations

import numpy as np


def payload_histogram_source(payload):
    """Return the array a tiled commit histograms for one payload."""

    source = getattr(payload, "semantic_histogram_data", None)
    if source is None:
        source = getattr(payload, "histogram_data", None)
    return source


def tiled_semantic_histogram_identity(tile_payloads):
    """Identity of the semantic histogram inputs of a tiled commit."""

    if not tile_payloads:
        return None
    result = []
    for tile, payload in sorted(dict(tile_payloads).items()):
        semantic = getattr(payload, "semantic_data", None)
        if semantic is None:
            semantic = getattr(payload, "image", None)
        result.append((int(tile), id(semantic), id(payload_histogram_source(payload))))
    return tuple(result)


def tiled_histogram_key(histogram_range, *, histogram_plot_data, tile_delta, semantic_identity):
    """Identity of the histogram stream for one tiled presentation commit.

    Texture/source identity is deliberately absent: presentation identity
    churn (display-LOD level swaps, atlas retargets) must never repaint or
    recompute the histogram (ADR 0050).
    """

    if tile_delta is None:
        raise ValueError("tiled histogram identity requires a TilePresentationDelta")
    source = None if histogram_plot_data is None else np.asarray(histogram_plot_data)
    return (
        "revision",
        int(getattr(tile_delta, "histogram_revision")),
        id(histogram_plot_data),
        semantic_identity,
        None if source is None else tuple(int(value) for value in source.shape),
        None if source is None else str(source.dtype),
        None if source is None else tuple(int(value) for value in source.strides),
        (float(histogram_range[0]), float(histogram_range[1])),
    )


__all__ = [
    "payload_histogram_source",
    "tiled_semantic_histogram_identity",
    "tiled_histogram_key",
]
