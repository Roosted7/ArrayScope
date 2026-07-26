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


def payload_histogram_display_source(payload):
    """Histogram source used to POPULATE a montage histogram from payloads.

    Prefers the dedicated histogram arrays (``payload_histogram_source``).  When
    a commit carries none — the histogram plot data is derived from level stats
    and may not be published on every backend/commit — fall back to the payload's
    real-valued semantic pixels (``semantic_data``).  ``semantic_data`` is
    retained unchanged across display-LOD level swaps (ADR 0050), so the
    histogram stays tied to semantic content, never to the reduced display plane
    in ``payload.image``.

    The fallback is skipped for complex ``semantic_data``: a complex plane's
    histogram source is its magnitude, which is carried explicitly in the
    dedicated histogram arrays.  Falling back to the raw complex plane would feed
    un-histogrammable values downstream, so a complex payload without a dedicated
    histogram array contributes nothing here (unchanged pre-fix behaviour).
    """

    source = payload_histogram_source(payload)
    if source is None:
        semantic = getattr(payload, "semantic_data", None)
        if semantic is not None and not np.iscomplexobj(semantic):
            source = semantic
    return source


def histogram_data_from_tile_payloads(payloads):
    """Concatenate the semantic histogram source arrays of a tiled commit.

    Backend-agnostic source of truth for the montage histogram: the histogram
    is built from the committed tile PAYLOADS, never from a single bound
    ImageItem (a tiled montage has none). Both maintained views
    feed their histogram from this.
    """

    parts = []
    for payload in dict(payloads or {}).values():
        source = payload_histogram_display_source(payload)
        if source is None:
            continue
        parts.append(np.asarray(source))
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return np.concatenate([np.ravel(part) for part in parts])


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
        int(tile_delta.histogram_revision),
        id(histogram_plot_data),
        semantic_identity,
        None if source is None else tuple(int(value) for value in source.shape),
        None if source is None else str(source.dtype),
        None if source is None else tuple(int(value) for value in source.strides),
        (float(histogram_range[0]), float(histogram_range[1])),
    )


__all__ = [
    "histogram_data_from_tile_payloads",
    "payload_histogram_display_source",
    "payload_histogram_source",
    "tiled_histogram_key",
    "tiled_semantic_histogram_identity",
]
