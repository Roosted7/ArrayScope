"""Qt-free semantic identity of the tiled histogram stream (ADR 0050).

Histogram/level work is driven by SEMANTIC tile content, never by texture or
display-LOD identity.  ``payload.image`` may be a reduced display plane;
``payload.semantic_data`` and the semantic histogram source are retained
unchanged across display-LOD level swaps, so a level swap leaves this identity
— and therefore the histogram stream — untouched, while a real content change
(new evaluation, slice change) replaces the arrays and refreshes it.
"""

from __future__ import annotations

import weakref

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


def _source_proof(source):
    """A weak, checkable claim that a later array IS this one.

    ``id()`` alone is not a proof of anything once the array it named can
    die: CPython reuses addresses, so a historical id can compare equal to a
    DIFFERENT array and a real content change would be skipped.  A weak
    reference is falsifiable — it either still yields this exact object or it
    yields ``None`` — and it holds no pixels alive, so proving the
    optimization never keeps a payload from being freed.

    ``None`` means no proof is obtainable; the caller must then treat the
    source as changed.
    """

    try:
        return weakref.ref(source)
    except TypeError:  # pragma: no cover - ndarrays are weakref-able
        return None


def _proves_same_source(proof, source) -> bool:
    """Whether ``proof`` establishes that ``source`` is the very same array.

    Anything that is not a live reference yielding this exact object is not a
    proof — including a bare ``id()`` value, which is what this replaced.
    Refusing to interpret it keeps a stale layout from silently degrading back
    into address comparison.
    """

    return callable(proof) and proof() is source


def _histogram_parts(payloads):
    """Per-tile histogram sources in build order, with their flat placement."""

    payload_map = dict(payloads or {})
    parts = []
    slices = {}
    offset = 0
    for tile, payload in payload_map.items():
        source = payload_histogram_display_source(payload)
        if source is None:
            continue
        source = np.asarray(source)
        parts.append(source)
        slices[int(tile)] = (_source_proof(source), offset, offset + source.size)
        offset += source.size
    population = tuple(int(tile) for tile in payload_map)
    return parts, slices, population


def histogram_data_from_tile_payloads(payloads):
    """Concatenate the semantic histogram source arrays of a tiled commit.

    Backend-agnostic source of truth for the montage histogram: the histogram
    is built from the committed tile PAYLOADS, never from a single bound
    ImageItem (a tiled montage has none). Both maintained views
    feed their histogram from this.
    """

    parts, _slices, _population = _histogram_parts(payloads)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return np.concatenate([np.ravel(part) for part in parts])


def histogram_plot_source_and_layout(payloads):
    """The montage histogram source, laid out as the image the plot draws.

    Returns ``(source, layout)``.  ``layout`` is ``(population, slices)``,
    where ``population`` is the payload key order this buffer was built from
    and ``slices`` maps a contributing tile to its ``(source_proof, start,
    stop)`` flat placement — or ``None`` when the result is not a
    concatenation this module laid out (no contributing tile, or the
    single-part passthrough below), and therefore cannot be patched later.

    The square layout is applied HERE rather than by the consumer.  Deriving
    it separately meant a second full-buffer pass on every commit, on top of
    the one that produced the buffer; doing it here lets both the build and
    the per-delta patch write the array the plot item actually receives.
    """

    parts, slices, population = _histogram_parts(payloads)
    if not parts:
        return None, None
    if len(parts) == 1:
        # Passthrough: the caller receives the tile's own array, not a copy,
        # so there is no concatenated buffer to patch later.
        return parts[0], None
    return _square_padded(np.concatenate([np.ravel(part) for part in parts])), (
        population,
        slices,
    )


def _square_padded(data):
    """Pad a flat source into the square image the histogram plot draws.

    The tail is NaN, exactly as the separate padding step produced.
    """

    width = max(1, int(np.ceil(np.sqrt(data.size))))
    height = int(np.ceil(data.size / width))
    padded = np.empty(height * width, dtype=data.dtype)
    padded[: data.size] = data
    if padded.size > data.size:
        padded[data.size :] = np.nan
    return padded.reshape(height, width)


def patched_histogram_plot_source(previous, previous_layout, payloads, *, upserts, removals):
    """Rewrite only the tiles this delta committed.

    Returns ``(None, None)`` when the previous buffer cannot represent the new
    commit — a removal, a tile arriving, a resized or re-typed contribution —
    and the caller rebuilds.  A reusable buffer whose values changed yields a
    NEW array, so a consumer keyed on array identity still sees a changed
    source; a commit that changed no histogram input returns the previous
    array unchanged, so it does not.

    The work here is bounded by the DELTA, not the montage.  Which tiles could
    have changed is what the delta says, so no untouched payload is inspected
    at all, and a refusal is decided before any payload is inspected — the
    earlier version scanned every payload just to discover the population had
    moved, which made a fill more expensive than not trying.

    The one montage-sized step left is comparing the payload key order against
    the population this buffer was built from.  That guard is not optional: a
    delta may change which tiles are ACTIVE without upserting or removing any,
    and reusing the buffer across that would publish a histogram for the wrong
    tile set.  It is an integer tuple comparison, some two orders of magnitude
    below the concatenation it protects.
    """

    if previous is None or previous_layout is None:
        return None, None
    previous_population, previous_slices = previous_layout
    payload_map = dict(payloads or {})
    if removals or tuple(int(tile) for tile in payload_map) != previous_population:
        return None, None
    patched = None
    slices = previous_slices
    for tile in upserts:
        tile = int(tile)
        placement = previous_slices.get(tile)
        payload = payload_map.get(tile)
        if placement is None or payload is None:
            # A tile with no slice never contributed (or is arriving now);
            # either way this buffer cannot represent the new commit.
            return None, None
        source = payload_histogram_display_source(payload)
        if source is None:
            return None, None
        source = np.asarray(source)
        proof, start, stop = placement
        if source.dtype != previous.dtype or source.size != stop - start:
            return None, None
        if _proves_same_source(proof, source):
            # Provably the same array, so this slice already holds its values.
            continue
        # Either a different array, or an unprovable one (the previous source
        # has been freed, so nothing can establish sameness). Both are handled
        # by rewriting the slice from what the payload holds now, which is
        # correct whatever the old contents were.
        if patched is None:
            patched = previous.copy()
            slices = dict(previous_slices)
        # Offsets are flat positions in the concatenation; the padded buffer
        # is C-contiguous, so this reshape is a view onto the same memory.
        patched.reshape(-1)[start:stop] = np.ravel(source)
        slices[tile] = (_source_proof(source), start, stop)
    if patched is None:
        return previous, previous_layout
    return patched, (previous_population, slices)


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
    "histogram_plot_source_and_layout",
    "patched_histogram_plot_source",
    "payload_histogram_display_source",
    "payload_histogram_source",
    "tiled_histogram_key",
    "tiled_semantic_histogram_identity",
]
