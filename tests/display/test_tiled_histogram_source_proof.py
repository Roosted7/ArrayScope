"""What counts as proof that a maintained histogram slice is still current.

The montage histogram source is maintained across commits: a bounded delta
rewrites only the slices whose tile it committed, and skips a slice when the
tile's source array is *the same array* as the one already written there.

That skip is the whole optimization, and it is only sound if "the same array"
is established by identity rather than inferred.  The first implementation
stored ``id(source)`` and compared later sources against that integer.  Nothing
held the original array alive, so once it was freed CPython could hand its
address to a different ndarray, the comparison would succeed, and a real
content change would be silently skipped — a wrong histogram, not a slow one.

These tests pin the proof rule directly.  None of them depends on
probabilistically forcing an address reuse: each constructs the state the rule
must handle and asserts what the rule does with it.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np

from arrayscope.display.model.frame import DisplayTilePayload
from arrayscope.display.model.tiled_histogram_identity import (
    histogram_plot_source_and_layout,
    patched_histogram_plot_source,
)

TILE = 8


def _payload(tile: int, source: np.ndarray) -> DisplayTilePayload:
    return DisplayTilePayload(
        tile_number=tile,
        source_index=tile,
        image=source,
        histogram_data=None,
        source_id=("tile", tile),
        semantic_data=source,
    )


def _source(tile: int, generation: int) -> np.ndarray:
    # Distinct per pixel, per tile and per generation, so a slice written from
    # the wrong array or at the wrong offset cannot compare equal by accident.
    return (
        float(tile) * 1e5
        + float(generation) * 1e3
        + np.arange(TILE * TILE, dtype=np.float32).reshape(TILE, TILE)
    ).astype(np.float32)


def _montage(tiles: int = 6, generation: int = 0):
    sources = {tile: _source(tile, generation) for tile in range(tiles)}
    payloads = {tile: _payload(tile, source) for tile, source in sources.items()}
    return payloads, sources


def _rebuilt(payloads):
    return histogram_plot_source_and_layout(payloads)[0]


def test_the_same_source_object_reuses_its_slice():
    """An unchanged array is provably unchanged, so nothing is rewritten."""

    payloads, _sources = _montage()
    previous, layout = histogram_plot_source_and_layout(payloads)

    patched, patched_layout = patched_histogram_plot_source(
        previous, layout, payloads, upserts=(2,), removals=()
    )

    assert patched is previous, (
        "re-presenting the very same source array produced a new buffer; the "
        "identity proof did not recognise it"
    )
    assert patched_layout is layout


def test_a_distinct_source_patches_even_when_a_numeric_id_is_forged():
    """A bare ``id()`` is not proof, whatever integer it happens to equal.

    This is the hole the weak reference closes, expressed without needing an
    address collision: store the exact integer the old implementation would
    have stored and compared equal, and require the patch to happen anyway.
    """

    payloads, _sources = _montage()
    previous, (population, slices) = histogram_plot_source_and_layout(payloads)

    replacement = _source(2, generation=1)
    payloads[2] = _payload(2, replacement)
    # Exactly what the old representation held, and it now compares equal to
    # the live source's id — the collision the weak reference makes impossible.
    _proof, start, stop = slices[2]
    forged = dict(slices)
    forged[2] = (id(replacement), start, stop)

    patched, _layout = patched_histogram_plot_source(
        previous, (population, forged), payloads, upserts=(2,), removals=()
    )

    assert patched is not None, (
        "the delta was refused outright, so the proof was never consulted and "
        "this test cannot say what a forged id would have done"
    )
    assert patched is not previous, (
        "a forged numeric id was accepted as proof; a real content change "
        "would be skipped and the histogram would be stale"
    )
    assert np.array_equal(patched, _rebuilt(payloads), equal_nan=True)


def test_a_dead_weak_reference_forces_the_slice_to_be_rewritten():
    """When the previous array is gone, nothing can prove sameness."""

    payloads, _sources = _montage()
    previous, (population, slices) = histogram_plot_source_and_layout(payloads)

    doomed = np.zeros((TILE, TILE), dtype=np.float32)
    dead = weakref.ref(doomed)
    del doomed
    gc.collect()
    assert dead() is None, "fixture failed to kill the referent"

    replacement = _source(3, generation=1)
    payloads[3] = _payload(3, replacement)
    _proof, start, stop = slices[3]
    with_dead_proof = dict(slices)
    with_dead_proof[3] = (dead, start, stop)

    patched, _layout = patched_histogram_plot_source(
        previous, (population, with_dead_proof), payloads, upserts=(3,), removals=()
    )

    assert patched is not None, (
        "the delta was refused outright, so the dead reference was never "
        "consulted and this test cannot say how it was treated"
    )
    assert patched is not previous, (
        "an unprovable slice was left untouched; a freed previous source is "
        "the absence of evidence, not evidence of sameness"
    )
    assert np.array_equal(patched, _rebuilt(payloads), equal_nan=True)


def test_the_maintained_source_equals_a_full_reconstruction_after_replacement():
    """Whatever path a replacement takes, the values must be the rebuild's."""

    payloads, _sources = _montage(tiles=6)
    maintained, layout = histogram_plot_source_and_layout(payloads)
    assert np.array_equal(maintained, _rebuilt(payloads), equal_nan=True)

    for generation, tiles in enumerate(((0,), (5,), (1, 2), (0, 1, 2, 3, 4, 5)), start=1):
        for tile in tiles:
            payloads[tile] = _payload(tile, _source(tile, generation))
        maintained, layout = patched_histogram_plot_source(
            maintained, layout, payloads, upserts=tiles, removals=()
        )
        assert maintained is not None, f"replacing {tiles} refused a patch it could serve"
        assert np.array_equal(maintained, _rebuilt(payloads), equal_nan=True), (
            f"after replacing {tiles} the maintained source diverged from a full reconstruction"
        )


def test_the_proof_does_not_keep_payload_pixels_alive():
    """The optimization must not be the reason an array cannot be freed."""

    payloads, sources = _montage(tiles=6)
    _previous, layout = histogram_plot_source_and_layout(payloads)
    watch = weakref.ref(sources[4])

    del payloads[4]
    del sources[4]
    gc.collect()

    assert watch() is None, (
        "the layout kept a dropped tile's histogram source alive; the identity "
        "proof must be weak, or maintaining the buffer retains the montage"
    )
    assert layout is not None
