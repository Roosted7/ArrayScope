"""G7 host-cache AUTO: the two-level compressed cache wired into the live evaluator.

These tests prove the three things the wiring must guarantee:

1. **Correctness (bit-identity).**  A payload put into a tier-backed cache and
   later recovered from the compressed tier (after a raw eviction) is bit-for-bit
   identical to the input -- for float32/complex64/int16, through the dataclass
   payload adapter and the aggressive per-dtype AUTO codec.
2. **The tier recovers by DECODE, not recompute.**  A raw-miss/tier-hit must not
   call the compute path (spied).
3. **AUTO changes nothing observable.**  The evaluator's outputs are identical
   with codec RAW vs AUTO -- only the cache internals differ.

Plus a live-engagement check: driving the real evaluator over more distinct
slices than the (shrunk) raw budget holds, then revisiting them, actually
engages the tier (``tier_recoveries > 0``) and serves revisits without a
recompute.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.core.cache_status import CacheStatus
from arrayscope.core.view_state import ViewState
from arrayscope.display.slice_engine import DisplayImage
from arrayscope.operations.compressed_tier import (
    TwoLevelArrayCache,
    split_payload_for_tier,
)
from arrayscope.operations.evaluator import OperationEvaluator, _build_array_cache
from arrayscope.operations.pipeline import ArrayDocument


def _image(dtype, i, shape=(96, 96)) -> np.ndarray:
    rng = np.random.default_rng(100 + i)
    dtype = np.dtype(dtype)
    if dtype == np.complex64:
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    if dtype == np.int16:
        return (rng.integers(-2000, 2000, size=shape) + i).astype(np.int16)
    return (rng.standard_normal(shape) * 100.0 + i).astype(dtype)


def _payload(dtype, i) -> DisplayImage:
    return DisplayImage(
        data=_image(dtype, i),
        histogram_data=np.linspace(0, 1, 64).astype(np.float32),
        default_levels=(0.0, 1.0),
    )


# ---------------------------------------------------------------------------
# 1 + 2: bit-identity through the tier + recover-by-decode, under AUTO.
#
# Raw eviction is forced by entry count (max_entries=2) with a roomy tier byte
# budget, so recovery is guaranteed independent of how well the (deliberately
# incompressible, random) data compresses -- the point here is losslessness.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.complex64, np.int16])
def test_payload_recovered_from_tier_is_bit_identical_auto(dtype):
    """A DisplayImage evicted from raw and recovered from the AUTO tier is exact."""

    payload = _payload(dtype, 0)
    # Adapter must accept the dataclass payload and pick the image as primary.
    parts = split_payload_for_tier(payload)
    assert parts is not None
    primary, _meta, aux_nbytes = parts
    assert primary is payload.data
    assert aux_nbytes == payload.histogram_data.nbytes

    cache = _build_array_cache(64 << 20, 2, "auto")  # raw holds 2 entries; tier roomy
    assert isinstance(cache, TwoLevelArrayCache)

    for i in range(6):
        cache.put((i,), _payload(dtype, i))  # entry 0 evicted into the tier

    recovered = cache.get((0,))
    assert recovered is not None
    assert isinstance(recovered, DisplayImage)
    # Lossless: bit-for-bit identical image AND verbatim auxiliary/metadata.
    assert recovered.data.dtype == np.dtype(dtype)
    assert np.array_equal(recovered.data, _payload(dtype, 0).data)
    assert np.array_equal(recovered.histogram_data, payload.histogram_data)
    assert recovered.default_levels == (0.0, 1.0)
    assert cache.tier_recoveries >= 1


def test_tier_recovery_does_not_recompute_auto():
    """The raw-miss/tier-hit path serves a decode, never the compute callback."""

    cache = _build_array_cache(64 << 20, 2, "auto")

    computed: list[int] = []

    def loader(i):
        def compute():
            computed.append(i)
            return _payload(np.float32, i)

        return compute

    for i in range(6):
        cache.get_or_compute((i,), loader(i))
    assert computed == list(range(6))

    computed.clear()
    value, hit = cache.get_or_compute((0,), loader(0))
    assert hit is True
    assert computed == [], "tier recovery must not recompute"
    assert np.array_equal(value.data, _payload(np.float32, 0).data)
    assert cache.tier_recoveries >= 1


def test_raw_codec_builds_plain_cache_byte_identical():
    """RAW yields a plain BoundedArrayCache: evicted entries are simply gone."""

    from arrayscope.operations.cache import BoundedArrayCache

    cache = _build_array_cache(64 << 20, 2, "raw")
    assert isinstance(cache, BoundedArrayCache)
    for i in range(6):
        cache.put((i,), _payload(np.float32, i))
    assert cache.get((0,)) is None  # no tier: evicted and gone


def test_diagnostics_surface_tier_counters():
    """The drop-in diagnostics snapshot exposes tier engagement + recoveries."""

    cache = _build_array_cache(64 << 20, 2, "auto")
    for i in range(6):
        cache.put((i,), _payload(np.float32, i))
    cache.get((0,))  # a tier recovery

    diag = cache.diagnostics(CacheStatus.READY, "hi")
    assert diag.tier_engaged is True
    assert diag.tier_codec == "auto"
    assert diag.tier_recoveries >= 1
    assert diag.tier_entries >= 1

    # RAW cache reports the plain snapshot (tier fields default-off).
    raw_cache = _build_array_cache(64 << 20, 2, "raw")
    raw_diag = raw_cache.diagnostics(CacheStatus.READY, "hi")
    assert raw_diag.tier_engaged is False
    assert raw_diag.tier_recoveries == 0


def test_unrecognized_value_declines_tier_storage():
    """A non-array, non-dataclass value is not tier-stored (evicts as today)."""

    assert split_payload_for_tier("not an array") is None
    assert split_payload_for_tier(42) is None


# ---------------------------------------------------------------------------
# 3: AUTO must not change any settled evaluator output vs RAW.
# ---------------------------------------------------------------------------


def _gradient_volume(n_slices=12, shape=(64, 64)) -> np.ndarray:
    """A smooth, compressible volume -- representative imaging data.

    Structured (a per-slice-shifted gradient) so the lossless codec compresses
    it well and the tier retains many more slices than the raw cache, which is
    exactly when the tier earns its keep.
    """

    ramp = np.linspace(0.0, 1.0, shape[1], dtype=np.float32)
    plane = np.outer(np.ones(shape[0], np.float32), ramp)
    return np.stack([(plane + z * 0.03) * 500.0 for z in range(n_slices)]).astype(np.float32)


def _memory_policy(display_bytes, profile_bytes=64 << 20, stage_bytes=64 << 20):
    return SimpleNamespace(
        display_cache_budget_bytes=int(display_bytes),
        profile_cache_budget_bytes=int(profile_bytes),
        stage_cache_budget_bytes=int(stage_bytes),
    )


def _slice_states(document, n):
    return [ViewState.from_shape(document.current_shape).with_slice(0, z) for z in range(n)]


def _two_tile_budget(document, state) -> int:
    """Byte budget that holds ~2 display tiles (probe one; the tile is downsampled)."""

    from arrayscope.operations.cache import _nbytes

    probe = OperationEvaluator(document, chunk_transport_codec="raw")
    tile_bytes = int(_nbytes(probe.image(state).data))
    return max(4096, tile_bytes * 2)


def test_evaluator_outputs_identical_raw_vs_auto():
    """Same document + requests: AUTO's outputs are bit-identical to RAW's."""

    document = ArrayDocument(_gradient_volume(8))
    states = _slice_states(document, 8)

    raw_eval = OperationEvaluator(document, chunk_transport_codec="raw")
    auto_eval = OperationEvaluator(document, chunk_transport_codec="auto")
    # Shrink so the raw cache overflows and (under AUTO) the tier engages.
    tiny = _two_tile_budget(document, states[0])
    raw_eval.apply_memory_policy(_memory_policy(display_bytes=tiny))
    auto_eval.apply_memory_policy(_memory_policy(display_bytes=tiny))

    # Pass 1 fills; pass 2 revisits in reverse (a scroll-back) so recently
    # evicted, still-resident tier entries are recovered under AUTO.
    for order in (states, list(reversed(states))):
        for state in order:
            r = np.asarray(raw_eval.image(state).data)
            a = np.asarray(auto_eval.image(state).data)
            assert np.array_equal(r, a), "AUTO must not change the settled output"

    assert auto_eval.display_cache_diagnostics().tier_recoveries >= 1


def test_evaluator_tier_engages_live_under_auto():
    """A real revisiting workload engages the tier (recoveries > 0), no recompute."""

    document = ArrayDocument(_gradient_volume(12))
    states = _slice_states(document, 12)
    reference = OperationEvaluator(document, chunk_transport_codec="raw")
    evaluator = OperationEvaluator(document, chunk_transport_codec="auto")
    # Raw cache holds ~2 slices; 12 distinct slices force eviction into the tier.
    tiny = _two_tile_budget(document, states[0])
    evaluator.apply_memory_policy(_memory_policy(display_bytes=tiny))
    for state in states:
        evaluator.image(state)  # pass 1: fill (each computed once; older evict)

    evals_after_fill = evaluator.image_evaluations
    recoveries_before = evaluator.display_cache_diagnostics().tier_recoveries

    # Pass 2: revisit in reverse (a scroll-back).  Slices retained in the
    # compressed tier are served by decode (no recompute); their output is
    # bit-identical to a plain RAW-codec evaluator's output for the same request.
    for state in reversed(states):
        got = evaluator.image(state)
        assert np.array_equal(np.asarray(got.data), np.asarray(reference.image(state).data))

    diag = evaluator.display_cache_diagnostics()
    assert diag.tier_engaged is True
    assert diag.tier_recoveries > recoveries_before, "tier must engage live"
    assert diag.tier_entries >= 1
    # Not every revisit recomputed: the tier (and the 2-slot raw cache) served
    # some revisits without a fresh evaluation.
    recomputes_on_pass2 = evaluator.image_evaluations - evals_after_fill
    assert recomputes_on_pass2 < len(states)
