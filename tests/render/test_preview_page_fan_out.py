"""ADR 0064 family 2: the coarse rung's per-tile fan-out as ``(entry, request)``.

Two things need pinning and nothing else does. First, the entry is the old
inline body: given the request the evaluator builds, the pages, histogram and
level evidence must be byte-identical to the composition that used to run at
the call site. Second, that request really does cross a process boundary and
the parent — not the worker — admits its pages.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.core.view_state import ViewState
from arrayscope.display.lod import LodDemand
from arrayscope.display.montage import RenderedTile, make_montage_plan
from arrayscope.display.pyramid import LodPageCache
from arrayscope.kernel.process_backend import ProcessPool
from arrayscope.kernel.remote_call import RemoteCall, remote_dispatch
from arrayscope.kernel.shm_slots import slottable
from arrayscope.operations.evaluator import OperationEvaluator
from arrayscope.operations.pipeline import ArrayDocument, CenteredFFT
from arrayscope.operations.stage_fanin import StageFanInState
from arrayscope.presentation import TileLifecycle
from arrayscope.render import effects


@pytest.fixture(scope="module")
def pool():
    pool = ProcessPool(processes=1).acquire()
    yield pool
    pool.release()


def _demand(level: int) -> LodDemand:
    factor = 2**level
    return LodDemand(
        desired_level=level,
        desired_factor=factor,
        desired_factor_xy=(factor, factor),
        acceptable_levels=(0, 1, 2, 3),
        source_texels_per_pixel_xy=(float(factor), float(factor)),
        reason="test",
    )


def _session(shader_display: bool = True, extent: int = 16):
    """A montage whose montage-axis FFT admits the shared reduced stage."""

    data = np.arange(extent * extent * 4, dtype=np.float32).reshape(extent, extent, 4)
    state = (
        ViewState.from_shape(data.shape)
        .with_image_axes(0, 1)
        .with_montage_axis(2, columns=2, indices=(0, 1, 2, 3), text=":")
    )
    plan = make_montage_plan(
        state,
        axis=2,
        indices=(0, 1, 2, 3),
        tile_shape=(extent, extent),
        columns=2,
        viewport_shape=(200, 200),
    )
    return SimpleNamespace(
        document=ArrayDocument(data, operations=(CenteredFFT(axis=2),)),
        view_state=state,
        plan=plan,
        montage_axis=2,
        colormap_lut=None,
        shader_display=shader_display,
        rgb=False,
        canonical_orientation=False,
        lod_preview_level=2,
        stage_fan_in=StageFanInState(),
        lifecycle=TileLifecycle(),
        rendered_tiles={},
        display_tile_payloads={},
        tile_presentation_state=SimpleNamespace(payloads={}),
        lod_page_cache=None,
        tile_semantic_source_id=lambda source_index: ("semantic", int(source_index)),
    )


def _evaluate(session, tile, *, level: int = 2, capture=None, monkeypatch=None):
    if capture is not None:
        original = effects.run_preview_page_fan_out

        def recording(request):
            capture.append(request)
            return original(request)

        monkeypatch.setattr(effects, "run_preview_page_fan_out", recording)
    evaluator = OperationEvaluator(session.document)
    return effects.evaluate_preview_tile(
        session,
        tile,
        demand=_demand(0),
        semantic_source_id=session.tile_semantic_source_id(tile.source_index),
        level=level,
        cancellation_token=None,
        shader_display=bool(session.shader_display),
        evaluation_context=None,
        stage_cache=evaluator.stage_cache,
        stage_materializer=evaluator.stage_materializer,
    )


def _inline_fan_out(request, tile):
    """The composition that ran at the call site before the entry existed.

    Deliberately a real ``RenderedTile``, not the entry's stand-in for one:
    the substitution is the part of the move that could silently change the
    display histogram, so the reference must not share it.
    """

    source = np.asarray(request.source)
    display_plane = request.display_plane
    rendered = RenderedTile(
        tile=tile,
        image=source if display_plane is None else np.asarray(display_plane),
        histogram_data=request.display_histogram,
        eval_ms=0.0,
        slab_shape=(),
        slab_nbytes=0,
        shader_mapping=request.shader_mapping,
        texture_kind=request.texture_kind,
        semantic_data=None,
        semantic_histogram_data=None,
        quality="preview",
    )
    reduced_source = effects.reduce_array_display_axes(
        source,
        (0, 1),
        request.factor_xy,
        source_starts={
            0: int(request.source_origin_yx[0]),
            1: int(request.source_origin_yx[1]),
        },
    )
    pages = effects._materialize_shared_preview_pages(reduced_source, plans=request.plans)
    histogram = effects._preview_display_histogram(
        rendered,
        reduced_source,
        request.texture_kind,
        None,
    )
    stats = effects._preview_level_stats_from_native_source(
        source,
        source_index=int(request.source_index),
        mapping=request.shader_mapping,
    )
    return pages, histogram, stats


def _assert_same_fan_out(produced, expected) -> None:
    pages, histogram, stats = produced
    expected_pages, expected_histogram, expected_stats = expected
    assert [page.key for page in pages] == [page.key for page in expected_pages]
    for page, reference in zip(pages, expected_pages, strict=True):
        np.testing.assert_array_equal(page.values, reference.values)
        assert page.values.dtype == reference.values.dtype
    if expected_histogram is None:
        assert histogram is None
    else:
        np.testing.assert_array_equal(np.asarray(histogram), np.asarray(expected_histogram))
    assert (stats is None) == (expected_stats is None)
    if stats is not None:
        assert stats.bounds == expected_stats.bounds
        np.testing.assert_array_equal(stats.sample, expected_stats.sample)


@pytest.mark.parametrize("shader_display", [True, False], ids=["wgpu", "pyqtgraph"])
def test_entry_reproduces_the_inline_fan_out_byte_for_byte(monkeypatch, shader_display):
    session = _session(shader_display=shader_display)
    tile = session.plan.tiles[1]
    captured = []

    payload = _evaluate(session, tile, capture=captured, monkeypatch=monkeypatch)

    assert payload is not None
    [request] = captured
    _assert_same_fan_out(
        (payload[1], payload[2], payload[6]),
        _inline_fan_out(request, tile),
    )
    # The pages are the whole point of the request; a request that carried none
    # would satisfy the comparison above vacuously.
    assert payload[1]


def test_request_and_reply_survive_a_pickle_round_trip(monkeypatch):
    import pickle

    session = _session()
    captured = []
    _evaluate(session, session.plan.tiles[0], capture=captured, monkeypatch=monkeypatch)
    [request] = captured

    revived = pickle.loads(pickle.dumps(request))
    produced = pickle.loads(pickle.dumps(effects.fan_out_preview_pages(revived)))

    _assert_same_fan_out(produced, effects.fan_out_preview_pages(request))


def test_a_reducing_fan_out_dispatches_and_a_passthrough_one_does_not(monkeypatch):
    """R5: nothing to compute means nothing to ship."""

    session = _session()
    captured = []
    _evaluate(session, session.plan.tiles[0], capture=captured, monkeypatch=monkeypatch)
    [request] = captured
    passthrough = effects.PreviewPageFanOut(
        source=np.zeros(request.plans[0].stored_shape, dtype=request.plans[0].key.dtype),
        plans=request.plans[:1],
        factor_xy=(1, 1),
        source_origin_yx=(0, 0),
        source_index=0,
    )

    assert effects._preview_pages_are_passthrough(request) is False
    assert effects._preview_pages_are_passthrough(passthrough) is True

    dispatched = []

    def dispatcher(call, token):
        dispatched.append(call.entry)
        return call.invoke(token)

    with remote_dispatch(dispatcher):
        effects.run_preview_page_fan_out(request)
        assert dispatched == [effects.PREVIEW_PAGE_FAN_OUT_ENTRY]
        effects.run_preview_page_fan_out(passthrough)
        assert dispatched == [effects.PREVIEW_PAGE_FAN_OUT_ENTRY]


def test_the_fan_out_runs_in_a_worker_process_and_the_parent_admits_its_pages(monkeypatch, pool):
    # A 256-square complex plane is 512 KiB: past the slot threshold and inside
    # one 1 MiB slot, which is the transport the family is supposed to use.
    session = _session(extent=256)
    tile = session.plan.tiles[2]
    captured = []
    payload = _evaluate(session, tile, capture=captured, monkeypatch=monkeypatch)
    [request] = captured
    key = payload[0]
    assert slottable(request.source)

    remote = pool.run(RemoteCall(effects.PREVIEW_PAGE_FAN_OUT_ENTRY, request))

    assert pool._slots.fallbacks == 0

    _assert_same_fan_out(remote, effects.fan_out_preview_pages(request))
    cache = LodPageCache(max_bytes=1 << 20)
    owner = ("test-remote-fan-out", key)
    for plan in cache.claim_plans(key.plans, owner):
        page = next(item for item in remote[0] if item.key == plan.key)
        cache.admit_as(plan.key, page, owner=owner)
    cache.release_owner_claims(owner)
    resolved = cache.resolved_page_set(key.plans)
    assert resolved is not None
    assert bool(resolved.exact)
