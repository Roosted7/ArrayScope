"""ADR 0064 family 6: the tile evaluation body as ``(entry, request)``.

Three claims need pinning. The entry must reproduce the inline composition
byte for byte on both display backends — a tile that is *nearly* right is a
wrong picture. The gate must refuse exactly the evaluations that would have
touched the parent's stage cache, because those are the ones a child cannot
honestly run. And the whole thing has to survive a real process boundary, with
the parent still owning what it owned before.
"""

from __future__ import annotations

import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.core.compute_policy import ComputeLane, EvaluationContext
from arrayscope.core.view_state import ViewState
from arrayscope.display.lod import LodDemand
from arrayscope.display.montage import make_montage_plan
from arrayscope.kernel.process_backend import ProcessPool
from arrayscope.kernel.remote_call import RemoteCall, remote_dispatch
from arrayscope.operations import remote_snapshot, source_transport, stage_segments
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.evaluator import (
    OperationEvaluator,
    evaluate_image_snapshot,
    stage_document_key,
)
from arrayscope.operations.pipeline import (
    ArrayDocument,
    CenteredFFT,
    CenteredIFFT,
    Crop,
    FFTShift,
)
from arrayscope.operations.remote_snapshot import (
    IMAGE_SNAPSHOT_ENTRY,
    evaluate_image_snapshot_request,
    portable_image_snapshot_request,
    run_image_snapshot,
    run_stage_backed_snapshot,
)
from arrayscope.operations.slabs import plan_slab, request_for_image, stage_key_for_candidate
from arrayscope.operations.stage_fanin import StageFanInState
from arrayscope.operations.stage_segments import release_all_stage_segments, segment_for_value
from arrayscope.presentation import TileLifecycle
from arrayscope.render import effects
from tests.kernel import remote_entries


@pytest.fixture(scope="module")
def pool():
    pool = ProcessPool(processes=1).acquire()
    yield pool
    pool.release()


def _view_state(shape):
    return ViewState.from_shape(shape).with_image_axes(0, 1)


def _lut():
    return np.linspace(0.0, 1.0, 256 * 4, dtype=np.float32).reshape(256, 4)


def _values(extent: int = 24):
    return np.arange(extent * extent * 4, dtype=np.float32).reshape(extent, extent, 4)


def _mapped_values(tmp_path, extent: int = 24):
    path = tmp_path / "montage.npy"
    np.save(path, _values(extent))
    return np.load(path, mmap_mode="r")


def _assert_same_result(produced, expected) -> None:
    np.testing.assert_array_equal(np.asarray(produced.value.data), np.asarray(expected.value.data))
    assert produced.value.data.dtype == expected.value.data.dtype
    for name in ("histogram_data", "semantic_data", "lod_source_data"):
        left = getattr(produced.value, name, None)
        right = getattr(expected.value, name, None)
        assert (left is None) == (right is None), name
        if left is not None:
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    assert produced.value.shader_mapping == expected.value.shader_mapping
    assert produced.value.texture_kind == expected.value.texture_kind
    assert produced.slab_shape == expected.slab_shape
    assert produced.slab_nbytes == expected.slab_nbytes


# ------------------------------------------------------------- equality pin


@pytest.mark.parametrize("shader_display", [True, False], ids=["wgpu", "pyqtgraph"])
@pytest.mark.parametrize("operations", [(), (Crop(2, 1, 3),)], ids=["raw", "cropped"])
def test_the_entry_reproduces_the_inline_evaluation(tmp_path, shader_display, operations):
    document = ArrayDocument(_mapped_values(tmp_path), operations=operations)
    view_state = _view_state(document.current_shape)
    lut = _lut()

    request = portable_image_snapshot_request(
        document,
        view_state,
        colormap_lut=lut,
        shader_display=shader_display,
        provisional_histogram=shader_display,
        stage_cache=OperationEvaluator(document).stage_cache,
    )
    assert request is not None
    # The seeded live document would make the entry a no-op re-run of the very
    # composition it is meant to reproduce; force the rebuild the child does.
    remote_snapshot._documents.clear()

    produced = evaluate_image_snapshot_request(request)

    _assert_same_result(
        produced,
        evaluate_image_snapshot(
            document,
            view_state,
            colormap_lut=lut,
            shader_display=shader_display,
            provisional_histogram=shader_display,
        ),
    )
    assert np.asarray(produced.value.data).size > 0


def test_the_request_survives_a_pickle_round_trip(tmp_path):
    document = ArrayDocument(_mapped_values(tmp_path))
    view_state = _view_state(document.current_shape)
    request = portable_image_snapshot_request(
        document, view_state, colormap_lut=_lut(), shader_display=True
    )

    revived = pickle.loads(pickle.dumps(request))
    remote_snapshot._documents.clear()

    _assert_same_result(
        evaluate_image_snapshot_request(revived),
        evaluate_image_snapshot(document, view_state, colormap_lut=_lut(), shader_display=True),
    )


# -------------------------------------------------------------------- gate


def test_a_stage_backed_tile_is_refused_and_the_same_tile_without_a_cache_is_not(tmp_path):
    """The gate is the evaluator's own branch, not a guess about FFTs.

    Over a mapped source, so the only thing that can decide either way is the
    stage cache — an unpublished eager array would refuse for its own reason.
    """

    document = ArrayDocument(_mapped_values(tmp_path), operations=(CenteredFFT(axis=2),))
    view_state = _view_state(document.current_shape)
    stage_cache = OperationEvaluator(document).stage_cache

    assert portable_image_snapshot_request(document, view_state, stage_cache=stage_cache) is None
    assert portable_image_snapshot_request(document, view_state, stage_cache=None) is not None


def _fft_chain():
    """The user's own chain, the one that dropped the viewer to ~2 draws/s."""

    return (CenteredFFT(axis=2), FFTShift(axis=2), CenteredIFFT(axis=2))


def _warmed_stage(document, view_state, *, publish: bool):
    """Materialize this document's stage into a real cache, as the montage does."""

    cache = OperationEvaluator(document).stage_cache
    with source_transport_publication(publish):
        evaluate_image_snapshot(
            document,
            view_state,
            shader_display=True,
            provisional_histogram=True,
            stage_cache=cache,
            stage_document_key=stage_document_key(document),
        )
    return cache


class source_transport_publication:
    """Turn shared publication on for a block, as ``window/main.py`` does once."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def __enter__(self):
        self.previous = source_transport._publication_enabled
        source_transport._publication_enabled = self.enabled
        return self

    def __exit__(self, *exc):
        source_transport._publication_enabled = self.previous
        return False


@pytest.fixture(autouse=True)
def released_segments():
    yield
    release_all_stage_segments()


@pytest.mark.parametrize("published", [True, False], ids=["published", "unpublished"])
def test_a_stage_backed_tile_moves_exactly_when_its_stage_can_travel(tmp_path, published):
    """The relaxation, and its floor.

    A resident stage that has a segment travels as a name. A resident stage
    without one refuses the move — a child that missed it would not be wrong,
    it would recompute the FFT prefix the parent already paid for, which is the
    whole cost this family exists to avoid.
    """

    document = ArrayDocument(_mapped_values(tmp_path), operations=_fft_chain())
    view_state = _view_state(document.current_shape)
    cache = _warmed_stage(document, view_state, publish=published)

    request = portable_image_snapshot_request(
        document,
        view_state,
        stage_cache=cache,
        stage_document_key=stage_document_key(document),
        shader_display=True,
    )

    if not published:
        assert request is None
        return
    assert request is not None
    assert len(request.stage_segments) == 1
    assert request.stage_cache_max_bytes == cache.max_bytes
    # The segment names the stage the parent actually holds, not a re-plan of it.
    resident = cache.resident_items()[0][1]
    assert request.stage_segments[0] is segment_for_value(resident)


def test_an_unmaterialized_stage_keeps_the_evaluation_in_process(tmp_path):
    """Materialization, single-flight and admission stay parent-side."""

    document = ArrayDocument(_mapped_values(tmp_path), operations=_fft_chain())
    view_state = _view_state(document.current_shape)

    with source_transport_publication(True):
        assert (
            portable_image_snapshot_request(
                document,
                view_state,
                stage_cache=OperationEvaluator(document).stage_cache,
                stage_document_key=stage_document_key(document),
            )
            is None
        )


@pytest.mark.parametrize("shader_display", [True, False], ids=["wgpu", "pyqtgraph"])
def test_the_entry_reproduces_the_stage_backed_evaluation(tmp_path, shader_display):
    """Byte identity where it is hardest: the composition reads a shared stage.

    The child re-plans, hits the attached segment at the same stage the parent
    would have hit in its own cache, and runs the remaining transitions. Any
    disagreement about *which* stage was hit shows up here as different pixels,
    not as a slower tile.
    """

    document = ArrayDocument(_mapped_values(tmp_path), operations=_fft_chain())
    view_state = _view_state(document.current_shape)
    lut = _lut()
    cache = _warmed_stage(document, view_state, publish=True)

    request = portable_image_snapshot_request(
        document,
        view_state,
        colormap_lut=lut,
        shader_display=shader_display,
        provisional_histogram=shader_display,
        stage_cache=cache,
        stage_document_key=stage_document_key(document),
    )
    assert request is not None
    # Force the rebuild a child does; otherwise the entry re-runs against the
    # very document (and the very stage array) it is meant to reproduce.
    remote_snapshot._documents.clear()
    attachments = stage_segments.attachments

    produced = evaluate_image_snapshot_request(request)

    # Equal pixels prove nothing on their own: an entry that missed the segment
    # and recomputed the FFT prefix would also match. One new attachment is
    # what says the shared stage is where the bytes came from.
    assert stage_segments.attachments == attachments + 1
    _assert_same_result(
        produced,
        evaluate_image_snapshot(
            document,
            view_state,
            colormap_lut=lut,
            shader_display=shader_display,
            provisional_histogram=shader_display,
            stage_cache=cache,
            stage_document_key=stage_document_key(document),
        ),
    )


def test_the_fan_ins_own_stage_value_is_enough_to_move_a_tile(tmp_path):
    """The montage hands a tile its stage directly and never consults the cache."""

    document = ArrayDocument(_mapped_values(tmp_path), operations=_fft_chain())
    view_state = _view_state(document.current_shape)
    cache = _warmed_stage(document, view_state, publish=True)
    stage_value = cache.resident_items()[0][1]

    dispatched = []
    with remote_dispatch(lambda call, token: dispatched.append(call.entry)):
        run_stage_backed_snapshot(
            document,
            view_state,
            stage_value=stage_value,
            shader_display=True,
            stage_cache_max_bytes=cache.max_bytes,
        )

    assert dispatched == [IMAGE_SNAPSHOT_ENTRY]


def test_an_unpublished_stage_value_leaves_the_caller_its_own_composition(tmp_path):
    document = ArrayDocument(_mapped_values(tmp_path), operations=_fft_chain())
    view_state = _view_state(document.current_shape)
    cache = _warmed_stage(document, view_state, publish=False)

    assert (
        run_stage_backed_snapshot(
            document,
            view_state,
            stage_value=cache.resident_items()[0][1],
            shader_display=True,
        )
        is None
    )


def test_an_operation_a_worker_could_not_re_resolve_is_refused(tmp_path):
    """A recipe a fresh interpreter cannot rebuild is a wrong tile, not a slow one."""

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Unregistered:
        def apply(self, data):
            return data

        def output_shape(self, shape):
            return tuple(shape)

        def output_dtype(self, dtype):
            return dtype

    document = ArrayDocument(_mapped_values(tmp_path), operations=(Unregistered(),))

    assert portable_image_snapshot_request(document, _view_state(document.current_shape)) is None


def test_an_undescribable_source_keeps_the_call_in_process(monkeypatch):
    """No descriptor, no move: ``run_image_snapshot`` still returns the answer."""

    monkeypatch.setattr(remote_snapshot, "describe_base_data", lambda *_args, **_kwargs: None)
    document = ArrayDocument(_values())
    view_state = _view_state(document.current_shape)

    dispatched = []
    with remote_dispatch(lambda call, token: dispatched.append(call.entry)):
        produced = run_image_snapshot(document, view_state, shader_display=True)

    assert dispatched == []
    _assert_same_result(
        produced, evaluate_image_snapshot(document, view_state, shader_display=True)
    )


# ------------------------------------------------------------- cancellation


def test_the_caller_token_reaches_the_remote_call(tmp_path):
    """Without it the pool takes ``pass_token=False`` and the child cannot look.

    The shared-byte mechanics themselves are pinned once, at the kernel
    (``tests/kernel/test_process_backend.py::test_the_child_observes_the_kernels_cancellation``);
    what this family owes is handing its token over so that pin applies to it.
    """

    document = ArrayDocument(_mapped_values(tmp_path))
    token = SimpleNamespace(cancelled=False)
    seen = []

    with remote_dispatch(lambda call, call_token: seen.append((call.entry, call_token))):
        run_image_snapshot(
            document,
            _view_state(document.current_shape),
            cancellation_token=token,
            shader_display=True,
        )

    assert seen == [(IMAGE_SNAPSHOT_ENTRY, token)]


def test_a_cancelled_token_stops_the_child_instead_of_letting_the_tile_finish(pool, tmp_path):
    """R6: a superseded scrub must not pay for every stale tile in full."""

    document = ArrayDocument(_mapped_values(tmp_path, extent=64))
    request = portable_image_snapshot_request(
        document, _view_state(document.current_shape), shader_display=True
    )
    token = SimpleNamespace(cancelled=True)

    with pytest.raises(EvaluationCancelled):
        pool.run(RemoteCall(IMAGE_SNAPSHOT_ENTRY, request), token, pass_token=True)

    # And the pool is still usable afterwards: a cancellation is an answer.
    assert pool.run(RemoteCall(remote_entries.__name__ + ":own_pid", None)) != os.getpid()


# ------------------------------------------------------------- end to end


def _session(base_data, *, extent: int, shader_display: bool, operations=()):
    state = (
        ViewState.from_shape(np.shape(base_data))
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
        document=ArrayDocument(base_data, operations=tuple(operations)),
        view_state=state,
        plan=plan,
        montage_axis=2,
        colormap_lut=None,
        shader_display=shader_display,
        rgb=False,
        canonical_orientation=False,
        lod_preview_level=0,
        stage_fan_in=StageFanInState(),
        lifecycle=TileLifecycle(),
        rendered_tiles={},
        display_tile_payloads={},
        tile_presentation_state=SimpleNamespace(payloads={}),
        lod_page_cache=None,
        source_anchoring=None,
        tile_semantic_source_id=lambda source_index: ("semantic", int(source_index)),
    )


def _demand() -> LodDemand:
    return LodDemand(
        desired_level=0,
        desired_factor=1,
        desired_factor_xy=(1, 1),
        acceptable_levels=(0,),
        source_texels_per_pixel_xy=(1.0, 1.0),
        reason="test",
    )


@pytest.mark.parametrize("mapped", [True, False], ids=["memmap", "shared-memory"])
def test_a_montage_fill_evaluates_its_tiles_in_worker_processes(
    pool, tmp_path, mapped, monkeypatch
):
    """The family end to end: every tile completes, and a child did the work."""

    # What ``window/main.py`` turns on when it chooses the process backend.
    monkeypatch.setattr(source_transport, "_publication_enabled", True)
    extent = 24
    base_data = _mapped_values(tmp_path, extent) if mapped else _values(extent)
    session = _session(base_data, extent=extent, shader_display=True)
    evaluator = OperationEvaluator(session.document)
    context = EvaluationContext(
        lane=ComputeLane.MONTAGE_TILE,
        cancellation_token=None,
        fft_workers=1,
        memory_policy=None,
    )

    routed = []

    def dispatcher(call, token):
        routed.append(call.entry)
        return pool.run(call, token, pass_token=token is not None)

    produced = {}
    with remote_dispatch(dispatcher):
        for tile in session.plan.tiles:
            produced[int(tile.montage_index)] = effects.evaluate_target_tile(
                session,
                tile,
                level=0,
                demand=_demand(),
                semantic_source_id=session.tile_semantic_source_id(tile.source_index),
                stage_cache=evaluator.stage_cache,
                stage_materializer=evaluator.stage_materializer,
                cancellation_token=None,
                shader_display=True,
                evaluation_context=context,
            )

    assert len(produced) == len(session.plan.tiles)
    assert routed == [IMAGE_SNAPSHOT_ENTRY] * len(session.plan.tiles)
    # ``pool.run`` has no in-process branch: a reply means a child produced it.
    # The pid probe names the process the same pool object answered from.
    assert pool.run(RemoteCall(remote_entries.__name__ + ":own_pid", None)) != os.getpid()

    for tile in session.plan.tiles:
        result = produced[int(tile.montage_index)]
        expected = evaluate_image_snapshot(
            session.document,
            tile.view_state,
            colormap_lut=None,
            shader_display=True,
            provisional_histogram=True,
        )
        _assert_same_result(result, expected)
        # The parent, not the worker, is what attached this tile's evidence.
        assert result.value.level_stats is not None
        assert result.value.level_stats.source_index == tile.source_index


@pytest.mark.parametrize("shader_display", [True, False], ids=["wgpu", "pyqtgraph"])
def test_an_fft_montage_fills_from_one_shared_stage_in_worker_processes(
    pool, tmp_path, shader_display
):
    """The workload the trace caught: an FFT prefix, filled off the GUI's GIL.

    The parent still owns the stage — it materializes it, admits it and hands
    it to the fan-in exactly as before. What moves is every tile's remainder:
    the cut out of the shared 4-D stage, the transitions after it, and the
    display build. The proof that this is what happened is threefold: a child
    answered, the child mapped the stage instead of recomputing it, and the
    pixels match the in-process stage-backed composition.
    """

    extent = 24
    base_data = _mapped_values(tmp_path, extent)
    session = _session(
        base_data, extent=extent, shader_display=shader_display, operations=_fft_chain()
    )
    evaluator = OperationEvaluator(session.document)
    context = EvaluationContext(
        lane=ComputeLane.MONTAGE_TILE,
        cancellation_token=None,
        fft_workers=1,
        memory_policy=None,
    )
    lead = session.plan.tiles[0]
    document_key = stage_document_key(session.document)

    # What the montage's stage fan-in does before any tile runs: materialize
    # the shared stage once, in the parent, and hand every tile the value.
    with source_transport_publication(True):
        evaluate_image_snapshot(
            session.document,
            lead.view_state,
            shader_display=shader_display,
            provisional_histogram=shader_display,
            stage_cache=evaluator.stage_cache,
            stage_document_key=document_key,
        )
        stage_key, stage_value = evaluator.stage_cache.resident_items()[0]
        candidate = next(
            item
            for item in plan_slab(
                session.document, request_for_image(lead.view_state)
            ).region_plan.cache_candidates
            if stage_key_for_candidate(document_key, item) == stage_key
        )
        for tile in session.plan.tiles:
            index = int(tile.montage_index)
            session.stage_fan_in.tile_stage_keys[index] = stage_key
            session.stage_fan_in.tile_stage_candidates[index] = candidate
            session.stage_fan_in.values[stage_key] = stage_value

        routed = []
        produced = {}
        mapped_before = pool.run(RemoteCall(remote_entries.__name__ + ":stage_attachments", None))

        def dispatcher(call, token):
            routed.append(call.entry)
            return pool.run(call, token, pass_token=token is not None)

        with remote_dispatch(dispatcher):
            for tile in session.plan.tiles:
                produced[int(tile.montage_index)] = effects.evaluate_target_tile(
                    session,
                    tile,
                    level=0,
                    demand=_demand(),
                    semantic_source_id=session.tile_semantic_source_id(tile.source_index),
                    stage_cache=evaluator.stage_cache,
                    stage_materializer=evaluator.stage_materializer,
                    cancellation_token=None,
                    shader_display=shader_display,
                    evaluation_context=context,
                )

    assert len(produced) == len(session.plan.tiles)
    assert routed == [IMAGE_SNAPSHOT_ENTRY] * len(session.plan.tiles)
    assert pool.run(RemoteCall(remote_entries.__name__ + ":own_pid", None)) != os.getpid()
    # One mapping, four tiles: the child read the stage instead of recomputing
    # it (which would be no new mapping at all), and it mapped it once — which
    # is the whole point of naming a segment instead of shipping an array.
    mapped = pool.run(RemoteCall(remote_entries.__name__ + ":stage_attachments", None))
    assert mapped == mapped_before + 1
    # And those are the parent's bytes: the probe attaches the same segment the
    # tiles were given, in the same worker.
    probe = pool.run(
        RemoteCall(remote_entries.__name__ + ":stage_segment_probe", segment_for_value(stage_value))
    )
    assert probe["pid"] != os.getpid()
    assert probe["checksum"] == pytest.approx(complex(np.asarray(stage_value.data).sum()))

    for tile in session.plan.tiles:
        result = produced[int(tile.montage_index)]
        assert result.compute_path == "stage_backed"
        _assert_same_result(
            result,
            evaluate_image_snapshot(
                session.document,
                tile.view_state,
                colormap_lut=None,
                shader_display=shader_display,
                provisional_histogram=shader_display,
                stage_cache=evaluator.stage_cache,
                stage_document_key=document_key,
            ),
        )
        assert result.value.level_stats is not None
        assert result.value.level_stats.source_index == tile.source_index
