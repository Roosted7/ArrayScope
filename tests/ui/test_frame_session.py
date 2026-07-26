import dataclasses
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.core.frame_targets import FrameTarget
from arrayscope.core.view_state import ViewState
from arrayscope.display.backend_contract import PYQTGRAPH_CAPABILITIES
from arrayscope.display.frame_planner import FramePlanner
from arrayscope.display.lod import LodInfo
from arrayscope.display.model.frame import (
    DisplayTilePayload,
    TileCommitReport,
    TilePresentationState,
)
from arrayscope.display.model.tile_identity import tile_ack_identity
from arrayscope.display.montage import MontageTileState, RenderedTile, make_montage_plan
from arrayscope.display.shader_mapping import (
    ShaderComponent,
    ShaderDisplayMode,
    ShaderMapping,
    ShaderScale,
    TexturePlaneKind,
)
from arrayscope.window.frame_session import FrameSession


def _session():
    state = ViewState.from_shape((2, 2, 4)).with_montage_axis(2, indices=(0, 1, 2, 3), text=":")
    plan = make_montage_plan(state, axis=2, indices=(0, 1, 2, 3), tile_shape=(2, 2), columns=4)
    return FrameSession(
        session_id=1,
        key="key",
        render_generation=1,
        level_key="levels",
        level_expected_indices=(0, 1, 2, 3),
        plan=plan,
        view_state=state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(10, 10),
        view_range=None,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
    )


def _present_exact_tiles(session, *tile_numbers):
    # Tests that mutate ``visible_tiles`` directly must publish the new
    # lifecycle targets just as the production viewport retarget does. A
    # presented payload without an active semantic target is intentionally
    # not sufficient evidence for visible completion.
    session.sync_lifecycle_scope()
    payloads = dict(getattr(session.tile_presentation_state, "payloads", {}) or {})
    for tile_number in tile_numbers:
        index = int(tile_number)
        tile = session.plan.tiles[index]
        image = np.ones((2, 2), dtype=np.float32)
        payload = DisplayTilePayload(
            tile_number=index,
            source_index=int(tile.source_index),
            image=image,
            histogram_data=image,
            source_id=("tile", int(tile.source_index), "exact"),
        )
        session.display_tile_payloads[index] = payload
        session.record_tile_payload(payload)
        payloads[index] = payload
        session.lifecycle.commit_emitted({index: payload})
        session.lifecycle.backend_ack({index: payload})
        session.lifecycle.acknowledge_presented(index, payload.source_id, payload.quality, 0)
    session.tile_presentation_state = TilePresentationState(payloads)
    session.lifecycle.presentation_confirmed(tile_numbers)


def test_montage_render_session_starts_with_required_targets_unsettled():
    session = _session()

    assert isinstance(session.pending_level_tiles, deque)
    assert session.required_target_unsettled_tiles() == (0, 1, 2, 3)


def test_stage_materialization_inherits_best_consumer_tile_rank():
    from arrayscope.kernel import UNRANKED_SCHEDULING_RANK
    from arrayscope.window.frame_effects import _stage_consumer_scheduling_rank

    session = _session()
    session.stage_fan_in.tile_stage_keys = {2: "shared-stage", 3: "shared-stage"}

    assert _stage_consumer_scheduling_rank(session, "shared-stage") == 2
    assert _stage_consumer_scheduling_rank(session, "unbound-stage") == UNRANKED_SCHEDULING_RANK


def test_first_pass_physical_completion_uses_required_tiles():
    session = _session()
    session.visible_tiles = (session.plan.tiles[0], session.plan.tiles[1])
    session.visible_tile_numbers = frozenset({0, 1})
    session.sync_lifecycle_scope()
    session.first_pass_quality = "exact"
    _present_exact_tiles(session, 0, 1)

    assert session.first_pass_pixels_presented()


def test_first_pass_completion_uses_adopted_targets_not_extra_frame_regions():
    session = _session()
    session.visible_tiles = (session.plan.tiles[0],)
    session.visible_tile_numbers = frozenset({0})
    session.frame_plan = SimpleNamespace(active_region_ids=(0, 1, 2, 3))

    _present_exact_tiles(session, 0)

    assert session.required_tile_numbers() == (0, 1, 2, 3)
    assert session.scheduling_policy.verdict.required_tiles == (0,)
    assert session.observe_physically_presented_first_pass_quality(
        session.tile_presentation_state.payloads
    )
    assert session.first_pass_quality == "exact"
    assert session.first_pass_pixels_presented()


def test_visible_first_pixels_use_the_canonical_required_scope():
    session = _session()
    session.visible_tile_numbers = frozenset({0, 1})
    _present_exact_tiles(session, 0, 1)

    assert session.required_first_pixels_presented()
    assert session.visible_first_pixels_presented()


def test_preview_first_pass_accepts_compatible_exact_overlap():
    """Exact retained sources are better first-pass pixels, not blockers."""

    session = _session()
    session.first_pass_quality = "preview"
    _present_exact_tiles(session, 0, 1, 2, 3)
    tile = session.plan.tiles[3]
    image = np.ones((2, 2), dtype=np.float32)
    preview = DisplayTilePayload(
        tile_number=3,
        source_index=int(tile.source_index),
        image=image,
        histogram_data=image,
        source_id=("tile", int(tile.source_index), "preview"),
        quality="preview",
        lod=LodInfo(level=2, factor=4, source_shape=image.shape, texture_shape=image.shape),
    )
    session.display_tile_payloads[3] = preview
    session.record_tile_payload(preview)
    session.lifecycle.commit_emitted({3: preview})
    session.lifecycle.backend_ack({3: preview})
    session.tile_presentation_state = TilePresentationState(
        {**session.tile_presentation_state.payloads, 3: preview}
    )

    assert session.first_pass_pixels_presented()


def test_retained_physical_fallback_seeds_first_pass_without_settling_exact_target():
    """A complete backend floor is coverage evidence, never exact acknowledgement."""

    session = _session()
    payloads = {}
    for tile in session.plan.tiles:
        tile_number = int(tile.montage_index)
        image = np.ones((2, 2), dtype=np.float32)
        payload = DisplayTilePayload(
            tile_number=tile_number,
            source_index=int(tile.source_index),
            image=image,
            histogram_data=image,
            source_id=("tile", int(tile.source_index), "preview"),
            quality="preview",
            lod=LodInfo(level=2, factor=4, source_shape=image.shape, texture_shape=image.shape),
        )
        payloads[tile_number] = payload
        session.display_tile_payloads[tile_number] = payload
        session.record_tile_payload(payload)
        session.lifecycle.commit_emitted({tile_number: payload})
        session.lifecycle.backend_ack({tile_number: payload})
        session.lifecycle.acknowledge_presented(
            tile_number,
            tile_ack_identity(payload),
            payload.quality,
            payload.lod.level,
        )

    assert session.first_pass_quality is None
    assert session.required_target_unsettled_tiles() == (0, 1, 2, 3)

    assert session.observe_physically_presented_first_pass_quality(payloads)

    assert session.first_pass_quality == "preview"
    assert session.first_pass_pixels_presented()
    assert session.required_target_unsettled_tiles() == (0, 1, 2, 3)
    assert all(
        (record := session.lifecycle.peek(tile_number)) is not None and not record.target_settled
        for tile_number in session.required_tile_numbers()
    )


def test_physical_preview_widens_latched_exact_first_pass(monkeypatch):
    """A later physical fallback must not strand the coverage barrier."""

    session = _session()
    _present_exact_tiles(session, 0, 1, 2, 3)
    assert session.observe_physically_presented_first_pass_quality(
        session.tile_presentation_state.payloads
    )
    assert session.first_pass_quality == "exact"

    tile = session.plan.tiles[3]
    image = np.ones((2, 2), dtype=np.float32)
    fallback = DisplayTilePayload(
        tile_number=3,
        source_index=int(tile.source_index),
        image=image,
        histogram_data=image,
        source_id=("tile", int(tile.source_index), "preview"),
        quality="preview",
        lod=LodInfo(
            level=2,
            factor=4,
            source_shape=image.shape,
            texture_shape=image.shape,
        ),
    )
    session.display_tile_payloads[3] = fallback
    session.record_tile_payload(fallback)
    session.lifecycle.commit_emitted({3: fallback})
    session.lifecycle.backend_ack({3: fallback})
    session.lifecycle.acknowledge_presented(
        3,
        tile_ack_identity(fallback),
        fallback.quality,
        fallback.lod.level,
    )
    session.tile_presentation_state = TilePresentationState(
        {**session.tile_presentation_state.payloads, 3: fallback}
    )

    events = []
    monkeypatch.setattr(
        "arrayscope.window.frame_session.emit_trace",
        lambda kind, **fields: events.append((kind, fields)),
    )
    assert session.observe_physically_presented_first_pass_quality(
        session.tile_presentation_state.payloads
    )
    assert session.first_pass_quality == "preview"
    assert session.first_pass_pixels_presented()
    assert events == [
        (
            "first_pass_quality",
            {
                "event": "widened_to_preview",
                "session_id": 1,
                "required_tiles": 4,
            },
        )
    ]


def test_montage_render_session_retarget_changes_canonical_priority():
    session = _session()
    session.view_range = ((0.0, 12.0), (0.0, 4.0))

    session.retarget_tile_priority(
        focus=(8.0, 1.0),
        active_tiles=(0, 1, 2, 3),
        near_tiles=(),
    )

    assert session._prioritized_tile_numbers((0, 1, 2))[:1] == (2,)
    assert session.priority_retargeted_tiles == 4


def test_montage_priority_retarget_preserves_payload_identity():
    session = _session()
    tile = session.plan.tiles[0]
    image = np.ones((2, 2), dtype=np.float32)
    rendered = RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes)
    session.mark_materialized(rendered)
    first = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]

    session.retarget_tile_priority(
        focus=(8.0, 1.0),
        active_tiles=(0, 1, 2, 3),
        near_tiles=(),
    )
    second = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]

    assert second is first
    assert second.source_id == first.source_id


def test_montage_render_session_materialized_tile_stays_loading_until_presented():
    session = _session()
    tile = session.plan.tiles[0]
    session.mark_loading(tile)
    session.active_tile_requests.add(int(tile.montage_index))
    rendered = RenderedTile(
        tile, np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32), 0.0, (2, 2), 16
    )

    session.mark_materialized(rendered)

    assert session.is_tile_loaded(tile)
    assert int(tile.montage_index) in session.loading_tiles

    session.mark_presented((tile.montage_index,))

    assert int(tile.montage_index) not in session.loading_tiles
    assert int(tile.montage_index) in session.lifecycle.presented_tiles


def test_stall_probe_row_classifies_presented_preview_pending_as_refinement_backlog():
    from arrayscope.window.frame_runtime import _stall_tile_probe_row_actionable

    session = _session()
    image = np.ones((2, 2), dtype=np.float32)
    payload = DisplayTilePayload(
        tile_number=0,
        source_index=0,
        image=image,
        histogram_data=image,
        source_id=("src", 0, "preview"),
        lod=LodInfo(level=2, factor=4, source_shape=(2, 2), texture_shape=(1, 1)),
        quality="preview",
    )
    session.display_tile_payloads[0] = payload
    session.tile_presentation_state = TilePresentationState({0: payload}, revision=1)
    session.lifecycle.backend_presented_snapshot({0: payload.source_id})
    session.lifecycle.acknowledge_presented(
        0, payload.source_id, payload.quality, payload.lod.level
    )

    row = next(row for row in session.diagnostic_tile_identity_rows() if row["tile"] == 0)

    assert row["target_unsettled"] is True
    assert "pending" not in row
    assert row["rendered"] is False
    assert row["presented"] is True
    assert row["visible_first_pixel_complete"] is True
    assert row["desired_payload_quality"] == "preview"
    assert row["desired_payload_lod"] == 2
    assert row["presented_quality"] == "preview"
    assert row["presented_lod"] == 2
    assert not _stall_tile_probe_row_actionable(row)


def test_stall_probe_row_keeps_stale_presented_identity_actionable():
    from arrayscope.window.frame_runtime import _stall_tile_probe_row_actionable

    session = _session()
    image = np.ones((2, 2), dtype=np.float32)
    payload = DisplayTilePayload(
        tile_number=0,
        source_index=1,
        image=image,
        histogram_data=image,
        source_id=("src", 1, "preview"),
        quality="preview",
    )
    session.display_tile_payloads[0] = payload
    session.tile_presentation_state = TilePresentationState({0: payload}, revision=1)
    session.lifecycle.backend_presented_snapshot({0: payload.source_id})
    session.lifecycle.acknowledge_presented(0, payload.source_id, payload.quality, 0)

    row = next(row for row in session.diagnostic_tile_identity_rows() if row["tile"] == 0)

    assert row["visible_first_pixel_complete"] is False
    assert _stall_tile_probe_row_actionable(row)


def test_stale_committed_state_payload_is_not_complete_after_retarget():
    session = _session()
    tile = session.plan.tiles[0]
    image = np.ones((2, 2), dtype=np.float32)
    rendered = RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes)
    session.rendered_tiles[0] = rendered
    session.visible_tiles = (tile,)
    session.visible_tile_numbers = frozenset({0})
    old_payload = DisplayTilePayload(
        tile_number=0,
        source_index=1,
        image=image + 1,
        histogram_data=image + 1,
        source_id=("src", 1, "lod", 2),
    )
    session.tile_presentation_state = TilePresentationState({0: old_payload}, revision=1)
    session.display_tile_payloads[0] = old_payload
    session.record_tile_payload(old_payload)
    session.lifecycle.commit_emitted({0: old_payload})
    session.lifecycle.backend_ack({0: old_payload})
    session.lifecycle.backend_presented_snapshot({0: old_payload.source_id})
    session.lifecycle.acknowledge_presented(0, old_payload.source_id, old_payload.quality, 0)
    session.lifecycle.presentation_confirmed((0,))

    assert not session.visible_plan_complete()
    assert session.backend_identity_mismatch_tiles() == (0,)

    _state, delta = session.build_tile_presentation({0: ("src", 0)}, max_upserts=4)

    assert delta.removals == ()
    assert 0 in delta.upserts
    assert delta.upserts[0].source_index == 0
    assert 0 in delta.active_tiles
    assert 0 not in session.lifecycle.presented_tiles


def test_unrendered_tile_without_payload_cannot_keep_pending_upsert_marker():
    session = _session()
    session.rendered_tiles.clear()
    session.visible_tiles = (session.plan.tiles[0],)
    session.visible_tile_numbers = frozenset({0})
    session.dirty_payloads[0] = None
    session.pending_payload_upserts[0] = None

    _state, delta = session.build_tile_presentation({0: ("src", 0)}, max_upserts=4)

    assert delta.upserts == {}
    assert 0 not in session.dirty_payloads
    assert 0 not in session.pending_payload_upserts


def test_backend_confirmed_current_payload_rehydrates_active_state():
    session = _session()
    tile = session.plan.tiles[0]
    image = np.full((2, 2), 7.0, dtype=np.float32)
    session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
    payload = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]
    session.tile_presentation_state = TilePresentationState()
    session.lifecycle.backend_presented_snapshot({0: tile_ack_identity(payload)})
    session.lifecycle.presentation_discarded(0)
    assert 0 in session.loading_tiles

    state, delta = session.build_tile_presentation({0: ("tile", 0)}, max_upserts=0)

    assert delta.upserts == {}
    assert state.active_payloads(delta)[0] is payload
    assert session.tile_presentation_state.payloads[0] is payload
    assert 0 in session.lifecycle.presented_tiles
    assert 0 not in session.loading_tiles
    assert 0 not in session.dirty_payloads


def test_tile_presentation_delta_carries_lifecycle_owned_typed_targets():
    session = _session()
    tile = session.plan.tiles[0]
    image = np.full((2, 2), 7.0, dtype=np.float32)
    session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))

    _state, delta = session.build_tile_presentation({0: ("tile", 0)})

    assert set(delta.target_identities) == set(delta.active_tiles)
    assert delta.upserts[0].tile_identity.satisfies_target(delta.target_identities[0])
    assert delta.target_identities[0].texture_kind.value == "scalar_r32f"


def test_backend_confirmed_current_payloads_do_not_trickle_through_upsert_cap():
    session = _session()
    source_ids = {}
    for tile in session.plan.tiles[:3]:
        index = int(tile.montage_index)
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
        source_ids[index] = ("tile", index)
    payloads = session.snapshot_display_tile_payloads(source_ids)
    session.display_tile_payloads.update(payloads)
    session.tile_presentation_state = TilePresentationState()
    session.lifecycle.backend_presented_snapshot(
        {tile: tile_ack_identity(payload) for tile, payload in payloads.items()}
    )
    for tile in payloads:
        session.pending_payload_upserts[int(tile)] = None
        session.dirty_payloads[int(tile)] = None

    state, delta = session.build_tile_presentation(source_ids, max_upserts=1)

    assert delta.upserts == {}
    # Active is the complete committed target scope. Tile 3 has no compatible
    # payload yet and therefore remains an explicit placeholder, not an
    # omitted obligation.
    assert delta.active_tiles == (0, 1, 2, 3)
    assert set(state.active_payloads(delta)) == {0, 1, 2}
    assert session.pending_payload_upserts == {}
    assert session.dirty_payloads == {}
    assert set(session.lifecycle.presented_tiles) >= {0, 1, 2}


def test_montage_render_session_skipped_tile_leaves_required_scope():
    session = _session()
    session.mark_skipped(session.plan.tiles[1])

    assert 1 in session.skipped_tiles
    assert 1 not in session.required_target_unsettled_tiles()
    assert set(session.required_target_unsettled_tiles()) == {0, 2, 3}


def test_montage_render_session_reuses_typed_payload_wrappers_until_tile_changes():
    session = _session()
    tile = session.plan.tiles[0]
    image = np.ones((2, 2), dtype=np.float32)
    histogram = image * 2.0
    first_rendered = RenderedTile(tile, image, histogram, 0.0, image.shape, image.nbytes)
    session.mark_materialized(first_rendered)

    first = session.snapshot_display_tile_payloads({0: ("tile", 0)})
    second = session.snapshot_display_tile_payloads({0: ("tile", 0)})
    state, delta = session.build_tile_presentation({0: ("tile", 0)})
    session.acknowledge_tile_presentation(delta, TileCommitReport(presented_tiles=(0,)))
    clean_state, clean_delta = session.build_tile_presentation({0: ("tile", 0)})

    assert second[0] is first[0]
    assert state.payloads[0] is first[0]
    assert delta.upserts == {0: first[0]}
    assert clean_state.payloads[0] is first[0]
    assert clean_delta.upserts == {}
    replacement = np.full((2, 2), 3.0, dtype=np.float32)
    session.mark_materialized(
        RenderedTile(tile, replacement, replacement, 0.0, replacement.shape, replacement.nbytes)
    )
    third_state, third_delta = session.build_tile_presentation({0: ("tile", 0, "replacement")})
    assert third_state.payloads[0] is not first[0]
    assert third_state.payloads[0].image is replacement
    assert third_delta.upserts[0] is third_state.payloads[0]


def test_montage_render_session_retries_capped_payload_until_backend_acknowledges():
    session = _session()
    source_ids = {}
    for tile in session.plan.tiles[:2]:
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
        source_ids[int(tile.montage_index)] = ("tile", int(tile.montage_index))

    proposed, delta = session.build_tile_presentation(source_ids, max_upserts=1)

    assert tuple(proposed.payloads) == (0,)
    assert tuple(delta.upserts) == (0,)
    assert session.tile_presentation_state.payloads == {}

    session.acknowledge_tile_presentation(delta, TileCommitReport(presented_tiles=(0,)))
    session.mark_presented((0,))
    retry_state, retry_delta = session.build_tile_presentation(source_ids)

    assert 0 in session.tile_presentation_state.payloads
    assert tuple(retry_delta.upserts) == (1,)
    assert tuple(retry_state.payloads) == (0, 1)

    session.acknowledge_tile_presentation(retry_delta, TileCommitReport(presented_tiles=(1,)))

    assert session.tile_presentation_state.payloads[0] is proposed.payloads[0]
    assert 0 not in session.dirty_payloads


def test_montage_render_session_does_not_acknowledge_deferred_visible_upsert():
    session = _session()
    source_ids = {}
    for tile in session.plan.tiles[:2]:
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
        source_ids[int(tile.montage_index)] = ("tile", int(tile.montage_index))

    _proposed, delta = session.build_tile_presentation(source_ids)
    session.acknowledge_tile_presentation(
        delta,
        TileCommitReport(
            presented_tiles=(0, 1),
            committed_upserts=(0,),
        ),
    )

    assert tuple(session.tile_presentation_state.payloads) == (0,)
    assert 0 not in session.dirty_payloads
    assert 1 in session.dirty_payloads

    retry_state, retry_delta = session.build_tile_presentation(source_ids)

    assert tuple(retry_delta.upserts) == (1,)
    assert tuple(retry_state.payloads) == (0, 1)


def test_level_snapshot_keeps_deferred_visible_upsert_pending_until_acknowledged():
    session = _session()
    source_ids = {}
    for tile in session.plan.tiles[:2]:
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
        source_ids[int(tile.montage_index)] = ("tile", int(tile.montage_index))

    first_state, first_delta = session.build_tile_presentation(source_ids)
    session.acknowledge_tile_presentation(
        first_delta,
        TileCommitReport(presented_tiles=first_state.active_payloads(first_delta)),
        levels=(0.0, 1.0),
    )
    session.mark_presented(first_state.active_payloads(first_delta))

    assert session.begin_level_presentation_update((2.0, 4.0)) is True
    proposed, delta = session.build_tile_presentation(source_ids, max_upserts=1)
    session.acknowledge_tile_presentation(
        delta,
        TileCommitReport(
            presented_tiles=(0, 1),
            committed_upserts=(0,),
        ),
        levels=(2.0, 4.0),
    )
    session.mark_presented((0, 1))

    snapshot = session.level_presentation_snapshot()
    assert tuple(proposed.payloads) == (0, 1)
    assert snapshot.revision == 1
    assert snapshot.target_levels == (2.0, 4.0)
    assert snapshot.stale_count == 1
    assert snapshot.pending_count == 1
    assert snapshot.settled is False

    retry_state, retry_delta = session.build_tile_presentation(source_ids)
    assert tuple(retry_delta.upserts) == (1,)

    session.acknowledge_tile_presentation(
        retry_delta,
        TileCommitReport(presented_tiles=retry_state.active_payloads(retry_delta)),
        levels=(2.0, 4.0),
    )
    session.mark_presented(retry_state.active_payloads(retry_delta))
    session.set_level_update_pending(session.has_stale_level_presentations())

    snapshot = session.level_presentation_snapshot()
    assert snapshot.stale_count == 0
    assert snapshot.pending_count == 0
    assert snapshot.settled is True


def _materialize_and_present(session, source_ids, *tile_numbers, levels=(0.0, 1.0)):
    for index in tile_numbers:
        tile = session.plan.tiles[int(index)]
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
        source_ids[int(index)] = ("tile", int(index))
    state, delta = session.build_tile_presentation(source_ids)
    session.acknowledge_tile_presentation(
        delta,
        TileCommitReport(presented_tiles=state.active_payloads(delta)),
        levels=levels,
    )
    session.mark_presented(state.active_payloads(delta))
    return state, delta


def test_pure_level_rewindow_commit_is_flagged_level_only_drain():
    # A commit whose only work is re-windowing already-resident, already-
    # presented exact tiles to new levels must be flagged ``level_only_drain``
    # so a CPU-windowing backend can bound its per-commit re-window work to the
    # emitted upsert slice instead of re-resolving every resident active tile.
    session = _session()
    source_ids = {}
    _materialize_and_present(session, source_ids, 0, 1, 2, 3)

    # Nothing dirty and no pending level target: not a level drain at all.
    _baseline_state, baseline_delta = session.build_tile_presentation(source_ids)
    assert baseline_delta.level_only_drain is False

    # Every visible exact tile now needs a new window: a pure level drain.
    assert session.begin_level_presentation_update((2.0, 4.0)) is True
    _state, delta = session.build_tile_presentation(source_ids, max_upserts=2)

    assert delta.level_only_drain is True
    # The upsert slice is bounded by the requested budget, and every emitted
    # tile is one of the already-presented tiles being re-levelled (no new
    # pixels, no coverage, no removals).
    assert 0 < len(delta.upserts) <= 2
    assert not delta.removals
    assert set(delta.upserts).issubset({0, 1, 2, 3})


def test_first_pixel_coverage_commit_is_not_level_only_drain():
    # A commit that still owes first-pixel coverage for an unpresented tile is
    # not a pure level drain even while a level target is pending: it uploads
    # new pixels, so the backend must take the general (full-resolve) path.
    session = _session()
    source_ids = {}
    # Present only two of the four visible tiles, then materialize (but do not
    # present) the other two so they still owe first-pixel coverage.
    _materialize_and_present(session, source_ids, 0, 1)
    for index in (2, 3):
        tile = session.plan.tiles[index]
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
        source_ids[int(index)] = ("tile", int(index))

    assert session.begin_level_presentation_update((2.0, 4.0)) is True
    _state, delta = session.build_tile_presentation(source_ids, max_upserts=4)

    # Tiles 2 and 3 have never presented first pixels, so this commit owes
    # coverage and must not be treated as a level-only drain.
    assert delta.level_only_drain is False


def test_montage_render_session_caps_upserts_without_clipping_active_scope():
    session = _session()
    source_ids = {}
    for tile in session.plan.tiles[:3]:
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
        source_ids[int(tile.montage_index)] = ("tile-source", int(tile.montage_index))

    first_state, first_delta = session.build_tile_presentation(source_ids, max_upserts=1)

    assert tuple(first_delta.upserts) == (0,)
    assert first_delta.active_tiles == (0, 1, 2, 3)
    assert first_state.active_payloads(first_delta) == {0: first_delta.upserts[0]}
    session.acknowledge_tile_presentation(
        first_delta, TileCommitReport(presented_tiles=first_state.active_payloads(first_delta))
    )
    session.mark_presented(first_state.active_payloads(first_delta))

    second_state, second_delta = session.build_tile_presentation(source_ids, max_upserts=1)

    assert tuple(second_delta.upserts) == (1,)
    assert second_delta.active_tiles == (0, 1, 2, 3)
    assert set(second_state.active_payloads(second_delta)) == {0, 1}
    assert session.ensure_tile_states()[2] == MontageTileState.LOADING
    assert 2 in session.dirty_payloads


def test_montage_render_session_capped_upserts_preserve_ready_priority_order():
    session = _session()
    source_ids = {}
    for index in (2, 0, 1):
        tile = session.plan.tiles[index]
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
        source_ids[int(tile.montage_index)] = ("tile-source", int(tile.montage_index))

    first_state, first_delta = session.build_tile_presentation(source_ids, max_upserts=1)
    session.acknowledge_tile_presentation(
        first_delta, TileCommitReport(presented_tiles=first_state.active_payloads(first_delta))
    )
    session.mark_presented(first_state.active_payloads(first_delta))
    second_state, second_delta = session.build_tile_presentation(source_ids, max_upserts=1)

    assert tuple(first_delta.upserts) == (2,)
    assert first_delta.active_tiles == (0, 1, 2, 3)
    assert tuple(second_delta.upserts) == (0,)
    assert second_delta.active_tiles == (0, 1, 2, 3)
    assert set(second_state.active_payloads(second_delta)) == {0, 2}
    assert 1 in session.dirty_payloads


def test_montage_render_session_dirty_payloads_keep_session_incomplete_until_acknowledged():
    session = _session()
    tile = session.plan.tiles[0]
    session.visible_tiles = (tile,)
    session.visible_tile_numbers = frozenset({0})
    session.sync_lifecycle_scope()
    image = np.ones((2, 2), dtype=np.float32)
    session.loading_tiles.clear()

    session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))

    assert not session.is_complete()

    state, delta = session.build_tile_presentation({0: ("tile", 0)})
    session.acknowledge_tile_presentation(
        delta, TileCommitReport(presented_tiles=state.active_payloads(delta))
    )
    session.mark_presented(state.active_payloads(delta))
    session.note_committed()

    assert session.is_complete()


def test_montage_render_session_completion_tracks_explicit_deferred_stage_debt():
    session = _session()
    session.visible_tiles = ()
    session.visible_tile_numbers = frozenset()
    session.sync_lifecycle_scope()
    session.loading_tiles.clear()

    assert session.is_complete()

    session.stage_planning_deferred = True
    session.deferred_missing_tiles = (session.plan.tiles[0],)
    assert not session.is_complete()

    session.stage_planning_deferred = False
    assert not session.is_complete()

    session.deferred_missing_tiles = ()
    assert session.is_complete()


def test_montage_render_session_visible_plan_ignores_deferred_offscreen_work():
    session = _session()
    session.loading_tiles.clear()
    session.visible_tiles = (session.plan.tiles[0],)
    session.visible_tile_numbers = frozenset({0})
    _present_exact_tiles(session, 0)
    session.loading_tiles.add(3)

    assert session.visible_first_pixels_presented()
    assert session.visible_plan_complete()
    assert not session.is_complete()


def test_montage_render_session_visible_plan_tracks_visible_work_only():
    session = _session()
    session.loading_tiles.clear()
    session.visible_tiles = (session.plan.tiles[0], session.plan.tiles[1])
    session.visible_tile_numbers = frozenset({0, 1})
    _present_exact_tiles(session, 0)

    assert not session.visible_plan_complete()

    assert not session.visible_plan_complete()
    session.loading_tiles.add(1)
    assert not session.visible_plan_complete()
    session.loading_tiles.clear()
    _present_exact_tiles(session, 1)

    assert session.visible_first_pixels_presented()
    assert session.visible_plan_complete()


def test_montage_render_session_uses_frame_plan_required_set_despite_coverage_drift():
    session = _session()
    session.loading_tiles.clear()
    session.visible_tiles = (session.plan.tiles[0], session.plan.tiles[1])
    session.visible_tile_numbers = frozenset({0, 1})
    session.frame_plan = FramePlanner().plan(
        target=FrameTarget(
            semantic_key="key",
            viewport_key="viewport",
            presentation_key="levels",
            quality="exact-visible",
        ),
        view_state=session.view_state,
        display_shape=session.plan.display_shape,
        backend_capabilities=PYQTGRAPH_CAPABILITIES,
        viewport_shape=session.viewport_shape,
        view_range=((0.0, 1.0), (0.0, 1.0)),
        montage_plan=session.plan,
    )
    assert session.frame_plan.active_region_ids == (0,)
    _present_exact_tiles(session, 0)

    assert session.required_tile_numbers() == (0,)
    assert session.required_target_settled()
    assert session.visible_plan_complete()


def test_descoped_payload_completion_does_not_leave_unowned_commit_debt():
    session = _session()
    tile = session.plan.tiles[0]
    image = np.ones((2, 2), dtype=np.float32)
    session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
    state, delta = session.build_tile_presentation({0: ("tile", 0)})
    session.acknowledge_tile_presentation(
        delta, TileCommitReport(presented_tiles=state.active_payloads(delta))
    )
    session.mark_presented(state.active_payloads(delta))

    session.visible_tiles = (session.plan.tiles[1],)
    session.visible_tile_numbers = frozenset({1})
    session.sync_lifecycle_scope()
    session.dirty_payloads[0] = None
    session.pending_payload_upserts[0] = None

    session.build_tile_presentation({0: ("tile", 0)})

    assert 0 not in session.dirty_payloads
    assert 0 not in session.pending_payload_upserts


def test_montage_render_session_replacement_materialization_reopens_visible_plan():
    session = _session()
    session.loading_tiles.clear()
    session.visible_tiles = (session.plan.tiles[0],)
    session.visible_tile_numbers = frozenset({0})
    _present_exact_tiles(session, 0)

    session.mark_materialized(
        RenderedTile(
            session.plan.tiles[0],
            np.ones((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
            0.0,
            (2, 2),
            16,
        )
    )

    # Atomic replacement keeps the last acknowledged pixels visible while the
    # replacement payload is dirty/in flight. Full completion remains open,
    # but first-pixel visible completion does not regress to black.
    assert 0 in session.lifecycle.presented_tiles
    assert session.visible_first_pixels_presented()
    assert not session.visible_plan_complete()
    assert not session.is_complete()


def test_montage_render_session_delta_carries_near_sources_without_payloads():
    session = _session()
    tile = session.plan.tiles[0]
    image = np.ones((2, 2), dtype=np.float32)
    session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))

    source_ids = {index: ("tile-source", index) for index in range(4)}
    _state, delta = session.build_tile_presentation(source_ids)

    assert delta.near_tiles == (0, 1, 2, 3)
    assert delta.near_tile_source_ids[0] == delta.upserts[0].source_id
    assert {key: delta.near_tile_source_ids[key] for key in (1, 2, 3)} == {
        key: source_ids[key] for key in (1, 2, 3)
    }
    assert set(delta.upserts) == {0}


def test_lod_payload_does_not_reduce_display_ready_rgb_phase_tiles():
    state = ViewState.from_shape((8, 8, 1)).with_montage_axis(2, indices=(0,), text=":")
    plan = make_montage_plan(state, axis=2, indices=(0,), tile_shape=(8, 8), columns=1)
    session = FrameSession(
        session_id=1,
        key="key",
        render_generation=1,
        level_key="levels",
        level_expected_indices=(0,),
        plan=plan,
        view_state=state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(2, 2),
        view_range=((0.0, 64.0), (0.0, 64.0)),
        output_dtype=np.dtype(np.uint8),
        rgb=True,
        window_mode=None,
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
    )
    rgb = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    histogram = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    rendered = RenderedTile(
        plan.tiles[0],
        rgb,
        histogram,
        0.0,
        rgb.shape,
        rgb.nbytes,
        texture_kind=TexturePlaneKind.RGB8,
        semantic_data=rgb,
    )
    session.mark_materialized(rendered)

    payload = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]

    assert payload.lod.factor == 1
    assert payload.image.shape[:2] == (8, 8)
    assert payload.histogram_data.shape[:2] == (8, 8)
    assert payload.texture_data.shape[:2] == (8, 8)
    assert payload.semantic_data.shape[:2] == (8, 8)
    assert payload.semantic_histogram_data.shape[:2] == (8, 8)
    assert payload.source_shape == (8, 8)


def _zoomed_out_session(*, dtype=np.float32, rgb=False):
    state = ViewState.from_shape((8, 8, 1)).with_montage_axis(2, indices=(0,), text=":")
    plan = make_montage_plan(state, axis=2, indices=(0,), tile_shape=(8, 8), columns=1)
    return FrameSession(
        session_id=1,
        key="key",
        render_generation=1,
        level_key="levels",
        level_expected_indices=(0,),
        plan=plan,
        view_state=state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(2, 2),
        view_range=((0.0, 64.0), (0.0, 64.0)),
        output_dtype=np.dtype(dtype),
        rgb=rgb,
        window_mode=None,
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
    )


def test_zoomed_out_scalar_payload_keeps_exact_texture_on_ui_commit_path():
    session = _zoomed_out_session()
    image = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    histogram = image + 100.0
    rendered = RenderedTile(
        session.plan.tiles[0],
        image,
        histogram,
        0.0,
        image.shape,
        image.nbytes,
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        semantic_data=image,
    )
    session.mark_materialized(rendered)

    payload = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]
    decision = session.lod_policy_decision

    assert decision.demand.desired_factor > 1
    assert decision.demand.desired_factor_xy[0] > 1
    assert decision.demand.desired_factor_xy[1] > 1
    assert decision.applied_factor == 1
    assert decision.applied_factor_xy == (1, 1)
    assert decision.demand.source_texels_per_pixel_xy == (32.0, 32.0)
    assert decision.policy == "native-only"
    assert "native-only montage LOD policy" in decision.reason
    assert payload.lod.factor == 1
    assert payload.texture_data is image
    assert payload.texture_data.shape[:2] == payload.image.shape[:2]
    assert payload.semantic_data.shape[:2] == (8, 8)
    assert payload.semantic_histogram_data.shape[:2] == (8, 8)


def test_zoomed_out_payload_reuses_clean_wrapper_without_cpu_lod_work():
    session = _zoomed_out_session()
    image = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    rendered = RenderedTile(
        session.plan.tiles[0],
        image,
        image,
        0.0,
        image.shape,
        image.nbytes,
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        semantic_data=image,
    )
    session.mark_materialized(rendered)

    first = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]
    second = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]

    assert second is first


def test_exact_payload_seeds_from_previous_session_without_materialization():
    first_session = _zoomed_out_session()
    image = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    rendered = RenderedTile(
        first_session.plan.tiles[0],
        image,
        image,
        0.0,
        image.shape,
        image.nbytes,
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        semantic_data=image,
    )
    first_session.mark_materialized(rendered)
    source_ids = {0: ("tile", 0)}
    first = first_session.snapshot_display_tile_payloads(source_ids)[0]

    second_session = _zoomed_out_session()
    second_session.mark_materialized(rendered)
    second_session.seed_display_tile_payloads({0: first}, source_ids)
    second = second_session.snapshot_display_tile_payloads(source_ids)[0]

    assert second is first


def test_retargeted_seed_rebuilds_typed_identity_for_current_source():
    original = _session()
    image = np.full((2, 2), 3.0, dtype=np.float32)
    original.mark_materialized(
        RenderedTile(
            original.plan.tiles[0],
            image,
            image,
            0.0,
            image.shape,
            image.nbytes,
        )
    )
    stale_wrapper = original.snapshot_display_tile_payloads({0: ("tile-source", 3)})[0]
    assert stale_wrapper.tile_identity.source_index == 0

    state = ViewState.from_shape((2, 2, 4)).with_montage_axis(
        2,
        indices=(3,),
        text="3",
    )
    plan = make_montage_plan(
        state,
        axis=2,
        indices=(3,),
        tile_shape=(2, 2),
        columns=1,
    )
    current = FrameSession(
        session_id=2,
        key="key",
        render_generation=2,
        level_key="levels",
        level_expected_indices=(3,),
        plan=plan,
        view_state=state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(10, 10),
        view_range=None,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={
            0: RenderedTile(
                plan.tiles[0],
                image,
                image,
                0.0,
                image.shape,
                image.nbytes,
            )
        },
        loading_tiles=set(),
        skipped_tiles=set(),
    )

    current.seed_display_tile_payloads(
        {0: stale_wrapper},
        {0: ("tile-source", 3)},
    )

    seeded = current.display_tile_payloads[0]
    target = current.tile_target_identity(plan.tiles[0], lod_level=0)
    assert seeded.source_index == 3
    assert seeded.tile_identity.source_index == 3
    assert seeded.tile_identity.semantic_key == target.semantic_key
    assert seeded.tile_identity.real_plane.pointer == int(
        seeded.texture_data.__array_interface__["data"][0]
    )
    assert 0 in current.pending_payload_upserts


def test_same_source_seed_rebuilds_typed_identity_for_current_axes():
    """Wrapper seeding must not preserve predecessor view semantics."""

    original = _session()
    image = np.full((2, 2), 3.0, dtype=np.float32)
    original.mark_materialized(
        RenderedTile(
            original.plan.tiles[0],
            image,
            image,
            0.0,
            image.shape,
            image.nbytes,
        )
    )
    source_ids = {0: ("tile-source", 0)}
    predecessor = original.snapshot_display_tile_payloads(source_ids)[0]

    current = _session()
    current.view_state = current.view_state.with_axis_flipped(0, True)
    current.plan = make_montage_plan(
        current.view_state,
        axis=2,
        indices=(0, 1, 2, 3),
        tile_shape=(2, 2),
        columns=4,
    )
    current.visible_tiles = current.plan.tiles
    current.visible_tile_numbers = frozenset(range(4))
    current.mark_materialized(
        RenderedTile(
            current.plan.tiles[0],
            image,
            image,
            0.0,
            image.shape,
            image.nbytes,
        )
    )

    current.seed_display_tile_payloads({0: predecessor}, source_ids, tile_numbers=(0,))

    seeded = current.display_tile_payloads[0]
    assert seeded is not predecessor
    assert seeded.texture_data is predecessor.texture_data
    assert seeded.source_id == predecessor.source_id
    assert seeded.tile_identity.axis_flips == current.view_state.axis_flipped


def test_zoomed_out_complex_payload_keeps_exact_semantics_and_texture():
    session = _zoomed_out_session(dtype=np.complex64, rgb=True)
    real = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    source = (real + 1j * (real + 100.0)).astype(np.complex64)
    histogram = np.log10(np.abs(source)).astype(np.float32)
    rendered = RenderedTile(
        session.plan.tiles[0],
        source,
        histogram,
        0.0,
        source.shape,
        source.nbytes,
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
        semantic_data=source,
        shader_mapping=ShaderMapping(
            component=ShaderComponent.ABS,
            scale=ShaderScale.LOG,
            display_mode=ShaderDisplayMode.PHASE_COLOR,
        ),
    )
    session.mark_materialized(rendered)

    payload = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]
    decision = session.lod_policy_decision

    assert decision.demand.desired_factor > 1
    assert decision.applied_factor == 1
    assert payload.texture_kind == TexturePlaneKind.COMPLEX_RG32F
    assert payload.lod.factor == 1
    assert np.iscomplexobj(payload.texture_data)
    assert payload.texture_data.shape[:2] == payload.image.shape[:2]
    np.testing.assert_array_equal(payload.semantic_data, source)
    np.testing.assert_array_equal(payload.semantic_histogram_data, histogram)
    assert payload.source_shape == (8, 8)


def test_shader_mapping_change_reuses_texture_content_identity():
    session = _zoomed_out_session(dtype=np.complex64, rgb=True)
    source = np.ones((8, 8), dtype=np.complex64)
    histogram = np.ones((8, 8), dtype=np.float32)
    first_mapping = ShaderMapping(
        component=ShaderComponent.ABS,
        scale=ShaderScale.LINEAR,
        display_mode=ShaderDisplayMode.PHASE_COLOR,
    )
    second_mapping = ShaderMapping(
        component=ShaderComponent.ABS,
        scale=ShaderScale.LOG,
        display_mode=ShaderDisplayMode.PHASE_COLOR,
    )
    first_rendered = RenderedTile(
        session.plan.tiles[0],
        source,
        histogram,
        0.0,
        source.shape,
        source.nbytes,
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
        semantic_data=source,
        shader_mapping=first_mapping,
    )
    session.mark_materialized(first_rendered)
    first = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]

    second_rendered = RenderedTile(
        session.plan.tiles[0],
        source,
        histogram,
        0.0,
        source.shape,
        source.nbytes,
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
        semantic_data=source,
        shader_mapping=second_mapping,
    )
    session.mark_materialized(second_rendered)
    second = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]

    assert second is not first
    assert second.source_id == first.source_id
    assert second.texture_data is first.texture_data
    assert second.shader_mapping == second_mapping


def test_retarget_viewport_separates_draw_set_from_loaded_residency():
    state = ViewState.from_shape((2, 2, 8)).with_montage_axis(2, indices=tuple(range(8)), text=":")
    plan = make_montage_plan(state, axis=2, indices=tuple(range(8)), tile_shape=(2, 2), columns=8)
    session = FrameSession(
        session_id=1,
        key="key",
        render_generation=1,
        level_key="levels",
        level_expected_indices=tuple(range(8)),
        plan=plan,
        view_state=state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(1, 1),
        view_range=((0.0, 1.0), (0.0, 1.0)),
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=plan.tiles[:2],
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
    )
    for tile in plan.tiles[:2]:
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))

    additions, changed = session.retarget_viewport(
        view_range=((8.0, 9.0), (0.0, 1.0)),
        viewport_shape=(1, 1),
        coverage_margin_tiles=1,
        near_margin_tiles=2,
    )

    assert changed
    assert tuple(tile.montage_index for tile in session.visible_tiles) == (2, 3)
    assert set(session.rendered_tiles) == {0, 1}
    assert tuple(tile.montage_index for tile in additions) == (2, 3, 4)

    state, delta = session.build_tile_presentation(
        {index: ("tile-source", index) for index in range(8)}
    )
    assert delta.active_tiles == (2, 3)
    assert state.active_payloads(delta) == {}


def test_retarget_viewport_range_change_with_same_tiles_is_camera_only():
    state = ViewState.from_shape((2, 2, 8)).with_montage_axis(2, indices=tuple(range(8)), text=":")
    plan = make_montage_plan(state, axis=2, indices=tuple(range(8)), tile_shape=(2, 2), columns=8)
    session = FrameSession(
        session_id=1,
        key="key",
        render_generation=1,
        level_key="levels",
        level_expected_indices=tuple(range(8)),
        plan=plan,
        view_state=state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(2, 4),
        view_range=((0.0, 4.0), (0.0, 2.0)),
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=(),
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
    )
    source_ids = {}
    for tile in plan.tiles:
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
        source_ids[int(tile.montage_index)] = ("tile-source", int(tile.montage_index))

    additions, changed = session.retarget_viewport(
        view_range=((0.0, 4.0), (0.0, 2.0)),
        viewport_shape=(2, 4),
        coverage_margin_tiles=0,
        near_margin_tiles=0,
    )
    assert additions == ()
    assert changed
    first_state, first_delta = session.build_tile_presentation(source_ids)
    session.acknowledge_tile_presentation(
        first_delta,
        TileCommitReport(
            presented_tiles=first_state.active_payloads(first_delta),
            committed_upserts=frozenset(first_delta.upserts),
        ),
    )
    session.mark_presented(first_state.active_payloads(first_delta))
    first_active = tuple(int(tile.montage_index) for tile in session.visible_tiles)
    first_viewport_revision = int(first_delta.viewport_revision)

    additions, changed = session.retarget_viewport(
        view_range=((0.1, 3.9), (0.0, 2.0)),
        viewport_shape=(2, 4),
        coverage_margin_tiles=0,
        near_margin_tiles=0,
    )

    assert additions == ()
    assert tuple(int(tile.montage_index) for tile in session.visible_tiles) == first_active
    assert not changed
    _state, delta = session.build_tile_presentation(source_ids)
    assert delta.upserts == {}
    assert delta.active_tiles == first_active
    assert delta.viewport_revision == first_viewport_revision + 1
    assert not session.presentation_geometry_changed

    additions, changed = session.retarget_viewport(
        view_range=((0.0, 8.0), (0.0, 2.0)),
        viewport_shape=(2, 8),
        coverage_margin_tiles=0,
        near_margin_tiles=0,
    )

    assert additions == ()
    assert changed, "a resident active-set expansion still needs a visibility commit"
    expanded_state, expanded_delta = session.build_tile_presentation(source_ids)
    assert set(expanded_delta.active_tiles) > set(first_active)
    session.acknowledge_tile_presentation(
        expanded_delta,
        TileCommitReport(
            presented_tiles=expanded_state.active_payloads(expanded_delta),
            committed_upserts=frozenset(expanded_delta.upserts),
        ),
    )
    session.mark_presented(expanded_state.active_payloads(expanded_delta))

    session.retarget_viewport(
        view_range=((0.1, 3.9), (0.0, 2.0)),
        viewport_shape=(2, 4),
        coverage_margin_tiles=0,
        near_margin_tiles=0,
    )
    session.build_tile_presentation(source_ids)
    _additions, changed = session.retarget_viewport(
        view_range=((0.0, 8.0), (0.0, 2.0)),
        viewport_shape=(2, 8),
        coverage_margin_tiles=0,
        near_margin_tiles=0,
    )
    _state, resident_expand = session.build_tile_presentation(source_ids)

    assert changed
    assert resident_expand.upserts == {}


def test_temporary_materialization_gap_does_not_remove_committed_payloads():
    session = _session()
    source_ids = {}
    for tile in session.plan.tiles[:2]:
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
        source_ids[int(tile.montage_index)] = ("tile-source", int(tile.montage_index))
    first_state, first_delta = session.build_tile_presentation(source_ids)
    session.acknowledge_tile_presentation(
        first_delta,
        TileCommitReport(presented_tiles=first_state.active_payloads(first_delta)),
    )
    session.mark_presented(first_state.active_payloads(first_delta))

    session.rendered_tiles.clear()
    session.display_tile_payloads.clear()
    next_state, next_delta = session.build_tile_presentation(source_ids)

    assert next_delta.removals == ()
    assert tuple(next_state.payloads) == (0, 1)
    assert session.ensure_tile_states()[0].value == "loaded"


def test_retarget_viewport_skips_target_settled_payload_tile_as_addition():
    # A tile already settled at its target must not be re-proposed as an
    # addition on a persistent-GPU-residency backend, where rendered_tiles is
    # empty and every pan otherwise re-planned the whole viewport (the settled
    # 400-tile pan slideshow). A tile whose payload does NOT satisfy its target
    # still IS an addition -- on that backend the additions path is how a
    # fallback tile gets its exact re-request, and dropping it strands the tile.
    session = _session()
    session.visible_tile_numbers = frozenset({0, 1, 2, 3})
    # Tiles 0 and 1 are presented exact -> their records are target-settled.
    # rendered_tiles stays empty, mimicking the GPU-resident backend.
    _present_exact_tiles(session, 0, 1)
    assert session.lifecycle.peek(0).target_settled
    assert not session.rendered_tiles

    additions, _changed = session.retarget_viewport(
        view_range=None,
        viewport_shape=(10, 10),
        coverage_margin_tiles=0,
        settled_payloads_are_known=True,
    )

    addition_numbers = {tile.montage_index for tile in additions}
    assert addition_numbers.isdisjoint({0, 1}), (
        "settled payload tiles must not re-enter as additions"
    )
    assert {2, 3} <= addition_numbers, "tiles without a settled payload remain additions"

    # The CPU/nonpersistent path (flag off) leaves the settled tiles as
    # additions -- widening ``known`` there dropped real refinement work.
    cpu_additions, _ = session.retarget_viewport(
        view_range=None,
        viewport_shape=(10, 10),
        coverage_margin_tiles=0,
        settled_payloads_are_known=False,
    )
    assert {0, 1} <= {tile.montage_index for tile in cpu_additions}


def test_retarget_viewport_requests_newly_required_unowned_tile():
    session = _session()
    session.visible_tiles = session.plan.tiles[:2]

    additions, _changed = session.retarget_viewport(
        view_range=((3.0, 4.0), (0.0, 1.0)),
        viewport_shape=(1, 1),
        coverage_margin_tiles=1,
    )

    assert 2 in {tile.montage_index for tile in additions}


def test_retarget_viewport_adopts_replacement_plan_with_same_geometry():
    session = _session()
    previous = session.plan
    # ``make_montage_plan`` memoizes identical layouts and would hand back the
    # very object already on the session. This test is about adopting a
    # DIFFERENT object whose geometry is EQUAL, so force the distinct instance.
    replacement = dataclasses.replace(
        make_montage_plan(
            session.view_state,
            axis=2,
            indices=(0, 1, 2, 3),
            tile_shape=(2, 2),
            columns=4,
        )
    )

    session.retarget_viewport(
        view_range=None,
        viewport_shape=(10, 10),
        plan=replacement,
        coverage_margin_tiles=1,
        near_margin_tiles=2,
    )

    assert replacement is not previous
    assert replacement.geometry == previous.geometry
    assert session.plan is replacement
    assert not session._layout_geometry_changed_pending


def test_layout_reflow_repositions_materialized_tiles_without_payload_upserts():
    state = ViewState.from_shape((2, 2, 6)).with_montage_axis(2, indices=tuple(range(6)), text=":")
    first_plan = make_montage_plan(
        state, axis=2, indices=tuple(range(6)), tile_shape=(2, 2), columns=2
    )
    second_plan = make_montage_plan(
        state, axis=2, indices=tuple(range(6)), tile_shape=(2, 2), columns=3
    )
    session = FrameSession(
        session_id=1,
        key="key",
        render_generation=1,
        level_key="levels",
        level_expected_indices=tuple(range(6)),
        plan=first_plan,
        view_state=state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(100, 100),
        view_range=None,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=first_plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
    )
    source_ids = {}
    for tile in first_plan.tiles:
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
        source_ids[int(tile.montage_index)] = ("tile-source", int(tile.montage_index))

    first_state, first_delta = session.build_tile_presentation(source_ids)
    session.acknowledge_tile_presentation(
        first_delta,
        TileCommitReport(presented_tiles=first_state.active_payloads(first_delta)),
    )
    session.mark_presented(first_state.active_payloads(first_delta))

    additions, changed = session.retarget_viewport(
        view_range=None,
        viewport_shape=(100, 100),
        plan=second_plan,
        coverage_margin_tiles=1,
        near_margin_tiles=2,
    )
    second_state, second_delta = session.build_tile_presentation(source_ids)

    assert changed
    assert additions == ()
    assert session.plan is second_plan
    assert second_delta.upserts == {}
    assert second_delta.removals == ()
    assert session.presentation_geometry_changed
    assert set(second_state.active_payloads(second_delta)) == {0, 1, 2, 3, 4, 5}
    session.acknowledge_tile_presentation(
        second_delta,
        TileCommitReport(presented_tiles=second_state.active_payloads(second_delta)),
    )
    session.mark_presented(second_state.active_payloads(second_delta))

    _clean_state, clean_delta = session.build_tile_presentation(source_ids)

    assert clean_delta.upserts == {}
    assert clean_delta.removals == ()
    assert not session.presentation_geometry_changed


def test_loaded_active_set_change_without_payload_delta_is_not_geometry_change():
    session = _session()
    source_ids = {}
    for tile in session.plan.tiles[:2]:
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
        source_ids[int(tile.montage_index)] = ("tile-source", int(tile.montage_index))

    first_state, first_delta = session.build_tile_presentation(source_ids)
    session.acknowledge_tile_presentation(
        first_delta,
        TileCommitReport(
            presented_tiles=first_state.active_payloads(first_delta),
            committed_upserts=frozenset(first_delta.upserts),
        ),
    )
    session.mark_presented(first_state.active_payloads(first_delta))
    # Simulate a viewport-scoped active-set expansion over already-presented
    # payloads. The active scope should update for the next real delta, but
    # it is not itself layout/viewport geometry that warrants a backend patch.
    session._last_active_tiles = (0,)

    _state, delta = session.build_tile_presentation(source_ids)

    assert delta.upserts == {}
    assert delta.removals == ()
    assert delta.active_tiles == (0, 1, 2, 3)
    assert session.visibility_revision == first_delta.visibility_revision + 1
    assert not session.presentation_geometry_changed


def test_montage_render_session_passes_cold_deadline_without_slicing_upserts():
    session = _session()
    source_ids = {}
    for tile in session.plan.tiles:
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
        source_ids[int(tile.montage_index)] = ("tile-source", int(tile.montage_index))

    first_state, first_delta = session.build_tile_presentation(source_ids, cold_deadline_ms=3.5)

    assert tuple(first_delta.upserts) == (0, 1, 2, 3)
    assert tuple(first_state.payloads) == (0, 1, 2, 3)
    assert first_delta.cold_deadline_ms == 3.5
    assert not session.is_complete()

    session.acknowledge_tile_presentation(
        first_delta,
        TileCommitReport(presented_tiles=first_state.active_payloads(first_delta)),
    )
    session.mark_presented(first_state.active_payloads(first_delta))
    session.note_committed()
    assert session.is_complete()


def test_montage_render_session_tile_states_keep_materialized_tiles_loading_until_presented():
    session = _session()
    session.visible_tiles = (session.plan.tiles[3], session.plan.tiles[1])
    source_ids = {}
    for tile in session.plan.tiles:
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
        source_ids[int(tile.montage_index)] = ("tile-source", int(tile.montage_index))

    state, delta = session.build_tile_presentation(source_ids)
    tile_states = session.ensure_tile_states()

    assert tuple(delta.upserts) == (1, 3)
    assert tuple(state.active_payloads(delta)) == (3, 1)
    assert {str(tile_states[index].value) for index in range(4)} == {"loading"}

    session.acknowledge_tile_presentation(
        delta,
        TileCommitReport(presented_tiles=state.active_payloads(delta)),
    )
    session.mark_presented(state.active_payloads(delta))
    tile_states = session.ensure_tile_states()

    assert {index for index in range(4) if tile_states[index].value == "loaded"} == {1, 3}
    assert {index for index in range(4) if tile_states[index].value == "loading"} == {0, 2}


def test_montage_render_session_reuses_tile_state_tuple_until_revision_changes():
    session = _session()

    first = session.ensure_tile_states()
    second = session.ensure_tile_states()
    assert second is first

    session.mark_loading(session.plan.tiles[0])
    third = session.ensure_tile_states()
    fourth = session.ensure_tile_states()

    assert third is not first
    assert fourth is third
    assert third[0].value == "loading"


def test_montage_overlay_refresh_caches_empty_and_repeated_state():
    from arrayscope.window.frame_controller import FrameControllerMixin

    class ImageView:
        def __init__(self):
            self.overlays = ()
            self.calls = 0

        def setMontageTileOverlays(self, overlays):
            self.overlays = tuple(overlays or ())
            self.calls += 1

        def montageTileOverlayCount(self):
            return len(self.overlays)

    session = _session()
    image_view = ImageView()
    owner = SimpleNamespace(img_view=image_view, _frame_session=session)
    owner.win = owner
    rect = (0, 0, 20, 20)

    FrameControllerMixin._update_montage_tile_overlays_for_plan(
        owner,
        session.plan,
        session.ensure_tile_states(),
        rect,
    )
    assert image_view.calls == 0

    session.mark_skipped(session.plan.tiles[1])
    FrameControllerMixin._update_montage_tile_overlays_for_plan(
        owner,
        session.plan,
        session.ensure_tile_states(),
        rect,
    )
    assert image_view.calls == 1
    assert len(image_view.overlays) == 1

    FrameControllerMixin._update_montage_tile_overlays_for_plan(
        owner,
        session.plan,
        session.ensure_tile_states(),
        rect,
    )
    assert image_view.calls == 1


def test_tile_truth_overlay_rows_include_each_plan_tile_rectangle():
    from arrayscope.window.frame_runtime import FrameRuntimeMixin

    class ImageView:
        def __init__(self):
            self.rows = ()

        def setTileTruthOverlayRows(self, rows):
            self.rows = tuple(rows or ())

        def tileTruthPhysicalRows(self):
            return {
                int(tile.montage_index): {
                    "physical_texture_kind": "scalar_r32f",
                    "physical_mapping_mode": 0.0,
                }
                for tile in session.plan.tiles
            }

    session = _session()
    image_view = ImageView()
    owner = SimpleNamespace(
        img_view=image_view,
        _frame_session=session,
        _tile_truth_overlay_enabled=True,
        _frame_session_is_current=lambda candidate: candidate is session,
    )
    owner.win = owner

    FrameRuntimeMixin._refresh_tile_truth_overlay(owner)

    expected = {
        int(tile.montage_index): (int(tile.x0), int(tile.y0), int(tile.width), int(tile.height))
        for tile in session.plan.tiles
    }
    assert {int(row["tile"]): row["tile_rect"] for row in image_view.rows} == expected
    assert all(row["physical_texture_kind"] == "scalar_r32f" for row in image_view.rows)


def test_montage_loading_state_does_not_create_per_tile_scene_overlays():
    from arrayscope.window.frame_controller import FrameControllerMixin

    class ImageView:
        def __init__(self):
            self.overlays = ()
            self.calls = 0

        def setMontageTileOverlays(self, overlays):
            self.overlays = tuple(overlays or ())
            self.calls += 1

        def montageTileOverlayCount(self):
            return len(self.overlays)

    session = _session()
    session.show_loading_overlays = True
    session.mark_loading(session.plan.tiles[0])
    image_view = ImageView()
    owner = SimpleNamespace(img_view=image_view, _frame_session=session)
    owner.win = owner
    rect = (0, 0, 20, 20)

    FrameControllerMixin._update_montage_tile_overlays_for_plan(
        owner,
        session.plan,
        session.ensure_tile_states(),
        rect,
    )
    assert image_view.calls == 0

    session.mark_loading(session.plan.tiles[1])
    FrameControllerMixin._update_montage_tile_overlays_for_plan(
        owner,
        session.plan,
        session.ensure_tile_states(),
        rect,
    )
    assert image_view.calls == 0


def test_level_presentation_finish_reuses_settled_generation():
    session = _session()
    session.level_generation.begin_target((2.0, 4.0), active_tiles=(0, 1))
    revision = session.level_revision
    session.level_generation.acknowledge_upserts(revision, (0, 1), levels=(2.0, 4.0))
    session.set_level_update_pending(False)

    assert session.begin_level_presentation_update((2.0, 4.0)) is False
    assert session.level_revision == revision
    assert session.has_pending_level_update() is False
    assert session.level_presentation_snapshot().stale_count == 0


def test_level_presentation_finish_drains_existing_generation_without_revising():
    session = _session()
    session.level_generation.begin_target((2.0, 4.0), active_tiles=(0, 1))
    revision = session.level_revision
    session.level_generation.acknowledge_upserts(revision, (0,), levels=(2.0, 4.0))
    session.level_generation.acknowledge_upserts(revision - 1, (1,), levels=(0.0, 1.0))
    session.level_generation.tile_values[1] = (0.0, 1.0)
    session.level_generation.tile_revisions[1] = revision - 1
    session.level_generation.set_active_tiles((0, 1))

    assert session.begin_level_presentation_update((2.0, 4.0)) is True
    assert session.level_revision == revision
    assert session.has_pending_level_update() is True
    assert session.level_presentation_snapshot().stale_count == 1


def test_level_scope_growth_reopens_pending_target_for_new_active_tile():
    session = _session()
    session.visible_tiles = (session.plan.tiles[0],)
    session.display_tile_payloads = {0: object()}
    session.lifecycle.presentation_confirmed((0,))
    session.level_generation.set_active_tiles((0,))

    assert session.begin_level_presentation_update((2.0, 4.0)) is True
    session.level_generation.acknowledge_upserts(
        session.level_revision,
        (0,),
        levels=(2.0, 4.0),
    )
    session.set_level_update_pending(False)

    session.visible_tiles = (session.plan.tiles[0], session.plan.tiles[1])
    session.display_tile_payloads[1] = object()
    session.lifecycle.presentation_confirmed((1,))
    session.update_level_presentation_scope()

    assert session.has_pending_level_update() is True
    assert session.level_presentation_snapshot().stale_count == 1


def test_shader_level_acknowledgement_settles_all_active_tiles():
    session = _session()
    session.level_generation.set_active_tiles((0, 2, 3))
    assert session.begin_level_presentation_update((2.0, 4.0)) is True

    session.acknowledge_uniform_level_presentation((2.0, 4.0))

    assert session.has_pending_level_update() is False
    assert session.level_presentation_snapshot().stale_count == 0
    assert session.level_generation.value_counts() == {(2.0, 4.0): 3}
    assert session.level_generation.tile_values == {0: (2.0, 4.0), 2: (2.0, 4.0), 3: (2.0, 4.0)}
    assert session.level_generation.tile_revisions == {0: 1, 2: 1, 3: 1}


def test_shader_level_acknowledgement_cannot_replace_the_target():
    session = _session()
    session.level_generation.set_active_tiles((0, 2, 3))
    assert session.begin_level_presentation_update((2.0, 4.0)) is True

    session.acknowledge_uniform_level_presentation((1.0, 5.0))

    assert session.level_generation.target_levels == (2.0, 4.0)
    assert session.level_generation.revision == 1
    assert session.has_pending_level_update() is True
    assert session.level_presentation_snapshot().stale_count == 3


def test_shader_preview_payload_with_level_evidence_enters_level_scope():
    session = _session()
    session.shader_display = True
    tile = session.plan.tiles[0]
    payload = DisplayTilePayload(
        tile_number=0,
        source_index=int(tile.source_index),
        image=np.ones((2, 2), dtype=np.complex64),
        histogram_data=None,
        source_id=session.tile_semantic_source_id(tile.source_index),
        level_data=np.asarray([2.0, 8.0], dtype=np.float32),
        quality="preview",
        lod=LodInfo(level=2, factor=4, source_shape=(2, 2), texture_shape=(2, 2)),
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
    )
    session.display_tile_payloads[0] = payload
    session.lifecycle.acknowledge_presented(0, payload.source_id, payload.quality, 2)

    session.update_level_presentation_scope()

    assert session.level_generation.active_tiles == frozenset({0})
    assert session.begin_level_presentation_update((2.0, 8.0)) is True
    session.acknowledge_uniform_level_presentation((2.0, 8.0))
    assert session.level_generation.value_counts() == {(2.0, 8.0): 1}


def test_cpu_preview_payload_stays_out_of_level_scope():
    session = _session()
    session.shader_display = False
    tile = session.plan.tiles[0]
    payload = DisplayTilePayload(
        tile_number=0,
        source_index=int(tile.source_index),
        image=np.ones((2, 2), dtype=np.float32),
        histogram_data=np.ones((2, 2), dtype=np.float32),
        source_id=session.tile_semantic_source_id(tile.source_index),
        level_data=np.asarray([2.0, 8.0], dtype=np.float32),
        quality="preview",
        lod=LodInfo(level=2, factor=4, source_shape=(2, 2), texture_shape=(2, 2)),
    )
    session.display_tile_payloads[0] = payload
    session.lifecycle.acknowledge_presented(0, payload.source_id, payload.quality, 2)

    session.update_level_presentation_scope()

    assert session.level_generation.active_tiles == frozenset()


def test_stale_level_delta_cannot_acknowledge_newer_target():
    session = _session()
    source_ids = {}
    for tile in session.plan.tiles[:2]:
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
        source_ids[int(tile.montage_index)] = ("tile", int(tile.montage_index))

    first_state, first_delta = session.build_tile_presentation(source_ids)
    session.acknowledge_tile_presentation(
        first_delta,
        TileCommitReport(presented_tiles=first_state.active_payloads(first_delta)),
        levels=(0.0, 1.0),
    )
    session.mark_presented(first_state.active_payloads(first_delta))

    assert session.begin_level_presentation_update((2.0, 4.0)) is True
    _old_state, old_delta = session.build_tile_presentation(source_ids)
    assert old_delta.level_revision == 1

    assert session.begin_level_presentation_update((3.0, 5.0)) is True
    session.acknowledge_tile_presentation(
        old_delta,
        TileCommitReport(presented_tiles=(0, 1), committed_upserts=(0, 1)),
        levels=(2.0, 4.0),
    )

    snapshot = session.level_presentation_snapshot()
    assert snapshot.revision == 2
    assert snapshot.target_levels == (3.0, 5.0)
    assert snapshot.stale_count == 2
    assert snapshot.pending_count == 2
    assert snapshot.settled is False
    assert session.level_generation.tile_values == {0: (0.0, 1.0), 1: (0.0, 1.0)}


def test_level_snapshot_tracks_active_set_changes_during_convergence():
    session = _session()
    session.level_revision = 4
    session.visible_tile_numbers = frozenset({0, 1, 2})
    session.lifecycle.presentation_confirmed((0, 1, 2))
    session.display_tile_payloads = {index: object() for index in range(3)}
    session.level_generation.tile_values = {
        0: (0.0, 1.0),
        1: (2.0, 4.0),
        2: (0.0, 1.0),
    }
    session.level_generation.tile_revisions = {0: 3, 1: 4, 2: 3}
    session.visible_tiles = tuple(session.plan.tiles[:3])
    session.level_generation.target_levels = (2.0, 4.0)
    session.set_level_update_pending(True)
    session.update_level_presentation_scope()

    snapshot = session.level_presentation_snapshot()
    assert snapshot.active_presented_tile_count == 3
    assert snapshot.stale_count == 2

    session.visible_tiles = (session.plan.tiles[1],)
    session.visible_tile_numbers = frozenset({1})
    session.update_level_presentation_scope()

    snapshot = session.level_presentation_snapshot()
    assert snapshot.active_tile_count == 1
    assert snapshot.active_presented_tile_count == 1
    assert snapshot.stale_count == 0
    assert snapshot.pending_count == 0
    assert snapshot.settled is True


def test_level_snapshot_tracks_tile_entering_active_set_during_convergence():
    session = _session()
    session.level_revision = 4
    session.visible_tile_numbers = frozenset({0})
    session.lifecycle.presentation_confirmed((0, 1))
    session.display_tile_payloads = {0: object(), 1: object()}
    session.level_generation.tile_values = {
        0: (2.0, 4.0),
        1: (0.0, 1.0),
    }
    session.level_generation.tile_revisions = {0: 4, 1: 3}
    session.visible_tiles = (session.plan.tiles[0],)
    session.level_generation.target_levels = (2.0, 4.0)
    session.set_level_update_pending(True)
    session.update_level_presentation_scope()

    snapshot = session.level_presentation_snapshot()
    assert snapshot.stale_count == 0
    assert snapshot.pending_count == 0
    assert snapshot.settled is True

    session.visible_tiles = (session.plan.tiles[0], session.plan.tiles[1])
    session.visible_tile_numbers = frozenset({0, 1})
    session.update_level_presentation_scope()

    snapshot = session.level_presentation_snapshot()
    assert snapshot.active_presented_tile_count == 2
    assert snapshot.stale_count == 1
    assert snapshot.pending_count == 1
    assert snapshot.settled is False


def test_montage_render_session_commits_ready_payloads_atomically():
    session = _session()
    source_ids = {}
    for tile in session.plan.tiles:
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
        source_ids[int(tile.montage_index)] = ("tile-source", int(tile.montage_index))

    state, delta = session.build_tile_presentation(source_ids)

    assert not delta.force_refresh
    assert delta.clear_reason == ""
    assert tuple(delta.upserts) == (0, 1, 2, 3)
    assert tuple(state.payloads) == (0, 1, 2, 3)


def test_tile_presentation_state_rejects_stale_delta():
    session = _session()
    tile = session.plan.tiles[0]
    image = np.ones((2, 2), dtype=np.float32)
    session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
    state, delta = session.build_tile_presentation({0: ("tile-source", 0)})

    stale = type(delta)(
        structure_revision=delta.structure_revision,
        payload_revision=delta.payload_revision + 1,
        visibility_revision=delta.visibility_revision,
        level_revision=delta.level_revision,
        histogram_revision=delta.histogram_revision,
        viewport_revision=delta.viewport_revision,
        base_revision=delta.base_revision,
        target_revision=delta.target_revision + 1,
        upserts=delta.upserts,
    )

    assert state.revision == delta.target_revision
    assert state.apply_delta(stale) is state


def test_seeded_payloads_retain_committed_state_across_retarget():
    original = _session()
    tile = original.plan.tiles[2]
    image = np.full((2, 2), 2, dtype=np.float32)
    rendered = RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes)
    original.mark_materialized(rendered)
    source_ids = {2: ("tile-source", 2)}
    state, delta = original.build_tile_presentation(source_ids)
    original.acknowledge_tile_presentation(
        delta,
        TileCommitReport(presented_tiles=state.active_payloads(delta)),
    )
    original.mark_presented(state.active_payloads(delta))

    shifted_state = ViewState.from_shape((2, 2, 4)).with_montage_axis(2, indices=(2, 3), text="2:4")
    shifted_plan = make_montage_plan(
        shifted_state, axis=2, indices=(2, 3), tile_shape=(2, 2), columns=2
    )
    shifted_rendered = RenderedTile(
        shifted_plan.tiles[0],
        image,
        image,
        0.0,
        image.shape,
        image.nbytes,
    )
    shifted = FrameSession(
        session_id=2,
        key="key",
        render_generation=1,
        level_key="levels",
        level_expected_indices=(2, 3),
        plan=shifted_plan,
        view_state=shifted_state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(10, 10),
        view_range=None,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=shifted_plan.tiles,
        rendered_tiles={0: shifted_rendered},
        loading_tiles=set(),
        skipped_tiles=set(),
    )

    shifted.seed_display_tile_payloads(state.payloads, {0: ("tile-source", 2)})
    next_state, next_delta = shifted.build_tile_presentation(
        {0: ("tile-source", 2), 1: ("tile-source", 3)}
    )

    assert 0 in next_state.payloads
    assert next_state.payloads[0].source_index == 2
    assert next_delta.removals == ()
    assert tuple(next_delta.upserts) == (0,)
    shifted.acknowledge_tile_presentation(
        next_delta,
        TileCommitReport(presented_tiles=next_state.active_payloads(next_delta)),
    )
    assert 0 not in shifted.pending_payload_upserts
    assert 0 in shifted.lifecycle.presented_tiles


def test_seeded_resident_payloads_reuse_base_identity_without_texture_lookup(monkeypatch):
    original = _session()
    tile = original.plan.tiles[2]
    image = np.full((2, 2), 2, dtype=np.float32)
    rendered = RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes)
    original.mark_materialized(rendered)
    state, delta = original.build_tile_presentation({2: ("tile-source", 2)})
    original.acknowledge_tile_presentation(
        delta,
        TileCommitReport(presented_tiles=state.active_payloads(delta)),
    )
    original.mark_presented(state.active_payloads(delta))

    shifted_state = ViewState.from_shape((2, 2, 4)).with_montage_axis(2, indices=(2, 3), text="2:4")
    shifted_plan = make_montage_plan(
        shifted_state, axis=2, indices=(2, 3), tile_shape=(2, 2), columns=2
    )
    shifted = FrameSession(
        session_id=2,
        key="key",
        render_generation=1,
        level_key="levels",
        level_expected_indices=(2, 3),
        plan=shifted_plan,
        view_state=shifted_state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(10, 10),
        view_range=None,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=shifted_plan.tiles,
        rendered_tiles={
            0: RenderedTile(shifted_plan.tiles[0], image, image, 0.0, image.shape, image.nbytes)
        },
        loading_tiles=set(),
        skipped_tiles=set(),
    )
    shifted.source_anchoring = SimpleNamespace(
        source_starts_yx=(0, 0),
        content_key=("current-windowless-view",),
    )
    monkeypatch.setattr(shifted, "_resident_lod_active", lambda: True)
    monkeypatch.setattr(
        shifted,
        "_texture_for_rendered_tile",
        lambda _rendered: (_ for _ in ()).throw(
            AssertionError("texture lookup should not seed resident payload")
        ),
    )

    shifted.seed_display_tile_payloads(
        state.payloads,
        {0: ("tile-source", 2)},
        tile_numbers=(0,),
    )

    assert shifted.display_tile_payloads[0].source_index == 2
    assert shifted.display_tile_payloads[0].source_anchor.content_key == (
        ("current-windowless-view",),
        "montage-source",
        2,
    )
    assert 0 not in shifted.tile_presentation_state.payloads
    assert 0 in shifted.pending_payload_upserts
    assert 0 not in shifted.lifecycle.presented_tiles


def test_seeded_payloads_only_confirm_when_backend_identity_matches():
    original = _session()
    tile = original.plan.tiles[2]
    image = np.full((2, 2), 2, dtype=np.float32)
    rendered = RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes)
    original.mark_materialized(rendered)
    state, delta = original.build_tile_presentation({2: ("tile-source", 2)})
    original.acknowledge_tile_presentation(
        delta,
        TileCommitReport(presented_tiles=state.active_payloads(delta)),
    )
    original.mark_presented(state.active_payloads(delta))

    shifted_state = ViewState.from_shape((2, 2, 4)).with_montage_axis(2, indices=(2, 3), text="2:4")
    shifted_plan = make_montage_plan(
        shifted_state, axis=2, indices=(2, 3), tile_shape=(2, 2), columns=2
    )
    shifted = FrameSession(
        session_id=2,
        key="key",
        render_generation=1,
        level_key="levels",
        level_expected_indices=(2, 3),
        plan=shifted_plan,
        view_state=shifted_state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(10, 10),
        view_range=None,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=shifted_plan.tiles,
        rendered_tiles={
            0: RenderedTile(shifted_plan.tiles[0], image, image, 0.0, image.shape, image.nbytes)
        },
        loading_tiles=set(),
        skipped_tiles=set(),
    )
    payload = state.payloads[2]
    shifted.lifecycle.backend_presented_snapshot({0: payload.tile_identity})

    shifted.seed_display_tile_payloads(
        state.payloads,
        {0: ("tile-source", 2)},
        tile_numbers=(0,),
    )

    assert shifted.tile_presentation_state.payloads[0] is shifted.display_tile_payloads[0]
    assert 0 not in shifted.pending_payload_upserts
    assert 0 in shifted.lifecycle.presented_tiles


def test_resident_retarget_upserts_bypass_cold_priority_cap():
    original = _session()
    images = {index: np.full((2, 2), float(index), dtype=np.float32) for index in range(7)}
    original_sources = {index: ("tile-source", index) for index in range(4)}
    for index in range(4):
        image = images[index]
        original.mark_materialized(
            RenderedTile(
                original.plan.tiles[index],
                image,
                image,
                0.0,
                image.shape,
                image.nbytes,
            )
        )
    state, delta = original.build_tile_presentation(original_sources)
    original.acknowledge_tile_presentation(
        delta,
        TileCommitReport(presented_tiles=state.active_payloads(delta)),
    )
    original.mark_presented(state.active_payloads(delta))

    shifted_state = ViewState.from_shape((2, 2, 7)).with_montage_axis(
        2, indices=(3, 4, 5, 6), text="3:7"
    )
    shifted_plan = make_montage_plan(
        shifted_state, axis=2, indices=(3, 4, 5, 6), tile_shape=(2, 2), columns=4
    )
    shifted = FrameSession(
        session_id=2,
        key="key",
        render_generation=1,
        level_key="levels",
        level_expected_indices=(3, 4, 5, 6),
        plan=shifted_plan,
        view_state=shifted_state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(10, 10),
        view_range=None,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=shifted_plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
    )
    shifted_sources = {tile: ("tile-source", tile + 3) for tile in range(4)}
    for tile_number, source_index in enumerate((3, 4, 5, 6)):
        image = images[source_index]
        shifted.mark_materialized(
            RenderedTile(
                shifted_plan.tiles[tile_number],
                image,
                image,
                0.0,
                image.shape,
                image.nbytes,
            )
        )

    shifted.seed_display_tile_payloads(state.payloads, shifted_sources)
    _next_state, next_delta = shifted.build_tile_presentation(
        shifted_sources,
        max_upserts=0,
    )

    assert 0 in next_delta.upserts
    assert next_delta.upserts[0].source_index == 3
    assert tuple(next_delta.upserts) == (0,)


def _priority_session():
    from arrayscope.operations.stage_fanin import StageFanInState

    state = ViewState.from_shape((2, 2, 4)).with_montage_axis(2, indices=(0, 1, 2, 3), text=":")
    plan = make_montage_plan(state, axis=2, indices=(0, 1, 2, 3), tile_shape=(2, 2), columns=4)
    return FrameSession(
        session_id=1,
        key="key",
        render_generation=1,
        level_key="levels",
        level_expected_indices=(0, 1, 2, 3),
        plan=plan,
        view_state=state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(10, 10),
        view_range=((0.0, 12.0), (0.0, 4.0)),
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        stage_fan_in=StageFanInState(),
        priority_focus=(8.0, 1.0),
    )


def test_montage_prefetch_candidates_prefer_focus_proximity():
    from arrayscope.window.montage_prefetch import _candidate_tiles

    session = _priority_session()
    session.visible_tiles = ()
    session.visible_tile_numbers = frozenset()

    ordered = [int(tile.montage_index) for tile in _candidate_tiles(session)]
    assert ordered == [2, 3, 1, 0]


def test_layout_reflow_retargets_priority_context_to_new_plan():
    session = _priority_session()

    state = ViewState.from_shape((2, 2, 4)).with_montage_axis(2, indices=(0, 1, 2, 3), text=":")
    reflowed = make_montage_plan(state, axis=2, indices=(0, 1, 2, 3), tile_shape=(2, 2), columns=2)
    assert reflowed.geometry != session.plan.geometry

    session.retarget_viewport(
        view_range=((0.0, 6.0), (0.0, 6.0)),
        viewport_shape=(10, 10),
        plan=reflowed,
    )

    assert session.plan is reflowed
    context = session.tile_priority_context()
    assert context.visible_tiles == session.visible_tile_numbers
    assert session.priority_retargeted_tiles == len(context.near_tiles)


@pytest.mark.parametrize("jsonl_sink", [True, False], ids=["with-jsonl-sink", "ring-only"])
def test_stranded_required_tile_emits_stall_trace_dump_and_visible_diagnostic(
    tmp_path, monkeypatch, qtbot, jsonl_sink
):
    """The dump must be complete and parseable from the ring alone.

    `_ensure_montage_watchdog` arms a ring-only bus in normal production, and
    that bus no longer encodes anything until `dump` — so the ring-only case
    is the one a real stall exercises, not the `--trace` case.
    """

    import json
    from pathlib import Path

    from pyqtgraph.Qt import QtWidgets

    from arrayscope.core.trace import close_trace, configure_trace
    from arrayscope.window import frame_runtime
    from arrayscope.window.frame_runtime import FrameRuntimeMixin

    monkeypatch.setattr(frame_runtime, "perf_counter", lambda: 12.1)
    dump_directory = tmp_path / "stall-dumps"
    monkeypatch.setenv("ARRAYSCOPE_STALL_DUMP_DIR", str(dump_directory))

    class Completions:
        @staticmethod
        def empty():
            return True

    kernel = SimpleNamespace(
        diagnostics=lambda: SimpleNamespace(
            queued=0,
            running=0,
            active=0,
            parked_deps=0,
            parked_quota=0,
        ),
        completions=Completions(),
    )
    session_id = abs(hash(str(tmp_path))) % 1_000_000 + 10_000
    session = SimpleNamespace(
        session_id=session_id,
        lifecycle=SimpleNamespace(evaluating_tiles=frozenset(), presented_tiles=frozenset()),
        active_tile_requests=frozenset(),
        dirty_payloads={},
        pending_payload_upserts={},
        pending_rung_materializations=(),
        stage_planning_deferred=False,
        pending_level_tiles=(),
        level_scan_remaining_tiles=0,
        semantic_level_evidence_progress=None,
        has_pending_level_update=lambda: False,
        required_target_unsettled_tiles=lambda: (5,),
        flush_pending=False,
        final_commit_pending=False,
        rendered_tiles={},
        diagnostic_tile_identity_rows=lambda **_kwargs: (
            {"tile": 5, "visible_first_pixel_complete": False},
        ),
        stage_fan_in=SimpleNamespace(
            active_requests=frozenset(),
            attached_requests=frozenset(),
            tile_stage_keys={},
        ),
        loading_tiles=frozenset(),
    )
    window = QtWidgets.QMainWindow()
    window.kernel = kernel
    qtbot.addWidget(window)

    class Renderer(FrameRuntimeMixin):
        def __init__(self):
            self.win = window
            self._frame_session = session
            self._montage_watchdog_state = (
                session_id,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                # Commit-progress terms: commit_batches and presentation-state
                # revision (both absent on this stub session, so 0).
                0,
                0,
            )
            self._montage_watchdog_state_since = 10.0

        @staticmethod
        def _frame_session_is_current(_session):
            return True

    configure_trace(tmp_path / "live.trace.jsonl" if jsonl_sink else None)
    renderer = Renderer()
    try:
        renderer._montage_watchdog_tick()
    finally:
        close_trace()

    dump_path = Path(renderer._montage_watchdog_last_trace_path)
    try:
        assert dump_path.parent == dump_directory
        rows = [json.loads(line) for line in dump_path.read_text().splitlines()]
        stall = next(row for row in rows if row.get("kind") == "stall")
        assert stall["session_id"] == session_id
        assert stall["owner_chain"]["required_unsettled"] == [5]
        assert stall["stalled_ms"] >= 2000.0
        label = window._arrayscope_status_message_widget
        assert str(dump_path) in label.property("arrayscope_status_message_text")
        assert window._arrayscope_status_message_timer is None
    finally:
        dump_path.unlink(missing_ok=True)


def test_watchdog_commit_progress_suppresses_stall_assertion(monkeypatch, qtbot):
    """A slow-but-live drain keeps every debt count constant across ticks
    (one upsert enters as one leaves) while commit batches keep landing; the
    monotonic commit-progress terms must keep the watchdog from firing
    (field session 2026-07-15: 22 Hz one-upsert batches read as a stall)."""

    import importlib

    from pyqtgraph.Qt import QtWidgets

    # Resolve through sys.modules, not the package attribute: the conftest
    # module-identity snapshot can leave a stale ``arrayscope.window``
    # attribute pointing at a purged module object when frame_runtime was
    # first imported inside an earlier test, and patching the stale object
    # would miss the module FrameRuntimeMixin actually executes in.
    frame_runtime = importlib.import_module("arrayscope.window.frame_runtime")
    FrameRuntimeMixin = frame_runtime.FrameRuntimeMixin

    monkeypatch.setattr(frame_runtime, "perf_counter", lambda: 12.1)

    session_id = 424_242
    session = SimpleNamespace(
        session_id=session_id,
        lifecycle=SimpleNamespace(evaluating_tiles=frozenset(), presented_tiles=frozenset()),
        active_tile_requests=frozenset(),
        dirty_payloads={},
        pending_payload_upserts={},
        pending_rung_materializations=(),
        stage_planning_deferred=False,
        pending_level_tiles=(),
        level_scan_remaining_tiles=0,
        semantic_level_evidence_progress=None,
        has_pending_level_update=lambda: False,
        required_target_unsettled_tiles=lambda: (5,),
        flush_pending=False,
        final_commit_pending=False,
        rendered_tiles={},
        stage_fan_in=SimpleNamespace(
            active_requests=frozenset(),
            attached_requests=frozenset(),
            tile_stage_keys={},
        ),
        loading_tiles=frozenset(),
        commit_batches=7,
        tile_presentation_state=SimpleNamespace(revision=3),
    )
    window = QtWidgets.QMainWindow()
    qtbot.addWidget(window)

    class Renderer(FrameRuntimeMixin):
        def __init__(self):
            self.win = window
            self._frame_session = session
            # Previous tick observed the same debt counts but OLDER commit
            # progress (commit_batches=6): commits are still landing.
            self._montage_watchdog_state = (
                session_id,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                6,
                3,
            )
            self._montage_watchdog_state_since = 10.0

        @staticmethod
        def _frame_session_is_current(_session):
            return True

    renderer = Renderer()
    renderer._montage_watchdog_tick()

    assert getattr(renderer, "_montage_stall_assertions", 0) == 0
    assert getattr(renderer, "_montage_watchdog_last_trace_path", None) is None
    # The tick re-based the progress clock on the new signature.
    assert renderer._montage_watchdog_state[-2:] == (7, 3)
    assert renderer._montage_watchdog_state_since == 12.1


def test_settled_tiles_with_stale_committed_frame_emit_stall_probe(
    tmp_path,
    monkeypatch,
    qtbot,
    capsys,
):
    """An empty tile backlog must not hide stale committed-frame semantics."""

    import json
    from pathlib import Path

    from pyqtgraph.Qt import QtWidgets

    from arrayscope.core.trace import close_trace, configure_trace
    from arrayscope.window import frame_runtime
    from arrayscope.window.frame_runtime import FrameRuntimeMixin

    monkeypatch.setattr(frame_runtime, "perf_counter", lambda: 12.1)
    monkeypatch.setenv("ARRAYSCOPE_STALL_DUMP_DIR", str(tmp_path))

    class Completions:
        @staticmethod
        def empty():
            return True

    kernel = SimpleNamespace(
        diagnostics=lambda: SimpleNamespace(
            queued=0,
            running=0,
            active=0,
            parked_deps=0,
            parked_quota=0,
        ),
        completions=Completions(),
    )
    session = SimpleNamespace(
        session_id=515_151,
        lifecycle=SimpleNamespace(
            evaluating_tiles=frozenset(),
            presented_tiles=frozenset({0}),
        ),
        active_tile_requests=frozenset(),
        dirty_payloads={},
        pending_payload_upserts={},
        pending_rung_materializations=(),
        stage_planning_deferred=False,
        pending_level_tiles=(),
        level_scan_remaining_tiles=0,
        semantic_level_evidence_progress=None,
        has_pending_level_update=lambda: False,
        required_target_unsettled_tiles=lambda: (),
        required_first_pixels_presented=lambda: True,
        flush_pending=False,
        final_commit_pending=False,
        rendered_tiles={0: object()},
        diagnostic_tile_identity_rows=lambda **_kwargs: (),
        stage_fan_in=SimpleNamespace(
            active_requests=frozenset(),
            attached_requests=frozenset(),
            tile_stage_keys={},
        ),
        loading_tiles=frozenset(),
    )
    window = QtWidgets.QMainWindow()
    window.kernel = kernel
    window._committed_display_frame = object()
    qtbot.addWidget(window)

    class Renderer(FrameRuntimeMixin):
        def __init__(self):
            self.win = window
            self._frame_session = session
            self._montage_watchdog_state = (
                515_151,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
            )
            self._montage_watchdog_state_since = 10.0

        @staticmethod
        def _frame_session_is_current(_session):
            return True

        @staticmethod
        def _is_committed_display_frame_current(_frame):
            return False

    configure_trace(tmp_path / "stale-frame.trace.jsonl")
    renderer = Renderer()
    try:
        renderer._montage_watchdog_tick()
    finally:
        close_trace()

    assert renderer._montage_stall_assertions == 1
    assert "STALL ASSERTION PROBE FIRED" in capsys.readouterr().err
    dump_path = Path(renderer._montage_watchdog_last_trace_path)
    rows = [json.loads(line) for line in dump_path.read_text().splitlines()]
    stall = next(row for row in rows if row.get("kind") == "stall")
    assert stall["owner_chain"]["committed_frame_stale"] is True
    dump_path.unlink(missing_ok=True)
