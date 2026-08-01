"""The retarget release pass must cost what changed, not what was ever shown.

``TileLifecycle._records`` is never pruned by design (park/re-arm and
``presentable_payloads`` both depend on a slot's record outliving its scope),
so the release loop's input grows to every tile the session has ever touched
while a pan changes one or two of them.  These tests pin the two halves of
that separately:

- the *accounting* half — how many records the pass mutates, asserted at a
  scale where "all of them" and "the ones that changed" are different numbers.
  A montage-sized fixture where every record leaves scope at once cannot fail
  this way, which is why the window here is a small fraction of the montage.
- the *equivalence* half — that skipping the already-released records changes
  no observable state, driven against a reference lifecycle that keeps the
  unconditional loop, over a script that covers scope entry, scope exit, park,
  re-arm, a source change and a session replacement.
"""

from dataclasses import fields

import pytest

from arrayscope.presentation import (
    Presentation,
    TileLifecycle,
    TilePayloadRef,
    TileTarget,
)
from arrayscope.presentation import tile_lifecycle as tile_lifecycle_module

MONTAGE_TILES = 400
WINDOW_TILES = 40


def _target(tile: int, *, source: int | None = None, level: int = 0) -> TileTarget:
    source_index = tile if source is None else source
    return TileTarget(
        tile_number=tile,
        source_index=source_index,
        semantic_source_id=("montage-tile", "sem", source_index),
        lod_level=level,
    )


def _window(start: int, *, level: int = 0, source_offset: int = 0) -> dict[int, TileTarget]:
    """The margin-0 visible set for a viewport whose first tile is ``start``."""

    return {
        tile: _target(tile, source=tile + source_offset, level=level)
        for tile in range(start, start + WINDOW_TILES)
    }


def _scroll_to(lc: TileLifecycle, last_start: int) -> None:
    """Accumulate records the way a long pan does: one tile in, one tile out."""

    for start in range(last_start + 1):
        lc.retarget(_window(start))


def _released_target_ids(lc: TileLifecycle, exclude: set[int]) -> dict[int, int]:
    return {
        tile: id(rec.target)
        for tile, rec in lc._records.items()
        if tile not in exclude and rec.target is not None
    }


def test_release_pass_mutates_only_the_records_that_left_the_viewport():
    """The pass is O(tiles that left), not O(tiles ever shown).

    Pinned as a count, not as a resulting state: re-releasing an already
    released record produces an equal ``TileTarget``, so a state-only
    assertion passes against the unconditional loop and proves nothing.
    """

    lc = TileLifecycle()
    _scroll_to(lc, MONTAGE_TILES - WINDOW_TILES)
    assert len(lc._records) == MONTAGE_TILES

    # One tile of pan back the way we came: tile 399 leaves, tile 359 returns.
    wanted = _window(MONTAGE_TILES - WINDOW_TILES - 1)
    before = _released_target_ids(lc, exclude=set(wanted))
    assert len(before) == MONTAGE_TILES - WINDOW_TILES  # 360 records out of scope

    lc.retarget(wanted)

    after = _released_target_ids(lc, exclude=set(wanted))
    mutated = {tile for tile, ident in after.items() if before.get(tile) != ident}
    assert mutated == {MONTAGE_TILES - 1}


def test_release_pass_emits_one_trace_edge_per_real_release(monkeypatch):
    """``target_released`` states a transition, not a per-pan census.

    The trace consumer (``tools/trace_verify``) pops the tile's target and
    acknowledgement on this edge, which is idempotent — but the edge's *count*
    changes meaning, so it is pinned here rather than left implicit.
    """

    events: list[dict] = []
    monkeypatch.setattr(tile_lifecycle_module, "trace_enabled", lambda: True)
    monkeypatch.setattr(
        tile_lifecycle_module,
        "emit_trace_fields",
        lambda kind, f: events.append(f) if kind == "lifecycle" else None,
    )

    lc = TileLifecycle()
    _scroll_to(lc, MONTAGE_TILES - WINDOW_TILES)
    events.clear()

    lc.retarget(_window(MONTAGE_TILES - WINDOW_TILES - 1))

    released = [f["tile"] for f in events if f["edge"] == "target_released"]
    assert released == [MONTAGE_TILES - 1]


def test_released_record_that_regains_a_claim_is_cleared_by_the_next_retarget():
    """Idempotence is not the same as "already visited".

    A record can pick up a task claim after it left scope (admission is not
    gated on visibility).  The fixed-point test therefore has to read the
    claim fields, not just the target's ``visible`` flag.
    """

    lc = TileLifecycle()
    lc.retarget({0: _target(0), 1: _target(1)})
    lc.retarget({0: _target(0)})
    assert lc.record(1).target is not None
    assert not lc.record(1).target.visible

    lc.task_admitted(1, ("task", 1), stage_key=("stage", 1), stage_producer_key=("prod", 1))
    assert lc.record(1).task_claim is not None

    lc.retarget({0: _target(0)})
    assert lc.record(1).task_claim is None
    assert lc.record(1).stage_producer_key is None


def test_re_entering_and_leaving_again_releases_again(monkeypatch):
    """The skip is a fixed point, not a latch: scope re-entry re-arms it."""

    events: list[dict] = []
    monkeypatch.setattr(tile_lifecycle_module, "trace_enabled", lambda: True)
    monkeypatch.setattr(
        tile_lifecycle_module,
        "emit_trace_fields",
        lambda kind, f: events.append(f) if kind == "lifecycle" else None,
    )

    def releases() -> list[int]:
        return [f["tile"] for f in events if f["edge"] == "target_released"]

    lc = TileLifecycle()
    lc.retarget({0: _target(0), 1: _target(1)})
    lc.retarget({0: _target(0)})
    assert lc.record(1).target.visible is False
    assert releases() == [1]

    lc.retarget({0: _target(0)})  # still out of scope: no second statement
    assert releases() == [1]

    lc.retarget({0: _target(0), 1: _target(1)})
    assert lc.record(1).target.visible is True
    lc.retarget({0: _target(0)})
    assert lc.record(1).target.visible is False
    assert releases() == [1, 1]


class _UnconditionalReleaseLifecycle(TileLifecycle):
    """``main``'s release loop: re-release every out-of-scope record.

    Layered on top of the real ``retarget`` rather than forked from it, so the
    reference stays true to every other clause of the method; the extra pass
    reproduces exactly what the pre-fix loop did to records the new loop
    skips.
    """

    def retarget(self, targets):
        wanted = {int(tile) for tile in dict(targets)}
        pruned = super().retarget(targets)
        for tile_number, rec in tuple(self._records.items()):
            if tile_number in wanted or rec.target is None:
                continue
            rec.target = TileTarget(
                tile_number=rec.target.tile_number,
                source_index=rec.target.source_index,
                semantic_source_id=rec.target.semantic_source_id,
                lod_level=rec.target.lod_level,
                visible=False,
            )
            rec.task_claim = None
            rec.stage_producer_key = None
        return pruned


_RECORD_FIELDS = tuple(f.name for f in fields(tile_lifecycle_module.TileRecord))


def _dump(lc: TileLifecycle) -> dict[str, object]:
    """Everything the machine can be asked about, by value."""

    records = {}
    for tile, rec in sorted(lc._records.items()):
        row = {name: repr(getattr(rec, name)) for name in _RECORD_FIELDS}
        row["phase"] = rec.phase.value
        row["active"] = rec.active
        row["target_settled"] = rec.target_settled
        row["first_pixel_presented"] = rec.first_pixel_presented
        records[tile] = row
    return {
        "records": records,
        "counters": lc.counters(),
        "parked": sorted(lc.parked_tiles),
        "presented": sorted(lc.presented_tiles),
        "loading": sorted(lc.loading_tiles),
        "evaluating": sorted(lc.evaluating_tiles),
        "active_requests": sorted(lc.active_request_tiles),
        "skipped": sorted(lc.skipped_tiles),
        "backend_identities": dict(sorted(lc.backend_presented_identities.items())),
        "dangling_claims": sorted(repr(claim) for claim in lc.dangling_claims()),
        "level_revision": lc.level_revision,
        "identity_rejections": lc.identity_rejections,
    }


def _payload(tile: int, *, source: int, quality: str = "exact", level: int = 0):
    return TilePayloadRef(
        source_id=("payload", tile, source, quality, level),
        quality=quality,
        lod_level=level,
        source_index=source,
    )


def _drive(lc: TileLifecycle) -> list[dict[str, object]]:
    """A pan/park/re-arm/scroll/replace script, dumped after every step.

    Each step returns state that both implementations must agree on; the
    per-step dump localises a divergence to the event that caused it instead
    of reporting only the end state.
    """

    dumps: list[dict[str, object]] = []

    def step() -> None:
        dumps.append(_dump(lc))

    # 1. Accumulate records: a long pan, one tile in and one tile out.
    for start in range(61):
        lc.retarget(_window(start))
    step()

    # 2. Settle part of the current window: evaluated, emitted, acknowledged.
    settled = range(60, 70)
    lc.plan_applied(settled)
    for tile in settled:
        lc.evaluation_started(tile)
        lc.evaluation_completed(tile)
        ref = _payload(tile, source=tile)
        lc.target_ready(tile, ref)
        lc.commit_emitted({tile: ref})
    lc.commit_acknowledged(
        emitted_tiles=tuple(settled),
        accepted_tiles=tuple(settled),
        active_scope=tuple(_window(60)),
        presented_identities={tile: ("payload", tile, tile, "exact", 0) for tile in settled},
    )
    step()

    # 3. A tile is declined out of scope and parks (rule 3).
    parked_tile = 62
    ref = _payload(parked_tile, source=parked_tile, quality="fallback")
    lc.commit_emitted({parked_tile: ref})
    lc.retarget(_window(80))
    lc.commit_acknowledged(
        emitted_tiles=(parked_tile,),
        accepted_tiles=(),
        active_scope=tuple(_window(80)),
    )
    step()
    assert lc.record(parked_tile).presentation is Presentation.PARKED

    # 4. Claims taken on tiles that are already out of scope, then a pan.
    lc.task_requested(30, stage_key=("stage", 30), stage_producer_key=("prod", 30))
    lc.task_admitted(31, ("task", 31), stage_key=("stage", 31), stage_producer_key=("prod", 31))
    lc.retarget(_window(81))
    step()

    # 5. Pan back so the parked tile re-enters scope and is re-armed.
    lc.retarget(_window(parked_tile - 1))
    rearmed = lc.rearm_for_scope(tuple(_window(parked_tile - 1)))
    dumps.append({"rearmed": list(rearmed)})
    step()

    # 6. An index scroll: same slots, different semantic sources.
    lc.retarget(_window(parked_tile - 1, source_offset=1000))
    step()

    # 7. A coarser LOD round over the same slots, then a pan away.
    lc.retarget(_window(parked_tile - 1, source_offset=1000, level=3))
    lc.retarget(_window(200))
    step()

    # 8. Session replacement: claims released, records survive.
    lc.level_claimed(200, ("page", 200), tile_lifecycle_module.ClaimOwner.PREVIEW)
    effects = lc.session_replaced()
    dumps.append({"released_claims": sorted(repr(effect) for effect in effects)})
    lc.retarget(_window(201))
    step()

    # 9. Return to the very first window, which has been out of scope
    #    since step 1 and is where a stale skip would show.
    lc.retarget(_window(0))
    step()

    return dumps


def test_release_skip_changes_no_observable_state():
    """Differential oracle against the unconditional loop."""

    fixed = _drive(TileLifecycle())
    reference = _drive(_UnconditionalReleaseLifecycle())

    assert len(fixed) == len(reference)
    for index, (lhs, rhs) in enumerate(zip(fixed, reference, strict=True)):
        assert lhs == rhs, f"divergence after script step {index}"


class _NoReleaseLifecycle(TileLifecycle):
    """A skip taken one step too far: no out-of-scope record is ever released.

    Present so the oracle above is demonstrably able to fail.  A differential
    test whose reference cannot disagree is the failure mode this repository
    keeps hitting; this is the cheapest available proof that it can.
    """

    def retarget(self, targets):
        wanted = {int(tile) for tile in dict(targets)}
        held = {
            tile: rec.target
            for tile, rec in self._records.items()
            if tile not in wanted and rec.target is not None
        }
        pruned = super().retarget(targets)
        for tile, target in held.items():
            self._records[tile].target = target
        return pruned


def test_the_differential_oracle_can_fail():
    """Sanity: the reference disagrees with a wrong skip."""

    broken = _drive(_NoReleaseLifecycle())
    reference = _drive(_UnconditionalReleaseLifecycle())

    def compare() -> None:
        for lhs, rhs in zip(broken, reference, strict=True):
            assert lhs == rhs

    with pytest.raises(AssertionError):
        compare()
