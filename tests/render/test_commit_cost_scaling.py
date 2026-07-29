"""R5: one tiled commit costs what its delta costs, not what the montage costs.

The open item under R5 was never the chunk size — it was the fixed per-commit
term.  ``63-69%`` of a commit iterated every presented payload, so an
empty-delta commit still cost ~90 ms and no cohort size could satisfy the
rule.  These tests pin the shape that made that true.

The gates here are **counters**, not milliseconds.  That is deliberate.  A
wall-clock ratio between a small and a field montage does not discriminate at
the cohort sizes the governor actually admits — measured before this change, a
32-tile commit was already only 1.3x more expensive into a 272-tile montage
than into a 34-tile one, because per-item work dominates at that size.  A
threshold that passes both before and after is worse than no test: it reports
confidence it has not earned.  Counting the per-tile work a commit performs is
exact, machine-independent, and separates the two regimes cleanly (measured
pre-change: 272 and 816 layout regions and 272 tile instances rebuilt for a
ONE-tile commit).

One wall-clock assertion survives, and it needs no threshold: a commit with
nothing to present must not cost more than one presenting 32 tiles.

Field scale is the point.  A 3-tile fixture cannot see any of this: the
whole-montage term and the per-delta term are the same size there.  Neither
can a fixture whose tiles are a constant fill (a misplaced patch is then
byte-identical) or whose payloads carry no ``TileIdentity`` (WGPU then never
reaches the binding-reuse path the app runs).  Both are pinned below, because
both silently turned a green test into no test at all.

**Still montage-proportional, and not covered here**: the presented-identity
scan, now the largest remaining whole-montage walk in a bounded PyQtGraph
commit at ~22%.  It asks Qt for each item's effective visibility, and
``state.visible`` and the item's own flag are deliberately allowed to
diverge, so the two-condition check is load-bearing.  Replacing it needs
maintained visibility ownership at the points where backend visibility
changes — not a cheaper loop — and is left for the GUI-handoff work rather
than pinned here by an assertion that would quietly succeed.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import replace
from statistics import median
from time import perf_counter

import numpy as np
import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from arrayscope.core.view_state import ViewState
from arrayscope.display import tile_layout
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.lod import LodInfo
from arrayscope.display.model.frame import (
    DisplayTilePayload,
    TilePresentationDelta,
    TilePresentationState,
)
from arrayscope.display.model.tile_identity import TileIdentity, TileLodIdentity
from arrayscope.display.montage import MontageTileState
from arrayscope.display.shader_mapping import TexturePlaneKind
from arrayscope.display.viewport import ViewportPolicy
from tests.display.test_imageview2d import _view_class

# The field montage this program is measured against (ADR 0059 / the R5
# dossier): 272 tiles.  Every count below is compared against this, so a
# per-montage cost and a per-delta cost differ by two orders of magnitude.
FIELD_TILES = 272
COLUMNS = 16
TILE = 64


def _wgpu_adapter_available() -> bool:
    try:
        import wgpu
        from wgpu.backends.wgpu_native.extras import set_instance_extras

        with contextlib.suppress(RuntimeError):
            set_instance_extras(backends=["Vulkan"])
        wgpu.gpu.request_adapter_sync(power_preference="low-power")
        return True
    except Exception:
        return False


_BACKENDS = ["pyqtgraph"]
if _wgpu_adapter_available():
    _BACKENDS.append("wgpu")


def _payload(tile: int, generation: object) -> DisplayTilePayload:
    # Deliberately NOT a constant fill.  Every pixel is distinct within the
    # tile and across tiles and generations, so a patch that writes the right
    # bytes to the wrong place — or the wrong order — is visible.  A constant
    # tile makes reordering undetectable, which is how a scrambling bug can
    # pass an oracle that compares values.
    image = (
        float(tile) * 1e4
        + float(hash((tile, generation)) % 997)
        + np.arange(TILE * TILE, dtype=np.float32).reshape(TILE, TILE)
    ).astype(np.float32)
    return DisplayTilePayload(
        tile_number=tile,
        source_index=tile,
        image=image,
        histogram_data=None,
        source_id=("tile", tile, generation),
        texture_data=image,
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        semantic_data=image,
        source_shape=image.shape,
        lod=LodInfo(0, 1, image.shape, image.shape, 0),
        tile_identity=TileIdentity(
            document_generation=1,
            operation_key="raw",
            source_index=int(tile),
            image_axes=(0, 1),
            axis_flips=(False, False),
            channel="scalar",
            complex_mapping=None,
            texture_kind=TexturePlaneKind.SCALAR_R32F,
            semantic_generation=generation,
            lod=TileLodIdentity(),
        ),
    )


def _montage(tiles: int):
    rows = -(-tiles // COLUMNS)
    view_state = (
        ViewState.from_shape((TILE, TILE, tiles))
        .with_image_axes(0, 1)
        .with_montage_axis(2, columns=COLUMNS, indices=tuple(range(tiles)))
    )
    geometry = DisplayGeometry(
        view_state=view_state,
        display_shape=(rows * TILE, COLUMNS * TILE),
        montage=MontageGeometry(
            indices=tuple(range(tiles)),
            tile_shape=(TILE, TILE),
            columns=COLUMNS,
            rows=rows,
            gap=0,
        ),
        montage_tile_states=tuple([MontageTileState.LOADED] * tiles),
    )
    return geometry, {tile: _payload(tile, 0) for tile in range(tiles)}


class _Montage:
    """A committed montage that can be driven with further bounded deltas."""

    def __init__(self, backend: str, tiles: int, *, present: int | None = None):
        self.view = _view_class(backend)()
        self.geometry, self.payloads = _montage(tiles)
        self.tiles = tiles
        self.revision = 1
        if present is None:
            self._commit(range(tiles))
        else:
            # Start partial, the way a fill does: the montage is planned at
            # full size but only some tiles have arrived.
            self.payloads = {tile: self.payloads[tile] for tile in range(present)}
            self.tiles = present
            self._commit(range(present))

    def _commit(self, upsert_tiles, *, removals=(), active=None, refresh_payloads=True):
        upsert_tiles = tuple(int(tile) for tile in upsert_tiles)
        removals = tuple(int(tile) for tile in removals)
        if refresh_payloads:
            for tile in upsert_tiles:
                self.payloads[tile] = _payload(tile, self.revision)
        for tile in removals:
            self.payloads.pop(tile, None)
        active_tiles = (
            tuple(range(self.tiles)) if active is None else tuple(sorted(int(t) for t in active))
        )
        delta = TilePresentationDelta(
            structure_revision=self.revision,
            payload_revision=self.revision,
            visibility_revision=self.revision,
            level_revision=self.revision,
            histogram_revision=self.revision,
            viewport_revision=self.revision,
            base_revision=self.revision - 1,
            target_revision=self.revision,
            upserts={tile: self.payloads[tile] for tile in upsert_tiles},
            removals=removals,
            active_tiles=active_tiles,
            planned_tiles=tuple(range(self.tiles)),
        )
        self.view.setTiledPresentation(
            geometry=self.geometry,
            tile_state=TilePresentationState(self.payloads, revision=self.revision - 1),
            tile_delta=delta,
            histogramPlotData=None,
            levels=(0.0, float(self.tiles)),
            histogramRange=(0.0, float(self.tiles)),
            viewport_policy=ViewportPolicy.PRESERVE,
        )
        self.revision += 1

    def commit(self, upsert_tiles, **kwargs):
        self._commit(upsert_tiles, **kwargs)

    def commit_ms(self, upsert_tiles, *, samples: int = 9) -> float:
        timings = []
        for _ in range(samples):
            start = perf_counter()
            self._commit(upsert_tiles)
            timings.append((perf_counter() - start) * 1000.0)
        # Discard the first two: the first bounded commit after a full one
        # still installs binding state a steady stream does not repay.
        return median(timings[2:])


class _Counter:
    """Count constructions of one type for the duration of a commit."""

    def __init__(self, monkeypatch, module, name):
        self.count = 0
        original = getattr(module, name)

        def counted(*args, **kwargs):
            self.count += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, counted)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_bounded_commit_does_not_rederive_montage_geometry(backend, qt_app, monkeypatch):
    """Tile placement is recomputed per changed tile, never per montage.

    Layout is a pure function of the montage geometry, but every commit asked
    for it up to three times and built one region object per tile each time.
    That is a fixed O(montage) term on a commit that touched one tile.
    """

    montage = _Montage(backend, FIELD_TILES)
    counter = _Counter(monkeypatch, tile_layout, "TileLayoutRegion")

    montage.commit((0,))

    assert counter.count == 0, (
        f"a 1-tile commit into a {FIELD_TILES}-tile montage rebuilt "
        f"{counter.count} tile layout regions; placement did not change, so it "
        "must be reused rather than re-derived"
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_placement_is_reused_across_rebuilt_geometry(backend, qt_app, monkeypatch):
    """Equal-but-new geometry must reuse placement, not rebuild it.

    The app does not hand the same ``DisplayGeometry`` object to consecutive
    commits — a crop or slice sweep constructs one per step, equal in every
    field that affects placement.  A cache keyed on object identity therefore
    misses every time; measured in the real workflow it hit 40 times and
    missed 412, while making each miss more expensive than having no cache.

    The test above cannot see that, because it reuses one geometry object
    forever.  This one rebuilds the geometry between commits, which is the
    regime that matters.
    """

    montage = _Montage(backend, FIELD_TILES)
    counter = _Counter(monkeypatch, tile_layout, "TileLayoutRegion")

    for _ in range(4):
        montage.geometry = replace(
            montage.geometry,
            montage=replace(montage.geometry.montage),
            montage_tile_states=tuple(montage.geometry.montage_tile_states),
        )
        montage.commit((0,))

    assert counter.count == 0, (
        f"four 1-tile commits with rebuilt (but equal) geometry into a "
        f"{FIELD_TILES}-tile montage rebuilt {counter.count} tile layout "
        "regions; placement is a function of the geometry's VALUE, so an "
        "equal geometry must reuse it"
    )


@pytest.mark.skipif("wgpu" not in _BACKENDS, reason="no wgpu adapter on this machine")
def test_wgpu_instance_rebuild_tracks_the_delta_not_the_montage(qt_app, monkeypatch):
    """The GPU instance buffer is patched at the changed tiles only.

    Rebuilding all 272 instances per commit was 22% of a bounded WGPU commit.
    The descriptors of untouched tiles are unchanged by construction, so the
    successor patches its predecessor's instances instead of rebuilding them.
    """

    from arrayscope.display import wgpu_imageview2d

    montage = _Montage("wgpu", FIELD_TILES)

    one = _Counter(monkeypatch, wgpu_imageview2d, "TileInstance")
    montage.commit((0,))
    one_tile = one.count

    eight = _Counter(monkeypatch, wgpu_imageview2d, "TileInstance")
    montage.commit(range(8))
    eight_tiles = eight.count

    assert one_tile <= 4, (
        f"a 1-tile commit into a {FIELD_TILES}-tile montage built {one_tile} "
        "tile instances; only the committed tile's instance changed"
    )
    assert eight_tiles <= 16, (
        f"an 8-tile commit into a {FIELD_TILES}-tile montage built {eight_tiles} tile instances"
    )
    assert eight_tiles > one_tile, (
        "instance construction must track the delta; it did not grow between a "
        f"1-tile ({one_tile}) and an 8-tile ({eight_tiles}) commit, so this "
        "test is no longer measuring per-delta work"
    )


@pytest.mark.skipif("wgpu" not in _BACKENDS, reason="no wgpu adapter on this machine")
def test_wgpu_instance_equality_checks_track_the_delta(qt_app, monkeypatch):
    """Reusing an equal instance tuple must not hide a whole-montage scan."""

    from arrayscope.gpu.command_protocol import TileInstance

    montage = _Montage("wgpu", FIELD_TILES)
    comparisons = 0
    original_eq = TileInstance.__eq__

    def counted_eq(self, other):
        nonlocal comparisons
        comparisons += 1
        return original_eq(self, other)

    monkeypatch.setattr(TileInstance, "__eq__", counted_eq)
    # The last tile makes a tuple equality fallback inspect the full montage
    # when the replacement's geometry is unchanged.
    montage.commit((FIELD_TILES - 1,))

    assert comparisons <= 4, (
        f"a 1-tile commit compared {comparisons} tile instances; equality "
        "reuse must inspect the delta, not scan the montage"
    )


def test_layout_cache_invalidates_on_every_input_that_moves_a_tile():
    """Semantic equality must never become stale physical geometry.

    The layout cache is keyed by value, which is the only way it hits at all
    (the app rebuilds geometry constantly). The risk of a value key is the
    opposite of a stale identity key: an input that moves a tile but is not
    part of the key would be served a placement from before it changed.

    So this walks the inputs and asserts each one moves the result. It is a
    completeness check on the key, not a behaviour check on placement.
    """

    from arrayscope.display.tile_layout import planned_tile_count, tile_layout_map

    geometry, _payloads = _montage(FIELD_TILES)
    base = tile_layout_map(geometry)
    baseline = {
        tile: (region.x, region.y, region.width, region.height) for tile, region in base.items()
    }

    def placement(changed):
        layout = tile_layout_map(changed)
        return {tile: (r.x, r.y, r.width, r.height) for tile, r in layout.items()}

    montage = geometry.montage
    cases = {
        "tile size": replace(geometry, montage=replace(montage, tile_shape=(TILE * 2, TILE * 2))),
        "columns": replace(geometry, montage=replace(montage, columns=COLUMNS // 2)),
        "gutter": replace(geometry, montage=replace(montage, gap=3)),
        "active set shrink": replace(
            geometry, montage=replace(montage, indices=tuple(range(FIELD_TILES // 2)))
        ),
    }
    for label, changed in cases.items():
        assert placement(changed) != baseline, (
            f"changing the {label} produced identical placement; that input is "
            "not part of the layout cache key and a stale layout can be served"
        )

    # A source-index remap keeps placement but must not keep the mapping.
    remapped = replace(
        geometry, montage=replace(montage, indices=tuple(reversed(range(FIELD_TILES))))
    )
    assert {tile: region.source_index for tile, region in tile_layout_map(remapped).items()} != {
        tile: region.source_index for tile, region in base.items()
    }, "remapping source indices reused the previous tile-to-source mapping"

    # An equal-but-new geometry must reuse, and an active-set growth must not.
    rebuilt = replace(
        geometry, montage=replace(montage), montage_tile_states=tuple(geometry.montage_tile_states)
    )
    assert placement(rebuilt) == baseline, "an equal geometry did not reuse its placement"
    assert planned_tile_count(cases["active set shrink"]) == FIELD_TILES // 2, (
        "the planned count did not follow the active set"
    )


def test_layout_map_cannot_be_mutated_by_a_caller():
    """The shared placement mapping is read-only by construction.

    Every caller resolving an equal geometry receives the same mapping, so one
    accidental write would misplace tiles for all of them until the entry is
    evicted — and it would surface far from the code that did it. Documenting
    it as read-only is not the same as it being read-only.
    """

    from arrayscope.display.tile_layout import tile_layout_map

    geometry, _payloads = _montage(8)
    layout = tile_layout_map(geometry)
    victim = next(iter(layout))

    with pytest.raises(TypeError):
        layout[victim] = None
    with pytest.raises((TypeError, AttributeError)):
        layout.pop(victim)

    assert dict(layout), "a caller must still be able to take its own copy"
    assert tile_layout_map(geometry)[victim] is layout[victim]


def test_bounded_commit_inspects_only_the_delta_for_the_histogram(qt_app, monkeypatch):
    """The montage histogram source is maintained, not re-derived.

    A tiled montage has no bound ImageItem, so its histogram source is built
    from the committed payloads.  Rebuilding that from every payload on every
    bounded commit was the largest single term in a PyQtGraph commit — and
    even deciding whether the previous buffer could be reused meant a
    montage-wide identity scan, so the cache HIT was montage-proportional too.

    The delta already says which tiles changed, so nothing else needs looking
    at.  This counts payload inspections in the refinement regime, where the
    population is stable and every commit could in principle reuse.
    """

    from arrayscope.display.model import tiled_histogram_identity

    montage = _Montage("pyqtgraph", FIELD_TILES)
    montage.commit((0,))  # settle the maintained buffer

    counter = _Counter(monkeypatch, tiled_histogram_identity, "payload_histogram_display_source")
    montage.commit((0,))

    assert counter.count <= 4, (
        f"a 1-tile commit into a {FIELD_TILES}-tile montage inspected "
        f"{counter.count} payload histogram sources; the delta names the only "
        "tile whose contribution can have changed"
    )


def test_histogram_source_matches_a_full_rebuild(qt_app):
    """The maintained histogram source must equal a from-scratch build.

    This one feeds drawn output, so a bounded patch that diverges is a wrong
    picture, not a slow one.  The oracle rebuilds from the committed payloads
    independently of the maintained buffer and compares values across the
    delta shapes that could break the mapping.

    PyQtGraph only: a tiled montage has no bound ImageItem to read a histogram
    from, so the CPU-LUT backend derives one from the payloads.  WGPU carries
    histogram evidence through the shader path instead.
    """

    backend = "pyqtgraph"

    from arrayscope.display.model.tiled_histogram_identity import (
        histogram_plot_source_and_layout,
    )

    montage = _Montage(backend, FIELD_TILES)

    def check(label):
        maintained = montage.view.histogramPlotSource
        expected = histogram_plot_source_and_layout(montage.payloads)[0]
        if maintained is None or expected is None:
            assert maintained is expected, f"[{label}] one source is None and the other is not"
            return
        maintained = np.asarray(maintained)
        expected = np.asarray(expected)
        assert maintained.shape == expected.shape, (
            f"[{label}] maintained histogram source has shape {maintained.shape}, "
            f"a full rebuild gives {expected.shape}"
        )
        assert np.array_equal(maintained, expected, equal_nan=True), (
            f"[{label}] the maintained histogram source diverged from a full rebuild"
        )

    check("initial")
    montage.commit((0,))
    check("first tile")
    montage.commit((FIELD_TILES - 1,))
    check("last tile")
    montage.commit(range(8))
    check("eight tiles")
    montage.commit(())
    check("empty delta")
    montage.commit((5,), refresh_payloads=False)
    check("re-presented unchanged payload")
    montage.commit((), removals=(FIELD_TILES - 1,), active=range(FIELD_TILES - 1))
    montage.tiles = FIELD_TILES - 1
    check("removal")
    montage.commit((3,), active=range(FIELD_TILES - 1))
    check("after removal")


def _instances_rebuilt_from_scratch(view):
    """Independent oracle: the instance tuple a full rebuild would produce.

    Deliberately re-derived from the committed record rather than from the
    incremental builder, so a position/identity mismatch in the patch path
    cannot agree with itself.
    """

    from arrayscope.gpu.command_protocol import TileInstance

    committed = view._wgpu_committed or {}
    tiles = committed.get("tiles", {}) or {}
    transposed = bool(committed.get("transposed", False))
    return tuple(
        TileInstance(
            tuple(float(value) for value in tiles[tile]["world_rect"]),
            tuple(float(value) for value in tiles[tile].get("src_origin", (0.0, 0.0))),
            tuple(float(value) for value in tiles[tile]["src_size"]),
            0,
            plane_index=int(tiles[tile]["plane_index"]),
            transposed=transposed,
        )
        for tile in sorted(tiles)
    )


@pytest.mark.skipif("wgpu" not in _BACKENDS, reason="no wgpu adapter on this machine")
def test_wgpu_patched_instances_match_a_full_rebuild(qt_app):
    """Every bounded path must agree with an independent full rebuild.

    Patching by position is only sound while position means tile identity.
    This sweeps the shapes where that could stop being true — the last
    position, a removal, an active-set change, an empty delta, and geometry
    rebuilt as a new but equal object — and checks the result against an
    oracle that re-derives the tuple from the committed record.
    """

    montage = _Montage("wgpu", FIELD_TILES)

    def check(label):
        instances = montage.view._wgpu_tile_instances()
        assert instances == _instances_rebuilt_from_scratch(montage.view), (
            f"[{label}] the incremental instance tuple disagrees with a full "
            "rebuild; a patched position no longer names the tile it belongs to"
        )

    check("initial")
    montage.commit((0,))
    check("first tile")
    montage.commit((FIELD_TILES - 1,))
    check("last tile")
    montage.commit(range(8))
    check("eight tiles")
    montage.commit(())
    check("empty delta")
    montage.geometry = replace(
        montage.geometry,
        montage=replace(montage.geometry.montage),
        montage_tile_states=tuple(montage.geometry.montage_tile_states),
    )
    montage.commit((5,))
    check("geometry rebuilt by value")
    montage.commit((), removals=(FIELD_TILES - 1,), active=range(FIELD_TILES - 1))
    check("removal")
    montage.tiles = FIELD_TILES - 1
    montage.commit((3,), active=range(FIELD_TILES - 1))
    check("after removal")


@pytest.mark.skipif("wgpu" not in _BACKENDS, reason="no wgpu adapter on this machine")
def test_wgpu_growing_population_places_every_new_tile(qt_app):
    """Tiles arriving must reach the instance buffer, at the right position.

    Growth is the shape a fill is in for its whole duration, and it is the
    one that can defeat an order reused from the predecessor: the new tile has
    no position in that order.  A patch path that trusts a stale order drops
    it silently — the montage simply never draws that tile.
    """

    montage = _Montage("wgpu", FIELD_TILES, present=FIELD_TILES // 2)

    for arrived in range(FIELD_TILES // 2, FIELD_TILES, 32):
        cohort = range(arrived, min(arrived + 32, FIELD_TILES))
        for tile in cohort:
            montage.payloads[tile] = _payload(tile, montage.revision)
        montage.tiles = min(arrived + 32, FIELD_TILES)
        montage.commit(cohort, refresh_payloads=False, active=range(montage.tiles))
        instances = montage.view._wgpu_tile_instances()
        assert instances == _instances_rebuilt_from_scratch(montage.view), (
            f"after growing to {montage.tiles} tiles the incremental instance "
            "tuple disagrees with a full rebuild"
        )
        assert len(instances) == montage.tiles, (
            f"the montage holds {montage.tiles} tiles but the instance buffer "
            f"carries {len(instances)}; an arriving tile was dropped"
        )


@pytest.mark.skipif("wgpu" not in _BACKENDS, reason="no wgpu adapter on this machine")
def test_wgpu_equal_replacement_retains_the_instance_tuple(qt_app):
    """An unchanged descriptor must reuse the tuple OBJECT, not an equal copy.

    The executor skips the instance-buffer upload on tuple identity, so an
    equal-but-new tuple is a montage-sized rewrite that changes no pixels.
    Identity — not equality — is what proves nothing was copied.
    """

    montage = _Montage("wgpu", FIELD_TILES)
    montage.commit((FIELD_TILES - 1,))
    before = montage.view._wgpu_tile_instances()

    # Re-present the same payloads: identities are unchanged, so every
    # descriptor this commit rewrites is equal to the one it replaces.
    montage.commit((FIELD_TILES - 1,), refresh_payloads=False)
    after = montage.view._wgpu_tile_instances()

    assert after is before, (
        "re-presenting an unchanged descriptor produced a new instance tuple; "
        "the executor will now rewrite the whole instance buffer for a frame "
        "whose geometry did not move"
    )


@pytest.mark.skipif("wgpu" not in _BACKENDS, reason="no wgpu adapter on this machine")
def test_empty_delta_takes_the_production_fast_path(qt_app, monkeypatch):
    """The fixture must reach the path production reaches.

    WGPU's whole-binding reuse needs a real ``TileIdentity`` on every payload:
    without one the identity comparison bails, the empty delta falls onto a
    full montage rebuild, and every timing taken through that fixture
    describes a path the app never runs. That is not hypothetical — an
    earlier measurement of this same empty-delta case reported 13.4 ms and a
    supposed inversion against a 32-tile delta, both artefacts of payloads
    that carried no identity.

    So this pins the fixture as much as the code: an empty delta must build
    no tile instances at all, which is only true on the fast path.
    """

    from arrayscope.display import wgpu_imageview2d
    from arrayscope.display.wgpu_imageview2d import _wgpu_physical_payload_identities

    montage = _Montage("wgpu", FIELD_TILES)
    assert _wgpu_physical_payload_identities(montage.payloads) is not None, (
        "fixture payloads carry no TileIdentity, so WGPU cannot reuse a "
        "binding; any commit cost measured through them is not the app's"
    )

    counter = _Counter(monkeypatch, wgpu_imageview2d, "TileInstance")
    montage.commit(())

    assert counter.count == 0, (
        f"an empty delta built {counter.count} tile instances; it left the "
        "whole-binding fast path and rebuilt the montage"
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_empty_delta_commit_is_cheaper_than_one_that_does_work(backend, qt_app):
    """An empty delta must not cost more than a delta that presents tiles.

    Ratio-free and threshold-free: whatever a commit costs on this machine, a
    commit with nothing to present cannot cost more than one presenting 32
    tiles.  It did before this change — an empty delta fell off the
    incremental path and rebuilt the whole montage binding.
    """

    montage = _Montage(backend, FIELD_TILES)

    empty_ms = montage.commit_ms(())
    working_ms = montage.commit_ms(range(32))

    assert empty_ms <= working_ms, (
        f"[{backend}] an empty-delta commit into a {FIELD_TILES}-tile montage "
        f"cost {empty_ms:.2f} ms while a 32-tile commit cost {working_ms:.2f} "
        "ms; the commit is paying a whole-montage fixed cost that its delta "
        "does not justify"
    )


def test_growing_montage_inspects_each_payload_once_per_commit(qt_app, monkeypatch):
    """A reuse attempt that cannot succeed must not cost a scan.

    While tiles are still arriving, every commit grows the payload
    population, so the montage histogram source cannot be patched from the
    previous buffer — it has to be rebuilt.  Deciding that by scanning the
    payloads first and only then discovering the population moved made a fill
    measurably MORE expensive than not trying at all — every scan was thrown
    away, two passes where one was needed.

    The refusal is therefore decided from the payload key order, before any
    payload is inspected.  This pins that: one pass over the payloads per
    commit, never two.  It is a regression this change introduced and then
    removed, which is exactly why it is pinned rather than trusted.
    """

    from arrayscope.display.model import tiled_histogram_identity

    montage = _Montage("pyqtgraph", FIELD_TILES)
    # Re-enter the coverage regime: a population that grows commit over
    # commit, the state a fill is in for its whole duration.
    montage.payloads = {tile: montage.payloads[tile] for tile in range(FIELD_TILES // 2)}
    montage.tiles = FIELD_TILES // 2
    montage.commit(())

    counter = _Counter(monkeypatch, tiled_histogram_identity, "payload_histogram_display_source")
    montage.tiles = FIELD_TILES // 2 + 32
    for tile in range(FIELD_TILES // 2, montage.tiles):
        montage.payloads[tile] = _payload(tile, montage.revision)
    montage.commit(range(FIELD_TILES // 2, montage.tiles))

    assert counter.count <= montage.tiles, (
        f"a commit that grew the montage to {montage.tiles} tiles inspected "
        f"{counter.count} payload histogram sources — more than one pass, so a "
        "reuse attempt that could not succeed was paid for anyway"
    )


@pytest.mark.skipif("wgpu" not in _BACKENDS, reason="no wgpu adapter on this machine")
def test_wgpu_residency_pins_are_rederived_per_delta_not_per_montage(qt_app, monkeypatch):
    """Re-pinning committed pages costs the delta, not the resident set.

    Replacing an owner's pin set walked the UNION of the old and new sets and
    re-derived ownership for every page in it, so a bounded montage delta
    re-derived every pinned page in the montage.  Only keys whose membership
    actually moved can change state, and that set is the delta.
    """

    from arrayscope.gpu import page_table as page_table_module

    montage = _Montage("wgpu", FIELD_TILES)
    table = montage.view._wgpu_executor.page_table
    resident = len(table.resident_keys())
    assert resident >= FIELD_TILES, (
        f"expected a montage-sized resident set to measure against, got {resident} "
        "pages; this test cannot see the defect at this scale"
    )

    counter = _Counter(monkeypatch, page_table_module.PageTable, "_owners_for")
    montage.commit((0,))

    assert counter.count <= 32, (
        f"a 1-tile commit re-derived pin ownership {counter.count} times over a "
        f"{resident}-page resident set; only pages whose pin membership changed "
        "can change state"
    )
