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

**No wall-clock assertion survives.**  One did until 2026-08-01 — ``a commit
with nothing to present must not cost more than one presenting 32 tiles`` —
and it called itself ratio-free and threshold-free.  That was true of the
comparison and false of the measurement: both numbers are ~1-7 ms, so under
three concurrent suites either could absorb a scheduling delay large enough to
invert them, and the ``[pyqtgraph]`` arm did exactly that while passing
serially.  No threshold was involved and it was still a stopwatch.

What it was standing in for is structural: an empty delta must stay on the
incremental path.  Both backends already publish that verdict, so the
restatement below asserts the fact instead of its duration.

Forcing the named defect back three ways on this machine — WGPU
``_wgpu_tiled_binding_reusable`` pinned to ``False`` (the historical
no-``TileIdentity`` shape), PyQtGraph committing with ``force_refresh``, and
PyQtGraph committing with no delta at all (the whole-active-set republication)
— every counter this test reads goes 0 -> 272, and the stopwatch it replaces
managed:

- WGPU binding reuse refused: **red**, 14.8 ms against 7.5 ms.
- PyQtGraph ``force_refresh``: **red by 14%**, 4.54 ms against 3.97 ms — the
  same margin three concurrent suites invert.
- PyQtGraph delta dropped: **green**, 4.29 ms against 6.69 ms, still
  "cheaper".

That last row is the finding.  PyQtGraph has no empty-delta-only fast path, so
a fixed cost it regains lands on *both* arms of a ``<=`` and largely cancels;
the arm that was flaking under load was the arm least able to fail for the
reason it named.

The commit *durations* are not lost and are not a gate: the field-scale table
lives in ``docs/architecture/progressive-render-contract.md``.  Per
``docs/testing/strategy.md``, a duration whose fact the product already
publishes needs neither a benchmark nor a looser constant — it needs the
counter.

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
        #: The backend's own acknowledgement of the most recent commit.
        self.report = None
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
        self.report = self.view.setTiledPresentation(
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
        return self.report

    def per_tile_work(self):
        """What this commit published about the tiles it touched.

        ``items_skipped`` is deliberately absent: the two backends do not mean
        the same thing by it.  PyQtGraph increments it per tile that entered
        the item loop and needed nothing — a walk.  WGPU derives it as
        ``len(committed_tiles) - len(updated)``, so it reads 272 on a commit
        that walked nothing at all.  Each backend's own reading of it is
        asserted where it means something, not here.
        """

        timing = self.view.lastImageUploadTiming()
        return {
            "items_created": int(timing.tile_layer_items_created),
            "items_updated": int(timing.tile_layer_items_updated),
            "image_replacements": int(timing.tile_layer_image_replacements),
            "texture_uploads": int(timing.tile_layer_texture_uploads),
            "existing_items_shown": int(timing.tile_layer_existing_items_shown),
            "relocated_tiles": int(timing.tile_layer_relocated_tiles),
        }


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
def test_an_empty_delta_stays_on_the_incremental_path(backend, qt_app):
    """A commit with nothing to present must touch no tile.

    This is what the wall-clock comparison it replaces was standing in for
    (see the module docstring).  It asks the same question in two steps: which
    path the commit took, in the vocabulary that backend decides in, and then
    what that path cost in tiles.

    The two backends reach the same place by different mechanisms, so the
    first step is per backend.  WGPU has a whole-binding fast path only an
    empty delta is eligible for and publishes which of its three paths it
    took.  PyQtGraph has no such path — its incremental commit is an empty
    item order — so the reading is the item loop's own: a tile that enters it
    and turns out to need nothing.

    The second step is shared and is the claim proper.  Forced off the
    incremental path on this machine (three ways: WGPU binding reuse refused,
    PyQtGraph committing with ``force_refresh``, PyQtGraph committing with no
    delta at all), every one of these counters goes 0 -> 272.

    The predecessor is a 32-tile commit rather than the initial full one, on
    purpose: reuse after a *bounded* commit is the state a refining montage is
    actually in, and ``test_empty_delta_takes_the_production_fast_path``
    already covers reuse straight after a full build.
    """

    montage = _Montage(backend, FIELD_TILES)

    working = montage.commit(range(32))
    working_work = montage.per_tile_work()
    empty = montage.commit(())
    empty_work = montage.per_tile_work()
    empty_skipped = int(montage.view.lastImageUploadTiming().tile_layer_items_skipped)

    # Control.  Every assertion below is a zero, and a zero proves nothing
    # about a commit path that presents nothing under any delta.  The 32-tile
    # commit has to be seen doing 32 tiles of work first, or this test is
    # reporting confidence it has not earned.
    control = (working.texture_uploads, working_work["items_updated"])
    assert control == (32, 32), (
        f"[{backend}] a 32-tile commit reported {working.texture_uploads} presented "
        f"tiles and {working_work['items_updated']} item updates, not 32 and 32; the "
        "empty-delta zeros below would then be the zeros of a backend that "
        "presents nothing, and this test cannot see the defect it names"
    )

    # 1. The path.
    if backend == "wgpu":
        paths = (
            empty.binding_fast_path_commits,
            empty.binding_incremental_commits,
            empty.binding_full_republications,
        )
        assert paths == (1, 0, 0), (
            "an empty delta did not reuse the submitted WGPU binding "
            f"(fast_path, incremental, full_republication) = {paths}; it left the "
            "fast path and re-established the montage binding"
        )
    else:
        assert empty_skipped == 0, (
            f"an empty delta walked {empty_skipped} tiles of a {FIELD_TILES}-tile "
            "montage in the PyQtGraph item loop and needed none of them; the delta "
            "names the tiles to visit and it named none"
        )

    # 2. What the path cost, in tiles.
    assert empty_work == dict.fromkeys(empty_work, 0), (
        f"[{backend}] an empty delta into a {FIELD_TILES}-tile montage touched "
        f"tiles: { ({name: count for name, count in empty_work.items() if count}) }; "
        "it has nothing to present, so it must leave every presented tile alone"
    )
    acknowledged = (empty.cold_count, empty.resident_rebinds)
    assert acknowledged == (0, 0), (
        f"[{backend}] the report for an empty delta claims {empty.cold_count} cold "
        f"tiles and {empty.resident_rebinds} resident rebinds; a commit that "
        "presents nothing acknowledges nothing"
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


def _presentation_state(tiles: int):
    _geometry, payloads = _montage(tiles)
    return TilePresentationState(payloads, revision=1), payloads


def _presentation(geometry, tile_state, payloads, upserts, *, level_revision=1):
    from arrayscope.display.model.commit import DisplayTiledPresentation

    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=level_revision,
        histogram_revision=1,
        viewport_revision=1,
        base_revision=0,
        target_revision=1,
        upserts={tile: payloads[tile] for tile in upserts},
        removals=(),
        active_tiles=tuple(sorted(payloads)),
        planned_tiles=tuple(sorted(payloads)),
    )
    return DisplayTiledPresentation(
        geometry=geometry,
        levels=(0.0, 1.0),
        histogram_range=(0.0, 1.0),
        viewport_policy=ViewportPolicy.PRESERVE,
        tile_state=tile_state,
        base_tile_state=tile_state,
        tile_delta=delta,
        tile_residency_budget_bytes=0,
    )


def test_presentation_rebinds_only_the_payloads_whose_identity_moved(monkeypatch):
    """Binding a transaction to its levels costs the delta, not the montage.

    Every payload the transaction carries must name the accepted levels, and
    the way to guarantee that was to rebuild every wrapper.  A montage's worth
    of ``replace`` per commit is a fixed term no cohort size can chunk away
    (R5), and it is unnecessary: a wrapper already carrying exactly this
    transaction's presentation identity is already bound.
    """

    from arrayscope.display.model import commit as commit_module

    geometry, payloads = _montage(FIELD_TILES)
    tile_state = TilePresentationState(payloads, revision=1)

    first = _presentation(geometry, tile_state, payloads, (0,))
    bound_state = first.tile_state

    counter = _Counter(monkeypatch, commit_module, "replace")
    second = _presentation(geometry, bound_state, dict(bound_state.payloads), (0,))

    assert counter.count == 0, (
        f"re-presenting a {FIELD_TILES}-tile montage at unchanged levels rebuilt "
        f"{counter.count} wrappers; every one of them already named this "
        "transaction's presentation identity"
    )
    assert second.tile_state is bound_state, (
        "nothing moved, so the transaction must keep the state it was handed"
    )
    for tile in (0, FIELD_TILES // 2, FIELD_TILES - 1):
        assert second.tile_state.payloads[tile] is bound_state.payloads[tile]


def test_a_moved_level_generation_still_rebinds_every_transaction_payload(monkeypatch):
    """The skip above is identity equality, never an assumption of sameness."""

    from arrayscope.display.model import commit as commit_module

    geometry, payloads = _montage(FIELD_TILES)
    tile_state = TilePresentationState(payloads, revision=1)
    bound = _presentation(geometry, tile_state, payloads, (0,)).tile_state

    counter = _Counter(monkeypatch, commit_module, "replace")
    moved = _presentation(
        geometry,
        bound,
        dict(bound.payloads),
        (0,),
        level_revision=2,
    )

    # One ``replace`` per payload, plus the delta and nothing else.
    assert counter.count >= FIELD_TILES, (
        f"a new level generation rebound only {counter.count} of {FIELD_TILES} "
        "payloads; every wrapper in the transaction names the generation"
    )
    assert all(
        int(payload.presentation_identity.levels_generation) == 2
        for payload in moved.tile_state.payloads.values()
    )


def test_plan_tile_index_is_derived_once_per_plan():
    """The slot -> tile map is a plan property, not a per-commit walk."""

    from arrayscope.display.montage import plan_tiles_by_number

    view_state = (
        ViewState.from_shape((TILE, TILE, FIELD_TILES))
        .with_image_axes(0, 1)
        .with_montage_axis(2, columns=COLUMNS, indices=tuple(range(FIELD_TILES)))
    )
    from arrayscope.display.montage import make_montage_plan

    plan = make_montage_plan(
        view_state,
        axis=2,
        indices=tuple(range(FIELD_TILES)),
        tile_shape=(TILE, TILE),
        columns=COLUMNS,
    )

    first = plan_tiles_by_number(plan)
    assert plan_tiles_by_number(plan) is first, (
        "a second reader re-derived the map; the plan is immutable, so the "
        "walk belongs to the plan and not to every commit that consults it"
    )
    assert first == {int(tile.montage_index): tile for tile in plan.tiles}

    other = make_montage_plan(
        view_state,
        axis=2,
        indices=tuple(range(FIELD_TILES - 1)),
        tile_shape=(TILE, TILE),
        columns=COLUMNS,
    )
    assert plan_tiles_by_number(other) is not first, "a different plan owns a different map"


# --------------------------------------------------------------------------
# The same rule one layer up: the SESSION's presentation delta build.
#
# Everything above measures what the backend does with a delta.  This section
# measures what building that delta costs, because that is where the field
# trace put the money: ``payload`` — the presentation delta build — is 39 % of
# ``montage_commit``, itself 17 % of session wall (trace 7, 2026-08-04).
#
# Measured on this fixture, a ONE-tile delta into a settled 272-tile montage
# executed 155 690 lines of arrayscope bytecode and a 272-tile delta executed
# 473 356.  The build is therefore ~97 % fixed cost at a one-tile delta, and
# the honest reading is that most of that fixed term is IRREDUCIBLE: a commit
# has to reconcile every drawn slot against backend identity truth, every
# retained payload against the camera, and every planned slot against its
# lifecycle target.  Those walks are the correctness of the commit and are
# deliberately not pinned smaller here.
#
# What is pinned here is the part that was never montage-proportional by
# necessity — asking one question twice, and deriving one pure value again for
# every reader.  Tile PIXELS are irrelevant to a count, so these use a small
# tile shape; the tile COUNT is the thing that must stay at field scale, for
# exactly the reason the module docstring gives.
# --------------------------------------------------------------------------

SESSION_TILE = (4, 4)


def _field_session(tiles: int = FIELD_TILES):
    """A settled montage of ``tiles`` presented, acknowledged, drawn tiles."""

    from arrayscope.display.model.tile_identity import tile_ack_identity
    from arrayscope.display.montage import RenderedTile, make_montage_plan
    from arrayscope.window.frame_session import FrameSession

    rows, cols = SESSION_TILE
    view_state = (
        ViewState.from_shape((rows, cols, tiles))
        .with_image_axes(0, 1)
        .with_montage_axis(2, columns=COLUMNS, indices=tuple(range(tiles)))
    )
    plan = make_montage_plan(
        view_state,
        axis=2,
        indices=tuple(range(tiles)),
        tile_shape=SESSION_TILE,
        columns=COLUMNS,
    )
    session = FrameSession(
        session_id=1,
        key="key",
        render_generation=1,
        level_key="levels",
        level_expected_indices=tuple(range(tiles)),
        plan=plan,
        view_state=view_state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(1024, 1024),
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
    session.visible_tile_numbers = frozenset(range(tiles))
    session.sync_lifecycle_scope()

    presented = {}
    for index in range(tiles):
        tile = plan.tiles[index]
        image = np.full(SESSION_TILE, float(index), dtype=np.float32)
        session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
        payload = DisplayTilePayload(
            tile_number=index,
            source_index=int(tile.source_index),
            image=image,
            histogram_data=image,
            source_id=("tile", int(tile.source_index), "exact"),
        )
        session.display_tile_payloads[index] = payload
        session.record_tile_payload(payload)
        presented[index] = payload
        session.lifecycle.commit_emitted({index: payload})
        session.lifecycle.backend_ack({index: payload})
        session.lifecycle.acknowledge_presented(index, payload.source_id, payload.quality, 0)
    session.tile_presentation_state = TilePresentationState(presented)
    session.lifecycle.presentation_confirmed(tuple(range(tiles)))
    session.lifecycle.backend_presented_snapshot(
        {index: tile_ack_identity(payload) for index, payload in presented.items()}
    )
    session.display_committed = True
    # Drain the residue of settling to quiescence, acknowledging each round the
    # way the backend does.  Without the acknowledgement the queue never
    # empties, the next build takes the paced-followup shortcut instead of the
    # reconciling one, and a counting test measures a build that never ran.
    settle_ids = {index: ("tile", index, "exact") for index in range(tiles)}
    for _ in range(40):
        if not session.dirty_payloads and not session.pending_payload_upserts:
            break
        state, delta = session.build_tile_presentation(settle_ids, max_upserts=tiles)
        if not delta.upserts:
            break
        session.lifecycle.commit_emitted(dict(delta.upserts))
        session.lifecycle.backend_ack(dict(delta.upserts))
        for index, payload in delta.upserts.items():
            session.lifecycle.acknowledge_presented(
                int(index), payload.source_id, payload.quality, 0
            )
        session.lifecycle.presentation_confirmed(tuple(delta.upserts))
        session.lifecycle.backend_presented_snapshot(
            {int(i): tile_ack_identity(p) for i, p in delta.upserts.items()}
        )
        session.tile_presentation_state = state
    unsettled = "the fixture montage never settled, so a build measured against it is "
    unsettled += "not the one-tile-delta build these tests claim to measure"
    assert not session.dirty_payloads, unsettled
    assert not session.pending_payload_upserts, unsettled
    return session


def _dirty_one_tile(session, index: int, tiles: int):
    from arrayscope.display.montage import RenderedTile

    tile = session.plan.tiles[index]
    image = np.full(SESSION_TILE, float(index) + 1e4, dtype=np.float32)
    session.mark_materialized(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
    session.dirty_payloads[index] = None
    return {i: ("tile", i, "moved" if i == index else "exact") for i in range(tiles)}


class _MethodCounter:
    """Count calls to one method or property for the duration of a build."""

    def __init__(self, monkeypatch, owner, name, *, prop=False):
        self.count = 0
        original = owner.__dict__[name]
        if prop:
            getter = original.fget

            def counted_get(instance):
                self.count += 1
                return getter(instance)

            monkeypatch.setattr(owner, name, property(counted_get))
        else:

            def counted(instance, *args, **kwargs):
                self.count += 1
                return original(instance, *args, **kwargs)

            monkeypatch.setattr(owner, name, counted)


def test_presentation_build_derives_each_payload_ref_once(monkeypatch):
    """A pure function of an immutable payload is derived once, not per reader.

    ``payload_ref_from_display_payload`` normalizes a ``DisplayTilePayload``
    into the identity metadata the lifecycle reasons with.  The payload is a
    frozen dataclass of frozen dataclasses, so that ref cannot change — yet a
    single 272-tile build asked for it 1115 times over 282 distinct payload
    objects (measured 2026-08-04), because the target scan, the fallback scan,
    the acknowledged-identity lookup and the currency check each rebuilt it.
    The derivation and the construction it drives were 24 % of the whole build.

    The gate is the RATIO, not a constant: however many payloads a build
    consults, it may run the derivation body at most once for each.
    """

    from arrayscope.display.model.frame import DisplayTilePayload as _Payload
    from arrayscope.presentation import tile_lifecycle

    session = _field_session()
    source_ids = _dirty_one_tile(session, 7, FIELD_TILES)

    consulted: set[int] = set()
    original = tile_lifecycle.payload_ref_from_display_payload

    def recording(payload):
        consulted.add(id(payload))
        return original(payload)

    monkeypatch.setattr(tile_lifecycle, "payload_ref_from_display_payload", recording)
    # Reached only on the derivation path; a reader served the cached ref never
    # touches it.  Counting it counts derivations, not lookups.
    derivations = _MethodCounter(monkeypatch, _Payload, "conservative_actual_lod_level", prop=True)

    session.build_tile_presentation(source_ids, max_upserts=1)

    assert consulted, "the build consulted no payload at all; the fixture is inert"
    assert derivations.count <= len(consulted), (
        f"a 1-tile delta into a {FIELD_TILES}-tile montage ran the payload-ref "
        f"derivation {derivations.count} times over {len(consulted)} distinct "
        "payloads; the payload is immutable, so every reader after the first "
        "is rebuilding a value that cannot have changed"
    )


def test_payload_ref_cache_still_answers_a_replaced_payload():
    """The memo is per-object, so new pixels get a new ref — not a stale one.

    A memo on an immutable object is only safe if a CHANGED payload is a
    different object.  ``dataclasses.replace`` makes one, and this pins that
    the derived ref follows the replacement rather than the slot.
    """

    from dataclasses import replace as _replace

    from arrayscope.presentation.tile_lifecycle import payload_ref_from_display_payload

    image = np.zeros(SESSION_TILE, dtype=np.float32)
    first = DisplayTilePayload(
        tile_number=3,
        source_index=3,
        image=image,
        histogram_data=None,
        source_id=("tile", 3, "a"),
        quality="preview",
    )
    ref = payload_ref_from_display_payload(first)
    assert payload_ref_from_display_payload(first) is ref, "the same object re-derived"

    second = _replace(first, source_id=("tile", 3, "b"), quality="exact")
    other = payload_ref_from_display_payload(second)
    assert other is not ref
    assert other.source_id == ("tile", 3, "b")
    assert other.quality == "exact"
    assert ref.source_id == ("tile", 3, "a"), "caching must not mutate the first ref"
    assert ref.quality == "fallback"


def test_pending_removal_reconciliation_does_not_scale_with_the_montage(monkeypatch):
    """Discharging an empty removal queue must not poll every planned tile.

    The transaction boundary retires queued removals whose slot turned out to
    hold current pixels after all.  It asked ``payload_is_current`` for every
    planned tile to do it — 272 lifecycle ref normalizations to discharge a
    queue that is usually empty.  ``discard`` is a no-op for anything not
    queued, so only the queued tiles can change the outcome.

    Two montage sizes, one identical delta: the count must not move.
    """

    from arrayscope.presentation.tile_lifecycle import TileLifecycle

    counts = {}
    for tiles in (FIELD_TILES // 4, FIELD_TILES):
        session = _field_session(tiles)
        source_ids = _dirty_one_tile(session, 5, tiles)
        counter = _MethodCounter(monkeypatch, TileLifecycle, "payload_is_current")
        session.build_tile_presentation(source_ids, max_upserts=1)
        counts[tiles] = counter.count
        monkeypatch.undo()

    small, field = counts[FIELD_TILES // 4], counts[FIELD_TILES]
    assert field == small, (
        f"one tile changed in both montages, yet the build asked "
        f"'is this payload current?' {small} times into a {FIELD_TILES // 4}-tile "
        f"montage and {field} times into a {FIELD_TILES}-tile one; the answer "
        "for a tile nobody queued for removal cannot change anything"
    )


def test_first_pixel_membership_is_asked_once_per_tile(monkeypatch):
    """Coverage and re-arm ask the same question; they may ask it once.

    ``unpresented_tiles`` and ``_rearm_required_first_pixel_payloads`` both
    need "does this slot own a first pixel", over scopes that in a settled
    montage are the same scope.  Asked separately, each answer cost a
    lifecycle record's acknowledged-payload read and a 272-tile montage paid
    for 544 of them per commit.
    """

    from arrayscope.presentation.tile_lifecycle import TileRecord

    session = _field_session()
    source_ids = _dirty_one_tile(session, 11, FIELD_TILES)
    counter = _MethodCounter(monkeypatch, TileRecord, "first_pixel_presented", prop=True)

    session.build_tile_presentation(source_ids, max_upserts=1)

    assert 0 < counter.count <= FIELD_TILES, (
        f"the build read first-pixel presentation {counter.count} times across "
        f"{FIELD_TILES} tiles; the two readers share one scope and one moment, "
        "so one pass answers both"
    )


# --------------------------------------------------------------------------
# The same rule applied to the commit's WRITES.
#
# Everything above counts what a commit READS.  This section counts what it
# CHANGES, which is the other half of "a commit costs what its delta costs"
# and the half with a consumer: ``FrameSession.mutation_revision`` is the
# epoch ``viewport_invariant_key`` carries as ``lifecycle_revision``, and a
# planner that finds it moved replans the whole viewport.
#
# Measured 2026-08-05 before the no-op-write rule landed, on this fixture: a
# commit into a settled 272-tile montage that created nothing, updated nothing
# and uploaded nothing moved the session epoch by **2460** and the ledger's by
# **1904**.  Every one of those calls stored the object the slot already held
# — 1088 record attribute stores, 816 set add/discard, 544 dict pops of absent
# keys.  A genuine one-tile delta moved the ledger by 1909: five parts signal
# in 1909.
#
# The gates below are threshold-free.  They run the SAME delta into two
# montage sizes and require the epoch movement to be equal, because the defect
# is not "this number is large" — it is "this number is a function of the
# montage instead of the delta".  Before the rule: 238 against 1904.
# --------------------------------------------------------------------------


def _settled_source_ids(tiles: int) -> dict[int, object]:
    return {index: ("tile", index, "exact") for index in range(tiles)}


def test_a_commit_that_changes_nothing_moves_no_epoch():
    """The absolute case, where the answer is a number and not a comparison.

    A settled montage re-committed with the same source ids creates nothing,
    updates nothing and uploads nothing.  There is no state for it to change,
    so there is nothing for the epoch to report, at any montage size.
    """

    for tiles in (FIELD_TILES // 4, FIELD_TILES):
        session = _field_session(tiles)
        before_session = session.mutation_revision
        before_ledger = session.lifecycle.mutation_revision

        _state, delta = session.build_tile_presentation(_settled_source_ids(tiles), max_upserts=8)

        unsettled = "the fixture is not settled, so this is not the commit the test names"
        assert not delta.upserts, unsettled
        assert not delta.removals, unsettled
        assert session.lifecycle.mutation_revision == before_ledger, (
            f"a {tiles}-tile commit with nothing to present moved the ledger epoch by "
            f"{session.lifecycle.mutation_revision - before_ledger}; every write it "
            "makes stores the object the record already holds"
        )
        assert session.mutation_revision - before_session <= 8, (
            f"a {tiles}-tile commit with nothing to present moved the session epoch by "
            f"{session.mutation_revision - before_session}; the handful that remain are "
            "the admission bookkeeping fields, which genuinely change"
        )


def test_the_epoch_a_commit_moves_is_the_size_of_its_delta_not_the_montage():
    """One tile changed in both montages: the same epoch movement, exactly.

    This is the gate.  A constant would have to be re-tuned as the commit
    gains and loses legitimate writes; the comparison holds whatever that
    number is, and fails the moment any per-montage-tile write returns.
    """

    moved = {}
    for tiles in (FIELD_TILES // 4, FIELD_TILES):
        session = _field_session(tiles)
        source_ids = _dirty_one_tile(session, 5, tiles)
        before = session.lifecycle.mutation_revision
        session.build_tile_presentation(source_ids, max_upserts=1)
        moved[tiles] = session.lifecycle.mutation_revision - before

    small, field = moved[FIELD_TILES // 4], moved[FIELD_TILES]
    assert field == small, (
        f"one tile changed in both montages, yet the commit moved the ledger epoch "
        f"by {small} into a {FIELD_TILES // 4}-tile montage and by {field} into a "
        f"{FIELD_TILES}-tile one. A commit is montage-wide exactly when this "
        "difference is nonzero, and ``viewport_invariant_key`` reads the result"
    )
    assert 0 < field < FIELD_TILES // 4, (
        f"the commit moved the ledger epoch by {field}; a one-tile delta is a "
        "handful of writes, and zero would mean the fixture never changed anything"
    )
