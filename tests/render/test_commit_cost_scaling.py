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
whole-montage term and the per-delta term are the same size there.

**Still montage-proportional, and not covered here** (measured shares of a
bounded PyQtGraph commit at 272 tiles, after this change): the montage-wide
histogram source rebuild (~28%) and the presented-identity scan (~24%).  Both
need a design change rather than a cheaper loop — an incrementally accumulated
histogram owner, and presentation truth maintained instead of re-read from Qt
per tile — so they are reported rather than pinned by a passing assertion.
"""

from __future__ import annotations

import contextlib
import os
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
    image = np.full((TILE, TILE), float(tile), dtype=np.float32)
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

    def __init__(self, backend: str, tiles: int):
        self.view = _view_class(backend)()
        self.geometry, self.payloads = _montage(tiles)
        self.tiles = tiles
        self.revision = 1
        self._commit(range(tiles))

    def _commit(self, upsert_tiles):
        upsert_tiles = tuple(int(tile) for tile in upsert_tiles)
        for tile in upsert_tiles:
            self.payloads[tile] = _payload(tile, self.revision)
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
            active_tiles=tuple(range(self.tiles)),
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

    def commit(self, upsert_tiles):
        self._commit(upsert_tiles)

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
