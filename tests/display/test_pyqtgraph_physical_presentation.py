"""Physical-presentation truth gates for the pyqtgraph tile layer.

Mirrors tests/display/test_vispy_physical_presentation.py for the CPU
backend: what the commit stats report must match what the ImageItems
physically draw.
"""

from types import SimpleNamespace

import numpy as np

from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.model.frame import DisplayTilePayload


class _Owner:
    def add_tile_item(self, *_args):
        pass

    def remove_tile_item(self, *_args):
        pass

    def move_tile_item(self, *_args):
        pass


def _make_layer():
    from arrayscope.display.backends.pyqtgraph.tiles import MontageTileLayer

    return MontageTileLayer(
        _Owner(),
        set_image_item_data=lambda *_args, **_kwargs: None,
        record_upload_timing=lambda *_args, **_kwargs: None,
        histogram_levels_for_display=lambda levels: levels,
        is_rgb_image=lambda _image: False,
    )


def test_identity_rejected_upserts_are_reported_not_silent(qt_app):
    """Session-148 gate (2026-07-16): typed-target rejection must be loud.

    A delta upsert whose payload identity cannot satisfy that tile's target
    identity is excluded from presentation.  On the pyqtgraph backend that
    exclusion was completely silent (no stat, no skip count), so a presenter
    re-emitting the same dead payload looped forever while the tile stayed
    empty on screen — invisible to the commit_batch trace, immune to the
    presenter's re-commit backoff, and undetectable by trace_verify.  The
    commit stats must name the rejected tiles so diagnostics and traces
    expose the loop on the first commit.
    """

    from arrayscope.display.model.tile_identity import TileIdentity, TileLodIdentity
    from arrayscope.display.shader_mapping import TexturePlaneKind

    def identity(semantic_generation):
        return TileIdentity(
            document_generation=("doc", 0),
            operation_key=("ops",),
            source_index=0,
            image_axes=(1, 0),
            axis_flips=(False, False),
            channel="real",
            complex_mapping=("scalar", "real", "mapped"),
            texture_kind=TexturePlaneKind.SCALAR_R32F,
            semantic_generation=semantic_generation,
            lod=TileLodIdentity(level=0, factor=1),
        )

    geometry = DisplayGeometry(
        view_state=None,
        display_shape=(4, 4),
        montage=MontageGeometry(indices=(0,), tile_shape=(4, 4), columns=1, rows=1, gap=0),
    )
    image = np.zeros((4, 4), dtype=np.float32)
    payload = DisplayTilePayload(
        0,
        0,
        image,
        None,
        ("tile", 0),
        tile_identity=identity(("stale",)),
    )
    layer = _make_layer()
    delta = SimpleNamespace(
        upserts={0: payload},
        active_tiles=(0,),
        target_identities={0: identity(("current",))},
        removals=(),
        near_tile_source_ids={},
        cold_deadline_ms=None,
    )
    stats = layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        rgb_already_windowed=False,
        dirty_tiles=None,
        tile_payloads={0: payload},
        tile_delta=delta,
    )

    assert stats.committed_upserts == ()
    assert stats.presented_tiles == ()
    assert stats.identity_rejected_items == 1
    assert stats.identity_rejected_tiles == (0,)

    # The matching identity presents normally and reports zero rejections.
    delta.target_identities = {0: identity(("stale",))}
    stats = layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        rgb_already_windowed=False,
        dirty_tiles=None,
        tile_payloads={0: payload},
        tile_delta=delta,
    )
    assert stats.committed_upserts == (0,)
    assert stats.presented_tiles == (0,)
    assert stats.identity_rejected_items == 0
    assert stats.identity_rejected_tiles == ()

    # A retained payload re-presented outside the delta's upserts is not a
    # rejected upsert: the presenter is not looping on it, so it must not
    # trip the loud path even when a new target has outrun the retained
    # payload's identity.
    delta.upserts = {}
    delta.target_identities = {0: identity(("newer",))}
    stats = layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        rgb_already_windowed=False,
        dirty_tiles=None,
        tile_payloads={0: payload},
        tile_delta=delta,
    )
    assert stats.identity_rejected_items == 0
    assert stats.identity_rejected_tiles == ()


def test_atomic_identity_rejection_preserves_complete_predecessor(qt_app):
    """A rejected successor transaction must not punch a physical hole."""

    from arrayscope.display.model.tile_identity import TileIdentity
    from arrayscope.display.shader_mapping import TexturePlaneKind

    def identity(generation):
        return TileIdentity(
            document_generation=("doc", 0),
            operation_key=(),
            source_index=0,
            image_axes=(0, 1),
            axis_flips=(False, False),
            channel="real",
            complex_mapping=("scalar", "real", "mapped"),
            texture_kind=TexturePlaneKind.SCALAR_R32F,
            semantic_generation=generation,
        )

    geometry = DisplayGeometry(
        view_state=None,
        display_shape=(4, 4),
        montage=MontageGeometry(indices=(0,), tile_shape=(4, 4), columns=1, rows=1, gap=0),
    )
    image = np.ones((4, 4), dtype=np.float32)
    predecessor = DisplayTilePayload(
        0,
        0,
        image,
        None,
        ("predecessor",),
        tile_identity=identity(("crop", 97, 197)),
    )
    layer = _make_layer()
    initial_delta = SimpleNamespace(
        upserts={0: predecessor},
        active_tiles=(0,),
        target_identities={0: predecessor.tile_identity},
        removals=(),
        near_tile_source_ids={},
        cold_deadline_ms=None,
        atomic_handoff=False,
    )
    initial = layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        rgb_already_windowed=False,
        dirty_tiles=None,
        tile_payloads={0: predecessor},
        tile_delta=initial_delta,
    )
    assert initial.presented_tiles == (0,)

    stale_successor = DisplayTilePayload(
        0,
        0,
        image * 2,
        None,
        ("stale-successor",),
        tile_identity=identity(("crop", 96, 196)),
    )
    current_target = identity(("crop", 94, 194))
    rejected_delta = SimpleNamespace(
        upserts={0: stale_successor},
        active_tiles=(0,),
        target_identities={0: current_target},
        removals=(),
        near_tile_source_ids={},
        cold_deadline_ms=None,
        atomic_handoff=True,
    )
    rejected = layer.update_presentation(
        None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        rgb_already_windowed=False,
        dirty_tiles=None,
        tile_payloads={0: stale_successor},
        tile_delta=rejected_delta,
    )

    assert rejected.presented_tiles == (0,)
    assert rejected.committed_upserts == ()
    assert rejected.identity_rejected_tiles == (0,)
    assert layer._states[0].acknowledged_identity == predecessor.tile_identity
    assert layer._states[0].visible is True
