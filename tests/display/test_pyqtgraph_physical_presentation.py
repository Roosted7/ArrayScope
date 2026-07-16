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
