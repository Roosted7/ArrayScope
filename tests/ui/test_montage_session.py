import numpy as np

from arrayscope.core.view_state import ViewState
from arrayscope.display.montage import RenderedTile, make_montage_plan
from arrayscope.window.montage_session import MontageRenderSession


def _session():
    state = ViewState.from_shape((2, 2, 4)).with_montage_axis(2, indices=(0, 1, 2, 3), text=":")
    plan = make_montage_plan(state, axis=2, indices=(0, 1, 2, 3), tile_shape=(2, 2), columns=4)
    return MontageRenderSession(
        session_id=1,
        key="key",
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
        previous_levels=None,
        previous_bounds=None,
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=list(plan.tiles[:3]),
    )


def test_montage_render_session_returns_pending_tiles_in_order():
    session = _session()

    assert session.next_tile().source_index == 0
    assert session.next_tile().source_index == 1


def test_montage_render_session_mark_loaded_moves_tile_from_loading_to_loaded():
    session = _session()
    tile = session.next_tile()
    rendered = RenderedTile(tile, np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32), 0.0, (2, 2), 16)

    session.mark_loaded(rendered)

    assert session.is_tile_loaded(tile)
    assert int(tile.montage_index) not in session.loading_tiles
    assert tile not in session.pending_tiles


def test_montage_render_session_keeps_skipped_separate_from_pending():
    session = _session()
    session.mark_skipped(session.plan.tiles[1])

    assert 1 in session.skipped_tiles
    assert session.plan.tiles[1] not in session.pending_tiles
    assert session.next_tile().source_index == 0
    assert session.next_tile().source_index == 2
