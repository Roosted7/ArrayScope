import numpy as np

from arrayscope.core.view_state import ViewState
from arrayscope.display.geometry import DisplayGeometry
from arrayscope.display.montage import make_montage_plan
from arrayscope.display.slice_engine import DisplayImage
from arrayscope.operations.evaluator import EvaluationResult, OperationEvaluator, _document_key
from arrayscope.operations.pipeline import ArrayDocument
from arrayscope.operations.tile_regions import TileRegionRequest
from arrayscope.window.display_frame import CommittedDisplayFrame, DisplayFrameKey
from arrayscope.window.tile_data_provider import TileDataProvider


def _setup():
    data = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    state = ViewState.from_shape(data.shape).with_image_axes(0, 1).with_line_axis(2).with_montage_axis(2, columns=2, indices=(0, 1, 2, 3), text=":")
    plan = make_montage_plan(state, axis=2, indices=(0, 1, 2, 3), tile_shape=(2, 3), columns=2, gap=1)
    document = ArrayDocument(data)
    evaluator = OperationEvaluator(document)
    return data, state, plan, document, evaluator


def _request(document, tile, state, region=(slice(0, 2), slice(0, 3))):
    return TileRegionRequest(
        document_key=_document_key(document),
        view_state=tile.view_state,
        montage_axis=state.montage_axis,
        source_index=tile.source_index,
        tile_number=tile.montage_index,
        tile_local_region=region,
        purpose="roi",
    )


def test_tile_region_provider_uses_committed_canvas_before_evaluation():
    data, state, plan, document, evaluator = _setup()
    tile = plan.tiles[1]
    canvas = np.full((2, 7), np.nan, dtype=float)
    canvas[:, 4:7] = 99.0
    geometry = DisplayGeometry(state, canvas.shape, montage=plan.geometry)
    frame = CommittedDisplayFrame(
        data=canvas,
        histogram_data=canvas.copy(),
        geometry=geometry,
        levels=(0.0, 100.0),
        histogram_range=(0.0, 100.0),
        key=DisplayFrameKey(_document_key(document), ("test",), 1),
    )
    provider = TileDataProvider(operation_evaluator=evaluator, document=document, committed_frame=frame, montage_plan=plan)

    result = provider.request_tile_region(_request(document, tile, state, (slice(0, 2), slice(0, 2))))

    assert result.source == "committed_canvas"
    np.testing.assert_array_equal(result.image, np.full((2, 2), 99.0))


def test_tile_region_provider_uses_committed_direct_tile_payload_before_canvas_placeholder():
    _data, state, plan, document, evaluator = _setup()
    from arrayscope.window.display_frame import DisplayTilePayload, TiledValueSource

    tile = plan.tiles[1]
    placeholder = np.full((2, 7), -1.0, dtype=float)
    geometry = DisplayGeometry(state, placeholder.shape, montage=plan.geometry)
    image = np.arange(6, dtype=float).reshape(2, 3) + 100.0
    histogram = image * 2.0
    frame = CommittedDisplayFrame(
        data=None,
        histogram_data=None,
        geometry=geometry,
        levels=(0.0, 200.0),
        histogram_range=(0.0, 200.0),
        key=DisplayFrameKey(_document_key(document), ("test",), 1),
        value_source=TiledValueSource(
            {
                1: DisplayTilePayload(
                    tile_number=1,
                    source_index=tile.source_index,
                    image=image,
                    histogram_data=histogram,
                    source_id=("payload", 1),
                )
            }
        ),
    )
    provider = TileDataProvider(operation_evaluator=evaluator, document=document, committed_frame=frame, montage_plan=plan)

    result = provider.request_tile_region(_request(document, tile, state, (slice(0, 2), slice(1, 3))))

    assert result.source == "committed_tile_payload"
    np.testing.assert_array_equal(result.image, image[:, 1:3])
    np.testing.assert_array_equal(result.histogram_data, histogram[:, 1:3])

def test_tile_region_provider_reuses_region_cache():
    _data, state, plan, document, evaluator = _setup()
    tile = plan.tiles[2]
    provider = TileDataProvider(operation_evaluator=evaluator, document=document, montage_plan=plan)
    request = _request(document, tile, state, (slice(0, 1), slice(0, 2)))

    first = provider.request_tile_region(request)
    second = provider.request_tile_region(request)

    assert first.source == "computed"
    assert second.source == "region_cache"
    np.testing.assert_array_equal(first.image, second.image)


def test_tile_region_provider_reuses_rendered_tile_cache():
    _data, state, plan, document, evaluator = _setup()
    tile = plan.tiles[3]
    cached = np.full((2, 3), 7.0, dtype=float)
    evaluator.store_montage_tile_result(
        tile,
        montage_axis=state.montage_axis,
        colormap_lut=None,
        result=EvaluationResult(DisplayImage(cached, histogram_data=cached), eval_ms=0.0, slab_shape=cached.shape, slab_nbytes=cached.nbytes),
    )
    provider = TileDataProvider(operation_evaluator=evaluator, document=document, montage_plan=plan)

    result = provider.request_tile_region(_request(document, tile, state, (slice(1, 2), slice(1, 3))))

    assert result.source == "tile_cache"
    np.testing.assert_array_equal(result.image, np.full((1, 2), 7.0))


def test_tile_region_demand_does_not_create_visible_montage_session():
    _data, state, plan, document, evaluator = _setup()
    provider = TileDataProvider(operation_evaluator=evaluator, document=document, montage_plan=plan)
    provider.visible_session = None

    result = provider.request_tile_region(_request(document, plan.tiles[0], state))

    assert result.source == "computed"
    assert getattr(provider, "visible_session", None) is None
