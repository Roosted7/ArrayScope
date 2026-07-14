import numpy as np

from arrayscope.core.view_state import ChannelMode, ScaleMode, ViewState
from arrayscope.display.model.montage_levels import sample_tile_level_stats
from arrayscope.display.shader_mapping import apply_scale
from arrayscope.display.slice_engine import apply_channel
from arrayscope.operations import dim_ops
from arrayscope.operations.evaluator import (
    OperationEvaluator,
    evaluate_level_evidence_snapshot,
)
from arrayscope.operations.pipeline import ArrayDocument, CenteredFFT


def _montage_state(shape, *, channel=ChannelMode.REAL, scale=ScaleMode.LINEAR):
    return (
        ViewState.from_shape(shape)
        .with_montage_axis(2, columns=4, indices=tuple(range(shape[2])), text=":")
        .with_channel(channel)
        .with_scale(scale)
    )


def test_raw_level_evidence_is_source_and_pixel_bounded_without_display_images(monkeypatch):
    import arrayscope.operations.evaluator as evaluator_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("level evidence must not construct a display image")

    monkeypatch.setattr(evaluator_module, "make_image_from_slab", forbidden)
    monkeypatch.setattr(evaluator_module, "make_shader_image_from_slab", forbidden)

    data = np.arange(96 * 128 * 20, dtype=np.float32).reshape(96, 128, 20)
    document = ArrayDocument(data)
    result = evaluate_level_evidence_snapshot(
        document,
        _montage_state(data.shape),
        (0, 5, 19),
        pixel_limit=257,
    )

    assert result.source_indices == (0, 5, 19)
    assert result.source_count == 3
    assert result.pixel_limit == 257
    assert result.max_source_pixels <= 257
    assert result.requested_pixels <= 3 * 257
    assert all(source.stats is not None for source in result.sources)
    assert all(source.stats.refined for source in result.sources)
    assert all(source.sampled_pixels <= 257 for source in result.sources)


def test_complex_level_evidence_uses_semantic_channel_and_scale_without_rgb(monkeypatch):
    import arrayscope.display.slice_engine as slice_engine

    def forbidden(*_args, **_kwargs):
        raise AssertionError("complex evidence must not construct RGB")

    monkeypatch.setattr(slice_engine, "complex_to_rgb", forbidden)
    real = np.arange(24 * 32 * 3, dtype=np.float32).reshape(24, 32, 3) + 1.0
    data = (real + 1j * (real * 0.5)).astype(np.complex64)
    state = _montage_state(data.shape, channel=ChannelMode.COMPLEX, scale=ScaleMode.LOG)

    result = evaluate_level_evidence_snapshot(
        ArrayDocument(data),
        state,
        (0, 2),
        pixel_limit=8192,
    )

    for source in result.sources:
        expected_values = apply_scale(np.abs(data[:, :, source.source_index]), state.scale.value)
        expected = sample_tile_level_stats(expected_values, source.source_index, refined=True)
        assert source.stats is not None
        np.testing.assert_allclose(source.stats.bounds, expected.bounds)
        np.testing.assert_allclose(source.stats.sample, expected.sample)


def test_operation_coupled_sources_reuse_the_existing_stage_cache(monkeypatch):
    calls = {"fft": 0}
    original = dim_ops.centered_fft

    def counted(data, axis, **kwargs):
        calls["fft"] += 1
        return original(data, axis, **kwargs)

    monkeypatch.setattr(dim_ops, "centered_fft", counted)
    data = np.arange(8 * 10 * 4, dtype=np.float32).reshape(8, 10, 4)
    evaluator = OperationEvaluator(ArrayDocument(data, operations=(CenteredFFT(axis=2),)))
    state = _montage_state(evaluator.document.current_shape, channel=ChannelMode.ABS)

    result = evaluator.level_evidence(state, (0, 1, 3), pixel_limit=8192)

    assert calls["fft"] == 1
    assert evaluator.stage_cache_diagnostics().stores == 1
    assert evaluator.stage_cache_diagnostics().hits >= 2
    materialized = evaluator.document.materialize()
    for source in result.sources:
        expected = sample_tile_level_stats(
            apply_channel(materialized[:, :, source.source_index], state.channel),
            source.source_index,
            refined=True,
        )
        assert source.stats is not None
        np.testing.assert_allclose(source.stats.bounds, expected.bounds)
        np.testing.assert_allclose(source.stats.sample, expected.sample)
