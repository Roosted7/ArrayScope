"""Headless replay preserves authored operations and owns only its input map."""

import json
import os

import numpy as np
import pytest

from arrayscope.core.array_source import NdArraySource, SourceReadRefused
from arrayscope.io import recipe_replay
from arrayscope.io.recipe_replay import (
    BatchExportNeedsConfirmation,
    BatchExportRequest,
    run_batch_export,
)
from arrayscope.operations import recipes
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.pipeline import CenteredFFTN, Crop, OperationStep, ReverseAxis, Scale


@pytest.fixture
def source_observations(monkeypatch):
    maps = []
    reads = []
    original_load = np.load
    original_read = NdArraySource.read_region

    def load(*args, **kwargs):
        result = original_load(*args, **kwargs)
        if isinstance(result, np.memmap):
            maps.append(result._mmap)
            assert not result.flags.writeable
        return result

    def read(source, spec, **kwargs):
        reads.append(tuple(spec))
        return original_read(source, spec, **kwargs)

    monkeypatch.setattr(recipe_replay.np, "load", load)
    monkeypatch.setattr(NdArraySource, "read_region", read)
    yield maps, reads
    assert all(mapping.closed for mapping in maps)


def _input(tmp_path, data):
    path = tmp_path / "source.npy"
    np.save(path, data)
    return path


def _recipe(tmp_path, steps):
    path = tmp_path / "operations.json"
    path.write_text(recipes.dumps_recipe(steps), encoding="utf-8")
    return path


@pytest.mark.parametrize("shape", [(), (2, 1, 1), (0, 3), (3, 4, 2)])
def test_dry_run_preserves_exact_source_shape_without_reads_or_output(
    tmp_path, source_observations, shape
):
    path = _input(tmp_path, np.empty(shape, dtype=np.complex64))
    output = tmp_path / "selected.npy"
    before = set(tmp_path.iterdir())

    result = run_batch_export(BatchExportRequest(path, output, squeeze=False), dry_run=True)

    maps, reads = source_observations
    assert maps
    assert maps[0].closed
    assert reads == []
    assert result.input_shape == shape
    assert result.output_shape == shape
    assert result.dtype == "complex64"
    assert result.output_bytes == np.prod(shape, dtype=int) * 8
    assert result.paths == ()
    assert result.dry_run
    assert set(tmp_path.iterdir()) == before
    json.dumps(result.summary)


def test_complex_recipe_and_output_ranges_match_numpy_and_keep_disabled_steps(
    tmp_path, source_observations
):
    rng = np.random.default_rng(7)
    data = rng.normal(size=(6, 5, 5)) + 1j * rng.normal(size=(6, 5, 5))
    path = _input(tmp_path, data)
    steps = (
        OperationStep(Crop(0, 1, 5)),
        OperationStep(Scale(factor=1000), enabled=False),
        OperationStep(ReverseAxis(1)),
        OperationStep(CenteredFFTN((0, 2))),
    )
    recipe = _recipe(tmp_path, steps)
    output = tmp_path / "selected.NPZ"

    result = run_batch_export(
        BatchExportRequest(
            path,
            output,
            recipe,
            ranges=((0, 1, 3), (1, 2, 3)),
            squeeze=False,
            allow_expensive=True,
        )
    )

    transformed = data[1:5, ::-1]
    # ArrayScope's MRI forward convention is inverse, centered, and unitary.
    # The odd axis makes shift direction observable independently of scale.
    expected = np.fft.ifftshift(
        np.fft.ifftn(np.fft.fftshift(transformed, axes=(0, 2)), axes=(0, 2), norm="ortho"),
        axes=(0, 2),
    )[1:3, 2:3, :]
    with np.load(output, allow_pickle=False) as saved:
        np.testing.assert_allclose(saved["array"], expected, rtol=1e-12, atol=1e-12)
        assert json.loads(saved["recipe_json"].item())["operations"][1]["enabled"] is False
        selection = json.loads(saved["selection_json"].item())
        assert selection["ranges"] == [[1, 3], [2, 3], [0, 5]]
        assert selection["output_shape"] == [2, 1, 5]
    assert result.output_shape == expected.shape
    assert result.dtype == str(expected.dtype)
    assert result.paths == (output,)
    np.testing.assert_array_equal(np.load(path), data)


@pytest.mark.parametrize(("squeeze", "shape"), [(True, ()), (False, (1, 1))])
def test_selected_singletons_are_only_squeezed_when_requested(
    tmp_path, source_observations, squeeze, shape
):
    path = _input(tmp_path, np.arange(12).reshape(3, 4))
    output = tmp_path / "scalar.npy"

    result = run_batch_export(
        BatchExportRequest(path, output, ranges=((0, 1, 2), (1, 2, 3)), squeeze=squeeze)
    )

    assert np.load(output).shape == shape
    assert np.load(output).item() == 6
    assert result.output_shape == shape


@pytest.mark.parametrize("enabled", [True, False])
def test_unsupported_ids_fail_before_registry_lookup_or_pixel_reads(
    tmp_path, monkeypatch, source_observations, enabled
):
    path = _input(tmp_path, np.arange(12).reshape(3, 4))
    recipe = tmp_path / "operations.json"
    recipe.write_text(
        json.dumps({"version": 2, "operations": [{"id": "user:command", "enabled": enabled}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        recipes, "create_operation", lambda *_args, **_kwargs: pytest.fail("unexpected lookup")
    )

    with pytest.raises(ValueError, match="built-in operations only"):
        run_batch_export(BatchExportRequest(path, tmp_path / "output.npy", recipe), dry_run=True)

    assert source_observations[1] == []
    assert not (tmp_path / "output.npy").exists()


@pytest.mark.parametrize(
    "ranges",
    [((3, 0, 1),), ((0, 0, 1), (0, 1, 2)), ((0, 1, 5),), ((0, 2, 1),), ((0, 0.0, 1),)],
)
def test_invalid_derived_ranges_fail_without_reads(tmp_path, source_observations, ranges):
    path = _input(tmp_path, np.arange(12).reshape(3, 4))

    with pytest.raises(ValueError, match="range"):
        run_batch_export(BatchExportRequest(path, tmp_path / "output.npy", ranges=ranges))

    assert source_observations[1] == []


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_input_alias_cannot_be_replaced_even_with_expensive_consent(
    tmp_path, source_observations, alias
):
    path = _input(tmp_path, np.arange(12).reshape(3, 4))
    original = path.read_bytes()
    output = path if alias == "same" else tmp_path / "alias.npy"
    if alias == "symlink":
        output.symlink_to(path)
    elif alias == "hardlink":
        output.hardlink_to(path)

    with pytest.raises(ValueError, match="choose a distinct output"):
        run_batch_export(BatchExportRequest(path, output, allow_expensive=True))

    assert path.read_bytes() == original
    assert source_observations == ([], [])


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_output_cannot_replace_an_alias_of_the_authored_recipe(
    tmp_path, source_observations, alias
):
    path = _input(tmp_path, np.arange(12).reshape(3, 4))
    recipe = _recipe(tmp_path, ())
    original = recipe.read_bytes()
    output = tmp_path / "alias.npz"
    if alias == "symlink":
        output.symlink_to(recipe)
    else:
        output.hardlink_to(recipe)

    with pytest.raises(ValueError, match="choose a distinct output"):
        run_batch_export(BatchExportRequest(path, output, recipe))

    assert recipe.read_bytes() == original
    assert source_observations == ([], [])


def test_cancellation_during_serialization_preserves_old_output_and_closes_source(
    tmp_path, source_observations
):
    path = _input(tmp_path, np.arange(100).reshape(10, 10))
    output = tmp_path / "output.npy"
    output.write_bytes(b"previous complete output")
    token = type("Token", (), {"cancelled": False})()

    def cancel(event):
        if event.phase == "writing" and event.completed_bytes:
            token.cancelled = True

    with pytest.raises(EvaluationCancelled):
        run_batch_export(
            BatchExportRequest(path, output), cancellation_token=token, progress_callback=cancel
        )

    assert output.read_bytes() == b"previous complete output"
    assert set(tmp_path.iterdir()) == {path, output}


def test_success_remains_success_when_cancel_arrives_after_publication(
    tmp_path, source_observations
):
    path = _input(tmp_path, np.arange(12).reshape(3, 4))
    output = tmp_path / "output.npy"
    token = type("Token", (), {"cancelled": False})()

    def late_cancel(event):
        if event.phase == "complete":
            token.cancelled = True

    result = run_batch_export(
        BatchExportRequest(path, output), cancellation_token=token, progress_callback=late_cancel
    )

    assert token.cancelled
    assert result.paths == (output,)
    np.testing.assert_array_equal(np.load(output), np.arange(12).reshape(3, 4))


def test_cost_gate_and_explicit_consent_share_the_same_plan(
    tmp_path, monkeypatch, source_observations
):
    from arrayscope.io import region_export

    path = _input(tmp_path, np.arange(12).reshape(3, 4))
    output = tmp_path / "output.npy"
    monkeypatch.setattr(region_export, "DEFAULT_VISIBLE_RENDER_BUDGET_BYTES", 1)

    preview = run_batch_export(BatchExportRequest(path, output), dry_run=True)
    assert preview.needs_confirmation
    assert source_observations[1] == []
    with pytest.raises(BatchExportNeedsConfirmation) as captured:
        run_batch_export(BatchExportRequest(path, output))
    assert captured.value.summary["output_bytes"] == 96
    assert source_observations[1] == []
    assert not output.exists()

    exported = run_batch_export(BatchExportRequest(path, output, allow_expensive=True))

    assert exported.needs_confirmation
    np.testing.assert_array_equal(np.load(output), np.arange(12).reshape(3, 4))


def test_expensive_consent_does_not_override_source_read_budget(
    tmp_path, monkeypatch, source_observations
):
    from arrayscope.operations import source_read

    path = _input(tmp_path, np.arange(12).reshape(3, 4))
    output = tmp_path / "output.npy"
    monkeypatch.setattr(source_read, "DEFAULT_SOURCE_READ_BUDGET_BYTES", 1)

    with pytest.raises(SourceReadRefused, match="read budget"):
        run_batch_export(BatchExportRequest(path, output, allow_expensive=True))

    assert source_observations[1] == []
    assert not output.exists()


@pytest.mark.parametrize("fifo_role", ["input", "recipe"])
def test_nonregular_inputs_are_rejected_without_opening_or_reading(
    tmp_path, monkeypatch, source_observations, fifo_role
):
    if not hasattr(os, "mkfifo"):
        pytest.skip("named pipes are unavailable on this platform")
    path = _input(tmp_path, np.arange(12).reshape(3, 4))
    recipe = tmp_path / "recipe.json"
    if fifo_role == "input":
        path.unlink()
        os.mkfifo(path)
    else:
        os.mkfifo(recipe)
    monkeypatch.setattr(
        recipe_replay.np,
        "load",
        lambda *_args, **_kwargs: pytest.fail("must refuse before opening an array"),
    )

    with pytest.raises(ValueError, match="regular"):
        run_batch_export(
            BatchExportRequest(
                path, tmp_path / "output.npy", recipe if fifo_role == "recipe" else None
            )
        )

    assert source_observations == ([], [])


def test_non_npy_payload_closes_the_opened_archive_before_rejecting_it(tmp_path, monkeypatch):
    path = tmp_path / "misnamed.npy"
    with path.open("wb") as stream:
        np.savez(stream, array=np.arange(12))
    opened = []
    original_load = np.load

    def load(*args, **kwargs):
        archive = original_load(*args, **kwargs)
        opened.append(archive.zip)
        return archive

    monkeypatch.setattr(recipe_replay.np, "load", load)

    with pytest.raises(ValueError, match="memory-mappable NumPy array"):
        run_batch_export(BatchExportRequest(path, tmp_path / "output.npy"))

    assert len(opened) == 1
    assert opened[0].fp is None
