"""Onscreen pixel + responsiveness assertions for the tile pipeline (ADR 0051).

Every test drives the production window on real hardware.  "Tile shows wrong
content" and "event loop hangs" fail HERE, not in someone's eyes.
See conftest for the opt-in invocation.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu_interaction

# Thomas's bar (gui-responsiveness memory): any >50 ms synchronous step in the
# GUI loop during interaction is a bug to profile.  The heartbeat samples every
# loop iteration, so the max gap IS the longest synchronous step.
MAX_INTERACTION_GAP_MS = 50.0


@pytest.mark.parametrize("backend", ("pyqtgraph", "vispy"))
def test_one_index_boundary_scroll_has_pixels_and_trace_clean(
    backend, tmp_path
):
    """V1: a tile touching the viewport boundary remains a render obligation."""

    import numpy as np

    from arrayscope.app.qt_binding import prefer_pyside6
    from arrayscope.core.trace import close_trace, configure_trace
    from arrayscope.tools.trace_verify import verify_trace

    prefer_pyside6()
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore

    from arrayscope.app.launch import _create_window
    from arrayscope.display.backend_contract import image_view_backend_capabilities
    from tests.gpu_interaction.conftest import Harness, TILE

    trace_path = tmp_path / f"v1-boundary-{backend}.trace.jsonl"
    app = pg.mkQApp()
    app.setOrganizationName("ArrayScope")
    app.setApplicationName("ArrayScope")
    settings = QtCore.QSettings()
    previous_backend = settings.value("image_rendering_backend")
    settings.setValue("image_rendering_backend", backend)
    settings.setValue("montage_quality_policy", "resident")
    settings.sync()
    configure_trace(trace_path)

    count = 37
    frames = np.repeat(np.arange(count, dtype=np.float32), TILE * TILE)
    data = frames.reshape(count, TILE, TILE).transpose(1, 2, 0).copy()
    app, win = _create_window(data, title=f"gpu-harness-v1-{backend}")
    try:
        if image_view_backend_capabilities(win.img_view).name != backend:
            pytest.skip(f"{backend} backend unavailable in this Qt environment")
        h = Harness(app, win)
        initial = tuple(range(36))
        shifted = tuple(range(1, 37))
        win._set_view_state(
            win.view_state.with_montage_axis(
                2, columns=6, indices=initial, text="0:36"
            )
        )
        win.render(reason="gpu-harness-v1-initial")
        assert h.wait_settled(timeout=20.0)
        win.img_view.setLevels(0.0, 36.0)
        assert h.wait_settled(timeout=20.0)

        tiles = h.session.plan.tiles
        boundary_tile = tiles[5]
        height, width = h.session.plan.display_shape
        view = win.img_view.getView()
        view.setRange(
            xRange=(0.0, float(boundary_tile.x0)),
            yRange=(0.0, float(height)),
            padding=0,
        )
        assert h.wait_settled(timeout=20.0)
        assert int(boundary_tile.montage_index) in h.session.required_tile_numbers()

        win._set_view_state(
            win.view_state.with_montage_axis(
                2, columns=6, indices=shifted, text="1:37"
            )
        )
        win.render(reason="gpu-harness-v1-one-index-boundary")
        assert h.wait_settled(timeout=20.0)
        assert h.session.required_target_settled()
        assert not h.session.required_target_unsettled_tiles()
        assert not (
            h.session.lifecycle.parked_tiles
            & set(h.session.required_tile_numbers())
        )
        boundary_row = h.session.lifecycle.row(int(boundary_tile.montage_index))
        assert boundary_row.target_settled

        # Revealing the boundary column must require no new materialization:
        # its exact pixels were already an obligation at the landing edge.
        view.setRange(
            xRange=(0.0, float(width)),
            yRange=(0.0, float(height)),
            padding=0,
        )
        app.processEvents()
        means = h.tile_means()
        expected = [255.0 * value / 36.0 for value in shifted]
        assert all(means[index] > means[index - 1] + 1.0 for index in range(1, 36)), means
        assert max(abs(actual - wanted) for actual, wanted in zip(means, expected)) <= 12.0, means
        h.assert_lifecycle_settled()
    finally:
        win.close()
        for _ in range(50):
            app.processEvents()
        close_trace()
        if previous_backend is None:
            settings.remove("image_rendering_backend")
        else:
            settings.setValue("image_rendering_backend", previous_backend)
        settings.sync()

    verification = verify_trace(trace_path)
    assert verification["ok"], verification
    assert verification["required_targets"] == 36


@pytest.mark.parametrize("backend", ("pyqtgraph", "vispy"))
def test_cold_scroll_records_center_out_acknowledgements(backend, tmp_path):
    """V2: final cold-scroll targets paint from viewport focus outward."""

    import numpy as np

    from arrayscope.app.qt_binding import prefer_pyside6
    from arrayscope.core.trace import close_trace, configure_trace
    from arrayscope.tools.trace_verify import verify_trace

    prefer_pyside6()
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore

    from arrayscope.app.launch import _create_window
    from arrayscope.display.backend_contract import image_view_backend_capabilities
    from arrayscope.display.model.tile_priority import prioritize_tiles
    from tests.gpu_interaction.conftest import Harness, TILE

    trace_path = tmp_path / f"v2-center-out-{backend}.trace.jsonl"
    app = pg.mkQApp()
    app.setOrganizationName("ArrayScope")
    app.setApplicationName("ArrayScope")
    settings = QtCore.QSettings()
    previous_backend = settings.value("image_rendering_backend")
    settings.setValue("image_rendering_backend", backend)
    settings.setValue("montage_quality_policy", "resident")
    settings.sync()
    configure_trace(trace_path)

    count = 72
    frames = np.repeat(np.arange(count, dtype=np.float32), TILE * TILE)
    data = frames.reshape(count, TILE, TILE).transpose(1, 2, 0).copy()
    app, win = _create_window(data, title=f"gpu-harness-v2-{backend}")
    expected_order = ()
    try:
        if image_view_backend_capabilities(win.img_view).name != backend:
            pytest.skip(f"{backend} backend unavailable in this Qt environment")
        h = Harness(app, win)
        win._set_view_state(
            win.view_state.with_montage_axis(
                2, columns=6, indices=tuple(range(36)), text="0:36"
            )
        )
        win.render(reason="gpu-harness-v2-initial")
        assert h.wait_settled(timeout=20.0)
        win.img_view.setLevels(0.0, 71.0)
        h.fit_view()
        assert h.wait_settled(timeout=20.0)

        for start in (36,):
            win._set_view_state(
                win.view_state.with_montage_axis(
                    2,
                    columns=6,
                    indices=tuple(range(start, start + 36)),
                    text=f"{start}:{start + 36}",
                )
            )
            win.render(reason="gpu-harness-v2-fast-scroll")
            app.processEvents()

        assert h.wait_settled(timeout=20.0)
        assert h.session.required_target_settled()
        expected_order = tuple(
            int(tile.montage_index)
            for tile in prioritize_tiles(
                h.session.plan.tiles,
                context=h.session.tile_priority_context(),
            )
        )
        means = h.tile_means()
        expected_values = [
            means[0] + (means[-1] - means[0]) * offset / 35.0
            for offset in range(36)
        ]
        assert all(
            means[index] >= means[index - 1] - 1.0
            for index in range(1, 36)
        ), means
        assert means[-1] - means[0] >= 100.0, means
        assert max(
            abs(actual - wanted)
            for actual, wanted in zip(means, expected_values)
        ) <= 12.0, means
        h.assert_lifecycle_settled()
    finally:
        win.close()
        for _ in range(50):
            app.processEvents()
        close_trace()
        if previous_backend is None:
            settings.remove("image_rendering_backend")
        else:
            settings.setValue("image_rendering_backend", previous_backend)
        settings.sync()

    verification = verify_trace(trace_path)
    assert verification["ok"], verification
    assert verification["required_targets"] == 36
    # Worker completion may permute tiles inside the backend's bounded first
    # exact commit; that first visible band must still be the nearest band.
    actual_order = tuple(verification["acknowledgement_order"])
    first_commit_size = 16 if backend == "vispy" else 8
    expected_first = set(expected_order[:first_commit_size])
    actual_first = set(actual_order[:first_commit_size])
    assert len(actual_first & expected_first) >= first_commit_size - 2, (
        actual_order,
        expected_order,
    )


def test_montage_presents_every_tile_with_its_own_content(montage_window):
    h = montage_window
    h.fit_view()
    h.assert_tile_identity_ramp()
    h.assert_lifecycle_settled()


def test_view_fits_montage_when_enabled_after_settle():
    """Reproduces the field report 'sessions often open with wrongly scaled
    items': the fast path (montage enabled right after open, as in the shared
    fixture) fits correctly, but enabling it after the single-frame view has
    fully settled must still fit the montage content."""

    from tests.gpu_interaction.conftest import Harness, synthetic_montage_data
    from arrayscope.app.qt_binding import prefer_pyside6

    prefer_pyside6()
    from arrayscope.app.launch import _create_window

    app, win = _create_window(
        synthetic_montage_data(), title="gpu-harness-late-montage"
    )
    try:
        h = Harness(app, win)
        h.pump(3.0)  # let the single-frame view settle completely
        win._set_view_state(win.view_state.with_montage_axis(2, text=":"))
        win.render(reason="gpu-harness-late-montage")
        assert h.wait_settled(timeout=20.0)
        (x0, x1), (y0, y1) = h.session.view_range
        height, width = h.session.plan.display_shape
        span_x, span_y = x1 - x0, y1 - y0
        assert span_x <= width * 1.5 and span_y <= height * 1.5, (
            f"fit shows {span_x:.0f}x{span_y:.0f} data units for a "
            f"{width}x{height} montage (wrongly scaled on montage enable)"
        )
    finally:
        win.close()
        for _ in range(50):
            app.processEvents()


def test_pan_keeps_event_loop_responsive_and_content_correct(montage_window):
    h = montage_window
    h.fit_view()
    view = h.win.img_view.getView()
    state = {"n": 0}

    def pan_step():
        (x0, x1), (y0, y1) = h.session.view_range
        dy = (y1 - y0) * (0.15 if state["n"] % 2 == 0 else -0.15)
        state["n"] += 1
        view.setRange(xRange=(x0, x1), yRange=(y0 + dy, y1 + dy), padding=0)

    gaps = h.heartbeat_gaps(3.0, step=pan_step, step_interval=0.1)
    worst = max(gaps)
    assert worst <= MAX_INTERACTION_GAP_MS, (
        f"event loop hung {worst:.0f} ms during pan (bar: {MAX_INTERACTION_GAP_MS} ms)"
    )
    # Back to rest: content must still be each tile's own.
    h.fit_view()
    assert h.wait_settled()
    h.assert_tile_identity_ramp()
    h.assert_lifecycle_settled()


def test_zoom_across_lod_threshold_keeps_content_and_levels_in_sync(montage_window):
    h = montage_window
    h.fit_view()
    assert h.wait_settled()
    view = h.win.img_view.getView()
    height, width = h.session.plan.display_shape

    ranges = (
        ((0.0, float(width)), (0.0, float(height))),
        ((0.0, float(width) * 0.45), (0.0, float(height) * 0.45)),
        ((float(width) * 0.25, float(width) * 0.75), (float(height) * 0.25, float(height) * 0.75)),
        ((0.0, float(width)), (0.0, float(height))),
    )
    for x_range, y_range in ranges:
        view.setRange(xRange=x_range, yRange=y_range, padding=0)
        assert h.wait_settled(timeout=20.0), f"never settled after zoom range {x_range}/{y_range}"
        h.assert_lifecycle_settled()

    h.fit_view()
    assert h.wait_settled()
    h.assert_tile_identity_ramp()
    h.assert_lifecycle_settled()


def test_index_scrub_and_return_shows_no_stale_content_or_wedged_claims(montage_window):
    """The scrub-back defect class: session replacement leaked singleflight
    claims, so returning to a previous index window presented wedged/stale
    LOD.  Scrub away and back; every tile must show its own content and the
    lifecycle machine must audit clean."""

    h = montage_window
    h.fit_view()
    axis = 2

    def select(text: str) -> None:
        h.win._set_view_state(h.win.view_state.with_montage_axis(axis, text=text))
        h.win.render(reason="gpu-harness-scrub")

    for text in ("9:27", "18:36", "0:18", ":"):
        select(text)
        assert h.wait_settled(timeout=20.0), f"never settled after scrub to {text!r}"

    h.fit_view()
    h.assert_tile_identity_ramp()
    h.assert_lifecycle_settled()
    pyramid = h.session.pyramid_cache
    if pyramid is not None:
        assert int(getattr(pyramid, "pending_count", 0) or 0) == 0, (
            "pyramid singleflight claims wedged after scrub-back"
        )


def test_fft_preview_refinement_settles_without_stalls(montage_window):
    """Exercise the transform-preview path that reproduced the 73 s floor wedge.

    The raw identity-ramp assertions above cover per-tile pixel ownership.
    This path instead proves that an FFT/shift/iFFT montage over the montage
    axis reaches idle, survives a scrub, and leaves the lifecycle machine clean.
    """

    from arrayscope.operations.pipeline import CenteredFFT, CenteredIFFT, FFTShift
    from arrayscope.tools.profile_montage_workflow import _profile_transform_operations

    h = montage_window
    axis = 2
    h.win.operation_coordinator.load_operations(
        _profile_transform_operations(
            axis,
            centered_fft=CenteredFFT,
            fftshift=FFTShift,
            centered_ifft=CenteredIFFT,
        )
    )
    h.win._set_document(h.win.operation_coordinator.document)
    h.win._coerce_channel_for_current_dtype()

    def select(text: str) -> None:
        h.win._set_view_state(h.win.view_state.with_montage_axis(axis, text=text))
        h.win.render(reason="gpu-harness-fft-preview")

    for text in (":", "6:24", ":"):
        select(text)
        assert h.wait_settled(timeout=30.0), (
            f"FFT preview/refinement montage never settled after scrub to {text!r}"
        )
        h.assert_lifecycle_settled()


def test_idle_stays_settled_after_interaction(montage_window):
    """The idle-loop defect class: parked upserts re-emitted every commit kept
    the app at ~120 commits+draws/s at idle.  After interaction settles, the
    machine must report no re-armable work and the dirty queues must drain."""

    h = montage_window
    h.fit_view()
    assert h.wait_settled()
    h.pump(1.0)
    s = h.session
    assert not s.dirty_payloads, (
        f"idle dirty queue never drains: {sorted(s.dirty_payloads)}"
    )
    assert not s.pending_payload_upserts, (
        f"idle upsert queue never drains: {sorted(s.pending_payload_upserts)}"
    )
    h.assert_lifecycle_settled()
