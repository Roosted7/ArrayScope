"""Targeted wgpu live-view tests: montage acks + complex/RGB commit modes.

Offscreen, adapter-skip pattern (mirrors ``test_imagesurface_contract``):
these pin the queue row 3(b) montage/complex slice — physical-truth per-tile
acknowledgement, content-keyed zero-upload behavior across montage scrolls
and mode/levels switches, the phase LUT, RGB display-ready bytes, and the
loud rejections at the honest scope boundary.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS

from tests.display.test_imageview2d import _present_tiled, _view_class


def _wgpu_adapter_available() -> bool:
    try:
        import wgpu
        from wgpu.backends.wgpu_native.extras import set_instance_extras

        try:
            # Vulkan-only instance BEFORE the first adapter request: letting
            # the probe create an all-backends instance makes GL adapter
            # enumeration re-init EGL, which SIGABRTs in workers that hold
            # live vispy GL state (gate-B Tier 0; full-suite crash 2026-07-18).
            set_instance_extras(backends=["Vulkan"])
        except RuntimeError:
            pass  # instance already exists
        wgpu.gpu.request_adapter_sync(power_preference="low-power")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _wgpu_adapter_available(), reason="no wgpu adapter on this machine"
)


def test_wgpu_camera_draw_publishes_presentation_ack(qt_app, monkeypatch):
    view = _view_class("wgpu")()
    requests = []
    acknowledgements = []
    try:
        monkeypatch.setattr(
            view._wgpu_canvas,
            "request_draw",
            lambda *args, **kwargs: requests.append((args, kwargs)),
        )
        view.presentationDrawn.connect(lambda: acknowledgements.append("drawn"))

        view._request_wgpu_canvas_draw()

        assert requests == [((), {})]
        assert view.presentationDrawPending() is True
        view._wgpu_canvas_update_pending = False
        view._publish_wgpu_draw_ack(0)
        assert acknowledgements == ["drawn"]
        assert view.presentationDrawPending() is False
    finally:
        view.close()


def test_wgpu_pool_headroom_clamps_to_device_limit_but_active_pages_do_not():
    from arrayscope.display.wgpu_imageview2d import _wgpu_pool_layer_budget

    assert (
        _wgpu_pool_layer_budget(
            previous=0,
            needed=272,
            preferred=2084,
            max_layers=2048,
        )
        == 2048
    )
    with pytest.raises(RuntimeError, match=r"needed=2049, max_layers=2048"):
        _wgpu_pool_layer_budget(previous=0, needed=2049, max_layers=2048)


def _montage_geometry(tile_shape, columns, rows, *, loaded, gap=0):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.montage import MontageTileState

    tile_h, tile_w = tile_shape
    height = rows * tile_h + max(0, rows - 1) * gap
    width = columns * tile_w + max(0, columns - 1) * gap
    return DisplayGeometry(
        view_state=ViewState.from_shape((height, width)).with_image_axes(0, 1),
        display_shape=(height, width),
        montage=MontageGeometry(
            indices=tuple(range(loaded)),
            tile_shape=(tile_h, tile_w),
            columns=columns,
            rows=rows,
            gap=gap,
        ),
        montage_tile_states=tuple([MontageTileState.LOADED] * loaded),
    )


def _payload(
    tile_number,
    image,
    *,
    source_id,
    shader_mapping=None,
    texture_kind=None,
    histogram_data=None,
):
    from arrayscope.display.model.frame import DisplayTilePayload

    return DisplayTilePayload(
        tile_number,
        tile_number,
        image,
        histogram_data,
        source_id,
        shader_mapping=shader_mapping,
        texture_kind=texture_kind,
    )


def _lod_payload(
    tile_number,
    image,
    *,
    base_source_id,
    level,
    source_shape,
    payload_source_shape=None,
    factor=None,
):
    from arrayscope.display.lod import LodInfo
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.display.model.tile_identity import TileIdentity, TileLodIdentity
    from arrayscope.display.shader_mapping import TexturePlaneKind

    factor = 1 << int(level) if factor is None else int(factor)
    lod = LodInfo(
        level=level,
        factor=factor,
        source_shape=source_shape,
        texture_shape=image.shape[:2],
    )
    identity = TileIdentity(
        document_generation="lod-doc",
        operation_key="lod-op",
        source_index=tile_number,
        image_axes=(0, 1),
        axis_flips=(False, False),
        channel="real",
        complex_mapping=None,
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        semantic_generation="lod-semantic",
        lod=TileLodIdentity(level=level, factor=factor),
    )
    return DisplayTilePayload(
        tile_number,
        tile_number,
        image,
        None,
        (
            base_source_id,
            "texture_kind",
            TexturePlaneKind.SCALAR_R32F.value,
            "lod",
            factor,
            level,
            0,
        ),
        source_shape=payload_source_shape or source_shape,
        lod=lod,
        tile_identity=identity,
    )


def _shown_view(qt_app):
    view = _view_class("wgpu")()
    view.resize(320, 260)
    view.show()
    return view


def _commit(view, geometry, payloads, *, levels, rgb_already_windowed=False):
    canvas = np.zeros(geometry.display_shape, dtype=np.float32)
    return _present_tiled(
        view,
        canvas,
        geometry=geometry,
        levels=levels,
        histogramRange=levels,
        montage_tile_payloads=payloads,
        rgb_already_windowed=rgb_already_windowed,
    )


def _rerender_internal(view):
    """Re-present the committed tiles to the executor's offscreen target."""

    from arrayscope.gpu.command_protocol import UpdateTileInstances

    camera = view._wgpu_camera_command()
    view._submit_wgpu(
        (camera, UpdateTileInstances(view._wgpu_camera_tiles(camera)))
    )


def _center_pixel(view):
    target = view._wgpu_executor.read_target()
    h, w = target.shape[:2]
    return target[h // 2, w // 2]


def _green_overlay_mask(target):
    pixels = np.asarray(target, dtype=np.int16)
    return (
        (pixels[..., 1] > 150)
        & (pixels[..., 1] > pixels[..., 0] + 45)
        & (pixels[..., 1] > pixels[..., 2] + 45)
    )


def _orange_overlay_mask(target):
    pixels = np.asarray(target, dtype=np.int16)
    return (
        (pixels[..., 0] > 150)
        & (pixels[..., 0] > pixels[..., 1] + 45)
        & (pixels[..., 2] < 120)
    )


def _mask_center(mask):
    rows, columns = np.nonzero(mask)
    assert len(rows), "expected physical pixels for this mask"
    return (float(columns.mean()), float(rows.mean()))


def test_roi_and_profile_marker_are_executor_pixels_and_clear(qt_app, qtbot):
    """Thomas's 2026-07-18 dogfood report: both overlays were invisible.

    The oracle reads the executor target, not the QWidget backing store, so a
    QGraphics mirror cannot satisfy it (nor can a bookkeeping-only hook).
    """

    from arrayscope.core.roi import RoiKind
    from arrayscope.display.interaction import InteractionTarget

    view = _shown_view(qt_app)
    try:
        image = np.full((64, 64), 0.02, dtype=np.float32)
        geometry = _montage_geometry(image.shape, 1, 1, loaded=1)
        _commit(
            view,
            geometry,
            {0: _payload(0, image, source_id="overlay-pixel-oracle")},
            levels=(0.0, 1.0),
        )
        view.getView().setRange(xRange=(0, 64), yRange=(0, 64), padding=0)
        roi = view.createRoi(
            RoiKind.RECTANGLE,
            rect=(10.0, 12.0, 20.0, 18.0),
            color=(40, 220, 80),
        )
        view.setProfileMarker(46.0, 42.0, visible=True)
        draws_before = int(view._wgpu_draw_count)
        view._wgpu_canvas_update_pending = False
        view._request_wgpu_canvas_draw()
        qtbot.waitUntil(
            lambda: int(view._wgpu_draw_count) > draws_before,
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        assert view._wgpu_last_draw_error == ""
        _rerender_internal(view)

        target = view._wgpu_executor.read_target()
        base_green = np.count_nonzero(_green_overlay_mask(target))
        base_orange = np.count_nonzero(_orange_overlay_mask(target))
        assert base_green > 100
        assert base_orange > 100

        state = view.interaction_controller.set_hover(
            InteractionTarget(
                "roi",
                object_id=roi.id,
                part="handle",
                geometry_kind="rectangle",
                handle_index=0,
            ),
            point=(10.0, 12.0),
        )
        view.sync_interaction_state(state)
        _rerender_internal(view)
        assert np.count_nonzero(
            _green_overlay_mask(view._wgpu_executor.read_target())
        ) > base_green
        view.highlightRoi(roi.id)
        _rerender_internal(view)

        state = view.interaction_controller.set_hover(
            InteractionTarget("profile", part="center"),
            point=(46.0, 42.0),
        )
        view.sync_interaction_state(state)
        _rerender_internal(view)
        assert np.count_nonzero(
            _orange_overlay_mask(view._wgpu_executor.read_target())
        ) > base_orange

        overlay_writes = view._wgpu_executor.overlay_buffer_writes_total
        view.getView().setRange(xRange=(64, 128), yRange=(0, 64), padding=0)
        _rerender_internal(view)
        assert not np.any(
            _orange_overlay_mask(view._wgpu_executor.read_target())
        )
        assert view._wgpu_executor.overlay_buffer_writes_total == overlay_writes
        view.getView().setRange(xRange=(0, 64), yRange=(0, 64), padding=0)
        _rerender_internal(view)
        assert np.count_nonzero(
            _orange_overlay_mask(view._wgpu_executor.read_target())
        ) > 100
        assert view._wgpu_executor.overlay_buffer_writes_total == overlay_writes

        assert view.removeRoi(roi.id)
        _rerender_internal(view)
        target = view._wgpu_executor.read_target()
        assert not np.any(_green_overlay_mask(target))
        assert np.count_nonzero(_orange_overlay_mask(target)) > 100

        view.hideProfileMarker()
        _rerender_internal(view)
        target = view._wgpu_executor.read_target()
        assert not np.any(_orange_overlay_mask(target))
    finally:
        view.close()


def test_world_overlay_and_tile_move_together_without_overlay_reupload(qt_app):
    """A camera-only frame must rigidly move tiles and world overlays."""

    from arrayscope.core.roi import RoiKind

    view = _shown_view(qt_app)
    try:
        image = np.zeros((64, 64), dtype=np.float32)
        image[22:34, 24:38] = 1.0
        geometry = _montage_geometry(image.shape, 1, 1, loaded=1)
        _commit(
            view,
            geometry,
            {0: _payload(0, image, source_id="overlay-camera-oracle")},
            levels=(0.0, 1.0),
        )
        view.getView().setRange(xRange=(0, 64), yRange=(0, 64), padding=0)
        view.createRoi(
            RoiKind.RECTANGLE,
            rect=(24.0, 22.0, 14.0, 12.0),
            color=(40, 220, 80),
        )
        _rerender_internal(view)
        before = view._wgpu_executor.read_target()
        tile_before = _mask_center(np.all(before[..., :3] > 180, axis=-1))
        overlay_before = _mask_center(_green_overlay_mask(before))
        overlay_writes = view._wgpu_executor.overlay_buffer_writes_total

        view.getView().setRange(xRange=(4, 68), yRange=(0, 64), padding=0)
        _rerender_internal(view)
        after = view._wgpu_executor.read_target()
        tile_after = _mask_center(np.all(after[..., :3] > 180, axis=-1))
        overlay_after = _mask_center(_green_overlay_mask(after))

        tile_shift = (tile_after[0] - tile_before[0], tile_after[1] - tile_before[1])
        overlay_shift = (
            overlay_after[0] - overlay_before[0],
            overlay_after[1] - overlay_before[1],
        )
        assert tile_shift[0] < -20.0
        assert tile_shift == pytest.approx(overlay_shift, abs=2.0)
        assert view._wgpu_executor.overlay_buffer_writes_total == overlay_writes

        view.getView().setRange(xRange=(0, 64), yRange=(0, 64), padding=0)
        view.getView().invertX(True)
        view.getView().invertY(False)
        _rerender_internal(view)
        mirrored = view._wgpu_executor.read_target()
        tile_mirrored = _mask_center(np.all(mirrored[..., :3] > 180, axis=-1))
        overlay_mirrored = _mask_center(_green_overlay_mask(mirrored))
        target_h, target_w = mirrored.shape[:2]
        assert tile_mirrored == pytest.approx(
            (target_w - 1 - tile_before[0], target_h - 1 - tile_before[1]),
            abs=2.0,
        )
        assert overlay_mirrored == pytest.approx(
            (target_w - 1 - overlay_before[0], target_h - 1 - overlay_before[1]),
            abs=2.0,
        )
        assert view._wgpu_executor.overlay_buffer_writes_total == overlay_writes
    finally:
        view.close()


def test_loading_and_skipped_tile_geometry_is_in_executor_target(qt_app):
    from arrayscope.display.overlays import MontageTileOverlay

    view = _shown_view(qt_app)
    try:
        image = np.full((32, 64), 0.02, dtype=np.float32)
        geometry = _montage_geometry((32, 32), 2, 1, loaded=2)
        _commit(
            view,
            geometry,
            {
                0: _payload(0, image[:, :32], source_id="loading-tile"),
                1: _payload(1, image[:, 32:], source_id="skipped-tile"),
            },
            levels=(0.0, 1.0),
        )
        view.getView().setRange(xRange=(0, 64), yRange=(0, 32), padding=0)
        view.setMontageTileOverlays(
            (
                MontageTileOverlay(0, 0, 32, 32, "loading", "not rendered"),
                MontageTileOverlay(32, 0, 32, 32, "skipped", "skipped"),
            )
        )
        _rerender_internal(view)
        target = view._wgpu_executor.read_target()
        bright = np.all(target[..., :3] > 150, axis=-1)
        midpoint = target.shape[1] // 2
        assert np.count_nonzero(bright[:, :midpoint]) > 20
        assert np.count_nonzero(bright[:, midpoint:]) > 20

        view.clearMontageTileOverlays()
        _rerender_internal(view)
        assert not np.any(
            np.all(view._wgpu_executor.read_target()[..., :3] > 150, axis=-1)
        )
    finally:
        view.close()


def _truth_text_mask(target):
    # Truth-label glyph ink is #a5f3fc (165, 243, 252); the label border
    # (#22d3ee, red 34) and tile pixels fail the red-channel bound.
    pixels = np.asarray(target, dtype=np.int16)
    return (pixels[..., 0] > 120) & (pixels[..., 1] > 180) & (pixels[..., 2] > 200)


def _truth_border_mask(target):
    # The DRAW-state label border is #22d3ee (34, 211, 238) at full alpha.
    pixels = np.asarray(target, dtype=np.int16)
    return (
        (np.abs(pixels[..., 0] - 34) < 30)
        & (pixels[..., 1] > 180)
        & (pixels[..., 2] > 200)
    )


def _label_anchor_px(view, target_shape, world_point):
    """Expected on-target pixel of a world-space label anchor."""

    camera = view._wgpu_camera_command()
    x0, y0, x1, y1 = camera.world_rect
    height, width = target_shape[:2]
    wx, wy = world_point
    px = (x1 - wx if camera.x_inverted else wx - x0) / (x1 - x0) * width
    py = (y1 - wy if not camera.y_inverted else wy - y0) / (y1 - y0) * height
    return px, py


def test_tile_truth_labels_are_native_glyph_pixels_pan_with_camera_and_clear(qt_app):
    """Queue row 3 text gap: truth labels are executor pixels, not QLabels.

    Offscreen GPU ring.  Red-first oracles: glyph pixels render at the
    tile's on-screen corner (derived from the shared camera command, the
    same transform that places tiles) and vanish on removal; a camera-only
    pan moves the text WITH the image with zero atlas uploads and zero
    overlay buffer rewrites; zooming out far enough hides unreadable labels.
    """

    view = _shown_view(qt_app)
    try:
        image = np.zeros((64, 64), dtype=np.float32)
        geometry = _montage_geometry(image.shape, 1, 1, loaded=1)
        _commit(
            view,
            geometry,
            {0: _payload(0, image, source_id="truth-label-oracle")},
            levels=(0.0, 1.0),
        )
        view.getView().setRange(xRange=(0, 64), yRange=(0, 64), padding=0)
        view.setTileTruthOverlayRows(
            ({"tile": 0, "drawable": True, "tile_rect": (0.0, 0.0, 64.0, 64.0)},)
        )
        _rerender_internal(view)
        target = view._wgpu_executor.read_target()
        assert np.count_nonzero(_truth_text_mask(target)) > 80, (
            "a truth label is many glyph pixels"
        )
        border_rows, border_columns = np.nonzero(_truth_border_mask(target))
        assert len(border_rows), "the label draws its state border"
        anchor_x, anchor_y = _label_anchor_px(view, target.shape, (0.0, 0.0))
        # Label box top-left sits at the anchor plus the 2 px inset.
        assert float(border_columns.min()) == pytest.approx(anchor_x + 2.0, abs=2.0)
        assert float(border_rows.min()) == pytest.approx(anchor_y + 2.0, abs=2.0)
        assert view.tileTruthOverlayText().startswith("slot 0  DRAW")

        atlas_uploads = view._wgpu_executor.glyph_atlas_uploads_total
        overlay_writes = view._wgpu_executor.overlay_buffer_writes_total

        view.getView().setRange(xRange=(-16, 48), yRange=(0, 64), padding=0)
        _rerender_internal(view)
        after = view._wgpu_executor.read_target()
        assert np.count_nonzero(_truth_text_mask(after)) > 80
        after_rows, after_columns = np.nonzero(_truth_border_mask(after))
        anchor_x, anchor_y = _label_anchor_px(view, after.shape, (0.0, 0.0))
        assert float(after_columns.min()) == pytest.approx(anchor_x + 2.0, abs=2.0)
        assert float(after_rows.min()) == pytest.approx(anchor_y + 2.0, abs=2.0)
        assert view._wgpu_executor.glyph_atlas_uploads_total == atlas_uploads, (
            "camera-only frames must never re-upload cached glyphs"
        )
        assert view._wgpu_executor.overlay_buffer_writes_total == overlay_writes, (
            "world-anchored text must ride the camera uniform, not a rewrite"
        )

        # Unreadably small tiles hide their labels (QLabel-layer parity).
        view.getView().setRange(xRange=(0, 6400), yRange=(0, 6400), padding=0)
        _rerender_internal(view)
        assert not np.any(_truth_text_mask(view._wgpu_executor.read_target()))
        assert view.tileTruthOverlayText() == ""

        view.getView().setRange(xRange=(0, 64), yRange=(0, 64), padding=0)
        view.setTileTruthOverlayRows(())
        _rerender_internal(view)
        assert not np.any(_truth_text_mask(view._wgpu_executor.read_target()))
        assert view.tileTruthOverlayText() == ""
    finally:
        view.close()


def test_montage_commit_acks_per_tile_and_scrolls_zero_upload(qt_app):
    view = _shown_view(qt_app)
    try:
        rng = np.random.default_rng(11)
        images = {
            name: rng.random((20, 30), dtype=np.float32) for name in ("p0", "p1", "p2")
        }
        geometry = _montage_geometry((20, 30), 2, 1, loaded=2)

        payloads = {
            0: _payload(0, images["p0"], source_id=("wgpu-montage", "p0")),
            1: _payload(1, images["p1"], source_id=("wgpu-montage", "p1")),
        }
        report = _commit(view, geometry, payloads, levels=(0.0, 1.0))
        assert set(report.presented_tiles) == {0, 1}
        assert report.presented_identities == {
            0: ("wgpu-montage", "p0"),
            1: ("wgpu-montage", "p1"),
        }
        assert report.texture_uploads == 2  # one 256^2 page per tile
        # Identical re-commit: content-keyed residency makes it physical no-op.
        report = _commit(view, geometry, payloads, levels=(0.0, 1.0))
        assert set(report.presented_tiles) == {0, 1}
        assert report.texture_uploads == 0
        initial_pixels = view._wgpu_executor.read_target()

        # Montage scroll: p1 moves to tile 0, new content p2 enters tile 1.
        # Only the genuinely new plane uploads; p1 stays warm across rebind.
        scrolled = {
            0: _payload(0, images["p1"], source_id=("wgpu-montage", "p1")),
            1: _payload(1, images["p2"], source_id=("wgpu-montage", "p2")),
        }
        report = _commit(view, geometry, scrolled, levels=(0.0, 1.0))
        assert set(report.presented_tiles) == {0, 1}
        assert report.texture_uploads == 1
        scrolled_pixels = view._wgpu_executor.read_target()
        assert not np.array_equal(scrolled_pixels, initial_pixels)

        # Scroll back: every plane was seen before — zero upload.
        report = _commit(view, geometry, payloads, levels=(0.0, 1.0))
        assert set(report.presented_tiles) == {0, 1}
        assert report.texture_uploads == 0
        assert np.array_equal(view._wgpu_executor.read_target(), initial_pixels)
    finally:
        view.close()


def test_phase1_exposes_fenced_resident_page_histogram(qt_app):
    view = _shown_view(qt_app)
    try:
        image = np.linspace(-2.0, 5.0, 20 * 30, dtype=np.float32).reshape(20, 30)
        geometry = _montage_geometry((20, 30), 1, 1, loaded=1)
        payloads = {0: _payload(0, image, source_id=("g6a", 0))}
        view.setResidentHistogramEvidenceRequired(True)

        report = _commit(view, geometry, payloads, levels=(-2.0, 5.0))

        assert report.presented_tiles == frozenset({0})
        (evidence,) = view.residentHistogramEvidence(payloads)
        evidence.wait_completed()
        counts, bounds = evidence.readback.resolve()
        assert bounds == pytest.approx((-2.0, 5.0))
        assert int(counts.sum()) == image.size
        assert evidence.frontier_keys == tuple(
            view._wgpu_committed["tiles"][0]["page_keys"]
        )

        view.acceptResidentHistogramEvidence((evidence.evidence_key,))
        report = _commit(view, geometry, payloads, levels=(-2.0, 5.0))
        assert set(report.presented_tiles) == {0}
        assert report.texture_uploads == 0

        view.setResidentHistogramEvidenceRequired(False)
        view.setResidentHistogramEvidenceRequired(True, ("next-coverage", 2))
        report = _commit(view, geometry, payloads, levels=(-2.0, 5.0))
        assert report.presented_tiles == frozenset({0})
        assert report.texture_uploads == 0
        (next_evidence,) = view.residentHistogramEvidence(payloads)
        assert next_evidence.evidence_key != evidence.evidence_key
    finally:
        view.close()


def test_histogram_frontier_evicted_in_same_submission_never_aborts_commit(
    qt_app, monkeypatch
):
    """Dogfood crash 2026-07-19: pool pressure inside one submission evicted a
    snapshotted histogram frontier page; the executor's loud KeyError then
    killed the whole commit mid-batch (ensures applied, present never ran).
    The commit must instead complete, drop that evidence spec with a loud
    bail trace, and let the normal re-queue machinery retry the evidence."""

    from arrayscope.display.wgpu_imageview2d import _shared_wgpu_device
    from arrayscope.gpu.wgpu_executor import WgpuPlaneExecutor

    view = _shown_view(qt_app)
    try:
        # One physical complex layer for two planes with evidence required:
        # tile 1's upload evicts tile 0's page after tile 0's frontier was
        # snapshotted, inside the same submission.
        small = WgpuPlaneExecutor(
            pool_layers={"complex_rg32f": 1}, device=_shared_wgpu_device()
        )
        view._wgpu_executor = small
        view._ensure_wgpu_executor = lambda required, **_kwargs: small
        view.setResidentHistogramEvidenceRequired(True)

        bail_events = []
        monkeypatch.setattr(
            "arrayscope.display.wgpu_imageview2d.emit_trace",
            lambda kind, **fields: bail_events.append((kind, fields)),
        )

        geometry = _montage_geometry((16, 24), 2, 1, loaded=2)
        payloads = {
            0: _payload(
                0,
                np.full((16, 24), 3.0 + 4.0j, np.complex64),
                source_id=("hist-race", 0),
            ),
            1: _payload(
                1,
                np.full((16, 24), 6.0 + 8.0j, np.complex64),
                source_id=("hist-race", 1),
            ),
        }
        report = _commit(view, geometry, payloads, levels=(0.0, 10.0))

        # Blast radius: the commit completes with partial residency and only
        # the evicted frontier's evidence spec is dropped — loudly.
        assert set(report.presented_tiles) == {1}
        assert [
            (kind, fields.get("reason"))
            for kind, fields in bail_events
            if kind == "wgpu_histogram_queue_bail"
        ] == [("wgpu_histogram_queue_bail", "evicted_in_batch")]
        (evidence,) = view.residentHistogramEvidence(payloads)
        assert evidence.tile_number == 1
        evidence.wait_completed()
        counts, _bounds = evidence.readback.resolve()
        assert int(counts.sum()) == 16 * 24

        # The frontier shield is submission-scoped: no permanent pins.
        remaining = small.page_table.resident_keys()
        assert remaining
        assert not any(small.page_table.is_pinned(key) for key in remaining)
        assert set(small.page_table.eviction_candidates()) == set(remaining)

        # Dropped evidence retries via the normal re-queue machinery: a
        # commit that fits the pool delivers tile 0's evidence after all.
        solo_geometry = _montage_geometry((16, 24), 1, 1, loaded=1)
        report = _commit(view, solo_geometry, {0: payloads[0]}, levels=(0.0, 10.0))
        assert set(report.presented_tiles) == {0}
        (retried,) = view.residentHistogramEvidence({0: payloads[0]})
        assert retried.tile_number == 0
        retried.wait_completed()
        counts, _bounds = retried.readback.resolve()
        assert int(counts.sum()) == 16 * 24
    finally:
        view.close()


def test_resident_histogram_obligation_survives_camera_only_coverage_reopen(qt_app):
    """Resident content must not dispatch/resolve again for a camera retarget."""

    view = _shown_view(qt_app)
    try:
        image = np.linspace(-2.0, 5.0, 20 * 30, dtype=np.float32).reshape(20, 30)
        geometry = _montage_geometry((20, 30), 1, 1, loaded=1)
        payloads = {0: _payload(0, image, source_id=("camera-histogram", 0))}
        obligation = ("content-and-mapping", 1)

        view.setResidentHistogramEvidenceRequired(True, obligation)
        _commit(view, geometry, payloads, levels=(-2.0, 5.0))
        (evidence,) = view.residentHistogramEvidence(payloads)
        evidence.wait_completed()
        evidence.readback.resolve()
        view.acceptResidentHistogramEvidence((evidence.evidence_key,))

        # Closing/reopening the coverage phase is what a camera-only retarget
        # does. The completed content+mapping evidence remains authoritative.
        view.setResidentHistogramEvidenceRequired(False)
        view.getView().setRange(xRange=(2, 28), yRange=(0, 20), padding=0)
        _rerender_internal(view)
        view.setResidentHistogramEvidenceRequired(True, obligation)
        _commit(view, geometry, payloads, levels=(-2.0, 5.0))

        assert view.residentHistogramEvidence(payloads) == ()
        assert view._wgpu_executor.histogram_dispatches_total == 1
        assert view._wgpu_executor.histogram_readback_resolves_total == 1
    finally:
        view.close()


def test_phase1_windowable_rgb_uses_resident_alpha_histogram_signal(qt_app):
    from arrayscope.display.shader_mapping import ShaderDisplayMode, ShaderMapping

    view = _shown_view(qt_app)
    try:
        histogram = np.linspace(2.0, 8.0, 20 * 30, dtype=np.float32).reshape(20, 30)
        image = np.full((20, 30, 3), 0.5, dtype=np.float32)
        geometry = _montage_geometry((20, 30), 1, 1, loaded=1)
        payloads = {
            0: _payload(
                0,
                image,
                source_id=("g6a-windowed-rgb", 0),
                shader_mapping=ShaderMapping(
                    display_mode=ShaderDisplayMode.RGB_WINDOWED
                ),
                histogram_data=histogram,
            )
        }
        view.setResidentHistogramEvidenceRequired(True)

        report = _commit(view, geometry, payloads, levels=(2.0, 8.0))

        assert report.presented_tiles == frozenset({0})
        (evidence,) = view.residentHistogramEvidence(payloads)
        evidence.wait_completed()
        counts, bounds = evidence.readback.resolve()
        assert bounds == pytest.approx((2.0, 8.0))
        assert int(counts.sum()) == histogram.size
    finally:
        view.close()


def test_coarse_payload_falls_back_then_native_payload_refines_same_plane(qt_app):
    view = _shown_view(qt_app)
    try:
        source_shape = (512, 512)
        geometry = _montage_geometry(source_shape, 1, 1, loaded=1)
        coarse = _lod_payload(
            0,
            np.full((32, 32), 0.25, np.float32),
            base_source_id="lod-plane",
            # Rung labels describe quality role, not pyramid exponent.  The
            # physical factor is the executor's LOD owner (ADR 0050).
            level=3,
            factor=16,
            source_shape=source_shape,
            payload_source_shape=(32, 32),
        )
        fine = _lod_payload(
            0,
            np.full(source_shape, 0.8, np.float32),
            base_source_id="lod-plane",
            level=0,
            source_shape=source_shape,
        )

        report = _commit(view, geometry, {0: coarse}, levels=(0.0, 1.0))
        assert report.texture_uploads == 1
        assert report.presented_identities == {0: coarse.tile_identity}
        assert view._wgpu_executor._bound_planes[0].max_lod == 4
        assert view._wgpu_camera_tiles()[0].lod_level == 0
        assert view._wgpu_camera_tiles()[0].src_size == (512.0, 512.0)
        view.getView().setRange(xRange=(0, 512), yRange=(0, 512), padding=0)
        _rerender_internal(view)
        assert np.allclose(_center_pixel(view)[:3], 64, atol=2)

        coarse_keys = set(view._wgpu_executor.page_table.resident_keys())
        report = _commit(view, geometry, {0: fine}, levels=(0.0, 1.0))
        assert report.texture_uploads == 4
        assert report.presented_identities == {0: fine.tile_identity}
        assert view._wgpu_executor._bound_planes[0].max_lod == 4
        assert coarse_keys <= set(view._wgpu_executor.page_table.resident_keys())
        assert all(view._wgpu_executor.page_table.is_pinned(key) for key in coarse_keys)
        assert {
            key.document_generation
            for key in view._wgpu_executor.page_table.resident_keys()
        } == {view._wgpu_executor._bound_planes[0].document_generation}
        _rerender_internal(view)
        assert np.allclose(_center_pixel(view)[:3], 204, atol=2)
    finally:
        view.close()


def test_coarser_mean_payload_generates_from_resident_fine_pages_zero_upload(qt_app):
    from arrayscope.display.pyramid import reduce_box_mean
    from arrayscope.gpu.keys import REDUCER_MEAN

    view = _shown_view(qt_app)
    try:
        source_shape = (512, 512)
        rng = np.random.default_rng(606)
        fine_values = rng.standard_normal(source_shape, dtype=np.float32)
        coarse_values = reduce_box_mean(fine_values, (4, 4))
        # Deliberately hostile descriptor payload: the live path must ignore
        # these CPU bytes and derive the requested page from resident L0.
        coarse_payload_values = np.full(coarse_values.shape, 123.0, np.float32)
        geometry = _montage_geometry(source_shape, 1, 1, loaded=1)
        fine = _lod_payload(
            0,
            fine_values,
            base_source_id="gpu-generated-lod-plane",
            level=0,
            source_shape=source_shape,
        )
        coarse = _lod_payload(
            0,
            coarse_payload_values,
            base_source_id="gpu-generated-lod-plane",
            level=2,
            source_shape=source_shape,
        )

        first = _commit(view, geometry, {0: fine}, levels=(-5.0, 5.0))
        assert first.texture_uploads == 4
        second = _commit(view, geometry, {0: coarse}, levels=(-5.0, 5.0))

        assert second.texture_uploads == 0
        assert second.presented_identities == {0: coarse.tile_identity}
        (generated_key,) = view._wgpu_committed["tiles"][0]["page_keys"]
        assert generated_key.lod.reducer == REDUCER_MEAN
        assert generated_key.lod.level == 2
        assert view._wgpu_executor.page_table.lookup(generated_key) is not None
        view._wgpu_executor.device.queue.on_submitted_work_done_sync()
        gpu_page = view._wgpu_executor.read_resident_page(generated_key)
        np.testing.assert_allclose(
            gpu_page[: coarse_values.shape[0], : coarse_values.shape[1]],
            coarse_values,
            rtol=1e-6,
            atol=1e-6,
        )
        assert not np.any(gpu_page[coarse_values.shape[0] :, :])
        assert not np.any(gpu_page[:, coarse_values.shape[1] :])
    finally:
        view.close()


def test_non_power_of_two_payload_factor_is_rejected_loudly(qt_app):
    view = _shown_view(qt_app)
    try:
        source_shape = (48, 48)
        geometry = _montage_geometry(source_shape, 1, 1, loaded=1)
        payload = _lod_payload(
            0,
            np.zeros((16, 16), np.float32),
            base_source_id="bad-lod-factor",
            level=2,
            factor=3,
            source_shape=source_shape,
        )

        with pytest.raises(NotImplementedError, match="power-of-two.*factor 3"):
            _commit(view, geometry, {0: payload}, levels=(0.0, 1.0))
    finally:
        view.close()


def test_partial_residency_acknowledges_only_resident_tiles(qt_app):
    from arrayscope.display.wgpu_imageview2d import _shared_wgpu_device
    from arrayscope.gpu.wgpu_executor import WgpuPlaneExecutor

    view = _shown_view(qt_app)
    try:
        # One-layer scalar pool: the second tile's upload must evict the
        # first tile's page inside the same submission.
        small = WgpuPlaneExecutor(
            pool_layers={"scalar_r32f": 1}, device=_shared_wgpu_device()
        )
        view._wgpu_executor = small
        view._ensure_wgpu_executor = lambda required, **_kwargs: small

        geometry = _montage_geometry((20, 30), 2, 1, loaded=2)
        payloads = {
            0: _payload(0, np.zeros((20, 30), np.float32), source_id=("partial", 0)),
            1: _payload(1, np.ones((20, 30), np.float32), source_id=("partial", 1)),
        }
        report = _commit(view, geometry, payloads, levels=(0.0, 1.0))
        assert set(report.presented_tiles) == {1}
        assert report.presented_identities == {1: ("partial", 1)}
    finally:
        view.close()


def test_complex_tile_mode_switch_is_zero_upload_with_physical_truth(qt_app):
    from arrayscope.display.shader_mapping import (
        ShaderComponent,
        ShaderDisplayMode,
        ShaderMapping,
    )

    view = _shown_view(qt_app)
    try:
        value = 3.0 + 4.0j  # magnitude exactly 5
        image = np.full((16, 24), value, dtype=np.complex64)
        geometry = _montage_geometry((16, 24), 1, 1, loaded=1)

        def mapping(component):
            return ShaderMapping(
                component=component, display_mode=ShaderDisplayMode.COMPLEX
            )

        payloads = {
            0: _payload(
                0,
                image,
                source_id=("wgpu-complex", 1),
                shader_mapping=mapping(ShaderComponent.ABS),
            )
        }
        report = _commit(view, geometry, payloads, levels=(0.0, 10.0))
        assert set(report.presented_tiles) == {0}
        assert report.texture_uploads == 1
        assert view._wgpu_mapping_state.mode == "magnitude"
        view.getView().setRange(xRange=(0, 24), yRange=(0, 16), padding=0)
        _rerender_internal(view)
        # |3+4j| = 5 → g = 0.5 → grayscale 128 (nearest LUT entry).
        assert np.allclose(_center_pixel(view), (128, 128, 128, 255), atol=2)

        # Mode switch (same content identity): physically zero-upload.
        payloads = {
            0: _payload(
                0,
                image,
                source_id=("wgpu-complex", 1),
                shader_mapping=mapping(ShaderComponent.REAL),
            )
        }
        report = _commit(view, geometry, payloads, levels=(0.0, 10.0))
        assert set(report.presented_tiles) == {0}
        assert report.texture_uploads == 0
        assert view._wgpu_mapping_state.mode == "real"
        _rerender_internal(view)
        # Re(3+4j) = 3 → g = 0.3 → grayscale round(76.5) ∈ {76, 77}.
        assert np.allclose(_center_pixel(view), (76, 76, 76, 255), atol=2)

        # Levels switch through the shared preview driver: zero-upload too.
        before = view._wgpu_executor.uploads_total
        view._apply_preview_levels_to_display((0.0, 5.0), final=True)
        assert view._wgpu_executor.uploads_total == before
        assert view._wgpu_mapping_state.level_hi == pytest.approx(5.0)
    finally:
        view.close()


def test_complex_montage_acknowledges_only_resident_content_planes(qt_app):
    from arrayscope.display.wgpu_imageview2d import _shared_wgpu_device
    from arrayscope.gpu.wgpu_executor import WgpuPlaneExecutor

    view = _shown_view(qt_app)
    try:
        # One physical complex layer for two requested ContentPlanes: the
        # second upload evicts the first, so only tile 1 can be acknowledged.
        small = WgpuPlaneExecutor(
            pool_layers={"complex_rg32f": 1}, device=_shared_wgpu_device()
        )
        view._wgpu_executor = small
        view._ensure_wgpu_executor = lambda required, **_kwargs: small

        geometry = _montage_geometry((16, 24), 2, 1, loaded=2)
        payloads = {
            0: _payload(
                0,
                np.full((16, 24), 3.0 + 4.0j, np.complex64),
                source_id=("complex-partial", 0),
            ),
            1: _payload(
                1,
                np.full((16, 24), 6.0 + 8.0j, np.complex64),
                source_id=("complex-partial", 1),
            ),
        }
        report = _commit(view, geometry, payloads, levels=(0.0, 10.0))
        assert set(report.presented_tiles) == {1}
        assert report.presented_identities == {1: ("complex-partial", 1)}
        assert len(view._wgpu_executor.bound_planes) == 2
        assert all(
            plane.representation == "complex_rg32f"
            for plane in view._wgpu_executor.bound_planes
        )
    finally:
        view.close()


def test_complex_montage_mode_switch_is_zero_upload_per_tile(qt_app):
    from arrayscope.display.shader_mapping import (
        ShaderComponent,
        ShaderDisplayMode,
        ShaderMapping,
    )

    view = _shown_view(qt_app)
    try:
        geometry = _montage_geometry((16, 24), 2, 1, loaded=2)
        images = {
            0: np.full((16, 24), 3.0 + 4.0j, np.complex64),
            1: np.full((16, 24), 6.0 + 8.0j, np.complex64),
        }

        def payloads(component):
            mapping = ShaderMapping(
                component=component, display_mode=ShaderDisplayMode.COMPLEX
            )
            return {
                tile: _payload(
                    tile,
                    image,
                    source_id=("complex-montage", tile),
                    shader_mapping=mapping,
                )
                for tile, image in images.items()
            }

        report = _commit(
            view, geometry, payloads(ShaderComponent.ABS), levels=(0.0, 10.0)
        )
        assert set(report.presented_tiles) == {0, 1}
        assert report.presented_identities == {
            0: ("complex-montage", 0),
            1: ("complex-montage", 1),
        }
        assert report.texture_uploads == 2
        assert len(view._wgpu_executor.bound_planes) == 2
        view.getView().setRange(xRange=(0, 48), yRange=(0, 16), padding=0)
        _rerender_internal(view)
        target = view._wgpu_executor.read_target()
        h, w = target.shape[:2]
        # Each tile samples its own ContentPlane: magnitudes 5 and 10.
        assert np.allclose(target[h // 2, w // 4], (128, 128, 128, 255), atol=2)
        assert np.allclose(target[h // 2, 3 * w // 4], (255, 255, 255, 255), atol=2)

        # Same per-tile content identities, new component uniform: both tiles
        # remain physically acknowledged without another texture upload.
        report = _commit(
            view, geometry, payloads(ShaderComponent.REAL), levels=(0.0, 10.0)
        )
        assert set(report.presented_tiles) == {0, 1}
        assert report.presented_identities == {
            0: ("complex-montage", 0),
            1: ("complex-montage", 1),
        }
        assert report.texture_uploads == 0
        assert view._wgpu_mapping_state.mode == "real"
        _rerender_internal(view)
        target = view._wgpu_executor.read_target()
        # Uniform-only mode switch exposes real components 3 and 6.
        assert np.allclose(target[h // 2, w // 4], (76, 76, 76, 255), atol=2)
        assert np.allclose(target[h // 2, 3 * w // 4], (153, 153, 153, 255), atol=2)
    finally:
        view.close()


def test_complex_phase_color_uses_phase_lut(qt_app):
    from arrayscope.display.shader_mapping import (
        ShaderComponent,
        ShaderDisplayMode,
        ShaderMapping,
        default_phase_lut,
    )

    view = _shown_view(qt_app)
    try:
        phase = np.pi / 2
        image = np.full((16, 24), np.exp(1j * phase), dtype=np.complex64)
        geometry = _montage_geometry((16, 24), 1, 1, loaded=1)
        payloads = {
            0: _payload(
                0,
                image,
                source_id=("wgpu-phase", 1),
                shader_mapping=ShaderMapping(
                    component=ShaderComponent.ANGLE,
                    display_mode=ShaderDisplayMode.PHASE_COLOR,
                ),
            )
        }
        report = _commit(view, geometry, payloads, levels=(-np.pi, np.pi))
        assert set(report.presented_tiles) == {0}
        assert view._wgpu_mapping_state.mode == "phase"
        assert view._wgpu_mapping_state.lut is not None
        view.getView().setRange(xRange=(0, 24), yRange=(0, 16), padding=0)
        _rerender_internal(view)
        # g = (π/2 + π) / 2π = 0.75 → nearest phase-LUT entry 191.
        expected = default_phase_lut()[191]
        assert np.allclose(_center_pixel(view)[:3], expected, atol=2)
    finally:
        view.close()


def test_magnitude_modulated_phase_color_matches_cpu_oracle_and_switches_zero_upload(qt_app):
    from dataclasses import replace

    from arrayscope.display.shader_mapping import (
        ShaderComponent,
        ShaderDisplayMode,
        ShaderMapping,
        cpu_display_rgba,
        default_phase_lut,
    )

    view = _shown_view(qt_app)
    try:
        phase = np.pi / 3.0
        image = np.full(
            (16, 24), 0.5 * np.exp(1j * phase), dtype=np.complex64
        )
        geometry = _montage_geometry((16, 24), 1, 1, loaded=1)
        scalar_mapping = ShaderMapping(
            component=ShaderComponent.ABS,
            display_mode=ShaderDisplayMode.COMPLEX,
        )
        phase_mapping = replace(
            scalar_mapping,
            display_mode=ShaderDisplayMode.PHASE_COLOR,
        )
        source_id = ("wgpu-phase-modulated", 1)

        report = _commit(
            view,
            geometry,
            {
                0: _payload(
                    0,
                    image,
                    source_id=source_id,
                    shader_mapping=scalar_mapping,
                )
            },
            levels=(0.0, 1.0),
        )
        assert set(report.presented_tiles) == {0}
        assert report.texture_uploads == 1

        report = _commit(
            view,
            geometry,
            {
                0: _payload(
                    0,
                    image,
                    source_id=source_id,
                    shader_mapping=phase_mapping,
                )
            },
            levels=(0.0, 1.0),
        )
        assert report.texture_uploads == 0
        view.getView().setRange(xRange=(0, 24), yRange=(0, 16), padding=0)
        _rerender_internal(view)
        expected_mapping = replace(
            phase_mapping,
            levels=(0.0, 1.0),
            lut_data=default_phase_lut(),
        )
        expected = cpu_display_rgba(image, expected_mapping)[0, 0]
        assert np.allclose(_center_pixel(view), expected, atol=3)
    finally:
        view.close()


def test_float_rgb_acknowledges_only_physically_resident_packed_pages(qt_app):
    from arrayscope.display.shader_mapping import ShaderDisplayMode, ShaderMapping
    from arrayscope.display.wgpu_imageview2d import _shared_wgpu_device
    from arrayscope.gpu.keys import RGB_WINDOWED_RGBA32F
    from arrayscope.gpu.wgpu_executor import WgpuPlaneExecutor

    view = _shown_view(qt_app)
    try:
        small = WgpuPlaneExecutor(
            pool_layers={RGB_WINDOWED_RGBA32F: 1},
            device=_shared_wgpu_device(),
        )
        view._wgpu_executor = small
        view._ensure_wgpu_executor = lambda required, **_kwargs: small
        geometry = _montage_geometry((20, 30), 2, 1, loaded=2)
        mapping = ShaderMapping(display_mode=ShaderDisplayMode.RGB_WINDOWED)
        payloads = {
            tile: _payload(
                tile,
                np.full((20, 30, 3), 0.25 + 0.25 * tile, np.float32),
                source_id=("wgpu-float-rgb-partial", tile),
                shader_mapping=mapping,
                histogram_data=np.full((20, 30), 0.5, np.float32),
            )
            for tile in (0, 1)
        }

        report = _commit(
            view,
            geometry,
            payloads,
            levels=(0.0, 1.0),
            rgb_already_windowed=False,
        )

        assert report.texture_uploads == 2
        assert set(report.presented_tiles) == {1}
        assert report.presented_identities == {
            1: ("wgpu-float-rgb-partial", 1)
        }
        assert {
            key.representation for key in small.page_table.resident_keys()
        } == {RGB_WINDOWED_RGBA32F}
    finally:
        view.close()


def test_log_and_symlog_scale_switch_is_zero_upload(qt_app):
    from arrayscope.display.shader_mapping import ShaderMapping, ShaderScale

    view = _shown_view(qt_app)
    try:
        image = np.full((16, 24), 100.0, dtype=np.float32)
        geometry = _montage_geometry((16, 24), 1, 1, loaded=1)

        def payload(scale, *, symlog_constant=0.0):
            return {
                0: _payload(
                    0,
                    image,
                    source_id=("wgpu-scale", 1),
                    shader_mapping=ShaderMapping(
                        scale=scale, symlog_constant=symlog_constant
                    ),
                )
            }

        report = _commit(
            view,
            geometry,
            payload(ShaderScale.LOG),
            levels=(0.0, 4.0),
        )
        assert set(report.presented_tiles) == {0}
        assert report.texture_uploads == 1
        assert view._wgpu_mapping_state.scale == "log"
        view.getView().setRange(xRange=(0, 24), yRange=(0, 16), padding=0)
        _rerender_internal(view)
        # log10(100) = 2 in [0, 4] -> nearest grayscale entry 128.
        assert np.allclose(_center_pixel(view), (128, 128, 128, 255), atol=2)

        report = _commit(
            view,
            geometry,
            payload(ShaderScale.SYMLOG, symlog_constant=1.0),
            levels=(0.0, 2.0),
        )
        assert set(report.presented_tiles) == {0}
        assert report.texture_uploads == 0
        assert view._wgpu_mapping_state.scale == "symlog"
        assert view._wgpu_mapping_state.symlog_constant == pytest.approx(1.0)
        _rerender_internal(view)
        # symlog(100, C=1) = log10(11), mapped through [0, 2].
        expected = round(np.log10(11.0) / 2.0 * 255.0)
        assert np.allclose(
            _center_pixel(view), (*([expected] * 3), 255), atol=2
        )
    finally:
        view.close()


def test_rgb_display_ready_tile_renders_raw_bytes(qt_app):
    view = _shown_view(qt_app)
    try:
        color = np.array([10, 200, 60], np.uint8)
        image = np.broadcast_to(color, (20, 30, 3)).copy()
        geometry = _montage_geometry((20, 30), 1, 1, loaded=1)
        payloads = {0: _payload(0, image, source_id=("wgpu-rgb", 1))}
        report = _commit(
            view, geometry, payloads, levels=(0.0, 1.0), rgb_already_windowed=True
        )
        assert set(report.presented_tiles) == {0}
        view.getView().setRange(xRange=(0, 30), yRange=(0, 20), padding=0)
        _rerender_internal(view)
        # Display-ready bytes: levels/LUT bypassed, rendered as-is.
        assert (_center_pixel(view) == (*color, 255)).all()
    finally:
        view.close()


def test_float_rgb_windowing_matches_cpu_reference_and_levels_switch_is_zero_upload(qt_app):
    from arrayscope.display.image_upload import rgb_display_for_levels
    from arrayscope.display.shader_mapping import (
        ShaderDisplayMode,
        ShaderMapping,
        TexturePlaneKind,
        pack_texture_data,
    )

    view = _shown_view(qt_app)
    try:
        color = np.array([0.25, 0.5, 1.0], np.float32)
        image = np.broadcast_to(color, (20, 30, 3)).copy()
        histogram = np.full((20, 30), 0.5, np.float32)
        geometry = _montage_geometry((20, 30), 1, 1, loaded=1)
        payloads = {
            0: _payload(
                0,
                image,
                source_id=("wgpu-float-rgb", 1),
                shader_mapping=ShaderMapping(
                    display_mode=ShaderDisplayMode.RGB_WINDOWED
                ),
                histogram_data=histogram,
            )
        }

        report = _commit(
            view,
            geometry,
            payloads,
            levels=(0.0, 1.0),
            rgb_already_windowed=False,
        )
        assert set(report.presented_tiles) == {0}
        assert report.texture_uploads == 1
        view.getView().setRange(xRange=(0, 30), yRange=(0, 20), padding=0)
        _rerender_internal(view)
        base = pack_texture_data(image, TexturePlaneKind.RGB8)
        expected = rgb_display_for_levels(base, histogram, (0.0, 1.0))[0, 0]
        assert np.allclose(_center_pixel(view), (*expected, 255), atol=2)

        before = view._wgpu_executor.uploads_total
        view._apply_preview_levels_to_display((0.0, 0.5), final=True)
        assert view._wgpu_executor.uploads_total == before
        _rerender_internal(view)
        expected = rgb_display_for_levels(base, histogram, (0.0, 0.5))[0, 0]
        assert np.allclose(_center_pixel(view), (*expected, 255), atol=2)
    finally:
        view.close()


def test_out_of_scope_commits_reject_loudly(qt_app):
    from arrayscope.display.shader_mapping import (
        ShaderComponent,
        ShaderDisplayMode,
        ShaderMapping,
    )

    view = _shown_view(qt_app)
    try:
        scalar = np.zeros((20, 30), np.float32)
        cplx = np.zeros((20, 30), np.complex64)
        geometry2 = _montage_geometry((20, 30), 2, 1, loaded=2)
        geometry1 = _montage_geometry((20, 30), 1, 1, loaded=1)

        # Mixed representations in one commit.
        with pytest.raises(NotImplementedError, match="one texture representation"):
            _commit(
                view,
                geometry2,
                {
                    0: _payload(0, scalar, source_id=("rej", 2)),
                    1: _payload(1, cplx.copy(), source_id=("rej", 3)),
                },
                levels=(0.0, 1.0),
            )
        # Display-ready promises are still strict: float RGB is not silently
        # quantized into the bypass pool. The supported float path is the
        # separately tested windowable-RGB representation.
        with pytest.raises(NotImplementedError, match="display-ready"):
            _commit(
                view,
                geometry1,
                {
                    0: _payload(
                        0,
                        np.zeros((20, 30, 3), np.float32),
                        source_id=("rej", 5),
                    )
                },
                levels=(0.0, 1.0),
                rgb_already_windowed=True,
            )
        # Phase-color has honest phase and magnitude variants only. A real
        # component plus cyclic hue has no established backend semantics.
        with pytest.raises(NotImplementedError, match="phase or magnitude"):
            _commit(
                view,
                geometry1,
                {
                    0: _payload(
                        0,
                        cplx,
                        source_id=("rej", 6),
                        shader_mapping=ShaderMapping(
                            component=ShaderComponent.REAL,
                            display_mode=ShaderDisplayMode.PHASE_COLOR,
                        ),
                    )
                },
                levels=(0.0, 1.0),
            )
        # The rejected commits must not have left a half-presented surface.
        assert view.montageDisplayMode() == "none"
    finally:
        view.close()


def test_warm_tiled_residency_accepts_the_commit_plan_contract(qt_app):
    """Regression: the live warm path must consume _wgpu_commit_plan's full
    return contract.  The 3c-prep branch unpacked two values while the
    rejection-lift branch grew the plan to four — no offscreen test drove
    warmTiledResidency, so only the real-Wayland journey matrix caught the
    ValueError.  This pins the seam offscreen."""

    import numpy as np

    view = _shown_view(qt_app)
    try:
        geometry = _montage_geometry((16, 24), 2, 1, loaded=2)
        images = {
            tile: np.linspace(0.0, 1.0, 16 * 24, dtype=np.float32).reshape(16, 24) + tile
            for tile in (0, 1)
        }
        payloads = {
            tile: _payload(tile, image, source_id=f"warm-src-{tile}")
            for tile, image in images.items()
        }
        _commit(view, geometry, {0: payloads[0]}, levels=(0.0, 2.0))
        resident_before = len(view._wgpu_executor.page_table.resident_keys())
        view.warmTiledResidency(
            payloads={1: payloads[1]},
            geometry=geometry,
            levels=(0.0, 2.0),
        )
        assert len(view._wgpu_executor.page_table.resident_keys()) >= resident_before
    finally:
        view.close()


def test_axis_inversion_mirrors_content_and_redraws_without_commit(qt_app):
    """Dogfood bugs 2026-07-18: (1) flips only took effect after the next
    commit — the view listened to sigRangeChanged but not sigStateChanged;
    (2) xInverted was ignored by the camera rect math entirely, so drawn
    content disagreed with the ViewBox's (correctly inverted) interaction
    mapping — drags and zoom rects landed on mirrored features."""

    from arrayscope.gpu.command_protocol import PresentGeneration, UpdateTileInstances

    view = _shown_view(qt_app)
    try:
        # One tile whose left half is dark and right half is bright.
        image = np.zeros((32, 64), dtype=np.float32)
        image[:, 32:] = 1.0
        geometry = _montage_geometry((32, 64), 1, 1, loaded=1)
        payloads = {0: _payload(0, image, source_id="flip-src")}
        _commit(view, geometry, payloads, levels=(0.0, 1.0))
        view.getView().setRange(xRange=(0, 64), yRange=(0, 32), padding=0)

        def render_columns():
            view._submit_wgpu(
                (UpdateTileInstances(view._wgpu_camera_tiles()), PresentGeneration(999))
            )
            target = view._wgpu_executor.read_target().astype(np.float32)
            h, w = target.shape[:2]
            row = target[h // 2, :, 0]
            return float(row[: w // 4].mean()), float(row[-w // 4 :].mean())

        left, right = render_columns()
        assert right > left + 50  # bright half on the right pre-flip

        uploads_before = view._wgpu_executor.uploads_total
        draws_before = int(getattr(view, "_wgpu_draw_count", 0) or 0)
        view._wgpu_canvas_update_pending = False
        view.getView().invertX(True)
        qt_app.processEvents()
        # Bug 1: the inversion toggle alone must request a redraw (pending
        # flag armed, or the ondemand draw already ran).
        assert bool(getattr(view, "_wgpu_canvas_update_pending", False)) or (
            int(getattr(view, "_wgpu_draw_count", 0) or 0) > draws_before
        )
        # Bug 2: the drawn content must mirror horizontally, with zero uploads.
        left, right = render_columns()
        assert left > right + 50, f"content not mirrored: left={left} right={right}"
        assert view._wgpu_executor.uploads_total == uploads_before
    finally:
        view.close()
