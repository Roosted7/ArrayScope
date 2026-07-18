"""wgpu executor behind the renderer command protocol: gate-B oracles A–G.

Skipped wherever wgpu or a Vulkan adapter is unavailable (CI runners); on
developer machines this is the default-ring proof that the protocol seam
preserves the experiment's zero-upload / never-black / physical-truth
guarantees.  Mirrors ``experiments/wgpu_gate_b/virtual_tensor.py``.
"""

import numpy as np
import pytest

wgpu = pytest.importorskip("wgpu")

from arrayscope.gpu.command_protocol import (  # noqa: E402
    BindContentPlanes,
    ContentPlane,
    DispatchHistogram,
    DisplayMapping,
    EnsureChunkResident,
    EvictChunk,
    FrameSubmission,
    GenerateLodPages,
    OverlayPrimitive,
    PresentGeneration,
    SetOverlayCamera,
    SetDisplayMapping,
    TileInstance,
    UpdateOverlayGeometry,
    UpdateTileInstances,
)
from arrayscope.gpu.chunk_summary import (  # noqa: E402
    HISTOGRAM_NORMALIZED_L1_TOLERANCE,
)
from arrayscope.gpu.keys import (  # noqa: E402
    COMPLEX_RG32F,
    REDUCER_MEAN_ABS,
    REDUCER_PHASE_VECTOR,
    RGB8,
    RGB_WINDOWED_RGBA32F,
    SCALAR_R32F,
)
from arrayscope.gpu.wgpu_executor import (  # noqa: E402
    PAGE,
    WgpuPlaneExecutor,
    plane_chunk_key,
)

PLANE = 1024
GRID0 = PLANE // PAGE
GRID1 = GRID0 // 2
CANVAS = (768, 768)
_MODES = {"magnitude": 0, "phase": 1, "real": 2, "imag": 3}

# One wgpu device for the whole module (mirrors the live view's shared
# device).  Each parameterless WgpuPlaneExecutor would otherwise request its
# own adapter+device; with several executors per file across parallel xdist
# workers that concurrency made wgpu-native abort inside
# request_adapter_sync (full-suite worker crash, 2026-07-18).
_DEVICE = None


def _shared_device():
    global _DEVICE
    if _DEVICE is None:
        from wgpu.backends.wgpu_native.extras import set_instance_extras

        try:
            # Vulkan-only instance: the GL backend's EGL re-init is fatal
            # under Wayland (gate-B Tier 0).  Harmless if already set.
            set_instance_extras(backends=["Vulkan"])
        except RuntimeError:
            pass
        adapter = wgpu.gpu.request_adapter_sync(power_preference="low-power")
        _DEVICE = adapter.request_device_sync()
    return _DEVICE


def _adapter_available() -> bool:
    try:
        from wgpu.backends.wgpu_native.extras import set_instance_extras

        try:
            # Vulkan-only instance BEFORE the first adapter request (the
            # GL backend's EGL re-init is fatal in workers with GL state).
            set_instance_extras(backends=["Vulkan"])
        except RuntimeError:
            pass  # instance already exists
        wgpu.gpu.request_adapter_sync(power_preference="low-power")
        return True
    except Exception:
        return False


def test_overlay_geometry_is_one_semantic_buffer_and_camera_is_uniform_only():
    executor = WgpuPlaneExecutor(
        target_size=(128, 96),
        pool_layers={SCALAR_R32F: 1},
        device=_shared_device(),
    )
    geometry = UpdateOverlayGeometry(
        (
            OverlayPrimitive(
                "world_rect",
                (0.20, 0.25),
                (0.40, 0.45),
                (0.1, 0.9, 0.2, 1.0),
            ),
        )
    )
    first = executor.submit(
        FrameSubmission(
            1,
            (
                geometry,
                SetOverlayCamera((0.0, 0.0, 1.0, 1.0)),
                PresentGeneration(1),
            ),
        )
    )
    before = executor.read_target()
    before_rows, before_columns = np.nonzero(before[..., 1] > 150)
    assert len(before_rows) > 100
    assert first.overlay_buffer_writes == 1
    assert executor.overlay_buffer_writes_total == 1

    camera_only = executor.submit(
        FrameSubmission(
            2,
            (
                SetOverlayCamera((0.10, 0.0, 1.10, 1.0)),
                PresentGeneration(2),
            ),
        )
    )
    after = executor.read_target()
    _after_rows, after_columns = np.nonzero(after[..., 1] > 150)
    assert float(after_columns.mean()) < float(before_columns.mean()) - 8.0
    assert camera_only.overlay_buffer_writes == 0
    assert executor.overlay_buffer_writes_total == 1

    cleared = executor.submit(
        FrameSubmission(
            3,
            (UpdateOverlayGeometry(()), PresentGeneration(3)),
        )
    )
    assert cleared.overlay_buffer_writes == 1
    assert not np.any(executor.read_target()[..., :3])


pytestmark = pytest.mark.skipif(
    not _adapter_available(), reason="no wgpu adapter on this machine"
)


def _scale_reference(values, mapping):
    """Independent CPU mirror of the WGSL display-scale formulas."""

    values = np.asarray(values, dtype=np.float32)
    if mapping.scale == "linear":
        return values
    if mapping.scale == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(np.maximum(values, 0.0)).astype(np.float32, copy=False)
    if mapping.scale == "symlog":
        with np.errstate(divide="ignore", invalid="ignore"):
            return (
                np.sign(values)
                * np.log10(
                    1.0
                    + np.abs(values) / (10.0 ** float(mapping.symlog_constant))
                )
            ).astype(np.float32, copy=False)
    raise AssertionError(f"unhandled scale {mapping.scale!r}")


def _plane() -> np.ndarray:
    rng = np.random.default_rng(42)
    re = rng.standard_normal((PLANE, PLANE), dtype=np.float32)
    im = rng.standard_normal((PLANE, PLANE), dtype=np.float32)
    yy, xx = np.mgrid[0:PLANE, 0:PLANE].astype(np.float32)
    re += np.sin(xx / 37.0) * 2 + (xx / PLANE)
    im += np.cos(yy / 23.0) * 2
    return np.stack([re, im], axis=-1)


def test_gpu_generated_complex_mean_page_matches_cpu_component_reference():
    from arrayscope.display.pyramid import reduce_box_mean

    rng = np.random.default_rng(20260718)
    source = (
        rng.standard_normal((PAGE * 2, PAGE * 2), dtype=np.float32)
        + 1j * rng.standard_normal((PAGE * 2, PAGE * 2), dtype=np.float32)
    ).astype(np.complex64)
    executor = WgpuPlaneExecutor(
        pool_layers={COMPLEX_RG32F: 8}, device=_shared_device()
    )
    sources = tuple(
        plane_chunk_key(
            "lod-doc",
            "lod-op",
            0,
            cx,
            cy,
            plane_shape=source.shape,
        )
        for cy in range(2)
        for cx in range(2)
    )
    destination = plane_chunk_key(
        "lod-doc", "lod-op", 1, 0, 0, plane_shape=source.shape
    )
    upload = executor.submit(
        FrameSubmission(
            1,
            (
                BindContentPlanes(
                    (
                        ContentPlane(
                            "lod-doc",
                            "lod-op",
                            source.shape,
                            max_lod=1,
                            representation=COMPLEX_RG32F,
                        ),
                    )
                ),
                *(
                    EnsureChunkResident(
                        key,
                        source[
                            cy * PAGE : (cy + 1) * PAGE,
                            cx * PAGE : (cx + 1) * PAGE,
                        ],
                    )
                    for key, (cy, cx) in zip(
                        sources,
                        ((0, 0), (0, 1), (1, 0), (1, 1)),
                        strict=True,
                    )
                ),
            ),
        )
    )
    assert upload.uploads == 4

    generated = executor.submit(
        FrameSubmission(2, (GenerateLodPages(sources, destination),))
    )
    generated.wait_completed()

    assert generated.uploads == 0
    assert generated.lod_pages_generated == (destination,)
    assert executor.page_table.lookup(destination) is not None
    gpu = executor.read_resident_page(destination)
    expected = reduce_box_mean(source, (2, 2))
    np.testing.assert_allclose(gpu[..., 0], expected.real, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gpu[..., 1], expected.imag, rtol=1e-6, atol=1e-6)


def test_gpu_lod_generation_rejects_non_mean_family_loudly():
    executor = WgpuPlaneExecutor(
        pool_layers={SCALAR_R32F: 8}, device=_shared_device()
    )
    source = plane_chunk_key(
        "lod-doc",
        "lod-op",
        0,
        0,
        0,
        dtype="float32",
        representation=SCALAR_R32F,
        plane_shape=(PAGE, PAGE),
    )
    destination = plane_chunk_key(
        "lod-doc",
        "lod-op",
        1,
        0,
        0,
        dtype="float32",
        representation=SCALAR_R32F,
        plane_shape=(PAGE, PAGE),
        reducer=REDUCER_MEAN_ABS,
    )
    executor.submit(
        FrameSubmission(
            1,
            (EnsureChunkResident(source, np.ones((PAGE, PAGE), np.float32)),),
        )
    )
    with pytest.raises(ValueError, match="reducer-honest.*component mean only"):
        executor.submit(
            FrameSubmission(2, (GenerateLodPages((source,), destination),))
        )


def test_bound_plane_reducer_selects_one_resident_complex_lod_family():
    executor = WgpuPlaneExecutor(
        pool_layers={COMPLEX_RG32F: 4}, device=_shared_device()
    )
    mean = plane_chunk_key("family-doc", "family-op", 1, 0, 0)
    phase = plane_chunk_key(
        "family-doc",
        "family-op",
        1,
        0,
        0,
        reducer=REDUCER_PHASE_VECTOR,
    )
    values = np.zeros((PAGE, PAGE, 2), np.float32)
    executor.submit(
        FrameSubmission(
            1,
            (
                BindContentPlanes(
                    (
                        ContentPlane(
                            "family-doc",
                            "family-op",
                            (PAGE * 2, PAGE * 2),
                            max_lod=1,
                            representation=COMPLEX_RG32F,
                        ),
                    )
                ),
                EnsureChunkResident(mean, values),
                EnsureChunkResident(phase, values),
            ),
        )
    )
    lod1_base = executor._plane_grids[0][1].base
    assert executor._flat_table[lod1_base] == executor.page_table.lookup(mean).page_index

    executor.submit(
        FrameSubmission(
            2,
            (
                BindContentPlanes(
                    (
                        ContentPlane(
                            "family-doc",
                            "family-op",
                            (PAGE * 2, PAGE * 2),
                            max_lod=1,
                            representation=COMPLEX_RG32F,
                            lod_reducer=REDUCER_PHASE_VECTOR,
                        ),
                    )
                ),
            ),
        )
    )
    lod1_base = executor._plane_grids[0][1].base
    assert executor._flat_table[lod1_base] == executor.page_table.lookup(phase).page_index


class Scene:
    """Executor + data + CPU mirror shared by every oracle."""

    def __init__(self):
        self.plane = _plane()
        p = self.plane
        self.plane_l1 = (p[0::2, 0::2] + p[1::2, 0::2] + p[0::2, 1::2] + p[1::2, 1::2]) / 4.0
        self.executor = WgpuPlaneExecutor(
            (PLANE, PLANE), max_lod=1, target_size=CANVAS, device=_shared_device()
        )
        self.doc, self.op = "doc-1", "op-identity"

        commands = [
            BindContentPlanes(
                (
                    ContentPlane(
                        self.doc,
                        self.op,
                        (PLANE, PLANE),
                        max_lod=1,
                        representation=COMPLEX_RG32F,
                    ),
                )
            )
        ]
        for cy in range(GRID1):  # pinned coarse coverage first (ADR 0056)
            for cx in range(GRID1):
                commands.append(
                    EnsureChunkResident(
                        self.key(1, cx, cy), self.l1_page(cx, cy), pinned=True
                    )
                )
        for cy in range(GRID0):
            for cx in range(GRID0):
                commands.append(EnsureChunkResident(self.key(0, cx, cy), self.l0_page(cx, cy)))
        report = self.executor.submit(FrameSubmission(0, commands))
        assert report.uploads == GRID1 * GRID1 + GRID0 * GRID0  # 20

    def key(self, lod, cx, cy):
        return plane_chunk_key(self.doc, self.op, lod, cx, cy)

    def l0_page(self, cx, cy):
        return self.plane[cy * PAGE : (cy + 1) * PAGE, cx * PAGE : (cx + 1) * PAGE]

    def l1_page(self, cx, cy):
        return self.plane_l1[cy * PAGE : (cy + 1) * PAGE, cx * PAGE : (cx + 1) * PAGE]

    def render(self, tiles, mapping, generation=1):
        report = self.executor.submit(
            FrameSubmission(
                generation,
                (
                    SetDisplayMapping(mapping),
                    UpdateTileInstances(tuple(tiles)),
                    PresentGeneration(generation),
                ),
            )
        )
        assert report.presented
        return report

    def reference(self, tiles, mapping, absent_l0=()):
        w, h = CANVAS
        out = np.zeros((h, w, 4), np.uint8)
        out[..., 3] = 255
        mode = _MODES[mapping.mode]
        lo, hi = mapping.level_lo, mapping.level_hi
        for t in tiles:
            x0 = int(round(t.dst_rect[0] * w))
            y0 = int(round(t.dst_rect[1] * h))
            tw = int(round(t.dst_rect[2] * w))
            th = int(round(t.dst_rect[3] * h))
            sx = t.src_origin[0] + (np.arange(tw) + 0.5) / tw * t.src_size[0]
            sy = t.src_origin[1] + (np.arange(th) + 0.5) / th * t.src_size[1]
            sxg, syg = np.meshgrid(sx, sy)
            if t.lod_level == 1:
                cx = np.clip(sxg / 2, 0, PLANE // 2 - 1).astype(np.int64)
                cy = np.clip(syg / 2, 0, PLANE // 2 - 1).astype(np.int64)
                v = self.plane_l1[cy, cx]
            else:
                cx = np.clip(sxg, 0, PLANE - 1).astype(np.int64)
                cy = np.clip(syg, 0, PLANE - 1).astype(np.int64)
                v = self.plane[cy, cx].copy()
                if absent_l0:
                    cx1 = np.clip(sxg / 2, 0, PLANE // 2 - 1).astype(np.int64)
                    cy1 = np.clip(syg / 2, 0, PLANE // 2 - 1).astype(np.int64)
                    for (acx, acy) in absent_l0:
                        m = (cx // PAGE == acx) & (cy // PAGE == acy)
                        v[m] = self.plane_l1[cy1[m], cx1[m]]
            re = v[..., 0].astype(np.float32)
            im = v[..., 1].astype(np.float32)
            x = {
                0: lambda: np.sqrt(re * re + im * im),
                1: lambda: np.arctan2(im, re),
                2: lambda: re,
                3: lambda: im,
            }[mode]()
            x = _scale_reference(x, mapping)
            g = np.clip((x.astype(np.float64) - lo) / (hi - lo), 0, 1)
            if bool(getattr(mapping, "phase_color", False)) and mode != 1:
                phase = np.arctan2(im, re)
                phase_g = np.clip(
                    (phase.astype(np.float64) + np.pi) / (2.0 * np.pi), 0, 1
                )
            else:
                phase_g = g
            phase_idx = np.clip(np.round(phase_g * 255).astype(np.int32), 0, 255)
            if mapping.lut is not None:
                table = np.frombuffer(mapping.lut, np.uint8).reshape(256, 4)
            else:  # executor's neutral grayscale ramp
                table = np.empty((256, 4), np.uint8)
                table[:, 0] = table[:, 1] = table[:, 2] = np.arange(256)
                table[:, 3] = 255
            rgba = table[phase_idx].copy()
            if bool(getattr(mapping, "phase_color", False)) and mode != 1:
                rgba[..., :3] = np.clip(
                    rgba[..., :3].astype(np.float64) * g[..., np.newaxis],
                    0.0,
                    255.0,
                ).astype(np.uint8)
            out[y0 : y0 + th, x0 : x0 + tw] = rgba
        return out

    def assert_matches(self, got, ref, tol=2, allow_px=0):
        """``allow_px`` absorbs GPU-f32 vs CPU-f64 LUT-index rounding at
        exact .5 boundaries — only meaningful for discontinuous LUTs, where
        an off-by-one index is a large color diff on a handful of pixels."""
        diff = np.abs(got.astype(np.int32) - ref.astype(np.int32))
        bad = int((np.any(diff > tol, axis=-1)).sum())
        assert bad <= allow_px, f"{bad} px over tol (max diff {diff.max()})"


@pytest.fixture(scope="module")
def scene():
    return Scene()


FULL = (TileInstance((0, 0, 1, 1), (0, 0), (512, 512)),)
WIN_A = (TileInstance((0, 0, 1, 1), (100, 100), (512, 512)),)
MAG = DisplayMapping("magnitude", 0.0, 6.0)


def test_physical_truth_full_residency(scene):
    scene.render(FULL, MAG)
    scene.assert_matches(scene.executor.read_target(), scene.reference(FULL, MAG))


def test_mode_and_levels_switches_render_exactly_with_zero_uploads(scene):
    for mapping in (
        DisplayMapping("phase", -3.2, 3.2),
        DisplayMapping("real", -4.0, 4.0),
        DisplayMapping("imag", -4.0, 4.0),
        DisplayMapping("magnitude", 0.5, 4.0),
    ):
        report = scene.render(FULL, mapping)
        assert report.uploads == 0
        scene.assert_matches(scene.executor.read_target(), scene.reference(FULL, mapping))


def test_log_and_symlog_switches_render_exactly_with_zero_uploads(scene):
    for generation, mapping in enumerate(
        (
            DisplayMapping("magnitude", -2.0, 1.0, scale="log"),
            DisplayMapping(
                "real", -1.0, 1.0, scale="symlog", symlog_constant=0.5
            ),
        ),
        start=6,
    ):
        report = scene.render(FULL, mapping, generation=generation)
        assert report.uploads == 0
        scene.assert_matches(scene.executor.read_target(), scene.reference(FULL, mapping))


def test_window_shift_is_descriptor_only(scene):
    shifted = (TileInstance((0, 0, 1, 1), (101, 100), (512, 512)),)
    assert scene.render(WIN_A, MAG).uploads == 0
    scene.assert_matches(scene.executor.read_target(), scene.reference(WIN_A, MAG))
    assert scene.render(shifted, MAG).uploads == 0
    scene.assert_matches(scene.executor.read_target(), scene.reference(shifted, MAG))


def test_montage_scroll_is_descriptor_only(scene):
    def montage(chunks):
        return tuple(
            TileInstance(
                (0.5 * (i % 2), 0.5 * (i // 2), 0.5, 0.5),
                (cx * 256.0, cy * 256.0),
                (256.0, 256.0),
            )
            for i, (cx, cy) in enumerate(chunks)
        )

    m1 = montage([(0, 0), (1, 0), (0, 1), (1, 1)])
    m2 = montage([(1, 0), (2, 0), (1, 1), (2, 1)])
    assert scene.render(m1, MAG).uploads == 0
    scene.assert_matches(scene.executor.read_target(), scene.reference(m1, MAG))
    assert scene.render(m2, MAG).uploads == 0
    scene.assert_matches(scene.executor.read_target(), scene.reference(m2, MAG))


def test_evicted_page_falls_back_to_pinned_ancestor_then_refills(scene):
    key = scene.key(0, 1, 1)
    report = scene.executor.submit(FrameSubmission(10, (EvictChunk(key),)))
    assert report.evictions == 1
    report = scene.render(WIN_A, MAG, generation=11)
    assert report.uploads == 0
    got = scene.executor.read_target()
    ref = scene.reference(WIN_A, MAG, absent_l0=[(1, 1)])
    scene.assert_matches(got, ref)
    # Never-black: the fallback region may contain DATA black (g=0 maps to
    # LUT[0]), but must not contain MORE black than the ancestor reference —
    # a missing-page hole would.
    black = lambda img: float((img[..., :3].sum(axis=-1) == 0).mean())  # noqa: E731
    assert black(got) <= black(ref) + 1e-6
    report = scene.executor.submit(
        FrameSubmission(
            12,
            (
                EnsureChunkResident(key, scene.l0_page(1, 1)),
                PresentGeneration(12),
            ),
        )
    )
    assert report.uploads == 1
    scene.assert_matches(scene.executor.read_target(), scene.reference(WIN_A, MAG))


def test_reensure_resident_chunk_is_zero_upload(scene):
    report = scene.executor.submit(
        FrameSubmission(
            20, (EnsureChunkResident(scene.key(0, 0, 0), scene.l0_page(0, 0)),)
        )
    )
    assert report.uploads == 0


def test_histogram_exact_over_all_l0_pages(scene):
    keys = tuple(scene.key(0, cx, cy) for cy in range(GRID0) for cx in range(GRID0))
    report = scene.executor.submit(
        FrameSubmission(30, (DispatchHistogram(keys, bins=64, lo=0.0, hi=6.0),))
    )
    (bins,) = report.histograms.values()
    re = scene.plane[..., 0]
    im = scene.plane[..., 1]
    mag = np.sqrt(re * re + im * im, dtype=np.float32)
    cpu = np.bincount(
        np.clip((mag / 6.0 * 64).astype(np.int32), 0, 63).ravel(), minlength=64
    )
    assert int(bins.sum()) == PLANE * PLANE
    assert (bins.astype(np.int64) == cpu.astype(np.int64)).all()


def test_custom_lut_is_applied_exactly_with_zero_uploads(scene):
    rng = np.random.default_rng(3)
    table = rng.integers(0, 256, size=(256, 4), dtype=np.uint8)
    table[:, 3] = 255
    mapping = DisplayMapping("magnitude", 0.0, 6.0, lut=table.tobytes())
    report = scene.render(FULL, mapping, generation=35)
    assert report.uploads == 0  # LUT changes are mapping state, not residency
    scene.assert_matches(
        scene.executor.read_target(), scene.reference(FULL, mapping), allow_px=64
    )
    # And back to the neutral ramp.
    report = scene.render(FULL, MAG, generation=36)
    assert report.uploads == 0
    scene.assert_matches(scene.executor.read_target(), scene.reference(FULL, MAG))


def test_magnitude_modulated_phase_color_matches_cpu_mirror_with_zero_upload_switch(scene):
    from arrayscope.display.shader_mapping import default_phase_lut

    table = np.empty((256, 4), np.uint8)
    table[:, :3] = default_phase_lut()
    table[:, 3] = 255
    mapping = DisplayMapping(
        "magnitude",
        0.0,
        6.0,
        lut=table.tobytes(),
        phase_color=True,
    )

    # The pages were already resident under ordinary magnitude rendering.
    # Phase-color is mapping state only: hue follows phase while brightness
    # follows the same normalized magnitude CPU formula.
    report = scene.render(FULL, mapping, generation=37)
    assert report.uploads == 0
    scene.assert_matches(
        scene.executor.read_target(),
        scene.reference(FULL, mapping),
        allow_px=64,
    )

    report = scene.render(FULL, MAG, generation=38)
    assert report.uploads == 0
    report = scene.render(FULL, mapping, generation=39)
    assert report.uploads == 0


def test_completion_token_is_callable_and_returns(scene):
    report = scene.render(FULL, MAG, generation=40)
    assert callable(report.wait_completed)
    report.wait_completed()  # must not raise; fences the submitted work


# ---- row 3(a) growth oracles: multi-plane binding + honest pools ------------

SP_H, SP_W = 256, 512  # one 1x2 page grid per scalar plane
SP_CANVAS = (512, 512)


def _gray_table():
    table = np.empty((256, 4), np.uint8)
    table[:, 0] = table[:, 1] = table[:, 2] = np.arange(256)
    table[:, 3] = 255
    return table


def _scalar_plane(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    plane = rng.standard_normal((SP_H, SP_W), dtype=np.float32)
    yy, xx = np.mgrid[0:SP_H, 0:SP_W].astype(np.float32)
    return plane + np.sin(xx / 29.0) * 2 + (yy / SP_H) * float(seed)


def _scalar_reference(tiles_with_data, mapping, canvas=SP_CANVAS):
    """CPU mirror for scalar planes: value IS the scalar, mode is ignored."""

    w, h = canvas
    out = np.zeros((h, w, 4), np.uint8)
    out[..., 3] = 255
    table = (
        np.frombuffer(mapping.lut, np.uint8).reshape(256, 4)
        if mapping.lut is not None
        else _gray_table()
    )
    for tile, data in tiles_with_data:
        x0 = int(round(tile.dst_rect[0] * w))
        y0 = int(round(tile.dst_rect[1] * h))
        tw = int(round(tile.dst_rect[2] * w))
        th = int(round(tile.dst_rect[3] * h))
        sx = tile.src_origin[0] + (np.arange(tw) + 0.5) / tw * tile.src_size[0]
        sy = tile.src_origin[1] + (np.arange(th) + 0.5) / th * tile.src_size[1]
        sxg, syg = np.meshgrid(sx, sy)
        cx = np.clip(sxg, 0, data.shape[1] - 1).astype(np.int64)
        cy = np.clip(syg, 0, data.shape[0] - 1).astype(np.int64)
        value = _scale_reference(data[cy, cx], mapping).astype(np.float64)
        g = np.clip((value - mapping.level_lo) / (mapping.level_hi - mapping.level_lo), 0, 1)
        idx = np.clip(np.round(g * 255).astype(np.int32), 0, 255)
        out[y0 : y0 + th, x0 : x0 + tw] = table[idx]
    return out


def _assert_matches(got, ref, tol=2, allow_px=0):
    diff = np.abs(got.astype(np.int32) - ref.astype(np.int32))
    bad = int((np.any(diff > tol, axis=-1)).sum())
    assert bad <= allow_px, f"{bad} px over tol (max diff {diff.max()})"


def _scalar_key(doc, cx, cy):
    return plane_chunk_key(
        doc, "op-live", 0, cx, cy, dtype="float32", representation=SCALAR_R32F
    )


def _scalar_plane_binding(doc):
    return ContentPlane(doc, "op-live", (SP_H, SP_W), max_lod=0, representation=SCALAR_R32F)


def _ensure_scalar_plane(doc, data, *, pinned=False):
    return tuple(
        EnsureChunkResident(
            _scalar_key(doc, cx, 0), data[:, cx * PAGE : (cx + 1) * PAGE], pinned=pinned
        )
        for cx in range(SP_W // PAGE)
    )


class MultiScene:
    """Three bound scalar planes sharing one executor (montage session)."""

    def __init__(self):
        self.planes = {doc: _scalar_plane(i + 1) for i, doc in enumerate("ABC")}
        self.executor = WgpuPlaneExecutor(
            target_size=SP_CANVAS, pool_layers=16, device=_shared_device()
        )
        commands = [BindContentPlanes(tuple(_scalar_plane_binding(doc) for doc in "ABC"))]
        for doc in "ABC":
            commands.extend(_ensure_scalar_plane(doc, self.planes[doc]))
        report = self.executor.submit(FrameSubmission(0, commands))
        assert report.uploads == 3 * (SP_W // PAGE)

    def render(self, tiles, mapping, generation=1):
        report = self.executor.submit(
            FrameSubmission(
                generation,
                (
                    SetDisplayMapping(mapping),
                    UpdateTileInstances(tuple(tiles)),
                    PresentGeneration(generation),
                ),
            )
        )
        assert report.presented
        return report


@pytest.fixture(scope="module")
def multi_scene():
    return MultiScene()


SCALAR_LEVELS = DisplayMapping("real", -4.0, 6.0)


def _montage_tiles(plane_indices):
    """Stack full planes vertically: montage rows selecting bound planes."""

    rows = len(plane_indices)
    return tuple(
        TileInstance(
            (0.0, i / rows, 1.0, 1.0 / rows),
            (0.0, 0.0),
            (float(SP_W), float(SP_H)),
            0,
            plane_index=index,
        )
        for i, index in enumerate(plane_indices)
    )


def test_multi_plane_montage_scroll_is_descriptor_only(multi_scene):
    docs = "ABC"
    view1 = _montage_tiles((0, 1))
    report = multi_scene.render(view1, SCALAR_LEVELS)
    assert report.uploads == 0  # everything resident: descriptor-only
    ref = _scalar_reference(
        [(t, multi_scene.planes[docs[t.plane_index]]) for t in view1], SCALAR_LEVELS
    )
    _assert_matches(multi_scene.executor.read_target(), ref)

    view2 = _montage_tiles((1, 2))  # scroll one plane down across planes
    report = multi_scene.render(view2, SCALAR_LEVELS, generation=2)
    assert report.uploads == 0
    ref = _scalar_reference(
        [(t, multi_scene.planes[docs[t.plane_index]]) for t in view2], SCALAR_LEVELS
    )
    _assert_matches(multi_scene.executor.read_target(), ref)


def test_scalar_plane_ignores_complex_mapping_modes(multi_scene):
    tiles = _montage_tiles((0,))
    baseline = None
    for mode in ("real", "magnitude", "phase", "imag"):
        mapping = DisplayMapping(mode, -4.0, 6.0)
        report = multi_scene.render(tiles, mapping, generation=5)
        assert report.uploads == 0
        got = multi_scene.executor.read_target()
        if baseline is None:
            baseline = got
            ref = _scalar_reference([(tiles[0], multi_scene.planes["A"])], mapping)
            _assert_matches(got, ref)
        else:
            # The scalar value IS the sample: mode switches change nothing.
            assert (got == baseline).all()


def test_histogram_over_scalar_pool_pages(multi_scene):
    keys = tuple(_scalar_key("B", cx, 0) for cx in range(SP_W // PAGE))
    report = multi_scene.executor.submit(
        FrameSubmission(30, (DispatchHistogram(keys, bins=64, lo=-4.0, hi=6.0),))
    )
    (bins,) = report.histograms.values()
    values = multi_scene.planes["B"]
    cpu = np.bincount(
        np.clip(((values - -4.0) / 10.0 * 64).astype(np.int32), 0, 63).ravel(),
        minlength=64,
    )
    assert int(bins.sum()) == SP_H * SP_W
    assert (bins.astype(np.int64) == cpu.astype(np.int64)).all()


def test_dynamic_histogram_discovers_mapped_bounds_and_fences_readback(scene):
    keys = tuple(scene.key(0, cx, cy) for cy in range(GRID0) for cx in range(GRID0))
    report = scene.executor.submit(
        FrameSubmission(
            31,
            (
                DispatchHistogram(
                    keys,
                    bins=64,
                    lo=None,
                    hi=None,
                    mode="real",
                ),
            ),
        )
    )

    report.wait_completed()
    readback = next(iter(report.histograms.values()))
    bins, bounds = readback.resolve()
    values = scene.plane[..., 0]
    assert bounds == pytest.approx((float(values.min()), float(values.max())))
    cpu = np.bincount(
        np.clip(
            ((values - bounds[0]) / (bounds[1] - bounds[0]) * 64).astype(np.int32),
            0,
            63,
        ).ravel(),
        minlength=64,
    )
    assert int(bins.sum()) == values.size
    assert np.array_equal(bins.astype(np.int64), cpu.astype(np.int64))


def test_refined_grade_histogram_supports_the_histogram_widget_bin_cap(scene):
    bins_requested = 500
    keys = tuple(scene.key(0, cx, cy) for cy in range(GRID0) for cx in range(GRID0))
    report = scene.executor.submit(
        FrameSubmission(
            31,
            (
                DispatchHistogram(
                    keys,
                    bins=bins_requested,
                    lo=None,
                    hi=None,
                    mode="real",
                ),
            ),
        )
    )

    report.wait_completed()
    bins, bounds = next(iter(report.histograms.values())).resolve()
    values = scene.plane[..., 0]
    assert bounds == pytest.approx((float(values.min()), float(values.max())))
    cpu = np.bincount(
        np.clip(
            (
                (values - bounds[0])
                / (bounds[1] - bounds[0])
                * bins_requested
            ).astype(np.int32),
            0,
            bins_requested - 1,
        ).ravel(),
        minlength=bins_requested,
    )
    assert bins.shape == (bins_requested,)
    assert int(bins.sum()) == values.size
    normalized_l1 = float(
        np.abs(bins.astype(np.int64) - cpu.astype(np.int64)).sum()
        / max(1, int(cpu.sum()))
    )
    # WGSL performs the bin coordinate in float32; NumPy can place values
    # exactly on one of the 500 boundaries a bin to either side after its
    # scalar promotion. Population and bounds remain exact, and the tiny
    # distribution delta is far below the G6 5% histogram oracle.
    assert normalized_l1 <= 0.001


def test_dynamic_complex_histogram_matches_cpu_over_same_resident_pages(scene):
    keys = tuple(scene.key(0, cx, cy) for cy in range(GRID0) for cx in range(GRID0))
    report = scene.executor.submit(
        FrameSubmission(
            31,
            (
                DispatchHistogram(
                    keys,
                    bins=64,
                    lo=None,
                    hi=None,
                    mode="magnitude",
                ),
            ),
        )
    )

    report.wait_completed()
    bins, bounds = next(iter(report.histograms.values())).resolve()
    values = np.sqrt(
        scene.plane[..., 0] ** 2 + scene.plane[..., 1] ** 2,
        dtype=np.float32,
    )
    assert bounds == pytest.approx((float(values.min()), float(values.max())))
    cpu = np.bincount(
        np.clip(
            ((values - bounds[0]) / (bounds[1] - bounds[0]) * 64).astype(np.int32),
            0,
            63,
        ).ravel(),
        minlength=64,
    )
    assert int(bins.sum()) == values.size
    normalized_l1 = float(
        np.abs(bins.astype(np.int64) - cpu.astype(np.int64)).sum()
        / max(1, int(cpu.sum()))
    )
    assert normalized_l1 <= HISTOGRAM_NORMALIZED_L1_TOLERANCE


def test_dynamic_histogram_excludes_padding_and_weights_source_coverage():
    shape = (17, 23)
    values = np.linspace(-3.0, 7.0, shape[0] * shape[1], dtype=np.float32).reshape(shape)
    page = np.zeros((PAGE, PAGE), dtype=np.float32)
    page[: shape[0], : shape[1]] = values
    executor = WgpuPlaneExecutor(
        pool_layers={SCALAR_R32F: 2}, device=_shared_device()
    )
    key = plane_chunk_key(
        "boundary-doc",
        "op-live",
        0,
        0,
        0,
        dtype="float32",
        representation=SCALAR_R32F,
        plane_shape=shape,
    )
    report = executor.submit(
        FrameSubmission(
            32,
            (
                BindContentPlanes(
                    (ContentPlane("boundary-doc", "op-live", shape, representation=SCALAR_R32F),)
                ),
                EnsureChunkResident(key, page),
                DispatchHistogram((key,), bins=64, lo=None, hi=None, mode="real"),
            ),
        )
    )

    report.wait_completed()
    readback = next(iter(report.histograms.values()))
    bins, bounds = readback.resolve()
    assert bounds == pytest.approx((float(values.min()), float(values.max())))
    assert int(bins.sum()) == values.size


def test_dynamic_histogram_uses_windowable_rgb_scalar_signal():
    values = np.linspace(1.0, 9.0, PAGE * PAGE, dtype=np.float32).reshape(PAGE, PAGE)
    page = np.zeros((PAGE, PAGE, 4), dtype=np.float32)
    page[..., :3] = 0.5
    page[..., 3] = values
    executor = WgpuPlaneExecutor(
        pool_layers={RGB_WINDOWED_RGBA32F: 2}, device=_shared_device()
    )
    key = plane_chunk_key(
        "windowed-rgb-doc",
        "op-live",
        0,
        0,
        0,
        dtype="float32",
        representation=RGB_WINDOWED_RGBA32F,
        plane_shape=values.shape,
    )
    report = executor.submit(
        FrameSubmission(
            33,
            (
                BindContentPlanes(
                    (
                        ContentPlane(
                            "windowed-rgb-doc",
                            "op-live",
                            values.shape,
                            representation=RGB_WINDOWED_RGBA32F,
                        ),
                    )
                ),
                EnsureChunkResident(key, page),
                DispatchHistogram((key,), bins=64, lo=None, hi=None, mode="real"),
            ),
        )
    )

    report.wait_completed()
    bins, bounds = next(iter(report.histograms.values())).resolve()
    assert bounds == pytest.approx((1.0, 9.0))
    assert int(bins.sum()) == values.size


def test_tile_plane_index_outside_bound_planes_is_loud(multi_scene):
    with pytest.raises(ValueError, match="plane_index"):
        multi_scene.executor.submit(
            FrameSubmission(
                40, (UpdateTileInstances(_montage_tiles((3,))),)
            )
        )


def test_plane_rebind_keeps_warm_residency_zero_upload():
    planes = {doc: _scalar_plane(10 + i) for i, doc in enumerate("AB")}
    executor = WgpuPlaneExecutor(
            target_size=SP_CANVAS, pool_layers=16, device=_shared_device()
        )
    tiles = _montage_tiles((0,))

    def commit(doc, generation):
        return executor.submit(
            FrameSubmission(
                generation,
                (
                    BindContentPlanes((_scalar_plane_binding(doc),)),
                    *_ensure_scalar_plane(doc, planes[doc]),
                    SetDisplayMapping(SCALAR_LEVELS),
                    UpdateTileInstances(tiles),
                    PresentGeneration(generation),
                ),
            )
        )

    assert commit("A", 1).uploads == SP_W // PAGE
    assert commit("B", 2).uploads == SP_W // PAGE
    _assert_matches(
        executor.read_target(), _scalar_reference([(tiles[0], planes["B"])], SCALAR_LEVELS)
    )
    # Scroll back: A's chunks stayed warm in the page table while unbound.
    report = commit("A", 3)
    assert report.uploads == 0
    _assert_matches(
        executor.read_target(), _scalar_reference([(tiles[0], planes["A"])], SCALAR_LEVELS)
    )


def test_rgb8_pool_bypasses_levels_and_lut():
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, size=(PAGE, PAGE, 3), dtype=np.uint8)
    executor = WgpuPlaneExecutor(
        target_size=(PAGE, PAGE), pool_layers={"rgb8": 4}, device=_shared_device()
    )
    key = plane_chunk_key("doc-rgb", "op-live", 0, 0, 0, dtype="uint8", representation=RGB8)
    lut = rng.integers(0, 256, size=(256, 4), dtype=np.uint8).tobytes()
    tiles = (TileInstance((0, 0, 1, 1), (0, 0), (PAGE, PAGE), 0),)
    report = executor.submit(
        FrameSubmission(
            1,
            (
                BindContentPlanes(
                    (ContentPlane("doc-rgb", "op-live", (PAGE, PAGE), representation=RGB8),)
                ),
                EnsureChunkResident(key, rgb),
                # Hostile levels + LUT: display-ready RGB must ignore both.
                SetDisplayMapping(DisplayMapping("magnitude", 0.25, 0.26, lut=lut)),
                UpdateTileInstances(tiles),
                PresentGeneration(1),
            ),
        )
    )
    assert report.uploads == 1
    got = executor.read_target()
    assert (got[..., :3] == rgb).all()
    assert (got[..., 3] == 255).all()
    # Re-ensure + re-present: identical bytes, zero uploads.
    report = executor.submit(
        FrameSubmission(2, (EnsureChunkResident(key, rgb), PresentGeneration(2)))
    )
    assert report.uploads == 0
    assert (executor.read_target()[..., :3] == rgb).all()


def test_per_pool_eviction_respects_budget_and_pins():
    executor = WgpuPlaneExecutor(
        target_size=(PAGE, PAGE),
        pool_layers={"scalar_r32f": 2, "complex_rg32f": 1},
        device=_shared_device(),
    )
    page = np.zeros((PAGE, PAGE), np.float32)
    complex_key = plane_chunk_key("doc-c", "op-live", 0, 0, 0)
    scalar_keys = [
        plane_chunk_key("doc-s", "op-live", 0, cx, 0, dtype="float32", representation=SCALAR_R32F)
        for cx in range(3)
    ]
    # The complex chunk becomes the LRU-oldest resident entry overall.
    report = executor.submit(
        FrameSubmission(
            1,
            (
                EnsureChunkResident(complex_key, np.zeros((PAGE, PAGE, 2), np.float32)),
                EnsureChunkResident(scalar_keys[0], page, pinned=True),
                EnsureChunkResident(scalar_keys[1], page),
                EnsureChunkResident(scalar_keys[2], page),
            ),
        )
    )
    assert report.uploads == 4
    resident = set(executor.page_table.resident_keys())
    # Scalar pool budget 2: the unpinned scalar LRU was evicted; the pinned
    # page and the (older) complex chunk in the OTHER pool were untouched.
    assert scalar_keys[0] in resident and scalar_keys[2] in resident
    assert scalar_keys[1] not in resident
    assert complex_key in resident
    assert executor.pool_free_layers(SCALAR_R32F) == 0
    assert executor.pool_free_layers(COMPLEX_RG32F) == 0

    # Pin everything in the scalar pool: overflow must fail loudly.
    executor.page_table.pin(scalar_keys[2])
    with pytest.raises(RuntimeError, match="pinned"):
        executor.submit(
            FrameSubmission(
                2,
                (
                    EnsureChunkResident(
                        plane_chunk_key(
                            "doc-s", "op-live", 0, 3, 0,
                            dtype="float32", representation=SCALAR_R32F,
                        ),
                        page,
                    ),
                ),
            )
        )


def test_unbudgeted_pool_rejects_residency_loudly():
    executor = WgpuPlaneExecutor(
        target_size=(PAGE, PAGE), pool_layers={"scalar_r32f": 2}, device=_shared_device()
    )
    with pytest.raises(RuntimeError, match="no layer budget"):
        executor.submit(
            FrameSubmission(
                1,
                (
                    EnsureChunkResident(
                        plane_chunk_key("doc", "op", 0, 0, 0),
                        np.zeros((PAGE, PAGE, 2), np.float32),
                    ),
                ),
            )
        )
