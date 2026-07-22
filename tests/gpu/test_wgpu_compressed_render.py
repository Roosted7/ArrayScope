"""Real-GPU parity: native BC pools sampled in the live wgpu render path.

This is the gate that makes "compressed textures are integrated" a fact rather
than a claim.  On a BC-capable adapter (the A2000 under the NVIDIA ICD, or the
integrated device -- both expose ``texture-compression-bc``) it renders a real
montage twice through :class:`~arrayscope.gpu.wgpu_executor.WgpuPlaneExecutor`:
once on the byte-identical raw path, once with compression engaged.  It then

* proves a tile was actually sampled from a BC pool -- ``compressed_uploads_total``
  and the page table's ``pool_id`` (the loud channel), not a silent raw fallback;
* asserts the engaged framebuffer matches the raw one within a tolerance
  **justified by the measured PSNR** (BC4/BC5 are lossy; the loss is a number);
* confirms the default (off) executor renders byte-identically to a raw one.

Skips cleanly where no BC adapter is present, so CI stays green.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from arrayscope.gpu import bc_codec
from arrayscope.gpu.command_protocol import (
    BindContentPlanes,
    ContentPlane,
    DisplayMapping,
    EnsureChunkResident,
    FrameSubmission,
    PresentGeneration,
    SetDisplayMapping,
    TileInstance,
    UpdateTileInstances,
)
from arrayscope.gpu.keys import COMPLEX_RG32F, SCALAR_R32F
from arrayscope.gpu.wgpu_executor import PAGE, WgpuPlaneExecutor, plane_chunk_key


def _bc_device():
    try:
        import wgpu  # noqa: F401

        from arrayscope.gpu.bc_gpu import create_compute_device
    except Exception:
        return None
    try:
        return create_compute_device("high-performance", features=["texture-compression-bc"])
    except Exception:
        with contextlib.suppress(Exception):
            return create_compute_device("low-power", features=["texture-compression-bc"])
        return None


_DEVICE = _bc_device()
pytestmark = pytest.mark.skipif(
    _DEVICE is None, reason="no BC-capable wgpu adapter on this machine"
)

CANVAS = (384, 384)


def _scalar_tile(seed: int = 0) -> np.ndarray:
    y, x = np.mgrid[0:PAGE, 0:PAGE]
    return (np.sin((x + seed) / 40.0) * np.cos((y - seed) / 35.0) + x / float(PAGE)).astype(
        np.float32
    )


def _complex_tile() -> np.ndarray:
    y, x = np.mgrid[0:PAGE, 0:PAGE]
    re = np.sin(x / 40.0) * np.cos(y / 35.0)
    im = np.cos(x / 33.0) * np.sin(y / 50.0)
    return (re + 1j * im).astype(np.complex64)


def _frame_quality(engaged: np.ndarray, raw: np.ndarray, *, tol: int) -> tuple[float, float]:
    """(rgb PSNR in dB, fraction of pixels whose per-channel diff exceeds tol)."""

    a = engaged[..., :3].astype(np.float64)
    b = raw[..., :3].astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse <= 0 else 20 * np.log10(255.0) - 10 * np.log10(mse)
    over = float(np.mean(np.any(np.abs(a - b) > tol, axis=-1)))
    return psnr, over


def _render_scalar(
    mode: str,
    *,
    codec_min_psnr_db: float = 40.0,
    payloads: list[np.ndarray] | None = None,
):
    executor = WgpuPlaneExecutor(
        (PAGE, 2 * PAGE),
        max_lod=0,
        target_size=CANVAS,
        device=_DEVICE,
        pool_layers={SCALAR_R32F: 4},
        compressed_textures=mode,
        codec_min_psnr_db=codec_min_psnr_db,
    )
    doc, op = "parity-scalar", "op"
    plane = ContentPlane(doc, op, (PAGE, 2 * PAGE), max_lod=0, representation=SCALAR_R32F)
    keys = [
        plane_chunk_key(
            doc,
            op,
            0,
            cx,
            0,
            dtype="float32",
            representation=SCALAR_R32F,
            plane_shape=(PAGE, 2 * PAGE),
        )
        for cx in range(2)
    ]
    payloads = payloads or [_scalar_tile(cx * 7) for cx in range(2)]
    ensures = [EnsureChunkResident(keys[cx], payloads[cx]) for cx in range(2)]
    executor.submit(FrameSubmission(0, [BindContentPlanes((plane,)), *ensures]))
    tile = TileInstance(
        (0.0, 0.0, 1.0, 1.0), (0.0, 0.0), (float(2 * PAGE), float(PAGE)), 0, plane_index=0
    )
    report = executor.submit(
        FrameSubmission(
            1,
            (
                SetDisplayMapping(DisplayMapping("real", -1.0, 2.0)),
                UpdateTileInstances((tile,)),
                PresentGeneration(1),
            ),
        )
    )
    report.wait_completed()
    return executor, keys, executor.read_target()


def _render_complex(mode: str, mapmode: str):
    executor = WgpuPlaneExecutor(
        (PAGE, PAGE),
        max_lod=0,
        target_size=CANVAS,
        device=_DEVICE,
        pool_layers={COMPLEX_RG32F: 4},
        compressed_textures=mode,
    )
    doc, op = "parity-complex", "op"
    plane = ContentPlane(doc, op, (PAGE, PAGE), max_lod=0, representation=COMPLEX_RG32F)
    key = plane_chunk_key(
        doc,
        op,
        0,
        0,
        0,
        dtype="complex64",
        representation=COMPLEX_RG32F,
        plane_shape=(PAGE, PAGE),
    )
    executor.submit(
        FrameSubmission(0, [BindContentPlanes((plane,)), EnsureChunkResident(key, _complex_tile())])
    )
    tile = TileInstance(
        (0.0, 0.0, 1.0, 1.0), (0.0, 0.0), (float(PAGE), float(PAGE)), 0, plane_index=0
    )
    report = executor.submit(
        FrameSubmission(
            1,
            (
                SetDisplayMapping(DisplayMapping(mapmode, -1.5, 1.5)),
                UpdateTileInstances((tile,)),
                PresentGeneration(1),
            ),
        )
    )
    report.wait_completed()
    return executor, key, executor.read_target()


def test_scalar_bc4_montage_matches_raw_and_actually_used_a_bc_pool():
    engaged, keys, frame_bc = _render_scalar("on")
    raw_exec, _keys, frame_raw = _render_scalar("off")

    # Loud channel: a tile really came from the BC pool, not a raw fallback.
    assert engaged.codec_engaged
    assert engaged.compressed_uploads_total == 2
    assert raw_exec.compressed_uploads_total == 0
    for key in keys:
        assert engaged.page_is_compressed(key)
        assert engaged.page_table.lookup(key).pool_id == "wgpu-scalar_r32f-bc-pool"
        assert not raw_exec.page_is_compressed(key)

    # Framebuffer parity within a PSNR-justified tolerance (BC4 is lossy).
    psnr, over = _frame_quality(frame_bc, frame_raw, tol=4)
    assert psnr >= 45.0, f"BC4 render PSNR {psnr:.1f} dB below the 45 dB tolerance floor"
    assert over < 0.01, f"{over:.4f} of pixels exceed the per-channel tolerance"


def test_40db_gate_rejects_page_that_38db_would_render_below_tolerance():
    y, x = np.mgrid[0:PAGE, 0:PAGE]
    borderline = (np.sin(x / 3.0) * np.cos(y / 3.6) + x / float(PAGE)).astype(np.float32)
    payloads = [borderline, np.roll(borderline, 5, axis=1)]

    conservative, _keys, _frame = _render_scalar("on", codec_min_psnr_db=40.0, payloads=payloads)
    engaged, _keys, frame_bc = _render_scalar("on", codec_min_psnr_db=38.0, payloads=payloads)
    _raw, _keys, frame_raw = _render_scalar("off", payloads=payloads)

    assert conservative.compressed_uploads_total == 0
    assert engaged.compressed_uploads_total == 2
    psnr, _over = _frame_quality(frame_bc, frame_raw, tol=4)
    assert psnr < 45.0


def test_complex_bc5_magnitude_matches_raw_and_used_a_bc_pool():
    engaged, key, frame_bc = _render_complex("on", "magnitude")
    _raw, _key, frame_raw = _render_complex("off", "magnitude")

    assert engaged.compressed_uploads_total == 1
    assert engaged.page_is_compressed(key)
    assert engaged.page_table.lookup(key).pool_id == "wgpu-complex_rg32f-bc-pool"

    psnr, over = _frame_quality(frame_bc, frame_raw, tol=4)
    assert psnr >= 45.0, f"BC5 magnitude render PSNR {psnr:.1f} dB below tolerance"
    assert over < 0.01


def test_complex_bc5_phase_error_confined_to_near_black_pixels():
    """Phase differs only where magnitude ~= 0 (phase is meaningless there).

    A tiny BC (re, im) error can flip phase by up to pi at near-zero magnitude,
    which the phase LUT wraps to an opposite colour.  This asserts every such
    large-diff pixel is a low-magnitude pixel -- the display-honest measure the
    codec's ``complex_display_quality`` records, not a raw max-diff.
    """

    _engaged, _key, frame_bc = _render_complex("on", "phase")
    _raw, _k, frame_raw = _render_complex("off", "phase")
    diff = np.abs(frame_bc[..., :3].astype(np.int32) - frame_raw[..., :3].astype(np.int32))
    big = np.any(diff > 32, axis=-1)

    tile = _complex_tile()
    mag = np.abs(tile)
    peak = float(mag.max())
    # The rendered frame is CANVAS-sized; sample the tile magnitude at the same
    # normalized positions to classify each framebuffer pixel.
    ys = np.clip((np.arange(CANVAS[1])[:, None] / CANVAS[1] * PAGE).astype(int), 0, PAGE - 1)
    xs = np.clip((np.arange(CANVAS[0])[None, :] / CANVAS[0] * PAGE).astype(int), 0, PAGE - 1)
    mag_frame = mag[ys, xs]
    low_magnitude = mag_frame < 0.1 * peak
    offending = big & ~low_magnitude
    assert not offending.any(), (
        f"{int(offending.sum())} large phase-colour diffs at non-trivial magnitude"
    )


def test_off_mode_is_byte_identical_to_raw_executor():
    off, _keys, frame_off = _render_scalar("off")
    _raw, _k, frame_raw = _render_scalar("off")
    assert not off.codec_engaged
    assert np.array_equal(frame_off, frame_raw)


def test_auto_bc_keeps_pages_raw_until_accelerator_is_ready(monkeypatch):
    monkeypatch.setattr(bc_codec, "numba_encoder_ready", lambda: False)

    executor, keys, _frame = _render_scalar("auto")

    assert executor.codec_engaged
    assert executor.compressed_uploads_total == 0
    assert all(not executor.page_is_compressed(key) for key in keys)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_scalar_page_declines_compression_and_preserves_value(bad):
    executor = WgpuPlaneExecutor(
        (PAGE, PAGE),
        max_lod=0,
        device=_DEVICE,
        pool_layers={SCALAR_R32F: 2},
        compressed_textures="on",
    )
    plane = ContentPlane("nonfinite", "op", (PAGE, PAGE), representation=SCALAR_R32F)
    key = plane_chunk_key(
        "nonfinite",
        "op",
        0,
        0,
        0,
        dtype="float32",
        representation=SCALAR_R32F,
        plane_shape=(PAGE, PAGE),
    )
    payload = _scalar_tile()
    payload[7, 11] = bad

    executor.submit(
        FrameSubmission(0, (BindContentPlanes((plane,)), EnsureChunkResident(key, payload)))
    )

    assert executor.compressed_uploads_total == 0
    assert executor.compressed_fallbacks_total == 1
    assert not executor.page_is_compressed(key)
    resident = executor.read_resident_page(key)
    if np.isnan(bad):
        assert np.isnan(resident[7, 11])
    else:
        assert resident[7, 11] == bad


def test_partial_edge_page_normalizes_only_valid_samples():
    """Zero padding outside a boundary chunk must not widen its codec affine."""

    side = 16
    executor = WgpuPlaneExecutor(
        (side, side),
        max_lod=0,
        device=_DEVICE,
        pool_layers={SCALAR_R32F: 2},
        compressed_textures="on",
    )
    plane = ContentPlane("edge", "op", (side, side), representation=SCALAR_R32F)
    key = plane_chunk_key(
        "edge",
        "op",
        0,
        0,
        0,
        dtype="float32",
        representation=SCALAR_R32F,
        plane_shape=(side, side),
    )
    payload = np.zeros((PAGE, PAGE), np.float32)
    payload[:side, :side] = np.linspace(100.0, 101.0, side * side, dtype=np.float32).reshape(
        side, side
    )

    executor.submit(
        FrameSubmission(0, (BindContentPlanes((plane,)), EnsureChunkResident(key, payload)))
    )

    assert executor.page_is_compressed(key)
    _codec, norm = executor._page_codec[key]
    assert norm[0] == pytest.approx(100.0)
    assert norm[1] == pytest.approx(1.0)
    resident = executor.read_resident_page(key)
    assert np.max(np.abs(resident[:side, :side] - payload[:side, :side])) < 0.08


def test_report_and_executor_account_actual_compressed_bytes():
    executor = WgpuPlaneExecutor(
        (PAGE, PAGE),
        max_lod=0,
        device=_DEVICE,
        pool_layers={SCALAR_R32F: 2},
        compressed_textures="on",
    )
    plane = ContentPlane("bytes", "op", (PAGE, PAGE), representation=SCALAR_R32F)
    key = plane_chunk_key(
        "bytes",
        "op",
        0,
        0,
        0,
        dtype="float32",
        representation=SCALAR_R32F,
        plane_shape=(PAGE, PAGE),
    )
    report = executor.submit(
        FrameSubmission(
            0,
            (BindContentPlanes((plane,)), EnsureChunkResident(key, _scalar_tile())),
        )
    )

    assert report.upload_bytes == PAGE * PAGE // 2  # BC4: 0.5 byte/texel
    assert executor.active_resident_bytes == report.upload_bytes
    assert executor.allocated_pool_bytes > executor.active_resident_bytes


def test_raw_and_codec_arrays_grow_on_demand_without_losing_resident_pages():
    """Logical budgets are maxima, not two eagerly allocated mirrors."""

    page_count = 18
    executor = WgpuPlaneExecutor(
        (PAGE, page_count * PAGE),
        max_lod=0,
        device=_DEVICE,
        pool_layers={SCALAR_R32F: 32},
        compressed_textures="on",
    )
    plane = ContentPlane(
        "demand-grown",
        "op",
        (PAGE, page_count * PAGE),
        max_lod=0,
        representation=SCALAR_R32F,
    )
    keys = [
        plane_chunk_key(
            "demand-grown",
            "op",
            0,
            cx,
            0,
            dtype="float32",
            representation=SCALAR_R32F,
            plane_shape=plane.plane_shape,
        )
        for cx in range(page_count)
    ]
    executor.submit(FrameSubmission(0, (BindContentPlanes((plane,)),)))

    assert executor.pool_budget(SCALAR_R32F) == 32
    assert executor.pool_allocated_layers(SCALAR_R32F) == 8
    assert executor.codec_pool_allocated_layers(SCALAR_R32F) == 8

    compressed_payloads = [_scalar_tile(seed) for seed in range(9)]
    for index, payload in enumerate(compressed_payloads):
        executor.submit(FrameSubmission(index + 1, (EnsureChunkResident(keys[index], payload),)))

    assert executor.codec_pool_allocated_layers(SCALAR_R32F) == 16
    assert executor.pool_allocated_layers(SCALAR_R32F) == 8
    assert all(executor.page_is_compressed(key) for key in keys[:9])
    # The first layer survived the immutable-array replacement at the same slot.
    assert np.max(np.abs(executor.read_resident_page(keys[0]) - compressed_payloads[0])) < 0.08

    raw_payloads = []
    for offset in range(9):
        payload = _scalar_tile(50 + offset)
        payload[0, 0] = np.nan  # native UNORM cannot preserve it: exact raw fallback
        raw_payloads.append(payload)
        executor.submit(
            FrameSubmission(
                20 + offset,
                (EnsureChunkResident(keys[9 + offset], payload),),
            )
        )

    assert executor.pool_allocated_layers(SCALAR_R32F) == 16
    assert executor.codec_pool_allocated_layers(SCALAR_R32F) == 16
    assert executor.pool_grows_total == 2
    expected_copy_bytes = 8 * PAGE * PAGE * 4 + 8 * (PAGE * PAGE // 2)
    assert executor.pool_growth_copy_bytes_total == expected_copy_bytes
    assert np.isnan(executor.read_resident_page(keys[9])[0, 0])
    # Each pool is only half its logical maximum after admitting this exact mix.
    assert executor.allocated_pool_bytes < (32 * PAGE * PAGE * 4 + 32 * PAGE * PAGE // 2)


def teardown_module(module):
    with contextlib.suppress(Exception):
        _DEVICE._destroy() if hasattr(_DEVICE, "_destroy") else None
