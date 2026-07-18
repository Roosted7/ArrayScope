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
    DispatchHistogram,
    DisplayMapping,
    EnsureChunkResident,
    EvictChunk,
    FrameSubmission,
    PresentGeneration,
    SetDisplayMapping,
    TileInstance,
    UpdateTileInstances,
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


def _adapter_available() -> bool:
    try:
        wgpu.gpu.request_adapter_sync(power_preference="low-power")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _adapter_available(), reason="no wgpu adapter on this machine"
)


def _plane() -> np.ndarray:
    rng = np.random.default_rng(42)
    re = rng.standard_normal((PLANE, PLANE), dtype=np.float32)
    im = rng.standard_normal((PLANE, PLANE), dtype=np.float32)
    yy, xx = np.mgrid[0:PLANE, 0:PLANE].astype(np.float32)
    re += np.sin(xx / 37.0) * 2 + (xx / PLANE)
    im += np.cos(yy / 23.0) * 2
    return np.stack([re, im], axis=-1)


class Scene:
    """Executor + data + CPU mirror shared by every oracle."""

    def __init__(self):
        self.plane = _plane()
        p = self.plane
        self.plane_l1 = (p[0::2, 0::2] + p[1::2, 0::2] + p[0::2, 1::2] + p[1::2, 1::2]) / 4.0
        self.executor = WgpuPlaneExecutor((PLANE, PLANE), max_lod=1, target_size=CANVAS)
        self.doc, self.op = "doc-1", "op-identity"

        commands = []
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
            g = np.clip((x.astype(np.float64) - lo) / (hi - lo), 0, 1)
            rgba = np.stack(
                [g * 255, g * g * 255, np.sqrt(g) * 255, np.full_like(g, 255)], axis=-1
            )
            out[y0 : y0 + th, x0 : x0 + tw] = np.round(rgba).astype(np.uint8)
        return out

    def assert_matches(self, got, ref, tol=2):
        diff = np.abs(got.astype(np.int32) - ref.astype(np.int32))
        assert int((diff > tol).sum()) == 0, f"max diff {diff.max()}"


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
    scene.assert_matches(got, scene.reference(WIN_A, MAG, absent_l0=[(1, 1)]))
    assert float((got[..., :3].sum(axis=-1) == 0).mean()) == 0.0  # never black
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


def test_completion_token_is_callable_and_returns(scene):
    report = scene.render(FULL, MAG, generation=40)
    assert callable(report.wait_completed)
    report.wait_completed()  # must not raise; fences the submitted work
