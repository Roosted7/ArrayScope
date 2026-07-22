"""Matched live-path benchmark for G7 host and WGPU texture compression.

The earlier G7 tools answer useful component questions (codec ratio and decoded
quality), but they do not time the synchronous encode/quality gate in
``WgpuPlaneExecutor.submit`` or account for the parallel raw + compressed texture
pools.  The host-cache model likewise charged a synthetic FFT to the display
cache and compared it with a production topology that used a different budget.

This tool measures the production seams without inventing an avoided operation:

* host cache: real ``DisplayImage`` payloads, one matched TOTAL byte budget,
  forward admission + reverse revisit, unique keys, combined raw/tier bytes, and
  maximum synchronous put latency;
* WGPU: executor construction, cold resident upload (including CPU encode and
  reference quality decode), fence, resident histogram, compressed-source LOD,
  actual submitted bytes, active resident bytes, and configured pool allocation.

Run each texture/host mode in a fresh process and compare the JSON records.  A
component ratio is not a product win unless cold latency, callback bars, pixels,
and final levels/LOD also pass the normal real-Wayland journey gate.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from arrayscope.display.slice_engine import DisplayImage
from arrayscope.gpu.command_protocol import (
    BindContentPlanes,
    ContentPlane,
    DispatchHistogram,
    DisplayMapping,
    EnsureChunkResident,
    FrameSubmission,
    GenerateLodPages,
    SetDisplayMapping,
)
from arrayscope.gpu.keys import COMPLEX_RG32F, SCALAR_R32F
from arrayscope.gpu.wgpu_executor import PAGE, WgpuPlaneExecutor, plane_chunk_key
from arrayscope.operations.evaluator import _build_array_cache

DEFAULT_DATA_PATH = Path(
    "/home/thomas/projects/ArrayScope/data/_WIPDelRec-tT2_20260223150234_14.nii"
)


def _load_tiles(path: Path, count: int, representation: str) -> list[np.ndarray]:
    import nibabel as nib

    volume = np.asanyarray(nib.load(str(path)).dataobj)
    while volume.ndim > 3:
        volume = volume[..., 0]
    axis = int(np.argmin(volume.shape))
    volume = np.moveaxis(volume, axis, 0).astype(np.float32, copy=False)
    tiles: list[np.ndarray] = []
    indices = np.linspace(0, volume.shape[0] - 1, count, dtype=int)
    for index in indices:
        plane = np.asarray(volume[index], dtype=np.float32)
        r0 = max(0, (plane.shape[0] - PAGE) // 2)
        c0 = max(0, (plane.shape[1] - PAGE) // 2)
        tile = plane[r0 : r0 + PAGE, c0 : c0 + PAGE]
        tile = np.pad(
            tile,
            ((0, PAGE - tile.shape[0]), (0, PAGE - tile.shape[1])),
            mode="edge",
        ).astype(np.float32, copy=False)
        if representation == COMPLEX_RG32F:
            imag = np.roll(tile, 3, axis=1) * 0.6 - np.roll(tile, 2, axis=0) * 0.4
            tile = (tile + 1j * imag).astype(np.complex64)
        tiles.append(np.ascontiguousarray(tile))
    return tiles


def _host_cache_case(tiles: list[np.ndarray], codec: str, budget_chunks: int) -> dict:
    payloads = [
        DisplayImage(data=tile, semantic_data=tile, lod_source_data=tile, level_data=tile)
        for tile in tiles
    ]
    chunk_bytes = int(payloads[0].data.nbytes)
    total_budget = int(budget_chunks) * chunk_bytes
    cache = _build_array_cache(total_budget, max(32, len(payloads) * 2), codec)
    put_ms: list[float] = []
    for index, payload in enumerate(payloads):
        start = time.perf_counter()
        cache.put((index,), payload)
        put_ms.append((time.perf_counter() - start) * 1000.0)
    start = time.perf_counter()
    revisited = [
        index for index in reversed(range(len(payloads))) if cache.get((index,)) is not None
    ]
    revisit_ms = (time.perf_counter() - start) * 1000.0

    tier = getattr(cache, "tier", None)
    raw = getattr(cache, "raw", cache)
    raw_keys = set(getattr(getattr(raw, "_cache", None), "_items", {}))
    tier_keys = set(getattr(getattr(tier, "_cache", None), "_items", {})) if tier else set()
    diagnostics = cache.diagnostics()
    return {
        "codec": codec,
        "total_budget_bytes": total_budget,
        "reported_max_bytes": int(cache.max_bytes),
        "combined_used_bytes": int(cache.bytes_used),
        "raw_budget_bytes": int(raw.max_bytes),
        "tier_budget_bytes": 0 if tier is None else int(tier.max_bytes),
        "raw_entries": len(raw_keys),
        "tier_entries": len(tier_keys),
        "unique_resident_keys": len(raw_keys | tier_keys),
        "overlapping_keys": len(raw_keys & tier_keys),
        "revisited_keys": len(revisited),
        "tier_recoveries": int(getattr(diagnostics, "tier_recoveries", 0)),
        "admit_total_ms": float(sum(put_ms)),
        "admit_max_ms": float(max(put_ms, default=0.0)),
        "revisit_total_ms": float(revisit_ms),
    }


def _device(power: str):
    import wgpu
    from wgpu.backends.wgpu_native.extras import set_instance_extras

    with contextlib.suppress(RuntimeError):
        set_instance_extras(backends=["Vulkan"])
    adapter = wgpu.gpu.request_adapter_sync(power_preference=power)
    available = {str(feature) for feature in adapter.features}
    wanted = [
        feature
        for feature in ("texture-compression-bc", "texture-compression-astc")
        if feature in available
    ]
    return adapter.request_device_sync(required_features=wanted)


def _payload_for_executor(tile: np.ndarray, representation: str) -> np.ndarray:
    if representation == SCALAR_R32F:
        return np.ascontiguousarray(tile, dtype=np.float32)
    return np.ascontiguousarray(np.stack((tile.real, tile.imag), axis=-1), dtype=np.float32)


def _gpu_case(tiles: list[np.ndarray], representation: str, texture_codec: str, power: str) -> dict:
    device = _device(power)
    info = dict(device.adapter.info)
    mode = "off" if texture_codec == "off" else "auto"
    shape = (PAGE, len(tiles) * PAGE)
    doc, operation = "g7-live", "identity"

    start = time.perf_counter()
    executor = WgpuPlaneExecutor(
        shape,
        max_lod=0,
        target_size=(128, 128),
        device=device,
        pool_layers={representation: len(tiles) + 4},
        compressed_textures=mode,
    )
    construct_ms = (time.perf_counter() - start) * 1000.0
    plane = ContentPlane(doc, operation, shape, max_lod=0, representation=representation)
    dtype = "float32" if representation == SCALAR_R32F else "complex64"
    keys = tuple(
        plane_chunk_key(
            doc,
            operation,
            0,
            index,
            0,
            dtype=dtype,
            representation=representation,
            plane_shape=shape,
        )
        for index in range(len(tiles))
    )
    commands = (
        BindContentPlanes((plane,)),
        *(
            EnsureChunkResident(key, _payload_for_executor(tile, representation))
            for key, tile in zip(keys, tiles, strict=True)
        ),
    )
    start = time.perf_counter()
    upload_report = executor.submit(FrameSubmission(0, commands))
    submit_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    upload_report.wait_completed()
    fence_ms = (time.perf_counter() - start) * 1000.0

    display_mode = "real" if representation == SCALAR_R32F else "magnitude"
    start = time.perf_counter()
    histogram = executor.submit(
        FrameSubmission(
            1,
            (
                SetDisplayMapping(DisplayMapping(display_mode, 0.0, 1.0)),
                DispatchHistogram(
                    keys,
                    bins=256,
                    lo=None,
                    hi=None,
                    mode=display_mode,
                    scale="linear",
                    symlog_constant=0.0,
                ),
            ),
        )
    )
    histogram.wait_completed()
    result = histogram.histograms[1]
    if hasattr(result, "resolve"):
        result.resolve()
    histogram_ms = (time.perf_counter() - start) * 1000.0

    return {
        "texture_codec": texture_codec,
        "representation": representation,
        "adapter": str(info.get("device", "")),
        "adapter_type": str(info.get("adapter_type", "")),
        "backend": str(info.get("backend_type", "")),
        "codec_family": executor.codec_family,
        "codec_block": list(executor.codec_block),
        "construct_ms": construct_ms,
        "cold_submit_ms": submit_ms,
        "cold_fence_ms": fence_ms,
        "histogram_ms": histogram_ms,
        "uploads": int(upload_report.uploads),
        "upload_bytes": int(upload_report.upload_bytes),
        "compressed_uploads": int(executor.compressed_uploads_total),
        "compressed_fallbacks": int(executor.compressed_fallbacks_total),
        "active_resident_bytes": int(executor.active_resident_bytes),
        "allocated_pool_bytes": int(executor.allocated_pool_bytes),
    }


def _lod_case(tile: np.ndarray, representation: str, texture_codec: str, power: str) -> dict:
    device = _device(power)
    mode = "off" if texture_codec == "off" else "auto"
    shape = (2 * PAGE, 2 * PAGE)
    doc, operation = "g7-live-lod", "identity"
    dtype = "float32" if representation == SCALAR_R32F else "complex64"
    executor = WgpuPlaneExecutor(
        shape,
        max_lod=1,
        target_size=(64, 64),
        device=device,
        pool_layers={representation: 8},
        compressed_textures=mode,
    )
    plane = ContentPlane(doc, operation, shape, max_lod=1, representation=representation)
    sources = tuple(
        plane_chunk_key(
            doc,
            operation,
            0,
            cx,
            cy,
            dtype=dtype,
            representation=representation,
            plane_shape=shape,
        )
        for cy in range(2)
        for cx in range(2)
    )
    destination = plane_chunk_key(
        doc,
        operation,
        1,
        0,
        0,
        dtype=dtype,
        representation=representation,
        plane_shape=shape,
    )
    payload = _payload_for_executor(tile, representation)
    executor.submit(
        FrameSubmission(
            0,
            (BindContentPlanes((plane,)), *(EnsureChunkResident(key, payload) for key in sources)),
        )
    ).wait_completed()
    start = time.perf_counter()
    report = executor.submit(FrameSubmission(1, (GenerateLodPages(sources, destination),)))
    report.wait_completed()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "lod_ms": elapsed_ms,
        "compressed_source_cpu_reductions": int(executor.lod_compressed_source_reductions_total),
        "generated": len(report.lod_pages_generated),
    }


def run_benchmark(
    *,
    data_path: Path,
    power: str,
    texture_codec: str,
    host_codec: str,
    representation: str,
    pages: int,
    budget_chunks: int,
    scope: str = "both",
) -> dict:
    tiles = _load_tiles(data_path, pages, representation)
    result = {
        "schema": "arrayscope.g7-live-compression-benchmark.v1",
        "git_revision": _git_revision(),
        "data_path": str(data_path.resolve()),
        "pages": pages,
        "power_preference": power,
    }
    if scope in ("host", "both"):
        result["host"] = _host_cache_case(tiles, host_codec, budget_chunks)
    if scope in ("gpu", "both"):
        gpu = _gpu_case(tiles, representation, texture_codec, power)
        gpu.update(_lod_case(tiles[0], representation, texture_codec, power))
        result["gpu"] = gpu
    return result


def _git_revision() -> str:
    try:
        return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    except Exception:
        return "unknown"


def _format(result: dict) -> str:
    lines = [f"data: {result['data_path']}  pages={result['pages']}"]
    host = result.get("host")
    if host is not None:
        lines.append(
            f"host {host['codec']}: budget={host['total_budget_bytes']}B "
            f"raw+tier={host['raw_budget_bytes']}+{host['tier_budget_bytes']}B "
            f"used={host['combined_used_bytes']}B unique={host['unique_resident_keys']} "
            f"recoveries={host['tier_recoveries']} admit={host['admit_total_ms']:.1f}ms "
            f"max-put={host['admit_max_ms']:.1f}ms revisit={host['revisit_total_ms']:.1f}ms"
        )
    gpu = result.get("gpu")
    if gpu is not None:
        lines.extend(
            (
                f"gpu {gpu['adapter']} ({gpu['adapter_type']}) {gpu['texture_codec']} -> "
                f"{gpu['codec_family']} {tuple(gpu['codec_block'])}: construct={gpu['construct_ms']:.1f}ms "
                f"cold-submit={gpu['cold_submit_ms']:.1f}ms fence={gpu['cold_fence_ms']:.1f}ms "
                f"histogram={gpu['histogram_ms']:.1f}ms lod={gpu['lod_ms']:.1f}ms",
                f"gpu bytes: upload={gpu['upload_bytes']} active={gpu['active_resident_bytes']} "
                f"allocated-pools={gpu['allocated_pool_bytes']} compressed={gpu['compressed_uploads']} "
                f"fallbacks={gpu['compressed_fallbacks']} cpu-lod={gpu['compressed_source_cpu_reductions']}",
            )
        )
    lines.append(
        "verdict is comparative: run fresh OFF/RAW and AUTO/AUTO processes; "
        "do not infer live benefit from encoded ratio alone."
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--power", choices=("low-power", "high-performance"), default="low-power")
    parser.add_argument("--texture-codec", choices=("off", "auto"), default="off")
    parser.add_argument("--host-codec", choices=("raw", "auto", "zfp", "blosc2"), default="raw")
    parser.add_argument("--representation", choices=("scalar", "complex"), default="scalar")
    parser.add_argument("--pages", type=int, default=16)
    parser.add_argument("--budget-chunks", type=int, default=8)
    parser.add_argument("--scope", choices=("host", "gpu", "both"), default="both")
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: tuple[str, ...] | None = None) -> int:
    args = _parser().parse_args(argv)
    representation = SCALAR_R32F if args.representation == "scalar" else COMPLEX_RG32F
    result = run_benchmark(
        data_path=args.data,
        power=args.power,
        texture_codec=args.texture_codec,
        host_codec=args.host_codec,
        representation=representation,
        pages=max(4, int(args.pages)),
        budget_chunks=max(2, int(args.budget_chunks)),
        scope=args.scope,
    )
    print(_format(result))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
