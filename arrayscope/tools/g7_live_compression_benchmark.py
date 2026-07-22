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
from arrayscope.gpu import bc_codec
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


def _histogram_case(
    executor,
    keys,
    *,
    display_mode: str,
    per_source: bool,
    batch_sources: int = 4,
) -> dict[str, float | int]:
    """Measure submit, fence, and readback for aggregate or live-shaped evidence."""

    source_rows = tuple((key,) for key in keys) if per_source else (tuple(keys),)
    submit_ms = []
    fence_ms = 0.0
    resolve_ms = 0.0
    started = time.perf_counter()
    for offset in range(0, len(source_rows), max(1, int(batch_sources))):
        rows = source_rows[offset : offset + max(1, int(batch_sources))]
        commands = tuple(
            DispatchHistogram(
                row,
                bins=64 if per_source else 256,
                lo=None,
                hi=None,
                mode=display_mode,
                scale="linear",
                symlog_constant=0.0,
            )
            for row in rows
        )
        submit_started = time.perf_counter()
        report = executor.submit(FrameSubmission(10_000 + offset, commands))
        submit_ms.append((time.perf_counter() - submit_started) * 1000.0)
        fence_started = time.perf_counter()
        report.wait_completed()
        fence_ms += (time.perf_counter() - fence_started) * 1000.0
        resolve_started = time.perf_counter()
        for value in report.histograms.values():
            if hasattr(value, "resolve"):
                value.resolve()
        resolve_ms += (time.perf_counter() - resolve_started) * 1000.0
    return {
        "dispatches": len(source_rows),
        "batch_sources": max(1, int(batch_sources)),
        "submit_ms": float(sum(submit_ms)),
        "submit_max_ms": float(max(submit_ms, default=0.0)),
        "fence_ms": float(fence_ms),
        "resolve_ms": float(resolve_ms),
        "total_ms": float((time.perf_counter() - started) * 1000.0),
    }


def _preencoded_transfer_case(executor, keys, tiles, representation: str) -> dict | None:
    """Measure only raw versus already-encoded queue transfer and completion.

    Encoding and quality verification intentionally happen before this cell.
    It answers the narrower question "would fewer bytes help if the source were
    already in the device-native compressed format?" rather than crediting the
    current synchronous runtime encoder with work it has not avoided.
    """

    if not executor.codec_engaged:
        return None
    rows = []
    for key, tile in zip(keys, tiles, strict=True):
        payload = _payload_for_executor(tile, representation)
        encoded = executor._encode_compressed(key, payload)
        if encoded is not None:
            rows.append((payload, encoded[0]))
    if not rows:
        return None

    import wgpu

    device = executor.device
    layers = len(rows)
    usage = wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST
    raw_format = "r32float" if representation == SCALAR_R32F else "rg32float"
    raw_texture = device.create_texture(
        size=(PAGE, PAGE, layers),
        format=raw_format,
        usage=usage,
    )
    compressed_texture = device.create_texture(
        size=(PAGE, PAGE, layers),
        format=executor.codec_pool_format(representation),
        usage=usage,
    )
    bx, by = executor.codec_block
    block_bytes = 8 if representation == SCALAR_R32F and executor.codec_family == "bc" else 16

    def transfer(compressed: bool) -> float:
        started = time.perf_counter()
        for layer, (raw, encoded) in enumerate(rows):
            if compressed:
                device.queue.write_texture(
                    {"texture": compressed_texture, "origin": (0, 0, layer)},
                    encoded,
                    {
                        "bytes_per_row": (PAGE // int(bx)) * int(block_bytes),
                        "rows_per_image": PAGE // int(by),
                    },
                    (PAGE, PAGE, 1),
                )
            else:
                device.queue.write_texture(
                    {"texture": raw_texture, "origin": (0, 0, layer)},
                    raw,
                    {
                        "bytes_per_row": PAGE * (4 if representation == SCALAR_R32F else 8),
                        "rows_per_image": PAGE,
                    },
                    (PAGE, PAGE, 1),
                )
        device.queue.on_submitted_work_done_sync()
        return (time.perf_counter() - started) * 1000.0

    # First pair warms driver allocation/staging. Alternate order thereafter.
    transfer(False)
    transfer(True)
    raw_ms = []
    compressed_ms = []
    for repetition in range(6):
        order = (False, True) if repetition % 2 == 0 else (True, False)
        for compressed in order:
            measured = transfer(compressed)
            (compressed_ms if compressed else raw_ms).append(measured)
    raw_median = float(np.median(raw_ms))
    compressed_median = float(np.median(compressed_ms))
    raw_bytes = sum(int(raw.nbytes) for raw, _encoded in rows)
    compressed_bytes = sum(len(encoded) for _raw, encoded in rows)
    return {
        "accepted_pages": layers,
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
        "raw_ms_median": raw_median,
        "compressed_ms_median": compressed_median,
        "saved_ms": raw_median - compressed_median,
        "speedup": raw_median / max(compressed_median, 1e-12),
        "excludes_encode_and_quality": True,
    }


def _gpu_case(
    tiles: list[np.ndarray],
    representation: str,
    texture_codec: str,
    power: str,
    *,
    numba_prewarm: bool,
) -> dict:
    device = _device(power)
    info = dict(device.adapter.info)
    mode = "off" if texture_codec == "off" else "auto"
    shape = (PAGE, len(tiles) * PAGE)
    doc, operation = "g7-live", "identity"

    prewarm_started = time.perf_counter()
    numba_enabled = bool(numba_prewarm and bc_codec.prewarm_numba_encoder())
    numba_prewarm_ms = (time.perf_counter() - prewarm_started) * 1000.0 if numba_prewarm else 0.0

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
    executor.submit(
        FrameSubmission(1, (SetDisplayMapping(DisplayMapping(display_mode, 0.0, 1.0)),))
    ).wait_completed()
    histogram_first_source_cold = _histogram_case(
        executor,
        keys[:1],
        display_mode=display_mode,
        per_source=True,
        batch_sources=1,
    )
    histogram_aggregate_warm = _histogram_case(
        executor,
        keys,
        display_mode=display_mode,
        per_source=False,
        batch_sources=len(keys),
    )
    histogram_live_per_source_warm = _histogram_case(
        executor,
        keys,
        display_mode=display_mode,
        per_source=True,
        batch_sources=4,
    )

    compressed_uploads = int(executor.compressed_uploads_total)
    compressed_fallbacks = int(executor.compressed_fallbacks_total)
    active_resident_bytes = int(executor.active_resident_bytes)
    allocated_pool_bytes = int(executor.allocated_pool_bytes)
    preencoded_transfer = _preencoded_transfer_case(executor, keys, tiles, representation)

    return {
        "texture_codec": texture_codec,
        "representation": representation,
        "adapter": str(info.get("device", "")),
        "adapter_type": str(info.get("adapter_type", "")),
        "backend": str(info.get("backend_type", "")),
        "codec_family": executor.codec_family,
        "codec_block": list(executor.codec_block),
        "numba_prewarm_requested": bool(numba_prewarm),
        "numba_encoder_enabled": numba_enabled,
        "numba_prewarm_ms": numba_prewarm_ms,
        "construct_ms": construct_ms,
        "cold_submit_ms": submit_ms,
        "cold_fence_ms": fence_ms,
        # Backward-compatible name for the old one-dispatch benchmark cell.
        "histogram_ms": float(histogram_aggregate_warm["total_ms"]),
        "histogram_first_source_cold": histogram_first_source_cold,
        "histogram_aggregate_warm": histogram_aggregate_warm,
        "histogram_live_per_source_warm": histogram_live_per_source_warm,
        "preencoded_transfer": preencoded_transfer,
        "uploads": int(upload_report.uploads),
        "upload_bytes": int(upload_report.upload_bytes),
        "compressed_uploads": compressed_uploads,
        "compressed_fallbacks": compressed_fallbacks,
        "active_resident_bytes": active_resident_bytes,
        "allocated_pool_bytes": allocated_pool_bytes,
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
    numba_prewarm: bool = False,
) -> dict:
    tiles = _load_tiles(data_path, pages, representation)
    result = {
        "schema": "arrayscope.g7-live-compression-benchmark.v3",
        "git_revision": _git_revision(),
        "data_path": str(data_path.resolve()),
        "pages": pages,
        "power_preference": power,
    }
    if scope in ("host", "both"):
        result["host"] = _host_cache_case(tiles, host_codec, budget_chunks)
    if scope in ("gpu", "both"):
        gpu = _gpu_case(
            tiles,
            representation,
            texture_codec,
            power,
            numba_prewarm=numba_prewarm,
        )
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
                f"numba={gpu['numba_encoder_enabled']} prewarm={gpu['numba_prewarm_ms']:.1f}ms "
                f"cold-submit={gpu['cold_submit_ms']:.1f}ms fence={gpu['cold_fence_ms']:.1f}ms "
                f"histogram-first-cold={gpu['histogram_first_source_cold']['total_ms']:.1f}ms "
                f"histogram-aggregate-warm={gpu['histogram_ms']:.1f}ms "
                f"histogram-live-warm={gpu['histogram_live_per_source_warm']['total_ms']:.1f}ms "
                f"lod={gpu['lod_ms']:.1f}ms",
                f"gpu live histogram: dispatches={gpu['histogram_live_per_source_warm']['dispatches']} "
                f"submit={gpu['histogram_live_per_source_warm']['submit_ms']:.1f}ms "
                f"fence={gpu['histogram_live_per_source_warm']['fence_ms']:.1f}ms "
                f"resolve={gpu['histogram_live_per_source_warm']['resolve_ms']:.1f}ms",
                f"gpu bytes: upload={gpu['upload_bytes']} active={gpu['active_resident_bytes']} "
                f"allocated-pools={gpu['allocated_pool_bytes']} compressed={gpu['compressed_uploads']} "
                f"fallbacks={gpu['compressed_fallbacks']} cpu-lod={gpu['compressed_source_cpu_reductions']}",
            )
        )
        transfer = gpu.get("preencoded_transfer")
        if transfer is not None:
            lines.append(
                f"preencoded transfer only: {transfer['accepted_pages']} pages "
                f"{transfer['raw_bytes']}->{transfer['compressed_bytes']}B "
                f"{transfer['raw_ms_median']:.2f}->{transfer['compressed_ms_median']:.2f}ms "
                f"({transfer['speedup']:.2f}x; encode/quality excluded)"
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
    parser.add_argument(
        "--numba-prewarm",
        action="store_true",
        help="compile the optional BC encoder before timing the live upload",
    )
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
        pages=max(1, int(args.pages)),
        budget_chunks=max(2, int(args.budget_chunks)),
        scope=args.scope,
        numba_prewarm=bool(args.numba_prewarm),
    )
    print(_format(result))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
