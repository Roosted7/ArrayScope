"""G7 gate: does compressing chunk bytes beat uploading them raw?

The G7 gate is a benchmark matrix proving the *compression inequality*

    compress + transfer(compressed) + decompress  <  transfer(raw)

per (dtype, scenario) before any default flips on.  This tool measures it on
real data (``data/``): it tiles a real volume into ``256x256`` transport-sized
chunks, casts each chunk to the transport dtypes (float32 scalar, complex64,
int16), and for every codec measures compress time, decompress time and
compressed bytes against the raw payload.

Transfer time is bandwidth-dependent, so rather than hard-code one PCIe number
the tool reports the **break-even bandwidth** per cell:

    break_even_gbps = (raw_bytes - compressed_bytes) / (compress_s + decompress_s)

The inequality holds only when the *actual* host->GPU link is slower than that
figure (a slow link makes the saved bytes worth the codec's CPU cost).  We also
evaluate the inequality directly at a measured host memcpy bandwidth and at a
nominal PCIe bandwidth (``--pcie-gbps``, default 12 GB/s), and print HOLDS/no
per cell.  On this hardware CPU (de)compression is the honest outcome to report:
compression saves host RAM, but for a fast PCIe link the CPU decode latency
usually makes the inequality fail -- so the default stays ``raw`` until a real
GPU-side decoder changes the transfer side.  See docs for the recorded verdict.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from arrayscope.gpu.chunk_codec import resolve_codec

PAGE = 256
DEFAULT_DATA_PATH = Path(
    "/home/thomas/projects/ArrayScope/data/_WIPDelRec-tT2_20260223150234_14.nii"
)
# Transport dtypes at the chunk seam (arrayscope.gpu.wgpu_executor payloads).
DTYPES = ("float32", "complex64", "int16")
CODECS = ("zfp", "blosc2")
NOMINAL_PCIE_GBPS = 12.0


@dataclass(frozen=True)
class Cell:
    dtype: str
    codec: str
    chunks: int
    raw_bytes: int
    compressed_bytes: int
    ratio: float
    compress_ms: float
    decompress_ms: float
    exact: bool
    break_even_gbps: float
    holds_at_memcpy: bool
    holds_at_pcie: bool


def _load_volume(path: Path) -> np.ndarray:
    import nibabel as nib

    data = np.asanyarray(nib.load(str(path)).dataobj)
    while data.ndim > 3:
        data = data[..., 0]
    if data.ndim == 2:
        data = data[None, ...]
    return data


def _chunk_as(volume: np.ndarray, dtype: str, limit: int) -> list[np.ndarray]:
    """Extract up to ``limit`` PAGE x PAGE chunks cast to a transport dtype."""

    chunks: list[np.ndarray] = []
    depth = volume.shape[0]
    base = volume.astype(np.float32)
    lo, hi = float(np.nanmin(base)), float(np.nanmax(base))
    span = (hi - lo) or 1.0
    for z in range(depth):
        plane = base[z]
        h, w = plane.shape[:2]
        for r in range(0, h - PAGE + 1, PAGE):
            for c in range(0, w - PAGE + 1, PAGE):
                tile = plane[r : r + PAGE, c : c + PAGE]
                if tile.shape != (PAGE, PAGE):
                    continue
                if dtype == "float32":
                    chunks.append(np.ascontiguousarray(tile, dtype=np.float32))
                elif dtype == "complex64":
                    imag = np.ascontiguousarray(np.roll(tile, 1, axis=0), dtype=np.float32)
                    chunks.append((tile + 1j * imag).astype(np.complex64))
                elif dtype == "int16":
                    norm = (tile - lo) / span
                    chunks.append((norm * 4000.0 - 2000.0).astype(np.int16))
                if len(chunks) >= limit:
                    return chunks
    return chunks


def _measure_memcpy_gbps(nbytes: int) -> float:
    """Empirical host memcpy bandwidth -- a lower bound proxy for a real link."""

    src = np.frombuffer(np.random.default_rng(0).bytes(max(nbytes, 1 << 20)), dtype=np.uint8)
    best = 0.0
    for _ in range(5):
        t0 = time.perf_counter()
        dst = src.copy()
        dt = time.perf_counter() - t0
        dst[0] = dst[0]  # keep the copy live
        best = max(best, src.nbytes / dt)
    return best / 1e9


def _inequality_holds(raw_bytes: int, comp_bytes: int, overhead_s: float, bw_gbps: float) -> bool:
    """compress + transfer(compressed) + decompress < transfer(raw) at ``bw_gbps``."""

    raw_t = raw_bytes / (bw_gbps * 1e9)
    comp_t = overhead_s + comp_bytes / (bw_gbps * 1e9)
    return comp_t < raw_t


def _time_codec(codec, chunks: list[np.ndarray]) -> tuple[int, float, float, bool]:
    """Return (compressed_bytes, compress_s, decompress_s, exact) over chunks."""

    compressed_bytes = 0
    compress_s = 0.0
    decompress_s = 0.0
    exact = True
    for chunk in chunks:
        t0 = time.perf_counter()
        data = codec.encode(chunk)
        compress_s += time.perf_counter() - t0
        compressed_bytes += len(data)
        t0 = time.perf_counter()
        back = codec.decode(data, shape=chunk.shape, dtype=chunk.dtype)
        decompress_s += time.perf_counter() - t0
        if exact and not np.array_equal(back, chunk):
            exact = False
    return compressed_bytes, compress_s, decompress_s, exact


def run_benchmark(
    data_path: Path,
    *,
    chunk_limit: int = 200,
    pcie_gbps: float = NOMINAL_PCIE_GBPS,
) -> dict:
    volume = _load_volume(data_path)
    memcpy_gbps = _measure_memcpy_gbps(PAGE * PAGE * 4)
    cells: list[Cell] = []
    for dtype in DTYPES:
        chunks = _chunk_as(volume, dtype, chunk_limit)
        if not chunks:
            continue
        raw_bytes = sum(int(np.ascontiguousarray(c).nbytes) for c in chunks)
        for codec_name in CODECS:
            codec = resolve_codec(codec_name, chunks[0].dtype)
            if codec.name != codec_name:
                # resolve fell back to raw (unavailable/unsupported): record it.
                cells.append(
                    Cell(
                        dtype=dtype,
                        codec=f"{codec_name}(->raw)",
                        chunks=len(chunks),
                        raw_bytes=raw_bytes,
                        compressed_bytes=raw_bytes,
                        ratio=1.0,
                        compress_ms=0.0,
                        decompress_ms=0.0,
                        exact=True,
                        break_even_gbps=0.0,
                        holds_at_memcpy=False,
                        holds_at_pcie=False,
                    )
                )
                continue
            comp_bytes, comp_s, decomp_s, exact = _time_codec(codec, chunks)
            saved = raw_bytes - comp_bytes
            overhead = comp_s + decomp_s
            break_even = (saved / overhead / 1e9) if overhead > 0 else float("inf")
            cells.append(
                Cell(
                    dtype=dtype,
                    codec=codec_name,
                    chunks=len(chunks),
                    raw_bytes=raw_bytes,
                    compressed_bytes=comp_bytes,
                    ratio=raw_bytes / comp_bytes if comp_bytes else float("inf"),
                    compress_ms=comp_s * 1e3,
                    decompress_ms=decomp_s * 1e3,
                    exact=exact,
                    break_even_gbps=break_even,
                    holds_at_memcpy=_inequality_holds(raw_bytes, comp_bytes, overhead, memcpy_gbps),
                    holds_at_pcie=_inequality_holds(raw_bytes, comp_bytes, overhead, pcie_gbps),
                )
            )
    any_win = any(c.holds_at_pcie for c in cells)
    return {
        "schema": "arrayscope.g7-transport-benchmark.v1",
        "data_path": str(Path(data_path).resolve()),
        "data_shape": [int(v) for v in volume.shape],
        "chunk_shape": [PAGE, PAGE],
        "chunk_limit": int(chunk_limit),
        "memcpy_gbps": memcpy_gbps,
        "pcie_gbps": pcie_gbps,
        "git_revision": _git_rev(),
        "verdict": ("flip-eligible" if any_win else "no-win: default stays raw (off)"),
        "cells": [asdict(c) for c in cells],
    }


def _git_rev() -> str:
    try:
        return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    except Exception:
        return "unknown"


def _format_matrix(result: dict) -> str:
    lines = []
    lines.append(f"data: {result['data_path']}  shape={result['data_shape']}")
    lines.append(
        f"chunk={result['chunk_shape']} limit={result['chunk_limit']}  "
        f"memcpy~{result['memcpy_gbps']:.1f} GB/s  pcie~{result['pcie_gbps']:.1f} GB/s"
    )
    header = (
        f"{'dtype':<10}{'codec':<14}{'chunks':>7}{'ratio':>8}"
        f"{'comp_ms':>10}{'decomp_ms':>11}{'break_even':>12}{'exact':>7}"
        f"{'@memcpy':>9}{'@pcie':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for c in result["cells"]:
        be = c["break_even_gbps"]
        be_s = "inf" if be == float("inf") else f"{be:.2f}"
        lines.append(
            f"{c['dtype']:<10}{c['codec']:<14}{c['chunks']:>7}{c['ratio']:>8.3f}"
            f"{c['compress_ms']:>10.1f}{c['decompress_ms']:>11.1f}{be_s:>12}"
            f"{('yes' if c['exact'] else 'NO'):>7}"
            f"{('HOLDS' if c['holds_at_memcpy'] else 'no'):>9}"
            f"{('HOLDS' if c['holds_at_pcie'] else 'no'):>7}"
        )
    lines.append("")
    lines.append(f"verdict: {result['verdict']}")
    lines.append(
        "note: break_even = link bandwidth below which compress+transfer+decompress "
        "< raw transfer.\n      A real PCIe link (~12 GB/s) is far above these figures, "
        "so CPU-decode\n      compression does NOT beat raw transfer here; the win it "
        "delivers is host RAM.\n      A GPU-side decoder is the path to a transfer-side "
        "win (G7 follow-up)."
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="G7 codec-aware chunk transport benchmark")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--chunk-limit", type=int, default=200)
    parser.add_argument("--pcie-gbps", type=float, default=NOMINAL_PCIE_GBPS)
    parser.add_argument("--json", type=Path, default=None, help="also write raw JSON here")
    return parser


def main(argv: tuple[str, ...] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_benchmark(
        data_path=Path(args.data),
        chunk_limit=max(1, int(args.chunk_limit)),
        pcie_gbps=float(args.pcie_gbps),
    )
    print(_format_matrix(result))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
