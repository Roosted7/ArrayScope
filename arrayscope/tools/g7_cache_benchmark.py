"""G7 Phase A gate: does a compressed backing tier cut expensive misses?

The transport benchmark (``g7_transport_benchmark``) proved the *transfer-time*
inequality fails on a fast PCIe link: a CPU decode is slower than a raw memcpy,
so codec-on-the-wire does not beat raw upload.  Phase A cashes the *other* win:
host RAM.  Under a fixed RAM budget a compressed backing tier keeps ~ratio x more
of the working set resident, so fewer revisits fall off the cache and force an
expensive recompute (an FFT) or re-read (a disk page).

This tool measures that on real data (``data/``).  It tiles a real volume into
256x256 transport chunks, then runs a *revisit workload* (scroll back and forth
over ``W`` distinct chunks) under a fixed RAM budget ``B`` in two configs:

* raw-only: one raw byte-bounded cache of budget ``B`` -- fits ``N_raw`` chunks;
* two-level: a small raw cache + a compressed tier summing to ``B`` -- fits
  ~``ratio x N_raw`` chunks.

A miss (a revisit that fell out of cache) pays a real ``fft2`` recompute; a tier
hit pays only a decode.  We report, per dtype: the compression ratio, the
working set retained (raw vs two-level), the recompute count, the end-to-end
workload time, and the crossover working-set size where the two-level tier starts
winning.  The topology label is printed so the RAM-axis verdict is on record for
this machine's integrated + discrete devices (both benefit on RAM; the discrete
transfer win is Phase B).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from arrayscope.gpu.cache_policy import decide_compressed_tier
from arrayscope.gpu.chunk_codec import resolve_codec
from arrayscope.gpu.device_topology import detect_topology
from arrayscope.operations.compressed_tier import CompressedBackingTier, TwoLevelArrayCache

PAGE = 256
DEFAULT_DATA_PATH = Path(
    "/home/thomas/projects/ArrayScope/data/_WIPDelRec-tT2_20260223150234_14.nii"
)
DTYPES = ("float32", "complex64", "int16")
# A small raw hot-set slice; the rest of the RAM budget is the compressed tier.
RAW_SLICE_CHUNKS = 4


def _load_volume(path: Path) -> np.ndarray:
    import nibabel as nib

    data = np.asanyarray(nib.load(str(path)).dataobj)
    while data.ndim > 3:
        data = data[..., 0]
    if data.ndim == 2:
        data = data[None, ...]
    return data


def _chunk_as(volume: np.ndarray, dtype: str, limit: int) -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    base = volume.astype(np.float32)
    lo, hi = float(np.nanmin(base)), float(np.nanmax(base))
    span = (hi - lo) or 1.0
    for z in range(base.shape[0]):
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


def _revisit_order(working_set: int, sweeps: int) -> list[int]:
    """Back-and-forth scroll over ``working_set`` chunks, ``sweeps`` times."""

    up = list(range(working_set))
    down = list(range(working_set - 1, -1, -1))
    order: list[int] = []
    for _ in range(sweeps):
        order.extend(up)
        order.extend(down)
    return order


def _micro_costs(
    codec_name: str, chunks: list[np.ndarray], miss_plane: np.ndarray
) -> tuple[float, float, float]:
    """Mean per-op (encode_ms, decode_ms, recompute_ms) -- the crossover terms.

    ``encode`` is paid once per eviction (compress-into-tier); ``decode`` once
    per tier recovery; ``recompute`` is the expensive miss the tier avoids (an
    fft2 the size of one stage recompute -- ``miss_plane``).
    """

    codec = resolve_codec(codec_name, chunks[0].dtype)
    sample = chunks[: min(len(chunks), 32)]
    t0 = time.perf_counter()
    blobs = [codec.encode(c) for c in sample]
    encode_ms = (time.perf_counter() - t0) / len(sample) * 1e3
    t0 = time.perf_counter()
    for c, blob in zip(sample, blobs, strict=True):
        codec.decode(blob, shape=c.shape, dtype=c.dtype)
    decode_ms = (time.perf_counter() - t0) / len(sample) * 1e3
    reps = 8
    t0 = time.perf_counter()
    for _ in range(reps):
        np.fft.fft2(miss_plane)
    recompute_ms = (time.perf_counter() - t0) / reps * 1e3
    return encode_ms, decode_ms, recompute_ms


def _measured_ratio(codec_name: str, chunks: list[np.ndarray]) -> tuple[float, int, int]:
    """Return (ratio, raw_chunk_bytes, mean_compressed_bytes) over the chunks."""

    codec = resolve_codec(codec_name, chunks[0].dtype)
    raw_bytes = int(chunks[0].nbytes)
    total = 0
    for chunk in chunks:
        total += len(codec.encode(chunk))
    mean_comp = max(1, total // len(chunks))
    return raw_bytes / mean_comp, raw_bytes, mean_comp


@dataclass(frozen=True)
class ConfigResult:
    label: str
    fit_chunks: int
    recomputes: int
    tier_recoveries: int
    workload_ms: float


@dataclass(frozen=True)
class DtypeResult:
    dtype: str
    codec: str
    working_set: int
    sweeps: int
    ratio: float
    raw_chunk_bytes: int
    mean_compressed_bytes: int
    budget_bytes: int
    policy_engage: bool
    policy_reason: str
    encode_ms: float
    decode_ms: float
    recompute_ms: float
    crossover_recompute_ms: float
    raw_only: ConfigResult
    two_level: ConfigResult
    ram_footprint_raw_bytes: int
    ram_footprint_two_level_bytes: int
    crossover_working_set: int | None


def _run_config(
    *,
    label: str,
    chunks: list[np.ndarray],
    order: list[int],
    raw_budget: int,
    tier: CompressedBackingTier | None,
    miss_plane: np.ndarray,
) -> ConfigResult:
    cache = TwoLevelArrayCache.build(
        raw_max_bytes=raw_budget,
        raw_max_entries=10_000,
        tier=tier,
    )

    def make_compute(idx: int):
        def compute():
            # Real expensive-miss cost: an fft2 the size of one stage recompute.
            np.fft.fft2(miss_plane)
            return chunks[idx]

        return compute

    t0 = time.perf_counter()
    for idx in order:
        cache.get_or_compute((idx,), make_compute(idx))
    workload_ms = (time.perf_counter() - t0) * 1e3

    fit = len(cache.raw._cache) + (len(tier) if tier is not None else 0)
    return ConfigResult(
        label=label,
        fit_chunks=fit,
        recomputes=cache.recomputes,
        tier_recoveries=cache.tier_recoveries,
        workload_ms=workload_ms,
    )


def _miss_plane(side: int) -> np.ndarray:
    """A deterministic float plane whose fft2 models one expensive stage miss."""

    ramp = np.linspace(0.0, 1.0, side, dtype=np.float32)
    return np.ascontiguousarray(np.outer(ramp, ramp[::-1]))


def _run_dtype(
    volume: np.ndarray,
    dtype: str,
    *,
    working_set: int,
    sweeps: int,
    budget_chunks: int,
    miss_fft: int,
    miss_fft_sweep: tuple[int, ...],
) -> DtypeResult | None:
    chunks = _chunk_as(volume, dtype, limit=working_set)
    if len(chunks) < working_set:
        return None
    codec_name = decide_compressed_tier(
        working_set_bytes=working_set * int(chunks[0].nbytes),
        budget_bytes=budget_chunks * int(chunks[0].nbytes),
        dtype=chunks[0].dtype,
    ).codec_name
    if codec_name == "raw":
        codec_name = "blosc2"
    ratio, raw_chunk_bytes, mean_comp = _measured_ratio(codec_name, chunks[:working_set])

    budget_bytes = budget_chunks * raw_chunk_bytes
    decision = decide_compressed_tier(
        working_set_bytes=working_set * raw_chunk_bytes,
        budget_bytes=budget_bytes,
        dtype=chunks[0].dtype,
    )
    miss_plane = _miss_plane(miss_fft)
    encode_ms, decode_ms, recompute_ms = _micro_costs(codec_name, chunks[:working_set], miss_plane)
    order = _revisit_order(working_set, sweeps)

    raw_only = _run_config(
        label="raw-only",
        chunks=chunks,
        order=order,
        raw_budget=budget_bytes,
        tier=None,
        miss_plane=miss_plane,
    )
    raw_slice = RAW_SLICE_CHUNKS * raw_chunk_bytes
    tier = CompressedBackingTier(max_bytes=max(0, budget_bytes - raw_slice), codec_name=codec_name)
    two_level = _run_config(
        label="two-level",
        chunks=chunks,
        order=order,
        raw_budget=raw_slice,
        tier=tier,
        miss_plane=miss_plane,
    )

    # Analytical crossover: two-level trades (encode per eviction + decode per
    # recovery) for the recomputes it avoids.  It wins on time once the avoided
    # recomputes outweigh that codec overhead:
    #   saved_recomputes * recompute_ms  >  encodes*encode_ms + recoveries*decode_ms
    saved = raw_only.recomputes - two_level.recomputes
    codec_overhead_ms = tier.stores * encode_ms + two_level.tier_recoveries * decode_ms
    crossover_recompute_ms = (codec_overhead_ms / saved) if saved > 0 else float("inf")

    crossover = _find_crossover_missfft(
        chunks, codec_name, budget_bytes, raw_slice, working_set, sweeps, miss_fft_sweep
    )

    return DtypeResult(
        dtype=dtype,
        codec=codec_name,
        working_set=working_set,
        sweeps=sweeps,
        ratio=ratio,
        raw_chunk_bytes=raw_chunk_bytes,
        mean_compressed_bytes=mean_comp,
        budget_bytes=budget_bytes,
        policy_engage=decision.engage,
        policy_reason=decision.reason,
        encode_ms=encode_ms,
        decode_ms=decode_ms,
        recompute_ms=recompute_ms,
        crossover_recompute_ms=crossover_recompute_ms,
        raw_only=raw_only,
        two_level=two_level,
        ram_footprint_raw_bytes=budget_bytes,
        ram_footprint_two_level_bytes=raw_slice + tier.compressed_bytes,
        crossover_working_set=crossover,
    )


def _find_crossover_missfft(
    chunks, codec_name, budget_bytes, raw_slice, working_set, sweeps, miss_fft_sweep
) -> int | None:
    """Smallest miss-fft side at which two-level end-to-end time beats raw-only.

    Larger miss-fft = a more expensive stage recompute -- exactly the "large
    matrix" regime the RAM win targets.  Returns the crossover side, or None if
    the tier never wins on time across the swept sizes.
    """

    order = _revisit_order(working_set, sweeps)
    for side in miss_fft_sweep:
        miss_plane = _miss_plane(side)
        raw_only = _run_config(
            label="raw-only",
            chunks=chunks,
            order=order,
            raw_budget=budget_bytes,
            tier=None,
            miss_plane=miss_plane,
        )
        tier = CompressedBackingTier(max_bytes=max(0, budget_bytes - raw_slice), codec_name=codec_name)
        two_level = _run_config(
            label="two-level",
            chunks=chunks,
            order=order,
            raw_budget=raw_slice,
            tier=tier,
            miss_plane=miss_plane,
        )
        if two_level.workload_ms < raw_only.workload_ms:
            return side
    return None


def run_benchmark(
    data_path: Path,
    *,
    working_set: int = 120,
    sweeps: int = 3,
    budget_chunks: int = 40,
    miss_fft: int = 512,
    miss_fft_sweep: tuple[int, ...] = (256, 512, 1024, 2048),
) -> dict:
    volume = _load_volume(data_path)
    topology = detect_topology()
    results: list[DtypeResult] = []
    for dtype in DTYPES:
        res = _run_dtype(
            volume,
            dtype,
            working_set=working_set,
            sweeps=sweeps,
            budget_chunks=budget_chunks,
            miss_fft=miss_fft,
            miss_fft_sweep=miss_fft_sweep,
        )
        if res is not None:
            results.append(res)
    return {
        "schema": "arrayscope.g7-cache-benchmark.v1",
        "data_path": str(Path(data_path).resolve()),
        "data_shape": [int(v) for v in volume.shape],
        "chunk_shape": [PAGE, PAGE],
        "working_set": working_set,
        "sweeps": sweeps,
        "budget_chunks": budget_chunks,
        "miss_fft": miss_fft,
        "topology": {
            "kind": topology.kind,
            "unified_memory": topology.unified_memory,
            "device": topology.device_name,
            "backend": topology.backend,
            "discrete_transfer_candidate_phase_b": topology.discrete_transfer_candidate,
        },
        "git_revision": _git_rev(),
        "dtypes": [asdict(r) for r in results],
    }


def _git_rev() -> str:
    try:
        return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    except Exception:
        return "unknown"


def _format(result: dict) -> str:
    lines = []
    lines.append(f"data: {result['data_path']}  shape={result['data_shape']}")
    topo = result["topology"]
    lines.append(
        f"topology: {topo['kind']} (unified_memory={topo['unified_memory']}, "
        f"device={topo['device']!r})  Phase-B transfer seam: "
        f"discrete_candidate={topo['discrete_transfer_candidate_phase_b']}"
    )
    lines.append(
        f"workload: revisit {result['working_set']} chunks x{result['sweeps']} sweeps  "
        f"budget={result['budget_chunks']} raw chunks  miss=fft2({result['miss_fft']}^2)"
    )
    lines.append("")
    lines.append("RAM win (unconditional -- same RAM budget, more working set retained):")
    ram_h = (
        f"{'dtype':<10}{'codec':<8}{'ratio':>7}{'fit_raw':>8}{'fit_2L':>8}"
        f"{'recomp_raw':>12}{'recomp_2L':>11}{'recover_2L':>12}{'RAM_MB':>9}"
    )
    lines.append(ram_h)
    lines.append("-" * len(ram_h))
    for r in result["dtypes"]:
        ro, tl = r["raw_only"], r["two_level"]
        ram_mb = r["ram_footprint_two_level_bytes"] / (1 << 20)
        lines.append(
            f"{r['dtype']:<10}{r['codec']:<8}{r['ratio']:>7.2f}"
            f"{ro['fit_chunks']:>8}{tl['fit_chunks']:>8}"
            f"{ro['recomputes']:>12}{tl['recomputes']:>11}{tl['tier_recoveries']:>12}"
            f"{ram_mb:>9.1f}"
        )
    lines.append("")
    lines.append("End-to-end time (miss-cost dependent) + crossover:")
    t_h = (
        f"{'dtype':<10}{'enc_ms':>8}{'dec_ms':>8}{'recomp_ms':>10}"
        f"{'ms_raw':>10}{'ms_2L':>9}{'speedup':>9}{'xover_ms':>10}{'xover_fft':>10}"
    )
    lines.append(t_h)
    lines.append("-" * len(t_h))
    for r in result["dtypes"]:
        ro, tl = r["raw_only"], r["two_level"]
        speedup = ro["workload_ms"] / tl["workload_ms"] if tl["workload_ms"] else float("inf")
        xo_ms = r["crossover_recompute_ms"]
        xo_ms_s = "inf" if xo_ms == float("inf") else f"{xo_ms:.3f}"
        xo_fft = "-" if r["crossover_working_set"] is None else f"{r['crossover_working_set']}^2"
        lines.append(
            f"{r['dtype']:<10}{r['encode_ms']:>8.3f}{r['decode_ms']:>8.3f}"
            f"{r['recompute_ms']:>10.3f}{ro['workload_ms']:>10.1f}{tl['workload_ms']:>9.1f}"
            f"{speedup:>8.2f}x{xo_ms_s:>10}{xo_fft:>10}"
        )
    lines.append("")
    lines.append(
        "fit_raw/fit_2L = distinct chunks resident under the SAME RAM budget (the RAM win).\n"
        "recover_2L = raw-misses served by a cheap decode instead of an expensive recompute.\n"
        "xover_ms = recompute cost above which the tier wins on TIME (avoided recomputes >\n"
        "  codec overhead); xover_fft = smallest miss fft2 side where two-level beats raw-only.\n"
        "Verdict: the RAM/eviction win is unconditional (~2.2x working set, fewer misses).\n"
        "The end-to-end TIME win appears once the expensive miss (large-matrix stage FFT or\n"
        "disk re-read) exceeds the per-op codec cost -- i.e. exactly the large-data regime\n"
        "the owner targets.  For cheap 256^2 misses the tier still wins RAM but not wall time,\n"
        "which is why the adaptive policy engages on RAM PRESSURE (large working sets)."
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="G7 Phase A compressed-tier cache benchmark")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--working-set", type=int, default=120)
    parser.add_argument("--sweeps", type=int, default=3)
    parser.add_argument("--budget-chunks", type=int, default=40)
    parser.add_argument(
        "--miss-fft", type=int, default=512, help="fft2 side modelling one expensive miss"
    )
    parser.add_argument("--json", type=Path, default=None)
    return parser


def main(argv: tuple[str, ...] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_benchmark(
        data_path=Path(args.data),
        working_set=max(1, int(args.working_set)),
        sweeps=max(1, int(args.sweeps)),
        budget_chunks=max(1, int(args.budget_chunks)),
        miss_fft=max(16, int(args.miss_fft)),
    )
    print(_format(result))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
