"""G7 Phase B gate: native block-compressed textures vs raw, per device x format.

The Phase-B transfer/VRAM win comes from handing the GPU a format its texture
sampler decompresses *for free*: a native BC (NVIDIA + Intel) or ASTC (Intel)
texture.  The compressed bytes cross PCIe, stay compressed resident in VRAM, and
there is no decode pass at all.  The cost is lossy compression -- so this tool
*measures* the loss instead of assuming it.

Per (device x format) it reports:

* **transfer/upload bytes** -- what crosses the bus (raw vs compressed);
* **VRAM resident bytes** -- the real texture footprint (matters most on the
  A2000's 4 GB);
* **quality** -- for a scalar tile, PSNR + max-abs of the decoded-then-normalized
  field vs raw; for a complex tile, DISPLAY-domain error (magnitude PSNR and
  wrapped-phase error) of magnitude/phase derived from decoded (re, im) vs raw.

It also verifies the encoded texture is a real, resident compressed texture on the
adapter, and (BC scalar) closes the loop by sampling it through the hardware
sampler and confirming the reference decode matches.

Only one Vulkan adapter is visible per process (env-selected ICD), so run once per
adapter:

    # discrete NVIDIA (BC only):
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
        __NV_PRIME_RENDER_OFFLOAD=1 \\
        python -m arrayscope.tools.g7_gpu_texture_benchmark --power-preference high-performance
    # integrated Intel (BC + ASTC block sweep):
    python -m arrayscope.tools.g7_gpu_texture_benchmark --power-preference low-power
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from arrayscope.gpu import astc_codec, bc_codec
from arrayscope.gpu.bc_gpu import (
    GpuBc4Encoder,
    GpuDecodeUnavailable,
    create_compute_device,
    sample_bc4_texture,
    upload_bc4_texture,
)

DEFAULT_DATA_PATH = Path(
    "/home/thomas/projects/ArrayScope/data/_WIPDelRec-tT2_20260223150234_14.nii"
)
TILE = 256


@dataclass
class FormatCell:
    kind: str  # "scalar" or "complex"
    format: str
    encoder: str  # "cpu" / "gpu" / "astc"
    raw_bytes: int
    upload_bytes: int
    vram_resident_bytes: int
    vram_ratio: float
    # scalar quality
    psnr_db: float = 0.0
    max_abs: float = 0.0
    # complex display-domain quality
    magnitude_psnr_db: float = 0.0
    phase_max_abs_rad: float = 0.0
    phase_weighted_rmse_rad: float = 0.0
    phase_max_abs_significant_rad: float = 0.0
    texture_created: bool = False
    hardware_sample_psnr_db: float = 0.0
    notes: str = ""


@dataclass
class BenchReport:
    adapter: str
    adapter_type: str
    backend: str
    bc_supported: bool
    astc_supported: bool
    tile: int
    cells: list = field(default_factory=list)


def _load_tiles(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One scalar float tile and a complex (re, im) tile from real data."""

    import nibabel as nib

    vol = np.asanyarray(nib.load(str(path)).dataobj).astype(np.float32)
    while vol.ndim > 2:
        vol = vol[..., vol.shape[-1] // 2] if vol.ndim > 3 else vol[vol.shape[0] // 2]
    h, w = vol.shape[:2]
    r0 = max(0, (h - TILE) // 2)
    c0 = max(0, (w - TILE) // 2)
    scalar = np.ascontiguousarray(vol[r0 : r0 + TILE, c0 : c0 + TILE])
    if scalar.shape != (TILE, TILE):
        scalar = np.pad(
            scalar, ((0, TILE - scalar.shape[0]), (0, TILE - scalar.shape[1])), mode="edge"
        )
    # a plausible complex tile: real = tile, imag = a smooth phase-carrying partner
    re = scalar
    im = np.ascontiguousarray(np.roll(scalar, 3, axis=1) * 0.6 - np.roll(scalar, 2, axis=0) * 0.4)
    return scalar, re, im


def _scalar_cells(device, scalar, bc_supported, astc_supported) -> list[FormatCell]:
    cells: list[FormatCell] = []
    unit, _ = bc_codec.normalize_tile(scalar)
    raw_bytes = scalar.size * 4  # r32float

    # BC4 -- CPU encoder
    data, h, w = bc_codec.bc4_encode(unit)
    q = bc_codec.quality_of(unit, bc_codec.bc4_decode(data, h, w))
    cell = FormatCell(
        kind="scalar",
        format="bc4-r-unorm",
        encoder="cpu",
        raw_bytes=raw_bytes,
        upload_bytes=len(data),
        vram_resident_bytes=len(data),
        vram_ratio=raw_bytes / len(data),
        psnr_db=q.psnr_db,
        max_abs=q.max_abs_diff,
    )
    if bc_supported:
        try:
            tex = upload_bc4_texture(device, data, h, w)
            cell.texture_created = True
            hw = sample_bc4_texture(device, tex, h, w)
            cell.hardware_sample_psnr_db = bc_codec.quality_of(
                bc_codec.bc4_decode(data, h, w), hw
            ).psnr_db
        except Exception as exc:  # pragma: no cover - driver dependent
            cell.notes = f"texture/sample failed: {exc}"
    cells.append(cell)

    # BC4 -- GPU compute encoder (proves on-device encode)
    if bc_supported:
        try:
            genc, gh, gw = GpuBc4Encoder(device).encode(unit)
            gq = bc_codec.quality_of(unit, bc_codec.bc4_decode(genc, gh, gw))
            cells.append(
                FormatCell(
                    kind="scalar",
                    format="bc4-r-unorm",
                    encoder="gpu",
                    raw_bytes=raw_bytes,
                    upload_bytes=len(genc),
                    vram_resident_bytes=len(genc),
                    vram_ratio=raw_bytes / len(genc),
                    psnr_db=gq.psnr_db,
                    max_abs=gq.max_abs_diff,
                    notes="on-GPU WGSL BC4 encode",
                )
            )
        except Exception as exc:  # pragma: no cover
            cells.append(
                FormatCell(
                    "scalar",
                    "bc4-r-unorm",
                    "gpu",
                    raw_bytes,
                    0,
                    0,
                    0.0,
                    notes=f"gpu encode failed: {exc}",
                )
            )

    # ASTC block sweep (Intel)
    if astc_supported and astc_codec.astc_available():
        for block in ((4, 4), (6, 6)):
            res = astc_codec.encode_scalar(unit, block=block)
            q = bc_codec.quality_of(unit, res.decoded[0][: scalar.shape[0], : scalar.shape[1]])
            cells.append(
                FormatCell(
                    kind="scalar",
                    format=res.wgpu_format,
                    encoder="astc",
                    raw_bytes=raw_bytes,
                    upload_bytes=res.bc_bytes,
                    vram_resident_bytes=res.bc_bytes,
                    vram_ratio=raw_bytes / res.bc_bytes,
                    psnr_db=q.psnr_db,
                    max_abs=q.max_abs_diff,
                )
            )
    return cells


def _complex_cells(re, im, bc_supported, astc_supported) -> list[FormatCell]:
    cells: list[FormatCell] = []
    unit_re, nre = bc_codec.normalize_tile(re)
    unit_im, nim = bc_codec.normalize_tile(im)
    raw_bytes = re.size * 8  # rg32float

    # BC5 (re, im)
    data, h, w = bc_codec.bc5_encode(unit_re, unit_im)
    d0, d1 = bc_codec.bc5_decode(data, h, w)
    q = bc_codec.complex_display_quality(re, im, nre.denormalize(d0), nim.denormalize(d1))
    cells.append(
        FormatCell(
            kind="complex",
            format="bc5-rg-unorm",
            encoder="cpu",
            raw_bytes=raw_bytes,
            upload_bytes=len(data),
            vram_resident_bytes=len(data),
            vram_ratio=raw_bytes / len(data),
            magnitude_psnr_db=q.magnitude_psnr_db,
            phase_max_abs_rad=q.phase_max_abs_rad,
            phase_weighted_rmse_rad=q.phase_weighted_rmse_rad,
            phase_max_abs_significant_rad=q.phase_max_abs_significant_rad,
            notes="(real, imag); display-domain quality",
        )
    )

    # ASTC two-channel (Intel)
    if astc_supported and astc_codec.astc_available():
        for block in ((4, 4), (6, 6)):
            res = astc_codec.encode_two_channel(unit_re, unit_im, block=block)
            re_d = nre.denormalize(res.decoded[0][: re.shape[0], : re.shape[1]])
            im_d = nim.denormalize(res.decoded[1][: im.shape[0], : im.shape[1]])
            q = bc_codec.complex_display_quality(re, im, re_d, im_d)
            cells.append(
                FormatCell(
                    kind="complex",
                    format=res.wgpu_format,
                    encoder="astc",
                    raw_bytes=raw_bytes,
                    upload_bytes=res.bc_bytes,
                    vram_resident_bytes=res.bc_bytes,
                    vram_ratio=raw_bytes / res.bc_bytes,
                    magnitude_psnr_db=q.magnitude_psnr_db,
                    phase_max_abs_rad=q.phase_max_abs_rad,
                    phase_weighted_rmse_rad=q.phase_weighted_rmse_rad,
                    phase_max_abs_significant_rad=q.phase_max_abs_significant_rad,
                    notes="(real, imag); display-domain quality",
                )
            )
    return cells


def run_benchmark(*, data_path: Path, power_preference: str) -> dict:
    device = create_compute_device(power_preference, features=["texture-compression-bc"])
    info = dict(device.adapter.info)
    have = {str(f) for f in device.adapter.features}
    bc_supported = "texture-compression-bc" in have
    astc_supported = "texture-compression-astc" in have and astc_codec.astc_available()

    scalar, re, im = _load_tiles(data_path)
    report = BenchReport(
        adapter=str(info.get("device", "")),
        adapter_type=str(info.get("adapter_type", "")),
        backend=str(info.get("backend_type", "")),
        bc_supported=bc_supported,
        astc_supported=astc_supported,
        tile=TILE,
    )
    report.cells = [
        asdict(c)
        for c in _scalar_cells(device, scalar, bc_supported, astc_supported)
        + _complex_cells(re, im, bc_supported, astc_supported)
    ]
    return {
        "schema": "arrayscope.g7-gpu-texture-benchmark.v1",
        "data_path": str(Path(data_path).resolve()),
        "power_preference": power_preference,
        "git_revision": _git_rev(),
        "report": asdict(report),
    }


def _git_rev() -> str:
    try:
        return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    except Exception:
        return "unknown"


def _format(result: dict) -> str:
    r = result["report"]
    lines = [
        f"data: {result['data_path']}",
        f"adapter: {r['adapter']} | {r['adapter_type']} | {r['backend']}  (pref={result['power_preference']})",
        f"bc={r['bc_supported']} astc={r['astc_supported']}  tile={r['tile']}x{r['tile']}",
        "-" * 96,
        f"{'kind':<8}{'format':<16}{'enc':<5}{'raw_B':>10}{'up_B':>9}{'vram_B':>9}"
        f"{'ratio':>7}{'PSNR':>8}{'maxabs':>9}{'magPSNR':>9}{'phSig':>8}{'phWt':>7}{'hwPSNR':>8}",
        "-" * 100,
    ]
    for c in r["cells"]:
        lines.append(
            f"{c['kind']:<8}{c['format']:<16}{c['encoder']:<5}{c['raw_bytes']:>10}"
            f"{c['upload_bytes']:>9}{c['vram_resident_bytes']:>9}{c['vram_ratio']:>7.2f}"
            f"{c['psnr_db']:>8.1f}{c['max_abs']:>9.4f}{c['magnitude_psnr_db']:>9.1f}"
            f"{c['phase_max_abs_significant_rad']:>8.3f}{c['phase_weighted_rmse_rad']:>7.3f}"
            f"{c['hardware_sample_psnr_db']:>8.1f}"
        )
        if c["notes"]:
            lines.append(f"    note: {c['notes']}")
    lines.append("-" * 100)
    lines.append(
        "ratio=raw/vram; PSNR/maxabs are scalar [0,1]-field quality; magPSNR/phSig/phWt\n"
        "are complex DISPLAY-domain: magnitude PSNR dB, worst phase err (rad) over\n"
        "significant (>=10% peak-magnitude) pixels, and magnitude-weighted phase RMSE\n"
        "(rad).  Phase is meaningless where magnitude~=0 (near-black), so unweighted\n"
        "max phase err ~pi there is not a displayed defect.  hwPSNR = hardware-sampler\n"
        "vs reference-decode (BC scalar closure)."
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="G7 Phase B GPU compressed-texture benchmark")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    p.add_argument(
        "--power-preference",
        choices=("high-performance", "low-power"),
        default="high-performance",
    )
    p.add_argument("--json", type=Path, default=None)
    return p


def main(argv: tuple[str, ...] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_benchmark(data_path=Path(args.data), power_preference=args.power_preference)
    except GpuDecodeUnavailable as exc:
        print(f"GPU texture benchmark unavailable: {exc}")
        return 2
    print(_format(result))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
