"""Adaptive policy for the G7 compressed host cache (Phase A, RAM axis).

"Always optimal for the RAM axis" means: engage the compressed backing tier
*exactly when it strictly helps* and stay out of the way otherwise.  The lever
is RAM pressure -- the tier only earns its keep once the working set exceeds the
cache budget, because that is when evictions (and the recompute/re-read they
force) actually happen.  When the working set already fits, the tier would only
add compression CPU for no eviction avoided, so the policy leaves it OFF and
tiny workloads pay nothing.

Inputs the decision is a function of:

* working-set size vs the cache byte budget -> engage only under RAM pressure;
* dtype -> pick the codec that round-trips that dtype losslessly and compresses
  it best (zfp's transform for float/complex/int16; blosc2's byte codec for the
  dtypes zfp declines, e.g. uint8);
* topology -> both integrated and discrete benefit on the RAM axis, so topology
  does NOT gate engagement here.  It is carried on the decision as the Phase-B
  seam: a discrete PCIe device additionally cares about host->VRAM transfer
  bytes (GPU-side decode), which Phase B will decide.

Losslessness is non-negotiable: the policy only ever selects a lossless codec,
and it double-checks via :func:`resolve_codec` (which degrades to ``raw`` rather
than lose pixels).  A lossy mode is never chosen automatically.

Justification for auto-enabling on large data even though the prior transport
benchmark did not win: that benchmark measured *transfer time*
(compress+transfer+decompress vs raw transfer) on a fast PCIe link, where CPU
decode loses.  The RAM/eviction win is a *different, measured* axis -- a decode
(microseconds) replaces a recompute/re-read (an FFT or a disk page, milliseconds
to seconds).  ``g7_cache_benchmark`` quantifies the crossover and the end-to-end
win; the policy engages only past that crossover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arrayscope.gpu.chunk_codec import resolve_codec
from arrayscope.gpu.device_topology import DeviceTopology, detect_topology

__all__ = [
    "CompressedTierDecision",
    "TextureCodecDecision",
    "decide_compressed_tier",
    "decide_texture_codec",
    "preferred_codec_for_dtype",
]

# Codec preference per dtype, best-ratio-lossless first.  ``resolve_codec`` is
# the final gate: any preference that cannot losslessly represent the dtype (or
# whose dependency is missing) degrades to ``raw`` and the next preference is
# tried, so correctness never depends on this table.
_ZFP_FRIENDLY = frozenset(
    np.dtype(t)
    for t in (
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
        np.int16,
        np.int32,
        np.int64,
    )
)


def preferred_codec_for_dtype(dtype) -> tuple[str, ...]:
    """Ordered lossless-codec preference for ``dtype`` (best ratio first)."""

    dtype = np.dtype(dtype)
    if dtype in _ZFP_FRIENDLY:
        # zfp's transform beats a byte codec on numeric arrays; blosc2 backs it
        # up for dtypes/deps zfp cannot cover.
        return ("zfp", "blosc2")
    # uint8 and other dtypes zfp declines: blosc2's byte codec is exact for all.
    return ("blosc2",)


@dataclass(frozen=True)
class CompressedTierDecision:
    """Whether/how to engage the compressed backing tier for a workload."""

    engage: bool
    codec_name: str
    reason: str
    working_set_bytes: int
    budget_bytes: int
    topology: DeviceTopology
    # Phase-B seam: a discrete PCIe device where compressing host->VRAM transfer
    # bytes could additionally pay once a GPU-side decoder exists.  Phase A does
    # not read this to decide the RAM win; it is here for the transfer decision.
    discrete_transfer_candidate: bool = False

    @property
    def pressure_ratio(self) -> float:
        if self.budget_bytes <= 0:
            return float("inf")
        return float(self.working_set_bytes) / float(self.budget_bytes)


@dataclass(frozen=True)
class TextureCodecDecision:
    """Which native compressed-texture format to use for the display path.

    This is the Phase-B *transfer/VRAM* decision (distinct from the Phase-A RAM
    decision above), and it is **topology-driven**:

    * discrete GPU (PCIe) -- the ``discrete_transfer_candidate`` seam: compressing
      the host->VRAM texture pays twice (fewer PCIe bytes AND less VRAM resident,
      which matters on the A2000's 4 GB).  NVIDIA exposes only BC, so scalar tiles
      use BC4 and two-channel complex tiles use BC5 (holding real, imag).
    * integrated GPU (unified memory) -- there is no PCIe transfer to shrink, but
      the *VRAM residency* win still applies, and Intel additionally advertises
      ASTC, which has a flexible block-size/quality knob and better two-channel
      handling.  So integrated prefers ASTC when available, else BC.

    Both decode for free in the hardware sampler (no decode pass).  The format is
    lossy, so this is **default OFF**: ``engage`` is False unless a caller opts in
    (``enable=True``) AND a compressed-texture path is actually available.  The
    default wgpu render path is byte-identical while ``engage`` is False.
    """

    engage: bool
    family: str  # "bc", "astc", or "none"
    scalar_format: str  # e.g. "bc4-r-unorm" / "astc-6x6-unorm" / "r32float"
    complex_format: str  # e.g. "bc5-rg-unorm" / "astc-4x4-unorm" / "rg32float"
    astc_block: tuple[int, int] | None
    topology: DeviceTopology
    reason: str

    @property
    def discrete_transfer_candidate(self) -> bool:
        return self.topology.discrete_transfer_candidate


def decide_texture_codec(
    *,
    topology: DeviceTopology | None = None,
    enable: bool = False,
    astc_supported: bool | None = None,
    astc_block: tuple[int, int] = (6, 6),
) -> TextureCodecDecision:
    """Pick the display-path texture format for a topology.  Default OFF.

    ``enable`` must be True to engage (the path is lossy and opt-in).
    ``astc_supported`` overrides adapter/library ASTC detection (tests); when None
    it is inferred from the ``astc_encoder`` dependency and the integrated adapter
    advertising ASTC.  On a discrete GPU BC is always chosen (NVIDIA has no ASTC);
    on integrated, ASTC is preferred when supported, else BC.
    """

    topology = topology or detect_topology()

    if not enable:
        return TextureCodecDecision(
            engage=False,
            family="none",
            scalar_format="r32float",
            complex_format="rg32float",
            astc_block=None,
            topology=topology,
            reason="texture codec disabled (default): render path byte-identical",
        )

    if astc_supported is None:
        # astc_codec imports cleanly (it guards the optional astc_encoder dep
        # internally), so this internal import is never swallowed.
        from arrayscope.gpu.astc_codec import astc_available

        astc_supported = astc_available() and topology.is_integrated

    if topology.is_discrete:
        return TextureCodecDecision(
            engage=True,
            family="bc",
            scalar_format="bc4-r-unorm",
            complex_format="bc5-rg-unorm",
            astc_block=None,
            topology=topology,
            reason=(
                "discrete PCIe GPU: BC compresses host->VRAM transfer AND VRAM "
                "residency (BC4 scalar, BC5 (re,im) complex); NVIDIA has no ASTC"
            ),
        )

    if astc_supported:
        return TextureCodecDecision(
            engage=True,
            family="astc",
            scalar_format=f"astc-{astc_block[0]}x{astc_block[1]}-unorm",
            complex_format=f"astc-{astc_block[0]}x{astc_block[1]}-unorm",
            astc_block=astc_block,
            topology=topology,
            reason=(
                "integrated GPU: no PCIe transfer to cut, but ASTC saves VRAM with "
                f"a tunable block ({astc_block[0]}x{astc_block[1]}) and 2-channel fit"
            ),
        )

    return TextureCodecDecision(
        engage=True,
        family="bc",
        scalar_format="bc4-r-unorm",
        complex_format="bc5-rg-unorm",
        astc_block=None,
        topology=topology,
        reason="integrated GPU without ASTC: BC saves VRAM residency (free decode)",
    )


def _first_lossless_codec(dtype, preferences) -> str:
    """Return the first preference that ``resolve_codec`` keeps (else ``raw``)."""

    for name in preferences:
        codec = resolve_codec(name, dtype)
        # resolve_codec only ever returns a lossless codec; if it kept our
        # requested name the codec is available and covers this dtype exactly.
        if codec.name == name:
            return name
    return "raw"


def decide_compressed_tier(
    *,
    working_set_bytes: int,
    budget_bytes: int,
    dtype,
    topology: DeviceTopology | None = None,
    pressure_margin: float = 1.0,
) -> CompressedTierDecision:
    """Decide whether to engage the compressed tier, and with which codec.

    Engages only under RAM pressure (``working_set_bytes`` exceeds
    ``budget_bytes * pressure_margin``) AND when a lossless codec can compress
    ``dtype`` (otherwise the tier cannot help and stays off).  Topology never
    gates the RAM decision; it is recorded, and the discrete-transfer seam is set
    for Phase B.
    """

    topology = topology or detect_topology()
    dtype = np.dtype(dtype)
    working_set_bytes = int(max(0, working_set_bytes))
    budget_bytes = int(max(0, budget_bytes))
    discrete_seam = topology.discrete_transfer_candidate

    fits = working_set_bytes <= budget_bytes * float(pressure_margin)
    if fits:
        return CompressedTierDecision(
            engage=False,
            codec_name="raw",
            reason=(
                f"working set {working_set_bytes}B fits budget {budget_bytes}B "
                f"(x{pressure_margin:g}); no eviction to avoid, tier stays off"
            ),
            working_set_bytes=working_set_bytes,
            budget_bytes=budget_bytes,
            topology=topology,
            discrete_transfer_candidate=discrete_seam,
        )

    codec_name = _first_lossless_codec(dtype, preferred_codec_for_dtype(dtype))
    if codec_name == "raw":
        return CompressedTierDecision(
            engage=False,
            codec_name="raw",
            reason=(
                f"under RAM pressure but no lossless codec covers dtype {dtype} "
                f"(zfpy/blosc2 unavailable or dtype unsupported); tier cannot help"
            ),
            working_set_bytes=working_set_bytes,
            budget_bytes=budget_bytes,
            topology=topology,
            discrete_transfer_candidate=discrete_seam,
        )

    return CompressedTierDecision(
        engage=True,
        codec_name=codec_name,
        reason=(
            f"RAM pressure (working set {working_set_bytes}B > budget "
            f"{budget_bytes}B): engage {codec_name} (lossless) to retain more of "
            f"the working set and cut expensive misses; topology={topology.kind}"
        ),
        working_set_bytes=working_set_bytes,
        budget_bytes=budget_bytes,
        topology=topology,
        discrete_transfer_candidate=discrete_seam,
    )
