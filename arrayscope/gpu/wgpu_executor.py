"""wgpu implementation of the renderer command protocol (gate-B seed, grown).

First real implementation of :mod:`arrayscope.gpu.command_protocol`, grown
from the proven gate-B harness (``experiments/wgpu_gate_b/virtual_tensor.py``
oracles A–G) to the live payload shapes (queue row 3a):

- **Multi-plane content binding**: :class:`BindContentPlanes` declares the
  set of 2-D plane pyramids tiles may sample (montage/session unit); the GPU
  flat page table + LOD spans are rebuilt from the *currently bound* planes,
  while the content-keyed :class:`~arrayscope.gpu.page_table.PageTable`
  keeps chunks of unbound planes warm (scroll-back across planes is a
  zero-upload rebind).
- **Honest pools per representation**: ``scalar_r32f`` chunks live in an
  ``r32float`` pool (no zero-imaginary waste), ``complex_rg32f`` in
  ``rg32float``, ``rgb8`` in ``rgba8unorm`` (display-ready: sampled as-is,
  levels/LUT bypassed), and windowable RGB in ``rgba32float`` (VisPy-faithful
  color RGB + scalar-window signal in alpha). Pool budgets are per
  representation; LRU eviction touches unpinned pages of the same pool only
  and pinned exhaustion is a loud error.  Keys a submission's own
  ``DispatchHistogram`` commands reference are shielded from that
  submission's eviction pressure (pre-scan pin, released before ``submit``
  returns); when pool pressure exceeds even the shield, the shielded page
  is yielded and the histogram reports the key via
  ``FrameReport.histogram_missing`` instead of aborting the batch.
- **Mapping correctness**: scalar planes ignore complex mapping modes (the
  value *is* the scalar); complex planes keep magnitude/phase/real/imag;
  both apply validated linear/log/symlog scales before levels/LUT mapping;
  ``DispatchHistogram`` reduces scalar pages too (magnitude == value).

The executor renders offscreen into its own target by default
(``read_target()`` is the test/audit oracle); a live canvas hands in a
texture view via ``present_to`` each frame (the bitmap-mode preview tool
does exactly that).  ``import wgpu`` happens lazily so this module is
importable everywhere; construction raises cleanly when wgpu or a Vulkan
adapter is unavailable.

Residency bookkeeping reuses :class:`arrayscope.gpu.page_table.PageTable`
(slot = pool layer); the GPU-side flat table mirrors it per submission.
"""

from __future__ import annotations

import contextlib
import struct
from dataclasses import dataclass, field

import numpy as np

from arrayscope.gpu import astc_codec, bc_codec
from arrayscope.gpu.command_protocol import (
    BindContentPlanes,
    DispatchHistogram,
    DisplayMapping,
    EnsureChunkResident,
    EvictChunk,
    FrameReport,
    FrameSubmission,
    GenerateLodPages,
    OverlayPrimitive,
    PresentGeneration,
    SetDisplayMapping,
    SetOverlayCamera,
    UpdateGlyphAtlas,
    UpdateOverlayGeometry,
    UpdateTileInstances,
    UpdateWidgetAtlas,
)
from arrayscope.gpu.keys import (
    COMPLEX_RG32F,
    REDUCER_MEAN,
    REPRESENTATIONS,
    RGB8,
    RGB_WINDOWED_RGBA32F,
    SCALAR_R32F,
    ChunkLod,
    DataChunkKey,
)
from arrayscope.gpu.page_table import PageSlot, PageTable

PAGE = 256
MAX_HISTOGRAM_BINS = 512

_MODE_INDEX = {"magnitude": 0, "phase": 1, "real": 2, "imag": 3}
_SCALE_INDEX = {"linear": 0, "log": 1, "symlog": 2}
_OVERLAY_KIND_INDEX = {
    "line": 0,
    "world_rect": 1,
    "handle_quad": 2,
    "screen_rect": 3,
    "glyph_quad": 4,
    "widget_quad": 5,
}
#: Bytes per packed overlay instance (must mirror the WGSL Overlay struct).
_OVERLAY_INSTANCE_BYTES = 96

#: Shader-side representation flags (PlaneInfo.rep / histogram layer entries).
_REP_INDEX = {
    SCALAR_R32F: 0,
    COMPLEX_RG32F: 1,
    RGB8: 2,
    RGB_WINDOWED_RGBA32F: 3,
}

_POOL_FORMATS = {
    SCALAR_R32F: "r32float",
    COMPLEX_RG32F: "rg32float",
    RGB8: "rgba8unorm",
    RGB_WINDOWED_RGBA32F: "rgba32float",
}
_POOL_TEXEL_BYTES = {
    SCALAR_R32F: 4,
    COMPLEX_RG32F: 8,
    RGB8: 4,
    RGB_WINDOWED_RGBA32F: 16,
}
_POOL_IDS = {rep: f"wgpu-{rep}-pool" for rep in REPRESENTATIONS}
_REP_BY_POOL_ID = {pool_id: rep for rep, pool_id in _POOL_IDS.items()}

#: Native block-compressed pool ids, parallel to the raw pools above.  A scalar
#: page compresses to one channel (BC4 / ASTC R), a complex page to two channels
#: (BC5, or ASTC R,G holding real, imag).  These live in *separate* textures (a
#: texture has one format), so the page table's ``pool_id`` is what tells the
#: render/histogram shaders which pool actually holds a page.  The pool id is
#: family-agnostic ("-bc-pool" is historical); the format the pool was created
#: with (``codec_pool_format``) is the loud channel proving BC vs ASTC.
_CODEC_POOL_IDS = {rep: f"wgpu-{rep}-bc-pool" for rep in (SCALAR_R32F, COMPLEX_RG32F)}
_REP_BY_POOL_ID.update({pool_id: rep for rep, pool_id in _CODEC_POOL_IDS.items()})

#: wgpu device features that must be enabled for the two codec families to exist.
_BC_FEATURE = "texture-compression-bc"
_ASTC_FEATURE = "texture-compression-astc"

#: BC family: scalar->BC4 (8 bytes / 4x4 block), complex->BC5 (16 bytes / block).
#: A 256² page is 64×64 blocks.  The discrete (NVIDIA) adapter exposes only BC.
_BC_POOL_FORMATS = {SCALAR_R32F: "bc4-r-unorm", COMPLEX_RG32F: "bc5-rg-unorm"}
_BC_BLOCK = (4, 4)
_BC_BLOCK_BYTES = {SCALAR_R32F: 8, COMPLEX_RG32F: 16}

#: ASTC family (integrated/Intel): a single ``astc-BxB-unorm`` format carries
#: both scalar (value in R, replicated) and complex (real, imag in R, G) — every
#: ASTC block is 16 bytes regardless of size.  wgpu rejects a compressed texture
#: whose width/height is not a multiple of the block, and the 256² page geometry
#: (uv = (texel+0.5)/256) assumes a 256² texture, so the block MUST divide 256:
#: 4x4 (default; highest quality, scalar 4x / complex 8x) or 8x8 (higher ratio,
#: scalar 16x / complex 32x, lower quality).  6x6 does NOT divide 256 and is
#: unavailable for the display pool even though the policy may name it.
_ASTC_DEFAULT_BLOCK = (4, 4)

# Texture-array extents are immutable in WebGPU, but the raw and codec pools do
# not need to reserve their full logical eviction budgets up front.  Start with
# a small usable extent and grow geometrically, copying the old layers at the
# queue boundary.  Eight pages keeps tiny/incremental commits copy-free while
# avoiding the former full raw + full codec mirror for large montages.
_POOL_INITIAL_LAYERS = 8


def _adapter_info(adapter_or_none) -> dict:
    """Best-effort ``adapter.info`` dict; ``{}`` when unavailable (never raises)."""

    if adapter_or_none is None:
        return {}
    try:
        return dict(adapter_or_none.info)
    except Exception:
        return {}


def _adapter_is_integrated(info: dict) -> bool:
    """True unless the adapter is explicitly discrete (unknown -> integrated).

    Mirrors :mod:`arrayscope.gpu.device_topology`: only a ``DiscreteGPU`` (the
    NVIDIA A2000) is treated as discrete; integrated/CPU/virtual/unknown are the
    ASTC-eligible, unified-memory side.
    """

    return "discrete" not in str(info.get("adapter_type", "")).lower()


def _decide_codec_family(mode: str, device_features: set[str], integrated: bool) -> str:
    """Pick the display-pool codec family for a device.  Mirrors the topology
    policy in :func:`arrayscope.gpu.cache_policy.decide_texture_codec`:

    * ``off`` -> ``"none"`` (render path byte-identical);
    * integrated + ASTC feature + ``astc_encoder`` available -> ``"astc"``
      (Intel: ASTC is the better fit and only offered there);
    * else BC when the device advertises it (discrete NVIDIA, or integrated
      without ASTC);
    * else ``"none"`` (no compressed format at all -> degrade to raw).
    """

    if mode == "off":
        return "none"
    astc_ok = _ASTC_FEATURE in device_features and astc_codec.astc_available()
    bc_ok = _BC_FEATURE in device_features
    if integrated and astc_ok:
        return "astc"
    if bc_ok:
        return "bc"
    if astc_ok:
        return "astc"
    return "none"


_BOUND_PLANES_PIN_OWNER = "wgpu-bound-content-planes"
_LOD_GENERATION_PIN_OWNER = "wgpu-lod-generation-sources"
_HISTOGRAM_SHIELD_PIN_OWNER = "wgpu-histogram-frontier-shield"


def _ordered_float32(value: float) -> int:
    bits = int(np.asarray(np.float32(value)).view(np.uint32))
    return int((~bits) & 0xFFFFFFFF) if bits & 0x80000000 else int(bits ^ 0x80000000)


def _float32_from_ordered(value: int) -> float:
    value = int(value) & 0xFFFFFFFF
    bits = ((~value) & 0xFFFFFFFF) if value < 0x80000000 else (value ^ 0x80000000)
    return float(np.asarray(np.uint32(bits)).view(np.float32))


def plane_chunk_key(
    document_generation: object,
    operation_key: object,
    lod_level: int,
    chunk_x: int,
    chunk_y: int,
    *,
    dtype: str = "complex64",
    representation: str = COMPLEX_RG32F,
    plane_shape: tuple[int, int] | None = None,
    reducer: str = REDUCER_MEAN,
) -> DataChunkKey:
    """Canonical key for one 256² page of a 2-D plane pyramid.

    Key geometry is canonical native-source space (ADR 0056).  The physical
    page remains 256² stored samples; ``lod.reduction`` tells the executor how
    to map this source footprint onto that page.
    """

    if lod_level == 0:
        lod = ChunkLod()
    else:
        lod = ChunkLod(
            level=lod_level,
            factor=1 << lod_level,
            reduction=(lod_level, lod_level),
            reducer=str(reducer),
        )
    factor = 1 << int(lod_level)
    origin = (chunk_y * PAGE * factor, chunk_x * PAGE * factor)
    shape = (PAGE * factor, PAGE * factor)
    if plane_shape is not None:
        plane_h, plane_w = (int(value) for value in plane_shape)
        shape = (
            min(shape[0], max(0, plane_h - origin[0])),
            min(shape[1], max(0, plane_w - origin[1])),
        )
        if any(value <= 0 for value in shape):
            raise ValueError(
                f"chunk ({chunk_y}, {chunk_x}) at LOD {lod_level} lies outside "
                f"plane shape {plane_shape}"
            )
    return DataChunkKey(
        document_generation=document_generation,
        operation_key=operation_key,
        lod=lod,
        chunk_origin=origin,
        chunk_shape=shape,
        dtype=dtype,
        representation=representation,
    )


_RENDER_WGSL = """
struct Mapping {
    mode: u32,
    scale: u32,
    level_lo: f32,
    level_hi: f32,
    symlog_constant: f32,
    phase_color: u32,
    pixel_grid: u32,
    clip_indicator: u32,
};
struct LodInfo { base: u32, grid_w: u32, grid_h: u32, _pad: u32 };
struct PlaneInfo { rep: u32, max_lod: u32, lod_base: u32, _pad: u32 };
struct Tile {
    dst: vec4<f32>,
    src: vec4<f32>,
    lod: u32,
    plane: u32,
    transposed: u32, _pad2: u32,
};
struct TileCamera {
    scale: vec2<f32>,
    offset: vec2<f32>,
    target_size: vec2<f32>,
    _pad: vec2<f32>,
};
@group(0) @binding(0) var<uniform> mapping: Mapping;
@group(0) @binding(1) var<storage, read> page_table: array<i32>;
@group(0) @binding(2) var<storage, read> lod_info: array<LodInfo>;
@group(0) @binding(3) var<storage, read> planes: array<PlaneInfo>;
@group(0) @binding(4) var<storage, read> tiles: array<Tile>;
@group(0) @binding(5) var scalar_pool: texture_2d_array<f32>;
@group(0) @binding(6) var complex_pool: texture_2d_array<f32>;
@group(0) @binding(7) var rgb_pool: texture_2d_array<f32>;
@group(0) @binding(8) var rgb_windowed_pool: texture_2d_array<f32>;
@group(0) @binding(9) var lut: texture_2d<f32>;
@group(0) @binding(10) var<uniform> camera: TileCamera;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) src: vec2<f32>,
    @location(1) @interpolate(flat) lod: u32,
    @location(2) @interpolate(flat) plane: u32,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0));
    let t = tiles[ii];
    let q = quad[vi];
    // World space in, NDC out: a pure pan only rewrites the camera uniform,
    // never the per-tile instances (O(1) instead of O(tiles) per frame).
    let world = t.dst.xy + q * t.dst.zw;
    let ndc = world * camera.scale + camera.offset;
    var out: VOut;
    out.pos = vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
    // A transpose is a pure display swap: the source plane is stored
    // canonically, so walk its UV axes swapped (q.yx) while the display quad
    // (t.dst) stays in screen orientation.
    out.src = t.src.xy + select(q, q.yx, t.transposed != 0u) * t.src.zw;
    out.lod = t.lod;
    out.plane = t.plane;
    return out;
}

struct Resolved { layer: i32, texel: vec2<i32> };

fn resolve(plane_index: u32, src_l0: vec2<f32>, lod_req: u32) -> Resolved {
    let p = planes[plane_index];
    for (var lod = lod_req; lod <= p.max_lod; lod = lod + 1u) {
        let info = lod_info[p.lod_base + lod];
        let scale = f32(1u << lod);
        let limit = vec2<f32>(f32(info.grid_w * 256u) - 1.0, f32(info.grid_h * 256u) - 1.0);
        let coord = vec2<u32>(clamp(src_l0 / scale, vec2<f32>(0.0), limit));
        let chunk = coord / 256u;
        let entry = page_table[info.base + chunk.y * info.grid_w + chunk.x];
        if (entry >= 0) {
            return Resolved(entry, vec2<i32>(coord % 256u));
        }
    }
    return Resolved(-1, vec2<i32>(0, 0));
}

fn apply_scale(value: f32) -> f32 {
    switch mapping.scale {
        case 0u: { return value; }
        case 1u: { return log(max(value, 0.0)) / log(10.0); }
        default: {
            return sign(value) * log(
                1.0 + abs(value) / pow(10.0, mapping.symlog_constant)
            ) / log(10.0);
        }
    }
}

fn finite_scalar(x: f32) -> bool {
    // No WGSL isFinite: a NaN fails self-equality; +/-Inf exceeds f32 max.
    return x == x && abs(x) <= 3.402823466e+38;
}

// A2: non-finite source -> high-contrast black/white diagonal hatch (45deg).
// Alternating opaque black and white reads against every colormap, so a
// NaN/Inf texel can never fall through clamp() into an arbitrary LUT entry.
fn nan_marker(pos: vec2<f32>) -> vec4<f32> {
    let stripe = fract((pos.x + pos.y) / 8.0);
    let shade = select(1.0, 0.0, stripe < 0.5);
    return vec4<f32>(shade, shade, shade, 1.0);
}

// A3: missing page -> dim gray hatch at the OPPOSITE angle (-45deg) and very
// low contrast, so "not loaded yet" never reads as an actual zero value.
fn missing_marker(pos: vec2<f32>) -> vec4<f32> {
    let stripe = fract((pos.x - pos.y) / 8.0);
    let shade = select(0.20, 0.12, stripe < 0.5);
    return vec4<f32>(shade, shade, shade, 1.0);
}

// A1: zoom-gated pixel grid, applied once to the final colour.  `fw` is
// fwidth(in.src) -- source texels per screen pixel -- computed in uniform
// control flow at the top of fs_main (WGSL derivatives require that, and
// resolve() returns early).  `1/fw` is screen px per texel; the grid fades
// from nothing below 12 px/texel to full above 24, so a normally-zoomed
// scene is returned byte-identical (fade == 0 -> multiply by 1.0).
fn pixel_grid(color: vec4<f32>, src: vec2<f32>, fw: vec2<f32>, enabled: u32) -> vec4<f32> {
    if (enabled == 0u) { return color; }
    let fwc = max(fw, vec2<f32>(1e-8, 1e-8));
    let px_per_texel = 1.0 / max(fwc.x, fwc.y);
    let fade = smoothstep(12.0, 24.0, px_per_texel);
    let edge = min(fract(src), vec2<f32>(1.0) - fract(src)) / fwc;
    let line = 1.0 - smoothstep(0.0, 1.0, min(edge.x, edge.y));
    let darken = 0.2 * line * fade;
    return vec4<f32>(color.rgb * (1.0 - darken), color.a);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    // A1: derivatives require uniform control flow, so take fwidth BEFORE
    // resolve()'s data-dependent early returns.  Only consumed by the
    // zoom-gated pixel grid at the single exit below.
    // A1: derivatives require uniform control flow, so take fwidth BEFORE
    // resolve()'s data-dependent early returns.  Only consumed by the
    // zoom-gated pixel grid at the single exit below.
    let fw = fwidth(in.src);
    let p = planes[in.plane];
    let r = resolve(in.plane, in.src, in.lod);
    var color: vec4<f32>;
    if (r.layer < 0) {
        // A3: page not resident at any LOD -> distinct missing-page hatch.
        color = missing_marker(in.pos.xy);
    } else if (p.rep == 2u) {
        // Display-ready RGB: sampled as-is, levels/LUT bypassed.
        let c = textureLoad(rgb_pool, r.texel, r.layer, 0);
        color = vec4<f32>(c.rgb, 1.0);
    } else if (p.rep == 3u) {
        // VisPy parity: preserve the color plane and modulate it by one
        // levels-normalized scalar plane (packed in alpha), not by three
        // independent per-channel windows.
        let c = textureLoad(rgb_windowed_pool, r.texel, r.layer, 0);
        let scalar = apply_scale(c.a);
        let intensity = clamp(
            (scalar - mapping.level_lo) / (mapping.level_hi - mapping.level_lo),
            0.0,
            1.0,
        );
        color = vec4<f32>(c.rgb * intensity, 1.0);
    } else {
        var v = vec2<f32>(0.0, 0.0);
        if (p.rep == 0u) {
            v = vec2<f32>(textureLoad(scalar_pool, r.texel, r.layer, 0).r, 0.0);
        } else {
            v = textureLoad(complex_pool, r.texel, r.layer, 0).rg;
        }
        if (!finite_scalar(v.x) || !finite_scalar(v.y)) {
            // A2: a non-finite source texel is a bad-data signal, not a value.
            color = nan_marker(in.pos.xy);
        } else {
            color = map_value(p, v);
        }
    }
    // A1: applied once to the final colour; identity at normal zoom.
    return pixel_grid(color, in.src, fw, mapping.pixel_grid);
}

// Scalar/complex value -> displayed colour: the mode/scale/levels/LUT path,
// plus the A4 clip markers.  `v` is the resident, finite source sample.
fn map_value(p: PlaneInfo, v: vec2<f32>) -> vec4<f32> {
    var x: f32;
    if (p.rep == 0u) {
        // Scalar planes ignore complex mapping modes: the value IS the scalar.
        x = v.x;
    } else {
        switch mapping.mode {
            case 0u: { x = length(v); }
            case 1u: { x = atan2(v.y, v.x); }
            case 2u: { x = v.x; }
            default: { x = v.y; }
        }
    }
    x = apply_scale(x);
    let g = clamp((x - mapping.level_lo) / (mapping.level_hi - mapping.level_lo), 0.0, 1.0);
    let phase_path = mapping.phase_color != 0u && p.rep == 1u && mapping.mode != 1u;
    // A4 (default off): mark values outside the levels window distinctly so
    // clipping is visible while windowing.  Skipped for the phase-colour path.
    if (mapping.clip_indicator != 0u && !phase_path) {
        if (x < mapping.level_lo) { return vec4<f32>(0.0, 0.2, 0.8, 1.0); }
        if (x > mapping.level_hi) { return vec4<f32>(0.9, 0.1, 0.0, 1.0); }
    }
    if (phase_path) {
        let phase = atan2(v.y, v.x);
        let phase_g = clamp(
            (phase + 3.141592653589793) / 6.283185307179586,
            0.0,
            1.0,
        );
        let phase_idx = clamp(i32(round(phase_g * 255.0)), 0, 255);
        let lut_color = textureLoad(lut, vec2<i32>(phase_idx, 0), 0);
        return vec4<f32>(lut_color.rgb * g, lut_color.a);
    }
    // Nearest-entry LUT indexing, mirroring the CPU display reference.
    let idx = clamp(i32(round(g * 255.0)), 0, 255);
    return textureLoad(lut, vec2<i32>(idx, 0), 0);
}
"""

#: Render shader with native block-compressed pools wired in.  Identical to
#: ``_RENDER_WGSL`` except: (a) three extra bindings — a per-flat-entry codec
#: flag buffer, a per-flat-entry (lo, span) normalization buffer, a nearest
#: sampler, and the BC4/BC5 pools; (b) ``resolve`` also returns the flat index
#: it hit so the fragment can look up that page's codec + norm; (c) scalar and
#: complex reads branch: a compressed page is decoded by the *hardware sampler*
#: (``textureSampleLevel``, normalized coords) and unscaled by the page's per-
#: tile (lo, span), while a raw page keeps the exact integer ``textureLoad``.
#: The compressed path is only ever selected for pages the executor actually
#: stored compressed (codec flag == 1); every raw page renders byte-identically
#: to the base shader.
_RENDER_WGSL_COMPRESSED = """
struct Mapping {
    mode: u32,
    scale: u32,
    level_lo: f32,
    level_hi: f32,
    symlog_constant: f32,
    phase_color: u32,
    pixel_grid: u32,
    clip_indicator: u32,
};
struct LodInfo { base: u32, grid_w: u32, grid_h: u32, _pad: u32 };
struct PlaneInfo { rep: u32, max_lod: u32, lod_base: u32, _pad: u32 };
struct Tile {
    dst: vec4<f32>,
    src: vec4<f32>,
    lod: u32,
    plane: u32,
    transposed: u32, _pad2: u32,
};
struct TileCamera {
    scale: vec2<f32>,
    offset: vec2<f32>,
    target_size: vec2<f32>,
    _pad: vec2<f32>,
};
@group(0) @binding(0) var<uniform> mapping: Mapping;
@group(0) @binding(1) var<storage, read> page_table: array<i32>;
@group(0) @binding(2) var<storage, read> lod_info: array<LodInfo>;
@group(0) @binding(3) var<storage, read> planes: array<PlaneInfo>;
@group(0) @binding(4) var<storage, read> tiles: array<Tile>;
@group(0) @binding(5) var scalar_pool: texture_2d_array<f32>;
@group(0) @binding(6) var complex_pool: texture_2d_array<f32>;
@group(0) @binding(7) var rgb_pool: texture_2d_array<f32>;
@group(0) @binding(8) var rgb_windowed_pool: texture_2d_array<f32>;
@group(0) @binding(9) var lut: texture_2d<f32>;
@group(0) @binding(10) var<uniform> camera: TileCamera;
@group(0) @binding(11) var<storage, read> page_codec: array<u32>;
@group(0) @binding(12) var<storage, read> page_norm: array<vec4<f32>>;
@group(0) @binding(13) var codec_samp: sampler;
@group(0) @binding(14) var scalar_bc_pool: texture_2d_array<f32>;
@group(0) @binding(15) var complex_bc_pool: texture_2d_array<f32>;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) src: vec2<f32>,
    @location(1) @interpolate(flat) lod: u32,
    @location(2) @interpolate(flat) plane: u32,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0));
    let t = tiles[ii];
    let q = quad[vi];
    let world = t.dst.xy + q * t.dst.zw;
    let ndc = world * camera.scale + camera.offset;
    var out: VOut;
    out.pos = vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
    // A transpose is a pure display swap: the source plane is stored
    // canonically, so walk its UV axes swapped (q.yx) while the display quad
    // (t.dst) stays in screen orientation.
    out.src = t.src.xy + select(q, q.yx, t.transposed != 0u) * t.src.zw;
    out.lod = t.lod;
    out.plane = t.plane;
    return out;
}

struct Resolved { layer: i32, texel: vec2<i32>, fidx: i32 };

fn resolve(plane_index: u32, src_l0: vec2<f32>, lod_req: u32) -> Resolved {
    let p = planes[plane_index];
    for (var lod = lod_req; lod <= p.max_lod; lod = lod + 1u) {
        let info = lod_info[p.lod_base + lod];
        let scale = f32(1u << lod);
        let limit = vec2<f32>(f32(info.grid_w * 256u) - 1.0, f32(info.grid_h * 256u) - 1.0);
        let coord = vec2<u32>(clamp(src_l0 / scale, vec2<f32>(0.0), limit));
        let chunk = coord / 256u;
        let fi = info.base + chunk.y * info.grid_w + chunk.x;
        let entry = page_table[fi];
        if (entry >= 0) {
            return Resolved(entry, vec2<i32>(coord % 256u), i32(fi));
        }
    }
    return Resolved(-1, vec2<i32>(0, 0), -1);
}

fn apply_scale(value: f32) -> f32 {
    switch mapping.scale {
        case 0u: { return value; }
        case 1u: { return log(max(value, 0.0)) / log(10.0); }
        default: {
            return sign(value) * log(
                1.0 + abs(value) / pow(10.0, mapping.symlog_constant)
            ) / log(10.0);
        }
    }
}

fn finite_scalar(x: f32) -> bool {
    // No WGSL isFinite: a NaN fails self-equality; +/-Inf exceeds f32 max.
    return x == x && abs(x) <= 3.402823466e+38;
}

// A2: non-finite source -> high-contrast black/white diagonal hatch (45deg).
// Alternating opaque black and white reads against every colormap, so a
// NaN/Inf texel can never fall through clamp() into an arbitrary LUT entry.
fn nan_marker(pos: vec2<f32>) -> vec4<f32> {
    let stripe = fract((pos.x + pos.y) / 8.0);
    let shade = select(1.0, 0.0, stripe < 0.5);
    return vec4<f32>(shade, shade, shade, 1.0);
}

// A3: missing page -> dim gray hatch at the OPPOSITE angle (-45deg) and very
// low contrast, so "not loaded yet" never reads as an actual zero value.
fn missing_marker(pos: vec2<f32>) -> vec4<f32> {
    let stripe = fract((pos.x - pos.y) / 8.0);
    let shade = select(0.20, 0.12, stripe < 0.5);
    return vec4<f32>(shade, shade, shade, 1.0);
}

// A1: zoom-gated pixel grid, applied once to the final colour.  `fw` is
// fwidth(in.src) -- source texels per screen pixel -- computed in uniform
// control flow at the top of fs_main (WGSL derivatives require that, and
// resolve() returns early).  `1/fw` is screen px per texel; the grid fades
// from nothing below 12 px/texel to full above 24, so a normally-zoomed
// scene is returned byte-identical (fade == 0 -> multiply by 1.0).
fn pixel_grid(color: vec4<f32>, src: vec2<f32>, fw: vec2<f32>, enabled: u32) -> vec4<f32> {
    if (enabled == 0u) { return color; }
    let fwc = max(fw, vec2<f32>(1e-8, 1e-8));
    let px_per_texel = 1.0 / max(fwc.x, fwc.y);
    let fade = smoothstep(12.0, 24.0, px_per_texel);
    let edge = min(fract(src), vec2<f32>(1.0) - fract(src)) / fwc;
    let line = 1.0 - smoothstep(0.0, 1.0, min(edge.x, edge.y));
    let darken = 0.2 * line * fade;
    return vec4<f32>(color.rgb * (1.0 - darken), color.a);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    // A1: fwidth before resolve()'s data-dependent early returns.
    let fw = fwidth(in.src);
    let p = planes[in.plane];
    let r = resolve(in.plane, in.src, in.lod);
    var color: vec4<f32>;
    if (r.layer < 0) {
        color = missing_marker(in.pos.xy);  // A3
    } else if (p.rep == 2u) {
        let c = textureLoad(rgb_pool, r.texel, r.layer, 0);
        color = vec4<f32>(c.rgb, 1.0);
    } else if (p.rep == 3u) {
        let c = textureLoad(rgb_windowed_pool, r.texel, r.layer, 0);
        let scalar = apply_scale(c.a);
        let intensity = clamp(
            (scalar - mapping.level_lo) / (mapping.level_hi - mapping.level_lo),
            0.0,
            1.0,
        );
        color = vec4<f32>(c.rgb * intensity, 1.0);
    } else {
        var v = vec2<f32>(0.0, 0.0);
        let compressed = page_codec[r.fidx] == 1u;
        let uv = (vec2<f32>(r.texel) + vec2<f32>(0.5)) / 256.0;
        let nrm = page_norm[r.fidx];
        if (p.rep == 0u) {
            if (compressed) {
                let s = textureSampleLevel(scalar_bc_pool, codec_samp, uv, r.layer, 0.0).r;
                v = vec2<f32>(s * nrm.y + nrm.x, 0.0);
            } else {
                v = vec2<f32>(textureLoad(scalar_pool, r.texel, r.layer, 0).r, 0.0);
            }
        } else {
            if (compressed) {
                let s = textureSampleLevel(complex_bc_pool, codec_samp, uv, r.layer, 0.0).rg;
                v = vec2<f32>(s.r * nrm.y + nrm.x, s.g * nrm.w + nrm.z);
            } else {
                v = textureLoad(complex_pool, r.texel, r.layer, 0).rg;
            }
        }
        if (!finite_scalar(v.x) || !finite_scalar(v.y)) {
            color = nan_marker(in.pos.xy);  // A2
        } else {
            color = map_value(p, v);
        }
    }
    return pixel_grid(color, in.src, fw, mapping.pixel_grid);  // A1
}

// Scalar/complex value -> displayed colour (mode/scale/levels/LUT + A4 clip).
fn map_value(p: PlaneInfo, v: vec2<f32>) -> vec4<f32> {
    var x: f32;
    if (p.rep == 0u) {
        x = v.x;
    } else {
        switch mapping.mode {
            case 0u: { x = length(v); }
            case 1u: { x = atan2(v.y, v.x); }
            case 2u: { x = v.x; }
            default: { x = v.y; }
        }
    }
    x = apply_scale(x);
    let g = clamp((x - mapping.level_lo) / (mapping.level_hi - mapping.level_lo), 0.0, 1.0);
    let phase_path = mapping.phase_color != 0u && p.rep == 1u && mapping.mode != 1u;
    if (mapping.clip_indicator != 0u && !phase_path) {
        if (x < mapping.level_lo) { return vec4<f32>(0.0, 0.2, 0.8, 1.0); }
        if (x > mapping.level_hi) { return vec4<f32>(0.9, 0.1, 0.0, 1.0); }
    }
    if (phase_path) {
        let phase = atan2(v.y, v.x);
        let phase_g = clamp(
            (phase + 3.141592653589793) / 6.283185307179586,
            0.0,
            1.0,
        );
        let phase_idx = clamp(i32(round(phase_g * 255.0)), 0, 255);
        let lut_color = textureLoad(lut, vec2<i32>(phase_idx, 0), 0);
        return vec4<f32>(lut_color.rgb * g, lut_color.a);
    }
    let idx = clamp(i32(round(g * 255.0)), 0, 255);
    return textureLoad(lut, vec2<i32>(idx, 0), 0);
}
"""

_OVERLAY_WGSL = """
struct OverlayCamera {
    scale: vec2<f32>,
    offset: vec2<f32>,
    target_size: vec2<f32>,
    _pad: vec2<f32>,
};
struct Overlay {
    p0: vec2<f32>,
    p1: vec2<f32>,
    color: vec4<f32>,
    anchor: vec2<f32>,
    width: f32,
    kind: u32,
    flags: u32,
    _pad0: u32,
    _pad1: vec2<f32>,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    screen_offset: vec2<f32>,
    size: vec2<f32>,
};
@group(0) @binding(0) var<uniform> camera: OverlayCamera;
@group(0) @binding(1) var<storage, read> overlays: array<Overlay>;
@group(0) @binding(2) var glyph_atlas: texture_2d<f32>;
@group(0) @binding(3) var widget_atlas: texture_2d<f32>;

struct OverlayOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) textured: u32,
};

fn world_to_ndc(point: vec2<f32>) -> vec2<f32> {
    return point * camera.scale + camera.offset;
}

@vertex
fn vs_overlay(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> OverlayOut {
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0));
    let primitive = overlays[instance_index];
    let q = quad[vertex_index];
    var ndc: vec2<f32>;
    var uv = vec2<f32>(0.0, 0.0);
    var textured = 0u;
    if (primitive.kind == 0u) {
        let a = world_to_ndc(primitive.p0);
        let b = world_to_ndc(primitive.p1);
        let delta_pixels = (b - a) * camera.target_size * 0.5;
        let length_pixels = max(length(delta_pixels), 0.0001);
        let normal_pixels = vec2<f32>(-delta_pixels.y, delta_pixels.x) / length_pixels;
        let normal_ndc = normal_pixels * primitive.width / camera.target_size;
        ndc = mix(a, b, q.x) + normal_ndc * (q.y * 2.0 - 1.0);
    } else if (primitive.kind == 1u) {
        ndc = world_to_ndc(mix(primitive.p0, primitive.p1, q));
    } else if (primitive.kind == 2u) {
        let center = world_to_ndc(primitive.p0);
        let pixel_offset = (q - vec2<f32>(0.5)) * primitive.width;
        ndc = center + pixel_offset * vec2<f32>(2.0) / camera.target_size;
    } else if (primitive.kind == 5u) {
        // Widget quad: window furniture, not scene content.  Placed purely
        // from y-down physical pixels off the target's top-left, so it never
        // moves with the camera the way an anchored screen_rect does.
        let pixels = primitive.screen_offset + q * primitive.size;
        ndc = vec2<f32>(-1.0, 1.0) + pixels * vec2<f32>(2.0, -2.0) / camera.target_size;
        uv = mix(primitive.uv0, primitive.uv1, q);
        textured = 2u;
    } else {
        // Screen-space-sized quad anchored at world p0: pixel offsets are
        // y-down physical pixels, so screen size is constant under zoom
        // while the anchor pans with the image.
        let anchor_ndc = world_to_ndc(primitive.p0);
        let pixels = primitive.screen_offset + q * primitive.size;
        ndc = anchor_ndc + pixels * vec2<f32>(2.0, -2.0) / camera.target_size;
        if (primitive.kind == 4u) {
            uv = mix(primitive.uv0, primitive.uv1, q);
            textured = 1u;
        }
    }
    if ((primitive.flags & 1u) != 0u) {
        let anchor_ndc = world_to_ndc(primitive.anchor);
        if (any(anchor_ndc < vec2<f32>(-1.0)) || any(anchor_ndc > vec2<f32>(1.0))) {
            ndc = vec2<f32>(3.0, 3.0);
        }
    }
    var out: OverlayOut;
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.color = primitive.color;
    out.uv = uv;
    out.textured = textured;
    return out;
}

@fragment
fn fs_overlay(in: OverlayOut) -> @location(0) vec4<f32> {
    var color = in.color;
    if (in.textured == 1u) {
        // Nearest texel load: glyph quads are laid out 1:1 with their atlas
        // cells, so exact loads keep text crisp (no sampler filtering).
        let dims = textureDimensions(glyph_atlas);
        let texel = clamp(
            vec2<i32>(in.uv * vec2<f32>(dims)),
            vec2<i32>(0),
            vec2<i32>(dims) - vec2<i32>(1),
        );
        color.a = color.a * textureLoad(glyph_atlas, texel, 0).r;
    } else if (in.textured == 2u) {
        // Widget quads carry Qt's own rasterized pixels, so the atlas REPLACES
        // the colour instead of masking it.  Straight (non-premultiplied)
        // alpha, matching the pipeline's src-alpha/one-minus-src-alpha blend,
        // so a translucent chip blends over the image exactly as Qt's painter
        // would have blended it.  1:1 texel load keeps text crisp.
        let dims = textureDimensions(widget_atlas);
        let texel = clamp(
            vec2<i32>(in.uv * vec2<f32>(dims)),
            vec2<i32>(0),
            vec2<i32>(dims) - vec2<i32>(1),
        );
        color = textureLoad(widget_atlas, texel, 0);
    }
    return color;
}
"""

#: Path A of the G7 levels/histogram work: the GPU histogram/bounds compute
#: shaders sample the BC pools directly, mirroring the render fragment's
#: codec-flag + ``(lo, span)``-unscale branch.  A compressed page is sampled
#: with the nearest/clamp ``codec_samp`` (reproducing the raw ``textureLoad``
#: texel centre) then unscaled by the SAME affine the render path applies,
#: sourced from ``_page_codec`` -- so no second scale source exists.  These
#: results come from the *lossy* decoded texels (measured against the exact
#: Path B raw-semantic reference by ``tools/g7_levels_histogram_benchmark.py``).
#: The raw (``compressed=False``) variant declares no BC bindings, so the
#: OFF/raw compute layout and every number stay byte-identical.


def _build_histo_wgsl(compressed: bool) -> str:
    # Histogram (``partial``) group BC bindings, appended after the raw pools
    # (7-9); empty when off so the raw layout is untouched.
    histo_bc_bindings = (
        """
@group(0) @binding(7) var scalar_bc_pool: texture_2d_array<f32>;
@group(0) @binding(8) var complex_bc_pool: texture_2d_array<f32>;
@group(0) @binding(9) var codec_samp: sampler;
"""
        if compressed
        else ""
    )
    # Bounds (``bounds_partial``) group BC bindings (6-8).
    bounds_bc_bindings = (
        """
@group(0) @binding(6) var bscalar_bc_pool: texture_2d_array<f32>;
@group(0) @binding(7) var bcomplex_bc_pool: texture_2d_array<f32>;
@group(0) @binding(8) var bcodec_samp: sampler;
"""
        if compressed
        else ""
    )

    if compressed:
        histo_scalar = (
            "if (page.codec == 1u) {\n"
            "            let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / 256.0;\n"
            "            value = textureSampleLevel(scalar_bc_pool, codec_samp, uv, page.layer,"
            " 0.0).r * page.span_r + page.lo_r;\n"
            "        } else {\n"
            "            value = textureLoad(scalar_pool, coord, page.layer, 0).r;\n"
            "        }"
        )
        histo_complex = (
            "var pair: vec2<f32>;\n"
            "        if (page.codec == 1u) {\n"
            "            let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / 256.0;\n"
            "            let s = textureSampleLevel(complex_bc_pool, codec_samp, uv, page.layer,"
            " 0.0).rg;\n"
            "            pair = vec2<f32>(s.r * page.span_r + page.lo_r,"
            " s.g * page.span_g + page.lo_g);\n"
            "        } else {\n"
            "            pair = textureLoad(complex_pool, coord, page.layer, 0).rg;\n"
            "        }"
        )
        bounds_scalar = (
            "if (page.codec == 1u) {\n"
            "            let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / 256.0;\n"
            "            value = textureSampleLevel(bscalar_bc_pool, bcodec_samp, uv, page.layer,"
            " 0.0).r * page.span_r + page.lo_r;\n"
            "        } else {\n"
            "            value = textureLoad(bscalar_pool, coord, page.layer, 0).r;\n"
            "        }"
        )
        bounds_complex = (
            "var pair: vec2<f32>;\n"
            "        if (page.codec == 1u) {\n"
            "            let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / 256.0;\n"
            "            let s = textureSampleLevel(bcomplex_bc_pool, bcodec_samp, uv, page.layer,"
            " 0.0).rg;\n"
            "            pair = vec2<f32>(s.r * page.span_r + page.lo_r,"
            " s.g * page.span_g + page.lo_g);\n"
            "        } else {\n"
            "            pair = textureLoad(bcomplex_pool, coord, page.layer, 0).rg;\n"
            "        }"
        )
    else:
        histo_scalar = "value = textureLoad(scalar_pool, coord, page.layer, 0).r;"
        histo_complex = "let pair = textureLoad(complex_pool, coord, page.layer, 0).rg;"
        bounds_scalar = "value = textureLoad(bscalar_pool, coord, page.layer, 0).r;"
        bounds_complex = "let pair = textureLoad(bcomplex_pool, coord, page.layer, 0).rg;"

    return (
        f"""
struct HArgs {{
    lo: f32,
    hi: f32,
    n_pages: u32,
    bins: u32,
    mode: u32,
    scale: u32,
    symlog_constant: f32,
    dynamic_bounds: u32,
}};
struct HPage {{
    layer: i32,
    rep: i32,
    source_h: u32,
    source_w: u32,
    factor: u32,
    codec: u32,
    lo_r: f32,
    span_r: f32,
    lo_g: f32,
    span_g: f32,
    _pad0: u32,
    _pad1: u32,
}};
@group(0) @binding(0) var<uniform> args: HArgs;
@group(0) @binding(1) var<storage, read> pages: array<HPage>;
@group(0) @binding(2) var scalar_pool: texture_2d_array<f32>;
@group(0) @binding(3) var complex_pool: texture_2d_array<f32>;
@group(0) @binding(4) var rgb_windowed_pool: texture_2d_array<f32>;
@group(0) @binding(5) var<storage, read_write> partials: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> final_bounds: array<u32>;
{histo_bc_bindings}
var<workgroup> local_bins: array<atomic<u32>, 512>;

fn ordered_float(value: f32) -> u32 {{
    let bits = bitcast<u32>(value);
    return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}}

fn float_from_ordered(value: u32) -> f32 {{
    let bits = select(value ^ 0x80000000u, ~value, value < 0x80000000u);
    return bitcast<f32>(bits);
}}

fn finite_value(value: f32) -> bool {{
    return value == value && abs(value) <= 3.402823466e+38;
}}

fn mapped_value(page: HPage, coord: vec2<i32>) -> f32 {{
    var value: f32;
    if (page.rep == 0) {{
        {histo_scalar}
    }} else if (page.rep == 1) {{
        {histo_complex}
        switch args.mode {{
            case 0u: {{ value = length(pair); }}
            case 1u: {{ value = atan2(pair.y, pair.x); }}
            case 2u: {{ value = pair.x; }}
            default: {{ value = pair.y; }}
        }}
    }} else {{
        value = textureLoad(rgb_windowed_pool, coord, page.layer, 0).a;
    }}
    switch args.scale {{
        case 0u: {{ return value; }}
        case 1u: {{ return log(max(value, 0.0)) / log(10.0); }}
        default: {{
            return sign(value) * log(
                1.0 + abs(value) / pow(10.0, args.symlog_constant)
            ) / log(10.0);
        }}
    }}
}}

fn stored_h(page: HPage) -> u32 {{
    return (page.source_h + page.factor - 1u) / page.factor;
}}

fn stored_w(page: HPage) -> u32 {{
    return (page.source_w + page.factor - 1u) / page.factor;
}}

fn source_weight(page: HPage, y: u32, x: u32) -> u32 {{
    let y0 = y * page.factor;
    let x0 = x * page.factor;
    return min(page.factor, page.source_h - y0) * min(page.factor, page.source_w - x0);
}}

@group(0) @binding(0) var<uniform> bargs: HArgs;
@group(0) @binding(1) var<storage, read> bpages: array<HPage>;
@group(0) @binding(2) var bscalar_pool: texture_2d_array<f32>;
@group(0) @binding(3) var bcomplex_pool: texture_2d_array<f32>;
@group(0) @binding(4) var brgb_windowed_pool: texture_2d_array<f32>;
@group(0) @binding(5) var<storage, read_write> page_bounds: array<u32>;
{bounds_bc_bindings}
var<workgroup> local_low: atomic<u32>;
var<workgroup> local_high: atomic<u32>;

fn bounds_mapped_value(page: HPage, coord: vec2<i32>) -> f32 {{
    var value: f32;
    if (page.rep == 0) {{
        {bounds_scalar}
    }} else if (page.rep == 1) {{
        {bounds_complex}
        switch bargs.mode {{
            case 0u: {{ value = length(pair); }}
            case 1u: {{ value = atan2(pair.y, pair.x); }}
            case 2u: {{ value = pair.x; }}
            default: {{ value = pair.y; }}
        }}
    }} else {{
        value = textureLoad(brgb_windowed_pool, coord, page.layer, 0).a;
    }}
    switch bargs.scale {{
        case 0u: {{ return value; }}
        case 1u: {{ return log(max(value, 0.0)) / log(10.0); }}
        default: {{
            return sign(value) * log(
                1.0 + abs(value) / pow(10.0, bargs.symlog_constant)
            ) / log(10.0);
        }}
    }}
}}
"""
        """

@compute @workgroup_size(256)
fn bounds_partial(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    if (li == 0u) {
        atomicStore(&local_low, 0xffffffffu);
        atomicStore(&local_high, 0u);
    }
    workgroupBarrier();
    let page = bpages[wg.x];
    if (li < stored_h(page)) {
        for (var x = 0u; x < stored_w(page); x = x + 1u) {
            let value = bounds_mapped_value(page, vec2<i32>(i32(x), i32(li)));
            if (finite_value(value)) {
                let ordered = ordered_float(value);
                atomicMin(&local_low, ordered);
                atomicMax(&local_high, ordered);
            }
        }
    }
    workgroupBarrier();
    if (li == 0u) {
        page_bounds[wg.x * 2u] = atomicLoad(&local_low);
        page_bounds[wg.x * 2u + 1u] = atomicLoad(&local_high);
    }
}

@group(0) @binding(0) var<uniform> bmargs: HArgs;
@group(0) @binding(1) var<storage, read> merged_bounds_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> merged_bounds_out: array<u32>;

var<workgroup> merged_low: atomic<u32>;
var<workgroup> merged_high: atomic<u32>;

@compute @workgroup_size(256)
fn bounds_merge(@builtin(local_invocation_index) li: u32) {
    if (li == 0u) {
        atomicStore(&merged_low, 0xffffffffu);
        atomicStore(&merged_high, 0u);
    }
    workgroupBarrier();
    for (var page = li; page < bmargs.n_pages; page = page + 256u) {
        let low = merged_bounds_in[page * 2u];
        let high = merged_bounds_in[page * 2u + 1u];
        if (low != 0xffffffffu) {
            atomicMin(&merged_low, low);
            atomicMax(&merged_high, high);
        }
    }
    workgroupBarrier();
    if (li == 0u) {
        merged_bounds_out[0] = atomicLoad(&merged_low);
        merged_bounds_out[1] = atomicLoad(&merged_high);
    }
}

@compute @workgroup_size(256)
fn partial(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    for (var bin = li; bin < args.bins; bin = bin + 256u) {
        atomicStore(&local_bins[bin], 0u);
    }
    workgroupBarrier();
    let page = pages[wg.x];
    var low = args.lo;
    var high = args.hi;
    if (args.dynamic_bounds != 0u) {
        low = float_from_ordered(final_bounds[0]);
        high = float_from_ordered(final_bounds[1]);
    }
    if (high <= low) {
        let radius = max(abs(low) * 0.03, 0.5);
        low = low - radius;
        high = high + radius;
    }
    if (li < stored_h(page) && final_bounds[0] != 0xffffffffu) {
        for (var x = 0u; x < stored_w(page); x = x + 1u) {
            let value = mapped_value(page, vec2<i32>(i32(x), i32(li)));
            if (finite_value(value)) {
                let t = (value - low) / (high - low);
                let bin = clamp(i32(t * f32(args.bins)), 0, i32(args.bins) - 1);
                atomicAdd(&local_bins[bin], source_weight(page, li, x));
            }
        }
    }
    workgroupBarrier();
    for (var bin = li; bin < args.bins; bin = bin + 256u) {
        atomicStore(&partials[wg.x * args.bins + bin], atomicLoad(&local_bins[bin]));
    }
}

@group(0) @binding(0) var<uniform> margs: HArgs;
@group(0) @binding(1) var<storage, read> merged_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> final_bins: array<u32>;

@compute @workgroup_size(256)
fn merge(@builtin(local_invocation_index) li: u32) {
    for (var bin = li; bin < margs.bins; bin = bin + 256u) {
        var acc = 0u;
        for (var p = 0u; p < margs.n_pages; p = p + 1u) {
            acc = acc + merged_in[p * margs.bins + bin];
        }
        final_bins[bin] = acc;
    }
}
"""
    )


def _reduce_wgsl(*, value_type: str, load_suffix: str, storage_format: str) -> str:
    """Build the component-mean shader for one honest pool representation."""

    zero = "0.0" if value_type == "f32" else "vec2<f32>(0.0)"
    stored = (
        "vec4<f32>(mean, 0.0, 0.0, 0.0)" if value_type == "f32" else "vec4<f32>(mean, 0.0, 0.0)"
    )
    return f"""
struct Args {{
    valid0: vec4<u32>,
    valid1: vec4<u32>,
    valid2: vec4<u32>,
    valid3: vec4<u32>,
}};
@group(0) @binding(0) var<uniform> args: Args;
@group(0) @binding(1) var src0: texture_2d<f32>;
@group(0) @binding(2) var src1: texture_2d<f32>;
@group(0) @binding(3) var src2: texture_2d<f32>;
@group(0) @binding(4) var src3: texture_2d<f32>;
@group(0) @binding(5) var dst: texture_storage_2d<{storage_format}, write>;

fn valid_size(page: u32) -> vec2<u32> {{
    switch page {{
        case 0u: {{ return args.valid0.xy; }}
        case 1u: {{ return args.valid1.xy; }}
        case 2u: {{ return args.valid2.xy; }}
        default: {{ return args.valid3.xy; }}
    }}
}}

fn load_value(page: u32, coord: vec2<i32>) -> {value_type} {{
    switch page {{
        case 0u: {{ return textureLoad(src0, coord, 0).{load_suffix}; }}
        case 1u: {{ return textureLoad(src1, coord, 0).{load_suffix}; }}
        case 2u: {{ return textureLoad(src2, coord, 0).{load_suffix}; }}
        default: {{ return textureLoad(src3, coord, 0).{load_suffix}; }}
    }}
}}

@compute @workgroup_size(16, 16)
fn reduce(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x >= 256u || gid.y >= 256u) {{ return; }}
    let source = gid.xy * 2u;
    var acc = {zero};
    var count = 0u;
    for (var dy = 0u; dy < 2u; dy = dy + 1u) {{
        for (var dx = 0u; dx < 2u; dx = dx + 1u) {{
            let coord = source + vec2<u32>(dx, dy);
            let page_xy = coord / 256u;
            let page = page_xy.y * 2u + page_xy.x;
            let local = coord % 256u;
            let valid = valid_size(page);
            if (local.x < valid.x && local.y < valid.y) {{
                acc = acc + load_value(page, vec2<i32>(local));
                count = count + 1u;
            }}
        }}
    }}
    var mean = {zero};
    if (count != 0u) {{ mean = acc / f32(count); }}
    textureStore(dst, vec2<i32>(gid.xy), {stored});
}}
"""


_REDUCE_WGSL = {
    SCALAR_R32F: _reduce_wgsl(value_type="f32", load_suffix="r", storage_format="r32float"),
    COMPLEX_RG32F: _reduce_wgsl(
        value_type="vec2<f32>", load_suffix="rg", storage_format="rg32float"
    ),
}


@dataclass
class _LodGrid:
    base: int
    grid_w: int
    grid_h: int


@dataclass
class _Pool:
    representation: str
    texture: object
    view: object
    free_layers: list[int] = field(default_factory=list)
    # Logical eviction budget.  This remains the public ``pool_budget`` and is
    # the maximum extent to which this physical array may grow.
    layer_count: int = 0
    # Current physical texture-array extent (at least one because bind groups
    # require a valid texture even for a representation with zero budget).
    allocated_layers: int = 1


@dataclass
class _HistogramReadbackBatch:
    """One staging buffer shared by every dynamic histogram in a frame."""

    device: object
    buffer: object
    spans: tuple[tuple[int, int, int, int, int, int], ...]
    on_resolve: object = None
    _raw: memoryview | None = None

    def resolve(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if self._raw is None:
            self._raw = memoryview(self.device.queue.read_buffer(self.buffer))
            if callable(self.on_resolve):
                self.on_resolve()
                self.on_resolve = None
        counts_offset, counts_size, bounds_offset, bounds_size, time_offset, time_size = self.spans[
            int(index)
        ]
        counts = np.frombuffer(
            self._raw[counts_offset : counts_offset + counts_size], np.uint32
        ).copy()
        bounds = np.frombuffer(
            self._raw[bounds_offset : bounds_offset + bounds_size], np.uint32
        ).copy()
        timestamps = (
            None
            if time_size <= 0
            else np.frombuffer(
                self._raw[time_offset : time_offset + time_size],
                np.uint64,
            ).copy()
        )
        return counts, bounds, timestamps


@dataclass
class _DeferredHistogramReadback:
    """Small readback resolved only after the report completion token fences."""

    device: object
    counts_buffer: object
    bounds_buffer: object
    bins: int
    timestamp_buffer: object | None = None
    timestamp_query_set: object | None = None
    timestamp_period_ns: float | None = None
    timestamp_indices: tuple[int, ...] = ()
    on_resolve: object = None
    _resolved: tuple[np.ndarray, tuple[float, float] | None] | None = None
    _gpu_elapsed_ms: float | None = None
    _batch: _HistogramReadbackBatch | None = None
    _batch_index: int = -1

    def resolve(self) -> tuple[np.ndarray, tuple[float, float] | None]:
        if self._resolved is None:
            if self._batch is None:
                counts = np.frombuffer(
                    self.device.queue.read_buffer(self.counts_buffer), np.uint32
                ).copy()
                raw_bounds = np.frombuffer(
                    self.device.queue.read_buffer(self.bounds_buffer), np.uint32
                ).copy()
                timestamps = (
                    None
                    if self.timestamp_buffer is None
                    else np.frombuffer(
                        self.device.queue.read_buffer(self.timestamp_buffer), np.uint64
                    ).copy()
                )
            else:
                counts, raw_bounds, timestamps = self._batch.resolve(self._batch_index)
            finite_bounds = (
                None
                if int(raw_bounds[0]) == 0xFFFFFFFF
                else (
                    _float32_from_ordered(int(raw_bounds[0])),
                    _float32_from_ordered(int(raw_bounds[1])),
                )
            )
            self._resolved = (counts[: int(self.bins)], finite_bounds)
            if callable(self.on_resolve):
                self.on_resolve()
                self.on_resolve = None
            if timestamps is not None:
                indices = tuple(int(index) for index in self.timestamp_indices)
                elapsed_ticks = sum(
                    max(0, int(timestamps[stop]) - int(timestamps[start]))
                    for start, stop in zip(indices[::2], indices[1::2], strict=False)
                )
                self._gpu_elapsed_ms = (
                    float(elapsed_ticks) * float(self.timestamp_period_ns or 1.0) / 1_000_000.0
                )
        return self._resolved

    @property
    def gpu_elapsed_ms(self) -> float | None:
        self.resolve()
        return self._gpu_elapsed_ms


class WgpuPlaneExecutor:
    """Protocol executor for bound 2-D plane pyramids on a wgpu device.

    ``pool_layers`` is either one integer budget applied to every
    representation pool or a dict ``{representation: layers}``; a
    representation missing from the dict gets no budget, and ensuring a
    chunk of that representation raises loudly.  ``plane_shape``/``max_lod``
    are retained for construction-time capacity context only — the GPU flat
    table and LOD spans always derive from the currently bound
    :class:`~arrayscope.gpu.command_protocol.ContentPlane` set.
    """

    def __init__(
        self,
        plane_shape: tuple[int, int] | None = None,
        *,
        max_lod: int = 1,
        pool_layers: int | dict[str, int] = 64,
        initial_pool_layers: dict[str, int] | None = None,
        initial_codec_pool_layers: dict[str, int] | None = None,
        target_size: tuple[int, int] = (768, 768),
        device: object = None,
        compressed_textures: str | bool = "off",
        codec_min_psnr_db: float = 40.0,
        histogram_codec_mode: str = "gpu_compressed",
        astc_block: tuple[int, int] = _ASTC_DEFAULT_BLOCK,
    ) -> None:
        import wgpu  # deferred: module import stays wgpu-free

        self._wgpu = wgpu
        # Normalize the compression mode: "off"/False (default: the render path
        # is byte-identical), "on"/True (force it, raise if the device cannot),
        # or "auto" (engage aggressively whenever the device advertises BC).
        # The AUTO that turns this on by default lives at the settings/capability
        # layer (see arrayscope.app.settings_state.TextureCodecChoice); the
        # executor default stays OFF so every existing exact-framebuffer oracle
        # and histogram test renders through the unchanged path.
        if compressed_textures is True:
            mode = "on"
        elif compressed_textures is False:
            mode = "off"
        else:
            mode = str(compressed_textures).lower()
        if mode not in ("off", "on", "auto"):
            raise ValueError(
                f"compressed_textures must be off/on/auto, got {compressed_textures!r}"
            )
        self._codec_mode = mode
        self._codec_min_psnr_db = float(codec_min_psnr_db)
        # How the GPU histogram/bounds compute treats a page that lives in a BC
        # pool (only reachable when compression is engaged):
        #   "gpu_compressed" -- Path A: the compute shaders sample the BC pool
        #       (lossy decoded texels) with the render path's codec branch, so a
        #       compressed page contributes present-but-lossy auto-levels.
        #   "skip" -- Path B posture: the compute excludes compressed pages and
        #       reports them partial (``histogram_missing``); their exact stats
        #       come from the CPU semantic plane / full-population refinement.
        # OFF has no compressed pages so the knob is inert; the raw path is
        # byte-identical regardless.
        histogram_codec_mode = str(histogram_codec_mode).lower()
        if histogram_codec_mode not in ("gpu_compressed", "skip"):
            raise ValueError(
                "histogram_codec_mode must be 'gpu_compressed' or 'skip', "
                f"got {histogram_codec_mode!r}"
            )
        self._histogram_codec_mode = histogram_codec_mode

        # The ASTC block for the compressed display pool must tile the 256² page
        # exactly (wgpu rejects a block-compressed texture whose dimensions are
        # not a multiple of the block, and uv = (texel+0.5)/256 assumes 256²).
        bx, by = int(astc_block[0]), int(astc_block[1])
        if PAGE % bx or PAGE % by:
            raise ValueError(
                f"astc_block {astc_block} must divide the {PAGE}px page (wgpu "
                "requires block-aligned compressed-texture dimensions); use "
                "(4,4) or (8,8) — 6x6 does not divide 256"
            )
        self._astc_block = (bx, by)

        if device is None:
            from wgpu.backends.wgpu_native.extras import set_instance_extras

            # Vulkan-only instance: the GL backend's EGL re-init is fatal
            # under Wayland (gate-B Tier 0). Harmless if already set; a
            # RuntimeError means the instance already exists (e.g. shared
            # with a canvas).
            with contextlib.suppress(RuntimeError):
                set_instance_extras(backends=["Vulkan"])
            adapter = wgpu.gpu.request_adapter_sync(power_preference="low-power")
            # Pick the codec family the policy prefers for this adapter (ASTC on
            # an integrated/ASTC-capable device, BC otherwise) and request only
            # that family's feature, so a self-owned device hosts the right pools.
            family = _decide_codec_family(
                mode,
                {str(f) for f in adapter.features},
                _adapter_is_integrated(_adapter_info(adapter)),
            )
            wanted = []
            if family == "bc":
                wanted.append(_BC_FEATURE)
            elif family == "astc":
                wanted.append(_ASTC_FEATURE)
            device = adapter.request_device_sync(required_features=wanted)
        self.device = device

        # Re-derive the effective family from the DEVICE's actual features and
        # its own adapter topology (a caller-supplied device may enable only one
        # family, or be the discrete adapter): integrated + ASTC -> ASTC, else BC
        # if available, else nothing.  Discrete NVIDIA never advertises ASTC, so
        # it always lands on BC — the existing BC path is unchanged there.
        device_features = {str(f) for f in self.device.features}
        family = _decide_codec_family(
            mode,
            device_features,
            _adapter_is_integrated(_adapter_info(getattr(self.device, "adapter", None))),
        )
        if mode == "on" and family == "none":
            raise RuntimeError(
                "compressed_textures forced on but the device enables neither "
                f"{_BC_FEATURE!r} nor a usable {_ASTC_FEATURE!r} "
                "(create the device with a compressed-texture feature; ASTC also "
                "needs the astc_encoder library)"
            )
        #: The codec family live this session: "bc", "astc", or "none".  AUTO
        #: degrades silently to the raw path ("none") when the device supports no
        #: compressed format; a forced "on" already raised.
        self._codec_family = family
        self._codec_engaged = mode != "off" and family != "none"
        # Per-family format / block geometry the pool, encode, upload and read
        # paths consult.  Left at BC defaults (inert) when nothing is engaged.
        if family == "astc":
            fmt = astc_codec.wgpu_format_for_block(self._astc_block)
            self._codec_feature = _ASTC_FEATURE
            self._codec_block = self._astc_block
            self._codec_pool_formats = {SCALAR_R32F: fmt, COMPLEX_RG32F: fmt}
            self._codec_block_bytes = {
                SCALAR_R32F: astc_codec.ASTC_BLOCK_BYTES,
                COMPLEX_RG32F: astc_codec.ASTC_BLOCK_BYTES,
            }
        else:
            self._codec_feature = _BC_FEATURE
            self._codec_block = _BC_BLOCK
            self._codec_pool_formats = dict(_BC_POOL_FORMATS)
            self._codec_block_bytes = dict(_BC_BLOCK_BYTES)

        if plane_shape is not None:
            h, w = (int(v) for v in plane_shape)
            if h <= 0 or w <= 0:
                raise ValueError(f"plane shape must be positive, got {plane_shape}")
            self.plane_shape = (h, w)
        else:
            self.plane_shape = None
        self.max_lod = int(max_lod)

        if isinstance(pool_layers, dict):
            budgets = {str(rep): int(layers) for rep, layers in pool_layers.items()}
            unknown = set(budgets) - set(REPRESENTATIONS)
            if unknown:
                raise ValueError(f"unknown pool representations {sorted(unknown)}")
        else:
            budgets = {rep: int(pool_layers) for rep in REPRESENTATIONS}
        self._pool_budgets = {rep: max(0, budgets.get(rep, 0)) for rep in REPRESENTATIONS}
        initial_pool_layers = {
            str(rep): max(0, int(layers)) for rep, layers in dict(initial_pool_layers or {}).items()
        }
        initial_codec_pool_layers = {
            str(rep): max(0, int(layers))
            for rep, layers in dict(initial_codec_pool_layers or {}).items()
        }
        unknown_initial = (set(initial_pool_layers) | set(initial_codec_pool_layers)) - set(
            REPRESENTATIONS
        )
        if unknown_initial:
            raise ValueError(f"unknown initial pool representations {sorted(unknown_initial)}")

        self.page_table = PageTable()
        self._bound_planes: tuple = ()
        self._plane_grids: list[list[_LodGrid]] = []
        self._plane_family_indices: dict[tuple[object, ...], tuple[int, ...]] = {}
        self._plane_lookup_candidates_total = 0
        self._flat_table = np.full(1, -1, dtype=np.int32)
        # Parallel per-flat-entry codec metadata (only consulted when the
        # compressed shader is live).  ``_flat_codec[i] == 1`` means the page at
        # flat slot ``i`` lives in a BC pool; ``_flat_norm[i] = (lo_r, span_r,
        # lo_g, span_g)`` is the affine the shader applies to the sampler's
        # [0,1] output to recover raw values.  Kept dirty-flushed with the table.
        self._flat_codec = np.zeros(1, dtype=np.uint32)
        self._flat_norm = np.zeros((1, 4), dtype=np.float32)
        #: key -> (codec_flag, (lo_r, span_r, lo_g, span_g)); the persistent
        #: source of truth _bind_planes rebuilds the flat codec arrays from.
        self._page_codec: dict[DataChunkKey, tuple[int, tuple[float, float, float, float]]] = {}
        self._compressed_uploads_total = 0
        self._compressed_fallbacks_total = 0
        self._texture_upload_bytes_total = 0
        self._lod_compressed_source_reductions_total = 0
        self._pool_grows_total = 0
        self._pool_growth_copy_bytes_total = 0
        self._last_pool_exhaustion = ""
        self._table_dirty = True
        self._tiles: tuple = ()
        self._mapping = DisplayMapping()
        self._uploads_total = 0
        # Submission-scoped histogram frontier shield (2026-07-19 dogfood
        # crash): keys a DispatchHistogram later in the CURRENT submission
        # will sample.  Populated by submit()'s pre-scan, pinned as they
        # become resident, always released before submit() returns.
        self._histogram_shield_wanted: frozenset[DataChunkKey] = frozenset()
        self._histogram_shield_pins: set[DataChunkKey] = set()
        self._histogram_dispatches_total = 0
        self._histogram_readback_resolves_total = 0
        self._histogram_batch_readbacks_total = 0
        self._overlay_geometry: tuple[OverlayPrimitive, ...] = ()
        self._overlay_camera = SetOverlayCamera((0.0, 0.0, 1.0, 1.0))
        self._overlay_buffer_writes_total = 0

        d = self.device
        self._pools: dict[str, _Pool] = {}
        for rep in REPRESENTATIONS:
            budget = self._pool_budgets[rep]
            requested = initial_pool_layers.get(rep, _POOL_INITIAL_LAYERS)
            allocated = max(1, min(max(1, requested), budget)) if budget else 1
            usage = (
                wgpu.TextureUsage.TEXTURE_BINDING
                | wgpu.TextureUsage.COPY_DST
                | wgpu.TextureUsage.COPY_SRC
            )
            if rep in _REDUCE_WGSL:
                usage |= wgpu.TextureUsage.STORAGE_BINDING
            texture = d.create_texture(
                size=(PAGE, PAGE, allocated),
                format=_POOL_FORMATS[rep],
                usage=usage,
            )
            self._pools[rep] = _Pool(
                representation=rep,
                texture=texture,
                view=texture.create_view(dimension="2d-array"),
                free_layers=list(range(min(allocated, budget))),
                layer_count=budget,
                allocated_layers=allocated,
            )

        # Native block-compressed pools, parallel to the raw ones and sharing
        # the same per-representation layer budget.  A page is stored in EITHER
        # its raw pool OR its BC pool (never both); the page table's pool_id is
        # the loud channel that records which.  Only created when compression is
        # engaged AND the representation has a BC format (scalar/complex).
        self._codec_pools: dict[str, _Pool] = {}
        self._codec_sampler = None
        if self._codec_engaged:
            for rep, fmt in self._codec_pool_formats.items():
                budget = self._pool_budgets[rep]
                if budget <= 0:
                    continue
                requested = initial_codec_pool_layers.get(rep, _POOL_INITIAL_LAYERS)
                allocated = min(max(1, requested), budget)
                texture = d.create_texture(
                    size=(PAGE, PAGE, allocated),
                    format=fmt,
                    # COPY_SRC so a BC page can be read back and reference-decoded
                    # (the read_resident_page oracle and the CPU LOD-from-
                    # compressed reduce path both copy blocks out of the pool).
                    usage=(
                        wgpu.TextureUsage.TEXTURE_BINDING
                        | wgpu.TextureUsage.COPY_DST
                        | wgpu.TextureUsage.COPY_SRC
                    ),
                )
                self._codec_pools[rep] = _Pool(
                    representation=rep,
                    texture=texture,
                    view=texture.create_view(dimension="2d-array"),
                    free_layers=list(range(allocated)),
                    layer_count=budget,
                    allocated_layers=allocated,
                )
            # Nearest + clamp: the compressed sampler must reproduce the exact
            # texel-centre decode textureLoad gives on the raw path (no filtering
            # across texels), so raw vs compressed differ only by BC's own loss.
            self._codec_sampler = d.create_sampler(
                mag_filter="nearest",
                min_filter="nearest",
                mipmap_filter="nearest",
                address_mode_u="clamp-to-edge",
                address_mode_v="clamp-to-edge",
                address_mode_w="clamp-to-edge",
            )
        # A rep whose BC pool could not be created (no layer budget) must never
        # take the compressed path even when engaged globally.  With no BC pool
        # at all (e.g. an RGB-only executor) there is nothing to compress, so the
        # unchanged base shader is used.
        self._codec_reps = frozenset(self._codec_pools)
        self._codec_engaged = self._codec_engaged and bool(self._codec_pools)

        # Bind-group epoch: bumped whenever a bound buffer is recreated
        # (plane rebind, table growth) so cached bind groups are rebuilt.
        self._bind_epoch = 0
        self._table_buf = d.create_buffer(
            size=max(16, self._flat_table.nbytes),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        # Codec side buffers mirror the flat table's length (only bound by the
        # compressed pipeline).  Start at one entry; grown with the table.
        self._codec_flag_buf = d.create_buffer(
            size=max(16, self._flat_codec.nbytes),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self._codec_norm_buf = d.create_buffer(
            size=max(16, self._flat_norm.nbytes),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self._lod_info_buf = d.create_buffer(
            size=16, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )
        self._planes_buf = d.create_buffer(
            size=16, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )
        self._tiles_cap = 512
        self._tiles_buf = d.create_buffer(
            size=48 * self._tiles_cap,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self._mapping_buf = d.create_buffer(
            size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self._overlay_cap = 256
        self._overlay_buf = d.create_buffer(
            size=_OVERLAY_INSTANCE_BYTES * self._overlay_cap,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self._camera_buf = d.create_buffer(
            size=32, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        # Glyph atlas: CPU-baked alpha mask sampled by glyph_quad instances.
        # Replaced wholesale by UpdateGlyphAtlas (rare); starts as one
        # transparent texel so the bind group is always valid.
        self._glyph_atlas_size = (1, 1)
        self._glyph_atlas_tex = d.create_texture(
            size=(1, 1, 1),
            format="r8unorm",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self._glyph_atlas_uploads_total = 0
        # Widget atlas: CPU-rasterized Qt chips sampled by widget_quad
        # instances (straight RGBA).  Same lifecycle as the glyph atlas —
        # replaced wholesale by UpdateWidgetAtlas, starts as one transparent
        # texel so the bind group is always valid.
        self._widget_atlas_size = (1, 1)
        self._widget_atlas_tex = d.create_texture(
            size=(1, 1, 1),
            format="rgba8unorm",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self._widget_atlas_uploads_total = 0
        self._lut_tex = d.create_texture(
            size=(256, 1, 1),
            format="rgba8unorm",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self._current_lut = object()  # sentinel: force the first write
        self._write_lut(None)
        self._write_mapping()

        self._target_size = tuple(int(v) for v in target_size)
        self._target = d.create_texture(
            size=(*self._target_size, 1),
            format="rgba8unorm",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )

        # The compressed variant is byte-for-byte the base shader plus the BC
        # bindings and branch, so when compression is off we compile the
        # unchanged source and the default render path is provably identical.
        self._shader = d.create_shader_module(
            code=_RENDER_WGSL_COMPRESSED if self._codec_engaged else _RENDER_WGSL
        )
        self._pipelines: dict[str, object] = {}
        self._binds: dict[str, tuple[object, int]] = {}
        self._overlay_shader = d.create_shader_module(code=_OVERLAY_WGSL)
        self._overlay_pipelines: dict[str, object] = {}
        self._overlay_binds: dict[str, object] = {}
        # Engaged compiles the BC-sampling histo/bounds variant (Path A bindings
        # present); OFF compiles the raw variant with no BC bindings, so the
        # compute layout and every histogram/auto-range number stay identical to
        # a build without this feature.  When engaged but the histogram mode is
        # "skip", the BC branch is simply never reached (no compressed entries).
        self._histo_mod = d.create_shader_module(
            code=_build_histo_wgsl(compressed=self._codec_engaged)
        )
        self._partial_pipe = d.create_compute_pipeline(
            layout="auto", compute={"module": self._histo_mod, "entry_point": "partial"}
        )
        self._merge_pipe = d.create_compute_pipeline(
            layout="auto", compute={"module": self._histo_mod, "entry_point": "merge"}
        )
        self._bounds_partial_pipe = d.create_compute_pipeline(
            layout="auto",
            compute={"module": self._histo_mod, "entry_point": "bounds_partial"},
        )
        self._bounds_merge_pipe = d.create_compute_pipeline(
            layout="auto",
            compute={"module": self._histo_mod, "entry_point": "bounds_merge"},
        )
        self._reduce_pipes = {
            representation: d.create_compute_pipeline(
                layout="auto",
                compute={
                    "module": d.create_shader_module(code=shader),
                    "entry_point": "reduce",
                },
            )
            for representation, shader in _REDUCE_WGSL.items()
        }

    # ---- internals ----------------------------------------------------------

    def _pipeline(self, fmt: str):
        if fmt not in self._pipelines:
            self._pipelines[fmt] = self.device.create_render_pipeline(
                layout="auto",
                vertex={"module": self._shader, "entry_point": "vs_main"},
                primitive={"topology": "triangle-list"},
                fragment={
                    "module": self._shader,
                    "entry_point": "fs_main",
                    "targets": [{"format": fmt}],
                },
            )
        pipe = self._pipelines[fmt]
        cached = self._binds.get(fmt)
        if cached is None or cached[1] != self._bind_epoch:
            bind = self.device.create_bind_group(
                layout=pipe.get_bind_group_layout(0),
                entries=[
                    {
                        "binding": 0,
                        "resource": {"buffer": self._mapping_buf, "offset": 0, "size": 32},
                    },
                    {
                        "binding": 1,
                        "resource": {
                            "buffer": self._table_buf,
                            "offset": 0,
                            "size": self._table_buf.size,
                        },
                    },
                    {
                        "binding": 2,
                        "resource": {
                            "buffer": self._lod_info_buf,
                            "offset": 0,
                            "size": self._lod_info_buf.size,
                        },
                    },
                    {
                        "binding": 3,
                        "resource": {
                            "buffer": self._planes_buf,
                            "offset": 0,
                            "size": self._planes_buf.size,
                        },
                    },
                    {
                        "binding": 4,
                        "resource": {
                            "buffer": self._tiles_buf,
                            "offset": 0,
                            "size": self._tiles_buf.size,
                        },
                    },
                    {"binding": 5, "resource": self._pools[SCALAR_R32F].view},
                    {"binding": 6, "resource": self._pools[COMPLEX_RG32F].view},
                    {"binding": 7, "resource": self._pools[RGB8].view},
                    {"binding": 8, "resource": self._pools[RGB_WINDOWED_RGBA32F].view},
                    {"binding": 9, "resource": self._lut_tex.create_view()},
                    {
                        "binding": 10,
                        "resource": {
                            "buffer": self._camera_buf,
                            "offset": 0,
                            "size": 32,
                        },
                    },
                    *self._codec_bind_entries(),
                ],
            )
            self._binds[fmt] = (bind, self._bind_epoch)
        return pipe, self._binds[fmt][0]

    def _codec_bind_entries(self) -> list[dict]:
        """Render bind-group entries 11-15 for the BC pools (empty when off)."""

        if not self._codec_engaged:
            return []
        return [
            {
                "binding": 11,
                "resource": {
                    "buffer": self._codec_flag_buf,
                    "offset": 0,
                    "size": self._codec_flag_buf.size,
                },
            },
            {
                "binding": 12,
                "resource": {
                    "buffer": self._codec_norm_buf,
                    "offset": 0,
                    "size": self._codec_norm_buf.size,
                },
            },
            {"binding": 13, "resource": self._codec_sampler},
            {"binding": 14, "resource": self._codec_pool_view(SCALAR_R32F)},
            {"binding": 15, "resource": self._codec_pool_view(COMPLEX_RG32F)},
        ]

    def _codec_pool_view(self, rep: str):
        """View of the BC pool for ``rep``, or the scalar BC pool as a stand-in.

        The bind group must supply a valid array view for both compressed pool
        bindings even when a representation has no BC budget; the shader never
        samples a pool it did not store a page into, so any live BC view is a
        safe filler for an unused binding slot.
        """

        pool = self._codec_pools.get(rep)
        if pool is None:
            pool = next(iter(self._codec_pools.values()))
        return pool.view

    def _pool_by_id(self, pool_id: str) -> _Pool:
        rep = _REP_BY_POOL_ID[pool_id]
        if pool_id in _CODEC_POOL_IDS.values():
            return self._codec_pools[rep]
        return self._pools[rep]

    def _pool_layer_bytes(self, pool_id: str) -> int:
        rep = _REP_BY_POOL_ID[pool_id]
        if pool_id in _CODEC_POOL_IDS.values():
            bx, by = self._codec_block
            return (PAGE // bx) * (PAGE // by) * self._codec_block_bytes[rep]
        return PAGE * PAGE * _POOL_TEXEL_BYTES[rep]

    def _grow_pool(self, pool_id: str, *, minimum_layers: int = 0) -> None:
        """Grow one immutable texture array while preserving layer indices.

        Raw and block-compressed formats must remain separate bindings, but
        their physical extents can follow actual admission independently.  GPU
        queue ordering makes the old->new copy precede later writes/draws; the
        page table stays valid because every occupied layer keeps its index.
        """

        pool = self._pool_by_id(pool_id)
        old_layers = int(pool.allocated_layers)
        if old_layers >= int(pool.layer_count):
            return
        new_layers = min(
            int(pool.layer_count),
            max(
                _POOL_INITIAL_LAYERS,
                old_layers + 1,
                old_layers * 2,
                int(minimum_layers),
            ),
        )
        codec = pool_id in _CODEC_POOL_IDS.values()
        if codec:
            texture_format = self._codec_pool_formats[pool.representation]
            usage = (
                self._wgpu.TextureUsage.TEXTURE_BINDING
                | self._wgpu.TextureUsage.COPY_DST
                | self._wgpu.TextureUsage.COPY_SRC
            )
        else:
            texture_format = _POOL_FORMATS[pool.representation]
            usage = (
                self._wgpu.TextureUsage.TEXTURE_BINDING
                | self._wgpu.TextureUsage.COPY_DST
                | self._wgpu.TextureUsage.COPY_SRC
            )
            if pool.representation in _REDUCE_WGSL:
                usage |= self._wgpu.TextureUsage.STORAGE_BINDING

        texture = self.device.create_texture(
            size=(PAGE, PAGE, new_layers),
            format=texture_format,
            usage=usage,
        )
        encoder = self.device.create_command_encoder()
        encoder.copy_texture_to_texture(
            {"texture": pool.texture},
            {"texture": texture},
            (PAGE, PAGE, old_layers),
        )
        self.device.queue.submit([encoder.finish()])

        pool.texture = texture
        pool.view = texture.create_view(dimension="2d-array")
        pool.free_layers.extend(range(old_layers, new_layers))
        pool.allocated_layers = new_layers
        self._pool_grows_total += 1
        self._pool_growth_copy_bytes_total += old_layers * self._pool_layer_bytes(pool_id)
        # Render bind groups capture texture views. Histogram and LOD bind groups
        # are submission-local and will naturally see the replacement view.
        self._bind_epoch += 1

    def _ensure_free_pool_layer(self, pool_id: str) -> None:
        pool = self._pool_by_id(pool_id)
        if pool.free_layers:
            return
        if pool.allocated_layers < pool.layer_count:
            self._grow_pool(pool_id)
            return
        self._evict_one_unpinned(_REP_BY_POOL_ID[pool_id], pool_id=pool_id)

    def _overlay_pipeline(self, fmt: str):
        if fmt not in self._overlay_pipelines:
            self._overlay_pipelines[fmt] = self.device.create_render_pipeline(
                layout="auto",
                vertex={"module": self._overlay_shader, "entry_point": "vs_overlay"},
                primitive={"topology": "triangle-list"},
                fragment={
                    "module": self._overlay_shader,
                    "entry_point": "fs_overlay",
                    "targets": [
                        {
                            "format": fmt,
                            "blend": {
                                "color": {
                                    "src_factor": "src-alpha",
                                    "dst_factor": "one-minus-src-alpha",
                                    "operation": "add",
                                },
                                "alpha": {
                                    "src_factor": "one",
                                    "dst_factor": "one-minus-src-alpha",
                                    "operation": "add",
                                },
                            },
                        }
                    ],
                },
            )
        pipe = self._overlay_pipelines[fmt]
        bind = self._overlay_binds.get(fmt)
        if bind is None:
            bind = self.device.create_bind_group(
                layout=pipe.get_bind_group_layout(0),
                entries=[
                    {
                        "binding": 0,
                        "resource": {
                            "buffer": self._camera_buf,
                            "offset": 0,
                            "size": 32,
                        },
                    },
                    {
                        "binding": 1,
                        "resource": {
                            "buffer": self._overlay_buf,
                            "offset": 0,
                            "size": self._overlay_buf.size,
                        },
                    },
                    {"binding": 2, "resource": self._glyph_atlas_tex.create_view()},
                    {"binding": 3, "resource": self._widget_atlas_tex.create_view()},
                ],
            )
            self._overlay_binds[fmt] = bind
        return pipe, bind

    def _set_overlay_geometry(self, primitives) -> int:
        primitives = tuple(primitives)
        self._overlay_geometry = primitives
        needed = max(1, len(primitives))
        if needed > self._overlay_cap:
            while self._overlay_cap < needed:
                self._overlay_cap *= 2
            self._overlay_buf = self.device.create_buffer(
                size=_OVERLAY_INSTANCE_BYTES * self._overlay_cap,
                usage=self._wgpu.BufferUsage.STORAGE | self._wgpu.BufferUsage.COPY_DST,
            )
            self._overlay_binds.clear()
        if not primitives:
            return 1
        packed = bytearray()
        for primitive in primitives:
            anchor = primitive.visibility_anchor or (0.0, 0.0)
            packed.extend(
                struct.pack(
                    "11f3I2f8f",
                    *primitive.p0,
                    *primitive.p1,
                    *primitive.rgba,
                    *anchor,
                    primitive.width,
                    _OVERLAY_KIND_INDEX[primitive.kind],
                    1 if primitive.visibility_anchor is not None else 0,
                    0,
                    0.0,
                    0.0,
                    *primitive.uv_rect,
                    *primitive.screen_offset,
                    *primitive.size,
                )
            )
        self.device.queue.write_buffer(self._overlay_buf, 0, packed)
        return 1

    def _update_glyph_atlas(self, cmd: UpdateGlyphAtlas) -> int:
        wgpu, d = self._wgpu, self.device
        size = (int(cmd.width), int(cmd.height))
        if size != self._glyph_atlas_size:
            self._glyph_atlas_tex = d.create_texture(
                size=(*size, 1),
                format="r8unorm",
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            )
            self._glyph_atlas_size = size
            self._overlay_binds.clear()
        d.queue.write_texture(
            {"texture": self._glyph_atlas_tex},
            cmd.data,
            {"bytes_per_row": size[0], "rows_per_image": size[1]},
            (*size, 1),
        )
        self._glyph_atlas_uploads_total += 1
        return 1

    def _update_widget_atlas(self, cmd: UpdateWidgetAtlas) -> int:
        wgpu, d = self._wgpu, self.device
        size = (int(cmd.width), int(cmd.height))
        if size != self._widget_atlas_size:
            self._widget_atlas_tex = d.create_texture(
                size=(*size, 1),
                format="rgba8unorm",
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            )
            self._widget_atlas_size = size
            self._overlay_binds.clear()
        d.queue.write_texture(
            {"texture": self._widget_atlas_tex},
            cmd.data,
            {"bytes_per_row": size[0] * 4, "rows_per_image": size[1]},
            (*size, 1),
        )
        self._widget_atlas_uploads_total += 1
        return 1

    def _write_camera(self, target_size: tuple[int, int]) -> None:
        """One world->NDC uniform, shared by the tile and overlay pipelines.

        Both read the same 32 bytes, so a pan costs exactly one small write
        no matter how many tiles or overlay primitives are drawn.
        """

        x0, y0, x1, y1 = self._overlay_camera.world_rect
        span_x = x1 - x0
        span_y = y1 - y0
        if self._overlay_camera.x_inverted:
            scale_x = -2.0 / span_x
            offset_x = 1.0 + 2.0 * x0 / span_x
        else:
            scale_x = 2.0 / span_x
            offset_x = -1.0 - 2.0 * x0 / span_x
        if self._overlay_camera.y_inverted:
            scale_y = -2.0 / span_y
            offset_y = 1.0 + 2.0 * y0 / span_y
        else:
            scale_y = 2.0 / span_y
            offset_y = -1.0 - 2.0 * y0 / span_y
        width, height = (max(1, int(value)) for value in target_size)
        self.device.queue.write_buffer(
            self._camera_buf,
            0,
            struct.pack(
                "8f",
                scale_x,
                scale_y,
                offset_x,
                offset_y,
                float(width),
                float(height),
                0.0,
                0.0,
            ),
        )

    def _write_lut(self, lut: bytes | None) -> None:
        if lut == self._current_lut:
            return
        if lut is None:  # neutral grayscale ramp
            ramp = np.empty((256, 4), np.uint8)
            ramp[:, 0] = ramp[:, 1] = ramp[:, 2] = np.arange(256)
            ramp[:, 3] = 255
            data = ramp.tobytes()
        else:
            data = lut
        self.device.queue.write_texture(
            {"texture": self._lut_tex},
            data,
            {"bytes_per_row": 256 * 4, "rows_per_image": 1},
            (256, 1, 1),
        )
        self._current_lut = lut

    def _write_mapping(self) -> None:
        self.device.queue.write_buffer(
            self._mapping_buf,
            0,
            struct.pack(
                "2I3f3I",
                _MODE_INDEX[self._mapping.mode],
                _SCALE_INDEX[self._mapping.scale],
                self._mapping.level_lo,
                self._mapping.level_hi,
                self._mapping.symlog_constant,
                int(self._mapping.phase_color),
                # Stage-A legibility flags in the two spare Mapping words:
                # pixel_grid (zoom-gated grid) and clip_indicator (windowing
                # aid).  Both default off, so the default render is unchanged.
                int(self._mapping.pixel_grid),
                int(self._mapping.clip_indicator),
            ),
        )

    # ---- plane binding -------------------------------------------------------

    def _bind_planes(self, cmd: BindContentPlanes) -> None:
        wgpu = self._wgpu
        self._bound_planes = tuple(cmd.planes)
        self._plane_grids = []
        family_indices: dict[tuple[object, ...], list[int]] = {}
        lod_rows: list[tuple[int, int, int, int]] = []
        plane_rows: list[tuple[int, int, int, int]] = []
        base = 0
        for plane_index, plane in enumerate(self._bound_planes):
            h, w = plane.plane_shape
            lod_base = len(lod_rows)
            grids: list[_LodGrid] = []
            for lod in range(plane.max_lod + 1):
                gw = -(-w // (PAGE << lod))
                gh = -(-h // (PAGE << lod))
                grids.append(_LodGrid(base=base, grid_w=gw, grid_h=gh))
                lod_rows.append((base, gw, gh, 0))
                base += gw * gh
            self._plane_grids.append(grids)
            plane_rows.append((_REP_INDEX[plane.representation], plane.max_lod, lod_base, 0))
            family = (
                plane.document_generation,
                plane.operation_key,
                plane.representation,
            )
            family_indices.setdefault((*family, None), []).append(plane_index)
            family_indices.setdefault((*family, plane.lod_reducer), []).append(plane_index)
        self._plane_family_indices = {
            family: tuple(indices) for family, indices in family_indices.items()
        }

        # Bound physical coverage is the active never-black fallback set.
        # Protect every currently resident page that feeds these plane spans
        # before later commands in the same submission ensure refinements;
        # rebinding atomically releases pages of planes that left the view.
        bound_keys = tuple(
            key for key in self.page_table.resident_keys() if self._flat_indices(key)
        )
        self.page_table.replace_pin_set(_BOUND_PLANES_PIN_OWNER, bound_keys)

        self._flat_table = np.full(max(base, 1), -1, dtype=np.int32)
        self._flat_codec = np.zeros(max(base, 1), dtype=np.uint32)
        self._flat_norm = np.zeros((max(base, 1), 4), dtype=np.float32)
        for key in self.page_table.resident_keys():
            slot = self.page_table.lookup(key)
            if slot is None:  # pragma: no cover - resident keys always resolve
                continue
            codec_flag, norm = self._page_codec.get(key, (0, (0.0, 0.0, 0.0, 0.0)))
            for flat in self._flat_indices(key):
                self._flat_table[flat] = slot.page_index
                self._flat_codec[flat] = codec_flag
                self._flat_norm[flat] = norm

        d = self.device
        lod_info = np.asarray(lod_rows or [(0, 0, 0, 0)], np.uint32)
        planes_info = np.asarray(plane_rows or [(0, 0, 0, 0)], np.uint32)
        self._lod_info_buf = d.create_buffer_with_data(
            data=lod_info.tobytes(), usage=wgpu.BufferUsage.STORAGE
        )
        self._planes_buf = d.create_buffer_with_data(
            data=planes_info.tobytes(), usage=wgpu.BufferUsage.STORAGE
        )
        if self._flat_table.nbytes > self._table_buf.size:
            self._table_buf = d.create_buffer(
                size=self._flat_table.nbytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
        if self._codec_engaged:
            if self._flat_codec.nbytes > self._codec_flag_buf.size:
                self._codec_flag_buf = d.create_buffer(
                    size=self._flat_codec.nbytes,
                    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
                )
            if self._flat_norm.nbytes > self._codec_norm_buf.size:
                self._codec_norm_buf = d.create_buffer(
                    size=self._flat_norm.nbytes,
                    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
                )
        self._bind_epoch += 1
        self._table_dirty = True

    def _flat_indices(self, key: DataChunkKey) -> tuple[int, ...]:
        """Flat-table entries for ``key`` across every bound plane it feeds."""

        if key.rank != 2:
            return ()
        out = []
        family = (
            key.document_generation,
            key.operation_key,
            key.representation,
            None if key.lod.is_native else key.lod.reducer,
        )
        plane_indices = self._plane_family_indices.get(family, ())
        self._plane_lookup_candidates_total += len(plane_indices)
        for plane_index in plane_indices:
            grids = self._plane_grids[plane_index]
            lod = key.lod.level
            if lod >= len(grids):
                continue
            grid = grids[lod]
            oy, ox = key.chunk_origin
            source_page = PAGE << int(lod)
            cx, cy = ox // source_page, oy // source_page
            if not (0 <= cx < grid.grid_w and 0 <= cy < grid.grid_h):
                continue
            out.append(grid.base + cy * grid.grid_w + cx)
        return tuple(out)

    # ---- residency -----------------------------------------------------------

    def _coerce_payload(self, key: DataChunkKey, payload: object) -> tuple[np.ndarray, int]:
        """Validate/pack ``payload`` for the key's pool; return (data, bytes/row)."""

        rep = key.representation
        data = np.asarray(payload)
        if rep == SCALAR_R32F:
            if np.iscomplexobj(data):
                raise ValueError("scalar_r32f chunk payload must be real-valued")
            data = np.ascontiguousarray(data, dtype=np.float32)
            if data.shape != (PAGE, PAGE):
                raise ValueError(f"scalar payload must be ({PAGE},{PAGE}), got {data.shape}")
            return data, PAGE * 4
        if rep == COMPLEX_RG32F:
            if np.iscomplexobj(data):
                packed = np.empty((*data.shape, 2), np.float32)
                packed[..., 0] = data.real
                packed[..., 1] = data.imag
                data = packed
            elif data.ndim == 2:  # real values as complex: zero imaginary plane
                data = np.stack(
                    [data.astype(np.float32), np.zeros_like(data, np.float32)],
                    axis=-1,
                )
            data = np.ascontiguousarray(data, dtype=np.float32)
            if data.shape != (PAGE, PAGE, 2):
                raise ValueError(f"complex payload must be ({PAGE},{PAGE}[,2]), got {data.shape}")
            return data, PAGE * 8
        if rep == RGB8:
            if (
                data.dtype != np.uint8
                or data.ndim != 3
                or data.shape[:2] != (PAGE, PAGE)
                or data.shape[2] not in (3, 4)
            ):
                raise ValueError(
                    f"rgb8 payload must be uint8 ({PAGE},{PAGE},3|4), got {data.dtype} {data.shape}"
                )
            if data.shape[2] == 3:
                rgba = np.empty((PAGE, PAGE, 4), np.uint8)
                rgba[..., :3] = data
                rgba[..., 3] = 255
                data = rgba
            return np.ascontiguousarray(data), PAGE * 4
        if rep == RGB_WINDOWED_RGBA32F:
            data = np.ascontiguousarray(data, dtype=np.float32)
            if data.shape != (PAGE, PAGE, 4):
                raise ValueError(
                    f"rgb_windowed_rgba32f payload must be ({PAGE},{PAGE},4), got {data.shape}"
                )
            return data, PAGE * 16
        raise ValueError(f"unknown chunk representation {rep!r}")  # pragma: no cover

    def _ensure(self, cmd: EnsureChunkResident) -> int:
        if self.page_table.lookup(cmd.key) is not None:
            self.page_table.touch(cmd.key)
            return 0
        rep = cmd.key.representation
        payload, bytes_per_row = self._coerce_payload(cmd.key, cmd.payload)
        pool = self._pools[rep]
        if pool.layer_count == 0:
            raise RuntimeError(f"no layer budget configured for representation {rep!r}")

        # Compression is attempted only when engaged AND the tile's BC round-trip
        # clears the quality gate; otherwise the raw pool takes it (the always-
        # correct fallback).  The page table's pool_id records which path won —
        # the loud channel a parity test asserts on.
        encoded = self._encode_compressed(cmd.key, payload) if self._codec_engaged else None
        if encoded is not None:
            data, norm4 = encoded
            codec_pool = self._codec_pools[rep]
            codec_pool_id = _CODEC_POOL_IDS[rep]
            self._ensure_free_pool_layer(codec_pool_id)
            layer = codec_pool.free_layers.pop()
            bx, by = self._codec_block
            block_bytes = self._codec_block_bytes[rep]
            self.device.queue.write_texture(
                {"texture": codec_pool.texture, "origin": (0, 0, layer)},
                data,
                {"bytes_per_row": (PAGE // bx) * block_bytes, "rows_per_image": PAGE // by},
                (PAGE, PAGE, 1),
            )
            slot = PageSlot(pool_id=codec_pool_id, page_index=layer, slot_index=0)
            self.page_table.bind(cmd.key, slot, nbytes=len(data), pinned=cmd.pinned)
            self._page_codec[cmd.key] = (1, norm4)
            for flat in self._flat_indices(cmd.key):
                self._flat_table[flat] = layer
                self._flat_codec[flat] = 1
                self._flat_norm[flat] = norm4
                self._table_dirty = True
            self._uploads_total += 1
            self._compressed_uploads_total += 1
            self._texture_upload_bytes_total += len(data)
            return 1

        self._ensure_free_pool_layer(_POOL_IDS[rep])
        layer = pool.free_layers.pop()
        self.device.queue.write_texture(
            {"texture": pool.texture, "origin": (0, 0, layer)},
            payload,
            {"bytes_per_row": bytes_per_row, "rows_per_image": PAGE},
            (PAGE, PAGE, 1),
        )
        slot = PageSlot(pool_id=_POOL_IDS[rep], page_index=layer, slot_index=0)
        self.page_table.bind(cmd.key, slot, nbytes=payload.nbytes, pinned=cmd.pinned)
        self._page_codec.pop(cmd.key, None)
        for flat in self._flat_indices(cmd.key):
            self._flat_table[flat] = layer
            self._flat_codec[flat] = 0
            self._flat_norm[flat] = (0.0, 0.0, 0.0, 0.0)
            self._table_dirty = True
        self._uploads_total += 1
        self._texture_upload_bytes_total += int(payload.nbytes)
        return 1

    def _encode_compressed(
        self, key: DataChunkKey, payload: np.ndarray
    ) -> tuple[bytes, tuple[float, float, float, float]] | None:
        """Encode a scalar/complex page to the live codec (BC or ASTC), or None
        to decline (keep raw).

        Declines when the representation has no codec pool or when the measured
        round-trip is below the quality gate — so quality is never silently
        sacrificed and a poorly-compressible tile falls back to the exact pool.
        Both families carry the SAME per-tile (lo, span) normalization (the
        render/compute shaders unscale by it), and complex stores (real, imag),
        never (magnitude, phase), so block compression never smears the ±pi wrap.
        """

        rep = key.representation
        if rep not in self._codec_reps:
            return None
        if (
            self._codec_family == "bc"
            and self._codec_mode == "auto"
            and not bc_codec.numba_encoder_ready()
        ):
            # AUTO never makes a visible page pay NumPy encoding or JIT load.
            # It stays raw until post-first-draw idle work publishes the
            # byte-identical accelerator. Forced ON retains the reference path
            # for deterministic codec tests and explicit experiments.
            return None
        factor = 1 << int(key.lod.level)
        valid_h = max(1, min(PAGE, -(-int(key.chunk_shape[0]) // factor)))
        valid_w = max(1, min(PAGE, -(-int(key.chunk_shape[1]) // factor)))
        valid = payload[:valid_h, :valid_w]
        if not bool(np.isfinite(valid).all()):
            # Scientific non-finites have semantic meaning. Native UNORM block
            # formats cannot represent them, so the exact raw pool is mandatory.
            self._compressed_fallbacks_total += 1
            return None

        def _pad_valid(array: np.ndarray) -> np.ndarray:
            if valid_h == PAGE and valid_w == PAGE:
                return array
            padding = ((0, PAGE - valid_h), (0, PAGE - valid_w))
            if array.ndim > 2:
                padding += tuple((0, 0) for _ in range(array.ndim - 2))
            return np.pad(array, padding, mode="edge")

        astc = self._codec_family == "astc"
        if rep == SCALAR_R32F:
            unit_valid, norm = bc_codec.normalize_tile(valid)
            unit = _pad_valid(unit_valid)
            if astc:
                res = astc_codec.encode_scalar(unit, block=self._codec_block)
                data, decoded = res.data, res.decoded[0]
                quality = bc_codec.quality_of(unit_valid, decoded[:valid_h, :valid_w])
            else:
                data, _h, _w, quality = bc_codec.bc4_encode_with_quality(
                    unit,
                    valid_shape=(valid_h, valid_w),
                )
            if quality.psnr_db < self._codec_min_psnr_db:
                self._compressed_fallbacks_total += 1
                return None
            return data, (float(norm.lo), float(norm.span), 0.0, 0.0)
        # complex_rg32f: two channels holding (real, imag) -- BC5, or ASTC R,G.
        re, im = valid[..., 0], valid[..., 1]
        unit_re_valid, norm_re = bc_codec.normalize_tile(re)
        unit_im_valid, norm_im = bc_codec.normalize_tile(im)
        unit_re = _pad_valid(unit_re_valid)
        unit_im = _pad_valid(unit_im_valid)
        if astc:
            res = astc_codec.encode_two_channel(unit_re, unit_im, block=self._codec_block)
            data, d0, d1 = res.data, res.decoded[0], res.decoded[1]
        else:
            data, height, width = bc_codec.bc5_encode(unit_re, unit_im)
            d0, d1 = bc_codec.bc5_decode(data, height, width)
        re_dec = bc_codec.denormalize_channel(d0[:valid_h, :valid_w], norm_re)
        im_dec = bc_codec.denormalize_channel(d1[:valid_h, :valid_w], norm_im)
        quality = bc_codec.complex_display_quality(re, im, re_dec, im_dec)
        if quality.magnitude_psnr_db < self._codec_min_psnr_db:
            self._compressed_fallbacks_total += 1
            return None
        return data, (
            float(norm_re.lo),
            float(norm_re.span),
            float(norm_im.lo),
            float(norm_im.span),
        )

    def _pool_for_slot(self, slot: PageSlot) -> _Pool:
        """The raw or BC pool object that owns ``slot``'s layer."""

        rep = _REP_BY_POOL_ID[slot.pool_id]
        if slot.pool_id in _CODEC_POOL_IDS.values():
            return self._codec_pools[rep]
        return self._pools[rep]

    def _generate_lod_page(self, cmd: GenerateLodPages) -> bool:
        """Run one resident 2x2 component-mean pass and bind its parent."""

        destination = cmd.destination_key
        if self.page_table.lookup(destination) is not None:
            self.page_table.touch(destination)
            return False
        if destination.rank != 2 or tuple(destination.lod.reduction) != (
            int(destination.lod.level),
            int(destination.lod.level),
        ):
            raise ValueError("wgpu LOD generation requires one isotropic 2-D destination")
        if destination.lod.level <= 0:
            raise ValueError("wgpu LOD generation destination must be reduced")
        if destination.lod.reducer != REDUCER_MEAN:
            raise ValueError(
                "wgpu LOD generation is reducer-honest for component mean only; "
                f"got {destination.lod.reducer!r}"
            )
        representation = destination.representation
        if representation not in self._reduce_pipes:
            raise ValueError(
                "wgpu component-mean LOD generation supports scalar_r32f and "
                f"complex_rg32f only; got {representation!r}"
            )

        source_level = int(destination.lod.level) - 1
        source_reducer = "native" if source_level == 0 else REDUCER_MEAN
        destination_origin = tuple(int(value) for value in destination.chunk_origin)
        source_page_extent = PAGE << source_level
        ordered: list[tuple[DataChunkKey, PageSlot] | None] = [None, None, None, None]
        for key in cmd.source_keys:
            if (
                key.rank != 2
                or key.document_generation != destination.document_generation
                or key.operation_key != destination.operation_key
                or key.dtype != destination.dtype
                or key.representation != representation
                or int(key.lod.level) != source_level
                or key.lod.reducer != source_reducer
            ):
                raise ValueError(
                    "wgpu LOD generation child disagrees with destination value family"
                )
            dy_num = int(key.chunk_origin[0]) - destination_origin[0]
            dx_num = int(key.chunk_origin[1]) - destination_origin[1]
            if (
                dy_num < 0
                or dx_num < 0
                or dy_num % source_page_extent
                or dx_num % source_page_extent
            ):
                raise ValueError("wgpu LOD generation child is off the canonical parent grid")
            dy, dx = dy_num // source_page_extent, dx_num // source_page_extent
            if dy not in (0, 1) or dx not in (0, 1):
                raise ValueError("wgpu LOD generation child lies outside its parent")
            index = dy * 2 + dx
            if ordered[index] is not None:
                raise ValueError("wgpu LOD generation has duplicate child quadrants")
            slot = self.page_table.lookup(key)
            if slot is None:
                raise KeyError(f"LOD generation source is not resident: {key}")
            ordered[index] = (key, slot)

        present = [item for item in ordered if item is not None]
        if not present:  # command shape already prevents this
            raise ValueError("wgpu LOD generation requires resident children")

        # The GPU reduce pass reads source pages by integer coord from the RAW
        # pool only; a BC-pool source layer would alias a different raw page.
        # So when any child is compressed (aggressive AUTO), decode the children
        # to raw on the CPU, box-reduce there, and store the destination as a raw
        # page.  Never a crash once compression is on — the destination is exact
        # w.r.t. the (already lossy) decoded children, and counted as a codec
        # LOD reduction.
        if any(
            item is not None and item[1].pool_id in _CODEC_POOL_IDS.values() for item in ordered
        ):
            return self._generate_lod_page_from_compressed(
                destination, ordered, present, representation
            )
        pool = self._pools[representation]
        if pool.layer_count == 0:
            raise RuntimeError(f"no layer budget configured for representation {representation!r}")
        self.page_table.replace_pin_set(
            _LOD_GENERATION_PIN_OWNER, tuple(key for key, _slot in present)
        )
        destination_layer = None
        try:
            self._ensure_free_pool_layer(_POOL_IDS[representation])
            destination_layer = pool.free_layers.pop()

            fallback = present[0]
            source_views = []
            valid_rows: list[tuple[int, int, int, int]] = []
            for item in ordered:
                key, slot = fallback if item is None else item
                source_views.append(
                    pool.texture.create_view(
                        dimension="2d",
                        base_array_layer=slot.page_index,
                        array_layer_count=1,
                    )
                )
                if item is None:
                    valid_rows.append((0, 0, 0, 0))
                else:
                    factor = 1 << int(key.lod.level)
                    valid_h = -(-int(key.chunk_shape[0]) // factor)
                    valid_w = -(-int(key.chunk_shape[1]) // factor)
                    valid_rows.append((valid_w, valid_h, 0, 0))

            wgpu, d = self._wgpu, self.device
            args = d.create_buffer_with_data(
                data=np.asarray(valid_rows, np.uint32).tobytes(),
                usage=wgpu.BufferUsage.UNIFORM,
            )
            destination_view = pool.texture.create_view(
                dimension="2d",
                base_array_layer=destination_layer,
                array_layer_count=1,
            )
            pipe = self._reduce_pipes[representation]
            bind = d.create_bind_group(
                layout=pipe.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": args, "offset": 0, "size": 64}},
                    *(
                        {"binding": index + 1, "resource": view}
                        for index, view in enumerate(source_views)
                    ),
                    {"binding": 5, "resource": destination_view},
                ],
            )
            encoder = d.create_command_encoder()
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(pipe)
            compute.set_bind_group(0, bind)
            compute.dispatch_workgroups(16, 16)
            compute.end()
            d.queue.submit([encoder.finish()])
        except Exception:
            if destination_layer is not None:
                pool.free_layers.append(destination_layer)
            raise
        finally:
            self.page_table.replace_pin_set(_LOD_GENERATION_PIN_OWNER, ())

        slot = PageSlot(
            pool_id=_POOL_IDS[representation],
            page_index=destination_layer,
            slot_index=0,
        )
        self.page_table.bind(
            destination,
            slot,
            nbytes=PAGE * PAGE * _POOL_TEXEL_BYTES[representation],
        )
        self._page_codec.pop(destination, None)
        for flat in self._flat_indices(destination):
            self._flat_table[flat] = destination_layer
            self._flat_codec[flat] = 0
            self._flat_norm[flat] = (0.0, 0.0, 0.0, 0.0)
            self._table_dirty = True
        return True

    def _generate_lod_page_from_compressed(
        self,
        destination: DataChunkKey,
        ordered: list,
        present: list,
        representation: str,
    ) -> bool:
        """CPU decode -> 2x2 mean box-reduce -> store the destination as a raw page.

        The GPU reduce shader cannot read BC pools by integer coord, so a LOD
        whose children are compressed is reduced on the CPU from the reference-
        decoded child texels (the same decode ``read_resident_page`` performs).
        The result is bound in the raw pool (codec 0) exactly like the GPU path,
        so downstream residency/rendering is oblivious to how it was produced.
        """

        d = self.device
        components = 1 if representation == SCALAR_R32F else 2
        super_shape = (2 * PAGE, 2 * PAGE) + ((components,) if components > 1 else ())
        super_tile = np.zeros(super_shape, np.float32)
        valid = np.zeros((2 * PAGE, 2 * PAGE), bool)
        for index, item in enumerate(ordered):
            if item is None:
                continue
            key, slot = item
            dy, dx = index // 2, index % 2
            factor = 1 << int(key.lod.level)
            valid_h = -(-int(key.chunk_shape[0]) // factor)
            valid_w = -(-int(key.chunk_shape[1]) // factor)
            if slot.pool_id in _CODEC_POOL_IDS.values():
                page = self._read_compressed_page(key, slot)
            else:
                page = self.read_resident_page(key)
            r0, c0 = dy * PAGE, dx * PAGE
            super_tile[r0 : r0 + valid_h, c0 : c0 + valid_w] = page[:valid_h, :valid_w]
            valid[r0 : r0 + valid_h, c0 : c0 + valid_w] = True

        # 2x2 box mean honouring the per-child valid extents (matches the GPU
        # reduce shader: mean over the valid contributors, zero where none).
        blocks = super_tile.reshape(PAGE, 2, PAGE, 2, *((components,) if components > 1 else ()))
        mask = valid.reshape(PAGE, 2, PAGE, 2)
        axes = (1, 3)
        count = mask.sum(axis=axes)
        if components > 1:
            acc = (blocks * mask[..., None]).sum(axis=axes)
            reduced = np.zeros((PAGE, PAGE, components), np.float32)
            nz = count > 0
            reduced[nz] = acc[nz] / count[nz][..., None]
        else:
            acc = (blocks * mask).sum(axis=axes)
            reduced = np.zeros((PAGE, PAGE), np.float32)
            nz = count > 0
            reduced[nz] = acc[nz] / count[nz]
        reduced = reduced.astype(np.float32)

        pool = self._pools[representation]
        self.page_table.replace_pin_set(
            _LOD_GENERATION_PIN_OWNER, tuple(key for key, _slot in present)
        )
        destination_layer = None
        try:
            self._ensure_free_pool_layer(_POOL_IDS[representation])
            destination_layer = pool.free_layers.pop()
            payload, bytes_per_row = self._coerce_payload(destination, reduced)
            d.queue.write_texture(
                {"texture": pool.texture, "origin": (0, 0, destination_layer)},
                payload,
                {"bytes_per_row": bytes_per_row, "rows_per_image": PAGE},
                (PAGE, PAGE, 1),
            )
        except Exception:
            if destination_layer is not None:
                pool.free_layers.append(destination_layer)
            raise
        finally:
            self.page_table.replace_pin_set(_LOD_GENERATION_PIN_OWNER, ())

        slot = PageSlot(
            pool_id=_POOL_IDS[representation],
            page_index=destination_layer,
            slot_index=0,
        )
        self.page_table.bind(
            destination,
            slot,
            nbytes=PAGE * PAGE * _POOL_TEXEL_BYTES[representation],
        )
        self._page_codec.pop(destination, None)
        for flat in self._flat_indices(destination):
            self._flat_table[flat] = destination_layer
            self._flat_codec[flat] = 0
            self._flat_norm[flat] = (0.0, 0.0, 0.0, 0.0)
            self._table_dirty = True
        self._lod_compressed_source_reductions_total += 1
        return True

    def _evict_one_unpinned(self, representation: str, *, pool_id: str | None = None) -> None:
        # ``pool_id`` (when given) confines eviction to one physical pool: the
        # raw and BC pools of a representation have independent layer budgets, so
        # a caller that needs a free BC layer must not be handed a freed raw one.
        def _matches(key: DataChunkKey) -> bool:
            if key.representation != representation:
                return False
            if pool_id is None:
                return True
            slot = self.page_table.lookup(key)
            return slot is not None and slot.pool_id == pool_id

        for key in self.page_table.eviction_candidates():
            if not _matches(key):
                continue
            self._evict(EvictChunk(key))
            return
        # Pool pressure inside one submission may exceed the histogram
        # frontier shield.  Yielding the least-recently-used shield-only pin
        # keeps eviction honest without turning evidence work into a dead
        # commit: the skipped key surfaces as FrameReport.histogram_missing
        # and the consumer retries.  Pins held by any other owner (bound
        # coverage, LOD sources) stay hard, so genuine exhaustion still
        # raises loudly below.
        for key in sorted(
            (key for key in self._histogram_shield_pins if _matches(key)),
            key=self.page_table.last_use,
        ):
            self._histogram_shield_pins.discard(key)
            self.page_table.replace_pin_set(
                _HISTOGRAM_SHIELD_PIN_OWNER, self._histogram_shield_pins
            )
            if self.page_table.is_pinned(key):
                # Another owner still protects this page; restore the shield.
                self._histogram_shield_pins.add(key)
                self.page_table.replace_pin_set(
                    _HISTOGRAM_SHIELD_PIN_OWNER, self._histogram_shield_pins
                )
                continue
            self._evict(EvictChunk(key))
            return
        pool = self._pool_by_id(pool_id or _POOL_IDS[representation])
        resident = tuple(
            key
            for key, slot in self.page_table.slot_items()
            if slot.pool_id == pool_id or (pool_id is None and key.representation == representation)
        )
        pinned = sum(self.page_table.is_pinned(key) for key in resident)
        self._last_pool_exhaustion = (
            f"page pool {representation!r} exhausted and every resident page is pinned: "
            f"pool={pool_id or 'all'} budget={pool.layer_count} "
            f"allocated={pool.allocated_layers} resident={len(resident)} "
            f"pinned={pinned} free={len(pool.free_layers)}"
        )
        raise RuntimeError(self._last_pool_exhaustion)

    def _evict(self, cmd: EvictChunk) -> int:
        # PageTable.unbind purges the key from its stored pin sets; the
        # executor's shield mirror must not drift or the next replace_pin_set
        # would name a non-resident key.
        self._histogram_shield_pins.discard(cmd.key)
        slot = self.page_table.unbind(cmd.key)
        if slot is None:
            return 0
        self._pool_for_slot(slot).free_layers.append(slot.page_index)
        self._page_codec.pop(cmd.key, None)
        for flat in self._flat_indices(cmd.key):
            self._flat_table[flat] = -1
            self._flat_codec[flat] = 0
            self._table_dirty = True
        return 1

    # ---- draw ----------------------------------------------------------------

    def _set_tiles(self, tiles) -> None:
        # Instances are camera-independent, so a pan re-submits the very
        # tuple already resident. Identity is the exact, O(1) test; a
        # producer that rebuilds an equal tuple just repacks as before.
        if tiles is self._tiles:
            return
        if len(tiles) > self._tiles_cap:
            raise ValueError(f"tile count {len(tiles)} exceeds capacity {self._tiles_cap}")
        for t in tiles:
            if t.plane_index >= len(self._bound_planes):
                raise ValueError(
                    f"tile plane_index {t.plane_index} outside bound planes "
                    f"(bound {len(self._bound_planes)})"
                )
        blob = b"".join(
            struct.pack(
                "8f4i",
                *t.dst_rect,
                *t.src_origin,
                *t.src_size,
                t.lod_level,
                t.plane_index,
                int(t.transposed),
                0,
            )
            for t in tiles
        )
        if blob:
            self.device.queue.write_buffer(self._tiles_buf, 0, blob)
        self._tiles = tuple(tiles)

    def _flush_table(self) -> None:
        if self._table_dirty:
            self.device.queue.write_buffer(self._table_buf, 0, self._flat_table.tobytes())
            if self._codec_engaged:
                self.device.queue.write_buffer(self._codec_flag_buf, 0, self._flat_codec.tobytes())
                self.device.queue.write_buffer(self._codec_norm_buf, 0, self._flat_norm.tobytes())
            self._table_dirty = False

    def _histogram(
        self, cmd: DispatchHistogram
    ) -> tuple[object, tuple[float, float] | None, tuple[DataChunkKey, ...]]:
        wgpu, d = self._wgpu, self.device
        if cmd.bins > MAX_HISTOGRAM_BINS:
            raise ValueError(f"executor supports up to {MAX_HISTOGRAM_BINS} histogram bins")
        entries = []
        missing: list[DataChunkKey] = []
        for key in cmd.keys:
            if key.representation == RGB8:
                raise ValueError(f"histogram over RGB presentation chunk {key}")
            slot = self.page_table.lookup(key)
            if slot is None:
                # Correct refusal, wrong blast radius as a raise (2026-07-19
                # dogfood crash): a key sacrificed to pool pressure must mark
                # this result partial, never abort the whole submission.
                missing.append(key)
                continue
            is_compressed = slot.pool_id in _CODEC_POOL_IDS.values()
            if is_compressed and self._histogram_codec_mode == "skip":
                # Path B posture: the compute never reads BC/ASTC texels.  A
                # compressed page is reported partial; its exact stats come from
                # the CPU semantic plane / full-population refinement.  (This is
                # also the only path when compression is off, where no page is
                # ever compressed and this branch is unreachable.)
                missing.append(key)
                continue
            factor = 1 << int(key.lod.level)
            # Path A: a compressed page carries codec 1 and the SAME (lo, span)
            # affine the render path applies, sourced from ``_page_codec`` (the
            # single scale source); the shader samples the BC pool and unscales.
            # A raw page carries codec 0 and an identity affine (exact
            # textureLoad).
            if is_compressed:
                codec_flag, norm = self._page_codec.get(key, (1, (0.0, 1.0, 0.0, 1.0)))
            else:
                codec_flag, norm = 0, (0.0, 0.0, 0.0, 0.0)
            entries.append(
                (
                    slot.page_index,
                    _REP_INDEX[key.representation],
                    int(key.chunk_shape[0]),
                    int(key.chunk_shape[1]),
                    factor,
                    int(codec_flag),
                    float(norm[0]),
                    float(norm[1]),
                    float(norm[2]),
                    float(norm[3]),
                )
            )
        self._histogram_dispatches_total += 1
        n = len(entries)
        if n == 0:
            return np.zeros(cmd.bins, dtype=np.uint32), None, tuple(missing)
        dynamic_bounds = cmd.lo is None
        lo = 0.0 if dynamic_bounds else float(cmd.lo)
        hi = 1.0 if dynamic_bounds else float(cmd.hi)
        uargs = d.create_buffer_with_data(
            data=struct.pack(
                "2f4IfI",
                lo,
                hi,
                n,
                cmd.bins,
                _MODE_INDEX[cmd.mode],
                _SCALE_INDEX[cmd.scale],
                float(cmd.symlog_constant),
                int(dynamic_bounds),
            ),
            usage=wgpu.BufferUsage.UNIFORM,
        )
        # HPage is 48 bytes (two trailing pad words keep the size a multiple of
        # 16, matching naga's storage-array stride): layer(i32) rep(i32)
        # source_h(u32) source_w(u32) factor(u32) codec(u32) then the
        # (lo_r, span_r, lo_g, span_g) f32 affine (identity for a raw page; the
        # render path's norm for a BC page) and two pad words.
        pages_bytes = b"".join(struct.pack("<iiIIIIffffII", *entry, 0, 0) for entry in entries)
        pages_buf = d.create_buffer_with_data(
            data=pages_bytes,
            usage=wgpu.BufferUsage.STORAGE,
        )
        page_stride = 48
        partials = d.create_buffer(size=4 * cmd.bins * n, usage=wgpu.BufferUsage.STORAGE)
        final = d.create_buffer(
            size=4 * cmd.bins, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )
        page_bounds = d.create_buffer(
            size=8 * n,
            usage=wgpu.BufferUsage.STORAGE,
        )
        initial_bounds = (
            np.asarray((0xFFFFFFFF, 0), dtype=np.uint32)
            if dynamic_bounds
            else np.asarray(
                (_ordered_float32(float(cmd.lo)), _ordered_float32(float(cmd.hi))),
                dtype=np.uint32,
            )
        )
        bounds = d.create_buffer_with_data(
            data=initial_bounds.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        bind1 = d.create_bind_group(
            layout=self._partial_pipe.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": uargs, "offset": 0, "size": 32}},
                {
                    "binding": 1,
                    "resource": {"buffer": pages_buf, "offset": 0, "size": page_stride * n},
                },
                {"binding": 2, "resource": self._pools[SCALAR_R32F].view},
                {"binding": 3, "resource": self._pools[COMPLEX_RG32F].view},
                {"binding": 4, "resource": self._pools[RGB_WINDOWED_RGBA32F].view},
                {
                    "binding": 5,
                    "resource": {"buffer": partials, "offset": 0, "size": 4 * cmd.bins * n},
                },
                {"binding": 6, "resource": {"buffer": bounds, "offset": 0, "size": 8}},
                # Path A BC bindings 7-9 (present whenever engaged; the shader's
                # codec branch only samples them for a compressed entry).
                *(
                    [
                        {"binding": 7, "resource": self._codec_pool_view(SCALAR_R32F)},
                        {"binding": 8, "resource": self._codec_pool_view(COMPLEX_RG32F)},
                        {"binding": 9, "resource": self._codec_sampler},
                    ]
                    if self._codec_engaged
                    else []
                ),
            ],
        )
        bind2 = d.create_bind_group(
            layout=self._merge_pipe.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": uargs, "offset": 0, "size": 32}},
                {
                    "binding": 1,
                    "resource": {"buffer": partials, "offset": 0, "size": 4 * cmd.bins * n},
                },
                {"binding": 2, "resource": {"buffer": final, "offset": 0, "size": 4 * cmd.bins}},
            ],
        )
        enc = d.create_command_encoder()
        timestamp_query_set = None
        timestamp_buffer = None
        timestamp_period_ns = None
        timestamp_indices: tuple[int, ...] = ()
        if "timestamp-query" in d.features:
            timestamp_query_set = d.create_query_set(type="timestamp", count=4)
            timestamp_buffer = d.create_buffer(
                size=32,
                usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC,
            )
            from wgpu.backends.wgpu_native._api import libf

            timestamp_period_ns = float(libf.wgpuQueueGetTimestampPeriod(d.queue._internal))
        if dynamic_bounds:
            bounds_bind1 = d.create_bind_group(
                layout=self._bounds_partial_pipe.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": uargs, "offset": 0, "size": 32}},
                    {
                        "binding": 1,
                        "resource": {"buffer": pages_buf, "offset": 0, "size": page_stride * n},
                    },
                    {"binding": 2, "resource": self._pools[SCALAR_R32F].view},
                    {"binding": 3, "resource": self._pools[COMPLEX_RG32F].view},
                    {"binding": 4, "resource": self._pools[RGB_WINDOWED_RGBA32F].view},
                    {"binding": 5, "resource": {"buffer": page_bounds, "offset": 0, "size": 8 * n}},
                    # Path A BC bindings 6-8 (present whenever engaged).
                    *(
                        [
                            {"binding": 6, "resource": self._codec_pool_view(SCALAR_R32F)},
                            {"binding": 7, "resource": self._codec_pool_view(COMPLEX_RG32F)},
                            {"binding": 8, "resource": self._codec_sampler},
                        ]
                        if self._codec_engaged
                        else []
                    ),
                ],
            )
            bounds_bind2 = d.create_bind_group(
                layout=self._bounds_merge_pipe.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": uargs, "offset": 0, "size": 32}},
                    {"binding": 1, "resource": {"buffer": page_bounds, "offset": 0, "size": 8 * n}},
                    {"binding": 2, "resource": {"buffer": bounds, "offset": 0, "size": 8}},
                ],
            )
            cp = enc.begin_compute_pass(
                timestamp_writes=(
                    None
                    if timestamp_query_set is None
                    else {
                        "query_set": timestamp_query_set,
                        "beginning_of_pass_write_index": 0,
                        "end_of_pass_write_index": 1,
                    }
                )
            )
            cp.set_pipeline(self._bounds_partial_pipe)
            cp.set_bind_group(0, bounds_bind1)
            cp.dispatch_workgroups(n)
            cp.set_pipeline(self._bounds_merge_pipe)
            cp.set_bind_group(0, bounds_bind2)
            cp.dispatch_workgroups(1)
            cp.end()
            timestamp_indices = (0, 1, 2, 3)
        elif timestamp_query_set is not None:
            timestamp_indices = (0, 1)
        histogram_timestamp_start = 2 if dynamic_bounds else 0
        cp = enc.begin_compute_pass(
            timestamp_writes=(
                None
                if timestamp_query_set is None
                else {
                    "query_set": timestamp_query_set,
                    "beginning_of_pass_write_index": histogram_timestamp_start,
                    "end_of_pass_write_index": histogram_timestamp_start + 1,
                }
            )
        )
        cp.set_pipeline(self._partial_pipe)
        cp.set_bind_group(0, bind1)
        cp.dispatch_workgroups(n)
        cp.set_pipeline(self._merge_pipe)
        cp.set_bind_group(0, bind2)
        cp.dispatch_workgroups(1)
        cp.end()
        if timestamp_query_set is not None:
            enc.resolve_query_set(
                timestamp_query_set,
                0,
                4 if dynamic_bounds else 2,
                timestamp_buffer,
                0,
            )
        d.queue.submit([enc.finish()])
        if dynamic_bounds:
            return (
                _DeferredHistogramReadback(
                    d,
                    final,
                    bounds,
                    cmd.bins,
                    timestamp_buffer=timestamp_buffer,
                    timestamp_query_set=timestamp_query_set,
                    timestamp_period_ns=timestamp_period_ns,
                    timestamp_indices=timestamp_indices,
                    on_resolve=self._note_histogram_readback_resolve,
                ),
                None,
                tuple(missing),
            )
        self._note_histogram_readback_resolve()
        counts = np.frombuffer(d.queue.read_buffer(final), np.uint32).copy()
        return counts, (float(cmd.lo), float(cmd.hi)), tuple(missing)

    def _note_histogram_readback_resolve(self) -> None:
        self._histogram_readback_resolves_total += 1

    def _batch_dynamic_histogram_readbacks(self, report: FrameReport) -> None:
        """Pack one frame's dynamic histogram outputs into one queue read."""

        rows = tuple(
            value
            for value in report.histograms.values()
            if isinstance(value, _DeferredHistogramReadback)
        )
        if len(rows) < 2:
            return

        def reserve(offset: int, size: int, *, alignment: int = 8) -> tuple[int, int]:
            start = (int(offset) + alignment - 1) // alignment * alignment
            return start, start + int(size)

        spans = []
        offset = 0
        for row in rows:
            counts_offset, offset = reserve(offset, 4 * int(row.bins))
            bounds_offset, offset = reserve(offset, 8)
            timestamp_size = (
                0
                if row.timestamp_buffer is None
                else 8 * (max(tuple(int(i) for i in row.timestamp_indices), default=-1) + 1)
            )
            timestamp_offset, offset = reserve(offset, timestamp_size)
            spans.append(
                (
                    counts_offset,
                    4 * int(row.bins),
                    bounds_offset,
                    8,
                    timestamp_offset,
                    timestamp_size,
                )
            )
        wgpu = self._wgpu
        batch_buffer = self.device.create_buffer(
            size=max(8, offset),
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
        )
        encoder = self.device.create_command_encoder()
        for row, span in zip(rows, spans, strict=True):
            counts_offset, counts_size, bounds_offset, bounds_size, time_offset, time_size = span
            encoder.copy_buffer_to_buffer(
                row.counts_buffer,
                0,
                batch_buffer,
                counts_offset,
                counts_size,
            )
            encoder.copy_buffer_to_buffer(
                row.bounds_buffer,
                0,
                batch_buffer,
                bounds_offset,
                bounds_size,
            )
            if time_size:
                encoder.copy_buffer_to_buffer(
                    row.timestamp_buffer,
                    0,
                    batch_buffer,
                    time_offset,
                    time_size,
                )
        self.device.queue.submit([encoder.finish()])
        batch = _HistogramReadbackBatch(
            self.device,
            batch_buffer,
            tuple(spans),
            on_resolve=self._note_histogram_batch_readback,
        )
        for index, row in enumerate(rows):
            row._batch = batch
            row._batch_index = int(index)

    def _note_histogram_batch_readback(self) -> None:
        self._histogram_batch_readbacks_total += 1

    def _present(
        self,
        target_view,
        fmt: str,
        *,
        target_size: tuple[int, int],
    ) -> None:
        self._flush_table()
        pipe, bind = self._pipeline(fmt)
        # The tile pipeline reads the same camera uniform, so it must be
        # current for every present, not just the ones carrying overlays.
        self._write_camera(target_size)
        if self._overlay_geometry:
            overlay_pipe, overlay_bind = self._overlay_pipeline(fmt)
        enc = self.device.create_command_encoder()
        rp = enc.begin_render_pass(
            color_attachments=[
                {
                    "view": target_view,
                    "load_op": "clear",
                    "store_op": "store",
                    "clear_value": (0, 0, 0, 1),
                }
            ]
        )
        if self._tiles:
            rp.set_pipeline(pipe)
            rp.set_bind_group(0, bind)
            rp.draw(6, len(self._tiles))
        if self._overlay_geometry:
            rp.set_pipeline(overlay_pipe)
            rp.set_bind_group(0, overlay_bind)
            rp.draw(6, len(self._overlay_geometry))
        rp.end()
        self.device.queue.submit([enc.finish()])

    # ---- RendererExecutor ---------------------------------------------------

    def submit(
        self,
        submission: FrameSubmission,
        *,
        present_to=None,
        present_format="rgba8unorm",
        present_size: tuple[int, int] | None = None,
    ) -> FrameReport:
        """Execute one ordered command batch.

        ``present_to`` (optional) is a texture view to render into instead of
        the internal offscreen target — the live-canvas path.
        """

        report = FrameReport(generation=submission.generation)
        upload_bytes_before = int(self._texture_upload_bytes_total)
        generated_pages = []
        # Histogram frontier shield: keys a DispatchHistogram in THIS batch
        # will sample must survive the batch's own residency churn (the view
        # snapshots frontiers before the submission is built; an earlier
        # ensure LRU-evicting one killed a live commit, 2026-07-19).  The
        # executor sees the whole ordered command tuple up front, so it pins
        # those keys — already-resident ones now, planned ones as they bind —
        # and releases every shield pin before returning.
        self._histogram_shield_wanted = frozenset(
            key
            for command in submission.commands
            if isinstance(command, DispatchHistogram)
            for key in command.keys
        )
        self._histogram_shield_pins = {
            key for key in self._histogram_shield_wanted if key in self.page_table
        }
        if self._histogram_shield_pins:
            self.page_table.replace_pin_set(
                _HISTOGRAM_SHIELD_PIN_OWNER, self._histogram_shield_pins
            )
        try:
            for index, cmd in enumerate(submission.commands):
                if isinstance(cmd, BindContentPlanes):
                    self._bind_planes(cmd)
                elif isinstance(cmd, EnsureChunkResident):
                    report.uploads += self._ensure(cmd)
                    self._shield_histogram_key(cmd.key)
                elif isinstance(cmd, EvictChunk):
                    report.evictions += self._evict(cmd)
                elif isinstance(cmd, GenerateLodPages):
                    if self._generate_lod_page(cmd):
                        generated_pages.append(cmd.destination_key)
                    self._shield_histogram_key(cmd.destination_key)
                elif isinstance(cmd, UpdateTileInstances):
                    self._set_tiles(cmd.tiles)
                elif isinstance(cmd, UpdateOverlayGeometry):
                    writes = self._set_overlay_geometry(cmd.primitives)
                    report.overlay_buffer_writes += writes
                    self._overlay_buffer_writes_total += writes
                elif isinstance(cmd, UpdateGlyphAtlas):
                    report.glyph_atlas_uploads += self._update_glyph_atlas(cmd)
                elif isinstance(cmd, UpdateWidgetAtlas):
                    report.widget_atlas_uploads += self._update_widget_atlas(cmd)
                elif isinstance(cmd, SetOverlayCamera):
                    self._overlay_camera = cmd
                elif isinstance(cmd, SetDisplayMapping):
                    self._mapping = cmd.mapping
                    self._write_mapping()
                    self._write_lut(cmd.mapping.lut)
                elif isinstance(cmd, DispatchHistogram):
                    counts, bounds, missing = self._histogram(cmd)
                    report.histograms[index] = counts
                    report.histogram_bounds[index] = bounds
                    if missing:
                        report.histogram_missing[index] = missing
                elif isinstance(cmd, PresentGeneration):
                    view = present_to if present_to is not None else self._target.create_view()
                    self._present(
                        view,
                        present_format if present_to is not None else "rgba8unorm",
                        target_size=(
                            tuple(int(value) for value in present_size)
                            if present_to is not None and present_size is not None
                            else self._target_size
                        ),
                    )
                    report.presented = True
                else:  # pragma: no cover - protocol/executor version skew guard
                    raise TypeError(f"unknown renderer command {type(cmd).__name__}")
        finally:
            self._histogram_shield_wanted = frozenset()
            self._histogram_shield_pins = set()
            self.page_table.replace_pin_set(_HISTOGRAM_SHIELD_PIN_OWNER, ())
        self._batch_dynamic_histogram_readbacks(report)
        report.lod_pages_generated = tuple(generated_pages)
        report.upload_bytes = int(self._texture_upload_bytes_total) - upload_bytes_before
        report.wait_completed = self.device.queue.on_submitted_work_done_sync
        return report

    def _shield_histogram_key(self, key: DataChunkKey) -> None:
        """Pin a freshly-bound key the current submission's histograms need."""

        if (
            key not in self._histogram_shield_wanted
            or key in self._histogram_shield_pins
            or key not in self.page_table
        ):
            return
        self._histogram_shield_pins.add(key)
        self.page_table.replace_pin_set(_HISTOGRAM_SHIELD_PIN_OWNER, self._histogram_shield_pins)

    # ---- audit oracles ------------------------------------------------------

    @property
    def uploads_total(self) -> int:
        return self._uploads_total

    @property
    def overlay_buffer_writes_total(self) -> int:
        return self._overlay_buffer_writes_total

    @property
    def histogram_dispatches_total(self) -> int:
        return int(self._histogram_dispatches_total)

    @property
    def histogram_readback_resolves_total(self) -> int:
        return int(self._histogram_readback_resolves_total)

    @property
    def histogram_batch_readbacks_total(self) -> int:
        return int(self._histogram_batch_readbacks_total)

    @property
    def plane_lookup_candidates_total(self) -> int:
        return int(self._plane_lookup_candidates_total)

    @property
    def glyph_atlas_uploads_total(self) -> int:
        return self._glyph_atlas_uploads_total

    @property
    def widget_atlas_uploads_total(self) -> int:
        return self._widget_atlas_uploads_total

    @property
    def bound_planes(self) -> tuple:
        return self._bound_planes

    @property
    def codec_engaged(self) -> bool:
        """Whether native compressed pools (BC or ASTC) are live and taking
        eligible tiles this session."""

        return bool(self._codec_engaged)

    @property
    def codec_family(self) -> str:
        """The live codec family: ``"bc"`` (discrete/NVIDIA), ``"astc"``
        (integrated/Intel), or ``"none"`` when compression is not engaged."""

        return self._codec_family if self._codec_engaged else "none"

    @property
    def codec_min_psnr_db(self) -> float:
        return float(self._codec_min_psnr_db)

    @property
    def codec_block(self) -> tuple[int, int]:
        """The block size of the live codec (BC is always 4x4; ASTC is the
        configured ``astc_block``)."""

        return tuple(self._codec_block)

    def codec_pool_format(self, representation: str) -> str | None:
        """The wgpu ``TextureFormat`` a representation's compressed pool was
        created with (e.g. ``"bc4-r-unorm"`` or ``"astc-4x4-unorm"``) — the loud
        channel proving ASTC vs BC — or None when that pool is not live."""

        if not self._codec_engaged or representation not in self._codec_pools:
            return None
        return self._codec_pool_formats[representation]

    @property
    def compressed_uploads_total(self) -> int:
        """Pages actually stored in a BC pool (the loud 'compression was used' channel)."""

        return int(self._compressed_uploads_total)

    @property
    def compressed_fallbacks_total(self) -> int:
        """Eligible tiles that declined BC (quality gate) and stayed raw."""

        return int(self._compressed_fallbacks_total)

    @property
    def texture_upload_bytes_total(self) -> int:
        """Actual raw or block-compressed bytes submitted by resident ensures."""

        return int(self._texture_upload_bytes_total)

    @property
    def active_resident_bytes(self) -> int:
        """Logical bytes owned by currently bound pages (raw + compressed)."""

        return int(self.page_table.resident_bytes())

    @property
    def allocated_pool_bytes(self) -> int:
        """Current physical texture-array extents, including idle layers.

        Pools grow toward their logical eviction budgets on demand; this reports
        the current arrays, not their maxima. Drivers may add alignment/metadata,
        so it remains payload arithmetic rather than process-wide VRAM telemetry.
        """

        total = 0
        for rep, pool in self._pools.items():
            total += int(pool.allocated_layers) * PAGE * PAGE * _POOL_TEXEL_BYTES[rep]
        bx, by = self._codec_block
        for rep, pool in self._codec_pools.items():
            layer_bytes = (PAGE // bx) * (PAGE // by) * self._codec_block_bytes[rep]
            total += int(pool.allocated_layers) * layer_bytes
        return int(total)

    @property
    def pool_grows_total(self) -> int:
        """Physical texture-array replacements performed to satisfy demand."""

        return int(self._pool_grows_total)

    @property
    def pool_growth_copy_bytes_total(self) -> int:
        """Bytes copied between old/new arrays during demand growth."""

        return int(self._pool_growth_copy_bytes_total)

    def page_is_compressed(self, key: DataChunkKey) -> bool:
        """Whether ``key``'s resident page lives in a BC pool (page-table truth)."""

        slot = self.page_table.lookup(key)
        return slot is not None and slot.pool_id in _CODEC_POOL_IDS.values()

    @property
    def lod_compressed_source_reductions_total(self) -> int:
        """LOD pages generated via the CPU decode+reduce path (compressed children)."""

        return int(self._lod_compressed_source_reductions_total)

    @property
    def histogram_codec_mode(self) -> str:
        """How the GPU histogram/bounds compute treats compressed pages.

        ``"gpu_compressed"`` (Path A) samples the BC pool's lossy texels;
        ``"skip"`` (Path B posture) excludes compressed pages and reports them
        ``histogram_missing`` so their exact stats come from the CPU semantic
        plane / full-population refinement.
        """

        return self._histogram_codec_mode

    def set_histogram_codec_mode(self, mode: str) -> None:
        """Switch the compressed-page histogram/bounds path (see the property)."""

        mode = str(mode).lower()
        if mode not in ("gpu_compressed", "skip"):
            raise ValueError(
                f"histogram_codec_mode must be 'gpu_compressed' or 'skip', got {mode!r}"
            )
        self._histogram_codec_mode = mode

    def pool_budget(self, representation: str) -> int:
        return int(self._pool_budgets[representation])

    def ensure_pool_budgets(self, budgets: dict[str, int]) -> dict[str, int]:
        """Grow logical pool ceilings in place while preserving resident pages."""

        grown: dict[str, int] = {}
        max_layers = int(self.device.limits["max-texture-array-layers"])
        for representation, requested in dict(budgets or {}).items():
            if representation not in self._pool_budgets:
                raise ValueError(f"unknown pool representation {representation!r}")
            requested = max(0, int(requested))
            if requested > max_layers:
                raise RuntimeError(
                    "wgpu active plane pages exceed the device texture-array limit: "
                    f"representation={representation!r}, needed={requested}, "
                    f"max_layers={max_layers}"
                )
            previous = int(self._pool_budgets[representation])
            if requested <= previous:
                continue
            self._pool_budgets[representation] = requested
            self._pools[representation].layer_count = requested
            codec_pool = self._codec_pools.get(representation)
            if codec_pool is not None:
                codec_pool.layer_count = requested
            grown[representation] = requested
        return grown

    def ensure_raw_pool_capacity(self, representation: str, required_layers: int) -> int:
        """Allocate a known raw working set in one copy-preserving growth.

        Page admission normally grows geometrically because future demand is
        unknown.  The display owner, however, computes the complete visible
        transaction before submitting it.  Jumping directly to that measured
        requirement avoids repeatedly copying an increasingly large texture
        array while preserving every resident layer index.
        """

        if representation not in self._pools:
            raise ValueError(f"unknown pool representation {representation!r}")
        required = max(0, int(required_layers))
        pool = self._pools[representation]
        if required > int(pool.layer_count):
            raise RuntimeError(
                "wgpu raw working set exceeds its logical pool capacity: "
                f"representation={representation!r}, needed={required}, "
                f"budget={pool.layer_count}"
            )
        if required > int(pool.allocated_layers):
            self._grow_pool(_POOL_IDS[representation], minimum_layers=required)
        return int(pool.allocated_layers)

    def replace_resident_pin_set(self, owner: object, keys) -> frozenset[DataChunkKey]:
        """Replace one owner's pins with the resident subset of ``keys``.

        Hidden presentation transactions know their complete future page set
        before every page has arrived.  Let that owner refresh its protection
        after each bounded upload batch without asking :class:`PageTable` to
        pin not-yet-resident pages.
        """

        resident = frozenset(key for key in keys if key in self.page_table)
        self.page_table.replace_pin_set(owner, resident)
        return resident

    def resident_pin_set(self, owner: object) -> frozenset[DataChunkKey]:
        return self.page_table.pin_set(owner)

    def pool_diagnostics_snapshot(self) -> tuple[dict[str, object], ...]:
        """Physical capacity, allocation, residency, and pins per representation."""

        rows = []
        for representation in REPRESENTATIONS:
            raw_pool_id = _POOL_IDS[representation]
            codec_pool_id = _CODEC_POOL_IDS.get(representation)
            raw_keys = tuple(
                key for key, slot in self.page_table.slot_items() if slot.pool_id == raw_pool_id
            )
            codec_keys = tuple(
                key
                for key, slot in self.page_table.slot_items()
                if codec_pool_id is not None and slot.pool_id == codec_pool_id
            )
            raw_pool = self._pools[representation]
            codec_pool = self._codec_pools.get(representation)
            rows.append(
                {
                    "representation": representation,
                    "budget_layers": int(self._pool_budgets[representation]),
                    "raw_allocated_layers": int(raw_pool.allocated_layers),
                    "raw_resident_layers": len(raw_keys),
                    "raw_pinned_layers": sum(self.page_table.is_pinned(key) for key in raw_keys),
                    "raw_free_layers": len(raw_pool.free_layers),
                    "codec_allocated_layers": (
                        0 if codec_pool is None else int(codec_pool.allocated_layers)
                    ),
                    "codec_resident_layers": len(codec_keys),
                    "codec_pinned_layers": sum(
                        self.page_table.is_pinned(key) for key in codec_keys
                    ),
                    "codec_free_layers": (0 if codec_pool is None else len(codec_pool.free_layers)),
                }
            )
        return tuple(rows)

    @property
    def last_pool_exhaustion(self) -> str:
        return str(self._last_pool_exhaustion)

    def pool_free_layers(self, representation: str) -> int:
        return len(self._pools[representation].free_layers)

    def pool_allocated_layers(self, representation: str) -> int:
        return int(self._pools[representation].allocated_layers)

    def codec_pool_free_layers(self, representation: str) -> int:
        pool = self._codec_pools.get(representation)
        return 0 if pool is None else len(pool.free_layers)

    def codec_pool_allocated_layers(self, representation: str) -> int:
        pool = self._codec_pools.get(representation)
        return 0 if pool is None else int(pool.allocated_layers)

    def read_target(self) -> np.ndarray:
        """Physical-truth oracle: the offscreen target as (h, w, 4) uint8."""

        w, h = self._target_size
        data = self.device.queue.read_texture(
            {"texture": self._target},
            {"bytes_per_row": w * 4, "rows_per_image": h},
            (w, h, 1),
        )
        return np.frombuffer(data, np.uint8).reshape(h, w, 4).copy()

    def read_resident_page(self, key: DataChunkKey) -> np.ndarray:
        """Physical-truth oracle: copy one exact page-table binding to CPU."""

        slot = self.page_table.lookup(key)
        if slot is None:
            raise KeyError(f"cannot read non-resident page {key}")
        representation = key.representation
        if slot.pool_id in _CODEC_POOL_IDS.values():
            return self._read_compressed_page(key, slot)
        pool = self._pools[representation]
        data = self.device.queue.read_texture(
            {"texture": pool.texture, "origin": (0, 0, slot.page_index)},
            {
                "bytes_per_row": PAGE * _POOL_TEXEL_BYTES[representation],
                "rows_per_image": PAGE,
            },
            (PAGE, PAGE, 1),
        )
        if representation == SCALAR_R32F:
            shape, dtype = (PAGE, PAGE), np.float32
        elif representation == COMPLEX_RG32F:
            shape, dtype = (PAGE, PAGE, 2), np.float32
        elif representation == RGB8:
            shape, dtype = (PAGE, PAGE, 4), np.uint8
        else:
            shape, dtype = (PAGE, PAGE, 4), np.float32
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()

    def _read_compressed_page(self, key: DataChunkKey, slot: PageSlot) -> np.ndarray:
        """Read a codec page's blocks back and reference-decode + denormalize them.

        Returns the same shape/dtype as the raw-pool oracle, so a parity test can
        compare a compressed page against the raw page it stands in for.  Handles
        both codec families: the block grid depends on the family's block size,
        the decode dispatches to BC or ASTC.
        """

        representation = key.representation
        pool = self._codec_pools[representation]
        bx, by = self._codec_block
        block_bytes = self._codec_block_bytes[representation]
        nbx, nby = PAGE // bx, PAGE // by
        block_row = nbx * block_bytes
        raw = self.device.queue.read_texture(
            {"texture": pool.texture, "origin": (0, 0, slot.page_index)},
            {"bytes_per_row": block_row, "rows_per_image": nby},
            (PAGE, PAGE, 1),
        )
        # read_texture sizes the readback by texel height (PAGE rows of
        # ``block_row`` bytes) even for a block-compressed texture; the block
        # data occupies only the first ``nby`` rows (verified on Intel ASTC and
        # NVIDIA BC).  Slice them out so the reference decoder sees exactly the
        # encoded blocks.
        data = np.frombuffer(bytes(raw), np.uint8).reshape(-1, block_row)[:nby].tobytes()
        _flag, (lo_r, span_r, lo_g, span_g) = self._page_codec.get(key, (1, (0.0, 1.0, 0.0, 1.0)))
        astc = self._codec_family == "astc"
        if representation == SCALAR_R32F:
            if astc:
                (unit,) = astc_codec.astc_decode(data, self._codec_block, PAGE, PAGE, 1)
            else:
                unit = bc_codec.bc4_decode(data, PAGE, PAGE)
            return (unit * np.float32(span_r) + np.float32(lo_r)).astype(np.float32)
        if astc:
            d0, d1 = astc_codec.astc_decode(data, self._codec_block, PAGE, PAGE, 2)
        else:
            d0, d1 = bc_codec.bc5_decode(data, PAGE, PAGE)
        out = np.empty((PAGE, PAGE, 2), np.float32)
        out[..., 0] = d0 * np.float32(span_r) + np.float32(lo_r)
        out[..., 1] = d1 * np.float32(span_g) + np.float32(lo_g)
        return out
