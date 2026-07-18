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
  and pinned exhaustion is a loud error.
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

import struct
from dataclasses import dataclass, field

import numpy as np

from arrayscope.gpu.command_protocol import (
    BindContentPlanes,
    DispatchHistogram,
    DisplayMapping,
    EnsureChunkResident,
    EvictChunk,
    FrameReport,
    FrameSubmission,
    GenerateLodPages,
    PresentGeneration,
    SetDisplayMapping,
    UpdateTileInstances,
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
_BOUND_PLANES_PIN_OWNER = "wgpu-bound-content-planes"
_LOD_GENERATION_PIN_OWNER = "wgpu-lod-generation-sources"


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
    _pad2: u32,
    _pad3: u32,
};
struct LodInfo { base: u32, grid_w: u32, grid_h: u32, _pad: u32 };
struct PlaneInfo { rep: u32, max_lod: u32, lod_base: u32, _pad: u32 };
struct Tile {
    dst: vec4<f32>,
    src: vec4<f32>,
    lod: u32,
    plane: u32,
    _pad1: u32, _pad2: u32,
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
    let cpos = t.dst.xy + q * t.dst.zw;
    var out: VOut;
    out.pos = vec4<f32>(cpos.x * 2.0 - 1.0, 1.0 - cpos.y * 2.0, 0.0, 1.0);
    out.src = t.src.xy + q * t.src.zw;
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

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let p = planes[in.plane];
    let r = resolve(in.plane, in.src, in.lod);
    if (p.rep == 2u) {
        // Display-ready RGB: sampled as-is, levels/LUT bypassed.
        if (r.layer < 0) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
        let c = textureLoad(rgb_pool, r.texel, r.layer, 0);
        return vec4<f32>(c.rgb, 1.0);
    }
    if (p.rep == 3u) {
        // VisPy parity: preserve the color plane and modulate it by one
        // levels-normalized scalar plane (packed in alpha), not by three
        // independent per-channel windows.
        if (r.layer < 0) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
        let c = textureLoad(rgb_windowed_pool, r.texel, r.layer, 0);
        let scalar = apply_scale(c.a);
        let intensity = clamp(
            (scalar - mapping.level_lo) / (mapping.level_hi - mapping.level_lo),
            0.0,
            1.0,
        );
        return vec4<f32>(c.rgb * intensity, 1.0);
    }
    var v = vec2<f32>(0.0, 0.0);
    if (r.layer >= 0) {
        if (p.rep == 0u) {
            v = vec2<f32>(textureLoad(scalar_pool, r.texel, r.layer, 0).r, 0.0);
        } else {
            v = textureLoad(complex_pool, r.texel, r.layer, 0).rg;
        }
    }
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
    if (mapping.phase_color != 0u && p.rep == 1u && mapping.mode != 1u) {
        let phase = atan2(v.y, v.x);
        let phase_g = clamp(
            (phase + 3.141592653589793) / 6.283185307179586,
            0.0,
            1.0,
        );
        let phase_idx = clamp(i32(round(phase_g * 255.0)), 0, 255);
        let color = textureLoad(lut, vec2<i32>(phase_idx, 0), 0);
        return vec4<f32>(color.rgb * g, color.a);
    }
    // Nearest-entry LUT indexing, mirroring the CPU display reference.
    let idx = clamp(i32(round(g * 255.0)), 0, 255);
    return textureLoad(lut, vec2<i32>(idx, 0), 0);
}
"""

_HISTO_WGSL = """
struct HArgs {
    lo: f32,
    hi: f32,
    n_pages: u32,
    bins: u32,
    mode: u32,
    scale: u32,
    symlog_constant: f32,
    dynamic_bounds: u32,
};
struct HPage {
    layer: i32,
    rep: i32,
    source_h: u32,
    source_w: u32,
    factor: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(0) @binding(0) var<uniform> args: HArgs;
@group(0) @binding(1) var<storage, read> pages: array<HPage>;
@group(0) @binding(2) var scalar_pool: texture_2d_array<f32>;
@group(0) @binding(3) var complex_pool: texture_2d_array<f32>;
@group(0) @binding(4) var rgb_windowed_pool: texture_2d_array<f32>;
@group(0) @binding(5) var<storage, read_write> partials: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> final_bounds: array<u32>;

var<workgroup> local_bins: array<atomic<u32>, 512>;

fn ordered_float(value: f32) -> u32 {
    let bits = bitcast<u32>(value);
    return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn float_from_ordered(value: u32) -> f32 {
    let bits = select(value ^ 0x80000000u, ~value, value < 0x80000000u);
    return bitcast<f32>(bits);
}

fn finite_value(value: f32) -> bool {
    return value == value && abs(value) <= 3.402823466e+38;
}

fn mapped_value(page: HPage, coord: vec2<i32>) -> f32 {
    var value: f32;
    if (page.rep == 0) {
        value = textureLoad(scalar_pool, coord, page.layer, 0).r;
    } else if (page.rep == 1) {
        let pair = textureLoad(complex_pool, coord, page.layer, 0).rg;
        switch args.mode {
            case 0u: { value = length(pair); }
            case 1u: { value = atan2(pair.y, pair.x); }
            case 2u: { value = pair.x; }
            default: { value = pair.y; }
        }
    } else {
        value = textureLoad(rgb_windowed_pool, coord, page.layer, 0).a;
    }
    switch args.scale {
        case 0u: { return value; }
        case 1u: { return log(max(value, 0.0)) / log(10.0); }
        default: {
            return sign(value) * log(
                1.0 + abs(value) / pow(10.0, args.symlog_constant)
            ) / log(10.0);
        }
    }
}

fn stored_h(page: HPage) -> u32 {
    return (page.source_h + page.factor - 1u) / page.factor;
}

fn stored_w(page: HPage) -> u32 {
    return (page.source_w + page.factor - 1u) / page.factor;
}

fn source_weight(page: HPage, y: u32, x: u32) -> u32 {
    let y0 = y * page.factor;
    let x0 = x * page.factor;
    return min(page.factor, page.source_h - y0) * min(page.factor, page.source_w - x0);
}

@group(0) @binding(0) var<uniform> bargs: HArgs;
@group(0) @binding(1) var<storage, read> bpages: array<HPage>;
@group(0) @binding(2) var bscalar_pool: texture_2d_array<f32>;
@group(0) @binding(3) var bcomplex_pool: texture_2d_array<f32>;
@group(0) @binding(4) var brgb_windowed_pool: texture_2d_array<f32>;
@group(0) @binding(5) var<storage, read_write> page_bounds: array<u32>;

var<workgroup> local_low: atomic<u32>;
var<workgroup> local_high: atomic<u32>;

fn bounds_mapped_value(page: HPage, coord: vec2<i32>) -> f32 {
    var value: f32;
    if (page.rep == 0) {
        value = textureLoad(bscalar_pool, coord, page.layer, 0).r;
    } else if (page.rep == 1) {
        let pair = textureLoad(bcomplex_pool, coord, page.layer, 0).rg;
        switch bargs.mode {
            case 0u: { value = length(pair); }
            case 1u: { value = atan2(pair.y, pair.x); }
            case 2u: { value = pair.x; }
            default: { value = pair.y; }
        }
    } else {
        value = textureLoad(brgb_windowed_pool, coord, page.layer, 0).a;
    }
    switch bargs.scale {
        case 0u: { return value; }
        case 1u: { return log(max(value, 0.0)) / log(10.0); }
        default: {
            return sign(value) * log(
                1.0 + abs(value) / pow(10.0, bargs.symlog_constant)
            ) / log(10.0);
        }
    }
}

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


def _reduce_wgsl(*, value_type: str, load_suffix: str, storage_format: str) -> str:
    """Build the component-mean shader for one honest pool representation."""

    zero = "0.0" if value_type == "f32" else "vec2<f32>(0.0)"
    stored = (
        "vec4<f32>(mean, 0.0, 0.0, 0.0)"
        if value_type == "f32"
        else "vec4<f32>(mean, 0.0, 0.0)"
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
    SCALAR_R32F: _reduce_wgsl(
        value_type="f32", load_suffix="r", storage_format="r32float"
    ),
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
    layer_count: int = 0


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
    _resolved: tuple[np.ndarray, tuple[float, float] | None] | None = None
    _gpu_elapsed_ms: float | None = None

    def resolve(self) -> tuple[np.ndarray, tuple[float, float] | None]:
        if self._resolved is None:
            counts = np.frombuffer(
                self.device.queue.read_buffer(self.counts_buffer), np.uint32
            ).copy()
            raw_bounds = np.frombuffer(
                self.device.queue.read_buffer(self.bounds_buffer), np.uint32
            ).copy()
            finite_bounds = (
                None
                if int(raw_bounds[0]) == 0xFFFFFFFF
                else (
                    _float32_from_ordered(int(raw_bounds[0])),
                    _float32_from_ordered(int(raw_bounds[1])),
                )
            )
            self._resolved = (counts[: int(self.bins)], finite_bounds)
            if self.timestamp_buffer is not None:
                timestamps = np.frombuffer(
                    self.device.queue.read_buffer(self.timestamp_buffer), np.uint64
                ).copy()
                indices = tuple(int(index) for index in self.timestamp_indices)
                elapsed_ticks = sum(
                    max(0, int(timestamps[stop]) - int(timestamps[start]))
                    for start, stop in zip(indices[::2], indices[1::2])
                )
                self._gpu_elapsed_ms = (
                    float(elapsed_ticks)
                    * float(self.timestamp_period_ns or 1.0)
                    / 1_000_000.0
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
        target_size: tuple[int, int] = (768, 768),
        device: object = None,
    ) -> None:
        import wgpu  # deferred: module import stays wgpu-free

        self._wgpu = wgpu
        if device is None:
            from wgpu.backends.wgpu_native.extras import set_instance_extras

            try:
                # Vulkan-only instance: the GL backend's EGL re-init is fatal
                # under Wayland (gate-B Tier 0). Harmless if already set.
                set_instance_extras(backends=["Vulkan"])
            except RuntimeError:
                pass  # instance already exists (e.g. shared with a canvas)
            adapter = wgpu.gpu.request_adapter_sync(power_preference="low-power")
            device = adapter.request_device_sync()
        self.device = device

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

        self.page_table = PageTable()
        self._bound_planes: tuple = ()
        self._plane_grids: list[list[_LodGrid]] = []
        self._flat_table = np.full(1, -1, dtype=np.int32)
        self._table_dirty = True
        self._tiles: tuple = ()
        self._mapping = DisplayMapping()
        self._uploads_total = 0

        d = self.device
        self._pools: dict[str, _Pool] = {}
        for rep in REPRESENTATIONS:
            layers = max(1, self._pool_budgets[rep])
            usage = (
                wgpu.TextureUsage.TEXTURE_BINDING
                | wgpu.TextureUsage.COPY_DST
                | wgpu.TextureUsage.COPY_SRC
            )
            if rep in _REDUCE_WGSL:
                usage |= wgpu.TextureUsage.STORAGE_BINDING
            texture = d.create_texture(
                size=(PAGE, PAGE, layers),
                format=_POOL_FORMATS[rep],
                usage=usage,
            )
            self._pools[rep] = _Pool(
                representation=rep,
                texture=texture,
                view=texture.create_view(dimension="2d-array"),
                free_layers=list(range(self._pool_budgets[rep])),
                layer_count=self._pool_budgets[rep],
            )

        # Bind-group epoch: bumped whenever a bound buffer is recreated
        # (plane rebind, table growth) so cached bind groups are rebuilt.
        self._bind_epoch = 0
        self._table_buf = d.create_buffer(
            size=max(16, self._flat_table.nbytes),
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

        self._shader = d.create_shader_module(code=_RENDER_WGSL)
        self._pipelines: dict[str, object] = {}
        self._binds: dict[str, tuple[object, int]] = {}
        self._histo_mod = d.create_shader_module(code=_HISTO_WGSL)
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
                    {"binding": 0, "resource": {"buffer": self._mapping_buf, "offset": 0, "size": 32}},
                    {"binding": 1, "resource": {"buffer": self._table_buf, "offset": 0, "size": self._table_buf.size}},
                    {"binding": 2, "resource": {"buffer": self._lod_info_buf, "offset": 0, "size": self._lod_info_buf.size}},
                    {"binding": 3, "resource": {"buffer": self._planes_buf, "offset": 0, "size": self._planes_buf.size}},
                    {"binding": 4, "resource": {"buffer": self._tiles_buf, "offset": 0, "size": self._tiles_buf.size}},
                    {"binding": 5, "resource": self._pools[SCALAR_R32F].view},
                    {"binding": 6, "resource": self._pools[COMPLEX_RG32F].view},
                    {"binding": 7, "resource": self._pools[RGB8].view},
                    {"binding": 8, "resource": self._pools[RGB_WINDOWED_RGBA32F].view},
                    {"binding": 9, "resource": self._lut_tex.create_view()},
                ],
            )
            self._binds[fmt] = (bind, self._bind_epoch)
        return pipe, self._binds[fmt][0]

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
                0,
                0,
            ),
        )

    # ---- plane binding -------------------------------------------------------

    def _bind_planes(self, cmd: BindContentPlanes) -> None:
        wgpu = self._wgpu
        self._bound_planes = tuple(cmd.planes)
        self._plane_grids = []
        lod_rows: list[tuple[int, int, int, int]] = []
        plane_rows: list[tuple[int, int, int, int]] = []
        base = 0
        for plane in self._bound_planes:
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
            plane_rows.append(
                (_REP_INDEX[plane.representation], plane.max_lod, lod_base, 0)
            )

        # Bound physical coverage is the active never-black fallback set.
        # Protect every currently resident page that feeds these plane spans
        # before later commands in the same submission ensure refinements;
        # rebinding atomically releases pages of planes that left the view.
        bound_keys = tuple(
            key for key in self.page_table.resident_keys() if self._flat_indices(key)
        )
        self.page_table.replace_pin_set(_BOUND_PLANES_PIN_OWNER, bound_keys)

        self._flat_table = np.full(max(base, 1), -1, dtype=np.int32)
        for key in self.page_table.resident_keys():
            slot = self.page_table.lookup(key)
            if slot is None:  # pragma: no cover - resident keys always resolve
                continue
            for flat in self._flat_indices(key):
                self._flat_table[flat] = slot.page_index

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
        self._bind_epoch += 1
        self._table_dirty = True

    def _flat_indices(self, key: DataChunkKey) -> tuple[int, ...]:
        """Flat-table entries for ``key`` across every bound plane it feeds."""

        if key.rank != 2:
            return ()
        out = []
        for plane, grids in zip(self._bound_planes, self._plane_grids):
            if (
                key.document_generation != plane.document_generation
                or key.operation_key != plane.operation_key
                or key.representation != plane.representation
                or (
                    not key.lod.is_native
                    and key.lod.reducer != plane.lod_reducer
                )
            ):
                continue
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
                packed = np.empty(data.shape + (2,), np.float32)
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
            if data.dtype != np.uint8 or data.ndim != 3 or data.shape[:2] != (PAGE, PAGE) or data.shape[2] not in (3, 4):
                raise ValueError(
                    f"rgb8 payload must be uint8 ({PAGE},{PAGE},3|4), got "
                    f"{data.dtype} {data.shape}"
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
                    f"rgb_windowed_rgba32f payload must be "
                    f"({PAGE},{PAGE},4), got {data.shape}"
                )
            return data, PAGE * 16
        raise ValueError(f"unknown chunk representation {rep!r}")  # pragma: no cover

    def _ensure(self, cmd: EnsureChunkResident) -> int:
        if self.page_table.lookup(cmd.key) is not None:
            self.page_table.touch(cmd.key)
            return 0
        payload, bytes_per_row = self._coerce_payload(cmd.key, cmd.payload)
        pool = self._pools[cmd.key.representation]
        if pool.layer_count == 0:
            raise RuntimeError(
                f"no layer budget configured for representation "
                f"{cmd.key.representation!r}"
            )
        if not pool.free_layers:
            self._evict_one_unpinned(cmd.key.representation)
        layer = pool.free_layers.pop()
        self.device.queue.write_texture(
            {"texture": pool.texture, "origin": (0, 0, layer)},
            payload,
            {"bytes_per_row": bytes_per_row, "rows_per_image": PAGE},
            (PAGE, PAGE, 1),
        )
        slot = PageSlot(
            pool_id=_POOL_IDS[cmd.key.representation], page_index=layer, slot_index=0
        )
        self.page_table.bind(cmd.key, slot, nbytes=payload.nbytes, pinned=cmd.pinned)
        for flat in self._flat_indices(cmd.key):
            self._flat_table[flat] = layer
            self._table_dirty = True
        self._uploads_total += 1
        return 1

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
            raise ValueError(
                "wgpu LOD generation requires one isotropic 2-D destination"
            )
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
        pool = self._pools[representation]
        if pool.layer_count == 0:
            raise RuntimeError(
                f"no layer budget configured for representation {representation!r}"
            )
        self.page_table.replace_pin_set(
            _LOD_GENERATION_PIN_OWNER, tuple(key for key, _slot in present)
        )
        destination_layer = None
        try:
            if not pool.free_layers:
                self._evict_one_unpinned(representation)
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
        for flat in self._flat_indices(destination):
            self._flat_table[flat] = destination_layer
            self._table_dirty = True
        return True

    def _evict_one_unpinned(self, representation: str) -> None:
        for key in self.page_table.eviction_candidates():
            if key.representation != representation:
                continue
            self._evict(EvictChunk(key))
            return
        raise RuntimeError(
            f"page pool {representation!r} exhausted and every resident page is pinned"
        )

    def _evict(self, cmd: EvictChunk) -> int:
        slot = self.page_table.unbind(cmd.key)
        if slot is None:
            return 0
        self._pools[_REP_BY_POOL_ID[slot.pool_id]].free_layers.append(slot.page_index)
        for flat in self._flat_indices(cmd.key):
            self._flat_table[flat] = -1
            self._table_dirty = True
        return 1

    # ---- draw ----------------------------------------------------------------

    def _set_tiles(self, tiles) -> None:
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
                "8f4i", *t.dst_rect, *t.src_origin, *t.src_size, t.lod_level, t.plane_index, 0, 0
            )
            for t in tiles
        )
        if blob:
            self.device.queue.write_buffer(self._tiles_buf, 0, blob)
        self._tiles = tuple(tiles)

    def _flush_table(self) -> None:
        if self._table_dirty:
            self.device.queue.write_buffer(self._table_buf, 0, self._flat_table.tobytes())
            self._table_dirty = False

    def _histogram(
        self, cmd: DispatchHistogram
    ) -> tuple[object, tuple[float, float] | None]:
        wgpu, d = self._wgpu, self.device
        if cmd.bins > MAX_HISTOGRAM_BINS:
            raise ValueError(
                f"executor supports up to {MAX_HISTOGRAM_BINS} histogram bins"
            )
        entries = []
        for key in cmd.keys:
            if key.representation == RGB8:
                raise ValueError(f"histogram over RGB presentation chunk {key}")
            slot = self.page_table.lookup(key)
            if slot is None:
                raise KeyError(f"histogram over non-resident chunk {key}")
            factor = 1 << int(key.lod.level)
            entries.append(
                (
                    slot.page_index,
                    _REP_INDEX[key.representation],
                    int(key.chunk_shape[0]),
                    int(key.chunk_shape[1]),
                    factor,
                    0,
                    0,
                    0,
                )
            )
        n = len(entries)
        if n == 0:
            return np.zeros(cmd.bins, dtype=np.uint32), None
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
        pages_buf = d.create_buffer_with_data(
            data=np.asarray(entries, np.int32).tobytes(),
            usage=wgpu.BufferUsage.STORAGE,
        )
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
                {"binding": 1, "resource": {"buffer": pages_buf, "offset": 0, "size": 32 * n}},
                {"binding": 2, "resource": self._pools[SCALAR_R32F].view},
                {"binding": 3, "resource": self._pools[COMPLEX_RG32F].view},
                {"binding": 4, "resource": self._pools[RGB_WINDOWED_RGBA32F].view},
                {"binding": 5, "resource": {"buffer": partials, "offset": 0, "size": 4 * cmd.bins * n}},
                {"binding": 6, "resource": {"buffer": bounds, "offset": 0, "size": 8}},
            ],
        )
        bind2 = d.create_bind_group(
            layout=self._merge_pipe.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": uargs, "offset": 0, "size": 32}},
                {"binding": 1, "resource": {"buffer": partials, "offset": 0, "size": 4 * cmd.bins * n}},
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

            timestamp_period_ns = float(
                libf.wgpuQueueGetTimestampPeriod(d.queue._internal)
            )
        if dynamic_bounds:
            bounds_bind1 = d.create_bind_group(
                layout=self._bounds_partial_pipe.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": uargs, "offset": 0, "size": 32}},
                    {"binding": 1, "resource": {"buffer": pages_buf, "offset": 0, "size": 32 * n}},
                    {"binding": 2, "resource": self._pools[SCALAR_R32F].view},
                    {"binding": 3, "resource": self._pools[COMPLEX_RG32F].view},
                    {"binding": 4, "resource": self._pools[RGB_WINDOWED_RGBA32F].view},
                    {"binding": 5, "resource": {"buffer": page_bounds, "offset": 0, "size": 8 * n}},
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
            return _DeferredHistogramReadback(
                d,
                final,
                bounds,
                cmd.bins,
                timestamp_buffer=timestamp_buffer,
                timestamp_query_set=timestamp_query_set,
                timestamp_period_ns=timestamp_period_ns,
                timestamp_indices=timestamp_indices,
            ), None
        counts = np.frombuffer(d.queue.read_buffer(final), np.uint32).copy()
        return counts, (float(cmd.lo), float(cmd.hi))

    def _present(self, target_view, fmt: str) -> None:
        self._flush_table()
        pipe, bind = self._pipeline(fmt)
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
        rp.end()
        self.device.queue.submit([enc.finish()])

    # ---- RendererExecutor ---------------------------------------------------

    def submit(
        self, submission: FrameSubmission, *, present_to=None, present_format="rgba8unorm"
    ) -> FrameReport:
        """Execute one ordered command batch.

        ``present_to`` (optional) is a texture view to render into instead of
        the internal offscreen target — the live-canvas path.
        """

        report = FrameReport(generation=submission.generation)
        generated_pages = []
        for index, cmd in enumerate(submission.commands):
            if isinstance(cmd, BindContentPlanes):
                self._bind_planes(cmd)
            elif isinstance(cmd, EnsureChunkResident):
                report.uploads += self._ensure(cmd)
            elif isinstance(cmd, EvictChunk):
                report.evictions += self._evict(cmd)
            elif isinstance(cmd, GenerateLodPages):
                if self._generate_lod_page(cmd):
                    generated_pages.append(cmd.destination_key)
            elif isinstance(cmd, UpdateTileInstances):
                self._set_tiles(cmd.tiles)
            elif isinstance(cmd, SetDisplayMapping):
                self._mapping = cmd.mapping
                self._write_mapping()
                self._write_lut(cmd.mapping.lut)
            elif isinstance(cmd, DispatchHistogram):
                counts, bounds = self._histogram(cmd)
                report.histograms[index] = counts
                report.histogram_bounds[index] = bounds
            elif isinstance(cmd, PresentGeneration):
                view = present_to if present_to is not None else self._target.create_view()
                self._present(view, present_format if present_to is not None else "rgba8unorm")
                report.presented = True
            else:  # pragma: no cover - protocol/executor version skew guard
                raise TypeError(f"unknown renderer command {type(cmd).__name__}")
        report.lod_pages_generated = tuple(generated_pages)
        report.wait_completed = self.device.queue.on_submitted_work_done_sync
        return report

    # ---- audit oracles ------------------------------------------------------

    @property
    def uploads_total(self) -> int:
        return self._uploads_total

    @property
    def bound_planes(self) -> tuple:
        return self._bound_planes

    def pool_budget(self, representation: str) -> int:
        return int(self._pool_budgets[representation])

    def pool_free_layers(self, representation: str) -> int:
        return len(self._pools[representation].free_layers)

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
