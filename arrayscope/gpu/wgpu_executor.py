"""wgpu implementation of the renderer command protocol (gate-B seed).

First real implementation of :mod:`arrayscope.gpu.command_protocol`, grown
from the proven gate-B harness (``experiments/wgpu_gate_b/virtual_tensor.py``
oracles A–G).  Scope is deliberately the experiment's: ONE 2-D plane pyramid
(native L0 plus mean-reduced coarser levels), one ``rg32float`` page pool
(scalar chunks store a zero imaginary plane — memory-honest pools per
representation are a follow-up), one instanced draw per present, and the
G6 two-pass magnitude histogram.

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
from dataclasses import dataclass

import numpy as np

from arrayscope.gpu.command_protocol import (
    DispatchHistogram,
    DisplayMapping,
    EnsureChunkResident,
    EvictChunk,
    FrameReport,
    FrameSubmission,
    PresentGeneration,
    SetDisplayMapping,
    UpdateTileInstances,
)
from arrayscope.gpu.keys import (
    COMPLEX_RG32F,
    REDUCER_MEAN,
    REDUCER_NATIVE,
    ChunkLod,
    DataChunkKey,
)
from arrayscope.gpu.page_table import PageSlot, PageTable

PAGE = 256
_POOL_ID = "wgpu-rg32f-pool"

_MODE_INDEX = {"magnitude": 0, "phase": 1, "real": 2, "imag": 3}


def plane_chunk_key(
    document_generation: object,
    operation_key: object,
    lod_level: int,
    chunk_x: int,
    chunk_y: int,
    *,
    dtype: str = "complex64",
) -> DataChunkKey:
    """Canonical key for one 256² page of a 2-D plane pyramid.

    ``chunk_origin`` is expressed in the LOD's own sample space (uniform
    plane-pixel pages across LODs, ADR 0056 §3).
    """

    if lod_level == 0:
        lod = ChunkLod()
    else:
        lod = ChunkLod(
            level=lod_level,
            factor=1 << lod_level,
            reduction=(lod_level, lod_level),
            reducer=REDUCER_MEAN,
        )
    return DataChunkKey(
        document_generation=document_generation,
        operation_key=operation_key,
        lod=lod,
        chunk_origin=(chunk_y * PAGE, chunk_x * PAGE),
        chunk_shape=(PAGE, PAGE),
        dtype=dtype,
        representation=COMPLEX_RG32F,
    )


_RENDER_WGSL = """
struct Mapping {
    mode: u32,
    max_lod: u32,
    level_lo: f32,
    level_hi: f32,
};
struct LodInfo { base: u32, grid_w: u32, grid_h: u32, _pad: u32 };
struct Tile {
    dst: vec4<f32>,
    src: vec4<f32>,
    lod: u32,
    _pad0: u32, _pad1: u32, _pad2: u32,
};
@group(0) @binding(0) var<uniform> mapping: Mapping;
@group(0) @binding(1) var<storage, read> page_table: array<i32>;
@group(0) @binding(2) var<storage, read> lod_info: array<LodInfo>;
@group(0) @binding(3) var<storage, read> tiles: array<Tile>;
@group(0) @binding(4) var pool: texture_2d_array<f32>;
@group(0) @binding(5) var lut: texture_2d<f32>;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) src: vec2<f32>,
    @location(1) @interpolate(flat) lod: u32,
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
    return out;
}

fn sample_value(src_l0: vec2<f32>, lod_req: u32) -> vec2<f32> {
    for (var lod = lod_req; lod <= mapping.max_lod; lod = lod + 1u) {
        let info = lod_info[lod];
        let scale = f32(1u << lod);
        let limit = vec2<f32>(f32(info.grid_w * 256u) - 1.0, f32(info.grid_h * 256u) - 1.0);
        let coord = vec2<u32>(clamp(src_l0 / scale, vec2<f32>(0.0), limit));
        let chunk = coord / 256u;
        let entry = page_table[info.base + chunk.y * info.grid_w + chunk.x];
        if (entry >= 0) {
            let texel = coord % 256u;
            return textureLoad(pool, vec2<i32>(texel), entry, 0).rg;
        }
    }
    return vec2<f32>(0.0, 0.0);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let v = sample_value(in.src, in.lod);
    var x: f32;
    switch mapping.mode {
        case 0u: { x = length(v); }
        case 1u: { x = atan2(v.y, v.x); }
        case 2u: { x = v.x; }
        default: { x = v.y; }
    }
    let g = clamp((x - mapping.level_lo) / (mapping.level_hi - mapping.level_lo), 0.0, 1.0);
    // Nearest-entry LUT indexing, mirroring the CPU display reference.
    let idx = clamp(i32(round(g * 255.0)), 0, 255);
    return textureLoad(lut, vec2<i32>(idx, 0), 0);
}
"""

_HISTO_WGSL = """
struct HArgs { lo: f32, hi: f32, n_pages: u32, bins: u32 };
@group(0) @binding(0) var<uniform> args: HArgs;
@group(0) @binding(1) var<storage, read> layers: array<i32>;
@group(0) @binding(2) var pool: texture_2d_array<f32>;
@group(0) @binding(3) var<storage, read_write> partials: array<atomic<u32>>;

var<workgroup> local_bins: array<atomic<u32>, 64>;

@compute @workgroup_size(256)
fn partial(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if (li < args.bins) { atomicStore(&local_bins[li], 0u); }
    workgroupBarrier();
    let layer = layers[wg.x];
    let y = i32(li);
    for (var x = 0; x < 256; x = x + 1) {
        let v = textureLoad(pool, vec2<i32>(x, y), layer, 0).rg;
        let mag = length(v);
        let t = (mag - args.lo) / (args.hi - args.lo);
        let b = clamp(i32(t * f32(args.bins)), 0, i32(args.bins) - 1);
        atomicAdd(&local_bins[b], 1u);
    }
    workgroupBarrier();
    if (li < args.bins) {
        atomicStore(&partials[wg.x * args.bins + li], atomicLoad(&local_bins[li]));
    }
}

@group(0) @binding(0) var<uniform> margs: HArgs;
@group(0) @binding(1) var<storage, read> merged_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> final_bins: array<u32>;

@compute @workgroup_size(64)
fn merge(@builtin(local_invocation_index) li: u32) {
    if (li >= margs.bins) { return; }
    var acc = 0u;
    for (var p = 0u; p < margs.n_pages; p = p + 1u) {
        acc = acc + merged_in[p * margs.bins + li];
    }
    final_bins[li] = acc;
}
"""


@dataclass
class _LodGrid:
    base: int
    grid_w: int
    grid_h: int


class WgpuPlaneExecutor:
    """Protocol executor for one 2-D plane pyramid on a wgpu device."""

    def __init__(
        self,
        plane_shape: tuple[int, int],
        *,
        max_lod: int = 1,
        pool_layers: int = 64,
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

        h, w = (int(v) for v in plane_shape)
        if h <= 0 or w <= 0:
            raise ValueError(f"plane shape must be positive, got {plane_shape}")
        self.plane_shape = (h, w)
        self.max_lod = int(max_lod)

        # Flat GPU page table: one span per LOD level.
        self._grids: list[_LodGrid] = []
        base = 0
        for lod in range(self.max_lod + 1):
            gw = -(-w // (PAGE << lod))
            gh = -(-h // (PAGE << lod))
            self._grids.append(_LodGrid(base=base, grid_w=gw, grid_h=gh))
            base += gw * gh
        self._flat_table = np.full(base, -1, dtype=np.int32)
        self._table_dirty = True

        self.page_table = PageTable()
        self._free_layers = list(range(pool_layers))
        self._tiles: tuple = ()
        self._mapping = DisplayMapping()
        self._uploads_total = 0

        d = self.device
        self.pool = d.create_texture(
            size=(PAGE, PAGE, pool_layers),
            format="rg32float",
            usage=(
                wgpu.TextureUsage.TEXTURE_BINDING
                | wgpu.TextureUsage.COPY_DST
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.STORAGE_BINDING
            ),
        )
        self._pool_view = self.pool.create_view(dimension="2d-array")
        self._table_buf = d.create_buffer(
            size=max(4, self._flat_table.nbytes),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        lod_info = np.zeros((self.max_lod + 1, 4), np.uint32)
        for i, g in enumerate(self._grids):
            lod_info[i] = (g.base, g.grid_w, g.grid_h, 0)
        self._lod_info_buf = d.create_buffer_with_data(
            data=lod_info.tobytes(), usage=wgpu.BufferUsage.STORAGE
        )
        self._tiles_cap = 512
        self._tiles_buf = d.create_buffer(
            size=48 * self._tiles_cap,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self._mapping_buf = d.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
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

        shader = d.create_shader_module(code=_RENDER_WGSL)
        self._pipelines: dict[str, object] = {}
        self._shader = shader
        self._bind_layout = None
        self._bind = None
        self._histo_mod = d.create_shader_module(code=_HISTO_WGSL)
        self._partial_pipe = d.create_compute_pipeline(
            layout="auto", compute={"module": self._histo_mod, "entry_point": "partial"}
        )
        self._merge_pipe = d.create_compute_pipeline(
            layout="auto", compute={"module": self._histo_mod, "entry_point": "merge"}
        )

    # ---- internals ----------------------------------------------------------

    def _pipeline(self, fmt: str):
        if fmt not in self._pipelines:
            pipe = self.device.create_render_pipeline(
                layout="auto",
                vertex={"module": self._shader, "entry_point": "vs_main"},
                primitive={"topology": "triangle-list"},
                fragment={
                    "module": self._shader,
                    "entry_point": "fs_main",
                    "targets": [{"format": fmt}],
                },
            )
            bind = self.device.create_bind_group(
                layout=pipe.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": self._mapping_buf, "offset": 0, "size": 16}},
                    {"binding": 1, "resource": {"buffer": self._table_buf, "offset": 0, "size": self._table_buf.size}},
                    {"binding": 2, "resource": {"buffer": self._lod_info_buf, "offset": 0, "size": self._lod_info_buf.size}},
                    {"binding": 3, "resource": {"buffer": self._tiles_buf, "offset": 0, "size": self._tiles_buf.size}},
                    {"binding": 4, "resource": self._pool_view},
                    {"binding": 5, "resource": self._lut_tex.create_view()},
                ],
            )
            self._pipelines[fmt] = (pipe, bind)
        return self._pipelines[fmt]

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
                "2I2f",
                _MODE_INDEX[self._mapping.mode],
                self.max_lod,
                self._mapping.level_lo,
                self._mapping.level_hi,
            ),
        )

    def _flat_index(self, key: DataChunkKey) -> int:
        lod = key.lod.level
        if lod > self.max_lod:
            raise ValueError(f"chunk lod {lod} exceeds executor max_lod {self.max_lod}")
        grid = self._grids[lod]
        oy, ox = key.chunk_origin
        cx, cy = ox // PAGE, oy // PAGE
        if not (0 <= cx < grid.grid_w and 0 <= cy < grid.grid_h):
            raise ValueError(f"chunk {key.chunk_origin} outside lod-{lod} grid")
        return grid.base + cy * grid.grid_w + cx

    def _ensure(self, cmd: EnsureChunkResident) -> int:
        if self.page_table.lookup(cmd.key) is not None:
            self.page_table.touch(cmd.key)
            return 0
        payload = np.asarray(cmd.payload)
        if payload.ndim == 2:  # scalar chunk: zero imaginary plane
            payload = np.stack(
                [payload.astype(np.float32), np.zeros_like(payload, np.float32)],
                axis=-1,
            )
        payload = np.ascontiguousarray(payload, dtype=np.float32)
        if payload.shape != (PAGE, PAGE, 2):
            raise ValueError(f"payload must be ({PAGE},{PAGE}[,2]), got {payload.shape}")
        if not self._free_layers:
            self._evict_one_unpinned()
        layer = self._free_layers.pop()
        self.device.queue.write_texture(
            {"texture": self.pool, "origin": (0, 0, layer)},
            payload,
            {"bytes_per_row": PAGE * 8, "rows_per_image": PAGE},
            (PAGE, PAGE, 1),
        )
        slot = PageSlot(pool_id=_POOL_ID, page_index=layer, slot_index=0)
        self.page_table.bind(cmd.key, slot, nbytes=payload.nbytes, pinned=cmd.pinned)
        self._flat_table[self._flat_index(cmd.key)] = layer
        self._table_dirty = True
        self._uploads_total += 1
        return 1

    def _evict_one_unpinned(self) -> None:
        for key in self.page_table.eviction_candidates():
            self._evict(EvictChunk(key))
            return
        raise RuntimeError("page pool exhausted and every resident page is pinned")

    def _evict(self, cmd: EvictChunk) -> int:
        slot = self.page_table.unbind(cmd.key)
        if slot is None:
            return 0
        self._free_layers.append(slot.page_index)
        self._flat_table[self._flat_index(cmd.key)] = -1
        self._table_dirty = True
        return 1

    def _set_tiles(self, tiles) -> None:
        if len(tiles) > self._tiles_cap:
            raise ValueError(f"tile count {len(tiles)} exceeds capacity {self._tiles_cap}")
        blob = b"".join(
            struct.pack("8f4i", *t.dst_rect, *t.src_origin, *t.src_size, t.lod_level, 0, 0, 0)
            for t in tiles
        )
        if blob:
            self.device.queue.write_buffer(self._tiles_buf, 0, blob)
        self._tiles = tuple(tiles)

    def _flush_table(self) -> None:
        if self._table_dirty:
            self.device.queue.write_buffer(self._table_buf, 0, self._flat_table.tobytes())
            self._table_dirty = False

    def _histogram(self, cmd: DispatchHistogram) -> np.ndarray:
        wgpu, d = self._wgpu, self.device
        if cmd.bins > 64:
            raise ValueError("seed executor supports up to 64 bins (workgroup array)")
        layers = []
        for key in cmd.keys:
            slot = self.page_table.lookup(key)
            if slot is None:
                raise KeyError(f"histogram over non-resident chunk {key}")
            layers.append(slot.page_index)
        n = len(layers)
        uargs = d.create_buffer_with_data(
            data=struct.pack("2f2I", cmd.lo, cmd.hi, n, cmd.bins),
            usage=wgpu.BufferUsage.UNIFORM,
        )
        layers_buf = d.create_buffer_with_data(
            data=np.asarray(layers, np.int32).tobytes(), usage=wgpu.BufferUsage.STORAGE
        )
        partials = d.create_buffer(size=4 * cmd.bins * n, usage=wgpu.BufferUsage.STORAGE)
        final = d.create_buffer(
            size=4 * cmd.bins, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )
        bind1 = d.create_bind_group(
            layout=self._partial_pipe.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": uargs, "offset": 0, "size": 16}},
                {"binding": 1, "resource": {"buffer": layers_buf, "offset": 0, "size": 4 * n}},
                {"binding": 2, "resource": self._pool_view},
                {"binding": 3, "resource": {"buffer": partials, "offset": 0, "size": 4 * cmd.bins * n}},
            ],
        )
        bind2 = d.create_bind_group(
            layout=self._merge_pipe.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": uargs, "offset": 0, "size": 16}},
                {"binding": 1, "resource": {"buffer": partials, "offset": 0, "size": 4 * cmd.bins * n}},
                {"binding": 2, "resource": {"buffer": final, "offset": 0, "size": 4 * cmd.bins}},
            ],
        )
        enc = d.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self._partial_pipe)
        cp.set_bind_group(0, bind1)
        cp.dispatch_workgroups(n)
        cp.set_pipeline(self._merge_pipe)
        cp.set_bind_group(0, bind2)
        cp.dispatch_workgroups(1)
        cp.end()
        d.queue.submit([enc.finish()])
        return np.frombuffer(d.queue.read_buffer(final), np.uint32).copy()

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
        for index, cmd in enumerate(submission.commands):
            if isinstance(cmd, EnsureChunkResident):
                report.uploads += self._ensure(cmd)
            elif isinstance(cmd, EvictChunk):
                report.evictions += self._evict(cmd)
            elif isinstance(cmd, UpdateTileInstances):
                self._set_tiles(cmd.tiles)
            elif isinstance(cmd, SetDisplayMapping):
                self._mapping = cmd.mapping
                self._write_mapping()
                self._write_lut(cmd.mapping.lut)
            elif isinstance(cmd, DispatchHistogram):
                report.histograms[index] = self._histogram(cmd)
            elif isinstance(cmd, PresentGeneration):
                view = present_to if present_to is not None else self._target.create_view()
                self._present(view, present_format if present_to is not None else "rgba8unorm")
                report.presented = True
            else:  # pragma: no cover - protocol/executor version skew guard
                raise TypeError(f"unknown renderer command {type(cmd).__name__}")
        report.wait_completed = self.device.queue.on_submitted_work_done_sync
        return report

    # ---- audit oracles ------------------------------------------------------

    @property
    def uploads_total(self) -> int:
        return self._uploads_total

    def read_target(self) -> np.ndarray:
        """Physical-truth oracle: the offscreen target as (h, w, 4) uint8."""

        w, h = self._target_size
        data = self.device.queue.read_texture(
            {"texture": self._target},
            {"bytes_per_row": w * 4, "rows_per_image": h},
            (w, h, 1),
        )
        return np.frombuffer(data, np.uint8).reshape(h, w, 4).copy()
