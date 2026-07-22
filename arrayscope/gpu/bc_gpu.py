"""GPU side of the BC-texture path (G7 Phase B): compute BC4 encoder + helpers.

Three GPU pieces the CPU :mod:`arrayscope.gpu.bc_codec` cannot provide:

* :func:`create_compute_device` -- a compute-only wgpu device (no surface), used
  by the encoder, the benchmark and the tests.  ``high-performance`` selects the
  discrete adapter (with the NVIDIA ICD env vars); ``low-power`` the integrated.
* :class:`GpuBc4Encoder` -- the "our own simple GPU encode" the owner asked for: a
  WGSL compute shader that packs a normalized f32 tile to BC4 blocks, one 4x4
  block per invocation (per-block min/max endpoints + nearest-index assignment).
  This lets GPU-resident derived tiles (LOD / compute outputs) be compressed on
  the device for host spill without a CPU round-trip.  It uses the same endpoints
  and palette as :func:`arrayscope.gpu.bc_codec.bc4_encode` and is quality-
  equivalent (f32 rounding can flip a few indices at exact ties).
* :func:`upload_bc4_texture` / :func:`sample_bc4_texture` -- create a real
  ``bc4-r-unorm`` texture from block bytes and (for the benchmark's honest
  quality closure) sample it through a render pass to r32float and read it back,
  proving the *hardware* sampler returns what the CPU reference decode predicts.

Import health: ``wgpu`` is imported lazily inside the functions/classes that need
a device, so importing this module never creates a device or requires an adapter.
"""

from __future__ import annotations

import contextlib

import numpy as np

__all__ = [
    "GpuBc4Encoder",
    "GpuDecodeUnavailable",
    "create_compute_device",
    "sample_bc4_texture",
    "upload_bc4_texture",
]


class GpuDecodeUnavailable(RuntimeError):
    """No usable wgpu compute device (no adapter / no Vulkan / software host)."""


def create_compute_device(power_preference: str = "high-performance", *, features=()):
    """Create a compute-only wgpu device (no surface/canvas), or raise.

    ``features`` are required device features (e.g. ``["texture-compression-bc"]``);
    a feature the adapter lacks raises :class:`GpuDecodeUnavailable`.
    """

    try:
        import wgpu

        with contextlib.suppress(Exception):
            from wgpu.backends.wgpu_native.extras import set_instance_extras

            with contextlib.suppress(RuntimeError):
                set_instance_extras(backends=["Vulkan"])
    except Exception as exc:  # pragma: no cover - import env dependent
        raise GpuDecodeUnavailable(f"wgpu unavailable: {exc}") from exc
    adapter = wgpu.gpu.request_adapter_sync(power_preference=power_preference)
    if adapter is None:
        raise GpuDecodeUnavailable("no wgpu adapter")
    have = {str(f) for f in adapter.features}
    missing = [f for f in features if f not in have]
    if missing:
        raise GpuDecodeUnavailable(f"adapter lacks features {missing}")
    device = adapter.request_device_sync(required_features=list(features))
    if device is None:  # pragma: no cover - driver dependent
        raise GpuDecodeUnavailable("adapter yielded no device")
    return device


# One invocation per 4x4 block.  Reads 16 f32 texels, quantizes to 8-bit, takes
# per-block min/max endpoints (red0 > red1 = the 8-value interpolated BC4 mode),
# assigns the nearest of the 8 palette levels to each texel, and packs the block
# into two little-endian u32 words: word0 = red0 | red1<<8 | (indices << 16),
# word1 = the high bits of the 48-bit index field.  Quality-equivalent to the CPU
# encoder in bc_codec.bc4_encode (f32 ties aside).
_BC4_ENCODE_WGSL = """
struct Dims { width: u32, height: u32, nbx: u32, nby: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> field: array<f32>;
@group(0) @binding(2) var<storage, read_write> blocks: array<u32>;

@compute @workgroup_size(64)
fn encode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let blk = gid.x;
    let nblocks = dims.nbx * dims.nby;
    if (blk >= nblocks) {
        return;
    }
    let bx = blk % dims.nbx;
    let by = blk / dims.nbx;

    var texels: array<f32, 16>;
    var lo: f32 = 1e30;
    var hi: f32 = -1e30;
    for (var t: u32 = 0u; t < 16u; t = t + 1u) {
        let r = t / 4u;
        let c = t % 4u;
        var px = bx * 4u + c;
        var py = by * 4u + r;
        // clamp to edge (matches the CPU encoder's edge padding)
        if (px >= dims.width) { px = dims.width - 1u; }
        if (py >= dims.height) { py = dims.height - 1u; }
        var v = field[py * dims.width + px];
        v = clamp(v, 0.0, 1.0);
        let q = round(v * 255.0);  // 8-bit quantize, as the CPU path does
        texels[t] = q;
        lo = min(lo, q);
        hi = max(hi, q);
    }
    let red0 = u32(hi);
    let red1 = u32(lo);
    let r0 = hi;
    let r1 = lo;

    var word0: u32 = red0 | (red1 << 8u);
    var word1: u32 = 0u;
    for (var t: u32 = 0u; t < 16u; t = t + 1u) {
        let value = texels[t];
        var best_i: u32 = 0u;
        var best_d: f32 = 1e30;
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            var level: f32;
            if (j == 0u) {
                level = r0;
            } else if (j == 1u) {
                level = r1;
            } else {
                // Palette index j>=2 is the (j-1)-th interpolation step:
                // ((8-j)*r0 + (j-1)*r1) / 7  (matches bc_codec._bc4_palette / the
                // BC4 spec the hardware sampler implements).
                let fj = f32(j);
                level = ((8.0 - fj) * r0 + (fj - 1.0) * r1) / 7.0;
            }
            let d = abs(value - level);
            if (d < best_d) {
                best_d = d;
                best_i = j;
            }
        }
        let absbit = 16u + 3u * t;
        if (absbit >= 32u) {
            word1 = word1 | (best_i << (absbit - 32u));
        } else {
            word0 = word0 | (best_i << absbit);
            if (absbit + 3u > 32u) {
                word1 = word1 | (best_i >> (32u - absbit));
            }
        }
    }
    blocks[blk * 2u] = word0;
    blocks[blk * 2u + 1u] = word1;
}
"""


class GpuBc4Encoder:
    """Compute-shader BC4 encoder (quality-equivalent to ``bc_codec.bc4_encode``)."""

    def __init__(self, device) -> None:
        import wgpu

        self._wgpu = wgpu
        self.device = device
        module = device.create_shader_module(code=_BC4_ENCODE_WGSL)
        self._pipeline = device.create_compute_pipeline(
            layout="auto", compute={"module": module, "entry_point": "encode"}
        )

    def encode(self, field_unit: np.ndarray) -> tuple[bytes, int, int]:
        """Encode a [0, 1] field to BC4 blocks on the GPU.  Returns (bytes, h, w)."""

        wgpu, d = self._wgpu, self.device
        field = np.ascontiguousarray(np.clip(field_unit, 0.0, 1.0), dtype=np.float32)
        h, w = int(field.shape[0]), int(field.shape[1])
        nbx, nby = (w + 3) // 4, (h + 3) // 4
        nblocks = nbx * nby
        dims = np.array([(w, h, nbx, nby)], dtype=np.dtype("<u4"))
        args = d.create_buffer_with_data(data=dims.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
        field_buf = d.create_buffer_with_data(
            data=field.tobytes(), usage=wgpu.BufferUsage.STORAGE
        )
        out_size = nblocks * 2 * 4
        out = d.create_buffer(
            size=out_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )
        bind = d.create_bind_group(
            layout=self._pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": args, "offset": 0, "size": 16}},
                {"binding": 1, "resource": {"buffer": field_buf, "offset": 0, "size": field.nbytes}},
                {"binding": 2, "resource": {"buffer": out, "offset": 0, "size": out_size}},
            ],
        )
        enc = d.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self._pipeline)
        cp.set_bind_group(0, bind)
        cp.dispatch_workgroups((nblocks + 63) // 64, 1, 1)
        cp.end()
        d.queue.submit([enc.finish()])
        raw = d.queue.read_buffer(out, size=out_size)
        # words are (nblocks, 2) u32 little-endian == the same bytes bc_codec emits
        return bytes(raw), h, w


def upload_bc4_texture(device, data: bytes, height: int, width: int):
    """Create a resident ``bc4-r-unorm`` texture from BC4 block bytes."""

    import wgpu

    ph = (-height) % 4
    pw = (-width) % 4
    H, W = height + ph, width + pw
    tex = device.create_texture(
        size=(W, H, 1),
        format="bc4-r-unorm",
        usage=(
            wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.COPY_DST
        ),
    )
    device.queue.write_texture(
        {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
        data,
        {"offset": 0, "bytes_per_row": (W // 4) * 8, "rows_per_image": H // 4},
        (W, H, 1),
    )
    device.queue.submit([])
    return tex


_SAMPLE_WGSL = """
@group(0) @binding(0) var tex: texture_2d<f32>;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var out: VOut;
    let xy = p[vi];
    out.pos = vec4<f32>(xy, 0.0, 1.0);
    // Framebuffer row 0 is NDC y=+1 (top); map it to texture row 0 so the
    // read-back image matches the texture's orientation (no vertical flip).
    out.uv = vec2<f32>((xy.x + 1.0) * 0.5, (1.0 - xy.y) * 0.5);
    return out;
}
@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let dims = textureDimensions(tex);
    let coord = vec2<i32>(i32(in.uv.x * f32(dims.x)), i32(in.uv.y * f32(dims.y)));
    let c = clamp(coord, vec2<i32>(0, 0), vec2<i32>(dims) - vec2<i32>(1, 1));
    return vec4<f32>(textureLoad(tex, c, 0).r, 0.0, 0.0, 1.0);
}
"""


def sample_bc4_texture(device, tex, height: int, width: int) -> np.ndarray:
    """Render the BC4 texture through the hardware sampler to r32float, read back.

    Proves the *hardware* decode: returns the [0, 1] field the GPU sampler yields
    for each texel centre, to be compared against the CPU reference decode.
    """

    import wgpu

    ph = (-height) % 4
    pw = (-width) % 4
    H, W = height + ph, width + pw
    target = device.create_texture(
        size=(W, H, 1),
        format="r32float",
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    module = device.create_shader_module(code=_SAMPLE_WGSL)
    pipeline = device.create_render_pipeline(
        layout="auto",
        vertex={"module": module, "entry_point": "vs"},
        fragment={"module": module, "entry_point": "fs", "targets": [{"format": "r32float"}]},
        primitive={"topology": "triangle-list"},
    )
    bind = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": tex.create_view()}],
    )
    enc = device.create_command_encoder()
    rp = enc.begin_render_pass(
        color_attachments=[
            {
                "view": target.create_view(),
                "clear_value": (0.0, 0.0, 0.0, 1.0),
                "load_op": "clear",
                "store_op": "store",
            }
        ]
    )
    rp.set_pipeline(pipeline)
    rp.set_bind_group(0, bind)
    rp.draw(3, 1, 0, 0)
    rp.end()
    device.queue.submit([enc.finish()])
    raw = device.queue.read_texture(
        {"texture": target, "mip_level": 0, "origin": (0, 0, 0)},
        {"offset": 0, "bytes_per_row": W * 4, "rows_per_image": H},
        (W, H, 1),
    )
    img = np.frombuffer(raw, dtype=np.float32).reshape(H, W)
    return img[:height, :width]
