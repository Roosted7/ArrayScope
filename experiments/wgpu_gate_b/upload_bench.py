"""Tier 4: upload-path microbenchmarks + completion contract (gate 3).

Paths measured on the UMA iGPU, all uploading RG32F 256^2 pages (512 KiB):

  wt_page      queue.write_texture, one page at a time
  wt_batch16   queue.write_texture x16 in one burst (one plane's worth)
  wt_plane     queue.write_texture of a whole 1024^2 plane into a big texture
  ring         persistent MAP_WRITE staging buffer: map_sync -> write_mapped
               -> unmap -> copy_buffer_to_texture x16 -> submit
  mappable     (feature mappable-primary-buffers) MAP_WRITE|STORAGE buffer
               written directly — the zero-copy UMA page-pool candidate

Completion contract: on_submitted_work_done_sync round-trip after a 16-page
burst — the token ArrayScope needs for staging-slot/page-slot recycling
(the thing Datoviz gate A could not provide).

Each measurement: N reps, wall ms; all submissions fenced by
on_submitted_work_done_sync so we time COMPLETED uploads, not queued ones.

Run: python upload_bench.py out.json
"""

import json
import statistics
import sys
import time

import numpy as np
import wgpu
from wgpu.backends.wgpu_native.extras import set_instance_extras

set_instance_extras(backends=["Vulkan"])

PAGE = 256
PAGE_BYTES = PAGE * PAGE * 8
REPS = 30

EV = {"harness": "wgpu-gate-b-tier4", "paths": {}, "completion": {}}


def stats(ms):
    s = sorted(ms)
    return {
        "n": len(s),
        "mean_ms": round(statistics.fmean(s), 3),
        "p50_ms": round(s[len(s) // 2], 3),
        "p95_ms": round(s[min(len(s) - 1, int(len(s) * 0.95))], 3),
    }


def gbps(nbytes, mean_ms):
    return round(nbytes / (mean_ms / 1e3) / 1e9, 2)


def main():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="low-power")
    features = []
    if "mappable-primary-buffers" in adapter.features:
        features.append("mappable-primary-buffers")
    device = adapter.request_device_sync(required_features=features)
    q = device.queue
    EV["adapter"] = adapter.info["device"]
    EV["features_requested"] = features

    rng = np.random.default_rng(7)
    page = rng.standard_normal((PAGE, PAGE, 2), dtype=np.float32)
    pages16 = [np.ascontiguousarray(page + i) for i in range(16)]
    plane = rng.standard_normal((1024, 1024, 2), dtype=np.float32)

    pool = device.create_texture(
        size=(PAGE, PAGE, 32),
        format="rg32float",
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    big = device.create_texture(
        size=(1024, 1024, 1),
        format="rg32float",
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )

    def fence():
        q.on_submitted_work_done_sync()

    # --- wt_page: one page per call, fenced per call (worst case).
    times = []
    for i in range(REPS):
        t0 = time.perf_counter()
        q.write_texture(
            {"texture": pool, "origin": (0, 0, i % 32)},
            pages16[i % 16],
            {"bytes_per_row": PAGE * 8, "rows_per_image": PAGE},
            (PAGE, PAGE, 1),
        )
        fence()
        times.append((time.perf_counter() - t0) * 1000)
    st = stats(times)
    st["gb_per_s"] = gbps(PAGE_BYTES, st["mean_ms"])
    EV["paths"]["wt_page_fenced"] = st

    # --- wt_batch16: 16 pages queued, one fence (the scroll-burst shape).
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        for i, p in enumerate(pages16):
            q.write_texture(
                {"texture": pool, "origin": (0, 0, i)},
                p,
                {"bytes_per_row": PAGE * 8, "rows_per_image": PAGE},
                (PAGE, PAGE, 1),
            )
        fence()
        times.append((time.perf_counter() - t0) * 1000)
    st = stats(times)
    st["gb_per_s"] = gbps(16 * PAGE_BYTES, st["mean_ms"])
    EV["paths"]["wt_batch16_fenced"] = st

    # --- wt_plane: one whole-plane write (same bytes as batch16).
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        q.write_texture(
            {"texture": big, "origin": (0, 0, 0)},
            plane,
            {"bytes_per_row": 1024 * 8, "rows_per_image": 1024},
            (1024, 1024, 1),
        )
        fence()
        times.append((time.perf_counter() - t0) * 1000)
    st = stats(times)
    st["gb_per_s"] = gbps(plane.nbytes, st["mean_ms"])
    EV["paths"]["wt_plane_fenced"] = st

    # --- ring: persistent staging buffer reused every rep (map_sync wait
    #     is the recycle cost the completion token would hide).
    staging = device.create_buffer(
        size=16 * PAGE_BYTES,
        usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
    )
    blob16 = np.concatenate([p.reshape(-1) for p in pages16]).tobytes()
    times, map_times = [], []
    for _ in range(REPS):
        t0 = time.perf_counter()
        staging.map_sync(wgpu.MapMode.WRITE)
        map_times.append((time.perf_counter() - t0) * 1000)
        staging.write_mapped(blob16)
        staging.unmap()
        enc = device.create_command_encoder()
        for i in range(16):
            enc.copy_buffer_to_texture(
                {
                    "buffer": staging,
                    "offset": i * PAGE_BYTES,
                    "bytes_per_row": PAGE * 8,
                    "rows_per_image": PAGE,
                },
                {"texture": pool, "origin": (0, 0, i)},
                (PAGE, PAGE, 1),
            )
        q.submit([enc.finish()])
        fence()
        times.append((time.perf_counter() - t0) * 1000)
    st = stats(times)
    st["gb_per_s"] = gbps(16 * PAGE_BYTES, st["mean_ms"])
    st["map_wait"] = stats(map_times)
    EV["paths"]["ring16_fenced"] = st

    # --- mappable-primary: direct write into a shader-usable STORAGE buffer.
    if features:
        prim = device.create_buffer(
            size=16 * PAGE_BYTES,
            usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.STORAGE,
        )
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            prim.map_sync(wgpu.MapMode.WRITE)
            prim.write_mapped(blob16)
            prim.unmap()
            times.append((time.perf_counter() - t0) * 1000)
        st = stats(times)
        st["gb_per_s"] = gbps(16 * PAGE_BYTES, st["mean_ms"])
        EV["paths"]["mappable_primary_write16"] = st

    # --- pure CPU memcpy baseline for the same bytes.
    dst = bytearray(16 * PAGE_BYTES)
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        dst[:] = blob16
        times.append((time.perf_counter() - t0) * 1000)
    st = stats(times)
    st["gb_per_s"] = gbps(16 * PAGE_BYTES, st["mean_ms"])
    EV["paths"]["cpu_memcpy16_baseline"] = st

    # --- completion contract: token round-trip after a 16-page burst.
    lat = []
    for _ in range(REPS):
        for i, p in enumerate(pages16):
            q.write_texture(
                {"texture": pool, "origin": (0, 0, i)},
                p,
                {"bytes_per_row": PAGE * 8, "rows_per_image": PAGE},
                (PAGE, PAGE, 1),
            )
        t0 = time.perf_counter()
        q.on_submitted_work_done_sync()
        lat.append((time.perf_counter() - t0) * 1000)
    EV["completion"] = {
        "api": "queue.on_submitted_work_done_sync (async variant also present)",
        "token_roundtrip_after_burst16": stats(lat),
        "ok": True,
    }

    print(json.dumps(EV, indent=2))
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            json.dump(EV, f, indent=2)


if __name__ == "__main__":
    main()
