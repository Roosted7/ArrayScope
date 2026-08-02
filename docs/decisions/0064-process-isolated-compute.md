# 0064 — Process-isolated compute and the frame-sovereign GUI process

Date: 2026-08-02. Status: accepted, staged delivery.

## Context

The UI stutters under load even though the GUI thread "only" coordinates. Three
mapping passes over the current code established why, with measurements:

1. **The wgpu draw path is already decoupled.** Tile instances carry camera-free
   world rects; a pan/zoom is a 32-byte `TileCamera` uniform write plus one
   render pass (`gpu/wgpu_executor.py` `_write_camera`, `_set_tiles` identity
   short-circuit). Free-pan measures ~58 draws/s at ~1 ms frame cost
   (`docs/redesign/wgpu-present-decoupling-2026-07-31.md`). The render path is
   not the problem.
2. **The problem is GUI-thread occupancy between frames.** A Qt callback is
   atomic, so any callback longer than a refresh period drops
   `ceil(ms / period)` frames deterministically. The measured occupants:
   - a whole-plan viewport retarget running **synchronously inside wheel
     dispatch** (up to 542 ms, p95 473 ms) because pyqtgraph's constraint clamp
     re-emits `sigRangeChanged` after the one-shot wheel flag was consumed, so
     the nested emit is classified "programmatic" and takes the synchronous
     branch (`window/viewport_bridge.py`);
   - GUI-thread page assembly and upload packing;
   - the atomic tiled commit;
   - GIL/scheduler contention: the cold-entry stage plan spent ~1.4 s of
     off-CPU wall blocked against the fill's own worker threads.
3. **Worker threads share one GIL, one GC, one crash domain** with the GUI.
   Pure-Python stages (the per-tile fan-out loop: 78–94 % of a shared
   preview/target stage, single-threaded Python) contend directly with input
   dispatch, and worker GC pauses land on GUI callbacks.

### Position on free threading

`app/free_threading.py` already relaunches with `PYTHON_GIL=0` on 3.13t/3.14t
builds. Processes do not replace that bet — they deliver what free threading
cannot: **GC isolation** (a worker's collection cannot pause the GUI), **crash
containment** (a plugin op cannot take down the viewer), **hard memory
ceilings**, and **per-worker native-thread-team control**. Both compose: a
free-threaded GUI process plus isolated compute processes is the end state.

## Decision

One **GUI/render process** that owns Qt, the wgpu device, all presentation
state, and only bounded coordination work — plus a **persistent pool of
spawn-context compute processes** fed by shared memory and tiny descriptors.
Alongside, the GUI-thread occupancy defects above are fixed directly, because
moving compute out of process does not shorten a 542 ms input-dispatch
callback.

### Execution seam

`kernel/workers.py` declares the `WorkerBackend` protocol as the intended seam
("a free-threaded or process backend must not require kernel changes"). We keep
that promise:

- **`RemoteCall(entry, request)`** — a frozen descriptor a submit site may use
  as `TaskSpec.fn`'s body: `entry` is a `"module:function"` string resolvable
  in a fresh interpreter, `request` is picklable data. A `RemoteCall` is also
  callable in-process (resolve + invoke), so every task keeps a same-process
  fallback and `InlineWorkerBackend` tests stay deterministic.
- **`ProcessWorkerBackend`** — implements the same three-method protocol. It
  routes tasks whose `fn` is a `RemoteCall` to the process pool and executes
  everything else (closures, `fn=lambda: True` continuation primitives) on its
  internal threads, exactly as `ThreadWorkerBackend` does today. The kernel
  scheduler, lanes, priorities, supersession, and the Qt bridge are untouched.
- Migration is therefore **per submit site**: a family moves by re-expressing
  its closure as `(entry, request)`; unconverted families keep working.

### Worker processes

- `multiprocessing.get_context("spawn")`, persistent for the viewer's life;
  `max(1, physical_cores − 2)` workers, one shared pool per process (not per
  window).
- Child env pins native thread teams: `OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `NUMBA_NUM_THREADS=1`. Today **no such handling exists anywhere** while
  numba kernels are `parallel=True` — N processes × uncapped teams is an
  oversubscription bomb. Raising a cap later is a measured decision.
- Crash containment: a dead worker fails its in-flight task with a real error,
  the pool respawns the worker, the viewer keeps running.
- Cancellation: cooperative, as today, but the token's flag is one byte in a
  shared control region; the parent sets it, the child's
  `CancellationToken.cancelled` reads it. `EvaluationCancelled` and
  traceback text travel back as data.
- Rollout: behind the `compute_process_pool` setting (`auto|on|off`),
  **off by default** until the pilot families prove the transport. Both
  default owners of the setting (settings-state and any profile default) are
  set together, per the two-owner settings trap.

### Data transport — descriptors, never pickled arrays

- **Sources.** File-backed sources (`.npy`, `.cfl`, uncompressed `.nii`) are
  already `np.memmap`: a worker re-opens the same path and the OS page cache
  shares physical memory — zero new machinery. The descriptor is a small tree
  (`scale` and `composite` nodes over memmap/shm leaves), carrying
  `(path|shm_name, shape, dtype, order, offset, slope, inter, out_dtype)`.
  Eager in-memory arrays get one `SharedMemory` segment written at open.
  Workers ship `(source descriptor, RegionSpec)` and call `read_base_region`
  themselves — `RegionSpec` is already fully picklable.
- **Results.** A bounded **slot pool** of shared memory: uniform 1 MiB slots
  (covers scalar/complex/rgba pages and whole tile planes at ≤2× waste; the
  page formats are 256 KiB / 512 KiB / 1 MiB with no row-pitch padding
  needed), total 512 MiB — inside the existing retention envelope. Slot
  lifecycle is a strict one-writer state machine:
  `FREE → LEASED(worker) → READY → HELD(GUI) → FREE`. Publication is by
  descriptor over the completion pipe; a slot is immutable from READY until
  the GUI releases it. Backpressure is structural: no free slot ⇒ the family's
  work waits, bounded memory by construction.
- **Small results** (`TileLevelStats` ≤32 KB, `ChunkHistogramSummary` 772 B,
  bounds, envelopes) go over the pipe directly; slots are for page/plane
  payloads only.
- The existing `PreparedUploadMailbox` keeps its identity keying and 64 MiB
  byte bound; a slot-backed buffer must pass `nbytes` explicitly (the bound
  reads `getattr(buffer, "nbytes")`).

### Prerequisites (small, named)

- Replace `id(document.base_data)` as document key
  (`operations/evaluator.py` `_document_key` / `stage_document_key`,
  `operations/slabs.py`) with a stable token — process-local ids cannot cross
  the boundary.
- `reusable=True` semantics change for migrated families: "finish into the
  cache" becomes "return the value; the parent admits it". Cache ownership
  (LodPageCache, StageCache) stays in the GUI process.

### Migration order (value ÷ risk, from the mapping)

| # | Family | Why this order |
|---|---|---|
| 1 | `compute_histogram_plot`, `aggregate_histogram_samples` | Already pure module functions over picklable data; proves backend + cancellation on work no frame depends on |
| 2 | Per-tile fan-out of shared preview/target evaluation | The single largest pure-Python block (78–94 % of the stage); one immutable input array (perfect slab), one page out per tile |

> **Correction, 2026-08-02.** Row 2's measurement priced
> `evaluate_shared_preview`'s 272-iteration worker loop — which `819789f8`
> deleted the day before this ADR was written; ADR 0059 had already turned it
> into one kernel task per tile. The surviving per-tile body (the tail of
> `_evaluate_tile_reduced_input_preview`) is what migrated: it is now the
> module-level entry `render.effects:fan_out_preview_pages` behind a
> `RemoteCall`, batch size one because supersession, lifecycle claims and
> R2/R3 admission are already per tile. The 78–94 % figure does not transfer
> to the successor; the per-tile cost is ~1.9–2.5 ms, and the win to measure
> is contention relief, not the retired loop's wall time.
| 3 | `materialize_lod_page` | Pure `(array, origin, plan) → page`; parent admits into `LodPageCache`; ship the 772 B histogram summary alongside so the parent does not redo it |
| 4 | Upload packing (`wgpu_packed_payload_texture`) into slot-backed mailbox buffers | Documented pure; output feeds `queue.write_texture` directly from the slot |
| 5 | Whole-array materialize/export | Rare, huge, user-initiated; subprocess precedent exists |
| 6 | Tile/stage evaluation | The real work; gated on request-snapshot types and the document-key prerequisite |

Stays in-process indefinitely: continuation no-ops, `resolve_wgpu_histogram_evidence`
(GPU fence — device-bound), ROI inspection (bound method of the window, needs a
rewrite first), FFT internals (already GIL-releasing with its own worker fan-out).

### The frame-sovereign GUI process

Process isolation alone does not produce 60–144 Hz; these land in the same
program:

1. **Gesture-cause classification** — a range change is classified by the
   gesture that caused it, held for the whole event dispatch, so pyqtgraph's
   constraint re-emit inherits the wheel classification and routes through the
   existing 16 ms coalescer instead of the synchronous whole-plan retarget.
   Straight deferral of the programmatic branch is already refuted
   (`docs/redesign/programmatic-viewport-replay-2026-07-31.md`); this changes
   *classification*, not *when programmatic replays run*.
2. **One `setRange` per gesture step** — the constraint clamp folds into the
   navigation driver, eliminating the nested `sigRangeChanged` entirely.
3. **R6 load shedding** (progressive-render contract): when production falls
   behind, existing tiles move with the camera and newly exposed area draws
   black/placeholder; the montage never freezes until idle.

## Consequences

- Worker GC, allocation storms, and plugin crashes stop reaching the GUI
  process at all; pan/zoom cost is bounded by the (already fast) draw path
  plus whatever GUI-thread work remains — which the classification and
  commit-slicing work bounds independently.
- Two schedulers' worth of complexity is avoided: the kernel stays the only
  scheduler; the pool is an executor detail behind `WorkerBackend`.
- Shared-memory lifetime is a real failure mode: the GUI process owns every
  segment and its unlink; workers only `close()`. Shutdown, killed-worker,
  and stale-segment cleanup are part of stage 1's test surface, not a
  follow-up.
- Zero-copy is one-sided by design: worker → GUI stays zero-copy into
  `write_texture`; there is still exactly one CPU→GPU transfer, and camera
  motion requires none.
