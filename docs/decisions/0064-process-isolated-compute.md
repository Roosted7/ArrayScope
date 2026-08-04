# 0064 — Process-isolated compute and the frame-sovereign GUI process

Date: 2026-08-02. Status: accepted, staged delivery.

> **Measured, 2026-08-04 — turning the pool on costs ~12 % of the frame rate.
> The shipped default already refuses it, and nothing here changes that.**
>
> The default is `ComputeProcessPoolChoice.AUTO`, which resolves to *off*
> (`app/settings_state.py:compute_process_pool_enabled`, locked by
> `tests/app/test_theme_settings.py`). **There was no default to change.** Only
> an explicit `compute_process_pool=on` builds a pool — and the operator's
> diagnostic recordings were all made with that pin, so every other improvement
> measured in those sessions was measured through this handicap. That, not the
> default, is the finding to act on.
>
> Four order-balanced sweeps of the real 272-tile montage workflow (wgpu
> screen, `tools/profile_montage_workflow`), 78 runs, arms interleaved in
> blocks of three so every arm carries the same mean position in the sequence.
> Adjudicated on `presentation_draw_count` per second.
>
> **The decisive sweep** — 21 runs, 7 per arm, on an otherwise quiet desktop
> (load 0.8–3.9, no competing batch job):
>
> | arm | n | draws/s | range | vs off |
> |---|---|---|---|---|
> | pool OFF | 7 | **8.990** | 8.514 – 9.203 | — |
> | pool ON (14 procs) | 7 | **7.913** | 7.558 – 8.442 | **−12.0 %** |
> | pool ON, 4 procs | 7 | **8.143** | 7.849 – 8.305 | −9.4 % |
>
> The pool-OFF and pool-ON distributions are **disjoint** — every pool-OFF run
> beats every pool-ON run — giving the smallest p an exact rank test can return
> at this n (**p = 0.0003**, one-sided Mann-Whitney).
>
> **The control says this is real.** Two earlier sweeps carried an explicit
> same-config control arm (`ctrl` running exactly the `off` configuration); it
> never separated from `off` (p = 0.37 and p = 0.52) while pool-ON did
> (p = 0.0020, p = 0.0001). In the decisive sweep the same question is asked by
> splitting the 7-run `off` arm in half: the two halves of one identical
> configuration differ by **2.13 %**, against a **12.0 %** effect — 5.6× the
> harness's own noise, and the arm's spread is only 2.6 % CV.
>
> **It is not merely pool start-up.** Wall clock rises ~2.3 s with the pool, but
> the long interactive scroll phase, measured against its *own* elapsed time so
> start-up is excluded entirely, is **−13.7 %** (14.93 → 12.89 draws/s,
> p = 0.0003). There is a deterministic setup cost *and* a steady-state cost.
>
> **It is mostly transport, not oversubscription.** Pool-ON runs
> `default_worker_count()` + `default_process_count()` = 28 compute contexts on
> 16 cores, so oversubscription was the competing explanation. Dropping to 4
> processes while keeping the transport (~1250 routed calls either way) recovers
> only **21 % of the whole-run gap and 31 % of the scroll-phase gap**; ~70–80 %
> of the loss survives. That is the cost `d3f5d1b3` priced directly: ~1290
> routed `RemoteCall` bodies at ~1.29 ms of parent CPU each (~0.93 ms
> GIL-holding) is ~1.6 s of added parent CPU over a ~26 s session, contending
> with the paint loop.
>
> Correcting the record on both predecessors: `d3f5d1b3`'s headline **6.33 vs
> 8.17 draws/s (−22 %) overstates the effect** — it came from a single
> back-to-back pair — and its own caveat, that its 9-slot sweep could not
> adjudicate this, was right and is now discharged. `f758e126`'s incidental
> control (8.32/8.20 ON vs 9.01 OFF, **−8.4 %**) was closer to the truth than
> the measurement it was a footnote to.
>
> **Nothing is retired and nothing is deleted.** `RemoteCall`, the four
> converted families, `ProcessWorkerBackend`, the shm publication path and the
> slot pool all still exist and still work; a `RemoteCall` with no pool still
> runs in-thread through its own `__call__`. What is measured is only that
> *enabling the pool today costs frames* — which is exactly why AUTO resolves to
> off. The strategy is not refused: see the 2026-08-04 graveyard rows for the
> shape of family that would pay, namely one whose routed body is much larger
> than the ~1.3 ms the transport costs to carry it.

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
