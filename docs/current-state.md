# Current state

**Snapshot:** ArrayScope `0.8.0` release-candidate / v28 baseline, reviewed on 2026-06-22. The audit fixes described below are included in this RC baseline.

ArrayScope has outgrown the original “small PyQtGraph image viewer” architecture and is in a deliberate transition. The project already has many of the right semantic boundaries and safety mechanisms; the remaining risk is that normal-image, montage, PyQtGraph, and VisPy paths still compose those mechanisms differently.

## Maturity map

| Area | State | Notes |
|---|---|---|
| Basic launch, slicing, image/line display | Established | Broad automated coverage; still validate platform/Qt integration. |
| Dimension roles, range parsing, axis flips/FFT shift | Established with recent change | Recent range/cropped-axis work deserves focused interaction regression. |
| Reversible operation document and recipes | Established | Optimizer preserves public step history. |
| Region planning, stage cache, cost/memory estimates | Substantial | Strong pure-core coverage; workload heuristics still need field evidence. |
| Profiles and ROI inspection | Substantial | Shared semantics exist; full pointer/drag ownership is not yet backend-neutral. |
| Histogram and window/level | Substantial, recently expanded | Adaptive bins/manual editing work is new and still performs some bounded NumPy work on the GUI thread. |
| Progressive montage | Advanced but transitional | Correct lifecycle distinctions and bounded caches exist; orchestration remains large and timer-heavy. |
| PyQtGraph backend | Default | Feature-complete baseline; many per-tile `ImageItem`s can become a CPU/scene bottleneck. |
| VisPy backend | Experimental | First-class tiled atlas/shader path exists, but the widget still subclasses the complete PyQtGraph view. |
| Diagnostics/benchmarks | Good internal foundation | Counters and JSONL traces are useful; real GPU/Wayland baselines are incomplete. |
| Packaging/release story | RC-ready | Package/runtime version identity is aligned for `0.8.0`; CI and RC artifacts still need publication evidence before release. |
| Documentation | Reorganized in this audit | Live guidance is now separated from archived phase notes. |

## What is working well

### Semantic state is mostly outside widgets

`ViewState`, `ArrayDocument`, operation declarations, geometry, frame/presentation models, and memory policy are largely Qt-free. This makes correctness testable and limits backend-specific meaning.

### Expensive work has explicit models

The project contains operation capability declarations, region plans, cost estimates, stage materialization/singleflight, separate caches, render decisions, cancellation tokens, lane worker policy, latency feedback, and a resource governor. This is a much stronger foundation than ad hoc “put it on a thread” code.

### Montage repair established important invariants

Recent work separates requested, materialized, resident, and presented tiles; rejects stale deltas; preserves retained residency; and distinguishes cold upload from warm rebind/visibility work. Automatic levels are tied to semantic coverage rather than merely to whether a histogram widget finished drawing.

### Tests protect architecture as well as values

The suite includes pure shape/value tests, property tests, UI interaction tests, architecture guards, memory stress, deterministic rendering counters, and smoke artifacts. Several tests intentionally prevent new renderer type-switches or widget-owned semantics.

## Current transition

The target architecture in [ADR 0039](decisions/0039-unified-image-surface-and-deadline-scheduler.md) is only partly implemented.

Implemented pieces include semantic display frames/presentations, backend capabilities/adapters, typed tile payloads, persistent VisPy residency, memory/resource policy, and montage sessions. Remaining gaps include:

- one frame planner for both normal image and montage;
- active-plus-latest progress-preserving visible scheduling;
- one deadline/admission model across visible, analysis, commit, and speculative lanes;
- storage-neutral tiled geometry for very large single planes;
- complete shared pointer capture and drag lifecycle;
- composition of an `ImageViewShell` with a thin `ImageSurface`, replacing `VisPyImageView2D(ImageView2D)` inheritance.

## Material risks

### 1. GUI callback budgets are not globally enforceable yet

Several callbacks still loop over a whole ready/waiting tile set, rebuild priorities, update many scene objects, or calculate histogram data in one event-loop turn. Item limits exist in some paths, but a universal item/byte/time contract does not.

### 2. Normal-image and montage control flow can diverge

They use different orchestration modules, generation/cancellation behavior, and timer/coalescing paths. A correctness fix in one path does not automatically protect the other.

### 3. Tile priority is not continuously retargeted

Recent priority work orders new plans around viewport center or the last hover point. Mouse movement itself only records a point; it does not safely reprioritize an already-active queue. Sorting on every mouse event would be worse, so this needs an indexed/coalesced design rather than another callback.

### 4. Renderer files remain too large

`montage_renderer.py`, `imageview2d.py`, `vispy_imageview2d.py`, and the VisPy tiled backend each exceed roughly 1,700–2,000 lines. Size is a symptom: orchestration, lifecycle, interaction bridging, diagnostics, and concrete scene mechanics are still interleaved.

### 5. Timer and generation interactions are hard to reason about

Debounce, quiet-period refresh, upload timers, warm-residency timers, prefetch timers, and slow-overlay timers can interact. Timers should be admission/rescheduling mechanisms, not the source of semantic ordering.

### 6. Hardware evidence is incomplete

Headless tests can verify contracts and deterministic work counters. They cannot prove GPU upload latency, texture limits, Wayland behavior, pointer feel, frame pacing, or real memory pressure.

### 7. Rapid local development increases integration risk

Recent work changed histogram, viewport, slicing, and tile priority behavior in quick succession. Before publication, keep the RC provenance, CI status, versioning, diagnostics trace, and benchmark baselines together.

## Audit fixes

The restored v28 fixes include:

- centered-slice and priority-aware montage test repairs, order-independent Qt settings cleanup, and empty Inspection-dock no-op refreshes;
- bounded evaluation-group invalidation with pruned completed per-tile generation bookkeeping;
- the `format_bytes` render-refusal import and single-action auto-window behavior when a manual histogram preview is open;
- consolidated CI and release tag/package/runtime version guards;
- canonical `0.8.0` package/runtime version identity and deterministic RC diagnostics artifacts;
- concurrent multi-path CLI launches while preserving single-path blocking;
- display resource shutdown and bounded benchmark lifecycle cleanup;
- viewport minimum-overlap preservation after max-span clamping;
- small hardening for montage priority inputs and early hover cleanup.

The active priorities and acceptance gates are in the [roadmap](roadmap.md). The full evidence and comparison are in the [v28 audit](reviews/v28-project-audit.md), with supplemental status detail in [project-status.md](project-status.md).
