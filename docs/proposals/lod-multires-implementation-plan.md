# LOD Multi-Resolution Implementation Plan

## Status

Blocked behind the native-only production policy. Demand selection is implemented and diagnostics
report desired quality, but production presentation applies native-resolution payloads only.

## Problem

The removed prototype built CPU pyramids from loaded tiles inside the presentation path, mixed reduced
tile dimensions into fixed atlas assumptions, and let display approximation approach semantic
histogram/value ownership. A safe implementation needs separate demand selection, asynchronous
materialization, and compatible residency as required by ADR 0041.

## Ownership

- `arrayscope.display.lod` owns Qt-free demand and policy objects.
- A future materializer owns reduced payload creation or source-provided pyramid reads outside Qt/GL
  callbacks.
- Backend residency owns storage classes, allocation, eviction, and transition acknowledgement.
- Committed frames, hover, ROI, profiles, and histograms keep exact semantic value sources unless a
  future approximation is explicitly labeled.

## Cache Keys And Storage Classes

Materialized LOD cache keys must include:

- semantic source identity and source revision;
- tile or region identity;
- component/scale representation used for reduction;
- LOD level and reduction algorithm version;
- source tile shape and resulting texture shape;
- texture format and channel layout;
- gutter policy and gutter width.

Residency keys must additionally include backend context and storage class. Compatible storage classes
are separate atlas pages or pools by `(level, tile shape, format, gutter)`, texture arrays grouped by
identical dimensions and format, or a virtual texture/page table. Arbitrary reduced dimensions must not
share fixed slots whose sampling assumes one native tile shape.

## Transition Behavior

- Keep the currently presented native tile visible while a requested reduced level materializes and
  becomes resident.
- Retain one adjacent level only when memory pressure allows; otherwise demote to native rather than
  clearing useful pixels.
- Repeated zoom threshold crossings reuse retained/materialized levels and must not rebuild the active
  set synchronously.
- A level transition changes physical residency identity, not semantic source identity.
- Exact hover, ROI/profile, histogram, export, and inspection values come from exact semantic sources,
  not approximate display textures.

## Benchmark Matrix

Before enabling non-native LOD, collect deterministic counters plus real-display traces for PyQtGraph
and VisPy:

| Scenario | Required evidence |
|---|---|
| Native baseline | request-to-first-frame, exact-visible time, upload bytes, resident bytes |
| Repeated threshold crossings | no synchronous rebuild storm; retained/native frame remains visible |
| Pan/zoom reuse | materialized/resident levels reused without redundant source evaluation |
| One dirty tile | only the dirty semantic tile rematerializes/reuploads at the selected level |
| Level-only update | levels/LUT do not change materialized LOD or upload source pixels |
| Source-provided pyramid | source level read wins over CPU reduction when available |
| Async materialization | worker requests are cancellable, singleflight, and cache bounded |
| Compatible residency | no mixed-shape fixed-slot atlas use; allocation failures recover cleanly |
| Context loss/failure | native exact frame or placeholder recovery is explicit and semantically safe |

Timing claims require the manual-regression hardware record: OS/session type, GPU/driver, backend,
dataset shape/dtype, operation stack, JSONL path, and diagnostics trace.

## Why This Is Not An ADR

ADR 0041 already records the durable architecture decision. This proposal is the implementation and
evidence checklist that must be satisfied before the roadmap can enable non-native LOD.
