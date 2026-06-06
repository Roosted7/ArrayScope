# 0012 — Lazy slab evaluation

ArrayScope display evaluation is slab-first. Image, profile, pixel-hover, and export-frame requests
ask for the smallest exact derived output region needed, and `arrayscope.operations.slabs` maps that
request back through the operation stack to a base-data slab.

This keeps ordinary inspection responsive on large arrays. Crop/reverse-style operations can remain
views. Reductions expand only the reduced axis for the requested output slab. FFT requests compute
only the requested slab when the FFT axis is visible, and expand the FFT axis only when a downstream
scalar index on that axis is requested.

`OperationEvaluator.current_data()` remains available, but it is an explicit full-materialization
path for materialize/save/export workflows. The main window tracks derived shape and dtype metadata
instead of storing a materialized derived array as normal state.

Display/profile/pixel requests are cached with bounded LRU caches. Defaults are fixed constants:
256 MiB for image/export-frame results and 64 MiB for profile/scalar results. Cache diagnostics
include entries, memory, hit/miss counts, evictions, and latest evaluation time.

Qt-visible evaluation runs through `arrayscope.window.evaluation_controller`. The controller uses a
monotonic generation counter rather than attempting to interrupt NumPy work. Workers evaluate against
captured immutable `ArrayDocument` and `ViewState` snapshots. They do not mutate the live
`OperationEvaluator`; the UI thread commits cache/status updates only after confirming the request is
still current. Late worker results are ignored when newer user intent exists. During slow image
evaluation the previous image remains visible and is marked stale with reduced opacity plus an
overlay.

Cache diagnostics are intentionally split by meaning: current view cache, profile/pixel cache, full
derived estimate, and last request slab details. Operation rows show structural shape/dtype estimates,
not per-operation cached progress, because slab evaluation does not permanently compute operation
rows left-to-right.

Operations without a slab rule may fall back to full materialization, but all currently registered
operations have exact slab behavior.
