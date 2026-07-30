# Crop-rebind R3 closure and draw liveness — 2026-07-30

Evidence for the R3 seam at a resident crop rebind. The retained-LOD handoff that
motivated the axis-slice replays is a separate change on its own branch; this
document owns only the R3 and draw-liveness conclusions.

## R3 closure and draw liveness

The crop-rebind R3 queue row closes with two explicit outcomes at the binding
seam. A page-backed payload that carries the complete native residency plane
derives its bounds through the payload's shader component and scale, then
widens the in-force WGPU levels atomically to that proven full-plane superset.
A payload without that exhaustive plane declines the rebind and follows normal
evaluation; predecessor-window statistics, sparse semantic samples, and
predecessor page descriptors still prove nothing about the current window.
The profiler independently maps the complete native plane and never reads the
clamp's `rebind_current_value_bounds` claim.

The intermittent final one-draw debt was a profiler deadline seam, not an
application liveness defect. At a captured failure the final tile-commit request
was only 30.7 ms old, rendercanvas still had `draw_requested=True`, and its
on-demand cadence is 33.3 ms. The phase-wide action deadline therefore expired a
few milliseconds before an already-scheduled frame. Retry callbacks and an
alternate canvas clock were rejected; a source-driven immediate-draw prototype
closed the gate but inflated physical draws from 65–72 to 137–184 and moved the
same stages from about 9.2 s to 10.0–10.5 s. The profiler now keeps the action
duration as R5/performance evidence, but once every semantic/lifecycle gate is
green it drains the existing request under the established 0.5 s physical-draw
target. It neither requests nor synthesizes a frame.

Across three final managed-Weston 336x336x272 geometry replays, both
`display_x_axis_slice` and `display_y_axis_slice` completed every time with
matched request/draw counts (x: 188/188, 194/194, 187/187; y: 321/321, 339/339,
326/326). R3 was absent on all six stage rows; the remaining R1/R2/R5 rows are
the inherited contract reds. Physical draws stayed at 74–76 for x and 65–67 for
y, preserving the accepted coalescing regime. The direct rebind counter measured
the 50-tile value-bound scan at 4.84–5.33 ms average for x (8.40–10.11 MiB
touched) and 4.30–4.83 ms for y (7.58–7.97 MiB), with a 13.14 ms maximum. This
is below both the 102–143 ms scrub-step cost and the approximately 0.5 s
end-to-end variance floor. The preceding cProfile ranking measured the old
compacting extrema scan at 0.098 s / 2391 calls (2.05 ms per 50 scans) and the
shared mapped scan at 0.154 s / 3440 calls (2.23 ms per 50 scans); no end-to-end
speed claim is made.

> Provenance: these replays ran with the retained-LOD handoff present in the
> same tree. Re-confirm the request/draw and scan figures once the two changes
> are integrated, since that feature is what drives the 272-tile rebind
> population these numbers were taken over.
