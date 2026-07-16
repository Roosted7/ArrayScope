# Reviews

Reviews are dated assessments and trace interpretations. They may identify risks or recommend direction, but the current source of truth remains tested code, accepted ADRs, and the live architecture/roadmap.

- [T1/V1 adversarial review](2026-07-14-t1-v1-adversarial-review.md): hardening of the import guard, workflow harness, and trace_verify; the idle re-commit/re-ack livelock finding (2026-07-14, addenda through P8/P9).
- [UI visual audit](2026-07-10-ui-visual-audit.md): dated visual findings (2026-07-10).
- [LOD resident A/B](lod-resident-ab-2026-07-04.md): resident-policy A/B measurements (2026-07-04).
- [X5a hardware telemetry](x5a-hardware-telemetry-linux-wayland.md): Linux/Wayland real-hardware baseline evidence (2026-07-03).
- [v33 optimization-roadmap review](v33-optimization-roadmap-review.md): optimization roadmap assessment.
- [v32 composition audit](v32-composition-audit.md): full-project ownership/structure review, measured coupling, crash root causes, dead-path removal, and the Y1-Y3 gates (2026-07-02).
- [v31 rendering and roadmap audit](v31-rendering-roadmap-audit.md): rendering architecture, residency, viewport, backend policy, and roadmap/ADR review (2026-06-27). Superseded by v32 where they overlap.
- [v30 rendering-consistency audit](v30-rendering-consistency-audit.md): histogram/level convergence, LOD, rendering control-plane, roadmap, and ADR review (2026-06-24).
- [v28 project audit](v28-project-audit.md): holistic code, performance, documentation, and ArrayShow/ArrayView review (2026-06-22).
- [v28 supplemental audit](project-audit-v28.md): alternate audit report restored from the v28 notes for completeness.
- [v27 rendering review](rendering-v27-review.md): detailed diagnosis preceding the tiled-montage repair.
- [v27 trace summary](rendering-v27-trace-summary.md): interpretation of rendering traces used during that repair.
- [v24 VisPy rendering review](vispy-v24-rendering-review.md): early backend experiment findings.

Create a new review for a dated evidence set. Do not append current requirements indefinitely to an old review; convert durable choices to ADRs and active work to the roadmap.
