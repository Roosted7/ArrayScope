# 0023 - Memory Policy and Developer Diagnostics

## Problem

ArrayScope had too many static and hidden memory limits. The render memory setting covered only part
of runtime behavior, montage tiles shared the normal image cache, cache budgets were not visible, and
developer debugging required reading internal state or print output.

## Decision

Use one psutil-backed `MemoryPolicy` to derive runtime budgets from a memory profile, sampled system
memory, input size, and the existing per-render hard cap. The profiles are conservative, balanced,
aggressive, and custom. Keep `render_memory_budget_mb` as the per-render hard cap for visible image
and montage tile/canvas allocations.

Split evaluator caches into image, montage tile, and profile/scalar caches, each with a policy budget.
Expose the current policy and runtime state through Developer -> Diagnostics. Diagnostics is a plain
`QDialog`, not a managed dock, and shows both color-coded filling bars and compact text sections.

Keep a `stage_cache_budget_bytes` policy field for Phase 4g diagnostics and planning, but do not
allocate or implement StageCache in this phase.

## Consequences

Budget behavior is more predictable and inspectable. Cache budgets adapt to profile and system memory
without treating the render cap as a total application memory cap. The user still controls the
per-render hard cap. There is no OS-level process hard limit.

## Rejected Alternatives

- Keep static constants: rejected because limits stayed scattered and opaque.
- Treat render cap as a total app memory cap: rejected because cache and render guardrails need
  different semantics.
- Add diagnostics as a managed dock: rejected because diagnostics should not affect panel layout or
  canvas-preservation behavior.
- Implement StageCache now: rejected to keep Phase 4f focused on policy, cache split, and diagnostics.

## Tests Required

Pure policy/formatter tests, settings roundtrip tests, cache resize/split tests, render/montage/prefetch
budget tests, diagnostics dialog tests, and architecture guards for Qt-free policy/formatter modules.

## Manual Checks Required

Open Developer -> Diagnostics, switch memory profiles and render caps, confirm the bars and text update,
confirm cache usage changes are visible, and confirm the dialog does not resize or disturb the main
viewer canvas.
