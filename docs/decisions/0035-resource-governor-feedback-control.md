# 0035 - Resource Governor Feedback Control

## Status

Accepted.

## Context

Hot-cache montage latency work removed most avoidable tile-layer uploads, but full operation redraws
still need to balance competing goals:

- use available CPU for stage-backed tile rendering;
- avoid UI freezes from too many result callbacks or display commits in one event-loop turn;
- keep prefetch and warmup useful without competing with visible work;
- make diagnostics explain whether work is stage compute, tile compute, upload, UI fan-in, or memory
  pressure.

Fixed duration tiers and independent lane limits were too local. They could keep a single path safe
while underusing the machine or overloading the UI when multiple subsystems became active.

## Decision

ArrayScope uses a Qt-free `ResourceGovernor` as the adaptive policy layer. `ComputePolicy` still
derives profile-based baseline worker and FFT-worker limits from settings. The governor combines that
policy with memory policy, resource telemetry, scheduler busy state, and latency feedback to choose
effective worker counts and UI fan-in budgets.

The governor is intentionally bounded:

- worker changes are damped and step-limited;
- UI pressure backs off immediately;
- recovery is gradual;
- prefetch is admitted per path and remains idle-only;
- montage tile FFT workers remain one by default to avoid native-worker oversubscription;
- stage/visible lanes can use more FFT workers because they run at most one job.

## Consequences

Diagnostics report resource pressure, feedback channels, effective lane worker decisions, and montage
compute paths. A cold montage redraw should show one reusable retained stage plus direct lead tiles
and stage-backed tiles, not ambiguous global cache hit/miss lines that look like one FFT per tile.

The prefetch controller queue remains available for cheap slice/profile prefetch. Expensive montage
tile and slice prefetch paths still perform their own stage-aware admission checks so the governor
does not accidentally turn one reusable FFT into many speculative tile FFTs.
