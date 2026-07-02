# v33 optimization and roadmap review

## Verdict

Do not throw away the current route. The recent work is right at the ownership
and semantic-contract level: a frame has one presentation meaning, interaction
semantics are shared, render staleness is centralized, and backend-specific code
is pushed toward physical mechanics. Throwing that away would reintroduce the
very duplication that caused stale callbacks, divergent levels, backend-only
interaction behavior, and cache-identity mistakes.

Do stop broad renderer churn now. The next risk is not another folder layout; it
is choosing physical strategies without evidence. X5 must prove what is fast on
real machines, then let the surface choose between singleton/direct, resident
tiled, virtual tiled, and eventually multi-resolution tiled mechanics under the
same semantic contract.

## Methods used in the recent work

The recent line used the right methods:

- **Ownership extraction.** Render state moved off the window into
  `RenderOrchestrator`; shared UI mirroring moved into `ViewStateBinder`; shared
  semantic display logic lives in the shell/surface rather than in each backend.
- **Semantic/physical separation.** `FramePlanner`, display model, value sources,
  and commit semantics describe what ArrayScope means; PyQtGraph item updates and
  VisPy texture/atlas updates describe how a backend draws it.
- **Centralized staleness and admission.** `window/render_contract.py` and
  `WorkGraph` replaced scattered token/counter/timer decisions.
- **Single bounded-cache core.** Display and stage caches now reuse
  `core.bounded_cache` instead of owning independent eviction loops.
- **Native-resolution production policy.** CPU-generated LOD is not the default;
  it waits for compatible residency mechanics and exact-value separation.
- **Contract tests and architecture guards.** The important tests target semantic
  contracts rather than concrete widget classes.

Those methods are sensible. The weakness is that they mostly prove consistency,
not end-user latency on hardware.

## How it integrates

Current flow:

```text
ViewState / document intent
  -> RenderOrchestrator
  -> render_contract generation checks + WorkGraph admission
  -> FramePlanner / tiled presentation model
  -> DisplayCommitter
  -> ImageSurface / ImageViewShell
  -> PyQtGraph or VisPy physical backend mechanics
```

This integration is clean enough to keep. The important boundary is that the
shared surface owns meaning and lifecycle, while the backend owns texture,
atlas, graphics item, draw invalidation, and capability mechanics.

The next integration point should be a physical strategy policy below the shared
surface, not another semantic split:

```python
class SurfacePhysicalStrategy(Enum):
    SINGLETON_DIRECT = "singleton_direct"
    RESIDENT_TILED = "resident_tiled"
    VIRTUAL_TILED = "virtual_tiled"


def choose_surface_strategy(frame, capabilities, budget, telemetry):
    """Return a physical strategy without changing frame semantics."""
    if frame.region_count == 1 and frame.estimated_gpu_bytes <= budget.direct_bytes:
        if capabilities.accepts_single_texture(frame.shape, frame.format):
            return SurfacePhysicalStrategy.SINGLETON_DIRECT
    if frame.requires_region_first or frame.estimated_gpu_bytes > budget.eager_bytes:
        return SurfacePhysicalStrategy.VIRTUAL_TILED
    return SurfacePhysicalStrategy.RESIDENT_TILED
```

This is deliberately policy-shaped, not hard-coded. The thresholds must come
from X5 traces.

## What it does well

- **It kills semantic forks.** A normal image, a huge plane, a one-tile montage,
  and a many-tile montage should not have different level semantics, hover/value
  rules, cache identity rules, or interaction state.
- **It protects interaction correctness.** Shared pointer/ROI/profile semantics
  reduce backend parity bugs.
- **It preserves last-good-frame behavior.** Slow work should not blank the UI.
- **It separates source identity from presentation identity.** Levels, LUTs, and
  shader mapping must not make unchanged texture data look like new source data.
- **It treats native-resolution residency as the safe production default.** That
  avoids mixing incompatible reduced tile shapes into fixed native atlas slots.
- **It moves toward truthful residency.** Requested tiles are not the same as
  backend-acknowledged resident tiles.

## What it does not do well enough yet

- **It does not prove performance.** Offscreen tests, contract tests, and
  software-GL runs do not tell us how the app feels on integrated GPUs, Wayland,
  Windows, macOS, or high-end render hosts.
- **It still contains large workflow knots.** `RenderOrchestrator` is the correct
  owner, but the file is still too large for precise performance debugging.
  Splitting should happen around proven workflows, not around another broad
  “renderer rewrite”.
- **It risks conflating semantic tiled presentation with physical tiling.** A
  small image should share semantics with tiled scenes, but it should not pay
  atlas/quad/residency overhead if a direct singleton surface is faster.
- **It still lacks the decisive normal-plane path.** Huge single images need
  region-first materialization and viewport retargeting based on tiled storage,
  not montage mode.
- **Backend defaults remain unearned.** VisPy may be the right large-scene path,
  but it is not a default until traces show latency, memory, context-loss, DPI,
  and parity behavior on real systems.

## Low-hanging fixes applied in this review branch

1. Retained tiled payload storage is now bounded before large insert batches.
   The old flow inserted into an initially unbounded `BoundedCache` and resized
   after the batch, which allowed avoidable memory spikes.
2. Retained tiled payload insertion no longer copies arbitrary payload mappings
   just to iterate values.
3. VisPy speculative/warm residency now stores an ordered queue and removes
   batches in constant-time queue operations. The previous timer path rebuilt a
   remaining mapping every tick, which is the wrong shape for huge montages.

These are intentionally small changes. They reduce obvious hot-path overhead
without changing the semantic architecture.

## What should not happen next

- Do not perform another large renderer rewrite before X5 evidence.
- Do not reintroduce the removed normal-image semantic branch, refuse/degraded
  preview decisions, or bespoke idle warmup scheduler.
- Do not make VisPy the default from theory or software-GL behavior.
- Do not enable the old synchronous CPU LOD pyramid.
- Do not force every small frame through identical atlas machinery merely to make
  the backend code look uniform.
- Do not accept payload caches or warm queues that are only bounded after the
  expensive work has already been admitted.

## Roadmap change

X5 is now the active gate and should be treated as five ordered sub-gates:

1. telemetry baseline;
2. acknowledged-residency conformance;
3. viewport-scoped tiled-scene retargeting;
4. region-first materialization plus physical strategy policy;
5. backend-default and LOD decisions.

The exit condition is not “VisPy works” or “tiling exists”. The exit condition is
a published trace matrix and conformance suite that justify the physical policy
on low-power and high-end systems.

## ADR change

ADR 0046 records the decision: keep one semantic surface contract, allow multiple
physical strategies beneath it, and make X5 evidence-first. It also rejects three
bad routes: another broad renderer rewrite, identical atlas storage for every
image, and resurrecting the old normal semantic path.

## Future implementation sketch

After X5a/X5b, introduce a small policy object that sees capabilities, budgets,
and telemetry:

```python
@dataclass(frozen=True)
class SurfaceStrategyDecision:
    strategy: SurfacePhysicalStrategy
    reason: str
    estimated_gpu_bytes: int
    expected_first_pixel_ms: float | None


class SurfaceStrategyPolicy:
    def choose(self, frame, *, capabilities, budget, telemetry) -> SurfaceStrategyDecision:
        ...
```

The policy should be deterministic and inspectable. The UI/debug HUD should be
able to say why a frame used direct texture, resident tiled, virtual tiled, or a
fallback path. Without that explanation, performance regressions will be hard to
trust and harder to fix.
