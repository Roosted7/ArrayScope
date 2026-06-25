# ADR 0040: Backend-aware presentation convergence

- **Status:** Implemented for the montage presentation path
- **Date:** 2026-06-24
- **Supersedes:** the backend-neutral interpretation of level-only commits in ADR 0039
- **Related:** ADR 0031, ADR 0032, ADR 0033, ADR 0037, ADR 0038, ADR 0039

## Context

ArrayScope presents one semantic level target through backends with materially different physical
costs.

For the VisPy tiled path, source pixels remain resident in textures and window/level is a shader
uniform. A new target can normally be applied to all active page visuals without rebuilding or
uploading tile data.

For the PyQtGraph tiled path, the answer depends on the payload:

- scalar `ImageItem` data can often use per-item `setLevels()`;
- complex/component and RGB display payloads are CPU-windowed and require a new display array plus an
  `ImageItem` update for every affected tile;
- a large visible montage therefore cannot be updated safely in one unbounded GUI callback.

The semantic command is global, but the PyQtGraph physical transition is necessarily progressive.
Treating it as an atomic backend-neutral operation either freezes the event loop or silently updates
only the admitted subset.

The v30 level-loop failure exposed a second problem. The backend report described every retained item
as *presented*, even when the current upsert had been deferred by a deadline. The session interpreted
visibility as acknowledgement of the new level revision, removed that tile from the pending set, and
could finish with a mixture of old and new levels. Rapid targets and auto-window work made the subset
vary between iterations.

A related race used `user_levels_override` both as persistence policy and as the current physical
presentation target. An older automatic commit could therefore restore an obsolete target while a
newer progressive update was still draining.

## Decision

ArrayScope will share one semantic **presentation-generation contract**, while each backend owns a
capability-specific convergence strategy.

### Semantic generation

A level command produces a target with at least:

```python
@dataclass(frozen=True)
class LevelPresentationTarget:
    revision: int
    levels: tuple[float, float]
    source: LevelSource
    semantic_key: object
    active_tiles: frozenset[int]
```

The target is authoritative independently of user-lock persistence. A new command increments the
revision and supersedes every older automatic or interactive target. It does not recreate the render
session or invalidate unchanged source payloads.

The implementation stores these concepts in `PresentationGenerationTracker`, with
`MontageRenderSession` retaining only session orchestration and committed presentation ownership.

### PyQtGraph strategy: progressive CPU/item convergence

For a PyQtGraph tile layer:

- every active tile whose acknowledged revision/value differs from the target becomes stale;
- stale tiles enter the existing viewport/hover priority order;
- each GUI callback admits work by item, byte, and elapsed-time budget;
- the old displayed tile remains visible until its replacement is accepted;
- the backend reports exactly which requested upserts were committed;
- only those tiles advance to the target revision;
- completion means every currently active tile is acknowledged at the target and no target upsert is
  pending.

A PyQtGraph level target is therefore semantically singular and physically progressive. Mixed old/new
pixels are allowed only as an observable transition with a non-zero stale count; they may never be
reported as converged.

The direct all-item MontageTileLayer.update_levels() path remains a small/standalone fallback. It is
not the application scheduling contract and must not be used to bypass the session queue for a large
committed montage.

### VisPy strategy: shader/uniform convergence

For a compatible VisPy tiled surface:

- source texture identity and residency remain unchanged;
- the target levels are applied to all active page visuals as uniforms;
- no texture upload is attributed to the level-only transition;
- the target is acknowledged after the relevant visuals contain the new uniform state and a draw is
  invalidated;
- context loss or an incompatible texture representation falls back to normal payload admission
  rather than pretending the uniform commit succeeded.

This is one backend transaction, not a per-tile CPU redraw queue.

### Acknowledgement vocabulary

Backend reports use distinct fields:

- `presented_tiles`: drawable after this commit, including retained old pixels;
- `committed_upserts`: requested payloads actually accepted in this commit;
- physical counters: image replacements, CPU windows, texture uploads, uniform updates, and related
  elapsed work;
- convergence state: target revision, stale active count, pending count, and settled flag.

Visibility, worker completion, materialization, residency, and presentation acknowledgement are never
substitutes for one another.

### Histogram and persistence

Detailed histogram plotting is a separate refinement lane. It may lag behind a valid semantic level
source and must not gate first pixels.

`user_levels_override` records window-mode/persistence intent.
`PresentationGenerationTracker.target_levels` records the latest physical presentation command.
Automatic, restored, and explicit-user sources retain their own ranks. A newer concrete command
clears obsolete `force_auto` work attached to the session.

### Benchmark completion

A level benchmark is complete only when the current target revision is settled. “All tiles are still
visible” is insufficient because retained tiles may still show a previous target.

## Consequences

Positive:

- PyQtGraph remains responsive without sacrificing eventual full coverage;
- VisPy retains its shader-update advantage without forcing its mechanics onto the fallback backend;
- rapid targets deterministically converge on the latest value;
- old pixels remain visible instead of flashing placeholders;
- budget feedback receives the actual amount of PyQtGraph tile work;
- tests can assert semantic parity while allowing backend-specific work counters.

Costs:

- transient mixed levels are possible on PyQtGraph and must be exposed as in-progress, not hidden;
- additional extracted model surface exists so generation/admission/fan-in can be tested without Qt;
- backend conformance tests need both target-state assertions and different physical-work assertions;
- final visual proof still requires real Qt/OpenGL/platform runs.

## Rejected alternatives

- **Apply levels to every PyQtGraph item synchronously.** This makes cost proportional to the
  visible tile count in one GUI callback and violates the responsiveness contract.
- **Treat the global target as an atomic backend operation.** This is true for the compatible VisPy
  uniform path, not for CPU-windowed PyQtGraph tiles.
- **Acknowledge every visible tile.** Retained visibility says nothing about whether the requested
  replacement was accepted.
- **Replace the montage session for auto-window.** This destroys useful materialization/residency
  and races with the active progressive generation.
- **Use user-lock state as the current target.** Persistence policy and physical convergence have
  different lifecycles.

## Migration and enforcement

The v30 review implemented the immediate corrections:

- explicit `committed_upserts` acknowledgement;
- latest-target revision/value tracking across partial commits;
- auto-window within the committed session;
- benchmark completion based on convergence;
- accurate PyQtGraph level-work counters;
- regressions for deferred visible upserts, one-tile batches, rapid supersession, and stale automatic
  work.

The N6 control-plane extraction then moved generation bookkeeping into
`PresentationGenerationTracker`, admission policy into `TileAdmissionQueue`, convergence behavior
behind `LevelConvergenceStrategy`, and stage batching into `StageFanInState`. Legacy session aliases
for these models were removed so future callers use the canonical owners directly.
