# ArrayView reference

**Observed source:** supplied checkout of [`oscarvanderheide/arrayview`](https://github.com/oscarvanderheide/arrayview), tag `v0.26.3` as reviewed on 2026-06-22.

ArrayView is a modern browser/webview-based array viewer whose product identity is “the array fills the screen.” It emphasizes visual restraint, keyboard operation, broad invocation environments, and many specialized viewing modes.

## What ArrayView optimizes for

- Minimal, polished chrome with progressive disclosure, immersive/zen/compact modes, and consistent visual tokens.
- Keyboard-first navigation with a command registry, command palette, and generated/validated help surfaces.
- Easy launch from CLI, Python, Jupyter, Julia, MATLAB, local/remote browser, and VS Code webviews.
- Normal slicing, orthogonal multiview, mosaic, projections, centered FFT, complex/RGB display.
- Rich compare workflows: side-by-side, signed/absolute/normalized difference, overlay, wipe, flicker, checkerboard, and registration-oriented views.
- ROI/ruler, overlays, segmentation controls, vector fields, MIP, qMRI, file/session management, and share/export flows.
- Lazy imports and optional format dependencies to keep launch paths convenient.

Its design documents are unusually explicit about visual restraint: reuse surfaces, avoid accumulating panels, treat modes as lenses, and reconcile UI state through common paths.

## Technical shape

The Python side owns launch routing, sessions, loading, analysis, rendering, routes, WebSocket/stdio transports, and host integration. A session stores data/metadata and LRU caches; CPU-heavy rendering runs on a dedicated daemon thread, with a separate one-thread prefetch executor.

The frontend is a self-contained HTML/JavaScript/CSS application. At the reviewed version:

- `src/arrayview/_viewer.html` is 29,561 lines;
- `src/arrayview/_launcher.py` is 3,858 lines;
- the frontend contains commands, modes, layouts, rendering, reconcilers, overlays, CSS, and host transport integration in one deliverable.

A newer View/Layer/Slicer/LayoutStrategy/ModeManager model is being introduced alongside legacy globals/rendering. The architecture document explicitly describes “sync blocks” and dual writes that keep legacy state and `displayState` aligned until migration completes. Reconcilers centralize UI visibility/layout after scattered mode toggles caused consistency problems.

The server generally extracts a slice, applies transforms/complex mode, renders RGBA/PNG on the Python side, transports it by WebSocket or stdio, and draws it to browser canvas. MIP and some browser-specific work use WebGL.

## What ArrayScope should adopt

### Product restraint

The image/array should dominate. Controls can appear contextually or live in compact existing surfaces. A new feature should not automatically create a permanent dock.

### One command model

Menus, shortcuts, command palette, help, and host integrations should call one semantic command registry with context predicates. Tests should verify command reachability and mode/state consistency.

### Invocation quality

ArrayView treats “works where the user works” as a feature. ArrayScope should improve notebook/editor/CLI invocation, but through one stable session API rather than separate behavior implementations.

### Compare as a focused first-class workflow

Side-by-side shared viewport/levels plus a small set of difference/overlay modes would add high value to scientific inspection. It should remain narrower than ArrayView’s full registration/qMRI/segmentation mode set.

### Visual and lifecycle audits

ArrayView’s mode matrix, visual checklists, Playwright artifacts, and explicit design rules are useful discipline. ArrayScope should add comparable backend/platform interaction matrices without copying browser-specific tooling wholesale.

### Reconciliation as a migration lesson

When state has already become scattered, one convergence path is better than ad hoc toggles. More importantly, ArrayScope should prevent duplicate state ownership so reconcilers are a temporary migration tool, not permanent architecture.

## What ArrayScope should adapt, not copy

### Browser/webview delivery

Broad deployment is attractive, but ArrayScope’s core value includes direct NumPy/Qt integration, precise local interaction, and GPU/CPU scheduling. A web frontend would introduce transport, encoding, host, and duplicated lifecycle complexity. Host adapters should remain optional unless usage evidence justifies them.

### Mode vocabulary

ArrayView’s modes are impressive but create a combinatorial consistency burden. ArrayScope should prefer composable state/overlays/storage strategies and add a mode only when it has a distinct coherent workflow and matrix tests.

### Server-rendered RGBA

Pre-rendered RGBA is portable and simple for a browser, but it can spend CPU/bandwidth and discard raw-value flexibility. ArrayScope’s shader/raw-texture path is better suited to local window/level and complex display when stable.

## What ArrayScope should avoid

- A 29,000-line self-contained frontend or similarly concentrated Qt module.
- Permanent dual-write state and legacy/new render pipelines.
- Duplicated WebSocket/stdio/browser protocols without demonstrated need.
- A global session registry as the semantic owner of all viewers.
- One daemon render thread becoming a universal bottleneck for all sessions/lanes.
- Feature/mode growth that requires a reconciler for every combination.
- UI polish implemented through hidden coupling or animation that delays input/presentation.
- CPU RGBA/PNG conversion for local presentation when raw texture/shader mapping is available.

## Position relative to ArrayScope

ArrayView is ahead in visual polish, invocation reach, compare workflows, keyboard discoverability, and explicit product design. ArrayScope is ahead in reversible operation planning, region/stage evaluation, explicit memory/admission policy, semantic progressive montage state, and backend-independent value/level correctness. ArrayScope should learn ArrayView’s restraint and delivery focus without importing its frontend concentration, transport stack, mode combinatorics, or migration dual writes.
