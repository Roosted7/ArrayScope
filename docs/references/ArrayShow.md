# ArrayShow reference

**Observed source:** [`tsumpf/arrShow`](https://github.com/tsumpf/arrShow), `develop` branch as reviewed on 2026-06-22.

ArrayShow is a mature MATLAB-oriented n-dimensional array viewer and the clearest product ancestor for ArrayScope’s dimension-first interaction model.

## What ArrayShow optimizes for

ArrayShow assumes a user already works inside MATLAB and wants immediate visual manipulation of workspace arrays. Its central strengths are proximity and breadth:

- a terse `as(...)` launch path and workspace-aware object discovery;
- dimension selection, slice navigation, plotting, FFT/reduction/crop-style actions close to dimension controls;
- complex-data display/windowing and cursor-position callbacks;
- ROI, markers, annotations, information text, and image statistics;
- multi-view scripts and synchronized “send groups” for related viewers;
- utility scripts for arranging windows, applying operations to many viewers, copying selections/ROIs, collecting ROI values, and saving collages/images.

The project demonstrates that n-dimensional inspection is most effective when axes are visible concepts and common operations are attached to the axis being discussed.

## Technical shape

The code is organized around MATLAB handle objects such as the main data/view object and auxiliary windowing, ROI, marker, cursor, icon, and send-group classes. The `as` entry point and utility scripts also use a global viewer collection (`asObjs`) and MATLAB workspace conventions.

The main object coordinates a broad set of responsibilities:

- source/current data and selections;
- dimension roles and display mode;
- complex representation and statistics;
- rendering/windowing;
- callbacks, cursor functions, ROI, text, and markers;
- synchronization with related viewers.

Send groups and scripts make cross-window workflows powerful, but communication and lifecycle are coupled to global MATLAB object/workspace state. Several interaction paths rely on callback ordering, draw/update recursion guards, pauses/timing, and figure-handle behavior. Playback rate is an intent/control value rather than a measured presentation deadline.

## What ArrayScope should adopt

### Dimension-local operations

A user should be able to point at an axis and choose an operation expressed in that axis’s terms: crop/select, reverse, reduce, FFT/IFFT, shift, montage, profile, export. ArrayScope already has the safer reversible operation pipeline; the improvement is to make the entry points as immediate as ArrayShow’s without hiding what will happen.

### Linked viewer groups

Synchronized windows are valuable for image/k-space, reconstruction variants, channels/coils, or parameter sweeps. Implement them as explicit typed group objects that independently link slice, cursor, viewport, levels, ROI, or operation recipe. Include origin/revision IDs to prevent feedback loops.

### Scriptable multi-viewer workflows

ArrayShow’s utility scripts reveal real user needs: align windows, propagate selections, set common levels, collect ROI values, and export a set. ArrayScope should expose stable semantic commands/session groups so Python users can automate these workflows without driving widgets.

### Fast contextual inspection

Cursor functions and ROI/profile tools should show useful local information immediately, then refine asynchronously when expensive. The interaction should stay close to the image rather than requiring a general analysis framework.

## What ArrayScope should adapt, not copy

### Workspace discovery

Python users benefit from easy launch, but implicit scanning of globals creates ambiguous ownership and lifetime. Prefer explicit array/session references, optional notebook/editor helpers, and clear process semantics.

### Operation behavior

ArrayShow’s direct manipulation can feel destructive because the active object/data and UI are tightly coupled. ArrayScope should retain immutable source/reversible operation steps, previews, undo/redo, and explicit materialization/export.

### Viewer synchronization

Global lists are convenient in MATLAB but fragile across processes, notebooks, and windows. Use scoped groups and weak/explicit lifecycle rather than a process-global viewer registry.

## What ArrayScope should avoid

- A single “god object” owning data, rendering, callbacks, ROI, windowing, synchronization, and UI state.
- Global workspace/viewer state as the routing mechanism.
- Recursive callback repair and `pause`/draw timing as a scheduler.
- Nominal playback FPS that does not account for evaluation and presented-frame time.
- Destructive defaults or hidden mutations during dimension operations.
- Figure-library quirks defining semantic state.

## Position relative to ArrayScope

ArrayShow remains a stronger model for dimension-local immediacy, linked windows, and scriptable groups. ArrayScope is stronger in reversible operations, explicit cache/memory/scheduling policy, backend-independent frame semantics, and Python integration. The right goal is not UI parity; it is ArrayShow’s workflow speed on top of ArrayScope’s safer document and rendering architecture.
