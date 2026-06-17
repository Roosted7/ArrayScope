# 0036 — Experimental VisPy rendering backend

## Problem

ArrayScope now avoids most unnecessary computation for large montage views, but the remaining hot path is often display upload/windowing rather than operation evaluation. PyQtGraph remains valuable for interaction and histogram controls, but its `ImageItem` path still requires repeated CPU-side RGB/windowing and large `setImage()` commits in some cases.

## Decision

Add an experimental VisPy image-rendering backend while keeping the existing PyQtGraph interaction layer. The experiment is selected from the Performance menu and is not enabled by default.

The first implementation is intentionally hybrid:

- PyQtGraph still owns interaction, ROI/profile state, HUD, and the histogram widget.
- VisPy owns image pixel presentation under the PyQtGraph interaction layer.
- Where native `QOpenGLWidget` stacking can hide PyQtGraph graphics on Wayland, VisPy draws passive visual mirrors for ROI outlines/handles, the live-profile crosshair/target, and montage loading/skipped placeholders. Those mirrors do not own application state; PyQtGraph callbacks still update ROI/profile state and downstream profile/inspection work.
- Scalar images use VisPy `ImageVisual` with `texture_format="auto"` and `clim` so window/level changes can be evaluated by the GPU where possible.
- RGB/complex views that carry scalar histogram/intensity data use an ArrayScope VisPy visual with separate color and scalar textures. Window/level changes update shader uniforms instead of rebuilding a CPU-windowed RGB image.
- Large montage tile-layer presentation maps to VisPy visuals per tile rather than a single full-canvas RGB upload.

This tests the question we actually care about: whether replacing the pixel display path improves large-view responsiveness without rewriting all tools at once.

## Consequences

Positive:

- Keeps the stable ArrayScope coordinate/ROI/profile systems intact for the first experiment.
- Gives us a fast path for scalar GPU windowing and a practical comparison against PyQtGraph `ImageItem` upload costs.
- Avoids investing in a complete custom VisPy ROI/editor stack before the value of VisPy is proven.

Trade-offs:

- This is still a hybrid renderer, not a pure VisPy viewer.
- RGB/complex views still receive CPU-prepared phase/color and scalar magnitude/intensity arrays from the current display pipeline, but hot window/level changes no longer rebuild RGB pixels on the CPU in the VisPy path.
- Tile-layer mode uses multiple VisPy image visuals and must be manually tested on target GPUs/compositors.
- The VisPy canvas is intentionally mouse-transparent and its camera is non-interactive. PyQtGraph remains the single viewport/interaction owner; VisPy follows the ViewBox range and axis inversion state.

## Rejected alternatives

- Replacing the entire image view with VisPy immediately. Too risky: ROI editing, histogram interactions, live profile, HUD, and context menus are already stable in PyQtGraph.
- Keeping only PyQtGraph and continuing to micro-optimize `ImageItem`. This would not answer whether GPU texture scaling materially improves ArrayScope's main bottleneck.
- Adding a compatibility shim layer for every old internal call. Internal compatibility is not a goal; the experimental backend preserves the public image-view contract used by the render pipeline only.

## Tests required

- Settings round-trip for the selected image rendering backend.
- Import/compile without VisPy installed.
- Manual launch with VisPy installed.
- Normal scalar image render, window/level drag, Fit, 1:1.
- Large scalar montage window/level drag.
- Large complex/RGB montage tile-layer display.
- ROI/profile overlays remain visible, editable, and aligned with the VisPy image layer, including Wayland smoke artifacts.

## Future work

- Moving phase/LUT generation itself to GPU-side complex-data shaders. The current shader handles intensity windowing from already-derived color and scalar textures.
- Dedicated VisPy tile visual state diagnostics.
- Optional pure-VisPy interaction/ROI stack only if the hybrid experiment proves clearly better.
