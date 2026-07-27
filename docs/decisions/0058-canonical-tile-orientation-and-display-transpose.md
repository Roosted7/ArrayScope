# ADR 0058: Canonical tile orientation and display-only axis swap

- **Status:** Accepted; implemented for the WGPU and PyQtGraph backends
  (2026-07-22). The former VisPy fallback is historical after
  [ADR 0061](0061-retire-vispy-rendering-backend.md); capability-gated fallback
  remains valid for any future non-canonical backend.

## Context

An X/Y axis-order swap (transpose) used to bake the display orientation into
tile **data** at materialization: the slab was read canonically but
`slice_engine._reorder_present_axes` transposed it into `image_axes` (display)
order before caching and upload. A swap therefore invalidated every tile and
forced a full re-render / re-upload — even though the underlying pixels are
identical, only their on-screen mapping changed.

An axis **flip** is already a pure display transform: `axis_flipped` is excluded
from the CPU cache key and applied at the camera (wgpu/vispy) or viewbox
(pyqtgraph). The requirement is that a transpose cost the same as a flip: data
must never be stored in an XY-dependent way (not in caches, not in GPU memory);
the render receives the X/Y mapping and its directions and samples correctly.

A transpose is not affine-diagonal, so unlike a flip it cannot ride the camera
scale; it needs a per-tile UV/geometry axis swap plus a layout swap.

## Decision

Materialize, cache, upload, and identify every tile **once, in canonical
(sorted-image-axes) orientation**, and apply the X/Y swap as a display
transform at draw time. Gate the whole behavior on a per-backend capability so
each backend can opt in independently and a non-capable backend keeps the
legacy path unchanged.

- **Capability.** `ImageViewBackendCapabilities.display_axis_transpose`
  (`display/backend_contract.py`). True for wgpu and PyQtGraph; false for VisPy.
- **Single gate.** The per-window `operation_evaluator` carries a
  `canonical_orientation` flag, set once per frame from the active backend's
  capability and mirrored onto the `FrameSession`. Canonical extraction skips
  the reorder (`slice_engine` `make_*` `canonical_orientation` param); cache
  keys and semantic identities sort `image_axes`/`keep_axes`. Because a
  transposed view's canonical key **equals** the unswapped key while a
  legacy display-order view keeps its distinct order, the two key spaces never
  collide — a flag mismatch degrades to a cache **miss (recompute)**, never a
  wrong-orientation reuse.
- **Display transform.**
  - wgpu: the vertex shader samples the canonical source with a swapped UV walk
    (`select(q, q.yx, Tile.transposed)`); the flag packs into the existing
    48-byte tile instance. Source windows are already canonical (resident
    texture) and world rects already display-oriented (layout), so the swap
    needs no rect resize (pages are square).
  - PyQtGraph: the `ImageItem` reads a transposed **view** (`swapaxes`) of the
    canonical buffer — a cheap view that shares memory, no copy or
    re-evaluation.
- **Re-present without re-upload.** A canonical swap re-presents no tiles (the
  payloads are unchanged), so the montage dirty/layout machinery sees no work —
  a SQUARE swap keeps even the tile shape. `retarget_index_window` flags
  `backend_refresh_pending` on an image-axes-order change to force one backend
  commit that re-lays-out over the resident textures.
- **Value readout.** `TiledValueSource.transposed` indexes the canonical array
  with swapped hover/ROI coordinates; derived from the backend capability in
  `DisplayCommitter._frame_for` and from the session in the montage direct-delta
  commit.
- **Page LOD is canonical.** The multi-resolution pyramid pages are canonical
  and transpose-invariant (shared across a swap), so their source rectangles and
  the reduced-floor `requested_lod.source_shape` must also be expressed in
  canonical order (`render.lod.canonical_source_tile_shape`). LOD **factor**
  selection stays display-oriented (it depends on how the tile appears on
  screen), as does montage layout topology.

## Consequences

- A transpose is instant on wgpu (GPU upload count flat across the swap) and
  PyQtGraph (no re-evaluation; a cheap transposed view), matching a flip.
- Extraction is a shared code path, so a single (non-montage) image transpose is
  display-only too — accepted as a consistent design rather than bifurcating the
  tile-only path.
- A page-LOD orientation trap: because pages are transpose-invariant but
  `plan.tile_shape` is display-oriented, any **source**-semantic use of the tile
  shape (page rects, `requested_lod.source_shape`) must be canonicalized or a
  transposed non-square montage trips
  `PageBackedPresentation`'s source-shape check inside floor admission.
- VisPy keeps the legacy re-render-on-swap path; the
  `frame_controller` `image-axes-order` retarget reject is now gated on
  `session.canonical_orientation` and remains the permanent fallback for
  non-canonical backends.
