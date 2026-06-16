# 0031: Render Presentation Model

## Problem

Progressive montage rendering still had multiple level decision paths. Full image commits, first
montage commits, progressive tile patches, histogram widget state, and partial viewport canvases could
all influence the levels that were displayed. In operation-backed montage views this made the
histogram range, shown levels, and actual image item levels diverge. Panning could appear to fix the
view because it re-entered rendering through a different cached-tile path.

## Decision

Add `arrayscope.window.render_model` as the Qt-free boundary model for visible display commits. Render
orchestration now builds a `PresentationInput` with the display payload, committed previous frame,
commit kind, semantic montage level source, and render request context. `arrayscope.window.presentation`
owns `decide_presentation()`, which returns the one `PresentationDecision` used by `DisplayCommitter`.

`DisplayCommitter` remains the only writer to `ImageView2D` and now validates shape compatibility and
finite increasing bounds before mutating Qt state. Progressive montage commits preserve committed/user
levels unless an explicit Auto Window request or a complete semantic montage level source permits an
upgrade. Visible-subset montage stats are never implicit level sources.

User histogram level edits are represented explicitly: `ImageView2D` emits `levelsChanged`, and the
window records those levels in committed display/session state instead of asking render policy to read
the widget later.

## Consequences

Panning, viewport culling, tile arrival order, cache hits, and worker scheduling no longer change
levels by accident. Partial montage canvases may update pixels and histogram sources for loaded
regions, but they cannot become implicit semantic window/level bounds. Degenerate bounds are
normalized before they reach Qt.

`render.py` still orchestrates much of visible rendering, but it no longer owns presentation policy or
partial-canvas level scans. Further controller extraction should move orchestration without
reintroducing alternate level decision paths.

## Tests Required

Pure tests cover normal relative/absolute reuse, explicit Auto Window, progressive montage preservation,
partial-source rejection, complete-source acceptance, and degenerate bounds. UI tests cover progressive
tile commits, loading canvases, panning without new tiles, and visible subset level stability.
Architecture guards prevent `render.py` from importing presentation-policy helpers or level-scan
helpers.
