# 0013 — ROI inspection workflows

ArrayScope supports first-class ROI inspection for line, rectangle, polyline, and freehand polygon
regions. The implementation keeps ROI geometry and sampling in Qt-free helpers, while `ImageView2D`
owns pyqtgraph ROI graphics and emits complete geometry changes to the window.

Line and polyline ROIs are sampled along their path with `scipy.ndimage.map_coordinates`. Rectangle
and freehand ROIs are area-based; freehand gestures are simplified and closed into polygons before
statistics are computed. Polygon containment uses a small NumPy ray-casting helper instead of adding
matplotlib as a dependency.

Basic ROI operations are available from the image view context menu. Creating an ROI does not open the
Inspection dock; instead, a movable semi-transparent overlay on top of the 2D view shows the most
important ROI values. The Inspection dock remains optional, defaults to a left-docked panel when opened,
and provides finite-value statistics and shared-range histogram comparisons. ROI statistics are computed
from the current displayed scalar image or the image histogram source, so complex RGB views use the same
magnitude source as the image histogram.

Phase 4c makes interaction ownership explicit. The Inspection dock is an analysis/management panel and
may create line and rectangle ROIs immediately because they have sensible defaults. Polyline and
freehand ROIs require a canvas drag path and are started as one-shot drawing commands from the image
context menu or command palette. Persistent dock tools do not own freehand/polyline drawing, and
`ImageView2D.createRoi(FREEHAND_POLYGON)` rejects missing or insufficient points instead of
synthesizing a fake rectangle-like freehand polygon.

Phase 4b added a shared ROI store and `QAbstractTableModel` for dock rows. ROI colors are assigned once
and reused by image graphics, table rows, and histogram curves. Table selection highlights the matching
ROI, and delete/clear paths synchronize back to the image view.

A minimal compare-layer scaffold exists for same-ROI histogram comparison against compatible 2D
arrays. It is intentionally not full Phase 5 session or synchronized-window support. Phase 4c
debounces ROI statistics/histogram refreshes and moves large ROI computations to the window's ROI
evaluation controller. ROI rows update immediately, but statistics and histograms commit only when
the debounced ROI/image request key is still current.
