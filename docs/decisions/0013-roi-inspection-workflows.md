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
important ROI values. The Inspection dock remains optional, floats by default when opened, and provides
the full ROI list, finite-value statistics, and shared-range histogram comparisons. ROI statistics are
computed from the current displayed scalar image or the image histogram source, so complex RGB views
use the same magnitude source as the image histogram.

A minimal compare-layer scaffold exists for same-ROI histogram comparison against compatible 2D
arrays. It is intentionally not full Phase 5 session or synchronized-window support.
