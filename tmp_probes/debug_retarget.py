import numpy as np, sys
sys.path.insert(0, "/home/thomas/projects/ArrayScope-lod-test")
from arrayscope.app.qt_binding import prefer_pyside6
prefer_pyside6()
from pyqtgraph.Qt import QtCore, QtWidgets
from arrayscope.app.settings_state import ImageRenderingBackendChoice

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
settings = QtCore.QSettings()
settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.VISPY.value)
settings.sync()
from arrayscope.window import ArrayScopeWindow

win = ArrayScopeWindow(np.arange(4 * 10 * 8, dtype=np.float32).reshape(4, 10, 8))
win.resize(360, 240)
win.show()
for _ in range(5):
    app.processEvents()
win.montage_tile_evaluation_controller.start_latest = (
    lambda _fn, **kwargs: len(getattr(win._montage_session, "active_tile_requests", ())) + 1
)
win._set_view_state(win.view_state.with_montage_axis(2, columns=8, indices=tuple(range(8)), text=":"))
win.update_image_view()
s = win._montage_session
print("after update: sid", s.session_id, "deferred", getattr(s, "stage_planning_deferred", None),
      "interaction", win._viewport_interaction_active, "visible", len(s.visible_tiles))
win.img_view.getView().setRange(xRange=(0.0, 10.0), yRange=(0.0, 4.0), padding=0)
print("token before:", getattr(win.renderer, "_montage_viewport_update_token", None),
      "running:", getattr(win.renderer, "_montage_viewport_update_running", False))
win.renderer._run_montage_viewport_update()
s2 = win._montage_session
print("after wrapper: sid", s2.session_id, "visible", len(s2.visible_tiles),
      "deferred", getattr(s2, "stage_planning_deferred", None))
win.close()
app.processEvents()
print("DONE")
