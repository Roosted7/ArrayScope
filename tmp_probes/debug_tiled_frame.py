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

win = ArrayScopeWindow(np.arange(2 * 2 * 8, dtype=np.float32).reshape(2, 2, 8))
for _ in range(5):
    app.processEvents()

def frame_info(tag):
    f = getattr(win, "_committed_display_frame", None)
    vs = getattr(f, "value_source", None)
    print(tag,
          "frame", type(f).__name__ if f is not None else None,
          "scene", getattr(f, "scene", None) is not None,
          "payloads_attr", hasattr(vs, "payloads"),
          "interaction", win._viewport_interaction_active,
          flush=True)
    s = win.renderer._montage_session
    if s is not None:
        print("   sid", s.session_id, "deferred", getattr(s, "stage_planning_deferred", None),
              "displaymode", win.img_view.montageDisplayMode(),
              "display_committed", s.display_committed, flush=True)

win._viewport_interaction_active = True  # force the deferral branch deterministically
win._set_view_state(win.view_state.with_montage_axis(2, columns=4, indices=tuple(range(8)), text=":"))
frame_info("after set_view_state")
win.update_image_view()
frame_info("after update_image_view")
win.close()
app.processEvents()
print("DONE")
