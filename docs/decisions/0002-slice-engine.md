# 0002 — Slice engine boundary

The slice engine converts `(data, ViewState)` into display-ready outputs.

It must not import Qt.

It owns:
- nD slicing
- selected image axes
- selected line axis
- channel conversion: real, imag, abs, angle, complex RGB
- scalar display scaling inputs

It does not own:
- widgets
- QAction/menu wiring
- file loading
- video export UI
- long-running worker management