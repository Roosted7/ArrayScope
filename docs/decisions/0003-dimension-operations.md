# 0003 — Dimension operations boundary

Dimension operations that create derived array values live in `arrayscope.operations.dim_ops`.

It must not import Qt or pyqtgraph.

It owns:
- centered FFT and inverse FFT transforms along one axis;
- fftshift and inverse fftshift along one axis;
- combining a size-2 real/imag axis into a singleton complex axis;
- splitting a singleton complex axis back into a size-2 real/imag axis.

It does not own:
- labels, icons, menus, or click handling;
- selected primary/secondary dimensions;
- line plot versus image mode;
- domain label styling;
- enabling or disabling channel controls.

The main window remains responsible for translating user actions into operations,
updating UI state, and refreshing the view. New array-valued dimension operations
should be added to `dim_ops.py` with shape/value tests before being wired into Qt.
