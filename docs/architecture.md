# Architecture

Main principle:
Qt displays and collects user intent. NumPy code transforms data.

Core boundaries:
- `ViewState`: what the user is currently looking at.
- `slice_engine`: converts ndarray + ViewState to displayable 2D/RGB/1D data.
- `dim_ops`: pure NumPy dimension operations such as centered FFT, fftshift, and real/complex axis conversion.
- `window`: assembles widgets and wires signals.

Avoid:
- putting array math in QWidget classes
- reading state back from widgets as the source of truth
- duplicating display conversion in export/video code
