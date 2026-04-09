# Changelog

## [0.5.1] - 2026-04-09

### Fixed
- **Colormap switching** — Fixed error when switching back to gray colormap on PyQt5 systems without matplotlib.

## [0.5.0] - 2026-02-18

### Added
- **PyQt6 support** — works with both PyQt5 (default) and PyQt6 via optional dependency: `pip install ndslice[pyqt6]`
- **HiDPI display support**
- **Colormaps** — Added colormaps with keyboard shortcuts:
  - Ctrl+1: Gray
  - Ctrl+2: [Viridis](https://bids.github.io/colormap/)
  - Ctrl+3: [Plasma](https://bids.github.io/colormap/)
  - Ctrl+4: PAL-relaxed (cyclic, hides phase wraps)
  - Ctrl+5: [Cividis](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0199239)
  - Ctrl+6: [Cubehelix](http://www.mrao.cam.ac.uk/~dag/CUBEHELIX/)
  - Ctrl+7: [Cool](https://d3js.org/d3-scale-chromatic/sequential)
  - Ctrl+8: [Warm](https://d3js.org/d3-scale-chromatic/sequential)
- **Video export**:
  - GIF, WebM, MP4, PNG (frames)
  - Window/Level can be per-slice or fixed
  
- **Update pyqtgraph to 0.14.0**

### Fixed
- Window/Level reset on re-clicking `linear` / `symlog`
- MATLAB v7.3 file loading — falls back to HDF5 loader when scipy.io.loadmat fails

