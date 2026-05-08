"""Screenshot smoke-test for ndslice.

Directly instantiates NDSliceWindow (bypassing multiprocessing), renders it
headlessly, and saves a PNG. Run via:
  xvfb-run -a python test/screenshot_test.py   # Linux
  python test/screenshot_test.py               # macOS / Windows
"""
import sys
import numpy as np
from pathlib import Path


def make_data():
    """3D complex Gaussian from the README."""
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    z = np.linspace(-5, 5, 50)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    mag = np.exp(-(X**2 + Y**2 + Z**2) / 10)
    pha = np.pi / 4 * (X + Y + Z)
    return (mag * np.exp(1j * pha)).astype(np.complex64)


def take_screenshot(win, path: Path):
    """Grab the window contents and save as PNG."""
    pixmap = win.grab()
    assert not pixmap.isNull(), "grab() returned a null pixmap"
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = pixmap.save(str(path), "PNG")
    assert ok, f"Failed to save screenshot to {path}"
    size = path.stat().st_size
    assert size > 1000, f"Screenshot suspiciously small ({size} bytes)"
    print(f"Screenshot saved: {path}  ({size} bytes)")


def main():
    import pyqtgraph as pg
    from ndslice.ndslice import NDSliceWindow

    data = make_data()

    app = pg.mkQApp()
    win = NDSliceWindow(data)
    win.setWindowTitle(f"CI test — {sys.platform}")
    win.resize(800, 800)
    win.show()

    # Let Qt process events so the image actually renders
    for _ in range(5):
        app.processEvents()

    out = Path(__file__).parent / "screenshots" / f"screenshot_{sys.platform}.png"
    take_screenshot(win, out)

    win.close()
    app.quit()
    print("All checks passed.")


if __name__ == "__main__":
    main()
