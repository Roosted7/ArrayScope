"""Screenshot smoke-test for arrayscope.

Directly instantiates ArrayScopeWindow (bypassing multiprocessing), renders it
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
    import argparse
    import pyqtgraph as pg
    from arrayscope.arrayscope import ArrayScopeWindow

    parser = argparse.ArgumentParser()
    parser.add_argument('--style', default='Fusion',
                        help='Qt style name, e.g. Fusion, Windows, macOS (default: Fusion)')
    parser.add_argument('--out', default=None,
                        help='Output directory for screenshot (default: test/screenshots/)')
    args = parser.parse_args()

    data = make_data()

    app = pg.mkQApp()
    app.setStyle(args.style)
    win = ArrayScopeWindow(data)
    win.setWindowTitle(f"CI test — {sys.platform} — style: {args.style}")
    win.resize(800, 800)
    win.show()

    # Let Qt process events so the image actually renders
    for _ in range(5):
        app.processEvents()

    out_dir = Path(args.out) if args.out else Path(__file__).parent / "screenshots"
    out = out_dir / f"screenshot_{sys.platform}.png"
    take_screenshot(win, out)

    win.close()
    app.quit()
    print("All checks passed.")


if __name__ == "__main__":
    main()
