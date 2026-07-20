# Packaging ArrayScope

Binary bundles for people who want ArrayScope as *a viewer* — no Python
visible, double-click a file and it opens. Python users should keep using
`pip install arrayscope` (see the README's install section); the bundles and
the wheel are the same code, the bundles just carry their own interpreter.

| Platform | Artifact | Built by |
|---|---|---|
| Linux | `ArrayScope-<v>-x86_64.AppImage` (portable, no install) | `linux/build_appimage.sh` |
| Windows | `ArrayScope-Setup-<v>.exe` (wizard installer) + `...-portable.zip` | `windows/build_installer.ps1` |
| macOS | `ArrayScope-<v>-macos-<arch>.dmg` (drag to Applications) | `macos/build_dmg.sh` |

Known limitation: DICOM *directory* conversion shells out to `dcm2niix`,
which is not bundled — users who need it install it separately on PATH
(single `.dcm` files work without it).

All three are PyInstaller one-directory bundles built from the shared spec
`pyinstaller/arrayscope.spec` with `pyinstaller/entry.py` as the frozen entry
point (adds `multiprocessing.freeze_support()`, a file-open dialog when
launched with no arguments, and the window icon). Expect roughly 200 MB
compressed / 500 MB on disk — PySide6 and the scientific stack dominate.

## Building locally

Each script must run **on its target platform** (PyInstaller does not
cross-compile). From the repository root, in a fresh environment:

```bash
pip install ".[installer]"          # the repo + PyInstaller
bash packaging/linux/build_appimage.sh                                   # Linux
powershell -ExecutionPolicy Bypass -File packaging\windows\build_installer.ps1  # Windows (needs Inno Setup 6)
bash packaging/macos/build_dmg.sh                                        # macOS
```

Outputs land in `dist/`. Notes:

- **Linux**: a local AppImage runs on machines with glibc >= yours. The CI
  build uses an `ubuntu:22.04` container (glibc 2.35) for wide compatibility;
  prefer the CI artifact for distribution. `APPIMAGETOOL=/path/to/appimagetool`
  skips the download.
- **Windows**: install Inno Setup 6 first (`winget install JRSoftware.InnoSetup`).
  The installer defaults to per-user (no admin prompt) with an
  all-users option; file associations for ArrayScope-owned formats
  (.npy, .npz, .cfl, .rec, .nii, .mat) are an installer task, and DICOM/HDF5
  only join the "Open with" menu — this mirrors the owned/shared split in the
  desktop-integration catalogue (`arrayscope/desktop/filetypes.py` on the
  `claude/file-loading-desktop-app-35eef4` branch).
- **macOS**: unsigned bundle; first launch is right-click → Open (Gatekeeper).
  Signing + notarization is a follow-up once an Apple Developer ID exists.

## CI (GitHub Actions)

`.github/workflows/installers.yml` builds all platforms on every
`workflow_dispatch` (artifacts downloadable from the run page) and on every
published release (artifacts attached to the release, next to the PyPI upload
from `pypi-release.yml`). Windows runners no longer preinstall Inno Setup, so
the workflow installs it with Chocolatey. macOS builds twice: `macos-latest`
(arm64) and `macos-15-intel` (x86_64; GitHub retires Intel runners August 2027).

Every job smoke-tests its bundle: `--help` must exit 0, and opening a
generated `.npy` offscreen must survive a few seconds without crashing.

## Frozen-app source contracts

Two pieces of `arrayscope/` exist specifically for these bundles — keep them
working:

- `arrayscope/app/qt_platform.py` relaunches the Wayland-supervised CLI as
  `[sys.executable, *argv]` when `sys.frozen` is set (there is no `-m` in a
  bundle). Covered by `tests/app/test_qt_platform.py`.
- `packaging/pyinstaller/entry.py` calls `multiprocessing.freeze_support()`
  first — selector windows and non-blocking launches spawn processes that
  re-execute the bundled binary.

When the desktop-integration branch merges, its launcher window supersedes the
no-args file dialog in `entry.py` (delete the dialog, keep freeze_support and
the icon), and `arrayscope --install-desktop` becomes the integration story
for pip installs while these bundles stay the zero-Python story.
