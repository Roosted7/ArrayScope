#!/usr/bin/env bash
# Build the ArrayScope AppImage (Linux, x86_64/aarch64).
#
# Prerequisites: a Python environment with the repo installed plus the
# installer extra —  pip install ".[installer]"  — and network access (or
# APPIMAGETOOL pointing at an existing appimagetool binary).
#
# Usage, from the repository root:
#   bash packaging/linux/build_appimage.sh
#
# Output: dist/ArrayScope-<version>-<arch>.AppImage
#
# glibc note: an AppImage runs on distros with glibc >= the build machine's.
# Release builds happen in an ubuntu:22.04 container (glibc 2.35) — see
# .github/workflows/installers.yml. A local build simply targets your own
# machine and newer.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VERSION="$(python -c 'from arrayscope._version import __version__; print(__version__)')"
ARCH="$(uname -m)"
BUILD_DIR="$REPO_ROOT/build/appimage"
APPDIR="$BUILD_DIR/ArrayScope.AppDir"

echo "==> PyInstaller bundle (ArrayScope $VERSION, $ARCH)"
pyinstaller --noconfirm --distpath "$BUILD_DIR/dist" --workpath "$BUILD_DIR/work" \
    packaging/pyinstaller/arrayscope.spec

echo "==> Assembling AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" \
    "$APPDIR/usr/share/icons/hicolor/256x256/apps"
cp -a "$BUILD_DIR/dist/ArrayScope/." "$APPDIR/usr/bin/"
cp packaging/linux/arrayscope.desktop "$APPDIR/"
cp packaging/linux/arrayscope.desktop "$APPDIR/usr/share/applications/"
cp arrayscope/resources/icons/arrayscope-256.png "$APPDIR/arrayscope.png"
cp arrayscope/resources/icons/arrayscope-256.png \
    "$APPDIR/usr/share/icons/hicolor/256x256/apps/arrayscope.png"
ln -sf usr/bin/ArrayScope "$APPDIR/AppRun"

# appimagetool: reuse $APPIMAGETOOL when provided, else fetch the pinned
# continuous build once into build/.
if [[ -z "${APPIMAGETOOL:-}" ]]; then
    APPIMAGETOOL="$BUILD_DIR/appimagetool-$ARCH.AppImage"
    if [[ ! -x "$APPIMAGETOOL" ]]; then
        echo "==> Downloading appimagetool"
        curl -fsSL -o "$APPIMAGETOOL" \
            "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$ARCH.AppImage"
        chmod +x "$APPIMAGETOOL"
    fi
fi

mkdir -p dist
OUTPUT="dist/ArrayScope-$VERSION-$ARCH.AppImage"
echo "==> Building $OUTPUT"
# --appimage-extract-and-run keeps this working where FUSE is unavailable
# (containers, CI).
ARCH="$ARCH" "$APPIMAGETOOL" --appimage-extract-and-run "$APPDIR" "$OUTPUT"

echo "==> Done: $OUTPUT"
