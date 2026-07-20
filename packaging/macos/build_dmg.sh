#!/usr/bin/env bash
# Build ArrayScope.app and a drag-to-Applications DMG (macOS only).
#
# Prerequisites: a Python environment with the repo installed plus the
# installer extra —  pip install ".[installer]"  — on the target
# architecture (the bundle matches the Python that builds it).
#
# Usage, from the repository root:
#   bash packaging/macos/build_dmg.sh
#
# Output: dist/ArrayScope-<version>-macos-<arch>.dmg
#
# The bundle is unsigned: first launch needs right-click -> Open (Gatekeeper).
# Signing/notarization is a follow-up once an Apple Developer ID exists.

set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "error: this script must run on macOS (iconutil, hdiutil)" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VERSION="$(python -c 'from arrayscope._version import __version__; print(__version__)')"
ARCH="$(uname -m)"   # arm64 or x86_64
ICON_SRC="arrayscope/resources/icons"

echo "==> Generating arrayscope.icns"
ICONSET="build/icons/arrayscope.iconset"
rm -rf "$ICONSET" && mkdir -p "$ICONSET"
cp "$ICON_SRC/arrayscope-16.png"  "$ICONSET/icon_16x16.png"
cp "$ICON_SRC/arrayscope-32.png"  "$ICONSET/icon_16x16@2x.png"
cp "$ICON_SRC/arrayscope-32.png"  "$ICONSET/icon_32x32.png"
cp "$ICON_SRC/arrayscope-64.png"  "$ICONSET/icon_32x32@2x.png"
cp "$ICON_SRC/arrayscope-128.png" "$ICONSET/icon_128x128.png"
cp "$ICON_SRC/arrayscope-256.png" "$ICONSET/icon_128x128@2x.png"
cp "$ICON_SRC/arrayscope-256.png" "$ICONSET/icon_256x256.png"
cp "$ICON_SRC/arrayscope-512.png" "$ICONSET/icon_256x256@2x.png"
cp "$ICON_SRC/arrayscope-512.png" "$ICONSET/icon_512x512.png"
iconutil -c icns "$ICONSET" -o build/icons/arrayscope.icns

echo "==> PyInstaller bundle (ArrayScope $VERSION, $ARCH)"
pyinstaller --noconfirm --distpath build/pyinstaller/dist --workpath build/pyinstaller/work \
    packaging/pyinstaller/arrayscope.spec

echo "==> Building DMG"
STAGING="build/dmg-staging"
rm -rf "$STAGING" && mkdir -p "$STAGING"
cp -R "build/pyinstaller/dist/ArrayScope.app" "$STAGING/"
ln -s /Applications "$STAGING/Applications"

mkdir -p dist
OUTPUT="dist/ArrayScope-$VERSION-macos-$ARCH.dmg"
rm -f "$OUTPUT"
hdiutil create -volname "ArrayScope" -srcfolder "$STAGING" -ov -format UDZO "$OUTPUT"

echo "==> Done: $OUTPUT"
