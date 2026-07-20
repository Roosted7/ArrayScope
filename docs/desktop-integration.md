# Desktop integration and asynchronous file loading

Status: implemented (2026-07). Code: `arrayscope/desktop/`, `arrayscope/app/open_flow.py`, `arrayscope/io/progressive.py`.

## Asynchronous, progressive file loading

Opening a file from the CLI (or a desktop shell association) never blocks
on file I/O:

1. **Immediate feedback.** `arrayscope <file>` shows a loading window
   *before any file I/O* — file name, stage ("Reading header…", "Reading
   data…", "Converting…"), a progress bar, byte counts, and a Cancel
   button. Formats whose libraries only offer a monolithic read (NIfTI,
   single DICOM, text, dcm2niix conversion) show an indeterminate bar with
   stage messages.
2. **Viewer as soon as possible.** Streaming-capable formats — `.npy`,
   BART `.cfl`, Philips `.REC` — pre-allocate the destination array after a
   cheap header probe and hand it to the viewer immediately. The window
   opens while the file streams in; a status-bar widget shows *"Loading…
   N% available"*. Unread regions render as zeros; the view refreshes on a
   throttle (`notify_data_changed()`, which bumps the document revision so
   no stale caches survive) and re-windows from the evidence available at
   each publication, including completion.

   The viewer does not receive the mutable destination `ndarray` directly.
   A synchronized progressive source owns that buffer: loader writes and
   bounded `read_region()` snapshots share one lock, and every evaluation
   read gets a detached array. This is the publication boundary that keeps
   partially written bytes out of pixels, levels, histograms, and caches.
3. **Cancellation.** Cancel in the loading window aborts the open; the ✕
   in the status-bar widget stops a stream but keeps the viewer open on
   the partial data.
4. **Large files** still take the lazy memory-mapped seam (ADR 0049) and
   open instantly; `--mmap` (wrapper handoff) is unchanged.

Layers:

- `arrayscope/io/progressive.py` — Qt-free chunked readers
  (`load_npy_progressive`, `load_cfl_progressive`), `LoadProgress`,
  `StreamingProbe`, `LoadCancelled`. `load_path(...)` accepts
  `progress=`, `cancel=`, `on_streaming_probe=`.
- `arrayscope/app/open_flow.py` — `FileOpenSession` runs the reader in a
  thread and marshals events onto the GUI thread via queued signals;
  `open_any_path()` also routes multi-dataset containers to the existing
  selectors. `arrayscope/ui/loading_window.py` and
  `arrayscope/ui/load_status.py` are the two progress surfaces.

## Desktop-shell registration

`arrayscope --install-desktop` / `--uninstall-desktop` (per-user,
idempotent, no elevation):

| Platform | What gets installed |
|---|---|
| Linux | XDG desktop entry (`~/.local/share/applications/arrayscope.desktop`), shared-mime-info package defining the MIME types ArrayScope owns (`application/x-numpy-array`, `-numpy-archive`, `-bart-cfl`, `-philips-rec`, `-nifti`, `-matlab-data`), hicolor icons (SVG + 8 PNG sizes), database refreshes, and `xdg-mime` defaults for owned types only |
| Windows | HKCU ProgIDs (`ArrayScope.<key>`) with `DefaultIcon` and open commands, extension associations (default only for owned types, `OpenWithProgids` otherwise), Start Menu shortcut, `SHChangeNotify` |
| macOS | `~/Applications/ArrayScope.app` bundle (Info.plist with `CFBundleDocumentTypes`, launcher script into the current Python environment, `.icns` via `iconutil`), LaunchServices registration |

Shared catalogue: `arrayscope/desktop/filetypes.py`. A type is **owned**
when no widely deployed application owns the format (we may claim default
handler); DICOM (`application/dicom`) and HDF5 (`application/x-hdf`) only
get "Open with" entries.

The launch command is resolved at install time (`launcher_cmd.py`):
the `arrayscope` console script next to `sys.executable` if present, else
`<python> -m arrayscope` — so associations work from login shells without
conda/venv activation.

Related entry points:

- `arrayscope` with no arguments → launcher window (open dialog +
  drag-and-drop). This also serves macOS launches with no document.
- Dropping supported files on a viewer window opens them in new windows.
- macOS Finder document opens arrive as `QFileOpenEvent` and are handled
  by an application-level event filter in `open_flow.py`.

Icons: `arrayscope/resources/icons/` (packaged). Regenerate PNG/ICO with
`python tools/generate_icons.py`; keep `arrayscope.svg` in sync by hand.
