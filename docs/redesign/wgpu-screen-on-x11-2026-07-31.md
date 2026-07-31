# The wgpu screen present path on X11 (2026-07-31)

`wgpu_present_method: AUTO` now resolves to the native swapchain on X
sessions as well as Wayland. This is the evidence that widening rests on,
and the list of what it still does not cover.

The reason to want it is the same one that justified the Wayland path:
bitmap presentation pays a GPU→CPU readback every frame — gate B measured
7.0 ms at 1300×650 and 26.0 ms at 4K — and the screen path skips it on any
platform. X sessions were getting bitmap unconditionally, not because the
recipe was known to fail there but because it had never been measured.

## 1. What changed

- `screen_present_unavailable_reason()` resolves a *native present target*
  instead of testing for one QPA: `wayland` via `QWaylandApplication`,
  `xcb` via `QX11Application`, and a named reason for anything else. The
  reason now names the platform it actually found, so an offscreen ring and
  a broken X session no longer produce the same message.
- `WgpuScreenCanvas.ensure_context()` builds its present-info from that
  target, which selects wgpu-py's `WGPUSurfaceSourceXlibWindow` for
  `platform: "x11"`. Nothing else in the canvas is platform-aware; pacing,
  Mailbox preference, resize coalescing and teardown are unchanged, and
  `wgpu_screen_swapchain_configured` now carries the platform so a screen
  timing can be attributed to one.
- `arrayscope/tools/headless_display.py` gained `xwayland=True` /
  `--qt-platform xcb`, which starts the managed compositor's own X server
  and points the child at it. It refuses to hand an xcb client an
  environment whose `DISPLAY` would resolve to the developer's session.

## 2. Evidence

All runs: managed headless Weston 15.0.1, Intel UHD (TGL GT1) via Mesa,
wgpu-py 0.31.1, PySide6/Qt 6.11.1, `set_instance_extras(backends=["Vulkan"])`.
The xcb arm talks X11 to that same compositor's XWayland. Machine load was
4.6–26 of 16 cores throughout (several sessions share this machine), so the
timing rows below are quoted as contrasts and the pixel rows — which are the
load-bearing ones — are counts, not times.

### 2a. The surface itself (`probe_native_x11.py`, ring-4 pinned)

Native child, top-level, and the production `createWindowContainer` shape
each presented **30/30 frames at status SuccessOptimal**, formats
`bgra8unorm`/`rgba8unorm`, present modes `immediate/mailbox/fifo/fiforelaxed`
— **Mailbox is available exactly as on Wayland**, which is what keeps the
blocking Fifo acquire off the GUI thread. Acquire p50 0.115–0.119 ms,
present p50 0.03–0.17 ms.

### 2b. Overlays reach the screen — the gate-B blocker is gone

Gate B measured the screen path under xcb as journey-clean but reproduced
the overlay-stacking failure: Qt cannot composite its own painting above a
native child window, so overlays were in the backing store and not on screen
(5,416 px vs 0). ArrayScope answered that by moving overlays *into* the wgpu
frame, keyed on the present method rather than the platform — which made X11
support plausible, not proven. Captured on real compositor output
(`x11_screen_evidence.py`, arm 1), pixels matching each overlay's own colour:

| pixels | X11 (xcb) | Wayland | control, overlays removed |
|---|---|---|---|
| ROI outline | 1,923 | 1,923 | 0 |
| floating chip | 8,172 | 8,172 | 0 |
| tile-truth label border | 626 | 626 | 0 |
| tile-truth label ink | 6 | 8 | 0 |

**The X11 counts are identical to Wayland's**, and collapse to zero in the
control, so they are the overlays and not the scene. Reproduced twice per
platform, byte-identical each time.

Qt's own `grab()` of the same window is the gate-B contrast inverted: ROI
outline **0** and label border **0** in the backing store while both are on
screen — they exist only inside the wgpu frame. The chip reads 8,172 in
*both*, which is correct and not a leak: a chip deliberately stays an
ordinary Qt widget and only its *pixels* are additionally drawn in the frame
(`_prepare_display_overlay_widget`), so Qt's own render still contains it.

### 2c. The Wayland resize trade has no X11 analogue

`_on_surface_resized` coalesces resizes into the next paced frame, and its
KNOWN TRADE is stale-buffer *spill*: a Wayland subsurface is not clipped by
its parent, so a shrinking window leaves an old, larger buffer hanging
outside it. Rather than race a ~17 ms transient with a ~100 ms screenshot,
arm 2 holds the state still — a production `WgpuScreenCanvas` driven 300 px
past its 420×320 top-level, with a flat clear colour, then captured:

| | X11 (xcb) | Wayland |
|---|---|---|
| clear-colour pixels, canvas within window | 134,400 | 134,400 |
| clear-colour pixels, canvas 300 px oversized | 134,400 | 446,400 |
| **pixels outside the window** | **0** | **312,000** |

134,400 is exactly the window's area and 446,400 is exactly the oversized
canvas: on Wayland the whole canvas reached the screen, on X11 the server
clipped it to its parent and not one pixel escaped. The oracle demonstrably
can fail — it fired on Wayland.

Resize *timing* in the real view, 60-step scripted shrink/grow sweep:

| | X11 (xcb) | Wayland |
|---|---|---|
| resize events → draws | 58 → 59 | 58 → 59 |
| stale window, mean | 22.7 / 22.8 ms | 8.1 / 8.4 ms |
| stale window, max | 33.3 / 29.5 ms | 15.1 / 12.0 ms |

Coalescing behaves identically (one draw per event, no storm). X11's stale
window is ~2.7× longer, but because it cannot spill, what it costs there is
only old-size content inside the window for one to two frames.

### 2d. The ring-4 screen-present oracles under xcb

`tests/gpu_interaction/test_wgpu_screen_present.py`, whole file, both QPAs
in the same session: **9 passed, 1 failed — identically on both.** Input
transparency (the `WindowTransparentForInput` contract, an XShape input
region on X11 rather than an empty Wayland input region), close-cancels-drag,
draw-ack on the real present edge, display-rate pacing, pacing override, and
resize coalescing all pass under xcb.

The one failure, `test_commit_presents_on_the_swapchain_and_drains_the_draw_gate`,
asserts `wgpu_screen_acquire_ms_last < 10.0` and is **inherited, not caused
here**: on unmodified `main` under Wayland it fails the same assert at 24.1
and 30.5 ms, against 16.5–26.0 ms on this branch under xcb and 14.6–15.8 ms
under Wayland. The harness already classifies it as an inherited red.

## 3. What this does NOT cover

- **A native X.org session.** All X evidence here is XWayland. That is the
  right server for anything decided by the X11 *protocol* — window
  hierarchy, clipping, stacking, input shapes, which is what §2b–2c turn on
  — but not for anything decided by the DDX or by real modesetting. Running
  Xorg from this session is blocked by `Xorg.wrap` ("Only console users are
  allowed to run the X server"), and changing that is a system security
  setting, not this chip's call. To close it, from a console on a real X
  session:

  ```bash
  ARRAYSCOPE_GPU_TESTS=1 tools/env pytest tests/gpu_interaction/test_wgpu_screen_present.py tests/gpu_interaction/test_wgpu_native_x11_pin.py -n 0 -q
  ```

- **Compositing vs non-compositing window managers.** Weston's XWayland is
  one WM's behaviour. A compositing X11 WM redirects the child window and
  composites it GPU-side (still no CPU readback), but that has not been
  measured, and neither has a bare non-compositing WM.
- **The readback saving itself, on X11.** The 7.0/26.0 ms bitmap readback
  figures are gate B's, on Wayland. The screen path avoids the readback by
  construction on either platform, but no before/after frame-cost contrast
  was run on an X session.
- **Exact-window capture under xcb.** `exact_window` disables Qt's
  *client-side* decoration, which is Wayland-only; an X11 window's frame is
  the WM's. Screen-evidence activities that need capture == window still owe
  an X treatment.
- **VRR and multi-monitor**, unchanged from the 2026-07-21 dossier.
- **Hidden-ROI / high-DPI scaling on X11** (`devicePixelRatio` under Xft
  scaling) was not exercised.

## 4. Reproducing

```bash
tools/env python -m arrayscope.tools.headless_display --qt-platform xcb --width 1600 --height 1200 -- python experiments/wgpu_gate_b/x11_screen_evidence.py out.json
```

Drop `--qt-platform xcb` for the Wayland arm. The captures land next to
`out.json`; they are evidence, not artifacts to commit.
