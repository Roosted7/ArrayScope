# 0047 — Capability-probed automatic image backend selection

**Status:** Superseded in backend choice by
[ADR 0061](0061-retire-vispy-rendering-backend.md) (2026-07-27). The X5a
measurements below remain historical evidence; AUTO now selects WGPU when its
device gate passes and otherwise PyQtGraph.

## Context

ADR 0046 required backend defaults to be earned from real-device traces, not
from theoretical throughput. The first X5a pass
([Linux/Wayland reference traces](../reviews/x5a-hardware-telemetry-linux-wayland.md))
measured both backends on a live Wayland session across four configurations
(Intel iGPU and NVIDIA dGPU, native Wayland and XWayland), through the
presented-frame micro/stress matrix, the production-window montage workflow,
and 60 Hz scroll interaction.

The traces are unambiguous for this platform class:

- VisPy presented first frames earlier in **every** measured scenario
  (1.4–13×), on both GPUs and both Qt platforms.
- The gap is structural where it matters most in real use: window/level
  changes are shader-uniform updates on VisPy (resident textures are reused,
  nothing is re-uploaded), versus CPU re-windowing of every visible tile on
  PyQtGraph. On a 272-tile FFT montage a level drag settles in ~0.26 s on
  VisPy versus ~8 s on PyQtGraph (and, before this pass's fixes, PyQtGraph
  did not converge at all).
- Production-scale progressive commits (272 tiles) are 3.5–4.4× faster to
  first frame on VisPy after the O(n²) fixes from the same pass.
- PyQtGraph keeps an edge only in CPU submission cost for tiny single-tile
  commits, and it remains the only backend that works without a usable
  hardware GL context.

What the traces do **not** cover: Windows, macOS, non-Mesa/NVIDIA Linux
stacks, long-session context loss, and multi-window residency pressure.

## Decision

Add `ImageRenderingBackendChoice.AUTO` and make it the settings default.
`AUTO` resolves once per process in the image-view factory:

1. Platform without reference traces (anything but Linux) → PyQtGraph.
2. Offscreen/minimal Qt platform → PyQtGraph.
3. Probe a short-lived GL context and read the renderer string through
   `query_gpu_device_limits`; probe failure or a software renderer
   (llvmpipe/softpipe/swrast) → PyQtGraph.
4. Otherwise (real hardware GL on Linux) → VisPy.

The explicit `pyqtgraph` and `vispy` settings keep their exact previous
meaning; `AUTO` is a resolution rule in front of them, and the resolution and
its reason are reported through the status notifier for inspectability.

## Consequences

- Users on the measured platform class get the backend that the traces show
  is faster in real use, most visibly for window/level interaction on
  montages, without losing the explicit override.
- Headless CI, software-GL, and unproven platforms keep the PyQtGraph
  behavior they had before; the offscreen guard means the test suite is
  unaffected by the new default.
- Extending `AUTO` to another platform requires publishing traces for it
  (the X5e matrix), then widening the platform gate — not editing defaults
  on intuition.
- The probe adds one short-lived GL context creation to the first
  image-view construction in AUTO mode (~tens of ms, cached per process).

## Alternatives considered

### Flip the default to VisPy outright

Rejected: the traces cover one platform class. ADR 0046 explicitly forbids
promoting VisPy by theory or partial evidence, and software-GL environments
are known to be unstable with VisPy.

### Keep PyQtGraph as default and leave selection manual

Rejected: the measured gap in the dominant interactive workflow (window/level
on tiled scenes) is 30×; leaving that on the table for users whose hardware
was measured contradicts the purpose of the evidence gate.

### Decide per-frame/per-scene instead of per-process

Deferred, not rejected: ADR 0046's physical strategy policy (X5d) will choose
singleton/tiled mechanics per frame beneath one semantic surface. Backend
choice stays per-process because the view widget, GL context, and residency
caches are per-window state.

## Related records

- ADR 0046: evidence-first performance strategy (parent gate).
- [X5a Linux/Wayland reference traces](../reviews/x5a-hardware-telemetry-linux-wayland.md).
