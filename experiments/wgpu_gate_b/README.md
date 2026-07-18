# wgpu renderer gate B

Experiment harness for the tiered wgpu plan in
[`docs/proposals/wgpu-renderer-experiment.md`](../../docs/proposals/wgpu-renderer-experiment.md).
This directory is not an ArrayScope renderer backend; live rendering paths
are unchanged.

Environment: conda env `arrayscope` with `wgpu 0.31.1` and
`rendercanvas 2.7.0` installed from wheels (no compiler, no source pin
needed — versions are recorded in the evidence JSON of each run).

Scripts (run with `XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0
QT_QPA_PLATFORM=wayland` unless a case says otherwise):

- `probe_native_wayland.py` — Tier 0: native-Wayland screen-presentation
  probe (paint-less native child + top-level, Vulkan-only instance).
- `run_gate_b.py` — Tier 1: presentation × overlay × journey matrix.
- `virtual_tensor.py` — Tier 2/3: page-pool virtual tensor slice with
  zero-upload oracles, compute histogram + LOD reduction.
- `upload_bench.py` — Tier 4: upload-path microbenchmarks + completion
  contract.

Machine-readable evidence is committed under
`tests/artifacts/wgpu-gate-b/`.
