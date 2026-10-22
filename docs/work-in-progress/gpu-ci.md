# GPU CI investigation — paused 2026-10-22

Branch `dev-gpu-ci`, based on `e8c4ad82d62aae676831cb015b492c200adc511c`.
This branch records a reproducible investigation; it changes no production
code, test assertions, tolerances, or GPU availability policy.

## Evidence

The failing CI run is
[39844643347](https://github.com/Roosted7/ArrayScope/actions/runs/39844643347),
job [120054892724](https://github.com/Roosted7/ArrayScope/actions/runs/39844643347/job/120054892724).
The downloaded job log remains at
`/workspace/work/arrayscope-ci/work/ci-logs/120054892724.log`.

The development container has no `/dev/dri` or system Vulkan ICD. A
workspace-local software Vulkan driver works with the locked checkout
environment: Debian 13, Python 3.13.5, wgpu 0.31.1, Mesa 25.0.7, llvmpipe
LLVM 19.1.7. WGPU reports a CPU Vulkan adapter, BC texture compression, and
2048 maximum texture-array layers. This provides software-backend correctness
evidence, with no hardware or GPU performance claim.

A single serial run of four exact nodes completed in 5.71 seconds:

| Node, under `tests/gpu/` | Result |
| --- | --- |
| `test_wgpu_command_protocol.py::test_physical_truth_full_residency` | FAIL: 23,696 pixels exceed existing tolerance 2; maximum channel difference 212 |
| `test_wgpu_command_protocol.py::test_histogram_exact_over_all_l0_pages` | PASS: exact 1,048,576-sample histogram and existing per-bin NumPy equality |
| `test_wgpu_compressed_render.py::test_complex_bc5_phase_error_confined_to_near_black_pixels` | FAIL: 24 large phase-color differences outside the existing low-magnitude mask, matching CI |
| `test_wgpu_compressed_levels.py::test_complex_bc5_histogram_bounds_match_raw_within_tolerance` | FAIL: upper-bound difference 0.00865924358 exceeds existing 0.005 × 1.34507894516, matching CI |

The local log is `work/gpu-baseline.log`; driver assets and logs are deliberately
untracked. No broader GPU suite was run before the requested winddown.

## What is established, and what remains uncertain

The CI 983,040-sample histograms represent exactly one absent 256 × 256 L0
page, but do not establish asynchronous publication or upload deferral.
`Scene` asserts that its initial synchronous executor submission reports all
20 uploads. The isolated histogram above sees all 16 native pages.

Source review identifies a concrete failure cascade: the module-scoped `Scene`
is shared; `test_evicted_page_falls_back_to_pinned_ancestor_then_refills` evicts
key `(LOD 0, x 1, y 1)`, asserts fallback pixels, and only then restores the
page. Its CI pixel assertion fails before that restoration. Later histogram
tests consequently inherit 15 pages. A second independent read review agrees.
The missing page is therefore consistent with test-state contamination after
the earlier pixel failure. This mechanism has not yet been pinned with a
dedicated ordered reproduction. Reviewed layer-zero handling is valid:
missing flat entries use -1 and shader checks accept layer zero.

The underlying raw framebuffer mismatch is unresolved. Next establish exact
per-page texture readback equality with the canonical source, page-table
completeness, and submission completion before comparing source-coordinate
selection and shader mapping with the independent CPU mirror. The noisy source
and 512-source-pixel to 768-framebuffer-pixel geometry make float-coordinate
boundary behavior a hypothesis to check, not an established cause.

The two BC5 failures reproduce independently. Distinguish actual decoded-texture
error from reference-coordinate errors before changing the codec or oracle.
The phase test currently classifies framebuffer magnitude using uncentered
coordinates, while rendering uses pixel centers. The complex admission gate
currently checks magnitude PSNR; computed phase/max-error metrics are not used
to reject a page. Neither observation alone proves the right fix. Preserve
the existing tolerances and prove the relevant scientific/display contract.

## Reproduce without changing the host

Create an isolated checkout and its own environment, then obtain the official
Debian package below. All extracted files stay under checkout `work/`.

```sh
UV_CACHE_DIR=/workspace/work/uv-cache uv sync --frozen --python /usr/bin/python3.13
mkdir -p work/mesa work/xdg
chmod 700 work/xdg
curl -fL https://deb.debian.org/debian/pool/main/m/mesa/mesa-vulkan-drivers_25.0.7-2+deb13u1_amd64.deb \
  -o work/mesa/mesa-vulkan-drivers_25.0.7-2+deb13u1_amd64.deb
dpkg-deb -x work/mesa/mesa-vulkan-drivers_25.0.7-2+deb13u1_amd64.deb work/mesa/root
```

The downloaded package SHA256 is
`caf2f7d296b7522efe74e40a74bce762a3ce84c33b53b5e54e1248ac2e0a13a5`.
The container already provides the required Vulkan loader, LLVM 19, and driver
shared-library dependencies. `ldd` found no missing dependencies. Configure
these variables with absolute paths to the isolated checkout:

```sh
export VK_DRIVER_FILES="$PWD/work/mesa/root/usr/share/vulkan/icd.d/lvp_icd.json"
export LD_LIBRARY_PATH="$PWD/work/mesa/root/usr/lib/x86_64-linux-gnu"
export XDG_RUNTIME_DIR="$PWD/work/xdg"
export MESA_SHADER_CACHE_DIR="$PWD/work/mesa-cache"
export QT_QPA_PLATFORM=offscreen
export TESTMON_DATAFILE="$PWD/work/gpu-testmon"
tools/env pytest -n0 --testmon-noselect \
  tests/gpu/test_wgpu_command_protocol.py::test_physical_truth_full_residency \
  tests/gpu/test_wgpu_command_protocol.py::test_histogram_exact_over_all_l0_pages \
  tests/gpu/test_wgpu_compressed_render.py::test_complex_bc5_phase_error_confined_to_near_black_pixels \
  tests/gpu/test_wgpu_compressed_levels.py::test_complex_bc5_histogram_bounds_match_raw_within_tolerance \
  >work/gpu-baseline.log 2>&1
```

The deliberate map bypass recorded this new isolated GPU environment; it was
used for the four reproduction nodes, not to broaden selection or hide reds.
No system package installation, home-directory writes, or source-environment
pointer sharing was needed. Resume with the smallest canonical-readback and
ordered-eviction probes above; do not blanket-disable the GPU tests.
