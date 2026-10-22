# Current state

**Snapshot: `dev`, 2026-10-22.** ArrayScope is a Python/Qt scientific array
viewer with maintained WGPU and PyQtGraph backends. WGPU serves GPU rendering;
PyQtGraph serves CPU and remote workflows. Historical hardware measurements
remain evidence for their named revisions, not certification of this branch.

## Product capabilities

- A home screen offers recent files, DICOM folders, and locally generated
  labelled examples. Searchable HDF5/NPZ/MAT browsing reads metadata before
  selecting an array.
- The scientific core supports n-dimensional slicing, montage views, complex
  values, ROI/profile inspection, linked windows, and ordered operations.
  Recipes travel through files and the clipboard. Bounded operation Undo/Redo
  restores edits on the current source, including compatible rank handoffs.
- Selected-array export chooses ranges before evaluation, reports planned
  costs, runs on the window kernel, and supports progress/cancellation with
  atomic publication. NPY keeps exact values; NPZ can keep the original recipe
  and separate selection/source provenance.
- [Terminal recipe replay](recipe-replay.md) runs built-in operations without
  Qt, over read-only NPY or explicitly selected HDF5 data. Dry-run reads
  metadata rather than array values and prints a versioned JSON plan.
- [HDF5 loading](hdf5-data.md) shares immutable metadata and lazy region sources
  between GUI, Python, and terminal workflows, with per-read file ownership,
  working-memory estimates, change detection, and process transport.

## Architecture

One execution kernel owns tasks and completion. The staged operation planner,
region evaluator, source reader, and atomic writer are shared across interfaces.
Rendering uses the backend-neutral frame/presentation contracts; caches and
physical backend storage retain separate budgets and identities.

## Current limits and next work

Direction and acceptance gates live in [the queue](queue.md). The next large-data
slice is bounded incremental export: current selected export still allocates
its selected result, and FFT/opaque stages can require larger intermediates.
Estimated peak memory is not a process-RSS cap. Plugin replay, indirect HDF5
storage, and lazy compound subarray decoding are outside the terminal contract.
File identity checks detect ordinary replacement and changes; they do not
provide cryptographic provenance or snapshot isolation against external writers.

Current validation includes numerical/resource tests, real subprocesses, and
offscreen Qt interactions with inspected dark/light artifacts. This environment
has no physical GPU device; it cannot establish compositor/pixel acceptance or
new GPU performance claims. Python 3.13 is available here; the wider declared
Python/platform matrix requires its corresponding CI/runtime environments.

The rendering pipeline remains complex, and historical callback/interaction
performance targets require fresh hardware measurements. Native code should
address a measured boundary with correct semantics and viable packaging; a
whole-application rewrite has no demonstrated performance justification.
