# Mission

Build the quickest trustworthy way for Python users to understand an n-dimensional scientific array.

## Primary users

- NumPy/SciPy users who need to inspect an unfamiliar result without writing plotting boilerplate.
- MRI and reconstruction developers moving repeatedly between image space, k-space, channels, coils, echoes, time, and derived dimensions.
- Researchers who value MATLAB ArrayShow’s dimension-first immediacy but want Python integration, reversible operations, and better control over large-data behavior.

## Product promise

A call such as `arrayscope(data)` should produce useful pixels quickly. Dimension selection, slicing, windowing, value inspection, profiles, and common transforms should feel direct. When a request is expensive, ArrayScope should remain responsive, preserve the last valid frame, expose progress/reasons, and refuse unsafe work clearly rather than freezing or silently allocating without bound.

## Principles

1. **Dimension-first.** Axes and selections are first-class, not hidden inside a generic layer model.
2. **Non-destructive.** Operations form a reversible document pipeline; the source array is not casually replaced.
3. **Progressive but semantically stable.** Partial pixels may arrive over time, but geometry, levels, values, and frame identity must not contradict one another.
4. **Bounded by default.** CPU work, GUI callbacks, caches, canvas sizes, GPU residency, and speculation all require explicit limits.
5. **Small surface, deep inspection.** Prefer coherent inspection workflows over accumulating unrelated analysis modes.
6. **Backend-independent meaning.** Rendering libraries may differ in mechanics, not in what an ArrayScope frame, ROI, level, or pointer target means.
7. **Evidence over folklore.** Performance decisions use traces, counters, request-to-frame latency, event-loop delay, memory measurements, and real-hardware checks.

## Non-goals

ArrayScope is not intended to become:

- a full napari replacement or general layer/plugin platform;
- a MATLAB clone or workspace manager;
- a diagnostic medical/DICOM workstation;
- a general registration, segmentation, or annotation suite;
- a remote collaboration/server product;
- a dashboard whose controls crowd out the array.

Capabilities that overlap those categories are acceptable only when they directly improve array inspection and preserve a simple mental model.

## Success

ArrayScope succeeds when a new user can open an array, identify its important dimensions, inspect values and structure, try a reversible transform, and understand any delay or limitation without reading internal documentation. It also succeeds when the same interaction remains bounded on arrays too large for eager full-frame processing.
