# 0008 — Live image profiles

ArrayScope supports opt-in live profiles from the image view. When `Live profile`
is enabled, a draggable image marker with vertical/horizontal crosshair lines
and a center handle is converted into a temporary `ViewState` for
`slice_engine.make_line(...)` through the existing `OperationEvaluator`.

Mouse `x` maps to the secondary image axis and mouse `y` maps to the primary
image axis, matching the existing image display orientation. If the selected
line axis is one of the image axes, that axis remains unsliced and the other
image axis is fixed at the hover coordinate. Non-image, non-line dimensions keep
their current slice indices.

The raw mouse handler does not perform array work and does not drive live
profiles. It only updates hover text. Crosshair movement records the latest
image-space position, and a short single-shot timer performs the profile query.
The first implementation still relies on the current evaluator cache, so
operation stacks that materialize derived arrays may pay that materialization
cost once per stack before cached line queries are reused.
