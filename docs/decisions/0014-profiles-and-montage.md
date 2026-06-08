# 0014 — Multi-profile and montage inspection

ArrayScope supports multiple profile axes by keeping `ArrayScopeWindow.profile_axes` as the active
axis tuple and plotting each evaluated profile as a separate curve with a legend in the Profile dock.
The existing single-axis combo remains a compatibility path that selects one active profile axis;
compact dimension-chip `P` buttons toggle additional axes. Live profile mode is controlled from the
image/context menus rather than a persistent toolbar button.

Complex line profiles preserve raw complex values when the global channel mode is `complex`. The
Profile dock then applies its own profile mode: magnitude, magnitude plus phase strip, phase, real,
or imaginary. This keeps image display channel behavior separate from profile inspection behavior and
allows the phase color strip to be generated from the same complex profile samples.

Montage is activated by entering range text in a non-image dimension slice field, for example `:` or
`0:2:100`. Three-part ranges use the user-facing `start:step:stop` convention and are clamped to the
axis size. Range text on image X/Y axes creates a display range instead of a montage, so image axes can
be sub-sampled directly. Montage rendering evaluates visible tiles through the existing image snapshot
path. Phase 4d added `MontagePlan` and tile-level cache keys so large montage ranges do not require
one giant all-tiles collage allocation. The Qt commit path assembles the currently loaded bounded tile
set into one stable display image, while the pure plan/cache contract keeps tile identity and source
indices separate from that display choice. When no explicit column count is stored, the montage helper
chooses the column count that maximizes tile size within the current 2D viewport. Montage provides a
dedicated histogram/ROI source with `NaN` in tile gaps for the committed tile set, so ROI statistics
and histograms ignore inter-tile spacing. Full session-level montage configuration can be expanded
later if needed.
