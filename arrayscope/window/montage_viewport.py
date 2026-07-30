"""Qt-free identity and coverage policy for montage viewport updates."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from arrayscope.core.roi import RoiGeometry, RoiKind, RoiSelection
from arrayscope.display.model.tile_priority import TilePriorityContext, prioritize_tiles
from arrayscope.display.montage import montage_rect_for_viewport, optimal_montage_columns
from arrayscope.display.viewport import view_ranges_near

MIN_VIEW_SPAN = 1e-9


def montage_priority_focus(owner, view_range) -> tuple[float, float] | None:
    """Return the one semantic attention point used by every render path.

    Backends may report pointer mechanics, but they do not own scheduling
    meaning.  A hover focus is valid only for the current committed frame;
    otherwise priority starts at the montage tile nearest viewport center.
    """

    focus = getattr(owner, "_last_image_hover_focus", None)
    if focus is not None:
        try:
            frame = getattr(owner.win, "_committed_display_frame", None)
            if getattr(owner, "_last_image_hover_focus_frame_key", None) != getattr(
                frame, "key", None
            ):
                raise ValueError("stored hover focus belongs to an older committed frame")
            x = float(focus[0])
            y = float(focus[1])
            x_range, y_range = view_range
            x0, x1 = sorted((float(x_range[0]), float(x_range[1])))
            y0, y1 = sorted((float(y_range[0]), float(y_range[1])))
            if x < x0 or x > x1 or y < y0 or y > y1:
                raise ValueError("stored hover focus is outside the current viewport")
            return (x, y)
        except Exception:
            pass
    try:
        plan = getattr(getattr(owner, "_frame_session", None), "plan", None)
        if plan is not None:
            focus = _nearest_montage_tile_center(plan, view_range)
            if focus is not None:
                return focus
        return _view_range_center(view_range)
    except Exception:
        return None


def _view_range_center(view_range) -> tuple[float, float] | None:
    try:
        x_range, y_range = view_range
        return (
            (float(x_range[0]) + float(x_range[1])) * 0.5,
            (float(y_range[0]) + float(y_range[1])) * 0.5,
        )
    except Exception:
        return None


def _nearest_montage_tile_center(plan, view_range) -> tuple[float, float] | None:
    center = _view_range_center(view_range)
    if center is None:
        return None
    tiles = getattr(plan, "tiles", ())
    if not tiles:
        return None
    try:
        tile_height, tile_width = (int(value) for value in plan.tile_shape[:2])
        gap = max(0, int(plan.gap))
        columns = max(1, int(plan.columns))
        rows = max(1, int(plan.rows))
        count = len(tiles)
        stride_x = max(1, tile_width + gap)
        stride_y = max(1, tile_height + gap)
        col = round((float(center[0]) - float(tile_width) * 0.5) / float(stride_x))
        row = round((float(center[1]) - float(tile_height) * 0.5) / float(stride_y))
        row = max(0, min(rows - 1, row))
        max_col = min(columns - 1, count - row * columns - 1)
        if max_col < 0:
            row = max(0, min((count - 1) // columns, rows - 1))
            max_col = min(columns - 1, count - row * columns - 1)
        col = max(0, min(max_col, col))
        tile = tiles[row * columns + col]
        return (
            float(tile.x0) + float(tile.width) * 0.5,
            float(tile.y0) + float(tile.height) * 0.5,
        )
    except Exception:
        return None


@dataclass(frozen=True)
class MontageViewportPlan:
    """Stable montage layout plus transient viewport scheduling state."""

    axis: int | None
    all_indices: tuple[int, ...]
    viewport_shape: tuple[int, int]
    tile_shape: tuple[int, int]
    plan: object
    view_range: object
    shader_display: bool
    persistent_tile_residency: bool
    priority_focus: tuple[float, float] | None = None

    def candidate_tiles(self, *, margin_tiles: int = 0, prioritize: bool = False):
        rect = montage_rect_for_viewport(
            self.plan,
            view_range=self.view_range,
            viewport_shape=self.viewport_shape,
        )
        tiles = self.plan.tiles_intersecting(
            ((rect[0], rect[2]), (rect[1], rect[3])),
            margin_tiles=max(0, int(margin_tiles)),
        )
        if prioritize:
            return prioritize_tiles(
                tiles,
                context=TilePriorityContext.from_tiles(
                    view_range=((rect[0], rect[2]), (rect[1], rect[3])),
                    focus=self.priority_focus,
                ),
            )
        return tiles

    def prioritize_tiles(self, tiles):
        rect = montage_rect_for_viewport(
            self.plan,
            view_range=self.view_range,
            viewport_shape=self.viewport_shape,
        )
        return prioritize_tiles(
            tiles,
            context=TilePriorityContext.from_tiles(
                view_range=((rect[0], rect[2]), (rect[1], rect[3])),
                focus=self.priority_focus,
            ),
        )


@dataclass(frozen=True)
class MontageViewportRetargetPolicy:
    """How a tiled backend handles viewport-only montage changes."""

    enabled: bool
    coverage_margin_tiles: int = 0
    near_margin_tiles: int = 0
    update_delay_ms: int = 120


@dataclass(frozen=True)
class MontageViewportReflow:
    """Pure viewport/layout reflow decision for a montage session update."""

    viewport_plan: MontageViewportPlan
    view_range_to_apply: tuple[tuple[float, float], tuple[float, float]] | None = None
    last_auto_view_range: tuple[tuple[float, float], tuple[float, float]] | None = None


@dataclass(frozen=True)
class MontageViewportIntent:
    """Canonical auto/manual facts consumed by montage viewport planning."""

    fit_locked: bool = False
    auto_active: bool = False

    @property
    def auto_like(self) -> bool:
        return bool(self.fit_locked or self.auto_active)


def montage_viewport_intent(viewport_controller, view_range) -> MontageViewportIntent:
    """Return semantic viewport intent without coupling callers to controller internals."""

    if viewport_controller is None:
        return MontageViewportIntent(auto_active=True)
    fit_locked = bool(viewport_controller.is_fit_locked())
    auto_active_fn = getattr(viewport_controller, "is_auto_active", None)
    auto_active = bool(auto_active_fn() if callable(auto_active_fn) else False)
    return MontageViewportIntent(
        fit_locked=fit_locked,
        auto_active=auto_active,
    )


def effective_montage_columns(
    count: int,
    tile_shape: tuple[int, int],
    viewport_shape: tuple[int, int],
    *,
    requested_columns: int | None,
    fit_locked: bool = False,
    auto_active: bool = False,
    latched_columns: int | None = None,
) -> int | None:
    """Choose applied montage columns without rewriting semantic view state.

    Automatic layout is recomputed from the viewport aspect whenever Fit owns
    the camera or the viewport is explicitly auto-owned -- those poses are
    meant to re-flow as the window changes.

    A *manual* pose (the user panned/zoomed) must NOT re-flow on resize: doing
    so visibly rearranges and rescales the tiles even though the user never
    asked for a different layout.  It keeps an explicit ``requested_columns``
    if one was pinned, else the ``latched_columns`` carried over from the last
    committed layout, so a manual montage holds its column count across window
    resizes and only reveals more/less of the same grid.
    """

    count = max(0, int(count))
    if count < 1:
        return None
    tile_shape = (max(1, int(tile_shape[0])), max(1, int(tile_shape[1])))
    viewport_shape = (max(1, int(viewport_shape[0])), max(1, int(viewport_shape[1])))
    automatic = optimal_montage_columns(count, tile_shape, viewport_shape)
    if bool(fit_locked) or bool(auto_active):
        return automatic
    if requested_columns is not None:
        return max(1, min(int(requested_columns), count))
    if latched_columns is not None:
        return max(1, min(int(latched_columns), count))
    return automatic


def montage_grid_mostly_fits(view_range, display_shape, *, grid_fit_fraction: float = 0.8) -> bool:
    """Whether >= ``grid_fit_fraction`` of the whole grid fits the viewport.

    Pan-independent (measured from the view span vs the grid extent), this is
    the "zoomed out" test: true when the montage is small enough relative to
    the view that seeing more tiles matters more than tile-relative shifts.
    """

    try:
        (x0, x1), (y0, y1) = view_range
        view_w = abs(float(x1) - float(x0))
        view_h = abs(float(y1) - float(y0))
    except (TypeError, ValueError):
        return False
    grid_h = max(1, int(display_shape[0]))
    grid_w = max(1, int(display_shape[1]))
    fits_w = min(1.0, view_w / grid_w)
    fits_h = min(1.0, view_h / grid_h)
    return fits_w >= grid_fit_fraction and fits_h >= grid_fit_fraction


def montage_manual_reflow_decision(
    view_range,
    *,
    display_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    gap: int,
    columns: int,
    rows: int,
    count: int,
    grid_fit_fraction: float = 0.8,
    moving_visible_fraction: float = 0.5,
) -> bool:
    """Should a *manual* montage RE-FLOW (recompute columns) on a resize?

    A manual (panned/zoomed) montage normally holds its committed column
    layout so the tiles do not rearrange under the user.  But at the two ends
    of the zoom range re-flowing is worth more than holding position, so this
    returns ``True`` (re-flow) when either applies -- and ``False`` (hold the
    layout) in the disorienting middle:

    * **Zoomed out** -- at least ``grid_fit_fraction`` (80%) of the whole grid
      fits in the viewport at this zoom, regardless of pan.  Seeing more tiles
      at once outweighs tile-relative shifts, so re-pack.
    * **Zoomed deep into one tile** -- every tile that a re-flow would move is
      at most ``moving_visible_fraction`` (50%) visible, so the shift is not
      seen.  The tiles that move are the rows above/below the centre; the
      side (same-row) tiles hold their place UNLESS the centre sits against an
      L/R edge, where a side tile can wrap to another row and must count too.

    ``view_range`` is ``((x0, x1), (y0, y1))`` in the montage's display/world
    coordinates, where tile ``(row, col)`` sits at ``col*(tw+gap),
    row*(th+gap)``.  Auto/Fit poses re-flow unconditionally and never reach
    this function.
    """

    try:
        (x0, x1), (y0, y1) = view_range
        x0, x1 = sorted((float(x0), float(x1)))
        y0, y1 = sorted((float(y0), float(y1)))
    except (TypeError, ValueError):
        return False
    columns = max(1, int(columns))
    rows = max(1, int(rows))
    count = max(1, int(count))
    tile_h, tile_w = (max(1, int(tile_shape[0])), max(1, int(tile_shape[1])))
    gap = max(0, int(gap))

    # Criterion 1: the grid nearly fits the viewport -> zoomed out -> re-flow.
    if montage_grid_mostly_fits(
        ((x0, x1), (y0, y1)), display_shape, grid_fit_fraction=grid_fit_fraction
    ):
        return True

    # Criterion 2: zoomed so deep that every tile a re-flow would move is
    # barely visible -> re-flow (the shift is not seen).
    stride_x = tile_w + gap
    stride_y = tile_h + gap
    center_x = (x0 + x1) * 0.5
    center_y = (y0 + y1) * 0.5
    center_col = min(columns - 1, max(0, int(center_x // stride_x)))
    center_row = min(rows - 1, max(0, int(center_y // stride_y)))

    def _exists(row: int, col: int) -> bool:
        return 0 <= row < rows and 0 <= col < columns and (row * columns + col) < count

    def _visible_fraction(row: int, col: int) -> float:
        tx0 = col * stride_x
        ty0 = row * stride_y
        overlap_x = max(0.0, min(tx0 + tile_w, x1) - max(float(tx0), x0))
        overlap_y = max(0.0, min(ty0 + tile_h, y1) - max(float(ty0), y0))
        return (overlap_x * overlap_y) / float(tile_w * tile_h)

    moving = [
        (row, center_col) for row in (center_row - 1, center_row + 1) if _exists(row, center_col)
    ]
    # Near an L/R edge the same-row neighbours can wrap to another row on a
    # column change, so they move too and must be checked for visibility.
    near_edge = center_col <= 0 or center_col >= columns - 1
    if near_edge:
        moving.extend(
            (center_row, col)
            for col in (center_col - 1, center_col + 1)
            if _exists(center_row, col)
        )

    if not moving:
        # A single row/column has nothing that a re-flow would move; criterion
        # 2 cannot apply (criterion 1 already handled the zoomed-out case).
        return False
    return max(_visible_fraction(row, col) for row, col in moving) <= moving_visible_fraction


def montage_latched_columns_for_plan(committed_plan, view_range) -> int | None:
    """Column count a manual montage should hold across this update, or None.

    Returns the committed plan's column count so the layout is held, except at
    the zoom extremes where ``montage_manual_reflow_decision`` says to re-flow
    (then None, letting the automatic aspect-based count take over). None too
    when there is no committed plan/geometry to hold.
    """

    columns = getattr(committed_plan, "columns", None)
    rows = getattr(committed_plan, "rows", None)
    tile_shape = getattr(committed_plan, "tile_shape", None)
    display_shape = getattr(committed_plan, "display_shape", None)
    if committed_plan is None or view_range is None or columns is None or rows is None:
        return None
    if tile_shape is None or display_shape is None:
        return None
    tiles = getattr(committed_plan, "tiles", None)
    reflow = montage_manual_reflow_decision(
        view_range,
        display_shape=display_shape,
        tile_shape=tile_shape,
        gap=getattr(committed_plan, "gap", 0),
        columns=columns,
        rows=rows,
        count=len(tiles) if tiles is not None else int(rows) * int(columns),
    )
    return None if reflow else int(columns)


def square_montage_fit_view_range(
    plan, viewport_shape
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return a square-pixel fit range for the full applied montage layout."""

    display_height, display_width = tuple(int(value) for value in plan.display_shape[:2])
    viewport_height, viewport_width = _normalized_viewport_shape(viewport_shape)
    content_width = max(MIN_VIEW_SPAN, float(display_width))
    content_height = max(MIN_VIEW_SPAN, float(display_height))
    viewport_aspect = float(viewport_width) / float(viewport_height)
    content_aspect = content_width / content_height
    fitted_width = content_width
    fitted_height = content_height
    if viewport_aspect > content_aspect:
        fitted_width = fitted_height * viewport_aspect
    elif viewport_aspect < content_aspect:
        fitted_height = fitted_width / viewport_aspect
    center_x = content_width * 0.5
    center_y = content_height * 0.5
    return (
        (center_x - fitted_width * 0.5, center_x + fitted_width * 0.5),
        (center_y - fitted_height * 0.5, center_y + fitted_height * 0.5),
    )


def plan_full_view_range(plan) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the exact world bounds of an applied montage plan."""

    height, width = tuple(int(value) for value in plan.display_shape[:2])
    return ((0.0, float(max(1, width))), (0.0, float(max(1, height))))


def retarget_montage_viewport_plan(
    previous_plan,
    viewport_plan: MontageViewportPlan,
    previous_viewport_shape,
    *,
    fit_locked: bool = False,
    auto_active: bool = False,
    skip_remap: bool = False,
    focus: tuple[float, float] | None = None,
) -> MontageViewportReflow:
    """Retarget a montage viewport after resize/layout reflow.

    Manual views are translated only when a same-source layout reflow moves the
    source tile underneath the focus point.  They are never refit by resize or
    column changes.  Fit and genuinely near-auto views take the fit paths.
    """

    next_plan = viewport_plan.plan
    if previous_plan is None or next_plan is None:
        return MontageViewportReflow(viewport_plan)
    layout_changed = getattr(previous_plan, "geometry", None) != getattr(
        next_plan, "geometry", None
    )
    viewport_shape_changed = tuple(previous_viewport_shape or ()) != tuple(
        viewport_plan.viewport_shape
    )
    if not layout_changed and not viewport_shape_changed:
        return MontageViewportReflow(viewport_plan)
    if viewport_plan.view_range is None or bool(skip_remap):
        return MontageViewportReflow(viewport_plan)

    current_range = _normalized_view_range(viewport_plan.view_range)
    last_auto_view_range = None
    if bool(fit_locked):
        next_range = plan_full_view_range(next_plan)
    elif bool(auto_active):
        auto_range = square_montage_fit_view_range(next_plan, viewport_plan.viewport_shape)
        next_range = auto_range
        last_auto_view_range = next_range
    else:
        # A zoomed-out re-flow re-packs the grid; anchor its frame centre so
        # the montage stays put instead of chasing the tile under the focus.
        frame_center_anchor = montage_grid_mostly_fits(
            current_range, getattr(previous_plan, "display_shape", (1, 1))
        )
        next_range = _manual_montage_reflow_range(
            previous_plan,
            next_plan,
            current_range,
            previous_viewport_shape,
            viewport_plan.viewport_shape,
            focus=focus,
            frame_center_anchor=frame_center_anchor,
        )
        if next_range is None:
            return MontageViewportReflow(viewport_plan)

    retargeted = replace(viewport_plan, view_range=next_range)
    apply_range = (
        None if view_ranges_near(current_range, next_range, tolerance_fraction=1e-9) else next_range
    )
    return MontageViewportReflow(
        retargeted,
        view_range_to_apply=apply_range,
        last_auto_view_range=last_auto_view_range,
    )


def _manual_montage_reflow_range(
    previous_plan,
    next_plan,
    current_range,
    previous_viewport_shape,
    next_viewport_shape,
    *,
    focus: tuple[float, float] | None = None,
    frame_center_anchor: bool = False,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    return remap_montage_view_range(
        previous_plan,
        next_plan,
        current_range,
        previous_viewport_shape,
        next_viewport_shape,
        focus=focus,
        frame_center_anchor=frame_center_anchor,
    )


def remap_montage_view_range(
    previous_plan,
    next_plan,
    view_range,
    previous_viewport_shape,
    next_viewport_shape,
    *,
    focus: tuple[float, float] | None = None,
    frame_center_anchor: bool = False,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Transfer a manual view through a montage layout change.

    Screen zoom (world units per viewport pixel) is always preserved. What
    stays put on screen depends on ``frame_center_anchor``:

    * default (tile anchor) -- the source-local point under ``focus`` keeps its
      screen fraction, right when zoomed in on a tile;
    * ``frame_center_anchor`` -- the montage's whole frame CENTER keeps its
      screen fraction, so a zoomed-out re-flow re-packs in place instead of
      lurching to follow one tile that moved.
    """

    try:
        view_range = _normalized_view_range(view_range)
        x0, x1 = float(view_range[0][0]), float(view_range[0][1])
        y0, y1 = float(view_range[1][0]), float(view_range[1][1])
    except Exception:
        return None

    span_x = abs(x1 - x0)
    span_y = abs(y1 - y0)
    if span_x <= MIN_VIEW_SPAN or span_y <= MIN_VIEW_SPAN:
        return None
    next_span_x, next_span_y = _resize_preserving_screen_zoom_spans(
        span_x,
        span_y,
        previous_viewport_shape,
        next_viewport_shape,
    )

    geometry_changed = getattr(previous_plan, "geometry", None) != getattr(
        next_plan, "geometry", None
    )
    if frame_center_anchor and geometry_changed:
        # Anchor the whole frame's centre, not a tile: a zoomed-out re-flow
        # keeps the montage centred where it was instead of chasing the tile
        # that happened to sit under the focus.
        prev_h, prev_w = (float(v) for v in previous_plan.display_shape[:2])
        next_h, next_w = (float(v) for v in next_plan.display_shape[:2])
        focus_x, focus_y = prev_w * 0.5, prev_h * 0.5
        next_focus_x, next_focus_y = next_w * 0.5, next_h * 0.5
    elif not geometry_changed:
        focus_x, focus_y = _focus_or_center(view_range, focus)
        next_focus_x, next_focus_y = focus_x, focus_y
    else:
        focus_x, focus_y = _focus_or_center(view_range, focus)
        anchor = _tile_local_anchor(
            previous_plan, view_range, (focus_x, focus_y), allow_nearest=True
        )
        if anchor is None:
            return None
        source_index, local_x, local_y = anchor
        next_tile = _tile_for_source(next_plan, source_index)
        if next_tile is None:
            return None
        next_focus_x = float(next_tile.x0) + min(max(0.0, local_x), float(next_tile.width))
        next_focus_y = float(next_tile.y0) + min(max(0.0, local_y), float(next_tile.height))
    fraction_x = min(1.0, max(0.0, (focus_x - min(x0, x1)) / span_x))
    fraction_y = min(1.0, max(0.0, (focus_y - min(y0, y1)) / span_y))
    return (
        (next_focus_x - fraction_x * next_span_x, next_focus_x + (1.0 - fraction_x) * next_span_x),
        (next_focus_y - fraction_y * next_span_y, next_focus_y + (1.0 - fraction_y) * next_span_y),
    )


def remap_montage_roi_geometry(
    previous_plan, next_plan, geometry: RoiGeometry
) -> RoiGeometry | None:
    """Move ROI geometry through a same-source montage layout reflow.

    Returns ``None`` when the source population changed.  Rectangle ROIs keep
    their shape and move by a source-local anchor, while point-based ROIs move
    each point by the source tile underneath that point.
    """

    previous_layout = _source_layout(previous_plan)
    next_layout = _source_layout(next_plan)
    return _remap_montage_roi_geometry_with_layout(previous_layout, next_layout, geometry)


def remap_montage_roi_selections(previous_plan, next_plan, selections) -> tuple[RoiSelection, ...]:
    """Return ROI selections remapped through a same-source montage layout reflow.

    The old/new source layout maps are built once per transition.  Geometry
    that cannot be represented cleanly in the existing ROI kind is left
    unchanged.
    """

    previous_layout = _source_layout(previous_plan)
    next_layout = _source_layout(next_plan)
    remapped = []
    for selection in tuple(selections or ()):
        if not isinstance(selection, RoiSelection):
            remapped.append(selection)
            continue
        geometry = _remap_montage_roi_geometry_with_layout(
            previous_layout, next_layout, selection.geometry
        )
        if geometry is not None and geometry != selection.geometry:
            remapped.append(replace(selection, geometry=geometry))
        else:
            remapped.append(selection)
    return tuple(remapped)


def _remap_montage_roi_geometry_with_layout(
    previous_layout, next_layout, geometry: RoiGeometry
) -> RoiGeometry | None:
    geometry = geometry if isinstance(geometry, RoiGeometry) else RoiGeometry(**geometry)
    if previous_layout.source_indices != next_layout.source_indices:
        return None
    if geometry.kind == RoiKind.RECTANGLE:
        return _remap_rectangle_roi(previous_layout, next_layout, geometry)
    if geometry.kind in (RoiKind.LINE, RoiKind.POLYLINE, RoiKind.FREEHAND_POLYGON):
        remapped = _remap_points(previous_layout, next_layout, geometry.points)
        if remapped is None:
            return None
        return replace(geometry, points=remapped)
    return None


def frame_session_key(
    document_key, view_state, viewport_plan: MontageViewportPlan, colormap_lut
) -> tuple[object, ...]:
    """Stable render-session identity, excluding transient viewport coverage."""

    if viewport_plan.axis is None:
        scope_state = view_state.with_montage_axis(None, columns=None, indices=None, text=None)
        axis_key = None
    else:
        scope_state = view_state.with_montage_axis(
            viewport_plan.axis,
            columns=None,
            indices=None,
            text=None,
        )
        axis_key = int(viewport_plan.axis)
    lut_key = None if colormap_lut is None else np.asarray(colormap_lut).tobytes()
    return (
        "montage_tiles",
        document_key,
        scope_state,
        axis_key,
        tuple(int(index) for index in viewport_plan.all_indices),
        tuple(int(value) for value in viewport_plan.tile_shape),
        int(viewport_plan.plan.gap),
        lut_key,
        bool(viewport_plan.shader_display),
    )


def montage_tile_semantic_key(
    document_key,
    view_state,
    viewport_plan: MontageViewportPlan,
    colormap_lut,
    *,
    canonical_orientation: bool = False,
) -> tuple[object, ...]:
    """Identity of one montage tile's TEXELS, shared across index windows.

    ``frame_session_key`` includes the sibling-index selection
    (``all_indices``) and layout gap — correct for session identity, but
    they change WHICH tiles exist, never what a given source index contains.
    Keying pyramid floors/previews by the session key made every index-window
    change rename identical texels, so previously computed tiles refilled
    cold from black (field defect 2026-07-05: missing corner tiles that
    "were there in other views").

    When ``canonical_orientation`` is set the texels are materialized in
    canonical (sorted-image-axes) order, so an X/Y swap must NOT rename them:
    both the scope state's image-axes order and the tile shape are squared to
    ascending-axis order so a transpose reuses the same floors/previews.
    """

    if viewport_plan.axis is None:
        scope_state = view_state.with_montage_axis(None, columns=None, indices=None, text=None)
        axis_key = None
    else:
        scope_state = view_state.with_montage_axis(
            viewport_plan.axis,
            columns=None,
            indices=None,
            text=None,
        )
        axis_key = int(viewport_plan.axis)
    tile_shape = tuple(int(value) for value in viewport_plan.tile_shape)
    if canonical_orientation:
        image_axes = getattr(scope_state, "image_axes", None)
        if image_axes is not None and len(image_axes) == 2 and image_axes[0] > image_axes[1]:
            scope_state = scope_state.with_image_axes(*sorted(int(a) for a in image_axes))
            # tile_shape is (size(image_axes[0]), size(image_axes[1])); a swap
            # transposes it, so square it back to ascending-axis order.
            tile_shape = (tile_shape[1], tile_shape[0], *tile_shape[2:])
    lut_key = None if colormap_lut is None else np.asarray(colormap_lut).tobytes()
    return (
        "montage_tile_semantics",
        document_key,
        scope_state,
        axis_key,
        tile_shape,
        lut_key,
        bool(viewport_plan.shader_display),
    )


def montage_viewport_retarget_policy(
    capabilities, display_mode: str
) -> MontageViewportRetargetPolicy:
    """Return viewport-retarget behaviour for the current presentation surface.

    Montage retargeting updates the tiled presentation when only camera/view
    coverage changes. Persistent residency backends additionally keep
    near-viewport warm coverage.
    """

    if bool(getattr(capabilities, "persistent_tile_residency", False)):
        return MontageViewportRetargetPolicy(
            enabled=True,
            coverage_margin_tiles=1,
            near_margin_tiles=2,
            update_delay_ms=90,
        )
    return MontageViewportRetargetPolicy(enabled=True)


def _normalized_view_range(view_range) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (float(view_range[0][0]), float(view_range[0][1])),
        (float(view_range[1][0]), float(view_range[1][1])),
    )


def _normalized_viewport_shape(viewport_shape) -> tuple[int, int]:
    return (max(1, int(viewport_shape[0])), max(1, int(viewport_shape[1])))


def _resize_preserving_screen_zoom_spans(
    span_x: float,
    span_y: float,
    previous_viewport_shape,
    next_viewport_shape,
) -> tuple[float, float]:
    try:
        previous_height, previous_width = _normalized_viewport_shape(previous_viewport_shape)
        next_height, next_width = _normalized_viewport_shape(next_viewport_shape)
    except Exception:
        return (float(span_x), float(span_y))
    return (
        max(MIN_VIEW_SPAN, float(span_x) * float(next_width) / float(previous_width)),
        max(MIN_VIEW_SPAN, float(span_y) * float(next_height) / float(previous_height)),
    )


def _focus_or_center(view_range, focus) -> tuple[float, float]:
    if focus is not None:
        try:
            return (float(focus[0]), float(focus[1]))
        except (IndexError, KeyError, TypeError, ValueError):
            pass
    return (
        (float(view_range[0][0]) + float(view_range[0][1])) * 0.5,
        (float(view_range[1][0]) + float(view_range[1][1])) * 0.5,
    )


def _tile_local_anchor(
    plan, view_range, focus, *, allow_nearest: bool = True
) -> tuple[int, float, float] | None:
    tile = _tile_at_point(plan, focus)
    if tile is None and allow_nearest:
        tile = _nearest_visible_tile(plan, view_range, focus)
    if tile is None:
        return None
    local_x = float(focus[0]) - float(tile.x0)
    local_y = float(focus[1]) - float(tile.y0)
    return (int(tile.source_index), local_x, local_y)


def _tile_at_point(plan, point):
    tile_at = getattr(plan, "tile_at", None)
    if callable(tile_at):
        try:
            tile = tile_at(int(point[0]), int(point[1]))
            if tile is not None:
                return tile
        except Exception:
            pass
    try:
        x = float(point[0])
        y = float(point[1])
    except Exception:
        return None
    for tile in tuple(getattr(plan, "tiles", ()) or ()):
        x0 = float(tile.x0)
        y0 = float(tile.y0)
        if x0 <= x < x0 + float(tile.width) and y0 <= y < y0 + float(tile.height):
            return tile
    return None


def _tile_for_source(plan, source_index: int):
    source_index = int(source_index)
    for tile in tuple(getattr(plan, "tiles", ()) or ()):
        if int(tile.source_index) == source_index:
            return tile
    return None


@dataclass(frozen=True)
class _SourceLayout:
    plan: object
    source_indices: tuple[int, ...]
    tiles_by_source: dict[int, object]


def _source_layout(plan) -> _SourceLayout:
    tiles = tuple(getattr(plan, "tiles", ()) or ())
    return _SourceLayout(
        plan=plan,
        source_indices=tuple(int(tile.source_index) for tile in tiles),
        tiles_by_source={int(tile.source_index): tile for tile in tiles},
    )


def _remap_points(previous_layout, next_layout, points):
    remapped = []
    for point in tuple(points or ()):
        mapped = _remap_point(previous_layout, next_layout, point)
        if mapped is None:
            return None
        remapped.append(mapped)
    return tuple(remapped)


def _remap_point(previous_layout, next_layout, point) -> tuple[float, float] | None:
    anchor = _tile_local_anchor(previous_layout.plan, _point_view_range(point), point)
    if anchor is None:
        return None
    source_index, local_x, local_y = anchor
    next_tile = next_layout.tiles_by_source.get(int(source_index))
    if next_tile is None:
        return None
    return (
        float(next_tile.x0) + float(local_x),
        float(next_tile.y0) + float(local_y),
    )


def _remap_rectangle_roi(previous_layout, next_layout, geometry: RoiGeometry) -> RoiGeometry | None:
    if geometry.rect is None:
        return None
    x, y, width, height = (float(value) for value in geometry.rect)
    remapped_anchor = _remap_point(previous_layout, next_layout, (x, y))
    if remapped_anchor is None:
        return None
    return replace(geometry, rect=(remapped_anchor[0], remapped_anchor[1], width, height))


def _point_view_range(point) -> tuple[tuple[float, float], tuple[float, float]]:
    x = float(point[0])
    y = float(point[1])
    return ((x - 0.5, x + 0.5), (y - 0.5, y + 0.5))


def _nearest_visible_tile(plan, view_range, focus):
    tile = _nearest_tile_by_grid(plan, focus)
    if tile is not None:
        return tile
    tiles_intersecting = getattr(plan, "tiles_intersecting", None)
    if callable(tiles_intersecting):
        try:
            candidates = tuple(tiles_intersecting(view_range, margin_tiles=0))
        except Exception:
            candidates = ()
    else:
        candidates = ()
    if not candidates:
        candidates = tuple(getattr(plan, "tiles", ()) or ())
    if not candidates:
        return None
    focus_x, focus_y = float(focus[0]), float(focus[1])

    def distance(tile) -> tuple[float, int]:
        closest_x = min(max(focus_x, float(tile.x0)), float(tile.x0 + tile.width))
        closest_y = min(max(focus_y, float(tile.y0)), float(tile.y0 + tile.height))
        dx = focus_x - closest_x
        dy = focus_y - closest_y
        return (dx * dx + dy * dy, int(tile.montage_index))

    return min(candidates, key=distance)


def _nearest_tile_by_grid(plan, focus):
    try:
        x = float(focus[0])
        y = float(focus[1])
        tile_height, tile_width = tuple(int(value) for value in plan.tile_shape)
        gap = max(0, int(plan.gap))
        columns = max(1, int(plan.columns))
        rows = max(1, int(plan.rows))
        tiles = tuple(getattr(plan, "tiles", ()) or ())
    except Exception:
        return None
    if tile_width <= 0 or tile_height <= 0 or not tiles:
        return None
    stride_x = tile_width + gap
    stride_y = tile_height + gap
    if stride_x <= 0 or stride_y <= 0:
        return None
    col = _nearest_grid_index(x, tile_width, stride_x, columns)
    row = _nearest_grid_index(y, tile_height, stride_y, rows)
    tile_number = row * columns + col
    if tile_number < 0 or tile_number >= len(tiles):
        return None
    return tiles[tile_number]


def _nearest_grid_index(value: float, tile_size: int, stride: int, count: int) -> int:
    if value <= 0.0:
        return 0
    raw = int(value // float(stride))
    local = value - float(raw * stride)
    if local >= float(tile_size) + max(0.0, float(stride - tile_size)) * 0.5:
        raw += 1
    return max(0, min(int(count) - 1, int(raw)))
