"""Profile a realistic full-montage workflow in a real ArrayScope window."""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import math
import os
import pstats
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from importlib import metadata
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from arrayscope.core.trace import emit_trace
from arrayscope.display.model.tile_identity import tile_ack_identity
from arrayscope.tools.headless_display import (
    capture_output,
    headless_display,
    is_headless_display,
    run_in_headless_display,
)
from arrayscope.tools.interaction_budget import (
    INTERACTION_SETTLE_HARD_LIMIT_S,
    bounded_interaction_settle_timeout_s,
)

DEFAULT_DATA_PATH = Path("data/_WIPDelRec-tT2_20260223150234_14.nii")
DEFAULT_SESSION_FIXTURE = Path("tests/fixtures/profile_montage_session.json")
PY_SPY_LOW_IMPACT_SAMPLE_RATE_HZ = 25
PY_SPY_FULL_SAMPLE_RATE_HZ = 50
PY_SPY_FULL_DURATION_S = 30
PY_SPY_FULL_DETACH_MARGIN_S = 1
PY_SPY_FULL_ALLOWED_MISSED_STACKS = 1

# Interaction-stress budgets (kept short so the benchmark stays fast).  The
# zoom/pan gestures chain continuously with NO settle between steps; only the
# slow scroll paces by target-LOD completion, and a single short settle runs
# after the whole zoom/pan storm.
SCROLL_FAST_DURATION_S = 5.0
# Build-time budget for the cold full-montage fills (raw and FFT). A 272-tile
# cold fill is a measured multi-second operation: an evidence sweep plus a
# bounded commit walk over every tile. The five-second interaction budget
# applies to per-gesture probes only; capping the build at the gesture budget
# turned a slow-but-progressing fill into a false "stall" verdict while the
# recorded fill milestones (perf-bars program) are the actual pass/fail
# authority for fill time (same rule as the churn harness, 2026-07-17). A
# genuine wedge still fails fast: the completion wait breaks after four
# seconds of an unchanged no-work stall signature regardless of this budget.
COLD_FILL_BUILD_TIMEOUT_S = 120.0

SCROLL_SLOW_LOD_BUDGET_S = min(3.0, INTERACTION_SETTLE_HARD_LIMIT_S)
ZOOMPAN_FINAL_SETTLE_S = min(3.0, INTERACTION_SETTLE_HARD_LIMIT_S)
ZOOMPAN_INPUT_FPS = 120.0
ZOOMPAN_MAX_OUT_REQUEST_SCALE = 1_000_000.0
ZOOMPAN_CENTRAL_SPAN_SCALE = 0.16
ZOOMPAN_PAN_FRACTION = 0.32
ZOOMPAN_DEEP_SPAN_SCALE = 0.06
ZOOMPAN_CHECKPOINT_SETTLE_S = min(3.0, INTERACTION_SETTLE_HARD_LIMIT_S)
ZOOMPAN_NEAR_OBSERVE_S = 0.20
R8_GUI_CALLBACK_MAX_MS = 50.0
R8_HEARTBEAT_MAX_GAP_MS = 16.0
R8_WARM_INPUT_MAX_MS = 15.0
PROFILE_RESIDENCY_PAGE_SAMPLES = 256
PROFILE_PHYSICAL_SAMPLES_PER_TILE = 64
DISPLAY_AXIS_CROP_SCENARIO_NAMES = (
    "primary-only-centered",
    "both-centered",
    "primary-minus-one",
    "secondary-plus-one",
    "both-diagonal",
    "both-return",
    "primary-page-edge",
    "primary-page-cross",
    "primary-page-return",
    "both-odd",
    "both-odd-primary-plus-one",
)
PROFILE_DEFAULT_BACKENDS = ("wgpu", "pyqtgraph")
PROFILE_MONTAGE_STAGES = (
    "load_data",
    "raw_full_tiled_montage",
    "fft_full_tiled_montage",
    "display_x_axis_slice",
    "display_y_axis_slice",
    "fft_level_refinement_preview",
    "montage_scroll_fft",
    "montage_scroll_scalar",
    "montage_zoompan_fft",
    "montage_zoompan_scalar",
)


@dataclass(frozen=True)
class _DisplayAxisCropScenario:
    """One backend-independent displayed-axis crop transition."""

    name: str
    axis_ranges: tuple[tuple[int, tuple[int, ...], str], ...]
    crosses_page_boundary: bool = False

    @property
    def cropped_axis_count(self) -> int:
        return len(self.axis_ranges)


def _display_axis_crop_scenarios(
    *,
    shape: tuple[int, ...],
    image_axes: tuple[int, int],
    primary_role: str,
) -> tuple[_DisplayAxisCropScenario, ...]:
    """Build the same crop-geometry stress matrix for every backend.

    The matrix varies geometry, direction, and page fan-in. It deliberately
    does not encode a renderer strategy: PyQtGraph and WGPU receive the same
    semantic states, while their diagnostics report the work each backend
    needed to present them.
    """

    shape = tuple(int(value) for value in shape)
    image_axes = tuple(int(value) for value in image_axes)
    role = str(primary_role)
    if role not in {"x", "y"}:
        raise ValueError(f"unsupported displayed-axis role {role!r}")
    primary_position = 1 if role == "x" else 0
    primary_axis = image_axes[primary_position]
    secondary_axis = image_axes[1 - primary_position]

    def window(axis: int, extent: int, *, offset: int = 0) -> tuple[int, int]:
        size = shape[axis]
        extent = max(1, min(int(extent), size))
        start = max(0, min((size - extent) // 2 + int(offset), size - extent))
        return start, start + extent

    def shifted(bounds: tuple[int, int], delta: int, axis: int) -> tuple[int, int]:
        start, stop = bounds
        extent = stop - start
        start = max(0, min(start + int(delta), shape[axis] - extent))
        return start, start + extent

    def axis_range(axis: int, bounds: tuple[int, int]) -> tuple[int, tuple[int, ...], str]:
        start, stop = bounds
        return axis, tuple(range(start, stop)), f"{start}:{stop}"

    def scenario(
        name: str,
        primary: tuple[int, int],
        secondary: tuple[int, int] | None,
        *,
        crosses_page_boundary: bool = False,
    ) -> _DisplayAxisCropScenario:
        ranges = [axis_range(primary_axis, primary)]
        if secondary is not None:
            ranges.append(axis_range(secondary_axis, secondary))
        return _DisplayAxisCropScenario(
            name=name,
            axis_ranges=tuple(ranges),
            crosses_page_boundary=bool(crosses_page_boundary),
        )

    primary_center = window(primary_axis, 100, offset=-21)
    secondary_center = window(secondary_axis, 100, offset=13)
    primary_minus_one = shifted(primary_center, -1, primary_axis)
    secondary_plus_one = shifted(secondary_center, 1, secondary_axis)
    primary_diagonal = shifted(primary_center, -2, primary_axis)
    secondary_diagonal = shifted(secondary_center, 2, secondary_axis)

    primary_size = shape[primary_axis]
    boundary_extent = min(100, primary_size)
    boundary_start = max(
        0,
        min(
            PROFILE_RESIDENCY_PAGE_SAMPLES - boundary_extent,
            primary_size - boundary_extent,
        ),
    )
    primary_page_edge = (boundary_start, boundary_start + boundary_extent)
    primary_page_cross = shifted(primary_page_edge, 1, primary_axis)
    crosses_page_boundary = (
        primary_page_cross[0] < PROFILE_RESIDENCY_PAGE_SAMPLES < primary_page_cross[1]
    )

    primary_odd = window(primary_axis, 99, offset=-7)
    secondary_odd = window(secondary_axis, 101, offset=9)
    primary_odd_plus_one = shifted(primary_odd, 1, primary_axis)

    return (
        scenario("primary-only-centered", primary_center, None),
        scenario("both-centered", primary_center, secondary_center),
        scenario("primary-minus-one", primary_minus_one, secondary_center),
        scenario("secondary-plus-one", primary_minus_one, secondary_plus_one),
        scenario("both-diagonal", primary_diagonal, secondary_diagonal),
        scenario("both-return", primary_center, secondary_center),
        scenario("primary-page-edge", primary_page_edge, secondary_center),
        scenario(
            "primary-page-cross",
            primary_page_cross,
            secondary_center,
            crosses_page_boundary=crosses_page_boundary,
        ),
        scenario("primary-page-return", primary_page_edge, secondary_center),
        scenario("both-odd", primary_odd, secondary_odd),
        scenario("both-odd-primary-plus-one", primary_odd_plus_one, secondary_odd),
    )


def _wgpu_cold_payload_binding_rows(win) -> tuple[dict[str, object], ...]:
    """Project current payloads through the real cold binding selector.

    The live crop workflow deliberately prewarms canonical source pages, so a
    successful zero-upload run does not necessarily exercise the crop-local
    fallback.  This read-only projection uses an empty resident set to expose
    the fallback identity selected for every real montage payload.  Different
    source windows must never alias one crop-local physical plane.
    """

    from arrayscope.display.shader_mapping import common_shader_mapping
    from arrayscope.display.wgpu_imageview2d import _wgpu_payload_binding

    view = getattr(win, "img_view", None)
    session = getattr(getattr(win, "renderer", None), "_frame_session", None)
    payloads = dict(getattr(session, "display_tile_payloads", {}) or {})
    if view is None or not payloads:
        return ()
    source_mapping = common_shader_mapping(
        getattr(payload, "shader_mapping", None) for payload in payloads.values()
    )
    representation, mapping_mode, *_rest = view._wgpu_commit_plan(
        payloads,
        source_mapping,
        False,
    )
    rows = []
    for tile, payload in sorted(payloads.items()):
        texture = view._wgpu_payload_texture(payload, representation)
        binding = _wgpu_payload_binding(
            payload,
            texture,
            representation=representation,
            mapping_mode=mapping_mode,
            resident_keys=(),
        )
        anchor = getattr(payload, "source_anchor", None)
        rows.append(
            {
                "tile": int(tile),
                "source_rect": tuple(
                    int(value) for value in tuple(getattr(anchor, "source_rect", ()) or ())
                ),
                "plane_identity": repr(binding.plane_identity),
                "source_anchored": bool(binding.source_anchored),
            }
        )
    return tuple(rows)


def _shift_profile_axis_window(state, axis: int, delta: int):
    """Shift one dimension without changing its current selection geometry."""

    axis = int(axis)
    delta = int(delta)
    axis_size = int(state.shape[axis])
    if getattr(state, "montage_axis", None) == axis:
        values = tuple(
            int(value)
            for value in (
                state.montage_indices
                if state.montage_indices is not None
                else tuple(range(axis_size))
            )
        )
        if not values:
            return state
        start = max(0, min(int(values[0]) + delta, axis_size - len(values)))
        stop = start + len(values)
        return state.with_montage_axis(
            axis,
            columns=state.montage_columns,
            indices=tuple(range(start, stop)),
            text=f"{start}:{stop}",
        )

    values = state.axis_range_indices[axis]
    if values is not None:
        values = tuple(int(value) for value in values)
        if not values:
            return state
        start = max(0, min(int(values[0]) + delta, axis_size - len(values)))
        stop = start + len(values)
        return state.with_axis_range(
            axis,
            indices=tuple(range(start, stop)),
            text=f"{start}:{stop}",
        )

    index = max(0, min(int(state.slice_indices[axis]) + delta, axis_size - 1))
    return state.with_slice(axis, index)


def _profile_axis_shift_direction(state, axis: int) -> int:
    """Choose a non-clamping scroll direction for the current dimension."""

    axis = int(axis)
    axis_size = int(state.shape[axis])
    if getattr(state, "montage_axis", None) == axis:
        values = tuple(
            state.montage_indices if state.montage_indices is not None else tuple(range(axis_size))
        )
    else:
        values = tuple(state.axis_range_indices[axis] or ())
    if values:
        return 1 if int(values[-1]) < axis_size - 3 else -1
    return 1 if int(state.slice_indices[axis]) < axis_size - 3 else -1


def _wgpu_source_window_truth(win) -> dict[str, object]:
    """Compare semantic crop coordinates with payload and shader sampling truth."""

    renderer = getattr(win, "renderer", None)
    session = getattr(renderer, "_frame_session", None)
    if session is None:
        return {
            "applicable": False,
            "passed": True,
            "payload_count": 0,
            "physical_tile_count": 0,
            "session_anchor_matches": True,
            "payload_anchor_mismatches": (),
            "global_origin_mismatches": (),
        }
    expected = renderer._session_source_anchoring(
        session.document,
        session.view_state,
        session.montage_axis,
    )
    if expected is None:
        return {
            "applicable": False,
            "passed": True,
            "payload_count": 0,
            "physical_tile_count": 0,
            "session_anchor_matches": True,
            "payload_anchor_mismatches": (),
            "global_origin_mismatches": (),
        }

    expected_starts = tuple(int(value or 0) for value in tuple(expected.source_starts_yx))
    actual = getattr(session, "source_anchoring", None)
    actual_starts = tuple(
        int(value or 0) for value in tuple(getattr(actual, "source_starts_yx", ()) or ())
    )
    session_anchor_matches = bool(
        actual_starts == expected_starts
        and getattr(actual, "content_key", None) == expected.content_key
    )

    payload_mismatches: list[dict[str, object]] = []
    payloads = dict(getattr(session, "display_tile_payloads", {}) or {})
    for tile, payload in sorted(payloads.items()):
        anchor = getattr(payload, "source_anchor", None)
        source_rect = tuple(int(value) for value in tuple(getattr(anchor, "source_rect", ()) or ()))
        if (
            len(source_rect) != 4
            or source_rect[0] != expected_starts[0]
            or source_rect[2] != expected_starts[1]
        ):
            payload_mismatches.append(
                {
                    "tile": int(tile),
                    "expected_start_yx": expected_starts,
                    "source_rect": source_rect,
                }
            )

    physical_rows_fn = getattr(getattr(win, "img_view", None), "tileTruthPhysicalRows", None)
    physical_rows = dict(physical_rows_fn() or {}) if callable(physical_rows_fn) else {}
    expected_origin = (float(expected_starts[1]), float(expected_starts[0]))
    origin_mismatches: list[dict[str, object]] = []
    for tile, row in sorted(physical_rows.items()):
        row = dict(row or {})
        bindings = tuple(row.get("physical_page_bindings", ()) or ())
        global_binding = False
        for binding in bindings:
            key = dict(binding or {}).get("actual_key")
            generation = getattr(key, "document_generation", None)
            if (
                isinstance(generation, tuple)
                and generation
                and generation[0] == "wgpu-source-plane"
            ):
                global_binding = True
                break
        if not global_binding:
            continue
        actual_origin = tuple(
            float(value) for value in tuple(row.get("physical_source_origin_xy", ()) or ())
        )
        if actual_origin != expected_origin:
            origin_mismatches.append(
                {
                    "tile": int(tile),
                    "expected_origin_xy": expected_origin,
                    "actual_origin_xy": actual_origin,
                }
            )

    passed = bool(session_anchor_matches and not payload_mismatches and not origin_mismatches)
    return {
        "applicable": True,
        "passed": passed,
        "payload_count": len(payloads),
        "physical_tile_count": len(physical_rows),
        "expected_start_yx": expected_starts,
        "actual_start_yx": actual_starts,
        "session_anchor_matches": session_anchor_matches,
        "payload_anchor_mismatches": tuple(payload_mismatches),
        "global_origin_mismatches": tuple(origin_mismatches),
    }


def _physical_frame_reference_truth(
    win,
    *,
    backend: str,
    sample_seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Compare the maintained backend's pixels with CPU truth.

    Returns the image verdict and, beside it, the ROI-placement verdict the
    same capture produced.  The image comparison withholds pixels the frame
    legitimately draws over the montage -- ROI strokes, the profile marker,
    the rasterized floating chips -- so the placement verdict is what keeps
    ROI-through-crop coverage: every enabled ROI must still paint its own
    colour inside the band its semantic geometry projects to, and nowhere
    else.  One capture answers both, so they can never disagree about which
    frame they saw.
    """

    from arrayscope.tools.framebuffer_reference import (
        qt_raster_matches_cpu_reference,
        wgpu_frame_matches_cpu_reference,
    )

    try:
        if backend == "wgpu":
            report = wgpu_frame_matches_cpu_reference(
                win,
                max_samples_per_tile=PROFILE_PHYSICAL_SAMPLES_PER_TILE,
                sample_seed=sample_seed,
            )
        elif backend == "pyqtgraph":
            report = qt_raster_matches_cpu_reference(
                win,
                max_samples_per_tile=PROFILE_PHYSICAL_SAMPLES_PER_TILE,
                sample_seed=sample_seed,
            )
        else:
            return ({"applicable": False, "passed": True}, {"applicable": False, "passed": True})
    except Exception as exc:
        failed = {"applicable": True, "passed": False, "error": repr(exc), "failures": ()}
        return (failed, dict(failed))
    failures = tuple(report.failures())
    image = {
        "applicable": True,
        "passed": not failures,
        "frame_shape": tuple(int(value) for value in report.frame_shape),
        "tile_count": len(report.tiles),
        "total_samples": int(report.total_samples),
        "overlay_excluded_samples": int(report.overlay_excluded_samples),
        "failures": tuple(asdict(failure) for failure in failures),
    }
    placement_report = report.roi_placement
    if placement_report is None:
        placement = {
            "applicable": True,
            "passed": False,
            "error": "frame reference report carried no ROI placement verdict",
            "failures": (),
        }
    else:
        placement_failures = tuple(placement_report.failures())
        placement = {
            "applicable": True,
            "passed": not placement_failures,
            "frame_shape": tuple(int(value) for value in placement_report.frame_shape),
            "roi_count": len(placement_report.rois),
            "stray_checked_roi_count": sum(1 for roi in placement_report.rois if roi.stray_checked),
            "failures": tuple(asdict(failure) for failure in placement_failures),
        }
    return (image, placement)


def _apply_all_dimension_scroll_stress(
    win,
    *,
    app,
    QtCore,
    backend: str,
    physical_sample_seed: int,
) -> dict[str, object]:
    """Exercise coalesced fast and fully settled slow scrolls on every axis."""

    base_state = win.view_state
    physical_rows_fn = getattr(getattr(win, "img_view", None), "tileTruthPhysicalRows", None)
    results: list[dict[str, object]] = []
    physical_counts: list[int] = []
    source_truth_checks: list[dict[str, object]] = []
    physical_reference_checks: list[dict[str, object]] = []
    roi_placement_checks: list[dict[str, object]] = []
    visual_checkpoint_count = 0
    uploads_before = _wgpu_upload_total(win)

    def apply_state(axis: int, state, *, reason: str) -> None:
        apply_slice_state = getattr(win, "_apply_slice_state", None)
        if callable(apply_slice_state):
            apply_slice_state(
                int(axis),
                state,
                reason=reason,
                interactive=True,
                immediate_axis_only=False,
            )
        else:
            win._set_view_state(state)
            win.render(reason=reason)

    def settle(
        predecessor_session_id: int,
        *,
        checkpoint: str,
    ) -> tuple[bool, bool, float]:
        nonlocal visual_checkpoint_count
        started = perf_counter()
        settled = _wait_for_montage_successor_settled(
            win=win,
            app=app,
            QtCore=QtCore,
            predecessor_session_id=predecessor_session_id,
            budget_s=INTERACTION_SETTLE_HARD_LIMIT_S,
        )
        current = _committed_display_frame_is_current(win)
        if callable(physical_rows_fn):
            physical_counts.append(len(dict(physical_rows_fn() or {})))
        if backend == "wgpu":
            source_truth_checks.append(_wgpu_source_window_truth(win))
        image_check, placement_check = _physical_frame_reference_truth(
            win,
            backend=backend,
            sample_seed=int(physical_sample_seed) + len(physical_reference_checks),
        )
        physical_reference_checks.append(image_check)
        roi_placement_checks.append(placement_check)
        visual_probe = getattr(win, "_arrayscope_visual_timeline_probe", None)
        capture = getattr(visual_probe, "capture", None)
        if callable(capture):
            capture(f"all-dimension-{checkpoint}")
            visual_checkpoint_count += 1
        return bool(settled), bool(current), max(0.0, (perf_counter() - started) * 1000.0)

    for axis in range(int(base_state.ndim)):
        axis_uploads_before = _wgpu_upload_total(win)
        direction = _profile_axis_shift_direction(base_state, axis)
        if axis == base_state.montage_axis:
            role = "montage"
        elif axis in tuple(base_state.image_axes or ()):
            role = "display"
        else:
            role = "slice"

        predecessor = getattr(win.renderer, "_frame_session", None)
        predecessor_id = int(getattr(predecessor, "session_id", -1) or -1)
        for step in (1, 2, 3):
            apply_state(
                axis,
                _shift_profile_axis_window(base_state, axis, direction * step),
                reason=f"profile-all-dims-axis-{axis}-fast",
            )
        fast_settled, fast_current, fast_settle_ms = settle(
            predecessor_id,
            checkpoint=f"axis-{axis}-fast",
        )

        predecessor = getattr(win.renderer, "_frame_session", None)
        predecessor_id = int(getattr(predecessor, "session_id", -1) or -1)
        apply_state(axis, base_state, reason=f"profile-all-dims-axis-{axis}-fast-restore")
        fast_restore_settled, fast_restore_current, fast_restore_ms = settle(
            predecessor_id,
            checkpoint=f"axis-{axis}-fast-restore",
        )

        predecessor = getattr(win.renderer, "_frame_session", None)
        predecessor_id = int(getattr(predecessor, "session_id", -1) or -1)
        apply_state(
            axis,
            _shift_profile_axis_window(base_state, axis, direction),
            reason=f"profile-all-dims-axis-{axis}-slow",
        )
        slow_forward_settled, slow_forward_current, slow_forward_ms = settle(
            predecessor_id,
            checkpoint=f"axis-{axis}-slow",
        )

        predecessor = getattr(win.renderer, "_frame_session", None)
        predecessor_id = int(getattr(predecessor, "session_id", -1) or -1)
        apply_state(axis, base_state, reason=f"profile-all-dims-axis-{axis}-slow-return")
        slow_return_settled, slow_return_current, slow_return_ms = settle(
            predecessor_id,
            checkpoint=f"axis-{axis}-slow-return",
        )
        axis_uploads_after = _wgpu_upload_total(win)

        results.append(
            {
                "axis": int(axis),
                "role": role,
                "direction": int(direction),
                "fast_input_steps": 3,
                "fast_settled": bool(fast_settled),
                "fast_committed_current": bool(fast_current),
                "fast_settle_ms": float(fast_settle_ms),
                "fast_restore_settled": bool(fast_restore_settled),
                "fast_restore_committed_current": bool(fast_restore_current),
                "fast_restore_settle_ms": float(fast_restore_ms),
                "slow_input_steps": 2,
                "slow_forward_settled": bool(slow_forward_settled),
                "slow_forward_committed_current": bool(slow_forward_current),
                "slow_forward_settle_ms": float(slow_forward_ms),
                "slow_return_settled": bool(slow_return_settled),
                "slow_return_committed_current": bool(slow_return_current),
                "slow_return_settle_ms": float(slow_return_ms),
                "wgpu_upload_delta": (
                    None
                    if axis_uploads_before is None or axis_uploads_after is None
                    else int(axis_uploads_after - axis_uploads_before)
                ),
            }
        )

    uploads_after = _wgpu_upload_total(win)
    all_settled = bool(
        len(results) == int(base_state.ndim)
        and all(
            bool(result["fast_settled"])
            and bool(result["fast_restore_settled"])
            and bool(result["slow_forward_settled"])
            and bool(result["slow_return_settled"])
            for result in results
        )
    )
    all_current = bool(
        len(results) == int(base_state.ndim)
        and all(
            bool(result["fast_committed_current"])
            and bool(result["fast_restore_committed_current"])
            and bool(result["slow_forward_committed_current"])
            and bool(result["slow_return_committed_current"])
            for result in results
        )
    )

    def role_upload_delta(role: str) -> int | None:
        values = [result["wgpu_upload_delta"] for result in results if result["role"] == role]
        if not values or any(value is None for value in values):
            return None
        return sum(int(value) for value in values)

    return {
        "axis_count": len(results),
        "expected_axis_count": int(base_state.ndim),
        "all_settled": all_settled,
        "all_committed_current": all_current,
        "results": tuple(results),
        "physical_sample_count": len(physical_counts),
        "minimum_physical_tile_count": min(physical_counts, default=0),
        "wgpu_upload_delta": (
            None
            if uploads_before is None or uploads_after is None
            else int(uploads_after - uploads_before)
        ),
        "display_role_wgpu_upload_delta": role_upload_delta("display"),
        "montage_role_wgpu_upload_delta": role_upload_delta("montage"),
        "slice_role_wgpu_upload_delta": role_upload_delta("slice"),
        "wgpu_source_truth_check_count": len(source_truth_checks),
        "wgpu_source_truth_passed": bool(
            backend != "wgpu"
            or (
                source_truth_checks
                and all(bool(check.get("passed", False)) for check in source_truth_checks)
            )
        ),
        "wgpu_source_truth_failures": tuple(
            check for check in source_truth_checks if not bool(check.get("passed", False))
        ),
        "physical_reference_check_count": len(physical_reference_checks),
        "physical_reference_passed": bool(
            physical_reference_checks
            and all(bool(check.get("passed", False)) for check in physical_reference_checks)
        ),
        "physical_reference_failures": tuple(
            check for check in physical_reference_checks if not bool(check.get("passed", False))
        ),
        # What the overlay-coverage mask withheld from the image comparison,
        # so the exclusion is a reported number rather than a silent cap.
        "physical_reference_overlay_excluded_samples": sum(
            int(check.get("overlay_excluded_samples", 0) or 0)
            for check in physical_reference_checks
        ),
        "roi_placement_check_count": len(roi_placement_checks),
        "roi_placement_applicable": bool(
            roi_placement_checks
            and all(bool(check.get("applicable", False)) for check in roi_placement_checks)
        ),
        "roi_placement_passed": bool(
            roi_placement_checks
            and all(bool(check.get("passed", False)) for check in roi_placement_checks)
        ),
        "roi_placement_failures": tuple(
            check for check in roi_placement_checks if not bool(check.get("passed", False))
        ),
        "physical_sample_seed": int(physical_sample_seed),
        "physical_samples_per_tile": int(PROFILE_PHYSICAL_SAMPLES_PER_TILE),
        "visual_checkpoint_count": int(visual_checkpoint_count),
    }


def _parse_stage_flags(values: tuple[str, ...] | None) -> tuple[str, ...]:
    parsed: list[str] = []
    for value in tuple(values or ()):
        for raw_token in str(value).split(","):
            token = raw_token.strip()
            if token:
                parsed.append(token)
    return tuple(parsed)


def _resolve_profile_stages(
    include_stages: tuple[str, ...] | None = None,
    skip_stages: tuple[str, ...] | None = None,
    *,
    stage_order: tuple[str, ...] = PROFILE_MONTAGE_STAGES,
) -> tuple[str, ...]:
    include_stages = tuple(str(stage).strip().lower() for stage in tuple(include_stages or ()))
    skip_stages = tuple(str(stage).strip().lower() for stage in tuple(skip_stages or ()))
    include_set = set(include_stages) if include_stages else set(stage_order)
    skip_set = set(skip_stages)
    unknown = sorted((set(include_stages) | set(skip_stages)) - set(stage_order))
    if unknown:
        raise ValueError(
            f"unknown montage workflow stage(s): {', '.join(unknown)}; expected one of: {', '.join(stage_order)}"
        )
    resolved = [stage for stage in stage_order if stage in include_set and stage not in skip_set]
    return tuple(resolved)


def run_profile_montage_workflow(
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    backend: str = "pyqtgraph",
    wgpu_present_method: str = "bitmap",
    wgpu_power_preference: str = "low-power",
    texture_codec: str = "off",
    jsonl: str | Path | None = None,
    timeout_s: float = INTERACTION_SETTLE_HARD_LIMIT_S,
    max_tiles: int | None = None,
    scroll_max_tiles: int = 60,
    columns: int | None = None,
    load_mode: str = "app",
    profiler_type: str = "plain",
    profiler_artifact_paths: tuple[str | Path, ...] = (),
    stages: tuple[str, ...] | None = None,
    screenshot_dir: str | Path | None = None,
    screenshot_interval_s: float = 0.0,
    session_fixture: str | Path | None = DEFAULT_SESSION_FIXTURE,
    verbose_tile_trace: bool = False,
    synthetic_scene: str | None = None,
    synthetic_shape: tuple[int, int, int] = (192, 256, 40),
    physical_sample_seed: int | None = None,
) -> tuple[dict[str, object], ...]:
    """Run raw full montage, then FFT/shift/iFFT-over-montage-axis montage.

    The function is intentionally suitable for wrapping with an external
    sampling profiler such as ``py-spy``.  Returned and JSONL records contain
    enough app diagnostics to correlate profiler stacks with UI-visible phases.
    """

    timeout_s = bounded_interaction_settle_timeout_s(timeout_s)

    from arrayscope.app.qt_binding import prefer_pyside6

    prefer_pyside6()
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui

    from arrayscope.app.settings_state import ImageRenderingBackendChoice
    from arrayscope.core.view_session import (
        loads_session,
        metadata_for_file,
        save_session_file,
        settings_key_for_metadata,
    )
    from arrayscope.operations.pipeline import CenteredFFT, CenteredIFFT, FFTShift
    from arrayscope.window import ArrayScopeWindow

    backend = _normalize_backend(backend)
    if backend == "wgpu":
        from arrayscope.display.wgpu_imageview2d import configure_wgpu_adapter_for_profile

        configure_wgpu_adapter_for_profile(wgpu_power_preference)
    data_path = Path(data_path)
    screenshot_dir = None if screenshot_dir is None else Path(screenshot_dir)
    run_id = uuid4().hex
    physical_sample_seed = (
        int(run_id[:16], 16) if physical_sample_seed is None else int(physical_sample_seed)
    )
    records: list[dict[str, object]] = []

    app = pg.mkQApp()
    previous_organization_name = str(app.organizationName())
    previous_application_name = str(app.applicationName())
    app.setOrganizationName("ArrayScope")
    app.setApplicationName("ArrayScopeProfileMontage")
    settings = QtCore.QSettings()
    settings.clear()
    backend_choice = {
        "pyqtgraph": ImageRenderingBackendChoice.PYQTGRAPH,
        "vispy": ImageRenderingBackendChoice.VISPY,
        "wgpu": ImageRenderingBackendChoice.WGPU,
    }[backend]
    settings.setValue(
        "image_rendering_backend",
        backend_choice.value,
    )
    wgpu_present_method = str(wgpu_present_method or "bitmap")
    settings.setValue("wgpu_present_method", wgpu_present_method)
    settings.setValue("texture_codec", str(texture_codec or "off"))
    settings.setValue("montage_quality_policy", "resident")
    settings.sync()

    win = None
    probe = None
    visual_probe = None
    try:
        stage_order = _resolve_profile_stages(stages)
        stage_enabled = set(stage_order)
        load_start = perf_counter()
        synthetic_scene = str(synthetic_scene or "").strip().lower()
        if synthetic_scene:
            data = _synthetic_profile_data(synthetic_scene, synthetic_shape)
            artifact_root = screenshot_dir or Path(os.environ.get("TMPDIR", "/tmp"))
            artifact_root.mkdir(parents=True, exist_ok=True)
            data_path = (
                artifact_root
                / f"synthetic-{synthetic_scene}-{'x'.join(str(v) for v in data.shape)}.npy"
            )
            np.save(data_path, data)
            load_mode = "synthetic"
            session_fixture = None
        else:
            data = _load_dataset(data_path, mode=load_mode)
        load_elapsed_ms = (perf_counter() - load_start) * 1000.0
        if np.ndim(data) < 3:
            raise ValueError(
                f"profile workflow requires at least 3 dimensions, got shape {np.shape(data)}"
            )
        restored_fixture = _install_profile_session_fixture(
            QtCore,
            data_path=data_path,
            data=data,
            session_fixture=session_fixture,
            settings=settings,
            loads_session=loads_session,
            metadata_for_file=metadata_for_file,
            save_session_file=save_session_file,
            settings_key_for_metadata=settings_key_for_metadata,
        )
        montage_axis = 2
        tile_count = int(np.shape(data)[montage_axis])
        max_tiles = None if max_tiles is None or int(max_tiles) <= 0 else int(max_tiles)
        large_indices = _centered_indices(tile_count, max_tiles)
        # Keep the displayed-axis regression stage bounded and invariant
        # across datasets. The attached failures used 50 simultaneously
        # visible montage tiles; running the whole source dimension changes
        # both the atomic transaction and the residency pressure being tested.
        display_axis_indices = _centered_indices(tile_count, min(50, tile_count))
        # ``None`` is the application's real default: choose a layout that
        # maximizes the montage in the current viewport.  Do not turn the
        # square-ish value used in the metrics table into a hidden semantic
        # preference.  Doing so made the restored wide session start at 6x10,
        # then snap to 8x8 as soon as the first zoom changed AUTO ownership to
        # USER and exposed the latent explicit preference.
        columns_large = None if columns is None else max(1, int(columns))
        reported_columns_large = (
            _default_columns(len(large_indices)) if columns_large is None else columns_large
        )
        scroll_max_tiles = (
            60 if scroll_max_tiles is None or int(scroll_max_tiles) <= 0 else int(scroll_max_tiles)
        )
        scroll_grid_size = min(tile_count, scroll_max_tiles)
        scroll_source_indices = tuple(range(tile_count))
        scroll_indices = _centered_indices(tile_count, scroll_grid_size)
        columns_small = None if columns is None else max(1, int(columns))
        reported_columns_small = (
            _default_columns(scroll_grid_size) if columns_small is None else columns_small
        )
        screenshot_timing_perturbed = bool(
            screenshot_dir is not None and float(screenshot_interval_s) > 0.0
        )
        base_large = _base_record(
            run_id=run_id,
            backend=backend,
            data_path=data_path,
            data=data,
            load_mode=load_mode,
            montage_axis=montage_axis,
            indices=large_indices,
            full_tile_count=tile_count,
            columns=reported_columns_large,
            max_tiles=max_tiles,
            profiler_type=profiler_type,
            profiler_artifact_paths=profiler_artifact_paths,
            run_temperature=_workflow_run_temperature(),
            qt_platform=str(app.platformName()),
            grid_kind="full",
            source_index_count=tile_count,
            screenshot_timing_perturbed=screenshot_timing_perturbed,
        )
        base_scroll = _base_record(
            run_id=run_id,
            backend=backend,
            data_path=data_path,
            data=data,
            load_mode=load_mode,
            montage_axis=montage_axis,
            indices=scroll_indices,
            full_tile_count=tile_count,
            columns=reported_columns_small,
            max_tiles=scroll_max_tiles,
            profiler_type=profiler_type,
            profiler_artifact_paths=profiler_artifact_paths,
            run_temperature=_workflow_run_temperature(),
            qt_platform=str(app.platformName()),
            grid_kind="scroll",
            source_index_count=len(scroll_source_indices),
            screenshot_timing_perturbed=screenshot_timing_perturbed,
        )
        base_display_axis = _base_record(
            run_id=run_id,
            backend=backend,
            data_path=data_path,
            data=data,
            load_mode=load_mode,
            montage_axis=montage_axis,
            indices=display_axis_indices,
            full_tile_count=tile_count,
            columns=(
                _default_columns(len(display_axis_indices))
                if columns_large is None
                else columns_large
            ),
            max_tiles=len(display_axis_indices),
            profiler_type=profiler_type,
            profiler_artifact_paths=profiler_artifact_paths,
            run_temperature=_workflow_run_temperature(),
            qt_platform=str(app.platformName()),
            grid_kind="display_axis",
            source_index_count=tile_count,
            screenshot_timing_perturbed=screenshot_timing_perturbed,
        )
        base_large["synthetic_scene"] = synthetic_scene or None
        base_scroll["synthetic_scene"] = synthetic_scene or None
        if "load_data" in stage_enabled:
            _append_record(
                records,
                jsonl,
                {
                    **base_large,
                    "phase": "load_data",
                    "elapsed_ms": load_elapsed_ms,
                    "complete": True,
                    "run_temperature": "cold",
                },
            )

        win = ArrayScopeWindow(data, filepath=str(data_path))
        win._arrayscope_profile_backend = str(backend)
        win._profile_session_fixture_viewport_shape = (
            None
            if restored_fixture is None or restored_fixture.viewport is None
            else restored_fixture.viewport.viewport_shape
        )
        win._profile_session_fixture_window_size = (
            None
            if restored_fixture is None or restored_fixture.panels is None
            else restored_fixture.panels.window_size
        )
        win._profile_session_fixture_image_axes = (
            None
            if restored_fixture is None
            else tuple(restored_fixture.recipe.view_state.image_axes or ())
        )
        win._profile_session_fixture_axis_flipped = (
            None
            if restored_fixture is None
            else tuple(restored_fixture.recipe.view_state.axis_flipped)
        )
        win.app_settings = _replace_settings(
            win.app_settings,
            backend=backend,
            image_choice=ImageRenderingBackendChoice,
        )
        win.show()
        _process_events(app, QtCore, count=20)
        if backend == "wgpu":
            effective = str(getattr(win.img_view, "wgpuPresentMethod", lambda: "bitmap")())
            base_large["wgpu_present_method"] = effective
            base_scroll["wgpu_present_method"] = effective
            if wgpu_present_method == "screen" and effective != "screen":
                # Evidence honesty: an explicit "screen" run must never
                # silently measure the bitmap path.  ("auto" records whatever
                # it resolved to — the effective method is in every record.)
                raise RuntimeError(
                    f"wgpu present method {wgpu_present_method!r} requested but "
                    f"the view activated {effective!r} "
                    f"({win.img_view.wgpuPresentMethodFallbackReason()})"
                )
        if screenshot_dir is not None and float(screenshot_interval_s) > 0.0:
            visual_probe = _VisualTimelineProbe(
                QtCore,
                QtGui,
                win,
                backend=backend,
                directory=screenshot_dir,
                interval_s=float(screenshot_interval_s),
            )
            win._arrayscope_visual_timeline_probe = visual_probe
            visual_probe.start()
        fixture_startup_settled = True
        fixture_startup_ms = 0.0
        if restored_fixture is not None:
            fixture_startup_settled, fixture_startup_ms = _wait_for_target_lod(
                win,
                app,
                QtCore,
                budget_s=bounded_interaction_settle_timeout_s(timeout_s),
                stall_grace_s=4.0,
            )
            if not fixture_startup_settled:
                session = getattr(win, "_frame_session", None)
                lifecycle = None if session is None else session.lifecycle_snapshot()
                level_summary = (
                    None
                    if session is None
                    else win.renderer._montage_level_tracker().summary_for(session.level_key)
                )
                raise RuntimeError(
                    "profile session fixture did not reach a settled frame before measurement: "
                    "target_unsettled="
                    f"{0 if session is None else len(session.required_target_unsettled_tiles())} "
                    f"loading={0 if session is None else len(getattr(session, 'loading_tiles', ()) or ())} "
                    f"active={0 if session is None else len(getattr(session, 'active_tile_requests', ()) or ())} "
                    f"dirty={0 if session is None else len(getattr(session, 'dirty_payloads', ()) or ())} "
                    f"presented={0 if session is None else len(session.lifecycle.presented_tiles)} "
                    f"force_auto={False if session is None else bool(getattr(session, 'force_auto', False))} "
                    f"user_levels={None if session is None else getattr(session, 'user_levels_override', None)} "
                    f"evidence={0 if session is None else len(getattr(session, 'pending_level_tiles', ()) or ())}/"
                    f"{False if session is None else bool(getattr(session, 'level_evidence_inflight', False))} "
                    f"histogram_aggregate={False if session is None else bool(getattr(session, 'histogram_aggregate_inflight', False))} "
                    f"level_decision={getattr(win.renderer, '_last_montage_level_decision', None)!r} "
                    f"level_summary={level_summary!r} "
                    f"gate_armed={bool(getattr(win.renderer, '_montage_presentation_gate_armed', False))} "
                    f"gate_backlog={getattr(win.renderer, '_montage_gate_last_backlog', None)!r} "
                    f"gate_no_progress={int(getattr(win.renderer, '_montage_gate_no_progress', 0) or 0)} "
                    f"kernel={getattr(getattr(win, 'kernel', None), 'diagnostics', lambda: None)()!r} "
                    f"lifecycle={None if lifecycle is None else dict(lifecycle.counts)}"
                )
            geometry_deadline = perf_counter() + timeout_s
            while not (
                _window_geometry_state(win)["session_viewport_shape_matches"]
                and _window_geometry_state(win)["session_axis_orientation_matches"]
            ):
                if perf_counter() >= geometry_deadline:
                    raise RuntimeError(
                        "profile session fixture frame settled before its viewport/orientation restore: "
                        f"{_window_geometry_state(win)!r}"
                    )
                _process_events(app, QtCore, count=1)
            _wait_for_physical_presentation_quiet(win, app, QtCore)
        probe = _EventLoopProbe(QtCore, app)
        probe.start()

        raw_state = win.view_state.with_montage_axis(
            montage_axis,
            columns=columns_large,
            indices=large_indices,
            text=":",
        )
        fft_operations = _profile_transform_operations(
            montage_axis,
            centered_fft=CenteredFFT,
            fftshift=FFTShift,
            centered_ifft=CenteredIFFT,
        )
        transform_pipeline = ("CenteredFFT", "FFTShift", "CenteredIFFT")
        fit_stretch_pulsed = {"raw": False, "fft": False}

        def apply_raw() -> dict[str, object]:
            clear_start = perf_counter()
            if tuple(getattr(win.document, "steps", ()) or ()):
                _set_operations(win, ())
            clear_operations_ms = (perf_counter() - clear_start) * 1000.0
            state_start = perf_counter()
            win._set_view_state(raw_state)
            set_view_state_ms = (perf_counter() - state_start) * 1000.0
            render_start = perf_counter()
            win.render(reason="profile-raw-full-montage")
            render_call_ms = (perf_counter() - render_start) * 1000.0
            fit_metrics: dict[str, float] = {}
            fit_stretch_pulsed["raw"] = _pulse_fit_stretch(
                win,
                app=app,
                QtCore=QtCore,
                metrics=fit_metrics,
            )
            return {
                "fit_stretch_pulsed": fit_stretch_pulsed["raw"],
                "action_clear_operations_ms": clear_operations_ms,
                "action_set_view_state_ms": set_view_state_ms,
                "action_render_call_ms": render_call_ms,
                "action_fit_stretch_ms": float(fit_metrics.get("fit_stretch_total_ms", 0.0)),
                **fit_metrics,
                "session_fixture_restored": restored_fixture is not None,
                "session_fixture_startup_settled": bool(fixture_startup_settled),
                "session_fixture_startup_ms": float(fixture_startup_ms),
            }

        if "raw_full_tiled_montage" in stage_enabled:
            raw_record = _run_phase(
                app,
                QtCore,
                win,
                probe,
                phase="raw_full_tiled_montage",
                timeout_s=COLD_FILL_BUILD_TIMEOUT_S,
                action=apply_raw,
                backend=backend,
                screenshot_dir=screenshot_dir,
                build_phase=True,
            )
            _attach_phase_screenshot(
                raw_record,
                win,
                phase="raw_full_tiled_montage",
                backend=backend,
                screenshot_dir=screenshot_dir,
            )
            _append_record(records, jsonl, {**base_large, **raw_record, "run_temperature": "cold"})

        def apply_fft() -> dict[str, object]:
            _set_operations(win, fft_operations)
            # Every stage owns its starting view. In particular, the preceding
            # displayed-axis slice stages leave a non-montage range window
            # active; inheriting that state makes a selected stage differ from
            # the same stage in the full workflow.
            fft_state = raw_state.with_montage_axis(
                montage_axis,
                columns=columns_large,
                indices=large_indices,
                text=":",
            )
            win._set_view_state(fft_state)
            win.render(reason="profile-fft-full-montage")
            fit_metrics: dict[str, float] = {}
            fit_stretch_pulsed["fft"] = _pulse_fit_stretch(
                win,
                app=app,
                QtCore=QtCore,
                metrics=fit_metrics,
            )
            return {
                "fit_stretch_pulsed": fit_stretch_pulsed["fft"],
                "action_fit_stretch_ms": float(fit_metrics.get("fit_stretch_total_ms", 0.0)),
                **fit_metrics,
            }

        if "fft_full_tiled_montage" in stage_enabled:
            fft_record = _run_phase(
                app,
                QtCore,
                win,
                probe,
                phase="fft_full_tiled_montage",
                timeout_s=COLD_FILL_BUILD_TIMEOUT_S,
                action=apply_fft,
                backend=backend,
                screenshot_dir=screenshot_dir,
                build_phase=True,
            )
            _attach_phase_screenshot(
                fft_record,
                win,
                phase="fft_full_tiled_montage",
                backend=backend,
                screenshot_dir=screenshot_dir,
            )
            _append_record(
                records,
                jsonl,
                {
                    **base_large,
                    **fft_record,
                    "operation_pipeline": transform_pipeline,
                    "run_temperature": "mixed",
                },
            )

        def apply_display_axis_slice(role: str) -> dict[str, object]:
            role = str(role)
            if role not in {"x", "y"}:
                raise ValueError(f"unsupported displayed-axis role {role!r}")
            presentation_before = _vispy_presentation_diagnostics(win)
            binding_hits_before = int(
                presentation_before.get("wgpu_residency_binding_cache_hits", 0) or 0
            )
            binding_misses_before = int(
                presentation_before.get("wgpu_residency_binding_cache_misses", 0) or 0
            )
            _set_operations(win, ())
            # Keep the third axis as a montage while cropping and scrolling a
            # displayed axis.  A single-image slice cannot exercise the
            # multi-tile atomic handoff used by the real user path.
            slice_source_state = raw_state.with_montage_axis(
                montage_axis,
                columns=columns_large,
                indices=display_axis_indices,
                text=":",
            ).with_image_axes(1, 0)
            slice_source_state = slice_source_state.with_axis_flipped(
                int(slice_source_state.image_axes[0]),
                True,
            )
            win._set_view_state(slice_source_state)
            win.render(reason=f"profile-display-{role}-axis-slice-source")
            _wait_for_montage_complete_soft(
                win=win,
                app=app,
                QtCore=QtCore,
                budget_s=INTERACTION_SETTLE_HARD_LIMIT_S,
            )
            fit_metrics: dict[str, float] = {}
            fit_stretch_pulsed = _pulse_fit_stretch(
                win,
                app=app,
                QtCore=QtCore,
                metrics=fit_metrics,
            )
            physical_rows = getattr(win.img_view, "tileTruthPhysicalRows", None)
            physical_tile_counts: list[int] = []
            crop_scenarios = _display_axis_crop_scenarios(
                shape=tuple(int(value) for value in win.view_state.shape),
                image_axes=tuple(int(value) for value in win.view_state.image_axes),
                primary_role=role,
            )
            crop_scenario_results: list[dict[str, object]] = []
            cold_binding_windows: dict[int, dict[tuple[int, ...], str]] = {}
            cold_binding_probe_ms = 0.0
            uploads_before_crop_matrix = _wgpu_upload_total(win)
            for scenario in crop_scenarios:
                scenario_state = slice_source_state
                for scenario_axis, indices, text in scenario.axis_ranges:
                    scenario_state = scenario_state.with_axis_range(
                        scenario_axis,
                        indices=indices,
                        text=text,
                    )
                before = _vispy_presentation_diagnostics(win)
                uploads_before = _wgpu_upload_total(win)
                step_started = perf_counter()
                predecessor_session = getattr(win.renderer, "_frame_session", None)
                predecessor_session_id = int(getattr(predecessor_session, "session_id", -1) or -1)
                win._set_view_state(scenario_state)
                win.render(reason=f"profile-display-{role}-axis-{scenario.name}")
                settled = _wait_for_montage_successor_settled(
                    win=win,
                    app=app,
                    QtCore=QtCore,
                    predecessor_session_id=predecessor_session_id,
                    budget_s=INTERACTION_SETTLE_HARD_LIMIT_S,
                )
                committed_current = _committed_display_frame_is_current(win)
                after = _vispy_presentation_diagnostics(win)
                uploads_after = _wgpu_upload_total(win)
                rows = dict(physical_rows() or {}) if callable(physical_rows) else {}
                physical_tile_counts.append(len(rows))
                binding_counts = tuple(
                    len(tuple(dict(row or {}).get("physical_page_bindings", ()) or ()))
                    for row in rows.values()
                )
                step_settle_ms = max(0.0, (perf_counter() - step_started) * 1000.0)
                cold_probe_started = perf_counter()
                cold_binding_rows = (
                    _wgpu_cold_payload_binding_rows(win) if backend == "wgpu" else ()
                )
                cold_binding_probe_ms += max(
                    0.0,
                    (perf_counter() - cold_probe_started) * 1000.0,
                )
                for binding_row in cold_binding_rows:
                    source_rect = tuple(binding_row["source_rect"])
                    if len(source_rect) != 4 or bool(binding_row["source_anchored"]):
                        continue
                    cold_binding_windows.setdefault(int(binding_row["tile"]), {})[source_rect] = (
                        str(binding_row["plane_identity"])
                    )
                crop_scenario_results.append(
                    {
                        "name": scenario.name,
                        "axis_ranges": tuple(
                            (scenario_axis, text)
                            for scenario_axis, _indices, text in scenario.axis_ranges
                        ),
                        "cropped_axis_count": scenario.cropped_axis_count,
                        "crosses_page_boundary": scenario.crosses_page_boundary,
                        "settled": bool(settled),
                        "committed_current": bool(committed_current),
                        "settle_ms": step_settle_ms,
                        "wgpu_upload_delta": (
                            None
                            if uploads_before is None or uploads_after is None
                            else int(uploads_after - uploads_before)
                        ),
                        "wgpu_binding_cache_hit_delta": (
                            None
                            if backend != "wgpu"
                            else int(after.get("wgpu_residency_binding_cache_hits", 0) or 0)
                            - int(before.get("wgpu_residency_binding_cache_hits", 0) or 0)
                        ),
                        "wgpu_binding_cache_miss_delta": (
                            None
                            if backend != "wgpu"
                            else int(after.get("wgpu_residency_binding_cache_misses", 0) or 0)
                            - int(before.get("wgpu_residency_binding_cache_misses", 0) or 0)
                        ),
                        "physical_tile_count": len(rows),
                        "minimum_page_bindings_per_tile": min(binding_counts, default=0),
                        "maximum_page_bindings_per_tile": max(binding_counts, default=0),
                        "wgpu_cold_local_binding_count": sum(
                            int(not bool(row["source_anchored"])) for row in cold_binding_rows
                        ),
                    }
                )
                if not settled or not committed_current:
                    break
            uploads_after_crop_matrix = _wgpu_upload_total(win)
            primary_axis = crop_scenarios[-1].axis_ranges[0][0]
            primary_indices = crop_scenarios[-1].axis_ranges[0][1]
            final_start = int(primary_indices[0])
            final_stop = int(primary_indices[-1]) + 1
            slice_size = len(primary_indices)
            initial_crop_settled = bool(
                crop_scenario_results and crop_scenario_results[0]["settled"]
            )
            final_settled = bool(
                len(crop_scenario_results) == len(crop_scenarios)
                and all(
                    bool(result["settled"]) and bool(result["committed_current"])
                    for result in crop_scenario_results
                )
            )
            final_settled = bool(final_settled) and _wait_for_montage_complete_soft(
                win=win,
                app=app,
                QtCore=QtCore,
                budget_s=INTERACTION_SETTLE_HARD_LIMIT_S,
            )
            all_dimension_scroll = _apply_all_dimension_scroll_stress(
                win,
                app=app,
                QtCore=QtCore,
                backend=backend,
                physical_sample_seed=physical_sample_seed,
            )
            if int(all_dimension_scroll["physical_sample_count"]) > 0:
                physical_tile_counts.append(
                    int(all_dimension_scroll["minimum_physical_tile_count"])
                )
            view_range = _montage_view_range(win)
            if view_range is not None:
                zoomed_out = _maximum_zoomout_view_range(
                    win,
                    app,
                    QtCore,
                    view_range,
                )
                _apply_view_range(win, zoomed_out[0], zoomed_out[1])
            target_lod_reached, target_lod_settle_ms = _wait_for_target_lod(
                win,
                app,
                QtCore,
                budget_s=INTERACTION_SETTLE_HARD_LIMIT_S,
            )
            # The field stall needs the combination, not an isolated crop:
            # retain the cropped source window, swap the displayed X/Y roles,
            # then swap back.  Both are presentation-only transforms for the
            # canonical source pages.  This also catches a phase-1 histogram
            # completion which synchronously reuses retained evidence but
            # forgets to wake the exact follow-up.
            original_image_axes = tuple(int(value) for value in win.view_state.image_axes)
            swapped_image_axes = tuple(reversed(original_image_axes))
            axis_swap_settled = True
            axis_swap_steps = 0
            uploads_before_axis_swap = _wgpu_upload_total(win)
            for image_axes in (swapped_image_axes, original_image_axes):
                predecessor_session = getattr(win.renderer, "_frame_session", None)
                predecessor_session_id = int(getattr(predecessor_session, "session_id", -1) or -1)
                win._set_view_state(win.view_state.with_image_axes(*image_axes))
                win.render(reason=f"profile-display-{role}-axis-swap")
                settled = _wait_for_montage_successor_settled(
                    win=win,
                    app=app,
                    QtCore=QtCore,
                    predecessor_session_id=predecessor_session_id,
                    budget_s=INTERACTION_SETTLE_HARD_LIMIT_S,
                )
                axis_swap_steps += 1
                axis_swap_settled = bool(axis_swap_settled and settled)
                if callable(physical_rows):
                    physical_tile_counts.append(len(dict(physical_rows() or {})))
                if not settled:
                    break
            uploads_after_axis_swap = _wgpu_upload_total(win)
            # Preserve the cropped displayed-axis window while leaving montage
            # mode, then advance the former montage axis by one and return.
            # Together with the three crop-window steps above, every X/Y
            # setting now exercises short scrolls on both relevant dimensions.
            # This is the field trajectory that exposed a silent semantic
            # stall: the backend had drawn and all queues were empty, but the
            # committed frame still named the predecessor slice.
            cropped_montage_state = win.view_state
            center_position = len(display_axis_indices) // 2
            center_index = int(display_axis_indices[center_position])
            adjacent_index = int(
                display_axis_indices[min(center_position + 1, len(display_axis_indices) - 1)]
            )
            single_slice_indices = (center_index, adjacent_index, center_index)
            single_slice_settled = True
            single_slice_committed_current = True
            single_slice_steps = 0
            single_slice_settle_ms: list[float] = []
            uploads_before_single_slice = _wgpu_upload_total(win)
            for source_index in single_slice_indices:
                step_started = perf_counter()
                predecessor_session = getattr(win.renderer, "_frame_session", None)
                predecessor_session_id = int(getattr(predecessor_session, "session_id", -1) or -1)
                win._set_view_state(
                    cropped_montage_state.tile_state_for_slice(
                        montage_axis,
                        source_index,
                    )
                )
                win.render(reason=f"profile-display-{role}-axis-single-slice")
                settled = _wait_for_montage_successor_settled(
                    win=win,
                    app=app,
                    QtCore=QtCore,
                    predecessor_session_id=predecessor_session_id,
                    budget_s=INTERACTION_SETTLE_HARD_LIMIT_S,
                )
                current = _committed_display_frame_is_current(win)
                single_slice_settle_ms.append(max(0.0, (perf_counter() - step_started) * 1000.0))
                single_slice_steps += 1
                single_slice_settled = bool(single_slice_settled and settled)
                single_slice_committed_current = bool(single_slice_committed_current and current)
                if not settled or not current:
                    break
            uploads_after_single_slice = _wgpu_upload_total(win)
            predecessor_session = getattr(win.renderer, "_frame_session", None)
            predecessor_session_id = int(getattr(predecessor_session, "session_id", -1) or -1)
            win._set_view_state(cropped_montage_state)
            win.render(reason=f"profile-display-{role}-axis-restore-montage")
            montage_restore_settled = _wait_for_montage_successor_settled(
                win=win,
                app=app,
                QtCore=QtCore,
                predecessor_session_id=predecessor_session_id,
                budget_s=INTERACTION_SETTLE_HARD_LIMIT_S,
            )
            montage_restore_committed_current = _committed_display_frame_is_current(win)
            if callable(physical_rows):
                physical_tile_counts.append(len(dict(physical_rows() or {})))
            presentation = _vispy_presentation_diagnostics(win)
            cold_binding_aliases = tuple(
                {
                    "tile": int(tile),
                    "source_windows": len(windows),
                    "plane_identities": len(set(windows.values())),
                }
                for tile, windows in sorted(cold_binding_windows.items())
                if len(windows) > 1 and len(set(windows.values())) != len(windows)
            )
            cold_binding_multiwindow_tiles = sum(
                int(len(windows) > 1) for windows in cold_binding_windows.values()
            )
            binding_hits_after = int(presentation.get("wgpu_residency_binding_cache_hits", 0) or 0)
            binding_misses_after = int(
                presentation.get("wgpu_residency_binding_cache_misses", 0) or 0
            )
            return {
                "display_axis_role": role,
                "display_axis": primary_axis,
                "display_axis_slice_start": final_start,
                "display_axis_slice_stop": final_stop,
                "display_axis_slice_size": slice_size,
                "display_axis_slice_scroll_steps": 3,
                "display_axis_slice_scroll_direction": "both",
                "display_axis_slice_scroll_cadence_ms": 0.0,
                "display_axis_initial_crop_settled": bool(initial_crop_settled),
                "display_axis_final_settled": bool(final_settled),
                "display_axis_crop_scenario_count": len(crop_scenario_results),
                "display_axis_crop_scenario_names": tuple(
                    str(result["name"]) for result in crop_scenario_results
                ),
                "display_axis_crop_scenarios_settled": bool(
                    len(crop_scenario_results) == len(crop_scenarios)
                    and all(bool(result["settled"]) for result in crop_scenario_results)
                ),
                "display_axis_crop_scenarios_committed_current": bool(
                    len(crop_scenario_results) == len(crop_scenarios)
                    and all(bool(result["committed_current"]) for result in crop_scenario_results)
                ),
                "display_axis_both_crop_scenario_count": sum(
                    int(scenario.cropped_axis_count == 2) for scenario in crop_scenarios
                ),
                "display_axis_page_boundary_scenario_count": sum(
                    int("page-" in scenario.name) for scenario in crop_scenarios
                ),
                "display_axis_crop_scenario_total_settle_ms": sum(
                    float(result["settle_ms"]) for result in crop_scenario_results
                ),
                "display_axis_crop_scenario_max_settle_ms": max(
                    (float(result["settle_ms"]) for result in crop_scenario_results),
                    default=0.0,
                ),
                "display_axis_crop_scenarios": tuple(crop_scenario_results),
                "display_axis_all_dimension_scroll_axis_count": int(
                    all_dimension_scroll["axis_count"]
                ),
                "display_axis_all_dimension_scroll_expected_axis_count": int(
                    all_dimension_scroll["expected_axis_count"]
                ),
                "display_axis_all_dimension_scrolls_settled": bool(
                    all_dimension_scroll["all_settled"]
                ),
                "display_axis_all_dimension_scrolls_committed_current": bool(
                    all_dimension_scroll["all_committed_current"]
                ),
                "display_axis_all_dimension_scroll_results": tuple(all_dimension_scroll["results"]),
                "display_axis_all_dimension_scroll_physical_sample_count": int(
                    all_dimension_scroll["physical_sample_count"]
                ),
                "display_axis_all_dimension_scroll_min_physical_tile_count": int(
                    all_dimension_scroll["minimum_physical_tile_count"]
                ),
                "display_axis_all_dimension_scroll_wgpu_upload_delta": (
                    all_dimension_scroll["wgpu_upload_delta"]
                ),
                "display_axis_all_dimension_display_roles_wgpu_upload_delta": (
                    all_dimension_scroll["display_role_wgpu_upload_delta"]
                ),
                "display_axis_all_dimension_montage_role_wgpu_upload_delta": (
                    all_dimension_scroll["montage_role_wgpu_upload_delta"]
                ),
                "display_axis_all_dimension_slice_roles_wgpu_upload_delta": (
                    all_dimension_scroll["slice_role_wgpu_upload_delta"]
                ),
                "display_axis_wgpu_source_truth_check_count": int(
                    all_dimension_scroll["wgpu_source_truth_check_count"]
                ),
                "display_axis_wgpu_source_truth_passed": bool(
                    all_dimension_scroll["wgpu_source_truth_passed"]
                ),
                "display_axis_wgpu_source_truth_failures": tuple(
                    all_dimension_scroll["wgpu_source_truth_failures"]
                ),
                "display_axis_physical_reference_check_count": int(
                    all_dimension_scroll["physical_reference_check_count"]
                ),
                "display_axis_physical_reference_passed": bool(
                    all_dimension_scroll["physical_reference_passed"]
                ),
                "display_axis_physical_reference_failures": tuple(
                    all_dimension_scroll["physical_reference_failures"]
                ),
                "display_axis_physical_reference_overlay_excluded_samples": int(
                    all_dimension_scroll["physical_reference_overlay_excluded_samples"]
                ),
                "display_axis_roi_placement_check_count": int(
                    all_dimension_scroll["roi_placement_check_count"]
                ),
                "display_axis_roi_placement_applicable": bool(
                    all_dimension_scroll["roi_placement_applicable"]
                ),
                "display_axis_roi_placement_passed": bool(
                    all_dimension_scroll["roi_placement_passed"]
                ),
                "display_axis_roi_placement_failures": tuple(
                    all_dimension_scroll["roi_placement_failures"]
                ),
                "display_axis_physical_sample_seed": int(
                    all_dimension_scroll["physical_sample_seed"]
                ),
                "display_axis_physical_samples_per_tile": int(
                    all_dimension_scroll["physical_samples_per_tile"]
                ),
                "display_axis_visual_checkpoint_count": int(
                    all_dimension_scroll["visual_checkpoint_count"]
                ),
                "display_axis_crop_matrix_wgpu_upload_delta": (
                    None
                    if uploads_before_crop_matrix is None or uploads_after_crop_matrix is None
                    else int(uploads_after_crop_matrix - uploads_before_crop_matrix)
                ),
                "display_axis_crop_wgpu_upload_delta": (
                    None
                    if not crop_scenario_results
                    else crop_scenario_results[0]["wgpu_upload_delta"]
                ),
                "display_axis_scroll_wgpu_upload_delta": (
                    None
                    if uploads_before_crop_matrix is None
                    or uploads_after_crop_matrix is None
                    or not crop_scenario_results
                    or crop_scenario_results[0]["wgpu_upload_delta"] is None
                    else int(uploads_after_crop_matrix - uploads_before_crop_matrix)
                    - int(crop_scenario_results[0]["wgpu_upload_delta"])
                ),
                "display_axis_xy_swap_settled": bool(axis_swap_settled),
                "display_axis_xy_swap_steps": int(axis_swap_steps),
                "display_axis_xy_swap_wgpu_upload_delta": (
                    None
                    if uploads_before_axis_swap is None or uploads_after_axis_swap is None
                    else int(uploads_after_axis_swap - uploads_before_axis_swap)
                ),
                "display_axis_single_slice_settled": bool(single_slice_settled),
                "display_axis_single_slice_committed_current": bool(single_slice_committed_current),
                "display_axis_single_slice_steps": int(single_slice_steps),
                "display_axis_single_slice_step_settle_ms": tuple(single_slice_settle_ms),
                "display_axis_single_slice_max_settle_ms": max(
                    single_slice_settle_ms,
                    default=0.0,
                ),
                "display_axis_single_slice_wgpu_upload_delta": (
                    None
                    if uploads_before_single_slice is None or uploads_after_single_slice is None
                    else int(uploads_after_single_slice - uploads_before_single_slice)
                ),
                "display_axis_montage_restore_settled": bool(montage_restore_settled),
                "display_axis_montage_restore_committed_current": bool(
                    montage_restore_committed_current
                ),
                "display_axis_wgpu_residency_binding_cache_hit_delta": (
                    None if backend != "wgpu" else binding_hits_after - binding_hits_before
                ),
                "display_axis_wgpu_residency_binding_cache_miss_delta": (
                    None if backend != "wgpu" else binding_misses_after - binding_misses_before
                ),
                "display_axis_wgpu_pool_exhaustion": str(
                    presentation.get("wgpu_last_pool_exhaustion", "") or ""
                ),
                "display_axis_wgpu_cold_binding_multiwindow_tiles": int(
                    cold_binding_multiwindow_tiles
                ),
                "display_axis_wgpu_cold_binding_aliases": cold_binding_aliases,
                "display_axis_wgpu_cold_binding_identity_unique": (
                    None
                    if backend != "wgpu"
                    else bool(cold_binding_multiwindow_tiles > 0 and not cold_binding_aliases)
                ),
                "display_axis_wgpu_cold_binding_probe_ms": float(cold_binding_probe_ms),
                "display_axis_physical_tile_sample_count": len(physical_tile_counts),
                "display_axis_min_physical_tile_count": min(
                    physical_tile_counts,
                    default=0,
                ),
                "fit_stretch_pulsed": fit_stretch_pulsed,
                "action_fit_stretch_ms": float(fit_metrics.get("fit_stretch_total_ms", 0.0)),
                **fit_metrics,
                "target_lod_reached": target_lod_reached,
                "target_lod_settle_ms": target_lod_settle_ms,
            }

        for role in ("x", "y"):
            phase = f"display_{role}_axis_slice"
            if phase not in stage_enabled:
                continue

            def apply_slice(role=role) -> dict[str, object]:
                return apply_display_axis_slice(role)

            slice_record = _run_phase(
                app,
                QtCore,
                win,
                probe,
                phase=phase,
                timeout_s=timeout_s,
                action=apply_slice,
                backend=backend,
                screenshot_dir=screenshot_dir,
            )
            _attach_phase_screenshot(
                slice_record,
                win,
                phase=phase,
                backend=backend,
                screenshot_dir=screenshot_dir,
            )
            _append_record(
                records,
                jsonl,
                {
                    **base_display_axis,
                    **slice_record,
                    "operation_pipeline": (),
                    "run_temperature": "mixed",
                },
            )

        if "fft_level_refinement_preview" in stage_enabled:

            def apply_fft_level_preview() -> dict[str, object]:
                apply_fft()
                return {
                    **_apply_fft_level_refinement_preview(win, app=app, QtCore=QtCore),
                    "fit_stretch_pulsed": bool(fit_stretch_pulsed["fft"]),
                }

            level_record = _run_phase(
                app,
                QtCore,
                win,
                probe,
                phase="fft_level_refinement_preview",
                timeout_s=timeout_s,
                action=apply_fft_level_preview,
                backend=backend,
                screenshot_dir=screenshot_dir,
            )
            _attach_phase_screenshot(
                level_record,
                win,
                phase="fft_level_refinement_preview",
                backend=backend,
                screenshot_dir=screenshot_dir,
            )
            _append_record(
                records,
                jsonl,
                {
                    **base_large,
                    **level_record,
                    "operation_pipeline": transform_pipeline,
                    "run_temperature": "warm",
                },
            )

        if tile_count >= 8:
            # Phase 5: improved scroll pattern over the FFT small montage (ops loaded).
            if "montage_scroll_fft" in stage_enabled:

                def _fft_scroll_action() -> dict[str, object]:
                    _set_operations(win, fft_operations)
                    win._set_view_state(raw_state)
                    _set_montage_indices(
                        win,
                        montage_axis=montage_axis,
                        columns=columns_small,
                        indices=scroll_indices,
                    )
                    _wait_for_montage_complete_soft(
                        win=win, app=app, QtCore=QtCore, budget_s=INTERACTION_SETTLE_HARD_LIMIT_S
                    )
                    return _apply_montage_scroll_pattern(
                        win,
                        montage_axis=montage_axis,
                        columns=columns_small,
                        indices=scroll_source_indices,
                        window_size=scroll_grid_size,
                        probe=probe,
                        app=app,
                        QtCore=QtCore,
                        verbose_tile_trace=verbose_tile_trace,
                    )

                scroll_fft_record = _run_phase(
                    app,
                    QtCore,
                    win,
                    probe,
                    phase="montage_scroll_fft",
                    timeout_s=timeout_s,
                    action=_fft_scroll_action,
                    backend=backend,
                    screenshot_dir=screenshot_dir,
                )
                _attach_phase_screenshot(
                    scroll_fft_record,
                    win,
                    phase="montage_scroll_fft",
                    backend=backend,
                    screenshot_dir=screenshot_dir,
                )
                _append_record(
                    records,
                    jsonl,
                    {
                        **base_scroll,
                        **scroll_fft_record,
                        "operation_pipeline": transform_pipeline,
                        "run_temperature": "warm",
                    },
                )

            # Phase 6: reset the small montage to the start, strip the ops, and run
            # the identical scroll pattern over the raw scalar data.
            if "montage_scroll_scalar" in stage_enabled:

                def _scalar_scroll_action() -> dict[str, object]:
                    _set_operations(win, ())
                    win._set_view_state(raw_state)
                    _set_montage_indices(
                        win,
                        montage_axis=montage_axis,
                        columns=columns_small,
                        indices=scroll_indices,
                    )
                    _wait_for_montage_complete_soft(
                        win=win, app=app, QtCore=QtCore, budget_s=INTERACTION_SETTLE_HARD_LIMIT_S
                    )
                    return _apply_montage_scroll_pattern(
                        win,
                        montage_axis=montage_axis,
                        columns=columns_small,
                        indices=scroll_source_indices,
                        window_size=scroll_grid_size,
                        probe=probe,
                        app=app,
                        QtCore=QtCore,
                        verbose_tile_trace=verbose_tile_trace,
                    )

                scroll_scalar_record = _run_phase(
                    app,
                    QtCore,
                    win,
                    probe,
                    phase="montage_scroll_scalar",
                    timeout_s=timeout_s,
                    action=_scalar_scroll_action,
                    backend=backend,
                    screenshot_dir=screenshot_dir,
                )
                _attach_phase_screenshot(
                    scroll_scalar_record,
                    win,
                    phase="montage_scroll_scalar",
                    backend=backend,
                    screenshot_dir=screenshot_dir,
                )
                _append_record(
                    records,
                    jsonl,
                    {**base_scroll, **scroll_scalar_record, "run_temperature": "warm"},
                )

            # Phase 7: grow to the full montage on scalar data, zoom out to the
            # enforced limit (tiles tiny), add the ops back, then a zoom/pan stress
            # sequence that hammers the LOD + visibility system on FFT data.
            def _fft_zoompan_action() -> dict[str, object]:
                _set_operations(win, ())
                win._set_view_state(raw_state)
                _set_montage_indices(
                    win, montage_axis=montage_axis, columns=columns_small, indices=scroll_indices
                )
                _wait_for_montage_complete_soft(
                    win=win, app=app, QtCore=QtCore, budget_s=INTERACTION_SETTLE_HARD_LIMIT_S
                )
                return _apply_montage_zoom_pan_stress(
                    win,
                    probe=probe,
                    app=app,
                    QtCore=QtCore,
                    mid_toggle=lambda: _set_operations(win, fft_operations),
                    montage_axis=montage_axis,
                    columns=columns_small,
                    indices=scroll_source_indices,
                    window_size=scroll_grid_size,
                )

            if "montage_zoompan_fft" in stage_enabled:
                zoompan_fft_record = _run_phase(
                    app,
                    QtCore,
                    win,
                    probe,
                    phase="montage_zoompan_fft",
                    timeout_s=timeout_s,
                    action=_fft_zoompan_action,
                    backend=backend,
                    screenshot_dir=screenshot_dir,
                )
                _attach_phase_screenshot(
                    zoompan_fft_record,
                    win,
                    phase="montage_zoompan_fft",
                    backend=backend,
                    screenshot_dir=screenshot_dir,
                )
                _append_record(
                    records,
                    jsonl,
                    {
                        **base_scroll,
                        **zoompan_fft_record,
                        "operation_pipeline": transform_pipeline,
                        "run_temperature": "warm",
                    },
                )

            # Phase 8: zoom back out to the limit on FFT data, strip the ops, then
            # run the identical zoom/pan stress sequence over the raw scalar data.
            def _scalar_zoompan_action() -> dict[str, object]:
                _set_operations(win, fft_operations)
                win._set_view_state(raw_state)
                _set_montage_indices(
                    win, montage_axis=montage_axis, columns=columns_small, indices=scroll_indices
                )
                _wait_for_montage_complete_soft(
                    win=win, app=app, QtCore=QtCore, budget_s=INTERACTION_SETTLE_HARD_LIMIT_S
                )
                return _apply_montage_zoom_pan_stress(
                    win,
                    probe=probe,
                    app=app,
                    QtCore=QtCore,
                    mid_toggle=lambda: _set_operations(win, ()),
                    montage_axis=montage_axis,
                    columns=columns_small,
                    indices=scroll_source_indices,
                    window_size=scroll_grid_size,
                )

            if "montage_zoompan_scalar" in stage_enabled:
                zoompan_scalar_record = _run_phase(
                    app,
                    QtCore,
                    win,
                    probe,
                    phase="montage_zoompan_scalar",
                    timeout_s=timeout_s,
                    action=_scalar_zoompan_action,
                    backend=backend,
                    screenshot_dir=screenshot_dir,
                )
                _attach_phase_screenshot(
                    zoompan_scalar_record,
                    win,
                    phase="montage_zoompan_scalar",
                    backend=backend,
                    screenshot_dir=screenshot_dir,
                )
                _append_record(
                    records,
                    jsonl,
                    {**base_scroll, **zoompan_scalar_record, "run_temperature": "warm"},
                )
        return tuple(records)
    finally:
        if probe is not None:
            probe.stop()
        if visual_probe is not None:
            visual_probe.stop()
        if win is not None:
            win.close()
            _process_events(app, QtCore, count=10)
        settings.clear()
        settings.sync()
        app.setOrganizationName(previous_organization_name)
        app.setApplicationName(previous_application_name)


def _scroll_montage_window(
    win,
    *,
    montage_axis,
    columns,
    indices,
    window_start,
    size,
    text=":",
    interactive=False,
):
    """Set the montage index window to `indices[window_start : window_start+size]`.

    ``interactive=True`` routes through the coalescing ``request_render`` path
    (the real user-scroll input path) so a 60fps input stream is delivered and
    coalesced rather than forcing a synchronous full render per frame.
    """

    indices = tuple(indices)
    window_size = max(1, min(int(size), len(indices)))
    if len(indices) <= 0 or window_size <= 0:
        return {"state_build_ms": 0.0, "state_apply_ms": 0.0, "render_request_ms": 0.0}
    start = min(max(int(window_start), 0), len(indices) - window_size)
    window_indices = indices[start : start + window_size]
    state_started = perf_counter()
    state = win.view_state.with_montage_axis(
        montage_axis, columns=columns, indices=window_indices, text=text
    )
    state_build_ms = (perf_counter() - state_started) * 1000.0
    apply_started = perf_counter()
    win._set_view_state(state)
    state_apply_ms = (perf_counter() - apply_started) * 1000.0
    request_started = perf_counter()
    if interactive:
        win.request_render(reason="profile-scroll", interactive=True)
    else:
        win.render(reason="profile-scroll")
    return {
        "state_build_ms": float(state_build_ms),
        "state_apply_ms": float(state_apply_ms),
        "render_request_ms": float((perf_counter() - request_started) * 1000.0),
    }


def _set_operations(win, operations) -> None:
    """Load (or clear, with an empty tuple) the montage operation pipeline."""

    operations = tuple(operations)
    current = tuple(getattr(win.operation_coordinator.document, "enabled_operations", ()) or ())
    if current == operations:
        # Reapplying an identical pipeline is not a realistic user action and
        # creates fresh OperationStep ids, invalidating stage/tile caches and
        # semantic predecessor identity. Back-to-back stages must exercise the
        # warm production path without a hidden hard reset.
        return
    win.operation_coordinator.load_operations(operations)
    win._set_document(win.operation_coordinator.document)
    win._coerce_channel_for_current_dtype()


def _set_montage_indices(win, *, montage_axis, columns, indices, text=":") -> None:
    """Set the montage index window to an explicit index sequence and render."""

    state = win.view_state.with_montage_axis(
        montage_axis, columns=columns, indices=tuple(indices), text=text
    )
    win._set_view_state(state)
    win.render(reason="profile-montage-window")


def _montage_at_target_lod(win) -> bool:
    """True when the montage has fully converged to its target (desired) LOD.

    Stricter than ``_montage_settled`` (which settles on first-pixels / preview
    floor): additionally requires the applied LOD level to equal the desired
    level under the resident policy and no pyramid materialization rungs still
    in flight.  Under the native-only policy the applied level is pinned to 0,
    so target LOD == settled with nothing pending.  ``quality`` is deliberately
    NOT used: under the resident policy a reduced-input tile at the *target*
    level is legitimately labelled ``preview`` (reduced input), so exact-quality
    would never be reached for a reduced fit.
    """

    return bool(_montage_target_lod_evidence(win)["reached"])


def _montage_target_lod_evidence(win, *, tile_numbers=None) -> dict[str, object]:
    """Explain target-LOD settlement without hiding mixed/stale tiles.

    This is deliberately payload- and source-aware. Aggregate policy state can
    truthfully describe the plurality while a bounded successor still has a
    few coarser or predecessor-mapped slots; the benchmark must identify those
    exact obligations instead of reporting only a boolean timeout.
    """

    session = getattr(win, "_frame_session", None)
    if session is None:
        return {
            "reached": False,
            "settled": False,
            "desired_level": 0,
            "payload_level_counts": (),
            "missing_tiles": (),
            "coarser_tiles": (),
            "source_mismatch_tiles": (),
            "backend_mismatch_tiles": (),
            "pending_materializations": 0,
        }
    scoped_tiles = (
        None if tile_numbers is None else {int(tile) for tile in tuple(tile_numbers or ())}
    )
    settled = _montage_settled(session)
    decision = getattr(session, "lod_policy_decision", None)
    demand = getattr(decision, "demand", None)
    desired_level = 0 if demand is None else int(getattr(demand, "desired_level", 0) or 0)
    policy = "" if decision is None else str(getattr(decision, "policy", "") or "")
    active_tiles = (
        set(_active_planned_montage_tiles(session)) if scoped_tiles is None else set(scoped_tiles)
    )
    if scoped_tiles is not None:
        scoped_backlog = _visible_backlog_state(session, active_tiles)
        settled = bool(active_tiles and not scoped_backlog["visible_has_backlog"])
    payloads = dict(getattr(session, "display_tile_payloads", {}) or {})
    plan_by_number = {
        int(tile.montage_index): tile
        for tile in tuple(getattr(getattr(session, "plan", None), "tiles", ()) or ())
    }
    missing: list[int] = []
    coarser: list[int] = []
    mismatched: list[int] = []
    backend_mismatched: list[int] = []
    backend_identities = dict(
        getattr(getattr(session, "lifecycle", None), "backend_presented_identities", {}) or {}
    )
    level_counts: dict[int, int] = {}
    for tile_number in sorted(active_tiles):
        payload = payloads.get(int(tile_number))
        if payload is None:
            missing.append(int(tile_number))
            continue
        payload_level = int(getattr(getattr(payload, "lod", None), "level", 0) or 0)
        level_counts[payload_level] = level_counts.get(payload_level, 0) + 1
        if policy == "resident" and payload_level > desired_level:
            coarser.append(int(tile_number))
        planned = plan_by_number.get(int(tile_number))
        if planned is not None and int(getattr(payload, "source_index", -1)) != int(
            planned.source_index
        ):
            mismatched.append(int(tile_number))
        if backend_identities.get(int(tile_number)) != tile_ack_identity(payload):
            backend_mismatched.append(int(tile_number))
    aggregate_coarser = bool(
        scoped_tiles is None
        and decision is not None
        and demand is not None
        and policy == "resident"
        and int(getattr(decision, "applied_level", 0) or 0) > desired_level
    )
    pending_materializations = (
        len(getattr(session, "pending_rung_materializations", ()) or ())
        if scoped_tiles is None
        else 0
    )
    reached = bool(
        settled
        and not aggregate_coarser
        and not missing
        and not coarser
        and not mismatched
        and not backend_mismatched
        and pending_materializations == 0
    )
    return {
        "reached": reached,
        "settled": bool(settled),
        "desired_level": int(desired_level),
        "payload_level_counts": tuple(
            sorted((int(level), int(count)) for level, count in level_counts.items())
        ),
        "missing_tiles": tuple(missing),
        "coarser_tiles": tuple(coarser),
        "source_mismatch_tiles": tuple(mismatched),
        "backend_mismatch_tiles": tuple(backend_mismatched),
        "aggregate_coarser": bool(aggregate_coarser),
        "pending_materializations": int(pending_materializations),
    }


def _journey_lod_trace_state(win) -> dict[str, object]:
    """Snapshot live-camera demand and session/applied LOD without inference."""

    session = getattr(win, "_frame_session", None)
    if session is None:
        return {
            "session_id": 0,
            "coverage_pass_open": False,
            "camera_desired_level": None,
            "session_desired_level": None,
            "applied_level": None,
        }
    decision = getattr(session, "lod_policy_decision", None)
    demand = getattr(decision, "demand", None)
    camera_desired = None
    view_range = _montage_view_range(win)
    tile_shape = getattr(getattr(session, "plan", None), "tile_shape", None)
    viewport_shape = getattr(session, "viewport_shape", None)
    if view_range is not None and tile_shape is not None and viewport_shape is not None:
        from arrayscope.display.lod import select_lod_demand

        camera_desired = int(
            select_lod_demand(
                view_range,
                tuple(viewport_shape),
                tuple(tile_shape),
            ).desired_level
        )
    return {
        "session_id": int(getattr(session, "session_id", 0) or 0),
        "coverage_pass_open": bool(session.scheduling_policy.verdict.coverage_open),
        "camera_desired_level": camera_desired,
        "session_desired_level": (
            None if demand is None else int(getattr(demand, "desired_level", 0) or 0)
        ),
        "applied_level": (
            None if decision is None else int(getattr(decision, "applied_level", 0) or 0)
        ),
    }


def _start_journey_gesture(win, journey: str) -> str:
    counter = int(getattr(win, "_arrayscope_journey_counter", 0) or 0) + 1
    win._arrayscope_journey_counter = counter
    gesture_id = f"{journey!s}-{counter}"
    win._arrayscope_active_journey = str(journey)
    win._arrayscope_active_gesture_id = gesture_id
    win._arrayscope_active_gesture_started_ns = time.monotonic_ns()
    visual_probe = getattr(win, "_arrayscope_visual_timeline_probe", None)
    if visual_probe is not None:
        visual_probe.capture("journey-start")
    emit_trace(
        "input",
        action="journey_gesture",
        edge="start",
        journey=str(journey),
        gesture_id=gesture_id,
        backend=str(getattr(win, "_arrayscope_profile_backend", "") or ""),
        **_journey_lod_trace_state(win),
    )
    return gesture_id


def _drain_presentation_draw_for_journey_sample(
    win,
    app,
    QtCore,
    *,
    timeout_s: float = min(2.0, INTERACTION_SETTLE_HARD_LIMIT_S),
) -> bool:
    """Run the dispatcher until the journey's physical quiet edge.

    The journey freshness sampler keys on presented pixels.  A
    descriptor-only gesture (wgpu zoom-out over resident content commits
    nothing by design) produces exactly one repaint, which the on-demand
    scheduler may run a tick after the last camera step; capturing the
    journey-end sample synchronously recorded the stale predecessor frame
    and reported first_new_pixels=None (matrix v6/v7, 2026-07-18). The
    production quiet edge also releases interaction-deferred residency and
    histogram evidence. Sampling while that edge or the resulting COVERAGE
    pass is still open can close the harness before preview evidence is ready.
    Bounded and non-raising: when an injected missed redraw, interaction, or
    coverage pass never clears, the sample is still taken and the oracle stays
    red.
    """

    timeout_s = bounded_interaction_settle_timeout_s(timeout_s)
    pending_fn = getattr(getattr(win, "img_view", None), "presentationDrawPending", None)
    if not callable(pending_fn):
        return True
    interaction_active = getattr(win, "_interaction_active_now", None)

    def sample_edge_pending() -> bool:
        session = getattr(win, "_frame_session", None)
        policy = getattr(session, "scheduling_policy", None)
        verdict = getattr(policy, "verdict", None)
        return (
            bool(pending_fn())
            or bool(callable(interaction_active) and interaction_active())
            or bool(getattr(verdict, "coverage_open", False))
        )

    deadline = perf_counter() + max(0.05, timeout_s)
    while sample_edge_pending():
        if perf_counter() >= deadline:
            return False
        _process_events(app, QtCore, count=2)
        time.sleep(0.005)
    return True


def _finish_journey_gesture(
    win,
    gesture_id: str,
    *,
    reached: bool | None = None,
    app=None,
    QtCore=None,
) -> None:
    if str(getattr(win, "_arrayscope_active_gesture_id", "")) != str(gesture_id):
        return
    visual_probe = getattr(win, "_arrayscope_visual_timeline_probe", None)
    presentation_drained = None
    if visual_probe is not None and app is not None and QtCore is not None:
        presentation_drained = _drain_presentation_draw_for_journey_sample(win, app, QtCore)
    if visual_probe is not None:
        visual_probe.capture("journey-end")
    started_ns = int(getattr(win, "_arrayscope_active_gesture_started_ns", 0) or 0)
    emit_trace(
        "input",
        action="journey_gesture",
        edge="complete",
        journey=str(getattr(win, "_arrayscope_active_journey", "") or ""),
        gesture_id=str(gesture_id),
        backend=str(getattr(win, "_arrayscope_profile_backend", "") or ""),
        elapsed_ms=(0.0 if started_ns <= 0 else (time.monotonic_ns() - started_ns) / 1_000_000.0),
        reached=None if reached is None else bool(reached),
        presentation_drained=presentation_drained,
        **_journey_lod_trace_state(win),
    )
    win._arrayscope_active_journey = ""
    win._arrayscope_active_gesture_id = ""
    win._arrayscope_active_gesture_started_ns = 0


def _wait_for_target_lod(
    win, app, QtCore, *, budget_s: float, stall_grace_s: float = 2.5
) -> tuple[bool, float]:
    """Run the real Qt dispatcher until the montage reaches full target LOD.

    Returns ``(reached, elapsed_ms)``.  Bails early when the montage is clearly
    frozen (stall signature stable while no kernel work is in flight) instead of
    staring at a dead session for the whole budget.
    """

    budget_s = bounded_interaction_settle_timeout_s(budget_s)
    stall_grace_s = min(bounded_interaction_settle_timeout_s(stall_grace_s), budget_s)
    t0 = perf_counter()
    if _montage_at_target_lod(win):
        return True, 0.0
    loop = QtCore.QEventLoop()
    poll = QtCore.QTimer(loop)
    poll.setInterval(15)
    poll.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
    timeout = QtCore.QTimer(loop)
    timeout.setSingleShot(True)
    timeout.setInterval(max(1, math.ceil(float(budget_s) * 1000.0)))
    result = {"reached": False, "stall_since": None, "last_sig": None}

    def inspect() -> None:
        if _montage_at_target_lod(win):
            result["reached"] = True
            loop.quit()
            return
        session = getattr(win, "_frame_session", None)
        sig = _montage_stall_signature(session)
        if _montage_work_in_flight(session):
            result["stall_since"] = None
        elif sig != result["last_sig"]:
            result["stall_since"] = perf_counter()
        elif result["stall_since"] is not None and perf_counter() - float(
            result["stall_since"]
        ) >= float(stall_grace_s):
            loop.quit()
            return
        result["last_sig"] = sig

    poll.timeout.connect(inspect)
    timeout.timeout.connect(loop.quit)
    poll.start()
    timeout.start()
    loop.exec()
    poll.stop()
    timeout.stop()
    return bool(result["reached"] or _montage_at_target_lod(win)), (perf_counter() - t0) * 1000.0


def _lod_priority_snapshot(win, *, include_details: bool = False) -> dict[str, object]:
    """Snapshot visible target truth and physical near-residency ordering."""

    session = getattr(win, "_frame_session", None)
    if session is None:
        return {
            "active_tiles": (),
            "near_tiles": (),
            "near_resident_identities": (),
            "resident_query_available": False,
            **_montage_target_lod_evidence(win),
        }
    frame_plan = getattr(session, "frame_plan", None)
    active = {int(tile) for tile in tuple(getattr(frame_plan, "active_region_ids", ()) or ())}
    if not active:
        active = set(_active_planned_montage_tiles(session))
    near = {int(tile) for tile in tuple(getattr(frame_plan, "near_region_ids", ()) or ())} - active
    payloads = dict(getattr(session, "display_tile_payloads", {}) or {})
    resident = getattr(getattr(win, "img_view", None), "tiledPayloadResident", None)
    target_evidence = _montage_target_lod_evidence(win, tile_numbers=active)
    lifecycle_target = getattr(getattr(session, "lifecycle", None), "visible_target_settled", None)
    pipeline = getattr(session, "pipeline", None)
    diagnostic_rows = getattr(session, "diagnostic_tile_identity_rows", None)
    desired_level = int(target_evidence.get("desired_level", 0) or 0)
    resident_identities = tuple(
        sorted(
            (
                int(tile),
                repr(getattr(payloads[int(tile)], "source_id", None)),
            )
            for tile in near
            if int(tile) in payloads
            and int(getattr(getattr(payloads[int(tile)], "lod", None), "level", 0) or 0)
            <= desired_level
            and callable(resident)
            and bool(resident(payloads[int(tile)]))
        )
    )
    all_resident_identities = tuple(
        sorted(
            (
                int(tile),
                repr(getattr(payload, "source_id", None)),
            )
            for tile, payload in payloads.items()
            if callable(resident) and bool(resident(payload))
        )
    )
    return {
        "active_tiles": tuple(sorted(active)),
        "near_tiles": tuple(sorted(near)),
        "near_resident_identities": resident_identities,
        "all_resident_identities": all_resident_identities,
        "resident_query_available": callable(resident),
        "lifecycle_visible_target_settled": bool(callable(lifecycle_target) and lifecycle_target()),
        "pipeline_states": (
            tuple(getattr(pipeline, "last_plan_states", ()) or ()) if include_details else ()
        ),
        "pipeline_steps": (
            tuple(getattr(pipeline, "last_plan_steps", ()) or ()) if include_details else ()
        ),
        "active_diagnostics": (
            tuple(
                row
                for row in tuple(
                    diagnostic_rows(limit=max(8, len(active))) if callable(diagnostic_rows) else ()
                )
                if int(row.get("tile", -1)) in active
            )
            if include_details
            else ()
        ),
        **target_evidence,
    }


def _wait_for_visible_target_then_observe_near(win, app, QtCore) -> dict[str, object]:
    """Require visible target settlement before newly resident near payloads.

    Once the visible set is correct, keep the dispatcher alive briefly to
    observe whether speculative residency starts. Existing near residency is
    a cache hit and is excluded from the ordering assertion.
    """

    started = perf_counter()
    deadline = started + float(ZOOMPAN_CHECKPOINT_SETTLE_S)
    baseline = _lod_priority_snapshot(win)
    baseline_near = {tuple(row) for row in baseline["near_resident_identities"]}
    baseline_resident = {tuple(row) for row in baseline.get("all_resident_identities", ())}
    before_visible: set[tuple[object, ...]] = set()
    first_near_before_visible_evidence = None
    visible_reached_at = None
    last = baseline
    while perf_counter() < deadline:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
        last = _lod_priority_snapshot(win)
        current_near = {tuple(row) for row in last["near_resident_identities"]}
        if bool(last.get("reached", False)):
            if visible_reached_at is None:
                visible_reached_at = perf_counter()
        else:
            # A range signal can precede its coalesced viewport-plan retarget.
            # The predecessor is briefly "settled" until that plan lands; it
            # is not evidence for the new camera. Require continuous target
            # truth through the observation window.
            visible_reached_at = None
            new_before = current_near - baseline_resident
            if new_before and first_near_before_visible_evidence is None:
                first_near_before_visible_evidence = _lod_priority_snapshot(
                    win,
                    include_details=True,
                )
            before_visible.update(new_before)
        if visible_reached_at is not None and perf_counter() - visible_reached_at >= float(
            ZOOMPAN_NEAR_OBSERVE_S
        ):
            break
        QtCore.QThread.msleep(1)
    final = _lod_priority_snapshot(
        win,
        include_details=visible_reached_at is None,
    )
    final_near = {tuple(row) for row in final["near_resident_identities"]}
    return {
        "visible_target_reached": bool(final.get("reached", False)),
        "visible_settle_ms": (
            float((visible_reached_at - started) * 1000.0)
            if visible_reached_at is not None
            else float((perf_counter() - started) * 1000.0)
        ),
        "active_tiles": final["active_tiles"],
        "near_tiles": final["near_tiles"],
        "near_resident_before_count": len(baseline_near),
        "near_new_before_visible": tuple(sorted(before_visible)),
        "near_new_before_visible_count": len(before_visible),
        "first_near_before_visible_evidence": first_near_before_visible_evidence,
        "near_new_after_visible_count": len(final_near - baseline_resident - before_visible),
        "resident_query_available": bool(final["resident_query_available"]),
        "target_evidence": final,
    }


def _lod_state_record(win) -> dict[str, object]:
    """Snapshot the LOD demand/applied/preview counters for a phase record."""

    try:
        montage = win.collect_runtime_diagnostics().montage
    except Exception:
        return {}
    session = getattr(win, "_frame_session", None)
    visible = 0 if session is None else len(getattr(session, "visible_tile_numbers", ()) or ())
    return {
        "lod_desired_factor": int(getattr(montage, "tile_lod_desired_factor", 0) or 0),
        "lod_applied_factor": int(getattr(montage, "tile_lod_applied_factor", 0) or 0),
        "lod_applied_level": int(getattr(montage, "tile_lod_applied_level", 0) or 0),
        "lod_desired_factor_xy": tuple(
            int(v) for v in getattr(montage, "tile_lod_desired_factor_xy", ()) or ()
        ),
        "lod_applied_factor_xy": tuple(
            int(v) for v in getattr(montage, "tile_lod_applied_factor_xy", ()) or ()
        ),
        "lod_preview_presentations": int(
            getattr(montage, "tile_lod_preview_presentations", 0) or 0
        ),
        "lod_pending_materializations": int(
            getattr(montage, "tile_lod_pending_materializations", 0) or 0
        ),
        "lod_pyramid_bytes": int(getattr(montage, "tile_lod_pyramid_bytes", 0) or 0),
        "lod_reason": str(getattr(montage, "tile_lod_reason", "")),
        "lod_visible_tiles": int(visible),
    }


def _montage_view_range(win):
    """Return the live camera range as ``((x0, x1), (y0, y1))`` or None."""

    try:
        view_range = win.img_view.getView().viewRange()
        return (
            (float(view_range[0][0]), float(view_range[0][1])),
            (float(view_range[1][0]), float(view_range[1][1])),
        )
    except Exception:
        return None


def _apply_view_range(win, x_range, y_range) -> None:
    """Push a new camera range; the range-changed signal retargets the montage."""

    image_view = win.img_view
    note_interaction = getattr(win, "_note_viewport_interaction", None)
    if callable(note_interaction):
        # setRange is the deterministic benchmark transport, but semantically
        # these are wheel/pan inputs. Use the production pointer reason so a
        # restored viewport-continuity transaction releases exactly as it does
        # for an accepted wheel/pan gesture; custom profile reasons leave the
        # saved camera authoritative and make demand measurements meaningless.
        note_interaction("range-pointer")
    # QApplication.mouseButtons() is empty for this deterministic transport.
    # Carry the same synchronous wheel identity that ImageViewShell.eventFilter
    # publishes for a real wheel event so ViewportBridge admits the production
    # interactive retarget path instead of classifying the gesture as restore
    # or fit replay.
    image_view._viewport_wheel_range_pending = True
    try:
        view = image_view.getView()
        view.setRange(
            xRange=(float(x_range[0]), float(x_range[1])),
            yRange=(float(y_range[0]), float(y_range[1])),
            padding=0,
        )
        # The benchmark transport has no real QWheelEvent. If the production
        # bridge cannot identify an extant committed tiled frame, its ordinary
        # signal path deliberately does not schedule. Deliver the synthetic
        # gesture through the same canonical retarget owner rather than
        # deriving or mutating LOD state in the harness.
        retarget = getattr(win, "retarget_montage_viewport", None)
        if callable(retarget):
            retarget()
    finally:
        # The bridge normally consumes the flag synchronously. Clear it here
        # as well for a constrained/no-op range which emits no signal.
        image_view._viewport_wheel_range_pending = False


def _scaled_view_range(view_range, span_scale, center_frac=(0.5, 0.5)):
    """Scale a view range's span (``>1`` zooms out, ``<1`` zooms in)."""

    (x0, x1), (y0, y1) = view_range
    cx = x0 + (x1 - x0) * float(center_frac[0])
    cy = y0 + (y1 - y0) * float(center_frac[1])
    half_x = (x1 - x0) * float(span_scale) * 0.5
    half_y = (y1 - y0) * float(span_scale) * 0.5
    return ((cx - half_x, cx + half_x), (cy - half_y, cy + half_y))


def _panned_view_range(view_range, dx_frac, dy_frac):
    """Shift a view range by a fraction of its span (pan)."""

    (x0, x1), (y0, y1) = view_range
    dx = (x1 - x0) * float(dx_frac)
    dy = (y1 - y0) * float(dy_frac)
    return ((x0 + dx, x1 + dx), (y0 + dy, y1 + dy))


def _maximum_zoomout_view_range(win, app, QtCore, base_range):
    """Ask the real viewport constraint for its maximum zoom-out range."""

    requested = _scaled_view_range(base_range, ZOOMPAN_MAX_OUT_REQUEST_SCALE)
    _apply_view_range(win, requested[0], requested[1])
    _process_events(app, QtCore, count=1)
    return _montage_view_range(win) or requested


def _few_tile_view_range(win, *, center_fraction: float = 0.5, tiles_across: float = 0.9):
    """Build a deterministic camera target spanning only a few montage tiles."""

    session = getattr(win, "_frame_session", None)
    tiles = tuple(getattr(getattr(session, "plan", None), "tiles", ()) or ())
    if not tiles:
        return None
    index = min(len(tiles) - 1, max(0, round((len(tiles) - 1) * float(center_fraction))))
    tile = tiles[index]
    center_x = float(tile.x0) + float(tile.width) * 0.5
    center_y = float(tile.y0) + float(tile.height) * 0.5
    half_x = max(1.0, float(tile.width) * float(tiles_across) * 0.5)
    half_y = max(1.0, float(tile.height) * float(tiles_across) * 0.5)
    return ((center_x - half_x, center_x + half_x), (center_y - half_y, center_y + half_y))


def _full_montage_view_range(win):
    """Return the tight content range, unlike the deliberately distant limit."""

    session = getattr(win, "_frame_session", None)
    shape = tuple(
        getattr(getattr(getattr(session, "plan", None), "geometry", None), "display_shape", ())
        or ()
    )
    if len(shape) < 2:
        shape = tuple(getattr(getattr(session, "plan", None), "display_shape", ()) or ())
    if len(shape) < 2:
        return None
    height, width = max(1, int(shape[0])), max(1, int(shape[1]))
    return ((0.0, float(width)), (0.0, float(height)))


def _glide_view_range(
    win,
    app,
    QtCore,
    probe,
    target_range,
    *,
    frames: int,
    fps: float = 60.0,
    frame_action=None,
) -> dict[str, object]:
    """Interpolate the camera range to ``target_range`` at ``fps``, one push/frame.

    Measures how the interaction hot path keeps up under a continuous, paced
    view change (the zoom/pan analogue of a fast scroll).  Never waits for LOD
    to settle between frames — that is what the settle step after it measures.
    """

    start_range = _montage_view_range(win)
    if start_range is None or int(frames) < 1:
        return {
            "frames": 0,
            "elapsed_ms": 0.0,
            "max_gap_ms": 0.0,
            "p95_gap_ms": 0.0,
            "p99_gap_ms": 0.0,
            "achieved_fps": 0.0,
        }
    (sx0, sx1), (sy0, sy1) = start_range
    (tx0, tx1), (ty0, ty1) = target_range
    interval = 1.0 / max(1.0, float(fps))
    note_interaction = getattr(win, "_note_viewport_interaction", None)
    if callable(note_interaction):
        # Establish interactive policy before the first frame timer is
        # admitted. Otherwise a quiet-edge residency continuation from the
        # preceding checkpoint can occupy the dispatcher before frame one has
        # a chance to announce the new gesture.
        note_interaction("profile-gesture")
    if probe is not None:
        probe.reset()
    ui_work_buffer = getattr(
        getattr(win, "resource_governor", None), "_recent_ui_work_observations", ()
    )
    ui_work_start = len(ui_work_buffer)
    vispy_draw_start = _vispy_draw_count(win)
    t0 = perf_counter()
    frames = int(frames)
    input_call_ms: list[float] = []
    view_range_call_ms: list[float] = []
    frame_action_call_ms: list[float] = []
    loop = QtCore.QEventLoop()
    frame_timer = QtCore.QTimer(loop)
    frame_timer.setInterval(max(1, round(interval * 1000.0)))
    frame_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
    callback_error: list[BaseException] = []
    frame_index = {"value": 0}

    def push_frame() -> None:
        index = int(frame_index["value"]) + 1
        frame_index["value"] = index
        alpha = index / frames
        x_range = (sx0 + (tx0 - sx0) * alpha, sx1 + (tx1 - sx1) * alpha)
        y_range = (sy0 + (ty0 - sy0) * alpha, sy1 + (ty1 - sy1) * alpha)
        input_start = perf_counter()
        try:
            if callable(note_interaction):
                # Programmatic ViewBox changes are correctly ignored by the
                # app's generic range bridge (restore/fit are not gestures).
                # This harness is deliberately synthesizing a pointer-speed
                # gesture, so drive the production interaction governor
                # explicitly before every frame and let its ordinary quiet
                # edge settle the final LOD afterwards.
                note_interaction("profile-gesture")
            view_range_start = perf_counter()
            _apply_view_range(win, x_range, y_range)
            view_range_call_ms.append((perf_counter() - view_range_start) * 1000.0)
            if callable(frame_action):
                frame_action_start = perf_counter()
                frame_action(index, frames)
                frame_action_call_ms.append((perf_counter() - frame_action_start) * 1000.0)
        except BaseException as exc:  # re-raise outside Qt callback boundary
            callback_error.append(exc)
            loop.quit()
            return
        input_call_ms.append((perf_counter() - input_start) * 1000.0)
        if index >= frames:
            loop.quit()

    frame_timer.timeout.connect(push_frame)
    frame_timer.start()
    loop.exec()
    frame_timer.stop()
    if callback_error:
        raise callback_error[0]
    elapsed_ms = (perf_counter() - t0) * 1000.0
    ui_work = tuple(ui_work_buffer)[ui_work_start:]
    tile_commits = tuple(item for item in ui_work if item.channel == "tile_layer_commit")
    physical_draws = tuple(
        item for item in ui_work if item.channel in {"vispy_canvas_draw", "graphics_view_paint"}
    )
    kernel_drains = tuple(item for item in ui_work if item.channel == "kernel_bridge_drain")
    return {
        "frames": frames,
        "elapsed_ms": elapsed_ms,
        "max_gap_ms": 0.0 if probe is None else float(probe.max_gap_ms),
        "p95_gap_ms": 0.0 if probe is None else float(probe.percentile_ms(95) or 0.0),
        "p99_gap_ms": 0.0 if probe is None else float(probe.percentile_ms(99) or 0.0),
        "achieved_fps": (frames / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0,
        "input_call_max_ms": float(max(input_call_ms) if input_call_ms else 0.0),
        "input_call_p95_ms": float(_percentile(tuple(input_call_ms), 95.0)),
        "view_range_call_max_ms": float(max(view_range_call_ms) if view_range_call_ms else 0.0),
        "view_range_call_p95_ms": float(_percentile(tuple(view_range_call_ms), 95.0)),
        "frame_action_call_max_ms": float(
            max(frame_action_call_ms) if frame_action_call_ms else 0.0
        ),
        "frame_action_call_p95_ms": float(_percentile(tuple(frame_action_call_ms), 95.0)),
        "tile_commit_count": len(tile_commits),
        "tile_commit_total_ms": float(sum(item.elapsed_ms for item in tile_commits)),
        "physical_draw_count": (
            max(0, _vispy_draw_count(win) - vispy_draw_start) or len(physical_draws)
        ),
        "physical_draw_total_ms": float(sum(item.elapsed_ms for item in physical_draws)),
        "kernel_drain_count": len(kernel_drains),
        "kernel_drain_total_ms": float(sum(item.elapsed_ms for item in kernel_drains)),
        "ui_work_total_ms": float(sum(item.elapsed_ms for item in ui_work)),
    }


class _VerboseTileTrace:
    """Opt-in transition history for source/LOD/backend slot correctness."""

    def __init__(self, win):
        self.win = win
        self.started = perf_counter()
        self.snapshots: list[dict[str, object]] = []
        self._target_changed_at: dict[int, float] = {}
        self._target_source_by_slot: dict[int, int] = {}
        self._presented_changed_at: dict[int, float] = {}
        self._presented_identity_by_slot: dict[int, object] = {}

    def capture(self, event: str, *, requested_start: int | None = None) -> None:
        now = perf_counter()
        desired_indices = tuple(
            int(index)
            for index in tuple(
                getattr(getattr(self.win, "view_state", None), "montage_indices", ()) or ()
            )
        )
        session = getattr(self.win, "_frame_session", None)
        plan_tiles = tuple(getattr(getattr(session, "plan", None), "tiles", ()) or ())
        planned = {int(tile.montage_index): int(tile.source_index) for tile in plan_tiles}
        payloads = dict(getattr(session, "display_tile_payloads", {}) or {})
        lifecycle = getattr(session, "lifecycle", None)
        backend = dict(getattr(lifecycle, "backend_presented_identities", {}) or {})
        physical_rows_fn = getattr(
            getattr(self.win, "img_view", None),
            "tileTruthPhysicalRows",
            None,
        )
        physical_rows = dict(physical_rows_fn() or {}) if callable(physical_rows_fn) else {}
        physical_layer = getattr(getattr(self.win, "img_view", None), "_montage_tile_layer", None)
        physical_states = dict(getattr(physical_layer, "states", {}) or {})
        composite = getattr(physical_layer, "_composite_item", None)
        revision = int(
            getattr(getattr(session, "tile_presentation_state", None), "revision", 0) or 0
        )
        rows: list[dict[str, object]] = []
        slot_count = max(len(desired_indices), len(plan_tiles), len(payloads), len(backend))
        for slot in range(slot_count):
            desired_source = desired_indices[slot] if slot < len(desired_indices) else None
            if (
                desired_source is not None
                and self._target_source_by_slot.get(slot) != desired_source
            ):
                self._target_source_by_slot[slot] = desired_source
                self._target_changed_at[slot] = now
            payload = payloads.get(slot)
            payload_source = None if payload is None else int(getattr(payload, "source_index", -1))
            backend_identity = backend.get(slot)
            if self._presented_identity_by_slot.get(slot) != backend_identity:
                self._presented_identity_by_slot[slot] = backend_identity
                self._presented_changed_at[slot] = now
            record = None if lifecycle is None else lifecycle.peek(slot)
            lod = None if payload is None else getattr(payload, "lod", None)
            physical = physical_states.get(slot)
            raw_physical_row = dict(physical_rows.get(slot) or {})
            physical_row = _verbose_physical_row(raw_physical_row)
            physical_image = (
                None
                if physical is None
                else getattr(getattr(physical, "item", None), "image", None)
            )
            physical_white_fraction = None
            if physical_image is not None:
                array = np.asarray(physical_image)
                stride_y = max(1, int(array.shape[0]) // 16)
                stride_x = max(1, int(array.shape[1]) // 16)
                sample = array[::stride_y, ::stride_x, ...]
                if sample.ndim >= 3 and int(sample.shape[-1]) in (3, 4):
                    physical_white_fraction = float(
                        np.mean(np.all(sample[..., :3] >= 250, axis=-1))
                    )
                else:
                    high = float(getattr(physical, "levels", (0.0, float("inf")))[1])
                    physical_white_fraction = float(np.mean(sample >= high))
            rows.append(
                {
                    "slot": int(slot),
                    "desired_source_index": desired_source,
                    "planned_source_index": planned.get(slot),
                    "payload_source_index": payload_source,
                    "payload_source_id": _trace_identity(getattr(payload, "source_id", None)),
                    "backend_source_id": _trace_identity(backend_identity),
                    "lod_level": None if lod is None else int(getattr(lod, "level", 0) or 0),
                    "lod_factor": None if lod is None else int(getattr(lod, "factor", 1) or 1),
                    "quality": None if payload is None else str(getattr(payload, "quality", "")),
                    "lifecycle_phase": None
                    if record is None
                    else str(
                        getattr(
                            getattr(record, "phase", None), "value", getattr(record, "phase", "")
                        )
                    ),
                    "target_age_ms": float((now - self._target_changed_at.get(slot, now)) * 1000.0),
                    "presented_age_ms": float(
                        (now - self._presented_changed_at.get(slot, now)) * 1000.0
                    ),
                    "payload_matches_desired": bool(payload_source == desired_source),
                    "plan_matches_desired": bool(planned.get(slot) == desired_source),
                    "backend_matches_payload": bool(
                        payload is not None and backend_identity == tile_ack_identity(payload)
                    ),
                    "physical_matches_payload": bool(
                        payload is not None
                        and raw_physical_row.get("physical_acknowledged_identity")
                        == tile_ack_identity(payload)
                    ),
                    **physical_row,
                    "physical_source_id": _trace_identity(
                        None if physical is None else getattr(physical, "source_array_id", None)
                    ),
                    "physical_levels": None
                    if physical is None
                    else tuple(float(v) for v in physical.levels),
                    "physical_white_fraction": physical_white_fraction,
                    "physical_render_required": bool(
                        physical is not None and getattr(physical.item, "_renderRequired", False)
                    ),
                }
            )
        self.snapshots.append(
            {
                "elapsed_ms": float((now - self.started) * 1000.0),
                "event": str(event),
                "requested_start": None if requested_start is None else int(requested_start),
                "session_id": None
                if session is None
                else int(getattr(session, "session_id", 0) or 0),
                "presentation_revision": revision,
                "atomic_successor_pending": bool(
                    session is not None and getattr(session, "atomic_successor_pending", False)
                ),
                "camera_view_range": _montage_view_range(self.win),
                "composite_cache_reason": str(
                    getattr(composite, "composite_cache_reason", "") or ""
                ),
                "composite_has_pixmap": bool(
                    getattr(composite, "_composite_pixmap", None) is not None
                    and not composite._composite_pixmap.isNull()
                ),
                "tiles": rows,
            }
        )


def _trace_identity(value, *, limit: int = 240) -> str | None:
    if value is None:
        return None
    text = repr(value)
    return text if len(text) <= int(limit) else text[: max(0, int(limit) - 3)] + "..."


def _verbose_physical_row(row) -> dict[str, object]:
    """Return JSON-safe physical evidence for one backend tile slot."""

    row = dict(row or {})
    keep = (
        "physical_page",
        "physical_slot",
        "physical_texture_kind",
        "physical_storage_mode",
        "physical_texture_dtype",
        "physical_texture_shape",
        "physical_real_plane_identity",
        "physical_imag_plane_identity",
        "physical_mapping_mode",
        "physical_component_mode",
        "physical_levels",
        "physical_shader_mapping_key",
    )
    return {
        **{key: row.get(key) for key in keep},
        "physical_acknowledged_identity": _trace_identity(
            row.get("physical_acknowledged_identity")
        ),
    }


def _fast_scroll_60fps(
    win,
    *,
    montage_axis,
    columns,
    indices,
    size,
    start,
    low,
    high,
    probe,
    app,
    QtCore,
    tile_trace=None,
) -> dict[str, object]:
    """Cold-forward scroll, short fast reverse, then a tiny oscillating tail."""

    interval = 1.0 / 60.0
    indices = tuple(indices)
    tile_count = len(indices)
    size = max(1, min(int(size), tile_count if tile_count > 0 else 1))
    low = max(0, int(low))
    high = min(max(0, int(tile_count) - int(size)), int(high))
    if high <= low:
        high = low
    camera_before = _montage_view_range(win)
    if probe is not None:
        probe.reset()
    draw_start = _vispy_tile_presentation_draw_count(win)
    ui_work_buffer = getattr(
        getattr(win, "resource_governor", None), "_recent_ui_work_observations", ()
    )
    ui_work_start = len(ui_work_buffer)
    t0 = perf_counter()
    t0 + float(SCROLL_FAST_DURATION_S)
    current = min(max(int(start), low), high)
    frames = 0
    reached_min = current
    reached_max = current
    primary_frames = 0
    reverse_frames = 0
    tail_frames = 0
    primary_deadline = t0 + float(SCROLL_FAST_DURATION_S) * 0.58
    reverse_deadline = t0 + float(SCROLL_FAST_DURATION_S) * 0.94
    reverse_target = low
    tail_low = max(low, reverse_target - max(3, size // 8))
    tail_high = min(high, reverse_target + max(3, size // 8))
    tail_direction = 1
    input_call_ms: list[float] = []
    input_state_build_ms: list[float] = []
    input_state_apply_ms: list[float] = []
    input_render_request_ms: list[float] = []
    # Begin at the settled centre and move continuously into cold slices.  The
    # former low->high formula teleported centre->low on the first tick, giving
    # the renderer zero resident overlap; every following target then outran
    # loading and the benchmark showed only centre-priority tiles moving until
    # its final pause.  A continuous centre->high sweep tests the actual cheap
    # N-k resident shuffle, then the faster high->low reverse stresses both
    # supersession and newly-entering slices.  The final 6% oscillates briefly.
    loop = QtCore.QEventLoop()
    frame_timer = QtCore.QTimer(loop)
    frame_timer.setInterval(max(1, round(interval * 1000.0)))
    frame_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
    stop_timer = QtCore.QTimer(loop)
    stop_timer.setSingleShot(True)
    stop_timer.setInterval(max(1, round(float(SCROLL_FAST_DURATION_S) * 1000.0)))
    callback_error: list[BaseException] = []

    def push_frame() -> None:
        nonlocal frames, current, reached_min, reached_max
        nonlocal primary_frames, reverse_frames, tail_frames, tail_direction
        frames += 1
        now = perf_counter()
        if now < primary_deadline:
            primary_frames += 1
            fraction = min(1.0, (now - t0) / max(1e-9, primary_deadline - t0))
            current = round(start + (high - start) * fraction)
        elif now < reverse_deadline:
            reverse_frames += 1
            fraction = min(
                1.0, (now - primary_deadline) / max(1e-9, reverse_deadline - primary_deadline)
            )
            current = round(high + (reverse_target - high) * fraction)
        else:
            tail_frames += 1
            current += tail_direction
            if current >= tail_high:
                current = tail_high
                tail_direction = -1
            elif current <= tail_low:
                current = tail_low
                tail_direction = 1
        reached_min = min(reached_min, current)
        reached_max = max(reached_max, current)
        if tile_trace is not None:
            tile_trace.capture("fast-before-input", requested_start=current)
        input_start = perf_counter()
        try:
            input_parts = _scroll_montage_window(
                win,
                montage_axis=montage_axis,
                columns=columns,
                indices=indices,
                window_start=current,
                size=size,
                interactive=True,
            )
        except BaseException as exc:  # re-raise outside the Qt callback boundary
            callback_error.append(exc)
            loop.quit()
            return
        input_call_ms.append((perf_counter() - input_start) * 1000.0)
        input_state_build_ms.append(float(input_parts["state_build_ms"]))
        input_state_apply_ms.append(float(input_parts["state_apply_ms"]))
        input_render_request_ms.append(float(input_parts["render_request_ms"]))
        if tile_trace is not None:
            tile_trace.capture("fast-after-input", requested_start=current)

    frame_timer.timeout.connect(push_frame)
    stop_timer.timeout.connect(loop.quit)
    frame_timer.start()
    stop_timer.start()
    loop.exec()
    frame_timer.stop()
    stop_timer.stop()
    if callback_error:
        raise callback_error[0]
    elapsed_ms = (perf_counter() - t0) * 1000.0
    camera_after = _montage_view_range(win)
    camera_drift = 0.0
    if camera_before is not None and camera_after is not None:
        camera_drift = max(
            abs(float(after) - float(before))
            for before_axis, after_axis in zip(camera_before, camera_after, strict=False)
            for before, after in zip(before_axis, after_axis, strict=False)
        )
    draw_delta = max(0, _vispy_tile_presentation_draw_count(win) - int(draw_start))
    ui_work = tuple(ui_work_buffer)[ui_work_start:]
    channel_counts: dict[str, int] = {}
    channel_ms: dict[str, float] = {}
    for item in ui_work:
        channel_counts[item.channel] = int(channel_counts.get(item.channel, 0)) + 1
        channel_ms[item.channel] = float(channel_ms.get(item.channel, 0.0)) + float(item.elapsed_ms)
    return {
        "fast_scroll_duration_s": float(SCROLL_FAST_DURATION_S),
        "fast_scroll_input_frames": int(frames),
        "fast_scroll_cold_forward_frames": int(primary_frames),
        "fast_scroll_reverse_frames": int(reverse_frames),
        "fast_scroll_tail_oscillation_frames": int(tail_frames),
        "fast_scroll_reverse_index_distance": int(max(0, high - reverse_target)),
        "fast_scroll_index_span_reached": int(reached_max - reached_min),
        "fast_scroll_draw_count_delta": int(draw_delta),
        "fast_scroll_elapsed_ms": elapsed_ms,
        "fast_scroll_max_gap_ms": 0.0 if probe is None else float(probe.max_gap_ms),
        "fast_scroll_gc_max_pause_ms": 0.0 if probe is None else float(probe.gc_max_pause_ms),
        "fast_scroll_gc_pause_count": 0 if probe is None else len(probe.gc_pauses_ms),
        "fast_scroll_gc_pauses": () if probe is None else tuple(probe.gc_pauses),
        "fast_scroll_p95_gap_ms": 0.0 if probe is None else float(probe.percentile_ms(95) or 0.0),
        "fast_scroll_p99_gap_ms": 0.0 if probe is None else float(probe.percentile_ms(99) or 0.0),
        "fast_scroll_achieved_input_fps": (frames / (elapsed_ms / 1000.0))
        if elapsed_ms > 0
        else 0.0,
        "fast_scroll_input_call_max_ms": float(max(input_call_ms) if input_call_ms else 0.0),
        "fast_scroll_input_call_p95_ms": float(_percentile(tuple(input_call_ms), 95.0)),
        "fast_scroll_state_build_max_ms": float(
            max(input_state_build_ms) if input_state_build_ms else 0.0
        ),
        "fast_scroll_state_apply_max_ms": float(
            max(input_state_apply_ms) if input_state_apply_ms else 0.0
        ),
        "fast_scroll_render_request_max_ms": float(
            max(input_render_request_ms) if input_render_request_ms else 0.0
        ),
        "fast_scroll_end_start": int(current),
        "fast_scroll_camera_before": camera_before,
        "fast_scroll_camera_after": camera_after,
        "fast_scroll_camera_max_drift": float(camera_drift),
        "fast_scroll_ui_work_total_ms": float(sum(item.elapsed_ms for item in ui_work)),
        "fast_scroll_ui_channel_counts": channel_counts,
        "fast_scroll_ui_channel_ms": channel_ms,
    }


def _slow_scroll_lod_paced(
    win,
    *,
    montage_axis,
    columns,
    indices,
    size,
    start,
    steps,
    probe,
    app,
    QtCore,
    tile_trace=None,
) -> dict[str, object]:
    """Slow index scrub paced by full target-LOD completion (not fixed time).

    Each single-index step advances the window, then waits until the montage has
    converged to full target LOD before the next step — measuring per-step
    convergence latency and any steps that never reach target LOD.
    """

    settle_times: list[float] = []
    max_gaps: list[float] = []
    step_evidence: list[dict[str, object]] = []
    unreached = 0
    indices = tuple(indices)
    tile_count = len(indices)
    size = max(1, min(int(size), tile_count if tile_count > 0 else 1))
    current = int(start)
    max_start = max(0, tile_count - size)
    for _step_index in range(int(steps)):
        current = min(current + 1, max_start)
        gesture_id = _start_journey_gesture(win, "index_scroll")
        if tile_trace is not None:
            tile_trace.capture("slow-before-input", requested_start=current)
        if probe is not None:
            probe.reset()
        _scroll_montage_window(
            win,
            montage_axis=montage_axis,
            columns=columns,
            indices=indices,
            window_start=current,
            size=size,
        )
        if tile_trace is not None:
            tile_trace.capture("slow-after-input", requested_start=current)
        reached, settle_ms = _wait_for_target_lod(
            win, app, QtCore, budget_s=SCROLL_SLOW_LOD_BUDGET_S
        )
        # Target-LOD settlement is semantic session truth.  The journey's
        # pixel oracle samples the physical surface, so do not close the
        # gesture (and capture its endpoint screenshot) until the matching
        # backend presentation request has actually drawn.
        _wait_for_tile_presentation_draw(win, app, QtCore)
        # Do not let the next index input supersede this step's phase-1
        # evidence. The target-LOD observation above deliberately records its
        # 3 s performance result, but journey coverage is eventual semantic
        # truth and must remain scoped to the gesture that produced it.
        _wait_for_coverage_pass_close(win, app, QtCore)
        _finish_journey_gesture(win, gesture_id, reached=bool(reached), app=app, QtCore=QtCore)
        if tile_trace is not None:
            tile_trace.capture("slow-settled", requested_start=current)
        settle_times.append(float(settle_ms))
        max_gaps.append(0.0 if probe is None else float(probe.max_gap_ms))
        evidence = _montage_target_lod_evidence(win)
        session = getattr(win, "_frame_session", None)
        pipeline = None if session is None else getattr(session, "pipeline", None)
        failure_details = {}
        if not reached and session is not None:
            diagnostic_rows = getattr(session, "diagnostic_tile_identity_rows", None)
            failure_details = {
                "dirty_tiles": tuple(
                    int(tile) for tile in getattr(session, "dirty_payloads", ()) or ()
                ),
                "pending_upserts": tuple(
                    int(tile) for tile in getattr(session, "pending_payload_upserts", ()) or ()
                ),
                "pending_removals": tuple(
                    int(tile) for tile in getattr(session, "pending_removals", ()) or ()
                ),
                "pipeline_states": tuple(getattr(pipeline, "last_plan_states", ()) or ()),
                "pipeline_steps": tuple(getattr(pipeline, "last_plan_steps", ()) or ()),
                "pipeline_intent_interactive": bool(
                    getattr(getattr(pipeline, "_current_intent", None), "interactive", False)
                ),
                "pipeline_pending_admissions": len(
                    tuple(getattr(pipeline, "_pending_admissions", ()) or ())
                ),
                "pipeline_admitted_steps": len(
                    tuple(getattr(pipeline, "_admitted_step_identities", ()) or ())
                ),
                "pipeline_admission_continuation_armed": bool(
                    getattr(pipeline, "_admission_continuation_armed", False)
                ),
                "kernel_lanes": dict(
                    getattr(
                        getattr(getattr(win, "kernel", None), "diagnostics", lambda: None)(),
                        "lanes",
                        {},
                    )
                    or {}
                ),
                "commit_outcome": str(
                    getattr(getattr(win, "renderer", None), "_last_montage_commit_outcome", "")
                    or ""
                ),
                "atomic_fast_reject_reason": str(
                    getattr(
                        getattr(win, "renderer", None),
                        "_last_montage_atomic_fast_reject_reason",
                        "",
                    )
                    or ""
                ),
                "atomic_successor_pending": bool(
                    getattr(session, "atomic_successor_pending", False)
                ),
                "flush_pending": bool(getattr(session, "flush_pending", False)),
                "final_commit_pending": bool(getattr(session, "final_commit_pending", False)),
                "replan_gate_armed": bool(
                    getattr(getattr(win, "renderer", None), "_montage_replan_gate_armed", False)
                ),
                "diagnostic_tiles": (
                    tuple(diagnostic_rows(limit=5)) if callable(diagnostic_rows) else ()
                ),
            }
        step_evidence.append(
            {
                "step": int(_step_index),
                "window_start": int(current),
                "reached": bool(reached),
                "settle_ms": float(settle_ms),
                "max_gap_ms": float(max_gaps[-1]),
                **evidence,
                **failure_details,
            }
        )
        if not reached:
            unreached += 1
    return {
        "slow_scroll_steps": int(steps),
        "slow_scroll_max_gap_ms": float(max(max_gaps) if max_gaps else 0.0),
        "slow_scroll_mean_settle_ms": float(sum(settle_times) / len(settle_times))
        if settle_times
        else 0.0,
        "slow_scroll_max_settle_ms": float(max(settle_times) if settle_times else 0.0),
        "slow_scroll_unreached_target_steps": int(unreached),
        "slow_scroll_step_evidence": step_evidence,
    }


def _apply_montage_scroll_pattern(
    win,
    *,
    montage_axis,
    columns,
    indices,
    window_size,
    probe=None,
    app=None,
    QtCore=None,
    verbose_tile_trace: bool = False,
) -> dict[str, object]:
    """Slide one explicit small grid across the full montage index range."""
    indices = tuple(indices)
    selected_count = len(indices)
    if selected_count <= 0:
        return {
            "scroll_window_size": 0,
            "scroll_center_band": [0, 0],
            "scroll_fast_input_frames": 0,
        }

    size = max(1, min(int(window_size), selected_count))
    max_start = max(0, selected_count - size)
    # Scroll within the content-bearing CENTRE of the stack: the outer slices are
    # near-empty and give meaningless levels.  Ping-pong across a central band.
    center = max(0, selected_count // 2 - size // 2)
    band = min(max_start // 2, max(size, 80))
    low = max(0, min(center - band, max_start))
    high = min(max_start, center + band)
    start = min(max(center, low), high)
    tile_trace = _VerboseTileTrace(win) if verbose_tile_trace else None
    _scroll_montage_window(
        win,
        montage_axis=montage_axis,
        columns=columns,
        indices=indices,
        window_start=start,
        size=size,
    )
    if app is not None and QtCore is not None:
        prep_reached, prep_settle_ms = _wait_for_target_lod(
            win,
            app,
            QtCore,
            budget_s=INTERACTION_SETTLE_HARD_LIMIT_S,
        )
    else:
        prep_reached, prep_settle_ms = False, 0.0
    auto = getattr(win, "auto_window_levels", None) or getattr(
        getattr(win, "renderer", None), "auto_window_levels", None
    )
    if callable(auto):
        auto()
    if app is not None and QtCore is not None:
        auto_reached, auto_settle_ms = _wait_for_target_lod(
            win,
            app,
            QtCore,
            budget_s=INTERACTION_SETTLE_HARD_LIMIT_S,
        )
    else:
        auto_reached, auto_settle_ms = False, 0.0
    fit_metrics: dict[str, float] = {}
    fit_stretch_pulsed = _pulse_fit_stretch(
        win,
        app=app,
        QtCore=QtCore,
        metrics=fit_metrics,
    )
    if app is not None and QtCore is not None:
        fit_reached, fit_settle_ms = _wait_for_target_lod(
            win,
            app,
            QtCore,
            budget_s=INTERACTION_SETTLE_HARD_LIMIT_S,
        )
    else:
        fit_reached, fit_settle_ms = False, 0.0

    shuffle_gesture_id = _start_journey_gesture(win, "scroll_shuffle")
    fast = _fast_scroll_60fps(
        win,
        montage_axis=montage_axis,
        columns=columns,
        indices=indices,
        size=size,
        start=start,
        low=low,
        high=high,
        probe=probe,
        app=app,
        QtCore=QtCore,
        tile_trace=tile_trace,
    )
    fast_end_levels = _levels_histogram_state(win)
    # The paced reverse tail is a separate contract: each one-index move must
    # start from a settled current window, not inherit dozens of hidden warm
    # uploads from the preceding five-second storm. This is a realistic brief
    # release/pause, not a reset; every cache and residency object stays live.
    if app is not None and QtCore is not None:
        fast_recovery_reached, fast_recovery_settle_ms = _wait_for_target_lod(
            win,
            app,
            QtCore,
            budget_s=INTERACTION_SETTLE_HARD_LIMIT_S,
        )
    else:
        fast_recovery_reached, fast_recovery_settle_ms = False, 0.0
    _finish_journey_gesture(
        win,
        shuffle_gesture_id,
        reached=bool(fast_recovery_reached),
        app=app,
        QtCore=QtCore,
    )
    if tile_trace is not None:
        tile_trace.capture(
            "fast-recovery-settled",
            requested_start=int(fast["fast_scroll_end_start"]),
        )
    slow = _slow_scroll_lod_paced(
        win,
        montage_axis=montage_axis,
        columns=columns,
        indices=indices,
        size=size,
        start=int(fast["fast_scroll_end_start"]),
        steps=min(10, max(0, max_start)),
        probe=probe,
        app=app,
        QtCore=QtCore,
        tile_trace=tile_trace,
    )
    return {
        "verbose_tile_trace_enabled": bool(verbose_tile_trace),
        "scroll_tile_trace": [] if tile_trace is None else tile_trace.snapshots,
        "scroll_window_size": int(size),
        "scroll_prep_reached_target_lod": bool(prep_reached),
        "scroll_prep_settle_ms": float(prep_settle_ms),
        "scroll_auto_reached_target_lod": bool(auto_reached),
        "scroll_auto_settle_ms": float(auto_settle_ms),
        "scroll_fit_reached_target_lod": bool(fit_reached),
        "scroll_fit_settle_ms": float(fit_settle_ms),
        "fit_stretch_pulsed": bool(fit_stretch_pulsed),
        "action_fit_stretch_ms": float(fit_metrics.get("fit_stretch_total_ms", 0.0)),
        "scroll_center_band": [int(low), int(high)],
        "scroll_fast_end_levels": fast_end_levels.get("display_levels"),
        "scroll_fast_end_levels_default": bool(fast_end_levels.get("levels_look_default")),
        "scroll_fast_end_histogram_empty": bool(fast_end_levels.get("histogram_empty")),
        "scroll_fast_recovery_reached_target_lod": bool(fast_recovery_reached),
        "scroll_fast_recovery_settle_ms": float(fast_recovery_settle_ms),
        **fast,
        **slow,
        **_lod_state_record(win),
        **fit_metrics,
    }


def _apply_montage_zoom_pan_stress(
    win,
    *,
    probe=None,
    app=None,
    QtCore=None,
    mid_toggle=None,
    montage_axis=None,
    columns=None,
    indices=(),
    window_size=0,
) -> dict[str, object]:
    """Continuous zoom/pan stress against the full montage (LOD + visibility).

    The gestures chain back-to-back with NO settle between them: zoom out to the
    enforced limit (tiles tiny), optionally toggle the op pipeline mid-storm,
    then zoom-in / pan-right / pan-down / deep-zoom / zoom-out — each glide
    starting from the previous (unsettled) view.  This sustained rapid view
    change is a harder stress on the LOD retarget + visibility path than settling
    between steps, and keeps the phase fast.  A single short target-LOD settle at
    the end measures how quickly the system recovers after the storm.
    """

    record: dict[str, object] = {}
    record["scroll_window_size"] = int(window_size) if int(window_size) > 0 else None
    fit_metrics: dict[str, float] = {}
    record["fit_stretch_pulsed"] = _pulse_fit_stretch(
        win,
        app=app,
        QtCore=QtCore,
        metrics=fit_metrics,
    )
    record["action_fit_stretch_ms"] = float(fit_metrics.get("fit_stretch_total_ms", 0.0))
    record.update(fit_metrics)

    def _glide(label: str, target_range, frames: int, *, frame_action=None) -> None:
        win._arrayscope_profile_action = str(label)
        stats = _glide_view_range(
            win,
            app,
            QtCore,
            probe,
            target_range,
            frames=frames,
            fps=ZOOMPAN_INPUT_FPS,
            frame_action=frame_action,
        )
        for key, value in stats.items():
            record[f"{label}_{key}"] = value

    base = _montage_view_range(win)
    if base is None:
        return {"zoompan_available": False}
    record["zoompan_available"] = True

    # 1) Hit the real enforced zoom-out limit, not a guessed span multiplier.
    win._arrayscope_profile_action = "maximum-zoomout-probe"
    maximum_out = _maximum_zoomout_view_range(win, app, QtCore, base)
    record["zoompan_input_fps"] = float(ZOOMPAN_INPUT_FPS)
    record["zoompan_max_out_request_scale"] = float(ZOOMPAN_MAX_OUT_REQUEST_SCALE)
    record["zoompan_max_out_range"] = maximum_out
    _glide("zoomout_limit", maximum_out, frames=4)
    record.update({f"zoomout_limit_lod_{k}": v for k, v in _lod_state_record(win).items()})

    # 2) Toggle the op pipeline at the tiny-tile vantage point (no settle; the
    #    following glides pump the document swap through as they run).
    if callable(mid_toggle):
        win._arrayscope_profile_action = "operation-toggle"
        mid_toggle()

    # 3) From the deliberately distant limit, make roughly the whole grid fill
    #    the viewport. This is the broadest useful LOD transition: many tiles
    #    become visible together and must all leave the coarse floor behind.
    full_grid_range = _full_montage_view_range(win)
    record["lod_full_grid_checkpoint_available"] = full_grid_range is not None
    if full_grid_range is not None:
        _glide("full_grid_zoomin", full_grid_range, frames=4)
        win._arrayscope_profile_action = "full-grid-target-settle"
        full_grid_checkpoint = _wait_for_visible_target_then_observe_near(win, app, QtCore)
        record["lod_full_grid_checkpoint"] = full_grid_checkpoint
        record["lod_full_grid_active_count"] = len(
            tuple(full_grid_checkpoint.get("active_tiles", ()) or ())
        )

    # The correctness storm must keep some content in view.  Basing the next
    # off-centre zoom on ``maximum_out`` can put its entire target tens of
    # thousands of source pixels outside the montage, where a black framebuffer
    # is correct and therefore proves nothing about retained coverage.
    interaction_range = full_grid_range or base

    # 4) Rapid, deterministic impulses repeatedly supersede unfinished LOD
    #    requests. Alternating off-centre zooms and opposing pans deliberately
    #    avoid the easy monotonic path of a human-smooth glide.
    record["zoompan_zoomin_span_scale"] = float(ZOOMPAN_CENTRAL_SPAN_SCALE)
    _glide(
        "zoomin",
        _scaled_view_range(interaction_range, ZOOMPAN_CENTRAL_SPAN_SCALE, center_frac=(0.22, 0.78)),
        frames=4,
    )
    # 5) Churn the visible set in opposing directions.
    record["zoompan_pan_right_dx_frac"] = float(ZOOMPAN_PAN_FRACTION)
    record["zoompan_pan_right_dy_frac"] = 0.0
    _glide(
        "pan_right",
        _panned_view_range(
            _montage_view_range(win) or interaction_range, ZOOMPAN_PAN_FRACTION, -0.21
        ),
        frames=3,
    )
    record["zoompan_pan_down_dx_frac"] = 0.0
    record["zoompan_pan_down_dy_frac"] = float(ZOOMPAN_PAN_FRACTION)
    _glide(
        "pan_down",
        _panned_view_range(
            _montage_view_range(win) or interaction_range, -0.41, ZOOMPAN_PAN_FRACTION
        ),
        frames=3,
    )
    _glide(
        "erratic_zoomout",
        _scaled_view_range(
            _montage_view_range(win) or interaction_range, 5.0, center_frac=(0.81, 0.18)
        ),
        frames=3,
    )
    _glide(
        "erratic_zoomin",
        _scaled_view_range(
            _montage_view_range(win) or interaction_range, 0.11, center_frac=(0.74, 0.31)
        ),
        frames=3,
    )
    # 6) Deep zoom into roughly one tile, then jump to the opposite side.
    record["zoompan_deep_zoom_span_scale"] = float(ZOOMPAN_DEEP_SPAN_SCALE)
    _glide(
        "deep_zoom",
        _scaled_view_range(
            _montage_view_range(win) or interaction_range,
            ZOOMPAN_DEEP_SPAN_SCALE,
            center_frac=(0.18, 0.82),
        ),
        frames=3,
    )
    _glide(
        "opposite_pan",
        _panned_view_range(_montage_view_range(win) or interaction_range, 0.72, -0.64),
        frames=2,
    )
    # 7) End at the same real maximum-out constraint for comparable recovery.
    _glide("zoomout_return", maximum_out, frames=4)

    # 8) Repeat the erratic path while the visible 60-tile window advances
    #    across a centre-weighted source band. This combines camera, source,
    #    LOD, evaluation, and presentation supersession in one input stream.
    source_indices = tuple(indices or ())
    combined_available = bool(montage_axis is not None and source_indices and int(window_size) > 0)
    record["combined_zoom_scroll_available"] = combined_available
    if combined_available:
        combined_size = max(1, min(int(window_size), len(source_indices)))
        max_start = max(0, len(source_indices) - combined_size)
        center_start = max(0, len(source_indices) // 2 - combined_size // 2)
        half_band = min(max_start // 2, max(8, combined_size))
        low = max(0, center_start - half_band)
        high = min(max_start, center_start + half_band)
        scroll_state = {"current": min(max(center_start, low), high), "direction": 1}
        combined_scroll_costs = {
            "state_build_ms": [],
            "state_apply_ms": [],
            "render_request_ms": [],
        }

        def advance_center_window(_frame_index, _frame_count):
            current = int(scroll_state["current"]) + int(scroll_state["direction"]) * 3
            if current >= high:
                current = high
                scroll_state["direction"] = -1
            elif current <= low:
                current = low
                scroll_state["direction"] = 1
            scroll_state["current"] = current
            costs = _scroll_montage_window(
                win,
                montage_axis=montage_axis,
                columns=columns,
                indices=source_indices,
                window_start=current,
                size=combined_size,
                interactive=True,
            )
            costs = dict(costs or {})
            for name, values in combined_scroll_costs.items():
                values.append(float(costs.get(name, 0.0) or 0.0))

        record["combined_scroll_window_size"] = int(combined_size)
        record["combined_scroll_center_band"] = [int(low), int(high)]
        if full_grid_range is not None:
            _glide(
                "combined_full_grid_zoomin",
                full_grid_range,
                frames=4,
                frame_action=advance_center_window,
            )
            # A stationary-camera glide is an intentional pause: source
            # indices continue changing at input cadence while LOD/evaluation
            # completions race the same visible slots.
            _glide(
                "combined_full_grid_scroll_pause",
                _montage_view_range(win) or full_grid_range,
                frames=3,
                frame_action=advance_center_window,
            )
        _glide(
            "combined_zoomin",
            _scaled_view_range(
                interaction_range, ZOOMPAN_CENTRAL_SPAN_SCALE, center_frac=(0.22, 0.78)
            ),
            frames=4,
            frame_action=advance_center_window,
        )
        _glide(
            "combined_zoom_scroll_pause",
            _montage_view_range(win) or interaction_range,
            frames=3,
            frame_action=advance_center_window,
        )
        _glide(
            "combined_pan_right",
            _panned_view_range(
                _montage_view_range(win) or interaction_range, ZOOMPAN_PAN_FRACTION, -0.21
            ),
            frames=3,
            frame_action=advance_center_window,
        )
        _glide(
            "combined_pan_scroll_pause",
            _montage_view_range(win) or interaction_range,
            frames=3,
            frame_action=advance_center_window,
        )
        _glide(
            "combined_pan_down",
            _panned_view_range(
                _montage_view_range(win) or interaction_range, -0.41, ZOOMPAN_PAN_FRACTION
            ),
            frames=3,
            frame_action=advance_center_window,
        )
        _glide(
            "combined_erratic_zoomout",
            _scaled_view_range(
                _montage_view_range(win) or interaction_range, 5.0, center_frac=(0.81, 0.18)
            ),
            frames=3,
            frame_action=advance_center_window,
        )
        _glide(
            "combined_erratic_zoomin",
            _scaled_view_range(
                _montage_view_range(win) or interaction_range, 0.11, center_frac=(0.74, 0.31)
            ),
            frames=3,
            frame_action=advance_center_window,
        )
        _glide(
            "combined_deep_zoom",
            _scaled_view_range(
                _montage_view_range(win) or interaction_range,
                ZOOMPAN_DEEP_SPAN_SCALE,
                center_frac=(0.18, 0.82),
            ),
            frames=3,
            frame_action=advance_center_window,
        )
        _glide(
            "combined_opposite_pan",
            _panned_view_range(_montage_view_range(win) or interaction_range, 0.72, -0.64),
            frames=2,
            frame_action=advance_center_window,
        )
        _glide("combined_zoomout_return", maximum_out, frames=4, frame_action=advance_center_window)
        record["combined_scroll_end_start"] = int(scroll_state["current"])
        for name, values in combined_scroll_costs.items():
            record[f"combined_scroll_{name}_max_ms"] = float(max(values) if values else 0.0)
            record[f"combined_scroll_{name}_p95_ms"] = float(_percentile(tuple(values), 95.0))
        win._arrayscope_profile_action = "combined-target-settle"
        combined_reached, combined_settle_ms = _wait_for_target_lod(
            win,
            app,
            QtCore,
            budget_s=ZOOMPAN_CHECKPOINT_SETTLE_S,
        )
        record["combined_recovery_reached_target_lod"] = bool(combined_reached)
        record["combined_recovery_settle_ms"] = float(combined_settle_ms)

        # Field regression 2026-07-21: populate reduced LODs, settle one
        # center tile at native resolution, then replace the source window by
        # a distant one while that deep camera remains fixed.  The old broad
        # storm zoomed out before settlement and therefore missed the
        # all-slot atomic wait that stranded the ready center behind two
        # ownerless offscreen shell slots.
        far_scroll_range = _few_tile_view_range(win, center_fraction=0.5)
        record["deep_zoom_far_scroll_available"] = far_scroll_range is not None
        if far_scroll_range is not None:
            _glide("deep_zoom_far_scroll_prep", far_scroll_range, frames=3)
            pre_checkpoint = _wait_for_visible_target_then_observe_near(win, app, QtCore)
            pre_reached = bool(pre_checkpoint.get("visible_target_reached", False))
            pre_settle_ms = float(pre_checkpoint.get("visible_settle_ms", 0.0) or 0.0)
            far_start = (
                high
                if abs(high - int(scroll_state["current"]))
                >= abs(int(scroll_state["current"]) - low)
                else low
            )
            gesture_id = _start_journey_gesture(win, "deep_zoom_far_scroll")
            _scroll_montage_window(
                win,
                montage_axis=montage_axis,
                columns=columns,
                indices=source_indices,
                window_start=far_start,
                size=combined_size,
                interactive=True,
            )
            post_checkpoint = _wait_for_visible_target_then_observe_near(win, app, QtCore)
            post_reached = bool(post_checkpoint.get("visible_target_reached", False))
            post_settle_ms = float(post_checkpoint.get("visible_settle_ms", 0.0) or 0.0)
            _finish_journey_gesture(
                win,
                gesture_id,
                reached=bool(post_reached),
                app=app,
                QtCore=QtCore,
            )
            record["deep_zoom_far_scroll_precondition_reached_target_lod"] = bool(pre_reached)
            record["deep_zoom_far_scroll_precondition_settle_ms"] = float(pre_settle_ms)
            record["deep_zoom_far_scroll_precondition_evidence"] = pre_checkpoint
            record["deep_zoom_far_scroll_start"] = int(far_start)
            record["deep_zoom_far_scroll_index_distance"] = abs(
                int(far_start) - int(scroll_state["current"])
            )
            record["deep_zoom_far_scroll_reached_target_lod"] = bool(post_reached)
            record["deep_zoom_far_scroll_settle_ms"] = float(post_settle_ms)
            record["deep_zoom_far_scroll_target_evidence"] = post_checkpoint

    # 9) A deterministic correctness checkpoint catches the visually subtle
    # failure where the storm ends with all slots populated but a few visible
    # tiles permanently stuck at fallback LOD. Zoom to only a few tiles, pause
    # just long enough for visible target work, then pan into the previous near
    # set and repeat. The monitor also proves that new speculative near uploads
    # do not outrank unfinished visible target work.
    checkpoint_range = _few_tile_view_range(win, center_fraction=0.48)
    record["lod_checkpoint_available"] = checkpoint_range is not None
    zoomout_gesture_id = None
    if checkpoint_range is not None:
        zoomin_gesture_id = _start_journey_gesture(win, "zoom_in")
        _glide("lod_checkpoint_zoomin", checkpoint_range, frames=3)
        win._arrayscope_profile_action = "lod-checkpoint-zoom-settle"
        first_checkpoint = _wait_for_visible_target_then_observe_near(win, app, QtCore)
        _finish_journey_gesture(
            win,
            zoomin_gesture_id,
            reached=bool(first_checkpoint.get("visible_target_reached", False)),
            app=app,
            QtCore=QtCore,
        )
        record["lod_checkpoint_zoom"] = first_checkpoint
        first_active = {int(tile) for tile in first_checkpoint["active_tiles"]}
        record["lod_checkpoint_zoom_active_count"] = len(first_active)

        pan_target = _panned_view_range(_montage_view_range(win) or checkpoint_range, 0.78, 0.34)
        _glide("lod_checkpoint_pan", pan_target, frames=3)
        win._arrayscope_profile_action = "lod-checkpoint-pan-settle"
        second_checkpoint = _wait_for_visible_target_then_observe_near(win, app, QtCore)
        record["lod_checkpoint_pan_result"] = second_checkpoint
        second_active = {int(tile) for tile in second_checkpoint["active_tiles"]}
        record["lod_checkpoint_pan_changed_visible_tiles"] = bool(first_active != second_active)
        record["lod_checkpoint_visible_tile_union"] = tuple(sorted(first_active | second_active))

        # Preserve the workflow's full-montage end state for the following
        # scalar/FFT stage and for the user's visual inspection.
        zoomout_gesture_id = _start_journey_gesture(win, "zoom_out")
        _glide("lod_checkpoint_zoomout_return", maximum_out, frames=3)

    # Single final settle after restoring the full montage.
    win._arrayscope_profile_action = "final-target-settle"
    reached, settle_ms = _wait_for_target_lod(win, app, QtCore, budget_s=ZOOMPAN_FINAL_SETTLE_S)
    if zoomout_gesture_id is not None:
        # Resident zoom-out is descriptor-only: target LOD can settle with no
        # payload commit at all. The final ViewBox range still queues a real
        # canvas paint, so keep the gesture-scoped sampler alive until that
        # presentation request is physically acknowledged. Otherwise the
        # journey-end grab can race the native child paint and falsely report
        # that no pixels ever changed.
        _wait_for_tile_presentation_draw(
            win,
            app,
            QtCore,
            timeout_s=INTERACTION_SETTLE_HARD_LIMIT_S,
        )
        _finish_journey_gesture(
            win,
            zoomout_gesture_id,
            reached=bool(reached),
            app=app,
            QtCore=QtCore,
        )
    record["final_settle_ms"] = float(settle_ms)
    record["final_reached_target_lod"] = bool(reached)
    record.update(
        {f"final_target_{key}": value for key, value in _montage_target_lod_evidence(win).items()}
    )
    record.update({f"final_lod_{k}": v for k, v in _lod_state_record(win).items()})

    # Rollups for the flat headline table.
    glide_max = [float(v) for k, v in record.items() if k.endswith("_max_gap_ms")]
    glide_p95 = [float(v) for k, v in record.items() if k.endswith("_p95_gap_ms")]
    record["zoompan_max_gap_ms"] = float(max(glide_max) if glide_max else 0.0)
    record["zoompan_worst_p95_gap_ms"] = float(max(glide_p95) if glide_p95 else 0.0)
    return record


def _tile_number_set(values) -> set[int]:
    iterable = tuple(values.keys()) if isinstance(values, dict) else tuple(values or ())
    numbers: set[int] = set()
    for value in iterable:
        try:
            numbers.add(int(value.montage_index))
        except AttributeError:
            numbers.add(int(value))
    return numbers


def _visible_backlog_state(session, expected: set[int]) -> dict[str, object]:
    if session is None:
        return {
            "visible_has_backlog": False,
            "visible_target_unsettled_tiles": 0,
            "visible_loading_tiles": 0,
            "visible_active_requests": 0,
            "visible_dirty_tiles": 0,
            "visible_upserts": 0,
            "visible_removals": 0,
            "visible_stage_waiters": 0,
            "atomic_successor_pending": False,
        }
    fan = getattr(session, "stage_fan_in", None)
    target_unsettled = _tile_number_set(session.required_target_unsettled_tiles())
    loading = _tile_number_set(getattr(session, "loading_tiles", ()))
    active = _tile_number_set(getattr(session, "active_tile_requests", ()))
    dirty = _tile_number_set(getattr(session, "dirty_payloads", ()))
    upserts = _tile_number_set(getattr(session, "pending_payload_upserts", ()))
    removals = _tile_number_set(getattr(session, "pending_removals", ()))
    stage_waiters = _tile_number_set(getattr(fan, "tile_stage_keys", {}) if fan is not None else {})
    visible_target_unsettled = target_unsettled & expected
    visible_loading = loading & expected
    visible_active = active & expected
    visible_dirty = dirty & expected
    visible_upserts = upserts & expected
    visible_removals = removals & expected
    visible_stage_waiters = stage_waiters & expected
    visible_changed = bool(visible_dirty or visible_upserts or visible_removals)
    atomic_successor_pending = bool(getattr(session, "atomic_successor_pending", False))
    visible_has_backlog = bool(
        visible_target_unsettled
        or visible_loading
        or visible_active
        or visible_stage_waiters
        or visible_changed
        or atomic_successor_pending
        or (visible_changed and bool(getattr(session, "final_commit_pending", False)))
        or (visible_changed and bool(getattr(session, "flush_pending", False)))
    )
    return {
        "visible_has_backlog": visible_has_backlog,
        "visible_target_unsettled_tiles": len(visible_target_unsettled),
        "visible_loading_tiles": len(visible_loading),
        "visible_active_requests": len(visible_active),
        "visible_dirty_tiles": len(visible_dirty),
        "visible_upserts": len(visible_upserts),
        "visible_removals": len(visible_removals),
        "visible_stage_waiters": len(visible_stage_waiters),
        "atomic_successor_pending": atomic_successor_pending,
    }


def _montage_settled(session) -> bool:
    if session is None:
        return False
    active = set(_active_planned_montage_tiles(session))
    expected = set(_expected_requested_montage_tiles(session))
    if not expected:
        expected = set(active)
    visible_checker = getattr(session, "visible_first_pixels_presented", None)
    visible_settled = (
        bool(visible_checker()) if callable(visible_checker) else bool(session.is_complete())
    )
    backlog = _visible_backlog_state(session, active)
    return bool(
        visible_settled
        and not bool(backlog["visible_has_backlog"])
        and not bool(getattr(session, "has_pending_level_update", lambda: False)())
    )


def _montage_work_in_flight(session) -> bool:
    """True when real computation is still running (kernel tasks in flight).

    A montage that is not settled but has work in flight is *progressing*; a
    montage that is not settled with nothing in flight is a lost wakeup — the
    signature the stall guard uses to bail fast instead of waiting the full
    budget.
    """

    if session is None:
        return False
    fan = getattr(session, "stage_fan_in", None)
    semantic_progress = getattr(session, "semantic_level_evidence_progress", None)
    # Only kernel-submitted work counts as in flight.  ``attached_requests`` are
    # stage keys bound to the fan-in but not necessarily running; an attached
    # stage that never activates while nothing else progresses is precisely the
    # stall the guard must catch, so it is deliberately excluded here.
    return bool(
        getattr(session, "active_tile_requests", None)
        or getattr(session, "pending_rung_materializations", None)
        or bool(getattr(session, "level_evidence_inflight", False))
        or bool(
            semantic_progress is not None
            and getattr(semantic_progress, "inflight_generation", None) is not None
        )
        or bool(getattr(session, "histogram_aggregate_inflight", False))
        or (fan is not None and getattr(fan, "active_requests", None))
    )


def _montage_stall_signature(session):
    if session is None:
        return None
    fan = getattr(session, "stage_fan_in", None)
    return (
        int(getattr(session, "session_id", -1)),
        len(getattr(session, "loading_tiles", ()) or ()),
        len(session.required_target_unsettled_tiles()),
        len(getattr(session, "active_tile_requests", ()) or ()),
        len(getattr(session, "dirty_payloads", {}) or {}),
        0 if fan is None else len(getattr(fan, "tile_stage_keys", {}) or {}),
        0 if fan is None else len(getattr(fan, "values", {}) or {}),
        len(getattr(session, "pending_rung_materializations", ()) or ()),
        len(session.lifecycle.presented_tiles),
        len(getattr(session, "rendered_tiles", {}) or {}),
    )


def _wait_for_montage_complete_soft(
    *, win, app, QtCore, budget_s: float, stall_grace_s: float = 2.5
) -> bool:
    """Pump the loop up to budget_s; return whether the montage settled.

    Bails early with False when the montage is *clearly frozen*: the stall
    signature holds constant for ``stall_grace_s`` while no kernel work is in
    flight (a lost wakeup — waiting the full budget would only stare at a dead
    session).  A montage still computing keeps the full budget.
    """

    budget_s = bounded_interaction_settle_timeout_s(budget_s)
    stall_grace_s = min(bounded_interaction_settle_timeout_s(stall_grace_s), budget_s)
    deadline = perf_counter() + budget_s
    stall_since = None
    last_sig = None
    while perf_counter() < deadline:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
        session = getattr(win, "_frame_session", None)
        if _montage_settled(session):
            return True
        sig = _montage_stall_signature(session)
        if _montage_work_in_flight(session):
            stall_since = None  # genuine progress possible; keep waiting
        elif sig != last_sig:
            stall_since = perf_counter()  # state changed; restart the grace window
        elif stall_since is not None and perf_counter() - stall_since >= float(stall_grace_s):
            print(
                f"[profile] STALL GUARD: montage frozen (no work in flight) — "
                f"signature stable {stall_grace_s:.1f}s: {sig}",
                file=sys.stderr,
                flush=True,
            )
            return False
        last_sig = sig
    return False


def _wait_for_montage_successor_settled(
    *,
    win,
    app,
    QtCore,
    predecessor_session_id: int,
    budget_s: float,
) -> bool:
    """Wait for an input's successor session, then for that target to settle.

    A retained predecessor can remain fully visible between the input callback
    and the coalesced render request.  Treating that old session as the input's
    completion made the displayed-axis profile skip the atomic handoff it was
    intended to exercise.
    """

    budget_s = bounded_interaction_settle_timeout_s(budget_s)
    started = perf_counter()
    deadline = started + budget_s
    while perf_counter() < deadline:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
        session = getattr(win, "_frame_session", None)
        if int(getattr(session, "session_id", -1) or -1) != int(predecessor_session_id):
            remaining = max(0.0, deadline - perf_counter())
            if remaining <= 0.0:
                return False
            return _wait_for_montage_complete_soft(
                win=win,
                app=app,
                QtCore=QtCore,
                budget_s=remaining,
                stall_grace_s=min(2.5, remaining),
            )
        time.sleep(0.001)
    return False


def _committed_display_frame_is_current(win) -> bool:
    frame = getattr(win, "_committed_display_frame", None)
    checker = getattr(getattr(win, "renderer", None), "_is_committed_display_frame_current", None)
    return bool(frame is not None and callable(checker) and checker(frame))


def _profile_transform_operations(
    montage_axis: int,
    *,
    centered_fft,
    fftshift,
    centered_ifft,
):
    return (
        centered_fft(axis=int(montage_axis)),
        fftshift(axis=int(montage_axis)),
        centered_ifft(axis=int(montage_axis)),
    )


def _pulse_fit_stretch(
    win, *, app=None, QtCore=None, metrics: dict[str, float] | None = None
) -> bool:
    """Exercise both user-visible fit/stretch states and expose their cost."""

    fit = getattr(win, "fit_image_to_view", None)
    if not callable(fit):
        return False
    total_start = perf_counter()
    enable_start = perf_counter()
    fit(True)
    enable_call_ms = (perf_counter() - enable_start) * 1000.0
    enable_delivery_start = perf_counter()
    if app is not None and QtCore is not None:
        _process_events(app, QtCore, count=10)
    enable_delivery_ms = (perf_counter() - enable_delivery_start) * 1000.0
    disable_start = perf_counter()
    fit(False)
    disable_call_ms = (perf_counter() - disable_start) * 1000.0
    disable_delivery_start = perf_counter()
    disable_modes: list[str] = []
    disable_ranges: list[object] = []
    if app is not None and QtCore is not None:
        for _step in range(5):
            _process_events(app, QtCore, count=1)
            image_view = getattr(win, "img_view", None)
            controller = getattr(image_view, "viewport_controller", None)
            mode = getattr(controller, "mode", None)
            disable_modes.append(str(getattr(mode, "value", mode) or ""))
            try:
                disable_ranges.append(image_view.getView().viewRange())
            except Exception:
                disable_ranges.append(None)
    disable_delivery_ms = (perf_counter() - disable_delivery_start) * 1000.0
    # Turning Fit off preserves the fitted camera range, so it does not emit a
    # second range-change signal.  The async render started immediately before
    # this pulse can therefore still own the predecessor's smaller visibility
    # plan even though the camera already shows the full montage.  Deliver the
    # final programmatic-camera obligation explicitly, matching the synchronous
    # path owned by ViewportBridge.on_view_range_changed().
    retarget = getattr(win, "retarget_montage_viewport", None)
    retarget_call_start = perf_counter()
    if callable(retarget):
        retarget()
    retarget_call_ms = (perf_counter() - retarget_call_start) * 1000.0
    retarget_delivery_start = perf_counter()
    if app is not None and QtCore is not None:
        _process_events(app, QtCore, count=2)
    retarget_delivery_ms = (perf_counter() - retarget_delivery_start) * 1000.0
    if metrics is not None:
        image_view = getattr(win, "img_view", None)
        controller = getattr(image_view, "viewport_controller", None)
        mode = getattr(controller, "mode", None)
        get_view = getattr(image_view, "getView", None)
        view_range = None
        if callable(get_view):
            try:
                view_range = get_view().viewRange()
            except Exception:
                view_range = None
        draw_pending = getattr(image_view, "presentationDrawPending", None)
        metrics.update(
            {
                "fit_stretch_enable_call_ms": float(enable_call_ms),
                "fit_stretch_enable_delivery_ms": float(enable_delivery_ms),
                "fit_stretch_disable_call_ms": float(disable_call_ms),
                "fit_stretch_disable_delivery_ms": float(disable_delivery_ms),
                "fit_stretch_retarget_call_ms": float(retarget_call_ms),
                "fit_stretch_retarget_delivery_ms": float(retarget_delivery_ms),
                "fit_disable_delivery_modes": disable_modes,
                "fit_disable_delivery_ranges": disable_ranges,
                "fit_stretch_total_ms": float((perf_counter() - total_start) * 1000.0),
                "fit_disable_viewport_mode": str(getattr(mode, "value", mode) or ""),
                "fit_disable_view_range": view_range,
                "fit_disable_draw_pending_after_delivery": bool(
                    callable(draw_pending) and draw_pending()
                ),
            }
        )
    return True


class _EventLoopProbe:
    def __init__(self, QtCore, app=None):
        # Timer category: UI cosmetic. Benchmark heartbeat samples event-loop
        # gaps and never gates render work.
        self._timer = QtCore.QTimer()
        self._timer.setInterval(1)
        self._timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        self._app = app
        self._last = perf_counter()
        self.max_gap_ms = 0.0
        self.tick_count = 0
        self.gaps_ms: list[float] = []
        self.gc_pauses_ms: list[float] = []
        self.gc_pauses: list[dict[str, object]] = []
        self.gc_max_pause_ms = 0.0
        self._gc_started: dict[int, float] = {}
        self._gc_callback_installed = False

    def start(self) -> None:
        self.reset()
        if not self._gc_callback_installed:
            gc.callbacks.append(self._gc_callback)
            self._gc_callback_installed = True
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        if self._gc_callback_installed:
            with contextlib.suppress(ValueError):
                gc.callbacks.remove(self._gc_callback)
            self._gc_callback_installed = False

    def reset(self) -> None:
        self._last = perf_counter()
        self.max_gap_ms = 0.0
        self.tick_count = 0
        self.gaps_ms = []
        self.gc_pauses_ms = []
        self.gc_pauses = []
        self.gc_max_pause_ms = 0.0
        self._gc_started = {}
        if self._app is not None:
            self._app._arrayscope_profile_event_pump_ms = []

    def _tick(self) -> None:
        now = perf_counter()
        gap_ms = (now - self._last) * 1000.0
        self.gaps_ms.append(float(gap_ms))
        self.max_gap_ms = max(self.max_gap_ms, gap_ms)
        self._last = now
        self.tick_count += 1

    def _gc_callback(self, phase, info) -> None:
        generation = int(dict(info or {}).get("generation", -1))
        if phase == "start":
            self._gc_started[generation] = perf_counter()
            return
        started = self._gc_started.pop(generation, None)
        if started is None:
            return
        elapsed_ms = (perf_counter() - started) * 1000.0
        self.gc_pauses_ms.append(float(elapsed_ms))
        self.gc_pauses.append(
            {
                "generation": int(generation),
                "elapsed_ms": float(elapsed_ms),
                "collected": int(dict(info or {}).get("collected", 0) or 0),
                "uncollectable": int(dict(info or {}).get("uncollectable", 0) or 0),
            }
        )
        self.gc_max_pause_ms = max(float(self.gc_max_pause_ms), float(elapsed_ms))

    def percentile_ms(self, percentile: float) -> float | None:
        if not self.gaps_ms:
            return None
        return _percentile(tuple(self.gaps_ms), percentile)


class _VisualTimelineProbe:
    """Periodic framebuffer + physical tile truth for interaction diagnosis."""

    def __init__(self, QtCore, QtGui, win, *, backend: str, directory: Path, interval_s: float):
        self._QtCore = QtCore
        self._QtGui = QtGui
        self._win = win
        self._backend = str(backend)
        self._directory = Path(directory)
        self._interval_ms = max(100, round(float(interval_s) * 1000.0))
        self._timer = QtCore.QTimer(win)
        self._timer.setInterval(self._interval_ms)
        self._timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._capture_interval)
        self._presentation_drawn = getattr(
            getattr(win, "img_view", None), "presentationDrawn", None
        )
        self._started_ns = time.monotonic_ns()
        self._last_sample_ns = self._started_ns
        self._last_drawn: frozenset[int] = frozenset()
        self._last_identities: dict[int, str] = {}
        self._images: list[tuple[Path, dict[str, object]]] = []
        self._index = 0
        self.timeline_path = self._directory / f"{self._backend}-visual-timeline.jsonl"
        self.contact_sheet_path = self._directory / f"{self._backend}-visual-contact-sheet.png"

    def start(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        if self.timeline_path.exists():
            self.timeline_path.unlink()
        if self._presentation_drawn is not None:
            self._presentation_drawn.connect(
                self._capture_presentation_draw_ack,
                self._QtCore.Qt.ConnectionType.QueuedConnection,
            )
        self.capture("start")
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        if self._presentation_drawn is not None:
            with contextlib.suppress(RuntimeError, TypeError):
                self._presentation_drawn.disconnect(self._capture_presentation_draw_ack)
        self.capture("stop")
        self._write_contact_sheet()

    def _capture_presentation_draw_ack(self) -> None:
        if not str(getattr(self._win, "_arrayscope_active_gesture_id", "") or ""):
            return
        elapsed_ms = (time.monotonic_ns() - self._last_sample_ns) / 1_000_000.0
        if elapsed_ms >= self._interval_ms:
            self.capture("presentation-draw-ack")

    def _capture_interval(self) -> None:
        # Wgpu gestures have a native presentation acknowledgement now. Avoid
        # duplicate full-window grabs between those physical edges; incumbents
        # and all non-gesture diagnostics retain the original periodic sampler.
        active_gesture = str(getattr(self._win, "_arrayscope_active_gesture_id", "") or "")
        if self._backend == "wgpu" and active_gesture:
            return
        self.capture("interval")

    def capture(self, reason: str) -> None:
        self._index += 1
        path = self._directory / f"{self._backend}-visual-{self._index:04d}.png"
        # Screen-path WGPU cannot be read from Qt's backing store. Keep the
        # labelled executor replay synchronous so the pixels, scene rows, and
        # trace metadata describe the same instant. Its GPU fence is why a
        # screenshot-enabled run is diagnostic rather than timing truth.
        saved = _save_view_screenshot(self._win, path, full_window=False)
        now_ns = time.monotonic_ns()
        screenshot_capture_kind = str(
            getattr(self._win, "_arrayscope_last_screenshot_capture_kind", "unknown")
        )
        screenshot_capture_error = str(
            getattr(self._win, "_arrayscope_last_screenshot_capture_error", "") or ""
        )
        session = getattr(self._win, "_frame_session", None)
        plan_tiles = tuple(getattr(getattr(session, "plan", None), "tiles", ()) or ())
        requested = frozenset(int(tile.montage_index) for tile in plan_tiles)
        visible = frozenset(
            int(tile) for tile in tuple(getattr(session, "visible_tile_numbers", ()) or ())
        )
        payloads = dict(getattr(session, "display_tile_payloads", {}) or {})
        lifecycle = getattr(session, "lifecycle", None)
        required_tiles = tuple(
            getattr(session, "required_tile_numbers", lambda: ())() if session is not None else ()
        )
        unsettled_tiles = tuple(
            getattr(session, "required_target_unsettled_tiles", lambda: ())()
            if session is not None
            else ()
        )
        lifecycle_snapshot = (
            getattr(session, "lifecycle_snapshot", lambda: None)() if session is not None else None
        )
        backend_identities = dict(getattr(lifecycle, "backend_presented_identities", {}) or {})
        physical_rows_fn = getattr(
            getattr(self._win, "img_view", None), "tileTruthPhysicalRows", None
        )
        physical_rows = dict(physical_rows_fn() or {}) if callable(physical_rows_fn) else {}
        presentation_diagnostics = dict(
            getattr(getattr(self._win, "img_view", None), "presentation_diagnostics", dict)() or {}
        )
        physical_acknowledged = frozenset(
            int(tile)
            for tile, row in physical_rows.items()
            if dict(row or {}).get("physical_acknowledged_identity") is not None
        )
        acknowledged = (
            frozenset(int(tile) for tile in backend_identities) | physical_acknowledged
        ) & requested
        scene_presented = _visual_scene_presented_tiles(
            self._backend,
            presentation_diagnostics=presentation_diagnostics,
            physical_rows=physical_rows,
        )
        drawn = scene_presented & requested
        identity_text: dict[int, str] = {}
        for tile in drawn:
            identity = backend_identities.get(int(tile))
            if identity is None:
                identity = dict(physical_rows.get(int(tile), {}) or {}).get(
                    "physical_acknowledged_identity"
                )
            identity_text[int(tile)] = _trace_identity(identity, limit=500) or ""
        changed = frozenset(
            tile
            for tile, identity in identity_text.items()
            if self._last_identities.get(tile) not in (None, identity)
        )
        appeared = drawn - self._last_drawn
        disappeared = self._last_drawn - drawn
        fallback_bindings = 0
        exact_bindings = 0
        binding_rows = 0
        for tile, row in physical_rows.items():
            if int(tile) not in drawn:
                continue
            for binding in tuple(dict(row or {}).get("physical_page_bindings", ()) or ()):
                binding_rows += 1
                if str(dict(binding or {}).get("quality", "")) == "exact":
                    exact_bindings += 1
                else:
                    fallback_bindings += 1
        physical_draw_rows = _visual_physical_draw_rows(physical_rows, drawn)
        geometry_state = _window_geometry_state(self._win)
        view_range = _montage_view_range(self._win)
        content_range = _full_montage_view_range(self._win)
        onscreen = frozenset(
            int(tile)
            for tile in drawn
            if _view_range_intersects_world_bounds(
                view_range,
                dict(physical_rows.get(int(tile), {}) or {}).get("physical_draw_world_bounds"),
            )
        )
        physical_visible_pages = int(
            presentation_diagnostics.get("physical_visible_page_count", 0)
            or presentation_diagnostics.get("tile_visual_visible_pages", 0)
            or 0
        )
        backend_presented_count = int(presentation_diagnostics.get("presented_tile_count", 0) or 0)
        physically_visible_tile_count = int(
            presentation_diagnostics.get("physically_visible_tile_count", 0) or 0
        )
        physical_visible = physically_visible_tile_count > 0
        camera_state = _visual_camera_state(
            self._win,
            session=session,
            live_view_range=view_range,
        )
        elapsed_ms = (now_ns - self._started_ns) / 1_000_000.0
        gap_ms = (now_ns - self._last_sample_ns) / 1_000_000.0
        journey_lod_state = _journey_lod_trace_state(self._win)
        record = {
            "record_type": "visual_sample",
            "backend": self._backend,
            "sample": int(self._index),
            "reason": str(reason),
            "phase": str(getattr(self._win, "_arrayscope_profile_phase", "setup") or "setup"),
            "action": str(getattr(self._win, "_arrayscope_profile_action", "idle") or "idle"),
            "journey": str(getattr(self._win, "_arrayscope_active_journey", "") or ""),
            "gesture_id": str(getattr(self._win, "_arrayscope_active_gesture_id", "") or ""),
            "monotonic_ns": int(now_ns),
            "elapsed_ms": float(elapsed_ms),
            "sample_gap_ms": float(gap_ms),
            "event_loop_freeze": bool(gap_ms > max(1500.0, self._interval_ms * 1.5)),
            "screenshot_path": str(path),
            "screenshot_saved": bool(saved),
            "screenshot_capture_kind": screenshot_capture_kind,
            "screenshot_capture_error": screenshot_capture_error,
            "session_id": None if session is None else int(getattr(session, "session_id", 0) or 0),
            "requested_tiles": sorted(requested),
            "visible_tiles": sorted(visible),
            "required_tiles": sorted(int(tile) for tile in required_tiles),
            "required_target_unsettled_tiles": sorted(int(tile) for tile in unsettled_tiles),
            "lifecycle_phase_counts": dict(getattr(lifecycle_snapshot, "counts", {}) or {}),
            "payload_tiles": sorted(int(tile) for tile in payloads if int(tile) in requested),
            "acknowledged_tiles": sorted(acknowledged),
            "resident_physical_row_tiles": sorted(
                {int(tile) for tile in physical_rows}.intersection(requested)
            ),
            "scene_presented_tiles": sorted(drawn),
            "onscreen_tiles": sorted(onscreen),
            "drawn_tiles": sorted(drawn),
            "appeared_tiles": sorted(appeared),
            "disappeared_tiles": sorted(disappeared),
            "changed_tiles": sorted(changed),
            "missing_draw_tiles": sorted(requested - drawn),
            "missing_visible_draw_tiles": sorted(visible - drawn),
            "requested_count": len(requested),
            "visible_count": len(visible),
            "payload_count": len(set(payloads).intersection(requested)),
            "drawn_count": len(drawn),
            "onscreen_count": len(onscreen),
            "acknowledged_count": len(acknowledged),
            "resident_physical_row_count": len(
                {int(tile) for tile in physical_rows}.intersection(requested)
            ),
            "exact_page_binding_count": int(exact_bindings),
            "fallback_page_binding_count": int(fallback_bindings),
            "page_binding_count": int(binding_rows),
            "physical_draw_rows": physical_draw_rows,
            "view_range": view_range,
            "content_range": content_range,
            "camera_intersects_content": _view_ranges_intersect(view_range, content_range),
            "camera_state": camera_state,
            "window_geometry": geometry_state,
            "montage_display_mode": str(
                presentation_diagnostics.get("montage_display_mode", "none")
            ),
            "physical_visible": physical_visible,
            "physically_visible_tile_count": physically_visible_tile_count,
            "backend_presented_tile_count": backend_presented_count,
            "presentation_draw_count": int(presentation_diagnostics.get("draw_count", 0) or 0),
            "tile_presentation_request_count": int(
                presentation_diagnostics.get("tile_presentation_request_count", 0) or 0
            ),
            "tile_presentation_draw_count": int(
                presentation_diagnostics.get("tile_presentation_draw_count", 0) or 0
            ),
            "presentation_draw_pending": bool(
                presentation_diagnostics.get("tile_presentation_draw_pending", False)
                or presentation_diagnostics.get("canvas_update_pending", False)
                or bool(
                    callable(
                        getattr(
                            getattr(self._win, "img_view", None),
                            "presentationDrawPending",
                            None,
                        )
                    )
                    and self._win.img_view.presentationDrawPending()
                )
            ),
            "physical_visible_page_count": physical_visible_pages,
            "physical_geometry": _visual_geometry_summary(
                physical_draw_rows,
                view_range=view_range,
                viewport_shape=geometry_state.get("viewport_shape"),
            ),
            "page_candidate_missing_tile_count": int(
                presentation_diagnostics.get("page_candidate_missing_tile_count", 0) or 0
            ),
            "page_candidate_missing_key_count": int(
                presentation_diagnostics.get("page_candidate_missing_key_count", 0) or 0
            ),
            "page_table_resident_count": int(
                presentation_diagnostics.get("page_table_resident_count", 0) or 0
            ),
            "atlas_page_classes": tuple(
                presentation_diagnostics.get("atlas_page_classes", ()) or ()
            ),
            "atlas_estimated_gpu_bytes": int(
                presentation_diagnostics.get("atlas_estimated_gpu_bytes", 0) or 0
            ),
            "atlas_budget_bytes": int(presentation_diagnostics.get("atlas_budget_bytes", 0) or 0),
            "presentation_revision": int(
                getattr(getattr(session, "tile_presentation_state", None), "revision", 0) or 0
            ),
            "lod_level_counts": _visual_lod_level_counts(payloads, requested),
            # Frame cadence (wgpu screen path only; absent elsewhere).  Passed
            # through wholesale rather than cherry-picked: this is the readout
            # the frame-pacing dossier's phase 2 exists to capture, and a
            # baseline that silently dropped a key would have to be re-run.
            **{
                key: value
                for key, value in presentation_diagnostics.items()
                if key.startswith("wgpu_screen_")
            },
            **journey_lod_state,
        }
        with self.timeline_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        emit_trace(
            "visual_sample",
            backend=self._backend,
            phase=record["phase"],
            sample=int(self._index),
            screenshot_path=str(path),
            drawn_tiles=tuple(sorted(drawn)),
            appeared_tiles=tuple(sorted(appeared)),
            disappeared_tiles=tuple(sorted(disappeared)),
            changed_tiles=tuple(sorted(changed)),
            missing_draw_tiles=tuple(sorted(requested - drawn)),
            onscreen_tiles=tuple(sorted(onscreen)),
            acknowledged_tiles=tuple(sorted(acknowledged)),
            event_loop_freeze=bool(record["event_loop_freeze"]),
            sample_gap_ms=float(gap_ms),
            action=record["action"],
            journey=record["journey"],
            gesture_id=record["gesture_id"],
            physical_visible=bool(record["physical_visible"]),
            presentation_draw_count=int(record["presentation_draw_count"]),
            tile_presentation_request_count=int(record["tile_presentation_request_count"]),
            tile_presentation_draw_count=int(record["tile_presentation_draw_count"]),
            presentation_draw_pending=bool(record["presentation_draw_pending"]),
            physical_visible_page_count=int(record["physical_visible_page_count"]),
            page_candidate_missing_tile_count=int(record["page_candidate_missing_tile_count"]),
            coverage_pass_open=bool(record["coverage_pass_open"]),
            camera_desired_level=record["camera_desired_level"],
            session_desired_level=record["session_desired_level"],
            applied_level=record["applied_level"],
        )
        self._images.append((path, record))
        self._last_sample_ns = now_ns
        self._last_drawn = drawn
        self._last_identities = identity_text

    def _write_contact_sheet(self) -> None:
        images = [(path, record) for path, record in self._images if path.exists()]
        if not images:
            return
        columns = min(4, len(images))
        thumb_w, thumb_h, label_h = 320, 200, 58
        rows = math.ceil(len(images) / columns)
        sheet = self._QtGui.QPixmap(columns * thumb_w, rows * (thumb_h + label_h))
        sheet.fill(self._QtCore.Qt.GlobalColor.black)
        painter = self._QtGui.QPainter(sheet)
        painter.setPen(self._QtCore.Qt.GlobalColor.white)
        for index, (path, record) in enumerate(images):
            row, column = divmod(index, columns)
            x, y = column * thumb_w, row * (thumb_h + label_h)
            pixmap = self._QtGui.QPixmap(str(path)).scaled(
                thumb_w,
                thumb_h,
                self._QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                self._QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(x + (thumb_w - pixmap.width()) // 2, y, pixmap)
            label = (
                f"{record['sample']:04d} {record['elapsed_ms'] / 1000.0:.1f}s "
                f"{record['phase']}\n"
                f"{record['action']}\n"
                f"capture {record['screenshot_capture_kind']} "
                f"scene {record['drawn_count']}/{record['requested_count']} "
                f"onscreen {record['onscreen_count']} "
                f"+{len(record['appeared_tiles'])} -{len(record['disappeared_tiles'])} "
                f"gap {record['sample_gap_ms']:.0f}ms"
            )
            painter.drawText(x + 4, y + thumb_h + 2, thumb_w - 8, label_h - 4, 0, label)
        painter.end()
        sheet.save(str(self.contact_sheet_path))


def _visual_scene_presented_tiles(
    backend: str,
    *,
    presentation_diagnostics,
    physical_rows,
) -> frozenset[int]:
    """Return named scene primitives, never residency/acknowledgement rows."""

    if str(backend) == "vispy":
        return frozenset(
            int(tile)
            for tile in tuple(dict(presentation_diagnostics or {}).get("presented_tiles", ()) or ())
        )
    # PyQtGraph's physical rows already exclude hidden/empty ImageItems.
    return frozenset(int(tile) for tile in dict(physical_rows or {}))


def _visual_lod_level_counts(payloads, requested) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tile, payload in dict(payloads or {}).items():
        if int(tile) not in requested:
            continue
        lod = getattr(payload, "lod", None)
        level = int(getattr(lod, "level", 0) or 0)
        quality = str(getattr(payload, "quality", "exact") or "exact")
        key = f"{quality}:L{level}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _visual_physical_draw_rows(physical_rows, requested) -> dict[str, dict[str, object]]:
    return {
        str(int(tile)): {
            "texture_kind": dict(row or {}).get("physical_texture_kind"),
            "storage_mode": dict(row or {}).get("physical_storage_mode"),
            "texture_dtype": dict(row or {}).get("physical_texture_dtype"),
            "texture_shape": dict(row or {}).get("physical_texture_shape"),
            "mapping_mode": dict(row or {}).get("physical_mapping_mode"),
            "component_mode": dict(row or {}).get("physical_component_mode"),
            "levels": dict(row or {}).get("physical_levels"),
            "shader_mapping_key": dict(row or {}).get("physical_shader_mapping_key"),
            "draw_world_rects": tuple(dict(row or {}).get("physical_draw_world_rects", ()) or ()),
            "draw_world_bounds": dict(row or {}).get("physical_draw_world_bounds"),
            "draw_uv_rects": tuple(dict(row or {}).get("physical_draw_uv_rects", ()) or ()),
            "expected_world_rect": dict(row or {}).get("physical_expected_world_rect"),
            "bounds_match_layout": dict(row or {}).get("physical_draw_bounds_match_layout"),
            "page_bindings": tuple(
                _visual_page_binding_row(binding)
                for binding in tuple(dict(row or {}).get("physical_page_bindings", ()) or ())
            ),
        }
        for tile, row in dict(physical_rows or {}).items()
        if int(tile) in requested
    }


def _visual_geometry_summary(
    physical_draw_rows,
    *,
    view_range,
    viewport_shape,
) -> dict[str, object]:
    """Compact world/screen geometry oracle for periodic visual evidence.

    The source of truth remains the physical draw rows.  Projecting their
    bounds through the live PyQtGraph camera range makes a mixed-size defect
    searchable in JSONL without trying to infer geometry from a screenshot.
    It also records enough context to distinguish a deliberate maximum-out or
    deep-zoom gesture from stale VisPy vertices/camera state.
    """

    world_sizes: list[tuple[float, float]] = []
    projected_sizes: list[tuple[float, float]] = []
    mismatched_tiles: list[int] = []
    x_span = y_span = 0.0
    viewport_height = viewport_width = 0
    try:
        (x0, x1), (y0, y1) = view_range
        x_span = abs(float(x1) - float(x0))
        y_span = abs(float(y1) - float(y0))
    except Exception:
        pass
    try:
        viewport_height = max(0, int(viewport_shape[0]))
        viewport_width = max(0, int(viewport_shape[1]))
    except Exception:
        pass
    for tile_text, row in dict(physical_draw_rows or {}).items():
        row = dict(row or {})
        if row.get("bounds_match_layout") is False:
            mismatched_tiles.append(int(tile_text))
        bounds = tuple(row.get("draw_world_bounds", ()) or ())
        if len(bounds) != 4:
            continue
        width = abs(float(bounds[2]) - float(bounds[0]))
        height = abs(float(bounds[3]) - float(bounds[1]))
        world_sizes.append((width, height))
        if x_span > 0.0 and y_span > 0.0 and viewport_width > 0 and viewport_height > 0:
            projected_sizes.append(
                (
                    width * float(viewport_width) / x_span,
                    height * float(viewport_height) / y_span,
                )
            )

    def unique(values) -> tuple[tuple[float, float], ...]:
        return tuple(
            sorted({(round(float(width), 6), round(float(height), 6)) for width, height in values})
        )

    unique_world = unique(world_sizes)
    unique_projected = unique(projected_sizes)
    return {
        "world_size_classes": unique_world,
        "projected_pixel_size_classes": unique_projected,
        "mixed_world_sizes": len(unique_world) > 1,
        "mixed_projected_pixel_sizes": len(unique_projected) > 1,
        "bounds_mismatch_tiles": tuple(sorted(mismatched_tiles)),
    }


def _visual_camera_state(win, *, session, live_view_range) -> dict[str, object]:
    """Record every camera representation without making one a new owner."""

    session_range = _normalized_view_range(getattr(session, "view_range", None))
    live_range = _normalized_view_range(live_view_range)
    image_view = getattr(win, "img_view", None)
    raw_key = getattr(image_view, "_vispy_camera_key", None)
    vispy_key_range = _normalized_view_range(None if raw_key is None else tuple(raw_key[:2]))
    camera_rect = None
    try:
        rect = image_view._vispy_view.camera.rect
        camera_rect = (
            float(rect.left),
            float(rect.bottom),
            float(rect.right),
            float(rect.top),
        )
    except Exception:
        pass
    return {
        "session_view_range": session_range,
        "live_view_range": live_range,
        "vispy_camera_key": _trace_identity(raw_key, limit=500),
        "vispy_camera_key_range": vispy_key_range,
        "vispy_camera_rect": camera_rect,
        "session_matches_live": _view_ranges_close(session_range, live_range),
        "vispy_key_matches_live": _view_ranges_close(vispy_key_range, live_range),
    }


def _normalized_view_range(value):
    try:
        return (
            (float(value[0][0]), float(value[0][1])),
            (float(value[1][0]), float(value[1][1])),
        )
    except Exception:
        return None


def _view_ranges_close(left, right, *, tolerance: float = 1e-6) -> bool | None:
    left = _normalized_view_range(left)
    right = _normalized_view_range(right)
    if left is None or right is None:
        return None
    return all(
        abs(float(a) - float(b)) <= float(tolerance) * max(1.0, abs(float(a)), abs(float(b)))
        for left_axis, right_axis in zip(left, right, strict=False)
        for a, b in zip(left_axis, right_axis, strict=False)
    )


def _view_ranges_intersect(left, right) -> bool:
    left = _normalized_view_range(left)
    right = _normalized_view_range(right)
    if left is None or right is None:
        return False
    return all(
        max(min(a0, a1), min(b0, b1)) < min(max(a0, a1), max(b0, b1))
        for (a0, a1), (b0, b1) in zip(left, right, strict=False)
    )


def _view_range_intersects_world_bounds(view_range, bounds) -> bool:
    try:
        world_range = (
            (float(bounds[0]), float(bounds[2])),
            (float(bounds[1]), float(bounds[3])),
        )
    except Exception:
        return False
    return _view_ranges_intersect(view_range, world_range)


def _visual_page_binding_row(binding) -> dict[str, object]:
    binding = dict(binding or {})
    target = binding.get("target_key")
    actual = binding.get("actual_key")
    slot = binding.get("slot")
    return {
        "target_key": _trace_identity(target, limit=500),
        "actual_key": _trace_identity(actual, limit=500),
        "target_origin_yx": tuple(getattr(target, "chunk_origin", ()) or ()),
        "target_shape_yx": tuple(getattr(target, "chunk_shape", ()) or ()),
        "target_reduction_yx": tuple(getattr(getattr(target, "lod", None), "reduction", ()) or ()),
        "actual_origin_yx": tuple(getattr(actual, "chunk_origin", ()) or ()),
        "actual_shape_yx": tuple(getattr(actual, "chunk_shape", ()) or ()),
        "actual_reduction_yx": tuple(getattr(getattr(actual, "lod", None), "reduction", ()) or ()),
        "scale_yx": tuple(float(value) for value in tuple(binding.get("scale", ()) or ())),
        "offset_yx": tuple(float(value) for value in tuple(binding.get("offset", ()) or ())),
        "quality": str(binding.get("quality", "")),
        "page_index": None if slot is None else int(getattr(slot, "page_index", -1)),
        "slot_index": None if slot is None else int(getattr(slot, "slot_index", -1)),
        "binding_generation": int(binding.get("binding_generation", 0) or 0),
    }


class _PresentationContinuityProbe:
    """Observe retained pixels while a successor frame is being prepared.

    A final screenshot cannot expose a transient clear or a camera move into a
    successor layout. Sampling on event-loop turns pins the actual invariant:
    while semantic identity and slot topology remain compatible, predecessor
    items and content extent remain unchanged. Document or topology changes
    are still photographed and timed, but their honest cold slots are not a
    retention failure.
    """

    def __init__(self, QtCore, win):
        self._win = win
        self._timer = QtCore.QTimer(win)
        self._timer.setInterval(1)
        self._timer.timeout.connect(self._sample)
        self._predecessor_frame = getattr(win, "_committed_display_frame", None)
        self._predecessor_count = _backend_visible_tile_count(win)
        self._predecessor_identity = _backend_presentation_identity(win)
        self._predecessor_semantic_key = _current_presentation_semantic_key(win)
        self._predecessor_extent = _viewport_content_extent(win)
        self._predecessor_topology = _current_montage_topology(win)
        self._minimum_count = self._predecessor_count
        self._samples = 0
        self._successor_observed = False
        self._successor_observed_ms = None
        self._successor_levels_state = None
        self._blackout_observed = False
        self._extent_changed_before_commit = False
        self._changed_extent = None
        self._continuity_expected = bool(
            self._predecessor_count > 0 and self._predecessor_semantic_key is not None
        )
        self._topology_changed = False
        self._started_at = None
        self._histogram_timeline: list[dict[str, object]] = []

    def start(self) -> None:
        self._started_at = perf_counter()
        self._sample()
        self._timer.start()

    def stop(self) -> None:
        self._sample()
        self._timer.stop()

    def _sample(self) -> None:
        if self._predecessor_count <= 0:
            return
        current_identity = _backend_presentation_identity(self._win)
        current_semantic_key = _current_presentation_semantic_key(self._win)
        self._topology_changed = bool(
            self._topology_changed
            or _current_montage_topology(self._win) != self._predecessor_topology
        )
        if current_semantic_key != self._predecessor_semantic_key or self._topology_changed:
            # A document/operation or slot-topology transition is not entitled
            # to retain the old mapping. Keep sampling so the first *physical*
            # successor is still timed, but do not grade honest cold slots as
            # a compatible-transition continuity failure.
            self._continuity_expected = False
        self._samples += 1
        count = _backend_visible_tile_count(self._win)
        self._minimum_count = min(self._minimum_count, count)
        successor_visible = bool(count > 0 and current_identity != self._predecessor_identity)
        _append_histogram_timeline_state(
            self._histogram_timeline,
            _levels_histogram_state(self._win),
            elapsed_ms=(
                0.0 if self._started_at is None else (perf_counter() - self._started_at) * 1000.0
            ),
            successor_visible=successor_visible,
        )
        if count <= 0:
            self._blackout_observed = True
        elif successor_visible:
            # The backend atomically accepted successor pixels. Camera/extent
            # changes after this point belong to that successor even when a
            # degraded preview does not yet own exact value semantics.
            self._observe_successor()
            return
        if _viewport_content_extent(self._win) != self._predecessor_extent:
            self._extent_changed_before_commit = True
            self._changed_extent = _viewport_content_extent(self._win)

    def _observe_successor(self) -> None:
        if self._successor_observed:
            return
        self._successor_observed = True
        if self._started_at is not None:
            self._successor_observed_ms = (perf_counter() - self._started_at) * 1000.0
        self._successor_levels_state = _levels_histogram_state(self._win)

    def record(self) -> dict[str, object]:
        expected = bool(self._continuity_expected)
        violation = bool(
            expected and (self._blackout_observed or self._extent_changed_before_commit)
        )
        successor_levels = self._successor_levels_state
        return {
            "presentation_continuity_expected": expected,
            "presentation_continuity_samples": int(self._samples),
            "presentation_predecessor_tile_count": int(self._predecessor_count),
            "presentation_minimum_retained_tile_count": int(self._minimum_count),
            "presentation_successor_observed": bool(self._successor_observed),
            # These canonical milestone fields deliberately override the
            # post-action waiter values in ``_run_phase``. Stress actions run
            # their own nested Qt event loops, so waiting until the action
            # returns records the end of the phase instead of the first
            # successor pixels actually shown to the user.
            **(
                {}
                if self._successor_observed_ms is None
                else {
                    "first_visible_tile_ms": float(self._successor_observed_ms),
                    "first_visible_display_levels": successor_levels.get("display_levels"),
                    "first_visible_histogram_data_bounds": successor_levels.get(
                        "histogram_data_bounds"
                    ),
                    "first_visible_levels_default": bool(
                        successor_levels.get("levels_look_default", True)
                    ),
                    "first_visible_histogram_empty": bool(
                        successor_levels.get("histogram_empty", True)
                    ),
                    "first_visible_level_source_rank": successor_levels.get("level_source_rank"),
                    "first_visible_level_source_count": successor_levels.get("level_source_count"),
                    "first_visible_level_evidence_quality": successor_levels.get(
                        "level_evidence_quality"
                    ),
                    "first_visible_level_decision": successor_levels.get("last_level_decision"),
                }
            ),
            "presentation_blackout_observed": bool(self._blackout_observed),
            "presentation_extent_changed_before_commit": bool(self._extent_changed_before_commit),
            "presentation_topology_changed": bool(self._topology_changed),
            "presentation_predecessor_extent": self._predecessor_extent,
            "presentation_changed_extent": self._changed_extent,
            "presentation_continuity_ok": not violation,
            **_histogram_continuity_metrics(self._histogram_timeline),
        }


def _viewport_content_extent(win) -> tuple[int, int] | None:
    extent = getattr(getattr(win, "img_view", None), "_viewport_content_extent", None)
    if extent is None:
        return None
    try:
        return int(extent[0]), int(extent[1])
    except (TypeError, ValueError, IndexError):
        return None


def _current_montage_topology(win) -> tuple | None:
    plan = getattr(getattr(win, "_frame_session", None), "plan", None)
    if plan is None:
        return None
    return (
        tuple(getattr(plan, "tile_shape", ()) or ()),
        int(getattr(plan, "columns", 0) or 0),
        int(getattr(plan, "rows", 0) or 0),
        int(getattr(plan, "gap", 0) or 0),
        len(tuple(getattr(plan, "tiles", ()) or ())),
    )


def _backend_visible_tile_count(win) -> int:
    image_view = getattr(win, "img_view", None)
    mode = str(getattr(image_view, "montageDisplayMode", lambda: "")())
    if mode == "vispy_tile_layer":
        return int(_vispy_presentation_diagnostics(win).get("presented_tile_count", 0) or 0)
    if mode == "wgpu_tile_layer":
        physical_count = getattr(image_view, "physicalVisibleTileCount", None)
        if callable(physical_count):
            return int(physical_count())
        return int(
            _vispy_presentation_diagnostics(win).get("physically_visible_tile_count", 0) or 0
        )
    layer = getattr(image_view, "_montage_tile_layer", None)
    states = getattr(layer, "states", {}) or {}
    return sum(1 for state in states.values() if bool(getattr(state, "visible", False)))


def _presentation_identity_token(identity):
    """Keep continuity sampling allocation-light without weakening equality.

    Production presentation identities are immutable/hashable tuples.  Retain
    those values directly so the 1 ms blackout probe does not repeatedly build
    multi-kilobyte repr strings for every visible tile.  Diagnostic fakes may
    supply mutable values; only those take the slower textual fallback.
    """

    try:
        hash(identity)
    except (TypeError, ValueError):
        return repr(identity)
    return identity


def _backend_presentation_identity(win) -> tuple[tuple[int, object], ...]:
    image_view = getattr(win, "img_view", None)
    mode = str(getattr(image_view, "montageDisplayMode", lambda: "")())
    if mode == "vispy_tile_layer":
        layer = getattr(image_view, "_vispy_gpu_montage_layer", None)
        stats = getattr(layer, "last_stats", None)
        identities = dict(getattr(stats, "presented_identities", {}) or {})
        if identities:
            return tuple(
                sorted(
                    (int(tile), _presentation_identity_token(identity))
                    for tile, identity in identities.items()
                )
            )
        tiles = tuple(getattr(stats, "presented_tiles", ()) or ())
        return tuple((int(tile), "") for tile in sorted(int(tile) for tile in tiles))
    if mode == "wgpu_tile_layer":
        committed = dict(getattr(image_view, "_wgpu_committed", None) or {})
        tiles = dict(committed.get("tiles", {}) or {})
        return tuple(
            (int(tile), _presentation_identity_token(info.get("identity")))
            for tile, info in sorted(tiles.items())
        )
    layer = getattr(image_view, "_montage_tile_layer", None)
    states = getattr(layer, "states", {}) or {}
    return tuple(
        sorted(
            (
                int(tile),
                _presentation_identity_token(getattr(state, "source_array_id", None)),
            )
            for tile, state in states.items()
            if bool(getattr(state, "visible", False))
        )
    )


def _current_presentation_semantic_key(win):
    frame = getattr(win, "_committed_display_frame", None)
    if frame is not None:
        return getattr(getattr(frame, "key", None), "semantic_key", None)
    session = getattr(win, "_frame_session", None)
    return None if session is None else getattr(session, "semantic_key", None)


def _apply_fft_level_refinement_preview(win, *, app=None, QtCore=None) -> dict[str, object]:
    bounds = None
    try:
        bounds = win.img_view.getHistogramDataBounds()
    except Exception:
        bounds = None
    if bounds is None:
        try:
            bounds = win.img_view.getLevels()
        except Exception:
            bounds = None
    if bounds is None:
        bounds = (0.0, 1.0)
    low, high = (float(bounds[0]), float(bounds[1]))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        low, high = 0.0, 1.0
    span = high - low
    try:
        base_levels = tuple(float(value) for value in win.img_view.getLevels())
    except Exception:
        base_levels = (low, high)
    if (
        len(base_levels) != 2
        or not math.isfinite(base_levels[0])
        or not math.isfinite(base_levels[1])
        or base_levels[1] <= base_levels[0]
    ):
        base_levels = (low, high)
    apply_preview = getattr(win.img_view, "_apply_histogram_preview_levels", None)
    action_start = perf_counter()
    first_half_stats = _new_histogram_loop_stats()
    second_half_stats = _new_histogram_loop_stats()
    for step in range(10):
        offset = -0.05 * step * span
        levels = (base_levels[0] + offset, base_levels[1] + offset)
        draw_start = _vispy_draw_count(win)
        preview_start = perf_counter()
        with contextlib.suppress(Exception):
            win.img_view.histogram.setLevels(float(levels[0]), float(levels[1]))
        target = first_half_stats if step < 5 else second_half_stats
        if step < 5:
            step_elapsed_ms = (perf_counter() - preview_start) * 1000.0
            _add_histogram_loop_timing(target, None, step_elapsed_ms=step_elapsed_ms)
            if app is not None and QtCore is not None:
                _flush_histogram_widget_redraw(win, app, QtCore)
        else:
            immediate_timing = None
            if callable(apply_preview):
                apply_preview(levels, final=True)
                immediate_timing = win.img_view.lastImageUploadTiming()
            else:
                win.img_view.setLevels(levels[0], levels[1])
            if app is not None and QtCore is not None:
                _wait_for_montage_complete(
                    app,
                    QtCore,
                    win,
                    timeout_s=INTERACTION_SETTLE_HARD_LIMIT_S,
                    start=preview_start,
                    draw_start=draw_start,
                    require_presentation_settled=True,
                )
            settled_timing = win.img_view.lastImageUploadTiming()
            timing = (
                immediate_timing if _timing_has_level_work(immediate_timing) else settled_timing
            )
            step_elapsed_ms = (perf_counter() - preview_start) * 1000.0
            _add_histogram_loop_timing(target, timing, step_elapsed_ms=step_elapsed_ms)
    finish = getattr(win.img_view, "_on_histogram_level_change_finished", None)
    if callable(finish):
        finish()
    total_stats = _combine_histogram_loop_stats(first_half_stats, second_half_stats)
    final_level_state = _montage_level_presentation_state(win)
    return {
        "histogram_loop_steps": 10,
        "histogram_loop_first_half_steps": 5,
        "histogram_loop_second_half_steps": 5,
        "histogram_loop_pacing": "5_histogram_redraw_then_5_full_render_redraw",
        "histogram_loop_direction": "down_from_auto_default_5_percent_steps",
        "histogram_loop_base_levels": [float(base_levels[0]), float(base_levels[1])],
        "histogram_loop_action_ms": (perf_counter() - action_start) * 1000.0,
        "histogram_loop_final_level_settled": bool(final_level_state["settled"]),
        "histogram_loop_final_level_revision": int(final_level_state["revision"]),
        "histogram_loop_final_stale_level_tiles": int(final_level_state["stale_tiles"]),
        "histogram_loop_final_pending_level_tiles": int(final_level_state["pending_tiles"]),
        "histogram_loop_final_active_level_value_count": int(
            final_level_state["active_level_value_count"]
        ),
        **_histogram_loop_record_fields("histogram_loop", total_stats),
        **_histogram_loop_record_fields("histogram_loop_first_half", first_half_stats),
        **_histogram_loop_record_fields("histogram_loop_second_half", second_half_stats),
    }


def _flush_histogram_widget_redraw(win, app, QtCore) -> None:
    histogram = getattr(getattr(win, "img_view", None), "histogram", None)
    if histogram is not None:
        with contextlib.suppress(Exception):
            histogram.update()
    _process_events(app, QtCore, count=2)


def _new_histogram_loop_stats() -> dict[str, object]:
    return {
        "steps": 0,
        "step_ms": [],
        "rgb_window_ms": 0.0,
        "rgb_window_tiles": 0,
        "texture_uploads": 0,
        "texture_upload_bytes": 0,
        "level_updates": 0,
        "shader_uniform_updates": 0,
        "items_updated": 0,
        "items_skipped": 0,
    }


def _add_histogram_loop_timing(stats: dict[str, object], timing, *, step_elapsed_ms: float) -> None:
    stats["steps"] = int(stats["steps"]) + 1
    cast_steps = stats["step_ms"]
    assert isinstance(cast_steps, list)
    cast_steps.append(float(step_elapsed_ms))
    if timing is None:
        return
    stats["rgb_window_ms"] = float(stats["rgb_window_ms"]) + float(
        getattr(timing, "tile_layer_rgb_window_ms", 0.0) or 0.0
    )
    stats["rgb_window_tiles"] = int(stats["rgb_window_tiles"]) + int(
        getattr(timing, "tile_layer_rgb_window_tiles", 0) or 0
    )
    stats["texture_uploads"] = int(stats["texture_uploads"]) + int(
        getattr(timing, "tile_layer_texture_uploads", 0) or 0
    )
    stats["texture_upload_bytes"] = int(stats["texture_upload_bytes"]) + int(
        getattr(timing, "tile_layer_texture_upload_bytes", 0) or 0
    )
    stats["level_updates"] = int(stats["level_updates"]) + int(
        getattr(timing, "tile_layer_level_updates", 0) or 0
    )
    stats["shader_uniform_updates"] = int(stats["shader_uniform_updates"]) + int(
        getattr(timing, "tile_layer_shader_uniform_updates", 0) or 0
    )
    stats["items_updated"] = int(stats["items_updated"]) + int(
        getattr(timing, "tile_layer_items_updated", 0) or 0
    )
    stats["items_skipped"] = int(stats["items_skipped"]) + int(
        getattr(timing, "tile_layer_items_skipped", 0) or 0
    )


def _timing_has_level_work(timing) -> bool:
    if timing is None:
        return False
    return any(
        int(getattr(timing, field, 0) or 0) > 0
        for field in (
            "tile_layer_rgb_window_tiles",
            "tile_layer_level_updates",
            "tile_layer_shader_uniform_updates",
            "tile_layer_items_updated",
        )
    )


def _combine_histogram_loop_stats(*stats_items: dict[str, object]) -> dict[str, object]:
    combined = _new_histogram_loop_stats()
    combined_steps = combined["step_ms"]
    assert isinstance(combined_steps, list)
    for stats in stats_items:
        combined["steps"] = int(combined["steps"]) + int(stats["steps"])
        step_ms = stats["step_ms"]
        assert isinstance(step_ms, list)
        combined_steps.extend(float(value) for value in step_ms)
        for key in (
            "rgb_window_ms",
            "rgb_window_tiles",
            "texture_uploads",
            "texture_upload_bytes",
            "level_updates",
            "shader_uniform_updates",
            "items_updated",
            "items_skipped",
        ):
            combined[key] = combined[key] + stats[key]
    return combined


def _histogram_loop_record_fields(prefix: str, stats: dict[str, object]) -> dict[str, object]:
    step_ms = stats["step_ms"]
    assert isinstance(step_ms, list)
    return {
        f"{prefix}_steps": int(stats["steps"]),
        f"{prefix}_step_max_ms": max(step_ms) if step_ms else 0.0,
        f"{prefix}_step_mean_ms": sum(step_ms) / len(step_ms) if step_ms else 0.0,
        f"{prefix}_rgb_window_ms": float(stats["rgb_window_ms"]),
        f"{prefix}_rgb_window_tiles": int(stats["rgb_window_tiles"]),
        f"{prefix}_texture_uploads": int(stats["texture_uploads"]),
        f"{prefix}_texture_upload_bytes": int(stats["texture_upload_bytes"]),
        f"{prefix}_level_updates": int(stats["level_updates"]),
        f"{prefix}_shader_uniform_updates": int(stats["shader_uniform_updates"]),
        f"{prefix}_items_updated": int(stats["items_updated"]),
        f"{prefix}_items_skipped": int(stats["items_skipped"]),
    }


def _run_phase(
    app,
    QtCore,
    win,
    probe: _EventLoopProbe,
    *,
    phase: str,
    timeout_s: float,
    action,
    backend: str = "",
    screenshot_dir: Path | None = None,
    build_phase: bool = False,
) -> dict[str, object]:
    # Cold-fill build phases carry a build-scale completion budget
    # (COLD_FILL_BUILD_TIMEOUT_S); the interaction clamp is for gesture
    # probes.  The stall detector inside the completion wait still fails a
    # genuine wedge fast under either budget.
    timeout_s = (
        max(float(timeout_s), bounded_interaction_settle_timeout_s(None))
        if build_phase
        else bounded_interaction_settle_timeout_s(timeout_s)
    )
    win._arrayscope_profile_phase = str(phase)
    visual_probe = getattr(win, "_arrayscope_visual_timeline_probe", None)
    if visual_probe is not None:
        visual_probe.capture("phase-start")
    emit_trace("input", action="phase_start", phase=str(phase), backend=str(backend))
    journey_gesture_id = (
        _start_journey_gesture(win, "cold_fill") if str(phase) == "raw_full_tiled_montage" else None
    )
    probe.reset()
    phase_start_geometry = _window_geometry_state(win)
    phase_start_controller = getattr(getattr(win, "img_view", None), "viewport_controller", None)
    phase_start_mode = getattr(getattr(phase_start_controller, "mode", None), "value", None)
    continuity_probe = _PresentationContinuityProbe(QtCore, win)
    continuity_probe.start()
    start = perf_counter()
    begin_physical_timeline = getattr(win.img_view, "beginPhysicalTileTimeline", None)
    end_physical_timeline = getattr(win.img_view, "endPhysicalTileTimeline", None)
    if callable(begin_physical_timeline):
        begin_physical_timeline()
    physical_tile_timeline: tuple[dict[str, object], ...] = ()
    draw_start = _vispy_draw_count(win)
    phase_ui_work_start = _recent_ui_work_observations(win)
    governor = getattr(win, "resource_governor", None)
    begin_ui_epoch = getattr(governor, "begin_ui_observation_epoch", None)
    phase_ui_work_epoch = begin_ui_epoch() if callable(begin_ui_epoch) else None
    phase_session = getattr(win, "_frame_session", None)
    predecessor_frame = getattr(win, "_committed_display_frame", None)
    predecessor_presentation_identity = _backend_presentation_identity(win)
    predecessor_semantic_key = _current_presentation_semantic_key(win)
    preview_floor_session_id = (
        None if phase_session is None else int(getattr(phase_session, "session_id", -1) or -1)
    )
    preview_floor_count_start = (
        0
        if phase_session is None
        else int(getattr(phase_session, "lod_preview_presentations", 0) or 0)
    )
    action_start = perf_counter()
    try:
        action_result = action()
        action_elapsed_ms = (perf_counter() - action_start) * 1000.0
        milestones = _wait_for_montage_complete(
            app,
            QtCore,
            win,
            timeout_s=timeout_s,
            build_budget=build_phase,
            start=start,
            draw_start=draw_start,
            preview_floor_session_id=preview_floor_session_id,
            preview_floor_count_start=preview_floor_count_start,
            preview_floor_screenshot_path=(
                None
                if screenshot_dir is None
                else screenshot_dir / f"{backend}-{phase}_preview_floor.png"
            ),
            predecessor_frame=predecessor_frame,
            predecessor_presentation_identity=predecessor_presentation_identity,
            predecessor_semantic_key=predecessor_semantic_key,
        )
    finally:
        if callable(end_physical_timeline):
            physical_tile_timeline = tuple(end_physical_timeline())
        if journey_gesture_id is not None:
            _finish_journey_gesture(win, journey_gesture_id, app=app, QtCore=QtCore)
        continuity_probe.stop()
        if visual_probe is not None:
            visual_probe.capture("phase-end")
    elapsed_ms = (perf_counter() - start) * 1000.0
    _process_events(app, QtCore, count=5)
    record = _phase_record(
        win,
        phase=phase,
        elapsed_ms=elapsed_ms,
        event_loop_p95_gap_ms=probe.percentile_ms(95),
        event_loop_p99_gap_ms=probe.percentile_ms(99),
        event_loop_max_gap_ms=probe.max_gap_ms,
        phase_ui_work_start=phase_ui_work_start,
        phase_ui_work_epoch=phase_ui_work_epoch,
    )
    record.update(
        {
            "phase_start_image_axes": phase_start_geometry.get("image_axes"),
            "phase_start_axis_flipped": phase_start_geometry.get("axis_flipped"),
            "phase_start_viewport_shape": phase_start_geometry.get("viewport_shape"),
            "phase_start_viewport_mode": phase_start_mode,
        }
    )
    if isinstance(action_result, dict):
        record.update(action_result)
    record["action_elapsed_ms"] = float(action_elapsed_ms)
    record.update(milestones)
    record.update(
        _physical_tile_timeline_metrics(
            physical_tile_timeline,
            phase_start_s=start,
            requested_tiles=int(milestones.get("requested_tile_count", 0) or 0),
            target_presentation_identity=tuple(
                (int(tile), str(row.get("physical_target_identity", "")))
                for tile, row in sorted(
                    dict(getattr(win.img_view, "tileTruthPhysicalRows", dict)()).items()
                )
            ),
        )
    )
    record.update(continuity_probe.record())
    record.update(_levels_histogram_state(win))
    record["event_loop_ticks"] = int(probe.tick_count)
    pump_ms = tuple(
        float(value) for value in getattr(app, "_arrayscope_profile_event_pump_ms", ()) or ()
    )
    record["event_pump_max_ms"] = float(max(pump_ms) if pump_ms else 0.0)
    record["event_pump_p95_ms"] = float(_percentile(pump_ms, 95.0))
    record["event_pump_count"] = len(pump_ms)
    emit_trace(
        "input",
        action="phase_complete",
        phase=str(phase),
        backend=str(backend),
        elapsed_ms=float(elapsed_ms),
    )
    return record


def _levels_histogram_state(win) -> dict[str, object]:
    """Capture display levels + histogram data state to expose the levels bug.

    Surfaces whether the montage is stranded at default ``(0, 1)`` levels or an
    empty histogram — the symptom of level/histogram metadata not being
    (re)published for the current montage payloads.
    """

    image_view = getattr(win, "img_view", None)
    levels = None
    try:
        raw = image_view.getLevels()
        if raw is not None:
            levels = (float(raw[0]), float(raw[1]))
    except Exception:
        levels = None
    bounds = None
    try:
        raw_bounds = image_view.getHistogramDataBounds()
        if raw_bounds is not None:
            bounds = (float(raw_bounds[0]), float(raw_bounds[1]))
    except Exception:
        bounds = None
    session = getattr(win, "_frame_session", None)
    committed_frame = getattr(win, "_committed_display_frame", None)
    applied = getattr(committed_frame, "level_source", None)
    if applied is None:
        applied = getattr(session, "applied_level_source", None)
    levels_look_default = bool(
        levels is not None and abs(levels[0]) <= 1e-9 and abs(levels[1] - 1.0) <= 1e-9
    )
    histogram_empty = bool(
        bounds is None
        or not math.isfinite(bounds[0])
        or not math.isfinite(bounds[1])
        or bounds[1] <= bounds[0]
    )
    renderer = getattr(win, "renderer", None)
    decision = getattr(renderer, "_last_montage_level_decision", None)
    refined_applied = int(getattr(renderer, "_montage_refined_level_applied_count", 0) or 0)
    return {
        "montage_refined_level_applied_count": refined_applied,
        "display_levels": None if levels is None else [levels[0], levels[1]],
        "histogram_data_bounds": None if bounds is None else [bounds[0], bounds[1]],
        "levels_look_default": levels_look_default,
        "histogram_empty": histogram_empty,
        "level_source_rank": None if applied is None else int(getattr(applied, "rank", 0) or 0),
        "level_source_count": None
        if applied is None
        else int(getattr(applied, "source_count", 0) or 0),
        "level_action_id": None
        if session is None
        else int(getattr(session, "session_id", -1) or -1),
        "level_semantic_key": None
        if applied is None
        else _trace_identity(getattr(applied, "semantic_key", None), limit=500),
        "level_evidence_quality": None
        if applied is None
        else int(getattr(applied, "evidence_quality", 0) or 0),
        "last_level_decision": None if decision is None else dict(decision),
    }


def _append_histogram_timeline_state(
    rows: list[dict[str, object]],
    state: dict[str, object],
    *,
    elapsed_ms: float,
    successor_visible: bool,
    limit: int = 128,
) -> None:
    """Retain only observable histogram/window-level state transitions."""

    row = {
        "elapsed_ms": float(elapsed_ms),
        "successor_visible": bool(successor_visible),
        "display_levels": state.get("display_levels"),
        "histogram_data_bounds": state.get("histogram_data_bounds"),
        "levels_look_default": bool(state.get("levels_look_default", True)),
        "histogram_empty": bool(state.get("histogram_empty", True)),
        "level_source_rank": state.get("level_source_rank"),
        "level_source_count": state.get("level_source_count"),
        "level_action_id": state.get("level_action_id"),
        "level_semantic_key": state.get("level_semantic_key"),
        "level_evidence_quality": state.get("level_evidence_quality"),
    }
    signature = tuple(
        (tuple(value) if isinstance(value, list) else value)
        for key, value in row.items()
        if key != "elapsed_ms"
    )
    if rows and rows[-1].get("_signature") == signature:
        return
    row["_signature"] = signature
    if len(rows) < max(1, int(limit)):
        rows.append(row)
        return
    # Preserve the first transition and the latest truth under pathological
    # churn. The truncation flag in the summary makes the lost detail loud.
    rows[-1] = row


def _histogram_continuity_metrics(rows) -> dict[str, object]:
    """Summarize visible window/level flicker from a compact state timeline."""

    timeline = tuple(dict(row) for row in tuple(rows or ()))
    visible = tuple(row for row in timeline if bool(row.get("successor_visible", False)))
    considered = visible if visible else timeline[-1:]
    transient_span_dip_ratio = 1.0
    center_excursion_fraction = 0.0
    source_count_regressed = False
    semantic_segments: list[list[dict[str, object]]] = []
    for row in considered:
        semantic_key = row.get(
            "level_action_id",
            row.get("level_semantic_key", "__legacy_single_semantic__"),
        )
        if (
            not semantic_segments
            or semantic_segments[-1][-1].get(
                "level_action_id",
                semantic_segments[-1][-1].get("level_semantic_key", "__legacy_single_semantic__"),
            )
            != semantic_key
        ):
            semantic_segments.append([])
        semantic_segments[-1].append(row)
    for segment in semantic_segments:
        bounds_rows = []
        for row in segment:
            bounds = row.get("histogram_data_bounds")
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                continue
            low, high = (float(bounds[0]), float(bounds[1]))
            if math.isfinite(low) and math.isfinite(high) and high > low:
                bounds_rows.append((row, low, high))
        if len(bounds_rows) >= 2:
            first_low, first_high = bounds_rows[0][1:]
            final_low, final_high = bounds_rows[-1][1:]
            first_span = first_high - first_low
            final_span = final_high - final_low
            endpoint_floor = min(first_span, final_span)
            if endpoint_floor > 0.0:
                transient_span_dip_ratio = min(
                    transient_span_dip_ratio,
                    min(high - low for _row, low, high in bounds_rows) / endpoint_floor,
                )
            first_center = 0.5 * (first_low + first_high)
            final_center = 0.5 * (final_low + final_high)
            center_low = min(first_center, final_center)
            center_high = max(first_center, final_center)
            normalization = max(first_span, final_span, np.finfo(float).eps)
            segment_excursion = max(
                max(
                    center_low - 0.5 * (low + high),
                    0.5 * (low + high) - center_high,
                    0.0,
                )
                / normalization
                for _row, low, high in bounds_rows
            )
            center_excursion_fraction = max(
                center_excursion_fraction,
                segment_excursion,
            )
        counts = tuple(
            int(row["level_source_count"])
            for row in segment
            if row.get("level_source_count") is not None
        )
        source_count_regressed = bool(
            source_count_regressed
            or (
                len(counts) >= 3
                and min(counts[1:-1], default=counts[0]) < min(counts[0], counts[-1])
            )
        )
    histogram_emptied = any(bool(row.get("histogram_empty", True)) for row in considered)
    levels_defaulted = any(bool(row.get("levels_look_default", True)) for row in considered)
    flicker_free = bool(
        not histogram_emptied
        and not levels_defaulted
        and transient_span_dip_ratio >= 0.75
        and center_excursion_fraction <= 0.25
        and not source_count_regressed
    )
    clean_rows = []
    for row in timeline:
        row.pop("_signature", None)
        clean_rows.append(row)
    return {
        "histogram_timeline": clean_rows,
        "histogram_timeline_transition_count": len(timeline),
        "histogram_timeline_truncated": bool(len(timeline) >= 128),
        "histogram_visible_state_count": len(considered),
        "histogram_emptied_after_successor_visible": bool(histogram_emptied),
        "levels_defaulted_after_successor_visible": bool(levels_defaulted),
        "level_transient_span_dip_ratio": float(transient_span_dip_ratio),
        "level_center_excursion_fraction": float(center_excursion_fraction),
        "level_source_count_regressed": bool(source_count_regressed),
        "window_level_flicker_free": flicker_free,
    }


def _post_visible_gate_blockers(
    *,
    fully_visible: bool,
    requested_grid_visible: bool,
    physical_drawn: bool,
    presentation_ready: bool,
    target_settled: bool,
    work_in_flight: bool,
    dirty_payloads: bool,
) -> tuple[str, ...]:
    """Name the completion gates a fully visible montage is still blocked on.

    Returns () while the montage is not yet fully visible, while real work is
    in flight, or while commit batches are still draining — those states are
    progress, not a stall.  A non-empty result that persists means the frame
    looks done but a completion gate can never close (e.g. the presentation
    layer re-committing forever, or level settlement deadlocked with an
    exhausted evidence tracker): the wait must bail with this diagnosis
    instead of burning the full phase timeout.
    """

    if not fully_visible or work_in_flight or dirty_payloads:
        return ()
    blockers = []
    if not requested_grid_visible:
        blockers.append("requested_grid_visible")
    if not physical_drawn:
        blockers.append("physical_drawn")
    if not presentation_ready:
        blockers.append("presentation_settled")
    if not target_settled:
        blockers.append("required_target_settled")
    return tuple(blockers)


def _physical_tile_timeline_metrics(
    rows: tuple[dict[str, object], ...],
    *,
    phase_start_s: float,
    requested_tiles: int,
    target_presentation_identity: tuple[tuple[int, str], ...],
) -> dict[str, object]:
    """Summarize physical WGPU draw edges without level/evidence settlement."""

    start_ns = int(float(phase_start_s) * 1_000_000_000.0)
    timeline = []
    for raw in sorted(rows, key=lambda row: int(row.get("timestamp_ns", 0) or 0)):
        row = dict(raw)
        timestamp_ns = int(row.pop("timestamp_ns", 0) or 0)
        row["elapsed_ms"] = max(0.0, (timestamp_ns - start_ns) / 1_000_000.0)
        identity = {
            int(tile): str(token)
            for tile, token in tuple(row.get("presentation_identity", ()) or ())
        }
        row["target_tile_count"] = sum(
            identity.get(int(tile)) == str(token) for tile, token in target_presentation_identity
        )
        timeline.append(row)

    visible = [row for row in timeline if int(row.get("target_tile_count", 0) or 0) > 0]
    first = visible[0] if visible else None
    full = next(
        (
            row
            for row in visible
            if requested_tiles > 0 and int(row.get("target_tile_count", 0) or 0) >= requested_tiles
        ),
        None,
    )
    first_ms = None if first is None else float(first["elapsed_ms"])
    full_ms = None if full is None else float(full["elapsed_ms"])
    rate = None
    if first is not None and full is not None and full_ms is not None and first_ms is not None:
        elapsed_s = (full_ms - first_ms) / 1000.0
        added = int(full.get("target_tile_count", 0) or 0) - int(
            first.get("target_tile_count", 0) or 0
        )
        rate = None if elapsed_s <= 0.0 else added / elapsed_s

    milestone_ms: dict[str, float | None] = {}
    for percent in (25, 50, 75, 100):
        target = math.ceil(requested_tiles * percent / 100.0) if requested_tiles else 0
        match = next(
            (row for row in visible if int(row.get("target_tile_count", 0) or 0) >= target),
            None,
        )
        milestone_ms[str(percent)] = None if match is None else float(match["elapsed_ms"])
    return {
        "physical_tile_draw_events": len(timeline),
        "physical_tile_first_ms": first_ms,
        "physical_tile_full_ms": full_ms,
        "physical_tile_rate_after_first_per_s": rate,
        "physical_tile_milestone_ms": milestone_ms,
        "physical_tile_timeline": timeline,
        "physical_tile_timeline_scope": (
            "WGPU canvas draw edges and page-backed tile rows only; excludes histogram, "
            "semantic evidence, and level-settlement gates"
        ),
    }


def _wait_for_montage_complete(
    app,
    QtCore,
    win,
    *,
    timeout_s: float,
    start: float,
    draw_start: int,
    require_presentation_settled: bool = True,
    preview_floor_session_id: int | None = None,
    preview_floor_count_start: int = 0,
    preview_floor_screenshot_path: Path | None = None,
    predecessor_frame=None,
    predecessor_presentation_identity=None,
    predecessor_semantic_key=None,
    build_budget: bool = False,
) -> dict[str, float | int | bool | None]:
    if not build_budget:
        timeout_s = bounded_interaction_settle_timeout_s(timeout_s)
    deadline = time.monotonic() + timeout_s
    first_materialized_tile_ms = None
    first_presented_tile_ms = None
    first_display_committed_ms = None
    first_overlay_clear_ms = None
    saw_overlays = _montage_overlay_count(win) > 0
    first_logical_complete_ms = None
    draw_after_complete_ms = None
    physical_draw_after_complete_ms = None
    fully_visible_ms = None
    first_visible_tile_ms = None
    first_display_payload_ms = None
    first_display_payload_fill_ms = None
    first_preview_payload_ms = None
    first_preview_payload_fill_ms = None
    first_preview_floor_fill_ms = None
    first_histogram_data_ms = None
    first_nondefault_levels_ms = None
    first_visible_levels_state: dict[str, object] | None = None
    first_visible_level_decision: dict[str, object] | None = None
    first_visible_reused_compatible_predecessor = False
    preview_floor_screenshot_saved = None
    preview_floor_screenshot_error = None
    preview_floor_physical_rows: list[dict[str, object]] = []
    presentation_settled_ms = None
    target_settled_ms = None
    final_visibility_state: dict[str, object] = {}
    final_level_state: dict[str, object] = {}
    final_payload_state: dict[str, object] = {}
    stall_since = None
    stall_last_sig = None
    stall_grace_s = 4.0
    stalled = False
    post_visible_since = None
    post_visible_blockers: tuple[str, ...] = ()
    histogram_timeline: list[dict[str, object]] = []
    initial_histogram_state = _levels_histogram_state(win)
    _append_histogram_timeline_state(
        histogram_timeline,
        initial_histogram_state,
        elapsed_ms=0.0,
        successor_visible=False,
    )
    while time.monotonic() < deadline:
        _process_events(app, QtCore, count=2)
        session = getattr(win, "_frame_session", None)
        mode = getattr(win.img_view, "montageDisplayMode", lambda: "")()
        level_state = _montage_level_presentation_state(win)
        final_level_state = level_state
        visibility_state = _montage_visibility_state(win, mode=str(mode))
        final_visibility_state = visibility_state
        levels_state = _levels_histogram_state(win)
        elapsed_now_ms = (perf_counter() - start) * 1000.0
        if first_histogram_data_ms is None or first_nondefault_levels_ms is None:
            if first_histogram_data_ms is None and not bool(levels_state["histogram_empty"]):
                first_histogram_data_ms = elapsed_now_ms
            if first_nondefault_levels_ms is None and not bool(levels_state["levels_look_default"]):
                first_nondefault_levels_ms = elapsed_now_ms
        presentation_ready = bool(level_state["settled"]) or not bool(require_presentation_settled)
        target_settled = bool(
            session is not None
            and callable(getattr(session, "required_target_settled", None))
            and session.required_target_settled()
            and not bool(getattr(session, "atomic_successor_pending", False))
        )
        # Fail-fast only while the visible frame is still owed.  Warm/offscreen
        # work must not make a completed visible frame look frozen.
        if not (
            bool(visibility_state["fully_visible"]) and presentation_ready
        ) and not _montage_settled(session):
            sig = _montage_stall_signature(session)
            if _montage_work_in_flight(session):
                stall_since = None
            elif sig != stall_last_sig:
                stall_since = time.monotonic()
            elif stall_since is not None and time.monotonic() - stall_since >= stall_grace_s:
                stalled = True
                break
            stall_last_sig = sig
        else:
            stall_since = None
        vispy_tiled = str(mode) == "vispy_tile_layer" and _vispy_canvas_visible(win)
        wgpu_tiled = str(mode) == "wgpu_tile_layer"
        pyqtgraph_tiled = str(mode) == "tile_layer"
        if session is not None:
            if first_materialized_tile_ms is None and bool(
                getattr(session, "rendered_tiles", None)
            ):
                first_materialized_tile_ms = (perf_counter() - start) * 1000.0
            if first_presented_tile_ms is None and bool(session.lifecycle.presented_tiles):
                first_presented_tile_ms = (perf_counter() - start) * 1000.0
            if first_display_committed_ms is None and bool(
                getattr(session, "display_committed", False)
            ):
                first_display_committed_ms = (perf_counter() - start) * 1000.0
        overlay_count = _montage_overlay_count(win)
        saw_overlays = bool(saw_overlays or overlay_count > 0)
        if saw_overlays and first_overlay_clear_ms is None and overlay_count == 0:
            first_overlay_clear_ms = (perf_counter() - start) * 1000.0
        logical_complete = (
            session is not None
            and bool(getattr(session, "display_committed", False))
            and session.is_complete()
            and mode in {"tile_layer", "vispy_tile_layer", "wgpu_tile_layer"}
        )
        if logical_complete and first_logical_complete_ms is None:
            first_logical_complete_ms = (perf_counter() - start) * 1000.0
        if bool(level_state["settled"]) and presentation_settled_ms is None:
            presentation_settled_ms = (perf_counter() - start) * 1000.0
        if target_settled and target_settled_ms is None:
            target_settled_ms = (perf_counter() - start) * 1000.0
        current_frame = getattr(win, "_committed_display_frame", None)
        current_presentation_identity = _backend_presentation_identity(win)
        successor_pixels_observed = bool(
            predecessor_frame is None
            or current_frame is not predecessor_frame
            or (
                predecessor_presentation_identity is not None
                and current_presentation_identity != predecessor_presentation_identity
            )
        )
        _append_histogram_timeline_state(
            histogram_timeline,
            levels_state,
            elapsed_ms=elapsed_now_ms,
            successor_visible=bool(
                successor_pixels_observed
                and int(visibility_state["active_presented_tile_count"]) > 0
            ),
        )
        if (
            first_visible_tile_ms is None
            and successor_pixels_observed
            and int(visibility_state["active_presented_tile_count"]) > 0
        ):
            first_visible_tile_ms = (perf_counter() - start) * 1000.0
            first_visible_levels_state = _levels_histogram_state(win)
            first_visible_level_decision = first_visible_levels_state.get("last_level_decision")
            first_visible_reused_compatible_predecessor = bool(
                predecessor_semantic_key is not None
                and predecessor_semantic_key == _current_presentation_semantic_key(win)
            )
        if session is not None:
            payload_state = _montage_display_payload_state(
                session, active_tiles=visibility_state["active_tiles"]
            )
            final_payload_state = payload_state
            preview_floor_count = int(getattr(session, "lod_preview_presentations", 0) or 0)
            session_id = int(getattr(session, "session_id", -1) or -1)
            preview_floor_delta = (
                preview_floor_count - int(preview_floor_count_start)
                if preview_floor_session_id is not None
                and session_id == int(preview_floor_session_id)
                else preview_floor_count
            )
            preview_floor_target = int(visibility_state["active_planned_tile_count"]) or int(
                visibility_state["requested_tile_count"]
            )
            if (
                first_preview_floor_fill_ms is None
                and preview_floor_target > 0
                and preview_floor_delta >= preview_floor_target
            ):
                first_preview_floor_fill_ms = (perf_counter() - start) * 1000.0
                if (
                    preview_floor_screenshot_path is not None
                    and preview_floor_screenshot_saved is None
                ):
                    _wait_for_tile_presentation_draw(win, app, QtCore)
                    try:
                        preview_floor_screenshot_saved = _save_view_screenshot(
                            win,
                            preview_floor_screenshot_path,
                        )
                        preview_floor_physical_rows = _preview_floor_physical_rows(win)
                    except Exception as exc:  # pragma: no cover - diagnostic path
                        preview_floor_screenshot_saved = False
                        preview_floor_screenshot_error = repr(exc)
            if first_display_payload_ms is None and payload_state["display_payload_count"] > 0:
                first_display_payload_ms = (perf_counter() - start) * 1000.0
            if first_preview_payload_ms is None and payload_state["preview_payload_count"] > 0:
                first_preview_payload_ms = (perf_counter() - start) * 1000.0
            if first_display_payload_fill_ms is None and payload_state["display_payload_fill"]:
                first_display_payload_fill_ms = (perf_counter() - start) * 1000.0
            if first_preview_payload_fill_ms is None and payload_state["preview_payload_fill"]:
                first_preview_payload_fill_ms = (perf_counter() - start) * 1000.0
                if (
                    preview_floor_screenshot_path is not None
                    and preview_floor_screenshot_saved is None
                ):
                    _wait_for_tile_presentation_draw(win, app, QtCore)
                    try:
                        preview_floor_screenshot_saved = _save_view_screenshot(
                            win,
                            preview_floor_screenshot_path,
                        )
                        preview_floor_physical_rows = _preview_floor_physical_rows(win)
                    except Exception as exc:  # pragma: no cover - diagnostic path
                        preview_floor_screenshot_saved = False
                        preview_floor_screenshot_error = repr(exc)
        fully_visible = bool(visibility_state["fully_visible"])
        requested_grid_visible = bool(
            int(visibility_state["requested_tile_count"]) > 0
            and int(visibility_state["active_planned_tile_count"])
            == int(visibility_state["requested_tile_count"])
        )
        if fully_visible and fully_visible_ms is None:
            fully_visible_ms = (perf_counter() - start) * 1000.0
        current_request_count = _vispy_tile_presentation_request_count(win)
        final_drawn = _vispy_tile_presentation_draw_count(win) >= int(current_request_count)
        draw_pending_fn = getattr(win.img_view, "presentationDrawPending", None)
        pyqtgraph_final_drawn = not bool(callable(draw_pending_fn) and draw_pending_fn())
        physical_drawn = bool(
            (not (vispy_tiled or wgpu_tiled) or final_drawn)
            and (not pyqtgraph_tiled or pyqtgraph_final_drawn)
        )
        if (
            fully_visible
            and requested_grid_visible
            and physical_drawn
            and presentation_ready
            and target_settled
        ):
            if (
                first_visible_levels_state is None
                and predecessor_semantic_key is not None
                and predecessor_semantic_key == _current_presentation_semantic_key(win)
                and int(visibility_state["active_presented_tile_count"]) > 0
            ):
                # A level-only/no-op phase can deliberately retain the same
                # compatible physical payload identity. There is no successor
                # pixel event to observe, but those pixels were visible from
                # phase start and their contemporaneous levels/histogram are
                # the correct first-visible evidence.
                first_visible_tile_ms = 0.0
                first_visible_levels_state = _levels_histogram_state(win)
                first_visible_reused_compatible_predecessor = True
            if vispy_tiled:
                draw_after_complete_ms = (perf_counter() - start) * 1000.0
            if vispy_tiled or wgpu_tiled or pyqtgraph_tiled:
                physical_draw_after_complete_ms = (perf_counter() - start) * 1000.0
            display_payload_fill_after_first_payload_ms = _elapsed_between_ms(
                first_display_payload_ms,
                first_display_payload_fill_ms,
            )
            fully_visible_after_first_payload_ms = _elapsed_between_ms(
                first_display_payload_ms,
                fully_visible_ms,
            )
            fully_visible_after_first_visible_tile_ms = _elapsed_between_ms(
                first_visible_tile_ms,
                fully_visible_ms,
            )
            return {
                "first_loaded_tile_ms": first_materialized_tile_ms,
                "first_materialized_tile_ms": first_materialized_tile_ms,
                "first_presented_tile_ms": first_presented_tile_ms,
                "first_display_committed_ms": first_display_committed_ms,
                "first_overlay_clear_ms": first_overlay_clear_ms,
                "first_display_payload_ms": first_display_payload_ms,
                "first_display_payload_fill_ms": first_display_payload_fill_ms,
                "first_display_payload_fill_after_first_payload_ms": display_payload_fill_after_first_payload_ms,
                "fully_visible_after_first_display_payload_ms": fully_visible_after_first_payload_ms,
                "first_visible_tile_ms": first_visible_tile_ms,
                "fully_visible_after_first_visible_tile_ms": fully_visible_after_first_visible_tile_ms,
                "first_preview_payload_ms": first_preview_payload_ms,
                "first_preview_payload_fill_ms": first_preview_payload_fill_ms,
                "first_preview_floor_fill_ms": first_preview_floor_fill_ms,
                "first_histogram_data_ms": first_histogram_data_ms,
                "first_nondefault_levels_ms": first_nondefault_levels_ms,
                "first_visible_display_levels": (
                    None
                    if first_visible_levels_state is None
                    else first_visible_levels_state.get("display_levels")
                ),
                "first_visible_histogram_data_bounds": (
                    None
                    if first_visible_levels_state is None
                    else first_visible_levels_state.get("histogram_data_bounds")
                ),
                "first_visible_levels_default": bool(
                    True
                    if first_visible_levels_state is None
                    else first_visible_levels_state.get("levels_look_default", True)
                ),
                "first_visible_histogram_empty": bool(
                    True
                    if first_visible_levels_state is None
                    else first_visible_levels_state.get("histogram_empty", True)
                ),
                "first_visible_level_source_rank": (
                    None
                    if first_visible_levels_state is None
                    else first_visible_levels_state.get("level_source_rank")
                ),
                "first_visible_level_source_count": (
                    None
                    if first_visible_levels_state is None
                    else first_visible_levels_state.get("level_source_count")
                ),
                "first_visible_level_evidence_quality": (
                    None
                    if first_visible_levels_state is None
                    else first_visible_levels_state.get("level_evidence_quality")
                ),
                "first_visible_level_decision": first_visible_level_decision,
                "first_visible_reused_compatible_predecessor": bool(
                    first_visible_reused_compatible_predecessor
                ),
                "preview_floor_screenshot_path": (
                    None
                    if preview_floor_screenshot_saved is None
                    else str(preview_floor_screenshot_path)
                ),
                "preview_floor_screenshot_saved": preview_floor_screenshot_saved,
                "preview_floor_screenshot_error": preview_floor_screenshot_error,
                "preview_floor_physical_rows": preview_floor_physical_rows,
                "final_display_payload_count": int(
                    final_payload_state.get("display_payload_count", 0)
                ),
                "final_preview_payload_count": int(
                    final_payload_state.get("preview_payload_count", 0)
                ),
                "final_exact_payload_count": int(final_payload_state.get("exact_payload_count", 0)),
                "preview_payload_reporting_scope": (
                    "display payloads with quality=preview; not evidence that a second transform pass ran"
                ),
                "logical_complete_ms": first_logical_complete_ms,
                "draw_after_complete_ms": draw_after_complete_ms,
                "physical_draw_after_complete_ms": physical_draw_after_complete_ms,
                "waited_for_pyqtgraph_draw_after_complete": bool(pyqtgraph_tiled),
                "pyqtgraph_draw_pending_after_complete": bool(
                    pyqtgraph_tiled and not pyqtgraph_final_drawn
                ),
                "fully_visible_ms": fully_visible_ms,
                "presentation_settled_ms": presentation_settled_ms,
                "presentation_settled": bool(level_state["settled"]),
                "required_target_settled_ms": target_settled_ms,
                "required_target_settled": bool(target_settled),
                "level_revision": int(level_state["revision"]),
                "stale_level_tiles": int(level_state["stale_tiles"]),
                "pending_level_tiles": int(level_state["pending_tiles"]),
                "active_level_value_count": int(level_state["active_level_value_count"]),
                "active_presented_tile_count": int(visibility_state["active_presented_tile_count"]),
                "active_planned_tile_count": int(visibility_state["active_planned_tile_count"]),
                "requested_tile_count": int(visibility_state["requested_tile_count"]),
                "requested_grid_fully_visible": bool(requested_grid_visible),
                "vispy_draw_count_start": int(draw_start),
                "vispy_draw_count_complete": _vispy_draw_count(win),
                "vispy_tile_presentation_request_count": _vispy_tile_presentation_request_count(
                    win
                ),
                "vispy_tile_presentation_draw_count": _vispy_tile_presentation_draw_count(win),
                "waited_for_vispy_draw_after_complete": bool(vispy_tiled),
                "waited_for_wgpu_draw_after_complete": bool(wgpu_tiled),
                **_histogram_continuity_metrics(histogram_timeline),
            }
        blockers = _post_visible_gate_blockers(
            fully_visible=fully_visible,
            requested_grid_visible=requested_grid_visible,
            physical_drawn=physical_drawn,
            presentation_ready=presentation_ready,
            target_settled=target_settled,
            work_in_flight=_montage_work_in_flight(session),
            dirty_payloads=bool(getattr(session, "dirty_payloads", None)),
        )
        if blockers:
            if post_visible_since is None:
                post_visible_since = time.monotonic()
            elif time.monotonic() - post_visible_since >= stall_grace_s:
                stalled = True
                post_visible_blockers = blockers
                break
        else:
            post_visible_since = None
        time.sleep(0.005)
    snapshot = win.collect_runtime_diagnostics()
    presentation_diagnostics = getattr(win.img_view, "presentation_diagnostics", dict)()
    session = getattr(win, "_frame_session", None)
    final_view_range = _montage_view_range(win)
    final_plan = None if session is None else getattr(session, "plan", None)
    viewport_controller = getattr(getattr(win, "img_view", None), "viewport_controller", None)
    viewport_mode = (
        None if viewport_controller is None else str(getattr(viewport_controller, "mode", None))
    )
    viewport_timer = getattr(getattr(win, "renderer", None), "_frame_viewport_update_timer", None)
    fan_in = None if session is None else getattr(session, "stage_fan_in", None)
    lifecycle_counts = {}
    lifecycle_phase_counts = {}
    active_samples = ()
    if session is not None:
        lifecycle = getattr(session, "lifecycle", None)
        if lifecycle is not None:
            lifecycle_counts = dict(getattr(lifecycle, "counters", dict)() or {})
        lifecycle_snapshot = getattr(session, "lifecycle_snapshot", lambda: None)()
        if lifecycle_snapshot is not None:
            lifecycle_phase_counts = dict(getattr(lifecycle_snapshot, "counts", {}) or {})
        samples = []
        for tile_number in tuple(sorted(getattr(session, "active_tile_requests", ()) or ()))[:4]:
            row = getattr(lifecycle, "row", lambda _tile: None)(int(tile_number))
            task_claim = None if row is None else getattr(row, "task_claim", None)
            claim = None if lifecycle is None else lifecycle.evaluation_claim_for(int(tile_number))
            samples.append(
                {
                    "tile": int(tile_number),
                    "eval_rung": None if claim is None else int(getattr(claim, "rung", -1)),
                    "eval_level": None if claim is None else int(getattr(claim, "level", -1)),
                    "task_key": None
                    if task_claim is None
                    else repr(getattr(task_claim, "task_key", None)),
                    "stage_key": None
                    if task_claim is None
                    else repr(getattr(task_claim, "stage_key", None)),
                    "has_payload": int(tile_number)
                    in getattr(session, "display_tile_payloads", {}),
                }
            )
        active_samples = tuple(samples)
    if post_visible_blockers:
        _stall_prefix = (
            "STALL GUARD: montage fully visible but completion gates "
            f"{list(post_visible_blockers)} stayed blocked with no work in "
            f"flight for {stall_grace_s:.1f}s: "
        )
    elif stalled:
        _stall_prefix = (
            f"STALL GUARD: montage frozen (no work in flight) after {stall_grace_s:.1f}s: "
        )
    else:
        _stall_prefix = "timed out waiting for montage completion: "
    raise TimeoutError(
        _stall_prefix + f"loaded={snapshot.montage.loaded_tiles} "
        f"target_unsettled={snapshot.montage.target_unsettled_tiles} "
        f"loading={snapshot.montage.loading_tiles} "
        f"active={0 if session is None else len(getattr(session, 'active_tile_requests', ()) or ())} "
        f"stage_active={0 if fan_in is None else len(getattr(fan_in, 'active_requests', ()) or ())} "
        f"stage_attached={0 if fan_in is None else len(getattr(fan_in, 'attached_requests', ()) or ())} "
        f"stage_deps={0 if fan_in is None else len(getattr(fan_in, 'tile_stage_keys', {}))} "
        f"lead_warmups={0 if fan_in is None else len(fan_in.lead_warmups)} "
        f"active_presented={final_visibility_state.get('active_presented_tile_count', 0)}/"
        f"{final_visibility_state.get('active_planned_tile_count', 0)} "
        f"fully_visible={final_visibility_state.get('fully_visible', False)} "
        f"requested={final_visibility_state.get('requested_tile_count', 0)} "
        "visible_target_unsettled="
        f"{final_visibility_state.get('visible_target_unsettled_tiles', 0)} "
        f"visible_loading={final_visibility_state.get('visible_loading_tiles', 0)} "
        f"visible_active={final_visibility_state.get('visible_active_requests', 0)} "
        f"visible_dirty={final_visibility_state.get('visible_dirty_tiles', 0)} "
        f"visible_upserts={final_visibility_state.get('visible_upserts', 0)} "
        f"visible_stage={final_visibility_state.get('visible_stage_waiters', 0)} "
        f"mode={getattr(win.img_view, 'montageDisplayMode', lambda: '')()} "
        f"display_committed={False if session is None else bool(getattr(session, 'display_committed', False))} "
        f"dirty={0 if session is None else len(getattr(session, 'dirty_payloads', ()) or ())} "
        f"upserts={0 if session is None else len(getattr(session, 'pending_payload_upserts', ()) or ())} "
        f"removals={0 if session is None else len(getattr(session, 'pending_removals', ()) or ())} "
        f"flush={False if session is None else bool(getattr(session, 'flush_pending', False))} "
        f"final={False if session is None else bool(getattr(session, 'final_commit_pending', False))} "
        f"present_gate={bool(getattr(getattr(win, 'renderer', None), '_montage_presentation_gate_armed', False))} "
        f"replan_gate={bool(getattr(getattr(win, 'renderer', None), '_montage_replan_gate_armed', False))} "
        f"gate_no_progress={int(getattr(getattr(win, 'renderer', None), '_montage_gate_no_progress', 0) or 0)} "
        f"gate_backlog={getattr(getattr(win, 'renderer', None), '_montage_gate_last_backlog', None)!r} "
        f"commit_outcome={getattr(getattr(win, 'renderer', None), '_last_montage_commit_outcome', None)!r} "
        f"commit_first={bool(getattr(getattr(win, 'renderer', None), '_last_montage_commit_first_display', False))} "
        f"commit_delta={int(getattr(getattr(win, 'renderer', None), '_last_montage_commit_delta_upserts', 0) or 0)} "
        f"report_presented={int(getattr(getattr(win, 'renderer', None), '_last_montage_report_presented', 0) or 0)} "
        f"report_committed={int(getattr(getattr(win, 'renderer', None), '_last_montage_report_committed', 0) or 0)} "
        f"report_stale={bool(getattr(getattr(win, 'renderer', None), '_last_montage_report_stale', False))} "
        f"report_ack={bool(getattr(getattr(win, 'renderer', None), '_last_montage_report_acknowledges', False))} "
        f"report_key={getattr(getattr(win, 'renderer', None), '_last_montage_report_delta_key', None)!r} "
        f"report_generation={getattr(getattr(win, 'renderer', None), '_last_montage_report_generation', None)!r} "
        f"ack_new={int(getattr(getattr(win, 'renderer', None), '_last_montage_ack_new_presented', 0) or 0)} "
        f"ack_lost={int(getattr(getattr(win, 'renderer', None), '_last_montage_ack_lost_presented', 0) or 0)} "
        f"gpu_bytes={int(getattr(getattr(snapshot, 'montage_timing', None), 'tile_layer_estimated_gpu_bytes', 0) or 0)} "
        f"gpu_budget={int(getattr(getattr(snapshot, 'montage_timing', None), 'tile_layer_budget_bytes', 0) or 0)} "
        f"gpu_warning={str(getattr(getattr(snapshot, 'montage_timing', None), 'tile_layer_capacity_warning', '') or '')!r} "
        f"gpu_pages={presentation_diagnostics.get('tile_atlas_pages', ())!r} "
        f"overlays={_montage_overlay_count(win)} vispy_draws={_vispy_draw_count(win)} "
        f"tile_draw={_vispy_tile_presentation_draw_count(win)}/{_vispy_tile_presentation_request_count(win)} "
        f"level_pending={final_level_state.get('pending', False)} "
        f"level_stale={final_level_state.get('stale_tiles', 0)} "
        f"level_values={final_level_state.get('active_level_value_count', 0)} "
        f"evidence_pending={0 if session is None else len(getattr(session, 'pending_level_tiles', ()) or ())} "
        f"evidence_refined={0 if session is None else len(getattr(session, 'pending_refined_level_tiles', ()) or ())} "
        f"evidence_scan={0 if session is None else int(getattr(session, 'level_scan_remaining_tiles', 0) or 0)} "
        f"evidence_inflight={False if session is None else bool(getattr(session, 'level_evidence_inflight', False))} "
        f"histogram_aggregate={False if session is None else bool(getattr(session, 'histogram_aggregate_inflight', False))} "
        f"atomic_warm_pending={0 if session is None else len((getattr(session, '_atomic_warm_job', None) or {}).get('pending', ()))} "
        f" pipeline_steps={(() if session is None else tuple(getattr(getattr(session, 'pipeline', None), 'last_plan_steps', ()) or ()))}"
        f" view_range={final_view_range!r}"
        f" plan_shape={None if final_plan is None else tuple(getattr(final_plan, 'display_shape', ()) or ())!r}"
        f" plan_columns={None if final_plan is None else int(getattr(final_plan, 'columns', 0) or 0)}"
        f" viewport_mode={viewport_mode!r}"
        f" viewport_pending={bool(getattr(win, '_montage_viewport_update_pending', False))}"
        f" viewport_timer_active={False if viewport_timer is None else bool(viewport_timer.isActive())}"
        f" lifecycle={lifecycle_counts}"
        f" lifecycle={lifecycle_phase_counts}"
        f" identity_rows={(() if session is None else session.diagnostic_tile_identity_rows(limit=12))!r}"
        f" active_samples={active_samples}"
    )


def _montage_level_presentation_state(win) -> dict[str, object]:
    """Return semantic completion for the current level generation."""

    session = getattr(win, "_frame_session", None)
    if session is None:
        return {
            "settled": True,
            "pending": False,
            "revision": 0,
            "target_levels": None,
            "stale_tiles": 0,
            "pending_tiles": 0,
            "active_level_value_count": 0,
            "active_tile_count": 0,
            "active_presented_tile_count": 0,
        }
    snapshot_getter = getattr(session, "level_presentation_snapshot", None)
    if callable(snapshot_getter):
        snapshot = snapshot_getter()
        return {
            "settled": bool(snapshot.settled),
            "pending": not bool(snapshot.settled),
            "revision": int(snapshot.revision),
            "target_levels": None
            if snapshot.target_levels is None
            else list(snapshot.target_levels),
            "stale_tiles": int(snapshot.stale_count),
            "pending_tiles": int(snapshot.pending_count),
            "active_level_value_count": len(session.level_generation.value_counts()),
            "active_tile_count": int(snapshot.active_tile_count),
            "active_presented_tile_count": int(snapshot.active_presented_tile_count),
        }
    pending = bool(session.has_pending_level_update())
    stale = int(session.level_presentation_snapshot().stale_count)
    counts = session.level_generation.value_counts()
    return {
        "settled": not pending and stale <= 0,
        "pending": pending,
        "revision": int(getattr(session, "level_revision", 0) or 0),
        "target_levels": None,
        "stale_tiles": max(0, stale),
        "pending_tiles": max(0, stale) if pending else 0,
        "active_level_value_count": len(counts),
        "active_tile_count": len(tuple(getattr(session, "visible_tiles", ()) or ())),
        "active_presented_tile_count": len(tuple(session.lifecycle.presented_tiles)),
    }


def _wgpu_frame_cadence(win) -> dict[str, object]:
    """Per-phase frame-cadence readout, or nothing on paths that lack one.

    Sampled per phase rather than once per run: the recorder holds a rolling
    window, so a run-level sample would describe whichever phase happened to
    end last instead of the interaction being measured.
    """

    view = getattr(win, "img_view", None)
    diagnostics = getattr(view, "presentation_diagnostics", None)
    if not callable(diagnostics):
        return {}
    try:
        sampled = diagnostics() or {}
    except Exception:  # diagnostics must never fail a profiling phase
        return {}
    return {key: value for key, value in sampled.items() if key.startswith("wgpu_screen_")}


def _phase_record(
    win,
    *,
    phase: str,
    elapsed_ms: float,
    event_loop_p95_gap_ms: float | None,
    event_loop_p99_gap_ms: float | None,
    event_loop_max_gap_ms: float,
    phase_ui_work_start: tuple[object, ...] = (),
    phase_ui_work_epoch: int | None = None,
) -> dict[str, object]:
    snapshot = win.collect_runtime_diagnostics()
    timing = snapshot.montage_timing
    montage = snapshot.montage
    resource = snapshot.resource_governor
    recent_callbacks = () if resource is None else tuple(resource.recent_over_warning_callbacks)
    recent_ui_work = () if resource is None else tuple(resource.recent_ui_work_observations)
    phase_recent_ui_work, phase_recent_ui_work_truncated = _ui_work_observation_delta(
        recent_ui_work,
        phase_ui_work_start,
    )
    epoch_evidence = getattr(
        getattr(win, "resource_governor", None), "ui_observation_epoch_evidence", None
    )
    epoch_count, epoch_max_ms, epoch_complete = (
        epoch_evidence(phase_ui_work_epoch)
        if phase_ui_work_epoch is not None and callable(epoch_evidence)
        else (0, 0.0, False)
    )
    feedback_channels = () if resource is None else tuple(resource.feedback_channels)
    lane_decisions = () if resource is None else tuple(resource.lane_decisions)
    vispy = _vispy_presentation_diagnostics(win)
    level_state = _montage_level_presentation_state(win)
    return {
        **_window_geometry_state(win),
        "phase": phase,
        "elapsed_ms": float(elapsed_ms),
        "event_loop_p95_gap_ms": _optional_float(event_loop_p95_gap_ms),
        "event_loop_p99_gap_ms": _optional_float(event_loop_p99_gap_ms),
        "event_loop_max_gap_ms": float(event_loop_max_gap_ms),
        "complete": True,
        "image_backend_actual": str(snapshot.image_rendering_backend_actual),
        "montage_display_mode": str(montage.display_mode),
        "montage_backend_chosen": str(montage.backend_chosen),
        "montage_quality_desired_factor": int(montage.tile_lod_desired_factor),
        "montage_quality_applied_factor": int(montage.tile_lod_applied_factor),
        "montage_quality_desired_factor_xy": tuple(
            int(value) for value in montage.tile_lod_desired_factor_xy
        ),
        "montage_quality_applied_factor_xy": tuple(
            int(value) for value in montage.tile_lod_applied_factor_xy
        ),
        "montage_quality_source_texels_per_pixel_xy": tuple(
            float(value) for value in montage.tile_lod_source_texels_per_pixel_xy
        ),
        "montage_quality_policy": str(montage.tile_lod_policy),
        "montage_quality_reason": str(montage.tile_lod_reason),
        "montage_quality_applied_level": int(getattr(montage, "tile_lod_applied_level", 0) or 0),
        "montage_quality_resident_tile_levels": tuple(
            (int(level), int(count))
            for level, count in tuple(getattr(montage, "tile_lod_resident_tile_levels", ()) or ())
        ),
        "montage_quality_pyramid_bytes": int(getattr(montage, "tile_lod_pyramid_bytes", 0) or 0),
        "montage_quality_pyramid_entries": int(
            getattr(montage, "tile_lod_pyramid_entries", 0) or 0
        ),
        "montage_quality_pyramid_hits": int(getattr(montage, "tile_lod_pyramid_hits", 0) or 0),
        "montage_quality_pyramid_misses": int(getattr(montage, "tile_lod_pyramid_misses", 0) or 0),
        "montage_quality_pyramid_evictions": int(
            getattr(montage, "tile_lod_pyramid_evictions", 0) or 0
        ),
        "montage_quality_pending_materializations": int(
            getattr(montage, "tile_lod_pending_materializations", 0) or 0
        ),
        "montage_quality_materializations_completed": int(
            getattr(montage, "tile_lod_materializations_completed", 0) or 0
        ),
        "montage_quality_ingest_reductions": int(
            getattr(montage, "tile_lod_ingest_reductions", 0) or 0
        ),
        "montage_quality_preview_reduced_scheduled": int(
            getattr(montage, "tile_lod_preview_reduced_scheduled", 0) or 0
        ),
        "montage_quality_preview_reduced_blocked": int(
            getattr(montage, "tile_lod_preview_reduced_blocked", 0) or 0
        ),
        "montage_quality_preview_reduced_failures": int(
            getattr(montage, "tile_lod_preview_reduced_failures", 0) or 0
        ),
        "montage_quality_preview_presentations": int(
            getattr(montage, "tile_lod_preview_presentations", 0) or 0
        ),
        "montage_quality_stats_cross_level_reuses": int(
            getattr(montage, "tile_lod_stats_cross_level_reuses", 0) or 0
        ),
        "montage_quality_stats_recomputes": int(
            getattr(montage, "tile_lod_stats_recomputes", 0) or 0
        ),
        "montage_quality_cross_level_reductions": int(
            getattr(montage, "tile_lod_cross_level_reductions", 0) or 0
        ),
        "montage_quality_pipeline_reruns_avoided": int(
            getattr(montage, "tile_lod_pipeline_reruns_avoided", 0) or 0
        ),
        "montage_quality_stage_hits_serving_derivations": int(
            getattr(montage, "tile_lod_stage_hits_serving_derivations", 0) or 0
        ),
        "montage_histogram_lod_swap_recomputes": int(
            getattr(montage, "tile_histogram_lod_swap_recomputes", 0) or 0
        ),
        "montage_histogram_cross_level_reuses": int(
            getattr(montage, "tile_histogram_cross_level_reuses", 0) or 0
        ),
        "montage_loaded_tiles": int(montage.loaded_tiles),
        "montage_loading_tiles": int(montage.loading_tiles),
        "montage_target_unsettled_tiles": int(montage.target_unsettled_tiles),
        "montage_tile_compute_cache_hits": int(montage.tile_compute_cache_hits),
        "montage_tile_compute_stage_backed": int(montage.tile_compute_stage_backed),
        "montage_tile_compute_direct": int(montage.tile_compute_direct),
        "montage_tile_compute_waiting_for_stage": int(montage.tile_compute_waiting_for_stage),
        "montage_tile_compute_stage_backed_ms": float(montage.tile_compute_stage_backed_ms),
        "montage_tile_compute_direct_ms": float(montage.tile_compute_direct_ms),
        "montage_tile_compute_stage_backed_max_ms": float(montage.tile_compute_stage_backed_max_ms),
        "montage_tile_compute_direct_max_ms": float(montage.tile_compute_direct_max_ms),
        "montage_lead_direct_tiles": int(montage.lead_direct_tiles),
        "montage_stage_backed_tiles_pending": int(montage.stage_backed_tiles_pending),
        "montage_retained_stage_index": montage.retained_stage_index,
        "montage_retained_stage_decision": str(montage.retained_stage_decision),
        "montage_repeated_expensive_stage_per_tile": bool(
            montage.repeated_expensive_stage_per_tile
        ),
        # Frame cadence, wgpu screen path only (absent on every other path).
        # Passed through wholesale rather than cherry-picked: this is the
        # readout the frame-pacing dossier's phase 2 exists to capture, and a
        # baseline that silently dropped a key would have to be re-run.
        **_wgpu_frame_cadence(win),
        "presentation_revision": int(level_state["revision"]),
        "presentation_target_levels": level_state["target_levels"],
        "presentation_stale_count": int(level_state["stale_tiles"]),
        "presentation_pending_count": int(level_state["pending_tiles"]),
        "presentation_settled": bool(level_state["settled"]),
        "presentation_active_tile_count": int(level_state["active_tile_count"]),
        "presentation_active_presented_tile_count": int(level_state["active_presented_tile_count"]),
        "last_render_sync_ms": _optional_float(snapshot.render_timing.last_render_sync_ms),
        "last_render_preamble_ms": _optional_float(
            getattr(win.renderer, "_last_render_preamble_ms", None)
        ),
        "last_control_sync_ms": _optional_float(snapshot.render_timing.last_control_sync_ms),
        "last_frame_update_ms": _optional_float(
            getattr(win.renderer, "_last_frame_update_ms", None)
        ),
        "last_side_panel_sync_ms": _optional_float(
            getattr(win.renderer, "_last_side_panel_sync_ms", None)
        ),
        "last_display_commit_ms": _optional_float(snapshot.render_timing.last_display_commit_ms),
        "last_viewport_plan_ms": _optional_float(timing.last_viewport_plan_ms),
        "last_cache_resolve_ms": _optional_float(timing.last_cache_resolve_ms),
        "last_stage_plan_ms": _optional_float(timing.last_stage_plan_ms),
        "last_session_setup_ms": _optional_float(timing.last_session_setup_ms),
        "last_retarget_source_ids_ms": _optional_float(
            getattr(win.renderer, "_last_montage_retarget_source_ids_ms", None)
        ),
        "last_retarget_frame_plan_ms": _optional_float(
            getattr(win.renderer, "_last_montage_retarget_frame_plan_ms", None)
        ),
        "last_retarget_release_ms": _optional_float(
            getattr(win.renderer, "_last_montage_retarget_release_ms", None)
        ),
        "last_retarget_model_ms": _optional_float(
            getattr(win.renderer, "_last_montage_retarget_model_ms", None)
        ),
        "last_retarget_hot_stage_ms": _optional_float(
            getattr(win.renderer, "_last_montage_retarget_hot_stage_ms", None)
        ),
        "last_retarget_hot_stage_cpu_ms": _optional_float(
            getattr(win.renderer, "_last_montage_retarget_hot_stage_cpu_ms", None)
        ),
        "last_retarget_hot_call_cpu_ms": _optional_float(
            getattr(win.renderer, "_last_montage_retarget_hot_call_cpu_ms", None)
        ),
        "last_retarget_hot_predicate_cpu_ms": _optional_float(
            getattr(win.renderer, "_last_montage_retarget_hot_predicate_cpu_ms", None)
        ),
        "last_retarget_hot_deferred_cpu_ms": _optional_float(
            getattr(win.renderer, "_last_montage_retarget_hot_deferred_cpu_ms", None)
        ),
        "last_retarget_attach_ms": _optional_float(
            getattr(win.renderer, "_last_montage_retarget_attach_ms", None)
        ),
        "last_retarget_level_setup_ms": _optional_float(
            getattr(win.renderer, "_last_montage_retarget_level_setup_ms", None)
        ),
        "hot_stage_match_cache_hits": int(
            getattr(win.renderer, "_hot_stage_match_cache_hits", 0) or 0
        ),
        "hot_stage_match_cache_misses": int(
            getattr(win.renderer, "_hot_stage_match_cache_misses", 0) or 0
        ),
        "last_hot_stage_resident_cpu_ms": _optional_float(
            getattr(win.renderer, "_last_hot_stage_resident_cpu_ms", None)
        ),
        "last_hot_stage_filter_cpu_ms": _optional_float(
            getattr(win.renderer, "_last_hot_stage_filter_cpu_ms", None)
        ),
        "last_hot_stage_sort_cpu_ms": _optional_float(
            getattr(win.renderer, "_last_hot_stage_sort_cpu_ms", None)
        ),
        "last_hot_stage_total_cpu_ms": _optional_float(
            getattr(win.renderer, "_last_hot_stage_total_cpu_ms", None)
        ),
        "last_initial_commit_ms": _optional_float(timing.last_initial_commit_ms),
        "last_tile_commit_ms": _optional_float(timing.last_tile_commit_ms),
        "last_tile_prepare_apply_ms": _optional_float(timing.last_tile_prepare_apply_ms),
        "last_tile_layer_apply_ms": _optional_float(timing.last_tile_layer_apply_ms),
        "last_tile_acknowledge_ms": _optional_float(timing.last_tile_acknowledge_ms),
        "last_tile_retained_store_ms": _optional_float(timing.last_tile_retained_store_ms),
        "last_tile_state_publish_ms": _optional_float(timing.last_tile_state_publish_ms),
        "last_tile_geometry_sync_ms": _optional_float(timing.last_tile_geometry_sync_ms),
        "last_tile_identity_check_ms": _optional_float(timing.last_tile_identity_check_ms),
        "last_tile_followup_ms": _optional_float(timing.last_tile_followup_ms),
        "last_histogram_recompute_ms": _optional_float(timing.last_histogram_recompute_ms),
        "last_level_sync_ms": _optional_float(timing.last_level_sync_ms),
        "last_tile_layer_upload_ms": _optional_float(timing.last_tile_layer_upload_ms),
        "last_tile_layer_rgb_window_ms": _optional_float(timing.last_tile_layer_rgb_window_ms),
        "last_overlay_update_ms": _optional_float(timing.last_overlay_update_ms),
        "tile_layer_visible_items": int(timing.tile_layer_visible_items),
        "tile_layer_items_updated": int(timing.tile_layer_items_updated),
        "tile_layer_items_skipped": int(timing.tile_layer_items_skipped),
        "tile_layer_rgb_window_tiles": int(timing.tile_layer_rgb_window_tiles),
        "tile_layer_texture_uploads": int(timing.tile_layer_texture_uploads),
        "tile_layer_texture_upload_bytes": int(timing.tile_layer_texture_upload_bytes),
        "tile_layer_texture_prepare_ms": _optional_float(timing.tile_layer_texture_prepare_ms),
        "tile_layer_texture_submit_ms": _optional_float(timing.tile_layer_texture_submit_ms),
        "tile_layer_vertex_uploads": int(timing.tile_layer_vertex_uploads),
        "tile_layer_level_updates": int(timing.tile_layer_level_updates),
        "tile_layer_level_update_pending_items": int(
            getattr(timing, "tile_layer_level_update_pending_items", 0)
        ),
        "tile_layer_shader_uniform_updates": int(timing.tile_layer_shader_uniform_updates),
        "tile_layer_estimated_gpu_bytes": int(timing.tile_layer_estimated_gpu_bytes),
        "tile_layer_page_count": int(timing.tile_layer_page_count),
        "tile_layer_active_pages": int(timing.tile_layer_active_pages),
        "persistent_tile_layer_fast_drain_last_enabled": bool(
            getattr(win, "_persistent_tile_layer_fast_drain_last_enabled", False)
        ),
        "persistent_tile_layer_fast_drain_enabled_count": int(
            getattr(win, "_persistent_tile_layer_fast_drain_enabled_count", 0) or 0
        ),
        "montage_overlay_count": _montage_overlay_count(win),
        "vispy_draw_count": int(vispy.get("draw_count", 0)),
        "vispy_last_draw_ms": float(vispy.get("last_draw_ms", 0.0) or 0.0),
        "vispy_max_draw_ms": float(vispy.get("max_draw_ms", 0.0) or 0.0),
        "vispy_tile_presentation_request_count": int(
            vispy.get("tile_presentation_request_count", 0)
        ),
        "vispy_tile_presentation_draw_count": int(vispy.get("tile_presentation_draw_count", 0)),
        "vispy_tile_presentation_draw_pending": bool(
            vispy.get("tile_presentation_draw_pending", False)
        ),
        "vispy_presented_tile_count": int(vispy.get("presented_tile_count", 0)),
        "vispy_presented_tiles": list(vispy.get("presented_tiles", ()) or ()),
        "wgpu_plane_lookup_candidates_total": int(
            vispy.get("wgpu_plane_lookup_candidates_total", 0) or 0
        ),
        "wgpu_uploads_total": int(vispy.get("wgpu_uploads_total", 0) or 0),
        "wgpu_last_report_uploads": int(vispy.get("wgpu_last_report_uploads", 0) or 0),
        "wgpu_page_pools": list(vispy.get("wgpu_page_pools", ()) or ()),
        "wgpu_atomic_warm_pinned_pages": int(vispy.get("wgpu_atomic_warm_pinned_pages", 0) or 0),
        "wgpu_last_pool_exhaustion": str(vispy.get("wgpu_last_pool_exhaustion", "") or ""),
        "wgpu_compressed_uploads_total": int(vispy.get("wgpu_compressed_uploads_total", 0) or 0),
        "wgpu_compressed_fallbacks_total": int(
            vispy.get("wgpu_compressed_fallbacks_total", 0) or 0
        ),
        "wgpu_active_resident_bytes": int(vispy.get("wgpu_active_resident_bytes", 0) or 0),
        "wgpu_allocated_pool_bytes": int(vispy.get("wgpu_allocated_pool_bytes", 0) or 0),
        "wgpu_pool_grows_total": int(vispy.get("wgpu_pool_grows_total", 0) or 0),
        "wgpu_pool_growth_copy_bytes_total": int(
            vispy.get("wgpu_pool_growth_copy_bytes_total", 0) or 0
        ),
        "wgpu_codec_family": str(vispy.get("wgpu_codec_family", "none") or "none"),
        "wgpu_codec_min_psnr_db": float(vispy.get("wgpu_codec_min_psnr_db", 0.0) or 0.0),
        "wgpu_adapter": str(vispy.get("wgpu_adapter", "") or ""),
        "wgpu_adapter_type": str(vispy.get("wgpu_adapter_type", "") or ""),
        "wgpu_power_preference": str(vispy.get("wgpu_power_preference", "") or ""),
        "vispy_tile_visual_visible_pages": int(vispy.get("tile_visual_visible_pages", 0)),
        "vispy_tile_visual_min_order": vispy.get("tile_visual_min_order"),
        "vispy_overlay_visual_visible_items": int(vispy.get("overlay_visual_visible_items", 0)),
        "vispy_overlay_visual_max_order": vispy.get("overlay_visual_max_order"),
        "vispy_overlays_above_tiles": bool(vispy.get("overlays_above_tiles", False)),
        "resource_feedback_channels": [asdict(channel) for channel in feedback_channels],
        "resource_lane_decisions": [
            {
                **asdict(decision),
                "lane": str(
                    getattr(getattr(decision, "lane", ""), "value", getattr(decision, "lane", ""))
                ),
            }
            for decision in lane_decisions
        ],
        "recent_ui_work_observations": [asdict(observation) for observation in recent_ui_work],
        "phase_recent_ui_work_observations": [
            asdict(observation) for observation in phase_recent_ui_work
        ],
        "phase_recent_ui_work_observations_truncated": bool(phase_recent_ui_work_truncated),
        "phase_ui_work_observation_count": int(epoch_count),
        "phase_ui_work_observation_max_ms": float(epoch_max_ms),
        "phase_ui_work_observation_evidence_complete": bool(epoch_complete),
        "recent_over_warning_callbacks": [asdict(callback) for callback in recent_callbacks],
    }


def _attach_phase_screenshot(
    record: dict[str, object],
    win,
    *,
    phase: str,
    backend: str,
    screenshot_dir: Path | None,
) -> None:
    if screenshot_dir is None:
        return
    path = screenshot_dir / f"{backend}-{phase}.png"
    try:
        ok = _save_view_screenshot(win, path)
    except Exception as exc:
        record["screenshot_error"] = repr(exc)
        return
    record["screenshot_path"] = str(path)
    record["screenshot_saved"] = ok
    record["screenshot_capture_kind"] = str(
        getattr(win, "_arrayscope_last_screenshot_capture_kind", "unknown")
    )
    record["screenshot_capture_error"] = str(
        getattr(win, "_arrayscope_last_screenshot_capture_error", "") or ""
    )


def _save_view_screenshot(win, path: Path, *, full_window: bool = True) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    win._arrayscope_last_screenshot_capture_error = ""
    present_method = getattr(win.img_view, "wgpuPresentMethod", lambda: "")()
    if str(present_method) == "screen" and full_window and is_headless_display():
        desktop_path = path.with_name(f".{path.stem}-weston{path.suffix}")
        try:
            capture_output(desktop_path)
            from pyqtgraph.Qt import QtGui

            desktop = QtGui.QImage(str(desktop_path))
            accepted_sizes = {win.size(), win.frameGeometry().size()}
            if desktop.isNull():
                raise RuntimeError("managed Weston returned no image")
            if desktop.size() not in accepted_sizes:
                # The exact-window compositor is sized to the session's window,
                # with no panel and no decoration, so the sole output IS the
                # window.  A mismatch means that identity broke — never save a
                # desktop-sized image as window evidence.
                expected = ", ".join(f"{size.width()}x{size.height()}" for size in accepted_sizes)
                raise RuntimeError(
                    "managed Weston must return exactly the profiled window; "
                    f"got {desktop.width()}x{desktop.height()}, expected one of {expected}"
                )
            win._arrayscope_last_screenshot_capture_kind = "managed-weston-window"
            return bool(desktop.save(str(path)))
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            win._arrayscope_last_screenshot_capture_kind = "managed-weston-failed"
            win._arrayscope_last_screenshot_capture_error = repr(exc)
            return False
        finally:
            desktop_path.unlink(missing_ok=True)

    # A screen-path fallback is an offscreen replay of the committed WGPU
    # command state.  It is useful pixel evidence, but it is not a compositor
    # or full-window screenshot and must be labelled as such in every record.
    grab_physical = getattr(win.img_view, "grabPresentedFramebuffer", None)
    if callable(grab_physical):
        frame = grab_physical()
        if frame is not None:
            from pyqtgraph.Qt import QtGui

            frame = np.ascontiguousarray(frame)
            height, width = frame.shape[:2]
            image = QtGui.QImage(
                frame.data,
                width,
                height,
                width * 4,
                QtGui.QImage.Format.Format_RGBA8888,
            )
            win._arrayscope_last_screenshot_capture_kind = "wgpu-offscreen-replay"
            return bool(image.save(str(path)))
    pixmap = win.grab()
    win._arrayscope_last_screenshot_capture_kind = "qt-window-grab"
    return bool(pixmap.save(str(path)))


def _preview_floor_physical_rows(win) -> list[dict[str, object]]:
    """Compact physical page state captured with the transient pixel proof."""

    physical_rows = getattr(getattr(win, "img_view", None), "tileTruthPhysicalRows", None)
    if not callable(physical_rows):
        return []
    keep = (
        "physical_page",
        "physical_slot",
        "physical_texture_kind",
        "physical_storage_mode",
        "physical_texture_dtype",
        "physical_texture_shape",
        "physical_mapping_mode",
        "physical_component_mode",
        "physical_levels",
        "physical_shader_mapping_key",
    )
    return [
        {
            "tile": int(tile),
            **{key: row.get(key) for key in keep},
        }
        for tile, row in sorted(dict(physical_rows() or {}).items())
    ]


def _wait_for_tile_presentation_draw(
    win,
    app,
    QtCore,
    *,
    timeout_s: float = 10.0,
    target_s: float = min(0.5, INTERACTION_SETTLE_HARD_LIMIT_S),
) -> None:
    """Wait until the screenshot milestone has physical draw evidence.

    Correctness is EVENTUAL settlement — a run that reaches it late is slow,
    not wrong. The strict ``target_s`` bound is recorded as a perf
    observation instead of aborting the whole workflow: a 0.5 s hard abort
    at 35/36 draws killed runs whose pipeline was demonstrably alive (202
    tiles acknowledged after the aborted step, 2026-07-17).

    GPU backends expose cumulative request/draw counts and must settle the
    requested generation.  QGraphics paints an actively filling scene and can
    re-arm its single pending bit in the same dispatcher turn that clears it;
    for that backend, one ``presentationDrawn`` edge after this milestone is
    the physical proof the following grab needs.  Requiring global idleness
    there can starve forever even while every paint is succeeding.
    """

    view = getattr(win, "img_view", None)
    paints_qgraphics_fn = getattr(view, "_paints_qgraphics_scene", None)
    paints_qgraphics = bool(callable(paints_qgraphics_fn) and paints_qgraphics_fn())
    presentation_drawn = getattr(view, "presentationDrawn", None)
    draw_edges = 0

    def record_draw_edge() -> None:
        nonlocal draw_edges
        draw_edges += 1

    observing_draw_edges = bool(paints_qgraphics and presentation_drawn is not None)
    if observing_draw_edges:
        presentation_drawn.connect(record_draw_edge)

    start = time.monotonic()
    deadline = start + max(float(timeout_s), float(target_s))
    try:
        while time.monotonic() < deadline:
            draw_pending_fn = getattr(view, "presentationDrawPending", None)
            draw_pending = bool(callable(draw_pending_fn) and draw_pending_fn())
            if (
                _vispy_tile_presentation_draw_count(win)
                >= _vispy_tile_presentation_request_count(win)
                and not draw_pending
            ):
                _process_events(app, QtCore, count=2)
                elapsed = time.monotonic() - start
                if elapsed > float(target_s):
                    print(
                        f"[perf] tile presentation draw settled in {elapsed:.3f}s "
                        f"(target {float(target_s):.3f}s)"
                    )
                return
            _process_events(app, QtCore, count=2)
            if draw_edges:
                return
            time.sleep(0.005)
    finally:
        if observing_draw_edges:
            with contextlib.suppress(RuntimeError, TypeError):
                presentation_drawn.disconnect(record_draw_edge)
    requested = _vispy_tile_presentation_request_count(win)
    drawn = _vispy_tile_presentation_draw_count(win)
    draw_pending_fn = getattr(view, "presentationDrawPending", None)
    draw_pending = bool(callable(draw_pending_fn) and draw_pending_fn())
    raise TimeoutError(
        "tile presentation draw did not settle within "
        f"{max(float(timeout_s), float(target_s)):.3f}s: requested={requested} "
        f"drawn={drawn} draw_pending={draw_pending}"
    )


def _wait_for_coverage_pass_close(
    win,
    app,
    QtCore,
    *,
    timeout_s: float = 10.0,
    target_s: float = min(0.5, INTERACTION_SETTLE_HARD_LIMIT_S),
) -> None:
    """Keep one paced gesture alive until its scheduling coverage closes."""

    start = time.monotonic()
    deadline = start + max(float(timeout_s), float(target_s))
    while time.monotonic() < deadline:
        session = getattr(win, "_frame_session", None)
        policy = None if session is None else getattr(session, "scheduling_policy", None)
        verdict = None if policy is None else getattr(policy, "verdict", None)
        if verdict is None or not bool(getattr(verdict, "coverage_open", False)):
            _process_events(app, QtCore, count=2)
            elapsed = time.monotonic() - start
            if elapsed > float(target_s):
                print(
                    f"[perf] coverage pass closed in {elapsed:.3f}s (target {float(target_s):.3f}s)"
                )
            return
        _process_events(app, QtCore, count=2)
        time.sleep(0.005)
    session = getattr(win, "_frame_session", None)
    policy = None if session is None else getattr(session, "scheduling_policy", None)
    verdict = None if policy is None else getattr(policy, "verdict", None)
    raise TimeoutError(
        "coverage pass did not close within "
        f"{max(float(timeout_s), float(target_s)):.3f}s: "
        f"coverage_open={bool(getattr(verdict, 'coverage_open', False))}"
    )


def _wait_for_physical_presentation_quiet(
    win,
    app,
    QtCore,
    *,
    timeout_s: float = min(3.0, INTERACTION_SETTLE_HARD_LIMIT_S),
) -> None:
    """Drain restore-time presentation work before measured phases start.

    A backend may redraw continuously for scene/cosmetic reasons even after
    every requested presentation is acknowledged. Generic draw-count churn is
    therefore diagnostic, not settlement truth; the shared pending contract is
    the owner of whether physical presentation work is still owed.
    """

    timeout_s = bounded_interaction_settle_timeout_s(timeout_s)
    deadline = perf_counter() + max(0.1, timeout_s)
    quiet_since = perf_counter()
    while perf_counter() < deadline:
        _process_events(app, QtCore, count=1)
        pending_fn = getattr(getattr(win, "img_view", None), "presentationDrawPending", None)
        pending = bool(callable(pending_fn) and pending_fn())
        if pending:
            quiet_since = perf_counter()
            continue
        if perf_counter() - quiet_since >= 0.1:
            return
    pending_fn = getattr(getattr(win, "img_view", None), "presentationDrawPending", None)
    raise TimeoutError(
        "physical presentation did not become quiet within "
        f"{timeout_s:.3f}s: draw_count={_vispy_draw_count(win)} "
        f"draw_pending={bool(callable(pending_fn) and pending_fn())}"
    )


def _elapsed_between_ms(start_ms, end_ms) -> float | None:
    if start_ms is None or end_ms is None:
        return None
    return max(0.0, float(end_ms) - float(start_ms))


def _recent_ui_work_observations(win) -> tuple[object, ...]:
    try:
        snapshot = win.collect_runtime_diagnostics()
    except Exception:
        return ()
    resource = snapshot.resource_governor
    return () if resource is None else tuple(resource.recent_ui_work_observations)


def _ui_work_observation_delta(
    current: tuple[object, ...], start: tuple[object, ...]
) -> tuple[tuple[object, ...], bool]:
    current = tuple(current or ())
    start = tuple(start or ())
    if not start:
        return current, False
    if len(current) >= len(start) and current[: len(start)] == start:
        return current[len(start) :], False
    return current, True


def _vispy_presentation_diagnostics(win) -> dict[str, object]:
    image_view = getattr(win, "img_view", None)
    for name in ("vispyPresentationDiagnostics", "wgpuPresentationDiagnostics"):
        getter = getattr(image_view, name, None)
        if callable(getter):
            try:
                return dict(getter())
            except Exception:
                return {}
    return {}


def _wgpu_upload_total(win) -> int | None:
    diagnostics = _vispy_presentation_diagnostics(win)
    if "wgpu_uploads_total" not in diagnostics:
        return None
    return int(diagnostics.get("wgpu_uploads_total", 0) or 0)


def _vispy_draw_count(win) -> int:
    diagnostics = _vispy_presentation_diagnostics(win)
    return int(diagnostics.get("draw_count", 0) or 0)


def _vispy_tile_presentation_request_count(win) -> int:
    diagnostics = _vispy_presentation_diagnostics(win)
    return int(diagnostics.get("tile_presentation_request_count", 0) or 0)


def _vispy_tile_presentation_draw_count(win) -> int:
    diagnostics = _vispy_presentation_diagnostics(win)
    return int(diagnostics.get("tile_presentation_draw_count", 0) or 0)


def _montage_visibility_state(win, *, mode: str | None = None) -> dict[str, object]:
    session = getattr(win, "_frame_session", None)
    if mode is None:
        mode = str(getattr(win.img_view, "montageDisplayMode", lambda: "")())
    if session is None:
        return {
            "fully_visible": False,
            "active_presented_tile_count": 0,
            "active_planned_tile_count": 0,
            "active_tiles": (),
        }
    active = set(_active_planned_montage_tiles(session))
    expected = set(_expected_requested_montage_tiles(session))
    if not expected:
        expected = set(active)
    presented = {int(tile) for tile in tuple(session.lifecycle.presented_tiles)}
    vispy = _vispy_presentation_diagnostics(win)
    overlay_count = _montage_overlay_count(win)
    overlays_above_tiles = bool(vispy.get("overlays_above_tiles", False))
    overlay_nonblocking = overlay_count == 0 or (
        str(mode) in {"vispy_tile_layer", "wgpu_tile_layer"}
        and not overlays_above_tiles
        and active
        and active.issubset(presented)
    )
    backlog = _visible_backlog_state(session, active)
    active_presented = active.intersection(presented)
    fully_visible = bool(
        str(mode) in {"tile_layer", "vispy_tile_layer", "wgpu_tile_layer"}
        and getattr(session, "display_committed", False)
        and not bool(backlog["visible_has_backlog"])
        and active
        and active.issubset(presented)
        and overlay_nonblocking
    )
    return {
        "fully_visible": fully_visible,
        "active_presented_tile_count": len(active_presented),
        "active_planned_tile_count": len(active),
        "requested_tile_count": len(expected),
        "active_tiles": tuple(sorted(active)),
        **backlog,
    }


def _montage_display_payload_state(session, *, active_tiles) -> dict[str, object]:
    active = {int(tile) for tile in tuple(active_tiles or ())}
    payloads = getattr(session, "display_tile_payloads", {}) or {}
    display_payload_tiles = {
        int(tile_number)
        for tile_number, payload in dict(payloads).items()
        if payload is not None and (not active or int(tile_number) in active)
    }
    preview_payload_tiles = {
        int(tile_number)
        for tile_number in display_payload_tiles
        if str(getattr(payloads.get(tile_number), "quality", "exact")) == "preview"
    }
    exact_payload_tiles = {
        int(tile_number)
        for tile_number in display_payload_tiles
        if str(getattr(payloads.get(tile_number), "quality", "exact")) == "exact"
    }
    fill_target = active if active else display_payload_tiles
    return {
        "display_payload_count": len(display_payload_tiles),
        "preview_payload_count": len(preview_payload_tiles),
        "exact_payload_count": len(exact_payload_tiles),
        "display_payload_fill": bool(fill_target) and fill_target.issubset(display_payload_tiles),
        "preview_payload_fill": bool(fill_target) and fill_target.issubset(preview_payload_tiles),
    }


def _active_planned_montage_tiles(session) -> tuple[int, ...]:
    skipped = {int(tile) for tile in tuple(getattr(session, "skipped_tiles", ()) or ())}
    visible = tuple(getattr(session, "visible_tiles", ()) or ())
    active = []
    for tile in visible:
        try:
            index = int(tile.montage_index)
        except Exception:
            continue
        if index not in skipped:
            active.append(index)
    return tuple(dict.fromkeys(active))


def _expected_requested_montage_tiles(session) -> tuple[int, ...]:
    skipped = {int(tile) for tile in tuple(getattr(session, "skipped_tiles", ()) or ())}
    indices = tuple(getattr(session, "level_expected_indices", ()) or ())
    if not indices:
        geometry = getattr(getattr(session, "plan", None), "geometry", None)
        indices = tuple(getattr(geometry, "indices", ()) or ())
    if not indices:
        return ()
    # level_expected_indices stores source indices for the montage request.  For
    # ordinary full-workflow montages the semantic tile numbers are positional.
    # When a custom subset is used, still require every requested tile slot.
    return tuple(index for index in range(len(indices)) if int(index) not in skipped)


def _vispy_canvas_visible(win) -> bool:
    native = getattr(getattr(win, "img_view", None), "_vispy_canvas_native", None)
    if native is None:
        return False
    try:
        return bool(native.isVisible())
    except Exception:
        return False


def _montage_overlay_count(win) -> int:
    getter = getattr(getattr(win, "img_view", None), "montageTileOverlayCount", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:
            return 0
    return 0


def _base_record(
    *,
    run_id: str,
    backend: str,
    data_path: Path,
    data,
    load_mode: str,
    montage_axis: int,
    indices: tuple[int, ...],
    columns: int,
    full_tile_count: int,
    max_tiles: int | None,
    profiler_type: str,
    profiler_artifact_paths: tuple[str | Path, ...],
    run_temperature: str = "mixed",
    qt_platform: str,
    grid_kind: str = "full",
    source_index_count: int | None = None,
    screenshot_timing_perturbed: bool = False,
) -> dict[str, object]:
    grid_kind = str(grid_kind)
    capped = grid_kind == "full" and max_tiles is not None and len(indices) < int(full_tile_count)
    smoke_only = bool(str(qt_platform).lower() == "offscreen" or capped)
    return {
        "run_id": run_id,
        "backend": backend,
        "data_path": str(data_path),
        "load_mode": str(load_mode),
        "profiler_type": str(profiler_type),
        "profiler_artifact_paths": [str(path) for path in tuple(profiler_artifact_paths or ())],
        "run_temperature": str(run_temperature),
        "qt_platform": str(qt_platform),
        "xdg_session_type": os.environ.get("XDG_SESSION_TYPE", ""),
        "wayland_display": os.environ.get("WAYLAND_DISPLAY", ""),
        "display": os.environ.get("DISPLAY", ""),
        "max_tiles": None if max_tiles is None else int(max_tiles),
        "tile_cap_applied": bool(capped),
        "smoke_only": bool(smoke_only),
        "screenshot_timing_perturbed": bool(screenshot_timing_perturbed),
        "pacing_evidence": bool(not smoke_only and not screenshot_timing_perturbed),
        "data_shape": tuple(int(value) for value in np.shape(data)),
        "data_dtype": str(getattr(getattr(data, "dtype", None), "str", getattr(data, "dtype", ""))),
        "montage_axis": int(montage_axis),
        "grid_kind": grid_kind,
        "grid_tile_count": len(indices),
        "source_index_count": int(
            full_tile_count if source_index_count is None else source_index_count
        ),
        "tile_count": len(indices),
        "full_tile_count": int(full_tile_count),
        "columns": int(columns),
    }


def _install_profile_session_fixture(
    QtCore,
    *,
    data_path: Path,
    data,
    session_fixture,
    settings,
    loads_session,
    metadata_for_file,
    save_session_file,
    settings_key_for_metadata,
):
    """Install a portable fixture through the production session store.

    The checked-in JSON intentionally has no machine-specific file metadata.
    Rebinding that metadata here keeps the fixture portable while exercising
    the same metadata check, QSettings indirection, parser, and restore path as
    a user session.
    """

    from arrayscope.window.file_view_session import _file_view_session_config_dir

    if session_fixture is None:
        return None
    path = Path(session_fixture)
    if not path.exists():
        raise FileNotFoundError(f"profile session fixture not found: {path}")
    try:
        template = loads_session(path.read_text(encoding="utf-8"), np.shape(data))
    except ValueError as exc:
        raise ValueError(
            f"profile session fixture {path} does not fit dataset shape "
            f"{np.shape(data)} ({exc}). The checked-in fixture pins the "
            "canonical 336x336x272 dataset's view (montage window 106:166). "
            "For other datasets pass --session-fixture '' to disable the "
            "restore, or provide a fixture saved from a compatible session."
        ) from exc
    metadata = metadata_for_file(data_path, data=data)
    session = replace(template, metadata=metadata)
    stored = save_session_file(_file_view_session_config_dir(), session)
    settings.setValue(settings_key_for_metadata(metadata), stored.name)
    settings.sync()
    return session


def _window_geometry_state(win) -> dict[str, object]:
    try:
        window_size = [int(win.width()), int(win.height())]
    except Exception:
        window_size = None
    try:
        minimum_size = [int(win.minimumWidth()), int(win.minimumHeight())]
    except Exception:
        minimum_size = None
    try:
        viewport = win.img_view.graphicsView.viewport()
        viewport_shape = [int(viewport.height()), int(viewport.width())]
    except Exception:
        viewport_shape = None
    try:
        canvas = win.img_view._vispy_canvas_native
        vispy_canvas_shape = [int(canvas.height()), int(canvas.width())]
        vispy_canvas_device_pixel_ratio = float(canvas.devicePixelRatioF())
    except Exception:
        vispy_canvas_shape = None
        vispy_canvas_device_pixel_ratio = None
    raw_target = getattr(win, "_profile_session_fixture_viewport_shape", None)
    target = None if raw_target is None else [int(raw_target[0]), int(raw_target[1])]
    shape_matches = bool(
        target is None
        or (
            viewport_shape is not None
            and abs(int(viewport_shape[0]) - int(target[0])) <= 1
            and abs(int(viewport_shape[1]) - int(target[1])) <= 1
        )
    )
    raw_window_target = getattr(win, "_profile_session_fixture_window_size", None)
    window_target = (
        None
        if raw_window_target is None
        else [int(raw_window_target[0]), int(raw_window_target[1])]
    )
    # Reported, deliberately NOT gated.  A window size is viewport plus
    # chrome, so it is a consequence of the restore rather than an input to
    # it: when the menu/tool/status bars grew 8 px after this fixture was
    # captured, the correctly-restored 739-row viewport started needing a
    # 948-tall window and an exact-match gate rejected every profile run --
    # on a live Wayland session as readily as headless.  The viewport shape
    # and axis orientation below are what decide aspect, montage layout, and
    # LOD, and a restore that genuinely did not happen fails those.
    window_size_delta = (
        None
        if window_target is None or window_size is None
        else [int(window_size[0]) - window_target[0], int(window_size[1]) - window_target[1]]
    )
    current_state = getattr(win, "view_state", None)
    current_image_axes = None if current_state is None else list(current_state.image_axes or ())
    current_axis_flipped = None if current_state is None else list(current_state.axis_flipped)
    expected_image_axes = getattr(win, "_profile_session_fixture_image_axes", None)
    expected_axis_flipped = getattr(win, "_profile_session_fixture_axis_flipped", None)
    return {
        "window_size": window_size,
        "session_window_size_target": window_target,
        "session_window_size_chrome_delta": window_size_delta,
        "window_minimum_size": minimum_size,
        "viewport_shape": viewport_shape,
        "vispy_canvas_shape": vispy_canvas_shape,
        "vispy_canvas_device_pixel_ratio": vispy_canvas_device_pixel_ratio,
        "session_viewport_shape_target": target,
        "session_viewport_shape_matches": shape_matches,
        "image_axes": current_image_axes,
        "axis_flipped": current_axis_flipped,
        "session_image_axes_target": None
        if expected_image_axes is None
        else list(expected_image_axes),
        "session_axis_flipped_target": None
        if expected_axis_flipped is None
        else list(expected_axis_flipped),
        "session_axis_orientation_matches": bool(
            expected_image_axes is None
            or (
                tuple(current_image_axes or ()) == tuple(expected_image_axes)
                and tuple(current_axis_flipped or ()) == tuple(expected_axis_flipped or ())
            )
        ),
    }


def _synthetic_profile_data(kind: str, shape: tuple[int, int, int]) -> np.ndarray:
    """Deterministic visual registration charts for live backend diagnosis."""

    height, width, depth = (int(value) for value in shape)
    if min(height, width, depth) < 4:
        raise ValueError(f"synthetic profile shape must be at least 4 on every axis, got {shape}")
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None, None]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :, None]
    zz = np.linspace(0.0, 1.0, depth, dtype=np.float32)[None, None, :]
    z_index = np.arange(depth, dtype=np.int32)[None, None, :]

    if kind == "geometry":
        values = 0.12 + 0.30 * (xx + 1.0) / 2.0 + 0.18 * (yy + 1.0) / 2.0
        center_x = -0.65 + 1.30 * zz
        center_y = 0.32 * np.sin(zz * np.float32(2.0 * np.pi))
        radius = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        values = values + np.where(radius < 0.24, 0.42, 0.0)
        values = values + np.where((radius > 0.31) & (radius < 0.34), 0.28, 0.0)
        diagonal = np.abs(yy - (0.72 * xx - 0.38 + 0.76 * zz)) < (3.0 / height)
        values = values + np.where(diagonal, 0.38, 0.0)
        border = (np.abs(xx) > 0.965) | (np.abs(yy) > 0.955)
        values = np.where(border, 0.90 - 0.35 * (z_index % 2), values)
        # Index-coded vertical and binary horizontal fiducials make a tile's
        # source position recoverable from one screenshot.
        bar_x = -0.86 + 0.22 * (z_index % 8)
        values = values + np.where((np.abs(xx - bar_x) < 0.018) & (yy > 0.46), 0.38, 0.0)
        for bit in range(6):
            bit_on = ((z_index >> bit) & 1) != 0
            band_y = -0.86 + bit * 0.11
            values = values + np.where(
                bit_on & (np.abs(yy - band_y) < 0.022) & (xx < -0.45), 0.32, 0.0
            )
        return np.ascontiguousarray(np.clip(values, 0.0, 1.0), dtype=np.float32)

    if kind == "complex-phase":
        center_x = -0.42 + 0.84 * zz
        center_y = 0.28 * np.cos(zz * np.float32(2.0 * np.pi))
        dx, dy = xx - center_x, yy - center_y
        radius = np.sqrt(dx * dx + dy * dy)
        amplitude = 0.12 + 0.38 * (xx + 1.0) / 2.0
        amplitude = amplitude + 0.55 * np.exp(-(((radius - 0.38) / 0.09) ** 2))
        amplitude = amplitude + 0.28 * ((np.abs(xx) < 0.18) & (np.abs(yy) < 0.56))
        phase = np.arctan2(dy, dx) + 5.0 * np.pi * xx + 2.0 * np.pi * zz
        y_index = np.arange(height, dtype=np.int32)[:, None, None]
        x_index = np.arange(width, dtype=np.int32)[None, :, None]
        opposed_patch = (xx < -0.48) & (yy < -0.42)
        opposed_phase = np.pi * ((x_index + y_index + z_index) & 1)
        phase = np.where(opposed_patch, opposed_phase, phase)
        zero_region = ((np.abs(xx) < 0.035) | (np.abs(yy) < 0.035)) & (radius > 0.18)
        amplitude = np.where(zero_region, 0.0, amplitude)
        values = amplitude * np.exp(1j * phase)
        return np.ascontiguousarray(values, dtype=np.complex64)

    raise ValueError("synthetic scene must be 'geometry' or 'complex-phase'")


def _parse_synthetic_shape(value: str) -> tuple[int, int, int]:
    parts = tuple(
        int(part.strip())
        for part in str(value).lower().replace("x", ",").split(",")
        if part.strip()
    )
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("synthetic shape must be HEIGHTxWIDTHxDEPTH")
    if min(parts) < 4:
        raise argparse.ArgumentTypeError("synthetic shape axes must each be at least 4")
    return parts


def _load_dataset(path: Path, *, mode: str):
    mode = str(mode)
    if mode == "app":
        from arrayscope.io.file_interpreters import load_path

        return load_path(path).data
    if mode == "native":
        if _is_nifti(path):
            import nibabel as nib

            return np.asanyarray(nib.load(str(path)).dataobj)
        if path.suffix.lower() == ".npy":
            return np.load(path)
    raise ValueError(f"unsupported load mode {mode!r}; expected 'app' or 'native'")


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".nii", ".nii.gz"))


def _centered_indices(full_count: int, max_tiles: int | None) -> tuple[int, ...]:
    full_count = max(0, int(full_count))
    if full_count <= 0:
        return ()
    if max_tiles is None or max_tiles <= 0 or max_tiles >= full_count:
        return tuple(range(full_count))
    max_tiles = max(1, min(full_count, int(max_tiles)))
    start = (full_count - max_tiles) // 2
    return tuple(range(start, start + max_tiles))


def _replace_settings(settings, *, backend: str, image_choice):
    from dataclasses import replace

    from arrayscope.app.settings_state import MontageQualityPolicyChoice

    backend_choice = {
        "pyqtgraph": image_choice.PYQTGRAPH,
        "vispy": image_choice.VISPY,
        "wgpu": image_choice.WGPU,
    }[_normalize_backend(backend)]
    return replace(
        settings,
        image_rendering_backend=backend_choice,
        montage_quality_policy=MontageQualityPolicyChoice.RESIDENT,
    )


def _append_record(
    records: list[dict[str, object]], jsonl: str | Path | None, record: dict[str, object]
) -> None:
    record.update(_r8_certification(record))
    records.append(record)
    line = json.dumps(record, sort_keys=True)
    if jsonl is None:
        return
    path = Path(jsonl)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _r8_certification(record: dict[str, object]) -> dict[str, object]:
    """Evaluate one workflow phase against the immutable R8 exit gates.

    Correctness gates apply to every real phase, including profiler-slowed
    runs. Timing gates apply only to plain, uncapped, onscreen evidence.
    Returning named failures keeps a JSONL useful without reverse-engineering
    a boolean from dozens of counters.
    """

    phase = str(record.get("phase", ""))
    if phase in {"", "load", "load_data"}:
        return {
            "r8_gate_applicable": False,
            "r8_gate_passed": None,
            "r8_gate_failures": [],
            "r8_gate_check_count": 0,
            "r8_performance_evidence": False,
        }

    failures: list[dict[str, object]] = []
    check_count = 0

    def require(
        name: str, condition: bool, *, evidence, target: str, category: str = "correctness"
    ) -> None:
        nonlocal check_count
        check_count += 1
        if bool(condition):
            return
        failures.append(
            {
                "gate": str(name),
                "category": str(category),
                "evidence": evidence,
                "target": str(target),
            }
        )

    requested = int(record.get("requested_tile_count", 0) or 0)
    planned = int(record.get("active_planned_tile_count", 0) or 0)
    presented = int(record.get("active_presented_tile_count", 0) or 0)
    require(
        "phase_complete",
        bool(record.get("complete", False)),
        evidence=record.get("complete"),
        target="true",
    )
    require(
        "requested_grid_fully_visible",
        bool(record.get("requested_grid_fully_visible", False)) and requested > 0,
        evidence={"requested": requested, "planned": planned, "presented": presented},
        target="requested == planned == presented > 0",
    )
    require(
        "all_requested_tiles_presented",
        requested > 0 and planned == requested and presented == requested,
        evidence={"requested": requested, "planned": planned, "presented": presented},
        target="requested == planned == presented",
    )
    if phase in {"display_x_axis_slice", "display_y_axis_slice"}:
        continuity_min = int(record.get("display_axis_min_physical_tile_count", 0) or 0)
        crop_scenario_names = tuple(record.get("display_axis_crop_scenario_names", ()) or ())
        require(
            "display_axis_crop_matrix_complete",
            int(record.get("display_axis_crop_scenario_count", 0) or 0)
            == len(DISPLAY_AXIS_CROP_SCENARIO_NAMES)
            and crop_scenario_names == DISPLAY_AXIS_CROP_SCENARIO_NAMES
            and int(record.get("display_axis_both_crop_scenario_count", 0) or 0) == 10
            and int(record.get("display_axis_page_boundary_scenario_count", 0) or 0) == 3,
            evidence={
                "names": crop_scenario_names,
                "both_cropped": record.get("display_axis_both_crop_scenario_count"),
                "page_boundary": record.get("display_axis_page_boundary_scenario_count"),
            },
            target=(
                "the full one-axis, both-axis, directional, odd-extent, "
                "and page-boundary crop matrix"
            ),
        )
        require(
            "display_axis_crop_matrix_settles",
            bool(record.get("display_axis_crop_scenarios_settled", False))
            and bool(record.get("display_axis_crop_scenarios_committed_current", False)),
            evidence={
                "settled": record.get("display_axis_crop_scenarios_settled"),
                "committed_current": record.get("display_axis_crop_scenarios_committed_current"),
                "maximum_ms": record.get("display_axis_crop_scenario_max_settle_ms"),
            },
            target="every crop-matrix successor settles with a current committed frame",
        )
        all_dimension_results = tuple(
            record.get("display_axis_all_dimension_scroll_results", ()) or ()
        )
        expected_axis_count = int(
            record.get("display_axis_all_dimension_scroll_expected_axis_count", 0) or 0
        )
        require(
            "display_axis_all_dimensions_scrolled_fast_and_slow",
            expected_axis_count > 0
            and int(record.get("display_axis_all_dimension_scroll_axis_count", 0) or 0)
            == expected_axis_count
            and len(all_dimension_results) == expected_axis_count
            and all(
                int(result.get("fast_input_steps", 0) or 0) == 3
                and int(result.get("slow_input_steps", 0) or 0) == 2
                for result in all_dimension_results
            ),
            evidence={
                "expected_axes": expected_axis_count,
                "axis_count": record.get("display_axis_all_dimension_scroll_axis_count"),
                "results": all_dimension_results,
            },
            target=(
                "every dimension receives a three-input coalesced fast burst "
                "and a settled forward/return slow scroll"
            ),
        )
        require(
            "display_axis_all_dimension_scrolls_settle_current",
            bool(record.get("display_axis_all_dimension_scrolls_settled", False))
            and bool(
                record.get(
                    "display_axis_all_dimension_scrolls_committed_current",
                    False,
                )
            ),
            evidence={
                "settled": record.get("display_axis_all_dimension_scrolls_settled"),
                "committed_current": record.get(
                    "display_axis_all_dimension_scrolls_committed_current"
                ),
                "results": all_dimension_results,
            },
            target="every fast and slow all-dimension checkpoint settles to the current frame",
        )
        require(
            "display_axis_all_dimension_pixels_match_cpu_reference",
            int(record.get("display_axis_physical_reference_check_count", 0) or 0)
            == expected_axis_count * 4
            and bool(record.get("display_axis_physical_reference_passed", False)),
            evidence={
                "checks": record.get("display_axis_physical_reference_check_count"),
                "failures": record.get("display_axis_physical_reference_failures"),
                "visual_checkpoints": record.get("display_axis_visual_checkpoint_count"),
            },
            target=(
                "the rendered WGPU target or PyQtGraph raster matches CPU semantic "
                "pixels after every fast, restore, slow-forward, and slow-return checkpoint"
            ),
        )
        require(
            "display_axis_roi_overlays_track_geometry",
            int(record.get("display_axis_roi_placement_check_count", 0) or 0)
            == expected_axis_count * 4
            and bool(record.get("display_axis_roi_placement_applicable", False))
            and bool(record.get("display_axis_roi_placement_passed", False)),
            evidence={
                "checks": record.get("display_axis_roi_placement_check_count"),
                "applicable": record.get("display_axis_roi_placement_applicable"),
                "failures": record.get("display_axis_roi_placement_failures"),
            },
            target=(
                "every enabled ROI paints its own colour inside the band its "
                "semantic geometry projects to, and nowhere else on the frame, "
                "after every crop-stage checkpoint"
            ),
        )
        require(
            "display_axis_physical_continuity",
            requested > 0 and continuity_min == requested,
            evidence={
                "requested": requested,
                "minimum": continuity_min,
                "samples": int(record.get("display_axis_physical_tile_sample_count", 0) or 0),
            },
            target="every post-crop scroll sample keeps every requested tile physically visible",
        )
        require(
            "display_axis_xy_swap_settles",
            bool(record.get("display_axis_xy_swap_settled", False))
            and int(record.get("display_axis_xy_swap_steps", 0) or 0) == 2,
            evidence={
                "settled": record.get("display_axis_xy_swap_settled"),
                "steps": record.get("display_axis_xy_swap_steps"),
            },
            target="cropped montage settles after X/Y swap and swap-back",
        )
        require(
            "display_axis_single_slice_scroll_settles",
            bool(record.get("display_axis_single_slice_settled", False))
            and bool(record.get("display_axis_single_slice_committed_current", False))
            and int(record.get("display_axis_single_slice_steps", 0) or 0) == 3,
            evidence={
                "settled": record.get("display_axis_single_slice_settled"),
                "committed_current": record.get("display_axis_single_slice_committed_current"),
                "steps": record.get("display_axis_single_slice_steps"),
            },
            target=(
                "the cropped single-slice current, +1, and return successors "
                "settle and publish the current committed frame"
            ),
        )
        require(
            "display_axis_montage_restore_settles",
            bool(record.get("display_axis_montage_restore_settled", False))
            and bool(record.get("display_axis_montage_restore_committed_current", False)),
            evidence={
                "settled": record.get("display_axis_montage_restore_settled"),
                "committed_current": record.get("display_axis_montage_restore_committed_current"),
            },
            target="the cropped montage restores with a current committed frame",
        )
        if str(record.get("backend", "")) == "wgpu":
            crop_uploads = record.get("display_axis_crop_wgpu_upload_delta")
            scroll_uploads = record.get("display_axis_scroll_wgpu_upload_delta")
            crop_matrix_uploads = record.get("display_axis_crop_matrix_wgpu_upload_delta")
            all_dimension_uploads = record.get(
                "display_axis_all_dimension_scroll_wgpu_upload_delta"
            )
            all_dimension_display_uploads = record.get(
                "display_axis_all_dimension_display_roles_wgpu_upload_delta"
            )
            all_dimension_montage_uploads = record.get(
                "display_axis_all_dimension_montage_role_wgpu_upload_delta"
            )
            all_dimension_slice_uploads = record.get(
                "display_axis_all_dimension_slice_roles_wgpu_upload_delta"
            )
            axis_swap_uploads = record.get("display_axis_xy_swap_wgpu_upload_delta")
            single_slice_uploads = record.get("display_axis_single_slice_wgpu_upload_delta")
            require(
                "display_axis_wgpu_source_window_truth",
                int(record.get("display_axis_wgpu_source_truth_check_count", 0) or 0) > 0
                and bool(record.get("display_axis_wgpu_source_truth_passed", False)),
                evidence={
                    "checks": record.get("display_axis_wgpu_source_truth_check_count"),
                    "failures": record.get("display_axis_wgpu_source_truth_failures"),
                },
                target=(
                    "every all-dimension checkpoint keeps the session anchor, "
                    "payload source rect, and committed global sampler origin current"
                ),
            )
            require(
                "display_axis_cold_crop_bindings_do_not_alias",
                bool(record.get("display_axis_wgpu_cold_binding_identity_unique", False)),
                evidence={
                    "multiwindow_tiles": record.get(
                        "display_axis_wgpu_cold_binding_multiwindow_tiles"
                    ),
                    "aliases": record.get("display_axis_wgpu_cold_binding_aliases"),
                },
                target=(
                    "the cold fallback gives every distinct displayed-axis "
                    "source window a distinct physical plane identity"
                ),
            )
            require(
                "display_axis_source_pages_reused",
                crop_uploads == 0
                and scroll_uploads == 0
                and crop_matrix_uploads == 0
                and all_dimension_display_uploads == 0
                and axis_swap_uploads == 0
                and single_slice_uploads == 0,
                evidence={
                    "initial_crop_uploads": crop_uploads,
                    "scroll_uploads": scroll_uploads,
                    "crop_matrix_uploads": crop_matrix_uploads,
                    "all_dimension_scroll_uploads": all_dimension_uploads,
                    "all_dimension_display_role_uploads": all_dimension_display_uploads,
                    "all_dimension_montage_role_uploads": all_dimension_montage_uploads,
                    "all_dimension_slice_role_uploads": all_dimension_slice_uploads,
                    "axis_swap_uploads": axis_swap_uploads,
                    "single_slice_uploads": single_slice_uploads,
                    "scroll_steps": int(record.get("display_axis_slice_scroll_steps", 0) or 0),
                },
                target=(
                    "the full montage prewarms reusable source pages; the "
                    "crop, every displayed-axis in-page +/-1 scroll, X/Y "
                    "swaps, and cached single-slice successors perform zero "
                    "uploads; montage-axis demand is reported separately"
                ),
                category="performance",
            )
            exhaustion = str(record.get("display_axis_wgpu_pool_exhaustion", "") or "")
            require(
                "display_axis_page_pool_has_headroom",
                not exhaustion,
                evidence={
                    "last_exhaustion": exhaustion,
                    "page_pools": record.get("wgpu_page_pools", ()),
                },
                target="no WGPU page pool exhaustion during the atomic crop handoff",
            )
    require(
        "presentation_levels_settled",
        bool(record.get("presentation_settled", False))
        and int(record.get("stale_level_tiles", 0) or 0) == 0
        and int(record.get("pending_level_tiles", 0) or 0) == 0,
        evidence={
            "settled": record.get("presentation_settled"),
            "stale": record.get("stale_level_tiles"),
            "pending": record.get("pending_level_tiles"),
        },
        target="settled with zero stale or pending level tiles",
    )
    require(
        "final_levels_semantic",
        not bool(record.get("levels_look_default", True)),
        evidence=record.get("display_levels"),
        target="finite non-default display levels",
    )
    require(
        "final_histogram_populated",
        not bool(record.get("histogram_empty", True)),
        evidence=record.get("histogram_data_bounds"),
        target="finite non-empty histogram data bounds",
    )
    shader_backend = str(record.get("backend", "")) in {"vispy", "wgpu"}
    required_evidence_quality = 1 if shader_backend else 3
    require(
        "first_visible_levels_semantic",
        not bool(record.get("first_visible_levels_default", True)),
        evidence=record.get("first_visible_display_levels"),
        target="first presented pixels already use semantic levels",
    )
    require(
        "first_visible_histogram_populated",
        not bool(record.get("first_visible_histogram_empty", True)),
        evidence=record.get("first_visible_histogram_data_bounds"),
        target="first presented pixels already publish their level sample in the histogram",
    )
    require(
        "window_level_flicker_free",
        bool(record.get("window_level_flicker_free", False)),
        evidence={
            "histogram_emptied": record.get("histogram_emptied_after_successor_visible"),
            "levels_defaulted": record.get("levels_defaulted_after_successor_visible"),
            "span_dip_ratio": record.get("level_transient_span_dip_ratio"),
            "center_excursion_fraction": record.get("level_center_excursion_fraction"),
            "source_count_regressed": record.get("level_source_count_regressed"),
            "transitions": record.get("histogram_timeline_transition_count"),
            "truncated": record.get("histogram_timeline_truncated"),
        },
        target=(
            "after successor pixels appear the histogram remains populated, levels never "
            "default, and no transient span/center/source-coverage excursion occurs"
        ),
    )
    first_quality = int(record.get("first_visible_level_evidence_quality", 0) or 0)
    compatible_predecessor = bool(record.get("first_visible_reused_compatible_predecessor", False))
    require(
        "first_visible_level_evidence_quality",
        first_quality >= required_evidence_quality or compatible_predecessor,
        evidence={"quality": first_quality, "compatible_predecessor": compatible_predecessor},
        target=(
            "rough-or-better evidence before first shader-renderer pixels"
            if required_evidence_quality == 1
            else "refined evidence before first PyQtGraph pixels"
        ),
    )
    predecessor_tiles = int(record.get("presentation_predecessor_tile_count", 0) or 0)
    minimum_retained = int(record.get("presentation_minimum_retained_tile_count", 0) or 0)
    continuity_expected = bool(
        record.get("presentation_continuity_expected", predecessor_tiles > 0)
    )
    required_transition_coverage = min(predecessor_tiles, requested) if predecessor_tiles > 0 else 0
    require(
        "presentation_continuity",
        not continuity_expected
        or (
            bool(record.get("presentation_continuity_ok", False))
            and minimum_retained >= required_transition_coverage
        ),
        evidence={
            "expected": continuity_expected,
            "blackout": record.get("presentation_blackout_observed"),
            "minimum_retained": minimum_retained,
            "required_transition_coverage": required_transition_coverage,
            "extent_changed_before_commit": record.get("presentation_extent_changed_before_commit"),
            "topology_changed": record.get("presentation_topology_changed"),
        },
        target="same-semantic predecessor/successor coverage and extent remain through successor acknowledgement",
    )
    require(
        "session_viewport_geometry_stable",
        bool(record.get("session_viewport_shape_matches", False)),
        evidence={
            "actual": record.get("viewport_shape"),
            "target": record.get("session_viewport_shape_target"),
        },
        target="restored viewport shape retained",
    )
    # No outer-window oracle: the window size a restore lands on is its
    # viewport plus whatever chrome the build currently has, so pinning it to
    # a recorded number fails on correct behaviour the moment a bar changes
    # height. Drift during a run would move the viewport, which the oracle
    # above already owns; the raw sizes and their delta stay in the record.
    require(
        "session_axis_orientation_stable",
        bool(record.get("session_axis_orientation_matches", False)),
        evidence={
            "axes": record.get("image_axes"),
            "flips": record.get("axis_flipped"),
            "target_axes": record.get("session_image_axes_target"),
            "target_flips": record.get("session_axis_flipped_target"),
        },
        target="restored XY dimensions and directions retained",
    )
    require(
        "single_expensive_stage_evaluation",
        not bool(record.get("montage_repeated_expensive_stage_per_tile", False)),
        evidence=record.get("montage_repeated_expensive_stage_per_tile"),
        target="false",
    )
    require(
        "fit_stretch_round_trip_exercised",
        bool(record.get("fit_stretch_pulsed", False)),
        evidence=record.get("fit_stretch_pulsed"),
        target="true",
    )
    if "fit_disable_viewport_mode" in record:
        require(
            "fit_disable_enters_regular_fit",
            str(record.get("fit_disable_viewport_mode", "")) == "auto_untouched",
            evidence={
                "mode": record.get("fit_disable_viewport_mode"),
                "view_range": record.get("fit_disable_view_range"),
            },
            target="auto_untouched regular square-pixel fit",
        )
    if str(record.get("backend", "")) == "pyqtgraph":
        require(
            "pyqtgraph_final_frame_physically_drawn",
            bool(record.get("waited_for_pyqtgraph_draw_after_complete", False))
            and record.get("physical_draw_after_complete_ms") is not None
            and not bool(record.get("pyqtgraph_draw_pending_after_complete", True)),
            evidence={
                "draw_ms": record.get("physical_draw_after_complete_ms"),
                "pending": record.get("pyqtgraph_draw_pending_after_complete"),
            },
            target="logical completion followed by a completed QGraphicsView paint",
        )

    grid_kind = str(record.get("grid_kind", ""))
    grid_count = int(record.get("grid_tile_count", 0) or 0)
    full_count = int(record.get("full_tile_count", 0) or 0)
    if grid_kind == "full":
        require(
            "full_grid_not_capped",
            grid_count == full_count and not bool(record.get("tile_cap_applied", False)),
            evidence={
                "grid": grid_count,
                "full": full_count,
                "capped": record.get("tile_cap_applied"),
            },
            target="full source dimension",
        )
    elif grid_kind == "scroll":
        selected = min(full_count, int(record.get("max_tiles", 60) or 60))
        require(
            "scroll_grid_has_exposed_size",
            grid_count == selected and int(record.get("scroll_window_size", 0) or 0) == selected,
            evidence={
                "grid": grid_count,
                "window": record.get("scroll_window_size"),
                "selected": selected,
            },
            target="the exposed scroll size (60 by default)",
        )

    if "slow_scroll_unreached_target_steps" in record:
        require(
            "slow_scroll_converged",
            int(record.get("slow_scroll_unreached_target_steps", 0) or 0) == 0,
            evidence=record.get("slow_scroll_unreached_target_steps"),
            target="zero unreached steps",
        )
        require(
            "fast_scroll_levels_semantic",
            not bool(record.get("scroll_fast_end_levels_default", True))
            and not bool(record.get("scroll_fast_end_histogram_empty", True)),
            evidence={
                "levels": record.get("scroll_fast_end_levels"),
                "histogram_empty": record.get("scroll_fast_end_histogram_empty"),
            },
            target="levels and histogram remain populated after cold-forward/reverse stress",
        )
        require(
            "fast_scroll_camera_stable",
            float(record.get("fast_scroll_camera_max_drift", float("inf"))) <= 1e-6,
            evidence={
                "before": record.get("fast_scroll_camera_before"),
                "after": record.get("fast_scroll_camera_after"),
                "max_drift": record.get("fast_scroll_camera_max_drift"),
            },
            target="montage source-window scrolling does not change XY camera range",
        )
    if "final_reached_target_lod" in record:
        require(
            "zoompan_recovered_to_target_lod",
            bool(record.get("final_reached_target_lod", False)),
            evidence=record.get("final_settle_ms"),
            target="target LOD reached after the interaction storm",
        )
    if bool(record.get("deep_zoom_far_scroll_available", False)):
        require(
            "deep_zoom_far_scroll_precondition_reaches_native",
            bool(record.get("deep_zoom_far_scroll_precondition_reached_target_lod", False)),
            evidence=record.get("deep_zoom_far_scroll_precondition_settle_ms"),
            target="the center reaches its deep-zoom target before the distant source scroll",
        )
        require(
            "deep_zoom_far_scroll_reaches_target_lod",
            bool(record.get("deep_zoom_far_scroll_reached_target_lod", False)),
            evidence=record.get("deep_zoom_far_scroll_target_evidence"),
            target="the distant successor replaces the retained center at target LOD",
        )
    if bool(record.get("lod_full_grid_checkpoint_available", False)):
        full_grid_checkpoint = dict(record.get("lod_full_grid_checkpoint", {}) or {})
        full_grid_active = int(record.get("lod_full_grid_active_count", 0) or 0)
        expected_grid = int(
            record.get("selected_tile_count", 0) or record.get("scroll_window_size", 0) or 0
        )
        require(
            "full_grid_visible_tiles_reach_target_lod",
            bool(full_grid_checkpoint.get("visible_target_reached", False)),
            evidence=full_grid_checkpoint,
            target="the broad full-grid zoom settles every visible tile at target LOD",
        )
        require(
            "full_grid_checkpoint_stresses_many_tiles",
            full_grid_active >= max(1, math.ceil(expected_grid * 0.8)),
            evidence={"active": full_grid_active, "selected": expected_grid},
            target="at least 80% of the selected montage is simultaneously visible",
        )
        if bool(full_grid_checkpoint.get("resident_query_available", False)):
            require(
                "near_residency_waits_for_full_grid_visible_target",
                int(full_grid_checkpoint.get("near_new_before_visible_count", 0) or 0) == 0,
                evidence=full_grid_checkpoint.get("near_new_before_visible"),
                target="no new near/offscreen residency before broad visible target settlement",
            )
    if bool(record.get("lod_checkpoint_available", False)):
        zoom_checkpoint = dict(record.get("lod_checkpoint_zoom", {}) or {})
        pan_checkpoint = dict(record.get("lod_checkpoint_pan_result", {}) or {})
        zoom_active_count = int(record.get("lod_checkpoint_zoom_active_count", 0) or 0)
        full_grid_active_count = int(record.get("lod_full_grid_active_count", 0) or 0)
        require(
            "zoomed_visible_tiles_reach_target_lod",
            bool(zoom_checkpoint.get("visible_target_reached", False)),
            evidence=zoom_checkpoint,
            target="all visible tiles reach target LOD during the short zoomed-in pause",
        )
        if full_grid_active_count > 0:
            require(
                "lod_checkpoint_reaches_small_visible_region",
                0 < zoom_active_count <= max(12, math.ceil(full_grid_active_count * 0.35)),
                evidence={
                    "zoom_active": zoom_active_count,
                    "full_grid_active": full_grid_active_count,
                },
                target="the deep checkpoint covers a small subset after the broad full-grid transition",
            )
        require(
            "panned_visible_tiles_reach_target_lod",
            bool(pan_checkpoint.get("visible_target_reached", False)),
            evidence=pan_checkpoint,
            target="all newly visible tiles reach target LOD during the short post-pan pause",
        )
        require(
            "lod_checkpoint_pan_changes_visible_set",
            bool(record.get("lod_checkpoint_pan_changed_visible_tiles", False)),
            evidence={
                "zoom_active": zoom_checkpoint.get("active_tiles"),
                "pan_active": pan_checkpoint.get("active_tiles"),
            },
            target="the checkpoint pan enters a different visible tile set",
        )
        if bool(zoom_checkpoint.get("resident_query_available", False)):
            require(
                "near_residency_waits_for_zoomed_visible_target",
                int(zoom_checkpoint.get("near_new_before_visible_count", 0) or 0) == 0,
                evidence=zoom_checkpoint.get("near_new_before_visible"),
                target="no new near/offscreen residency before visible target settlement",
            )
        if bool(pan_checkpoint.get("resident_query_available", False)):
            require(
                "near_residency_waits_for_panned_visible_target",
                int(pan_checkpoint.get("near_new_before_visible_count", 0) or 0) == 0,
                evidence=pan_checkpoint.get("near_new_before_visible"),
                target="no new near/offscreen residency before newly visible target settlement",
            )

    performance_evidence = (
        bool(record.get("pacing_evidence", False))
        and str(record.get("profiler_type", "plain")) == "plain"
    )
    max_observed_callback_ms = max(
        [
            float(record.get("phase_ui_work_observation_max_ms", 0.0) or 0.0),
            *(
                float(item.get("elapsed_ms", 0.0) or 0.0)
                for item in tuple(record.get("phase_recent_ui_work_observations", ()) or ())
                if isinstance(item, dict)
            ),
        ],
        default=0.0,
    )
    direct_ui_fields = {
        key: float(value or 0.0)
        for key, value in record.items()
        if (
            key.endswith("_call_ms")
            or key
            in {"action_render_call_ms", "action_set_view_state_ms", "action_clear_operations_ms"}
        )
        and isinstance(value, (int, float))
    }
    reported_heartbeat_fields = {
        key: float(value or 0.0)
        for key, value in record.items()
        if (key == "event_loop_max_gap_ms" or key.endswith("_max_gap_ms"))
        and isinstance(value, (int, float))
    }
    heartbeat_fields = {
        key: value
        for key, value in reported_heartbeat_fields.items()
        if key != "event_loop_max_gap_ms"
    }
    input_fields = {
        key: float(value or 0.0)
        for key, value in record.items()
        if key.endswith("input_call_max_ms") and isinstance(value, (int, float))
    }
    phase_name = str(record.get("phase", ""))
    heartbeat_gate_applicable = any(token in phase_name for token in ("scroll", "zoompan"))
    if performance_evidence:
        require(
            "gui_callbacks_below_50ms",
            (
                bool(record.get("phase_ui_work_observation_evidence_complete", False))
                or not bool(record.get("phase_recent_ui_work_observations_truncated", False))
            )
            and max([max_observed_callback_ms, *direct_ui_fields.values()], default=0.0)
            < R8_GUI_CALLBACK_MAX_MS,
            evidence={
                "observed_max_ms": max_observed_callback_ms,
                "direct_calls_ms": direct_ui_fields,
            },
            target=f"every synchronous GUI step < {R8_GUI_CALLBACK_MAX_MS:.0f} ms",
            category="performance",
        )
        if heartbeat_gate_applicable:
            require(
                "event_loop_heartbeat",
                bool(heartbeat_fields)
                and max(heartbeat_fields.values(), default=0.0) <= R8_HEARTBEAT_MAX_GAP_MS,
                evidence=heartbeat_fields,
                target=f"pan/scrub max gap <= {R8_HEARTBEAT_MAX_GAP_MS:.0f} ms",
                category="performance",
            )
        if input_fields:
            require(
                "warm_input_dispatch",
                max(input_fields.values(), default=0.0) <= R8_WARM_INPUT_MAX_MS,
                evidence=input_fields,
                target=f"every scrub/zoom input call <= {R8_WARM_INPUT_MAX_MS:.0f} ms",
                category="performance",
            )

    return {
        "r8_gate_applicable": True,
        "r8_gate_passed": not failures,
        "r8_gate_failures": failures,
        "r8_gate_failure_count": len(failures),
        "r8_gate_check_count": int(check_count),
        "r8_performance_evidence": bool(performance_evidence),
        "r8_heartbeat_gate_applicable": bool(performance_evidence and heartbeat_gate_applicable),
        "r8_max_observed_callback_ms": float(max_observed_callback_ms),
        "r8_direct_ui_call_ms": direct_ui_fields,
        "r8_heartbeat_max_gap_ms": max(reported_heartbeat_fields.values(), default=0.0),
        "r8_input_call_max_ms": max(input_fields.values(), default=0.0),
        "r8_targets": {
            "gui_callback_max_ms": R8_GUI_CALLBACK_MAX_MS,
            "heartbeat_max_gap_ms": R8_HEARTBEAT_MAX_GAP_MS,
            "warm_input_max_ms": R8_WARM_INPUT_MAX_MS,
        },
    }


def _process_events(app, QtCore, *, count: int) -> None:
    flags = QtCore.QEventLoop.ProcessEventsFlag.AllEvents
    for _ in range(max(1, int(count))):
        # Use Qt's snapshot overload: it processes the events queued before
        # entry and then returns.  The QDeadlineTimer overload also processes
        # events posted while it runs; under this manual benchmark loop that
        # chained several individually bounded continuation callbacks into a
        # single 50-150 ms pump and measured harness-created starvation.  The
        # independent callback observations still expose every genuinely long
        # handler, while this overload models one production dispatcher turn.
        pump_start = perf_counter()
        app.processEvents(flags)
        pump_ms = (perf_counter() - pump_start) * 1000.0
        samples = getattr(app, "_arrayscope_profile_event_pump_ms", None)
        if isinstance(samples, list):
            samples.append(float(pump_ms))
        # ``processEvents`` is a non-blocking manual pump, not ``app.exec``.
        # Without a scheduler yield this benchmark immediately re-enters it
        # from a Python polling loop; on Wayland that can prevent the platform
        # dispatcher from polling timers/input even though every application
        # callback is bounded. A one-millisecond yield models the natural idle
        # wait of the production event loop. Long callbacks remain visible in
        # both the heartbeat gap and the independent callback observations.
        time.sleep(0.001)


def _restore_setting(settings, key: str, previous) -> None:
    if previous is None:
        settings.remove(key)
    else:
        settings.setValue(key, previous)


def _default_columns(tile_count: int) -> int:
    return max(1, math.ceil(math.sqrt(max(1, int(tile_count)))))


def _normalize_backend(backend: str) -> str:
    backend = str(backend).strip().lower()
    if backend not in {"pyqtgraph", "vispy", "wgpu"}:
        raise ValueError(
            f"unsupported backend {backend!r}; expected 'pyqtgraph', 'vispy', or 'wgpu'"
        )
    return backend


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def py_spy_command(argv: tuple[str, ...] | None = None) -> str:
    args = tuple(sys.argv[1:] if argv is None else argv)
    native = _py_spy_native_requested(args)
    args = _py_spy_filtered_args(args)
    return shlex.join(
        (
            "py-spy",
            "record",
            *(("--native",) if native else ()),
            "--rate",
            str(PY_SPY_LOW_IMPACT_SAMPLE_RATE_HZ),
            "--gil",
            "--nonblocking",
            "-o",
            "arrayscope-montage-workflow.svg",
            "--",
            sys.executable,
            "-m",
            "arrayscope.tools.profile_montage_workflow",
            *args,
        )
    )


def cprofile_command(argv: tuple[str, ...], output: str | Path) -> str:
    return shlex.join(
        (
            sys.executable,
            "-m",
            "cProfile",
            "-o",
            str(output),
            "-m",
            "arrayscope.tools.profile_montage_workflow",
            *tuple(argv),
        )
    )


def py_spy_raw_command(
    argv: tuple[str, ...],
    output: str | Path,
    *,
    rate_hz: int,
    nonblocking: bool = False,
    gil: bool = False,
    duration_s: int | None = None,
    detach_margin_s: int = 0,
) -> str:
    native = _py_spy_native_requested(tuple(argv))
    target = (
        (
            sys.executable,
            "-c",
            "import sys, time; from arrayscope.tools.profile_montage_workflow import main; "
            "started = time.monotonic(); duration = float(sys.argv[-2]); margin = float(sys.argv[-1]); "
            "rc = main(tuple(sys.argv[1:-2])); "
            "time.sleep(max(0.0, started + duration + margin - time.monotonic())); raise SystemExit(rc)",
            *tuple(argv),
            str(int(duration_s or 0)),
            str(int(detach_margin_s)),
        )
        if duration_s is not None
        else (sys.executable, "-m", "arrayscope.tools.profile_montage_workflow", *tuple(argv))
    )
    return shlex.join(
        (
            "py-spy",
            "record",
            *(("--native",) if native else ()),
            *(("--duration", str(int(duration_s))) if duration_s is not None else ()),
            "--rate",
            str(int(rate_hz)),
            *(("--nonblocking",) if nonblocking else ()),
            *(("--gil",) if gil else ()),
            "--format",
            "raw",
            "-o",
            str(output),
            "--",
            *target,
        )
    )


def perf_record_command(argv: tuple[str, ...], output: str | Path) -> str:
    return shlex.join(
        (
            "perf",
            "record",
            "-F",
            "99",
            "-g",
            "-o",
            str(output),
            "--",
            sys.executable,
            "-m",
            "arrayscope.tools.profile_montage_workflow",
            *tuple(argv),
        )
    )


def profiler_suite_commands(
    argv: tuple[str, ...], suite_dir: str | Path
) -> tuple[dict[str, object], ...]:
    suite_dir = Path(suite_dir)
    plain_jsonl = suite_dir / "plain.jsonl"
    base = _suite_child_args(argv)
    include_cprofile = _include_cprofile_requested(argv)
    py_spy_native = _py_spy_native_requested(argv)
    py_spy_low_type = "py-spy-raw-low-impact-native" if py_spy_native else "py-spy-raw-low-impact"
    py_spy_full_type = "py-spy-raw-full-native" if py_spy_native else "py-spy-raw-full"
    backends = _suite_profiler_backends(base)
    split_backend_artifacts = len(backends) > 1
    commands: list[dict[str, object]] = [
        {
            "step_id": "plain",
            "profiler_type": "plain",
            "backend": "all" if split_backend_artifacts else backends[0],
            "required": True,
            "jsonl": str(plain_jsonl),
            "artifact_paths": (str(plain_jsonl),),
            "command": shlex.join(
                (
                    sys.executable,
                    "-m",
                    "arrayscope.tools.profile_montage_workflow",
                    *base,
                    "--jsonl",
                    str(plain_jsonl),
                    "--profiler-type",
                    "plain",
                    "--profiler-artifact",
                    str(plain_jsonl),
                )
            ),
        },
    ]
    if include_cprofile:
        for backend in backends:
            backend_base = _suite_args_for_backend(base, backend)
            suffix = f".{backend}" if split_backend_artifacts else ""
            cprofile_artifact = suite_dir / f"montage{suffix}.cprofile"
            cprofile_jsonl = suite_dir / f"cprofile{suffix}.jsonl"
            commands.append(
                {
                    "step_id": f"cprofile:{backend}",
                    "profiler_type": "cprofile",
                    "backend": backend,
                    "required": True,
                    "jsonl": str(cprofile_jsonl),
                    "artifact_paths": (str(cprofile_artifact), str(cprofile_jsonl)),
                    "command": cprofile_command(
                        (
                            *backend_base,
                            "--jsonl",
                            str(cprofile_jsonl),
                            "--profiler-type",
                            "cprofile",
                            "--profiler-artifact",
                            str(cprofile_artifact),
                        ),
                        cprofile_artifact,
                    ),
                }
            )
    for backend in backends:
        backend_base = _suite_args_for_backend(base, backend)
        py_spy_base = (*backend_base, "--py-spy-native") if py_spy_native else backend_base
        suffix = f".{backend}" if split_backend_artifacts else ""
        py_spy_low_artifact = suite_dir / f"montage{suffix}.pyspy.low-impact.raw"
        py_spy_low_jsonl = suite_dir / f"py-spy-low-impact{suffix}.jsonl"
        py_spy_full_artifact = suite_dir / f"montage{suffix}.pyspy.full.raw"
        py_spy_full_jsonl = suite_dir / f"py-spy-full{suffix}.jsonl"
        perf_artifact = suite_dir / f"montage{suffix}.perf.data"
        perf_jsonl = suite_dir / f"perf{suffix}.jsonl"
        commands.extend(
            (
                {
                    "step_id": f"{py_spy_low_type}:{backend}",
                    "profiler_type": py_spy_low_type,
                    "backend": backend,
                    "required": True,
                    "jsonl": str(py_spy_low_jsonl),
                    "artifact_paths": (str(py_spy_low_artifact), str(py_spy_low_jsonl)),
                    "command": py_spy_raw_command(
                        (
                            *py_spy_base,
                            "--jsonl",
                            str(py_spy_low_jsonl),
                            "--profiler-type",
                            py_spy_low_type,
                            "--profiler-artifact",
                            str(py_spy_low_artifact),
                        ),
                        py_spy_low_artifact,
                        rate_hz=PY_SPY_LOW_IMPACT_SAMPLE_RATE_HZ,
                        nonblocking=True,
                        gil=True,
                    ),
                },
                {
                    "step_id": f"{py_spy_full_type}:{backend}",
                    "profiler_type": py_spy_full_type,
                    "backend": backend,
                    "required": True,
                    "jsonl": str(py_spy_full_jsonl),
                    "artifact_paths": (str(py_spy_full_artifact), str(py_spy_full_jsonl)),
                    "command": py_spy_raw_command(
                        (
                            *py_spy_base,
                            "--jsonl",
                            str(py_spy_full_jsonl),
                            "--profiler-type",
                            py_spy_full_type,
                            "--profiler-artifact",
                            str(py_spy_full_artifact),
                        ),
                        py_spy_full_artifact,
                        rate_hz=PY_SPY_FULL_SAMPLE_RATE_HZ,
                        nonblocking=True,
                        duration_s=PY_SPY_FULL_DURATION_S,
                        detach_margin_s=PY_SPY_FULL_DETACH_MARGIN_S,
                    ),
                },
                {
                    "step_id": f"perf-record:{backend}",
                    "profiler_type": "perf-record",
                    "backend": backend,
                    "required": False,
                    "jsonl": str(perf_jsonl),
                    "artifact_paths": (str(perf_artifact), str(perf_jsonl)),
                    "command": perf_record_command(
                        (
                            *backend_base,
                            "--jsonl",
                            str(perf_jsonl),
                            "--profiler-type",
                            "perf-record",
                            "--profiler-artifact",
                            str(perf_artifact),
                        ),
                        perf_artifact,
                    ),
                },
            )
        )
    return tuple(commands)


def run_profile_suite(argv: tuple[str, ...], suite_dir: str | Path) -> int:
    suite_dir = Path(suite_dir)
    suite_dir.mkdir(parents=True, exist_ok=True)
    commands = profiler_suite_commands(argv, suite_dir)
    manifest_path = suite_dir / "suite-manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()
    tool_versions = _suite_tool_versions()
    repository = _repository_state()
    step_records: list[dict[str, object]] = []
    for item in commands:
        profiler = str(item["profiler_type"])
        step_temperature = _suite_step_temperature(step_records)
        command = str(item["command"])
        executable = shlex.split(command)[0]
        log_stem = _artifact_stem(str(item.get("step_id", profiler)))
        stdout_path = suite_dir / f"{log_stem}.stdout.log"
        stderr_path = suite_dir / f"{log_stem}.stderr.log"
        required = bool(item.get("required", False))
        if profiler.startswith("py-spy") and shutil.which("py-spy") is None:
            status = "failed" if required else "skipped"
            reason_key = "failure_reason" if status == "failed" else "skip_reason"
            record = _suite_step_record(
                item,
                command_executable=executable,
                returncode=127,
                elapsed_ms=0.0,
                missing_artifacts=tuple(item.get("artifact_paths", ()) or ()),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                status=status,
                reason_key=reason_key,
                reason="py-spy executable not found",
                tool_versions=tool_versions,
                repository=repository,
                run_temperature=step_temperature,
            )
            _write_manifest_record(manifest_path, record)
            step_records.append(record)
            continue
        if profiler.startswith("perf") and shutil.which("perf") is None:
            record = _suite_step_record(
                item,
                command_executable=executable,
                returncode=127,
                elapsed_ms=0.0,
                missing_artifacts=tuple(item.get("artifact_paths", ()) or ()),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                status="skipped",
                reason_key="skip_reason",
                reason="perf executable not found",
                tool_versions=tool_versions,
                repository=repository,
                run_temperature=step_temperature,
            )
            _write_manifest_record(manifest_path, record)
            step_records.append(record)
            continue
        started = perf_counter()
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            completed = subprocess.run(
                shlex.split(command), cwd=Path.cwd(), check=False, stdout=stdout, stderr=stderr
            )
        elapsed_ms = (perf_counter() - started) * 1000.0
        artifacts = tuple(str(path) for path in tuple(item.get("artifact_paths", ()) or ()))
        missing = [
            path for path in artifacts if not Path(path).exists() or Path(path).stat().st_size <= 0
        ]
        profiler_diagnostics = _profiler_log_diagnostics(profiler, stdout_path, stderr_path)
        sample_issue = _profiler_sample_issue(profiler, profiler_diagnostics)
        complete = int(completed.returncode) == 0 and not missing and sample_issue == ""
        if complete:
            status = "completed"
            reason_key = None
            reason = ""
        else:
            status = "failed" if required else "degraded"
            reason_key = "failure_reason" if status == "failed" else "degraded_reason"
            if int(completed.returncode) != 0:
                reason = f"command exited with {int(completed.returncode)}"
            elif sample_issue:
                reason = sample_issue
            else:
                reason = "missing or empty expected artifacts"
        record = _suite_step_record(
            item,
            command_executable=executable,
            returncode=int(completed.returncode),
            elapsed_ms=elapsed_ms,
            missing_artifacts=missing,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            profiler_diagnostics=profiler_diagnostics,
            status=status,
            reason_key=reason_key,
            reason=reason,
            tool_versions=tool_versions,
            repository=repository,
            run_temperature=step_temperature,
        )
        _write_manifest_record(manifest_path, record)
        step_records.append(record)
    summary = _suite_summary_record(
        step_records, tool_versions=tool_versions, repository=repository
    )
    interpretation_path = suite_dir / "suite-summary.md"
    summary["interpretation_path"] = str(interpretation_path)
    _write_suite_interpretation(interpretation_path, step_records, summary)
    print(interpretation_path.read_text(encoding="utf-8"), end="")
    _write_manifest_record(manifest_path, summary)
    return 0 if bool(summary["overall_valid"]) else _suite_exit_code(step_records)


def _write_manifest_record(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _suite_step_record(
    item: dict[str, object],
    *,
    command_executable: str,
    returncode: int,
    elapsed_ms: float,
    missing_artifacts,
    stdout_path: Path,
    stderr_path: Path,
    profiler_diagnostics: dict[str, object] | None = None,
    status: str,
    reason_key: str | None = None,
    reason: str = "",
    tool_versions: dict[str, str],
    repository: dict[str, object],
    run_temperature: str,
) -> dict[str, object]:
    complete = str(status) == "completed"
    record = {
        **item,
        "record_type": "suite_step",
        "required": bool(item.get("required", False)),
        "status": str(status),
        "valid": bool(complete),
        "complete": bool(complete),
        "command_executable": str(command_executable),
        "returncode": int(returncode),
        "elapsed_ms": float(elapsed_ms),
        "missing_artifacts": [str(path) for path in tuple(missing_artifacts or ())],
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "profiler_diagnostics": dict(profiler_diagnostics or {}),
        "tool_versions": dict(tool_versions),
        "repository_revision": str(repository["repository_revision"]),
        "repository_dirty": bool(repository["repository_dirty"]),
        "run_temperature": str(run_temperature),
    }
    if reason_key is not None and reason:
        record[str(reason_key)] = str(reason)
    return record


def _suite_summary_record(
    records: list[dict[str, object]],
    *,
    tool_versions: dict[str, str],
    repository: dict[str, object],
) -> dict[str, object]:
    statuses = {
        str(record.get("step_id", record["profiler_type"])): str(record["status"])
        for record in records
    }
    overall_valid = bool(records) and all(bool(record.get("valid", False)) for record in records)
    if overall_valid:
        overall_status = "completed"
    elif any(str(record.get("status")) == "failed" for record in records):
        overall_status = "failed"
    else:
        overall_status = "degraded"
    return {
        "record_type": "suite_summary",
        "overall_valid": bool(overall_valid),
        "overall_status": overall_status,
        "step_statuses": statuses,
        "step_count": len(records),
        "tool_versions": dict(tool_versions),
        "repository_revision": str(repository["repository_revision"]),
        "repository_dirty": bool(repository["repository_dirty"]),
        "run_temperature": _aggregate_run_temperature(
            tuple(str(record.get("run_temperature", "")) for record in records)
        ),
    }


def _suite_exit_code(records: list[dict[str, object]]) -> int:
    for record in records:
        if str(record.get("status")) == "failed":
            return int(record.get("returncode") or 1)
    for record in records:
        if str(record.get("status")) in {"degraded", "skipped"}:
            return int(record.get("returncode") or 2)
    return 2


def _artifact_stem(profiler: str) -> str:
    return str(profiler).replace("/", "-").replace(" ", "-")


def _profiler_log_diagnostics(
    profiler: str, stdout_path: Path, stderr_path: Path
) -> dict[str, object]:
    stdout = _read_text_if_exists(stdout_path)
    stderr = _read_text_if_exists(stderr_path)
    diagnostics: dict[str, object] = {
        "warning_count": stderr.count("WARN  "),
    }
    if str(profiler).startswith("py-spy"):
        rate_match = re.search(r"Sampling process\s+(\d+)\s+times a second", stdout)
        match = re.search(r"Samples:\s*(\d+)\s+Errors:\s*(\d+)", stdout)
        diagnostics["sample_rate_hz"] = int(rate_match.group(1)) if rate_match else 0
        diagnostics["sample_count"] = int(match.group(1)) if match else 0
        diagnostics["error_count"] = int(match.group(2)) if match else 0
        diagnostics["missed_stack_count"] = stderr.count("Failed to get stack trace")
        diagnostics["scope"] = (
            "low_impact_python_gil_holders"
            if "low-impact" in str(profiler)
            else "complete_sampling_all_python_threads"
        )
        diagnostics["sampling_mode"] = (
            "nonblocking_gil_samples"
            if "low-impact" in str(profiler)
            else "blocking_all_python_thread_samples"
        )
        diagnostics["allowed_missed_stack_count"] = (
            0 if "low-impact" in str(profiler) else PY_SPY_FULL_ALLOWED_MISSED_STACKS
        )
        diagnostics["sampling_complete"] = bool(
            int(diagnostics["sample_count"]) > 0
            and (
                ("low-impact" in str(profiler))
                or (
                    int(diagnostics["error_count"]) <= PY_SPY_FULL_ALLOWED_MISSED_STACKS
                    and int(diagnostics["missed_stack_count"]) <= PY_SPY_FULL_ALLOWED_MISSED_STACKS
                )
            )
        )
    elif str(profiler).startswith("perf"):
        diagnostics["scope"] = "native_and_python_process_samples"
    elif str(profiler) == "cprofile":
        diagnostics["scope"] = "deterministic_python_calls"
    else:
        diagnostics["scope"] = "workflow_timing_jsonl"
    return diagnostics


def _profiler_sample_issue(profiler: str, diagnostics: dict[str, object]) -> str:
    if str(profiler).startswith("py-spy") and int(diagnostics.get("sample_count", 0) or 0) <= 0:
        return "py-spy produced no samples"
    if (
        str(profiler).startswith("py-spy")
        and "full" in str(profiler)
        and not bool(diagnostics.get("sampling_complete", False))
    ):
        return f"py-spy full profile missed more than {PY_SPY_FULL_ALLOWED_MISSED_STACKS} stack sample(s)"
    return ""


def _read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _write_suite_interpretation(
    path: Path, records: list[dict[str, object]], summary: dict[str, object]
) -> None:
    lines = [
        "# Profile Suite Summary",
        "",
        f"- Overall status: `{summary.get('overall_status')}`",
        f"- Overall valid: `{summary.get('overall_valid')}`",
        f"- Run temperature: `{summary.get('run_temperature')}`",
        "",
        "## Run Metadata",
        "",
        *_run_metadata_summary(summary),
        "",
        "## Tool Status",
        "",
        *_tool_status_summary(records),
        "",
        "## Timing Evidence",
        "",
    ]
    plain = _find_step(records, "plain")
    if plain is None:
        lines.append("Plain JSONL timing evidence was not produced.")
    else:
        lines.extend(_plain_timing_summary(Path(str(plain.get("jsonl", "")))))
        lines.extend(["", "## Tooling Slowdown", ""])
        lines.extend(_tooling_slowdown_summary(plain, records))
    lines.extend(["", "## Python Attribution", ""])
    backend_names = _summary_backend_names(records)
    py_spy_count = 0
    for backend in backend_names:
        backend_lines: list[str] = []
        low = _find_step(records, "py-spy-raw-low-impact", backend=backend)
        full = _find_step(records, "py-spy-raw-full", backend=backend)
        generic = (
            _find_step(records, "py-spy-raw", backend=backend)
            if low is None and full is None
            else None
        )
        if low is not None:
            backend_lines.extend(_py_spy_summary(low, title="Low-impact py-spy", heading="####"))
        if full is not None:
            backend_lines.extend(_py_spy_summary(full, title="Full sampled py-spy", heading="####"))
        if generic is not None:
            backend_lines.extend(_py_spy_summary(generic, title="py-spy", heading="####"))
        if backend_lines:
            lines.extend([f"### {backend}", "", *backend_lines])
            py_spy_count += 1
    if py_spy_count == 0:
        lines.append("No py-spy artifacts were produced.")
    lines.extend(["", "## Deterministic Python Calls", ""])
    cprofile_count = 0
    for backend in backend_names:
        cprofile = _find_step(records, "cprofile", backend=backend)
        if cprofile is not None:
            lines.extend([f"### {backend}", ""])
            lines.extend(_cprofile_summary(cprofile))
            cprofile_count += 1
    if cprofile_count == 0:
        lines.append(
            "cProfile was not run. Use `--include-cprofile` when deterministic Python call counts are worth the slowdown."
        )
    lines.extend(["", "## Native Attribution", ""])
    perf_count = 0
    for backend in backend_names:
        perf = _find_step(records, "perf-record", backend=backend)
        if perf is not None:
            lines.extend([f"### {backend}", ""])
            lines.extend(_perf_summary(perf))
            perf_count += 1
    if perf_count == 0:
        lines.append("perf was not run.")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _run_metadata_summary(summary: dict[str, object]) -> list[str]:
    versions = dict(summary.get("tool_versions", {}) or {})
    return [
        f"- Revision: `{summary.get('repository_revision', 'unknown')}`",
        f"- Dirty worktree: `{summary.get('repository_dirty')}`",
        "- Tools: "
        + ", ".join(
            f"`{name} {versions.get(name, 'unknown')}`"
            for name in ("python", "PySide6", "pyqtgraph", "vispy", "py-spy", "perf")
        ),
    ]


def _tool_status_summary(records: list[dict[str, object]]) -> list[str]:
    rows = [
        "| Step | backend | required | status | rc | artifacts | profiler health |",
        "|---|---|---:|---|---:|---|---|",
    ]
    rows.extend(
        "| "
        + " | ".join(
            (
                f"`{record.get('step_id', record.get('profiler_type', ''))}`",
                f"`{record.get('backend', '')}`",
                str(bool(record.get("required", False))),
                str(record.get("status", "")),
                str(record.get("returncode", "")),
                _artifact_status(record),
                _profiler_health(record),
            )
        )
        + " |"
        for record in records
    )
    return rows


def _artifact_status(record: dict[str, object]) -> str:
    artifacts = tuple(record.get("artifact_paths", ()) or ())
    missing = tuple(record.get("missing_artifacts", ()) or ())
    if not artifacts:
        return "none"
    if missing:
        return f"{len(artifacts) - len(missing)}/{len(artifacts)} present"
    return f"{len(artifacts)}/{len(artifacts)} present"


def _profiler_health(record: dict[str, object]) -> str:
    diagnostics = dict(record.get("profiler_diagnostics", {}) or {})
    profiler = str(record.get("profiler_type", ""))
    if profiler.startswith("py-spy"):
        return (
            f"samples {diagnostics.get('sample_count', 0)}, "
            f"errors {diagnostics.get('error_count', 0)}, "
            f"missed {diagnostics.get('missed_stack_count', 0)}"
        )
    warning_count = diagnostics.get("warning_count")
    if warning_count is not None:
        return f"warnings {warning_count}"
    return str(diagnostics.get("scope", "n/a"))


def _find_step(
    records: list[dict[str, object]], profiler_type: str, *, backend: str | None = None
) -> dict[str, object] | None:
    for record in records:
        record_backend = str(record.get("backend", "all") or "all")
        if backend is not None and record_backend != str(backend):
            continue
        if str(record.get("profiler_type", "")).startswith(profiler_type):
            return record
    return None


def _summary_backend_names(records: list[dict[str, object]]) -> tuple[str, ...]:
    names = sorted(
        {
            str(record.get("backend", ""))
            for record in records
            if str(record.get("backend", "")) not in {"", "all"}
        }
    )
    return tuple(names) if names else ("all",)


def _plain_timing_summary(path: Path) -> list[str]:
    records = _read_jsonl(path)
    if not records:
        return [f"No readable timing records found at `{path}`."]
    lines = [
        f"Source: `{path}`",
        "",
        "Headline only; detailed counters remain in JSONL.",
        "",
        "| Backend | phase | temp | pacing ms | event-loop gap ms | tiles | work |",
        "|---|---|---|---|---|---:|---|",
    ]
    for record in records:
        backend = str(record.get("backend", ""))
        phase = str(record.get("phase", ""))
        temperature = str(record.get("run_temperature", ""))
        pacing = _pacing_summary(record)
        gaps = _event_loop_summary(record)
        tiles = _tile_summary(record)
        work = _work_summary(record)
        lines.append(
            f"| `{backend}` | `{phase}` | `{temperature}` | {pacing} | {gaps} | {tiles} | {work} |"
        )
    return lines


def _workflow_timing_summary(records: tuple[dict[str, object], ...]) -> str:
    if not records:
        return "No workflow timing records were produced.\n"
    lines = [
        "Workflow timing summary",
        "| Backend | phase | R8 | first tile | preview floor | initial fill | full refined | visible after first | elapsed | event-loop max | histogram-loop action | level/rgb | textures | histogram | sync | tiles |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|",
    ]
    lines.extend(
        "| "
        + " | ".join(
            (
                f"`{record.get('backend', '')}`",
                f"`{record.get('phase', '')}`",
                _r8_gate_summary(record),
                _format_ms(
                    record.get("first_visible_tile_ms", record.get("first_display_payload_ms"))
                ),
                _format_ms(
                    _first_non_none(
                        record, "first_preview_floor_fill_ms", "first_preview_payload_fill_ms"
                    )
                ),
                _format_ms(
                    record.get("first_display_payload_fill_ms", record.get("fully_visible_ms"))
                ),
                _format_ms(
                    _first_non_none(
                        record,
                        "required_target_settled_ms",
                        "draw_after_complete_ms",
                        "fully_visible_ms",
                    )
                ),
                _format_ms(record.get("fully_visible_after_first_visible_tile_ms")),
                _format_ms(record.get("elapsed_ms")),
                _format_ms(record.get("event_loop_max_gap_ms")),
                _histogram_loop_action_summary(record),
                _level_work_summary(record),
                _texture_work_summary(record),
                _format_ms(record.get("last_histogram_recompute_ms")),
                _format_ms(record.get("last_level_sync_ms")),
                _tile_summary(record),
            )
        )
        + " |"
        for record in records
    )
    return "\n".join(lines) + "\n"


def _r8_gate_summary(record: dict[str, object]) -> str:
    if not bool(record.get("r8_gate_applicable", False)):
        return "n/a"
    if bool(record.get("r8_gate_passed", False)):
        return "PASS"
    failures = tuple(record.get("r8_gate_failures", ()) or ())
    names = [str(item.get("gate", "?")) for item in failures if isinstance(item, dict)]
    return "FAIL: " + ", ".join(names)


def _first_non_none(record: dict[str, object], *keys: str):
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def _histogram_loop_action_summary(record: dict[str, object]) -> str:
    if "histogram_loop_action_ms" not in record:
        return "n/a"
    return (
        f"hist {_format_ms(record.get('histogram_loop_first_half_step_mean_ms'))}; "
        f"full {_format_ms(record.get('histogram_loop_second_half_step_mean_ms'))}; "
        f"max {_format_ms(record.get('histogram_loop_step_max_ms'))}"
    )


def _level_work_summary(record: dict[str, object]) -> str:
    rgb_tiles = int(
        record.get("histogram_loop_rgb_window_tiles", record.get("tile_layer_rgb_window_tiles", 0))
        or 0
    )
    uniform_updates = int(
        record.get("histogram_loop_level_updates", record.get("tile_layer_level_updates", 0)) or 0
    )
    shader_uniform_updates = int(
        record.get(
            "histogram_loop_shader_uniform_updates",
            record.get("tile_layer_shader_uniform_updates", 0),
        )
        or 0
    )
    rgb_ms = _format_ms(
        record.get("histogram_loop_rgb_window_ms", record.get("last_tile_layer_rgb_window_ms"))
    )
    steps = record.get("histogram_loop_steps")
    prefix = f"{steps}x; " if steps is not None else ""
    return f"{prefix}rgb {rgb_tiles} / {rgb_ms}; level {uniform_updates}; shader {shader_uniform_updates}"


def _texture_work_summary(record: dict[str, object]) -> str:
    uploads = int(
        record.get("histogram_loop_texture_uploads", record.get("tile_layer_texture_uploads", 0))
        or 0
    )
    bytes_text = _format_bytes(
        record.get(
            "histogram_loop_texture_upload_bytes", record.get("tile_layer_texture_upload_bytes")
        )
    )
    vertex = int(record.get("tile_layer_vertex_uploads", 0) or 0)
    return f"upload {uploads} / {bytes_text}; vertex {vertex}"


def _py_spy_summary(record: dict[str, object], *, title: str, heading: str = "###") -> list[str]:
    diagnostics = dict(record.get("profiler_diagnostics", {}) or {})
    lines = [
        f"{heading} {title}",
        "",
        f"- Status: `{record.get('status')}`",
        f"- Mode: `{diagnostics.get('sampling_mode', 'unknown')}`",
        f"- Samples/errors: `{diagnostics.get('sample_count', 0)}` / `{diagnostics.get('error_count', 0)}`",
        f"- Missed stacks: `{diagnostics.get('missed_stack_count', 0)}`",
    ]
    artifacts = tuple(record.get("artifact_paths", ()) or ())
    raw_paths = [Path(str(path)) for path in artifacts if str(path).endswith(".raw")]
    if raw_paths:
        top = _top_py_spy_stacks(raw_paths[0])
        if top:
            lines.extend(["", "Top sampled stacks (cropped to leaf context):"])
            for stack, count in top:
                lines.append(f"- `{count}` {stack}")
        else:
            lines.append(f"- No readable raw stack samples found in `{raw_paths[0]}`.")
    return lines


def _tooling_slowdown_summary(
    plain: dict[str, object], records: list[dict[str, object]]
) -> list[str]:
    plain_phases = _backend_phase_elapsed_map(Path(str(plain.get("jsonl", ""))))
    if not plain_phases:
        return ["No readable plain timing records were available for slowdown comparisons."]
    rows: list[str] = []
    for record in records:
        profiler_type = str(record.get("profiler_type", ""))
        if profiler_type == "plain":
            continue
        phases = _backend_phase_elapsed_map(Path(str(record.get("jsonl", ""))))
        record_backend = str(record.get("backend", "") or "")
        compared_backends = (
            (record_backend,)
            if record_backend and record_backend != "all"
            else tuple(sorted(plain_phases))
        )
        for backend in compared_backends:
            if backend not in plain_phases:
                continue
            baseline = plain_phases.get(backend, {})
            values = phases.get(backend, {})
            rows.append(
                "| "
                + " | ".join(
                    (
                        f"`{profiler_type}`",
                        f"`{backend}`",
                        str(record.get("status", "unknown")),
                        _delta_cell(
                            values.get("raw_full_tiled_montage"),
                            baseline.get("raw_full_tiled_montage"),
                        ),
                        _delta_cell(
                            values.get("fft_full_tiled_montage"),
                            baseline.get("fft_full_tiled_montage"),
                        ),
                        _delta_cell(_combined_elapsed_ms(values), _combined_elapsed_ms(baseline)),
                    )
                )
                + " |"
            )
    if not rows:
        return ["No tooled runs were available for slowdown comparisons."]
    return [
        "Compared with the non-tooled `plain` workflow. Negative values mean the tooled run happened to finish faster.",
        "",
        "| Tool | backend | status | normal delta | FFT delta | combined delta |",
        "|---|---|---|---:|---:|---:|",
        *rows,
    ]


def _cprofile_summary(record: dict[str, object]) -> list[str]:
    artifacts = tuple(record.get("artifact_paths", ()) or ())
    profiles = [Path(str(path)) for path in artifacts if str(path).endswith(".cprofile")]
    if not profiles or not profiles[0].exists():
        return ["cProfile was requested, but no readable `.cprofile` artifact was found."]
    stream = io.StringIO()
    try:
        stats = pstats.Stats(str(profiles[0]), stream=stream).strip_dirs().sort_stats("cumulative")
        stats.print_stats(8)
    except Exception as exc:
        return [f"Could not read cProfile artifact `{profiles[0]}`: {exc}"]
    lines = [
        f"Source: `{profiles[0]}`",
        "",
        "Top cumulative Python call entries (cropped):",
        "```text",
    ]
    lines.extend(stream.getvalue().strip().splitlines()[:14])
    lines.append("```")
    return lines


def _perf_summary(record: dict[str, object]) -> list[str]:
    artifacts = tuple(record.get("artifact_paths", ()) or ())
    perf_paths = [Path(str(path)) for path in artifacts if str(path).endswith(".data")]
    if not perf_paths or not perf_paths[0].exists():
        return ["perf was requested, but no readable `perf.data` artifact was found."]
    completed = None
    if shutil.which("perf") is not None:
        try:
            completed = subprocess.run(
                (
                    "perf",
                    "report",
                    "--stdio",
                    "-i",
                    str(perf_paths[0]),
                    "--no-children",
                    "--sort",
                    "comm,dso,symbol",
                ),
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            completed = None
    lines = [f"Source: `{perf_paths[0]}`"]
    if completed is None or completed.returncode != 0:
        lines.append("Run `perf report -i <artifact>` locally for native attribution.")
        return lines
    report_lines = _interesting_perf_lines(completed.stdout.splitlines())[:12]
    lines.extend(["", "Top perf report lines (cropped):", "```text", *report_lines, "```"])
    return lines


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception:
        return []


def _backend_phase_elapsed_map(path: Path) -> dict[str, dict[str, float]]:
    phases: dict[str, dict[str, float]] = {}
    for record in _read_jsonl(path):
        backend = str(record.get("backend", "unknown"))
        phase = str(record.get("phase", ""))
        try:
            phases.setdefault(backend, {})[phase] = float(record["elapsed_ms"])
        except Exception:
            continue
    return phases


def _combined_elapsed_ms(phases: dict[str, float]) -> float | None:
    raw = phases.get("raw_full_tiled_montage")
    fft = phases.get("fft_full_tiled_montage")
    if raw is None or fft is None:
        return None
    return raw + fft


def _delta_cell(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None or baseline == 0:
        return "n/a"
    delta = value - baseline
    percent = (delta / baseline) * 100.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f} ms ({sign}{percent:.1f}%)"


def _top_py_spy_stacks(path: Path, *, limit: int = 8) -> list[tuple[str, int]]:
    stacks: list[tuple[str, int]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        stack, sep, raw_count = line.rpartition(" ")
        if not sep:
            continue
        try:
            count = int(raw_count)
        except ValueError:
            continue
        frames = [frame.strip() for frame in stack.split(";") if frame.strip()]
        if not frames:
            continue
        interesting = " -> ".join(frames[-4:]) if frames else stack
        stacks.append((interesting, count))
    stacks.sort(key=lambda item: item[1], reverse=True)
    return stacks[:limit]


def _interesting_perf_lines(lines: list[str]) -> list[str]:
    result = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "%" not in stripped and "Overhead" not in stripped:
            continue
        result.append(line[:180])
    return result


def _format_ms(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}"
    except Exception:
        return "n/a"


def _percentile(values: tuple[float, ...], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (float(percentile) / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _tile_summary(record: dict[str, object]) -> str:
    tile_count = record.get("tile_count")
    full = record.get("full_tile_count")
    requested = record.get("requested_tile_count")
    presented = record.get("active_presented_tile_count")
    if requested is not None and presented is not None:
        return f"{presented}/{requested}"
    if tile_count is not None and full is not None:
        return f"{tile_count}/{full}"
    return "n/a"


def _pacing_summary(record: dict[str, object]) -> str:
    first = record.get("first_loaded_tile_ms") or record.get("first_display_committed_ms")
    first_fill = record.get("first_display_payload_fill_ms")
    fill_after_first_payload = record.get("first_display_payload_fill_after_first_payload_ms")
    visible_after_first_tile = record.get("fully_visible_after_first_visible_tile_ms")
    preview_fill = record.get("first_preview_payload_fill_ms")
    exact_visible = record.get("fully_visible_ms")
    draw = record.get("draw_after_complete_ms")
    full = draw if draw is not None else record.get("elapsed_ms")
    return " / ".join(
        (
            f"first {_format_ms(first)}",
            f"fill {_format_ms(first_fill)}",
            f"fill-after-first {_format_ms(fill_after_first_payload)}",
            f"visible-fill {_format_ms(visible_after_first_tile)}",
            f"preview {_format_ms(preview_fill)}",
            f"visible {_format_ms(exact_visible)}",
            f"full {_format_ms(full)}",
        )
    )


def _event_loop_summary(record: dict[str, object]) -> str:
    p95 = _format_ms(record.get("event_loop_p95_gap_ms"))
    p99 = _format_ms(record.get("event_loop_p99_gap_ms"))
    max_gap = _format_ms(record.get("event_loop_max_gap_ms"))
    return f"p95 {p95} / p99 {p99} / max {max_gap}"


def _work_summary(record: dict[str, object]) -> str:
    parts = []
    compute = _compute_summary(record)
    if compute != "n/a":
        parts.append(compute)
    upload_bytes = _format_bytes(record.get("tile_layer_texture_upload_bytes"))
    uploads = record.get("tile_layer_texture_uploads")
    if uploads is not None:
        parts.append(f"upload {uploads} / {upload_bytes}")
    updated = record.get("tile_layer_items_updated")
    skipped = record.get("tile_layer_items_skipped")
    if updated is not None and skipped is not None:
        parts.append(f"items up {updated}, skip {skipped}")
    level_updates = record.get("tile_layer_level_updates")
    vertex_uploads = record.get("tile_layer_vertex_uploads")
    if level_updates is not None or vertex_uploads is not None:
        parts.append(f"rebind level {level_updates or 0}, vertex {vertex_uploads or 0}")
    pages = record.get("tile_layer_active_pages")
    gpu_bytes = record.get("tile_layer_estimated_gpu_bytes")
    if pages is not None and gpu_bytes is not None:
        parts.append(f"resident {pages} pages / {_format_bytes(gpu_bytes)}")
    return "; ".join(parts) if parts else "n/a"


def _compute_summary(record: dict[str, object]) -> str:
    direct = record.get("montage_tile_compute_direct")
    stage = record.get("montage_tile_compute_stage_backed")
    cache = record.get("montage_tile_compute_cache_hits")
    parts = []
    if direct is not None:
        parts.append(f"direct {direct}")
    if stage is not None:
        parts.append(f"stage {stage}")
    if cache is not None:
        parts.append(f"cache {cache}")
    return ", ".join(parts) if parts else "n/a"


def _format_bytes(value) -> str:
    if value is None:
        return "n/a"
    try:
        amount = float(value)
    except Exception:
        return "n/a"
    units = ("B", "KiB", "MiB", "GiB")
    unit = units[0]
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    if unit == "B":
        return f"{int(amount)} {unit}"
    return f"{amount:.1f} {unit}"


def _suite_step_temperature(previous_records: list[dict[str, object]]) -> str:
    return "cold" if not previous_records else "warm"


def _aggregate_run_temperature(temperatures: tuple[str, ...]) -> str:
    observed = {
        temperature for temperature in temperatures if temperature in {"cold", "warm", "mixed"}
    }
    if not observed:
        return "mixed"
    if len(observed) == 1:
        return next(iter(observed))
    return "mixed"


def _workflow_run_temperature() -> str:
    return "mixed"


def _repository_state() -> dict[str, object]:
    revision = _run_text_command(("git", "rev-parse", "--verify", "HEAD"))
    dirty = bool(_run_text_command(("git", "status", "--short")))
    return {
        "repository_revision": revision or "unknown",
        "repository_dirty": dirty,
    }


def _suite_tool_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "arrayscope": _package_version("ArrayScope"),
        "numpy": getattr(np, "__version__", ""),
        "py-spy": _run_text_command(("py-spy", "--version"))
        if shutil.which("py-spy")
        else "unavailable",
        "perf": _run_text_command(("perf", "--version")) if shutil.which("perf") else "unavailable",
    }
    for package in ("PySide6", "pyqtgraph", "vispy", "nibabel"):
        versions[package] = _package_version(package)
    return versions


def _package_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _run_text_command(command: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5)
    except Exception:
        return ""
    output = (completed.stdout or completed.stderr or "").strip()
    return output.splitlines()[0] if output else ""


def _suite_child_args(argv: tuple[str, ...]) -> tuple[str, ...]:
    blocked = {"--profile-suite", "--print-py-spy-command", "--py-spy-native", "--include-cprofile"}
    result: list[str] = []
    skip_next = False
    for _index, arg in enumerate(tuple(argv)):
        if skip_next:
            skip_next = False
            continue
        if arg in blocked:
            skip_next = arg == "--profile-suite"
            continue
        if arg.startswith("--profile-suite="):
            continue
        if arg in {"--jsonl", "--profiler-type", "--profiler-artifact"}:
            skip_next = True
            continue
        if arg.startswith(("--jsonl=", "--profiler-type=", "--profiler-artifact=")):
            continue
        result.append(arg)
    return tuple(result)


def _suite_profiler_backends(argv: tuple[str, ...]) -> tuple[str, ...]:
    backend = "all"
    args = tuple(argv)
    for index, arg in enumerate(args):
        if arg == "--backend" and index + 1 < len(args):
            backend = str(args[index + 1])
            break
        if str(arg).startswith("--backend="):
            backend = str(arg).split("=", 1)[1]
            break
    if backend == "all":
        return PROFILE_DEFAULT_BACKENDS
    return (backend,)


def _suite_args_for_backend(argv: tuple[str, ...], backend: str) -> tuple[str, ...]:
    args = tuple(argv)
    result: list[str] = []
    replaced = False
    skip_next = False
    for index, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--backend":
            result.extend(("--backend", str(backend)))
            replaced = True
            skip_next = index + 1 < len(args)
            continue
        if str(arg).startswith("--backend="):
            result.append(f"--backend={backend}")
            replaced = True
            continue
        result.append(str(arg))
    if not replaced:
        result.extend(("--backend", str(backend)))
    return tuple(result)


def _py_spy_native_requested(argv: tuple[str, ...]) -> bool:
    return any(str(arg) == "--py-spy-native" for arg in tuple(argv))


def _include_cprofile_requested(argv: tuple[str, ...]) -> bool:
    return any(str(arg) == "--include-cprofile" for arg in tuple(argv))


def _py_spy_filtered_args(argv: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(arg for arg in tuple(argv) if str(arg) != "--py-spy-native")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the realistic tiled montage + FFT/shift/iFFT profiling workflow"
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help="Dataset path; defaults to the bundled realistic NIfTI",
    )
    parser.add_argument(
        "--synthetic-scene",
        choices=("geometry", "complex-phase"),
        default=None,
        help="Use a deterministic visual registration chart instead of --data",
    )
    parser.add_argument(
        "--synthetic-shape",
        type=_parse_synthetic_shape,
        default=(192, 256, 40),
        metavar="HEIGHTxWIDTHxDEPTH",
    )
    parser.add_argument("--backend", choices=("pyqtgraph", "vispy", "wgpu", "all"), default="all")
    parser.add_argument(
        "--wgpu-present-method",
        choices=("bitmap", "screen", "auto"),
        default="bitmap",
        help=(
            "wgpu backend presentation path: bitmap (default), the "
            "native-Wayland screen swapchain (fails loudly if unavailable), "
            "or auto (screen where possible; effective method recorded); "
            "ignored by other backends"
        ),
    )
    parser.add_argument(
        "--wgpu-power-preference",
        choices=("low-power", "high-performance"),
        default="low-power",
        help="adapter selection for a fresh wgpu profile process",
    )
    parser.add_argument(
        "--texture-codec",
        choices=("off", "auto"),
        default="off",
        help="wgpu display texture codec experiment; ignored by other backends",
    )
    parser.add_argument("--jsonl", default=None, help="Optional JSONL metrics output")
    parser.add_argument("--trace", default=None, help="Structured event trace JSONL output")
    parser.add_argument(
        "--screenshot-dir", default=None, help="Optional directory for phase screenshots"
    )
    parser.add_argument(
        "--screenshot-interval-s",
        type=float,
        default=0.0,
        help="Periodic framebuffer/physical-truth sampling interval; requires --screenshot-dir",
    )
    parser.add_argument(
        "--physical-sample-seed",
        type=int,
        default=None,
        help=(
            "Replay the bounded CPU-reference pixel samples; the default varies "
            "per run and is recorded in JSONL"
        ),
    )
    parser.add_argument(
        "--verbose-tile-trace",
        action="store_true",
        help=(
            "Record per-scroll-delivery tile source, backend identity, LOD, lifecycle, "
            "revision, and age snapshots in scroll_tile_trace (diagnostic; large JSONL)"
        ),
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=INTERACTION_SETTLE_HARD_LIMIT_S,
        help=(
            "Per-interaction settlement deadline in seconds; values above the "
            f"repository hard limit ({INTERACTION_SETTLE_HARD_LIMIT_S:g} s) are capped"
        ),
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=0,
        help="Optional tile cap for local smoke runs; 0 means full dim 2",
    )
    parser.add_argument(
        "--scroll-max-tiles",
        type=int,
        default=60,
        help="Tile cap for scroll/zoom-pan phases; default 60",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=0,
        help="Montage columns; 0 uses the application's viewport-maximizing automatic layout",
    )
    parser.add_argument("--load-mode", choices=("app", "native"), default="app")
    parser.add_argument(
        "--session-fixture",
        default=str(DEFAULT_SESSION_FIXTURE),
        help="Portable file-view session restored before profiling; empty disables restore",
    )
    parser.add_argument(
        "--stages",
        action="append",
        default=[],
        help=(
            "Comma-separated workflow phases to include. Repeat to accumulate. "
            "Omit to run all stages. Supported: " + ", ".join(PROFILE_MONTAGE_STAGES)
        ),
    )
    parser.add_argument(
        "--skip-stages",
        action="append",
        default=[],
        help=(
            "Comma-separated workflow phases to skip. Repeat to accumulate. "
            "Skip wins over --stages when names overlap."
        ),
    )
    parser.add_argument(
        "--print-py-spy-command",
        action="store_true",
        help="Print an external py-spy command for this invocation and exit",
    )
    parser.add_argument(
        "--profile-suite",
        default=None,
        help="Run plain JSONL, py-spy raw, and perf record into this directory",
    )
    parser.add_argument(
        "--include-cprofile",
        action="store_true",
        help="Include cProfile call-count attribution; slower and not timing evidence",
    )
    parser.add_argument(
        "--py-spy-native",
        action="store_true",
        help="Include native stacks in py-spy artifacts; useful diagnostically but not pacing evidence",
    )
    parser.add_argument("--profiler-type", default="plain", help=argparse.SUPPRESS)
    parser.add_argument("--profiler-artifact", action="append", default=[], help=argparse.SUPPRESS)
    return parser


_CHROME_PROBE = """\
import numpy as np
from PySide6 import QtWidgets
from arrayscope.window.main import ArrayScopeWindow

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
win = ArrayScopeWindow(np.zeros((16, 16, 4), dtype=np.float32), filepath="chrome-probe.nii")
win.show()
for _ in range(60):
    app.processEvents()
win.resize(900, 800)
for _ in range(60):
    app.processEvents()
viewport = win.img_view.graphicsView.viewport()
print(f"CHROME {win.width() - viewport.width()} {win.height() - viewport.height()}")
"""


def _measure_window_chrome() -> tuple[int, int] | None:
    """Pixels of window that are not viewport, measured on a real compositor.

    Chrome is what stands between the viewport a session pins and the window
    size that holds it.  It is constant in the window size (verified across
    900-1000 px) but NOT portable across platforms: the same build measures
    153x209 under Wayland and 153x235 under the offscreen QPA plugin, so this
    has to run somewhere the screenshot run would also run.
    """

    try:
        with headless_display(log_dir=None) as display:
            completed = subprocess.run(
                (sys.executable, "-c", _CHROME_PROBE),
                check=False,
                capture_output=True,
                text=True,
                env=display.child_environment(),
                timeout=180,
            )
    except Exception:
        return None
    for line in reversed((completed.stdout or "").splitlines()):
        if line.startswith("CHROME "):
            _, width, height = line.split()
            return int(width), int(height)
    return None


def _managed_weston_output_size(session_fixture: str | Path | None) -> tuple[int, int]:
    """Size the compositor output to the window the session will restore.

    Screen evidence is exact-window evidence, and a Wayland client cannot ask
    where it sits on screen.  So we read the session FIRST, size the sole
    output to the window it asks for, and run with no panel and no window
    decoration — then the window fills the output and one capture is the
    window.  This replaces the kiosk shell, which achieved the same identity
    by force-fullscreening and in doing so changed viewport aspect and
    montage layout.

    What the session pins is its VIEWPORT; the window size it recorded is
    that viewport plus the chrome of whatever build recorded it.  Sizing the
    output from the recorded window size silently loses the identity as soon
    as a bar changes height — the fixture captured at `0f11a22` asks for
    1400x940 while today's chrome needs 1400x948, so every exact-window
    capture was short by 8 px of the window it claimed to be.  Derive the
    output the way the app derives the window instead: pinned viewport plus
    measured chrome, falling back to the recorded size when the probe cannot
    run.
    """

    default = (1400, 940)
    payload: dict = {}
    if session_fixture is not None and str(session_fixture).strip():
        try:
            payload = json.loads(Path(session_fixture).read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            payload = {}
    try:
        recorded = payload["panels"]["window_size"]
        recorded_size = (max(1, int(recorded[0])), max(1, int(recorded[1])))
    except (KeyError, TypeError, ValueError):
        recorded_size = default
    try:
        viewport_shape = payload["viewport"]["viewport_shape"]
        viewport_height, viewport_width = (int(viewport_shape[0]), int(viewport_shape[1]))
    except (KeyError, TypeError, ValueError):
        return recorded_size
    chrome = _measure_window_chrome()
    if chrome is None:
        print(
            "profile: could not measure window chrome; sizing the screenshot output from "
            f"the session's recorded window {recorded_size[0]}x{recorded_size[1]}, which is "
            "only exact-window evidence if this build's chrome matches the recording's",
            file=sys.stderr,
        )
        return recorded_size
    measured = (max(1, viewport_width + chrome[0]), max(1, viewport_height + chrome[1]))
    if measured != recorded_size:
        print(
            f"profile: session records window {recorded_size[0]}x{recorded_size[1]} for viewport "
            f"{viewport_width}x{viewport_height}, but this build's chrome "
            f"({chrome[0]}x{chrome[1]}) needs {measured[0]}x{measured[1]}; sizing the "
            "screenshot output to the latter so the capture is still the window",
            file=sys.stderr,
        )
    return measured


def _managed_weston_requested(args) -> bool:
    return bool(
        args.backend == "wgpu"
        and str(args.wgpu_present_method) in {"screen", "auto"}
        and args.screenshot_dir
        and not is_headless_display()
    )


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    stages = _resolve_profile_stages(
        include_stages=_parse_stage_flags(tuple(args.stages)),
        skip_stages=_parse_stage_flags(tuple(args.skip_stages)),
    )

    if args.print_py_spy_command:
        filtered = tuple(
            arg
            for arg in (argv if argv is not None else sys.argv[1:])
            if arg != "--print-py-spy-command"
        )
        print(py_spy_command(filtered))
        return 0
    if args.profile_suite:
        source_argv = tuple(argv if argv is not None else sys.argv[1:])
        return run_profile_suite(source_argv, args.profile_suite)
    if _managed_weston_requested(args):
        source_argv = tuple(argv if argv is not None else sys.argv[1:])
        return run_in_headless_display(
            (
                sys.executable,
                "-m",
                "arrayscope.tools.profile_montage_workflow",
                *source_argv,
            ),
            log_dir=args.screenshot_dir,
            output_size=_managed_weston_output_size(args.session_fixture),
            # Screen evidence owns its compositor: never share a batch's.
            exact_window=True,
        )

    jsonl = None if args.jsonl is None else Path(args.jsonl)
    if jsonl is not None and jsonl.exists():
        jsonl.unlink()
    trace = None if args.trace is None else Path(args.trace)
    if trace is not None:
        from arrayscope.core.trace import configure_trace

        configure_trace(trace)
    all_records: list[dict[str, object]] = []
    try:
        for backend in PROFILE_DEFAULT_BACKENDS if args.backend == "all" else (args.backend,):
            all_records.extend(
                run_profile_montage_workflow(
                    data_path=args.data,
                    backend=backend,
                    wgpu_present_method=str(args.wgpu_present_method),
                    wgpu_power_preference=str(args.wgpu_power_preference),
                    texture_codec=str(args.texture_codec),
                    jsonl=jsonl,
                    timeout_s=bounded_interaction_settle_timeout_s(args.timeout_s),
                    max_tiles=None if args.max_tiles <= 0 else args.max_tiles,
                    scroll_max_tiles=args.scroll_max_tiles,
                    columns=None if args.columns <= 0 else args.columns,
                    load_mode=args.load_mode,
                    profiler_type=args.profiler_type,
                    profiler_artifact_paths=tuple(args.profiler_artifact or ()),
                    stages=stages,
                    screenshot_dir=args.screenshot_dir,
                    screenshot_interval_s=float(args.screenshot_interval_s),
                    session_fixture=None
                    if not str(args.session_fixture).strip()
                    else args.session_fixture,
                    verbose_tile_trace=bool(args.verbose_tile_trace),
                    synthetic_scene=args.synthetic_scene,
                    synthetic_shape=tuple(args.synthetic_shape),
                    physical_sample_seed=args.physical_sample_seed,
                )
            )
    finally:
        if trace is not None:
            from arrayscope.core.trace import close_trace

            close_trace()
    print(_workflow_timing_summary(tuple(all_records)), end="")
    failed = [
        record
        for record in all_records
        if bool(record.get("r8_gate_applicable", False))
        and not bool(record.get("r8_gate_passed", False))
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
