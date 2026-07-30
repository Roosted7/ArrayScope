"""Field-scale WGPU gate for a crop that sharpens the demanded LOD."""

from __future__ import annotations

import json
from time import monotonic, sleep

import numpy as np
import pytest

from arrayscope.tools.framebuffer_reference import assert_wgpu_frame_matches_cpu_reference
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_S
from tests.gpu_interaction.conftest import wait_for_qt_condition
from tests.ui.helpers import frame_session_settled, restore_default_backend, use_wgpu_backend

pytestmark = pytest.mark.gpu_interaction


def _wait_for_background_refinement(app, predicate, *, timeout_s: float = 20.0) -> bool:
    """Deadlock guard for non-blocking refinement after first-pixel acceptance."""

    deadline = monotonic() + float(timeout_s)
    while monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        sleep(0.001)
    app.processEvents()
    return bool(predicate())


def _crop_state_current(win, axis: int, start: int, previous_session_id: int) -> bool:
    session = win.renderer._frame_session
    ranges = tuple(session.view_state.axis_range_indices or ())
    indices = tuple(ranges[axis] or ()) if axis < len(ranges) else ()
    return bool(
        int(session.session_id) != int(previous_session_id)
        and indices
        and int(indices[0]) == int(start)
    )


@pytest.mark.parametrize("image_axes", [(0, 1), (1, 0)], ids=("yx", "xy"))
def test_second_display_axis_crop_presents_resident_lod_before_refining(
    qt_app,
    image_axes,
    tmp_path,
):
    """The retained LOD is an atomic first frame; T remains governed work."""

    from arrayscope.display.backend_contract import image_view_backend_capabilities
    from arrayscope.window.main import ArrayScopeWindow

    settings = use_wgpu_backend(
        extra_settings={
            "montage_quality_policy": "resident",
            "resident_crop_rebind": True,
        }
    )
    data = np.random.default_rng(20260730).standard_normal((336, 336, 272), dtype=np.float32)
    win = ArrayScopeWindow(data)
    trace_open = False
    try:
        assert image_view_backend_capabilities(win.img_view).name == "wgpu"
        win.resize(900, 850)
        win.show()
        state = win.view_state.with_axis_flipped(1, True)
        if image_axes != (0, 1):
            state = state.with_image_axes(*image_axes)
        state = state.with_montage_axis(
            2,
            columns=16,
            indices=tuple(range(272)),
            text=":",
        )
        win._set_view_state(state)
        assert win.view_state.image_axes == image_axes
        assert win.view_state.axis_flipped == (False, True, False)
        win.update_image_view()
        assert _wait_for_background_refinement(
            qt_app,
            lambda: frame_session_settled(win),
            timeout_s=20.0,
        )

        first_crop = win.view_state.with_axis_range(
            0,
            indices=tuple(range(80, 280)),
            text="80:280",
        )
        previous_session_id = int(win.renderer._frame_session.session_id)
        win._apply_slice_state(
            0,
            first_crop,
            reason="slice-range",
            interactive=True,
            immediate_axis_only=False,
        )
        assert _wait_for_background_refinement(
            qt_app,
            lambda: (
                _crop_state_current(win, 0, 80, previous_session_id) and frame_session_settled(win)
            ),
            timeout_s=20.0,
        )

        epoch = win.resource_governor.begin_ui_observation_epoch()
        from arrayscope.core.trace import close_trace, configure_trace

        trace_path = tmp_path / f"second-axis-crop-{image_axes[0]}{image_axes[1]}.jsonl"
        configure_trace(trace_path)
        trace_open = True
        second_crop = win.view_state.with_axis_range(
            1,
            indices=tuple(range(50, 250)),
            text="50:250",
        )
        previous_session_id = int(win.renderer._frame_session.session_id)
        win._apply_slice_state(
            1,
            second_crop,
            reason="slice-range",
            interactive=True,
            immediate_axis_only=False,
        )

        assert wait_for_qt_condition(
            qt_app,
            lambda: (
                _crop_state_current(win, 1, 50, previous_session_id)
                and win.renderer._frame_session.lifecycle.presented_tiles
                and win.renderer._frame_session.resident_crop_rebind_stats.get("rebound", 0) == 272
            ),
            timeout_s=INTERACTION_SETTLE_HARD_LIMIT_S,
        )
        session = win.renderer._frame_session
        stats = dict(session.resident_crop_rebind_stats)
        assert stats.get("considered") == 272
        assert stats.get("crop_local_subset") == 272
        assert stats.get("rebound") == 272
        assert stats.get("pages_not_resident", 0) == 0

        assert _wait_for_background_refinement(
            qt_app,
            lambda: (
                session.required_first_pixels_presented()
                and len(win.img_view.tileTruthPhysicalRows()) == 272
            ),
            timeout_s=20.0,
        )

        observation_count, max_callback_ms, current = (
            win.resource_governor.ui_observation_epoch_evidence(epoch)
        )
        assert current
        assert observation_count > 0
        assert max_callback_ms <= 50.0, (
            f"retained crop handoff exceeded R5: {max_callback_ms:.3f} ms"
        )

        assert _wait_for_background_refinement(
            qt_app,
            lambda: frame_session_settled(win),
            timeout_s=20.0,
        )
        close_trace()
        trace_open = False
        events = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        preview_acks = [
            event
            for event in events
            if event.get("kind") == "backend_ack"
            and event.get("accepted") is True
            and event.get("quality") in {"preview", "fallback"}
        ]
        target_acks = [
            event
            for event in events
            if event.get("kind") == "backend_ack"
            and event.get("accepted") is True
            and event.get("quality") == "exact"
        ]
        assert len({int(event["tile"]) for event in preview_acks}) == 272
        assert target_acks
        assert max(int(event["sequence"]) for event in preview_acks) < min(
            int(event["sequence"]) for event in target_acks
        ), "target acknowledgements overtook the atomic retained-preview handoff"
        assert int(getattr(win.renderer, "_montage_stall_assertions", 0) or 0) == 0
        assert_wgpu_frame_matches_cpu_reference(win)
    finally:
        if trace_open:
            from arrayscope.core.trace import close_trace

            close_trace()
        win.close()
        restore_default_backend(settings)
        for _ in range(30):
            qt_app.processEvents()
