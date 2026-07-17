"""Journey oracle: LOD demand tracks the camera after every gesture.

Field defect 2026-07-18 (both backends, screenshots + diagnostics): after a
hard zoom-in the session still reported the FIT-view texels-per-pixel
(texpp ~5) with "demanded LOD level is resident and presented" — the demand
was never re-derived from the new camera, so no finer work was ever wanted
and the screen stayed blocky forever. The unit suite could not see this:
every mechanism (ladder, policy, replan) was individually correct; the
broken thing was the TRAJECTORY gesture -> session.view_range ->
selected_lod_factor. This gate drives the real widget path.
"""

from __future__ import annotations

import numpy as np
import pytest

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS

# Build-time budget for the cold montage fill (see the churn harness):
# the 5 s interaction cap governs per-gesture waits only.
_FILL_TIMEOUT_MS = 30000

from arrayscope.display.lod import select_lod_demand


def _demand_for_current_camera(win):
    session = win.renderer._frame_session
    view = win.img_view.getView()
    (x0, x1), (y0, y1) = view.viewRange()
    return select_lod_demand(
        ((x0, x1), (y0, y1)),
        session.viewport_shape,
        session.plan.tile_shape,
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "OPEN (2026-07-18, both backends, field screenshots + this offscreen "
        "repro): a zoom under AUTO camera intent rebuilds the session with "
        "the FIT range instead of the live camera (probe: camera "
        "[[706,835],[470,556]] vs session.view_range ((0,1541),(-1,1028))) — "
        "the auto_like planning override in _montage_viewport_plan erases a "
        "live zoomed camera, so LOD demand freezes at the fit level and "
        "quality never upgrades. Fix owner: scope the AUTO/FIT replay "
        "override to extent-change/no-camera cases."
    ),
)
def test_zoom_in_rederives_lod_demand(qtbot):
    from tests.ui.helpers import make_backend_window, restore_default_backend, use_vispy_backend

    settings = use_vispy_backend(extra_settings={"montage_quality_policy": "resident"})
    data = np.random.default_rng(5).normal(size=(256, 256, 24)).astype(np.float32)
    win = make_backend_window(qtbot, data)
    try:
        win.resize(900, 700)
        win.show()
        state = win.view_state.with_montage_axis(
            2, columns=None, indices=tuple(range(24)), text="0:24"
        )
        win._set_view_state(state)
        win.update_image_view()
        session = win.renderer._frame_session
        qtbot.waitUntil(
            lambda: bool(getattr(win.renderer._frame_session, "display_committed", False)),
            timeout=_FILL_TIMEOUT_MS,
        )
        session = win.renderer._frame_session
        fit_desired = int(session.lod_policy_decision.demand.desired_level)
        assert fit_desired > 0, "fit view of a large montage should demand a reduced level"

        # Hard zoom into ~1.5 tiles: true texels-per-pixel drops below 1,
        # so the demanded level must become finer (smaller) than the fit's.
        calls = {"plan": 0, "retarget_viewport": 0, "viewport_only": 0, "retarget_mv": 0, "apply_mv": 0, "bridge": 0}
        bridge = getattr(win.img_view, "viewport_bridge", None) or getattr(win.renderer, "viewport_bridge", None)
        orig_rmv = win.renderer.retarget_montage_viewport
        win.renderer.retarget_montage_viewport = lambda *a, **k: (calls.__setitem__("retarget_mv", calls["retarget_mv"] + 1), orig_rmv(*a, **k))[1]
        orig_amv = win.renderer.apply_montage_viewport_retarget
        win.renderer.apply_montage_viewport_retarget = lambda *a, **k: (calls.__setitem__("apply_mv", calls["apply_mv"] + 1), orig_amv(*a, **k))[1]
        renderer = win.renderer
        orig_plan = renderer._montage_viewport_plan
        renderer._montage_viewport_plan = lambda *a, **k: (calls.__setitem__("plan", calls["plan"] + 1), orig_plan(*a, **k))[1]
        orig_only = renderer._try_update_montage_viewport_only
        renderer._try_update_montage_viewport_only = lambda *a, **k: (calls.__setitem__("viewport_only", calls["viewport_only"] + 1), orig_only(*a, **k))[1]
        session_now = win.renderer._frame_session
        orig_rv = session_now.retarget_viewport
        session_now.retarget_viewport = lambda *a, **k: (calls.__setitem__("retarget_viewport", calls["retarget_viewport"] + 1), orig_rv(*a, **k))[1]

        view = win.img_view.getView()
        (x0, x1), (y0, y1) = view.viewRange()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        span_x = (x1 - x0) / 12
        span_y = (y1 - y0) / 12
        view.setRange(
            xRange=(cx - span_x / 2, cx + span_x / 2),
            yRange=(cy - span_y / 2, cy + span_y / 2),
            padding=0,
        )

        def demand_fresh() -> bool:
            current = win.renderer._frame_session
            wanted = int(_demand_for_current_camera(win).desired_level)
            return int(current.lod_policy_decision.demand.desired_level) == wanted

        try:
            qtbot.waitUntil(demand_fresh, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        except Exception:
            # Decisive probe: is only the SIGNAL dead, or the handler chain?
            manual = {"worked": None, "error": None}
            try:
                win.renderer._on_view_range_changed()
                qtbot.wait(500)
                manual["worked"] = demand_fresh()
            except Exception as exc:
                manual["error"] = repr(exc)
            print("manual handler call:", manual)
            frame = getattr(win, "_committed_display_frame", None)
            vs = getattr(frame, "value_source", None)
            print("bridge inputs: frame=", type(frame).__name__ if frame is not None else None,
                  "value_source=", type(vs).__name__ if vs is not None else None,
                  "has_payloads=", hasattr(vs, "payloads"),
                  "commit_active=", getattr(win.renderer, "_montage_presentation_commit_active", None),
                  "montage_axis=", getattr(win.view_state, "montage_axis", None))
            print("view object id stable:", id(win.img_view.getView()))
            current = win.renderer._frame_session
            view = win.img_view.getView()
            print("camera:", view.viewRange())
            print("session.view_range:", current.view_range)
            print("calls:", calls)
            print("session desired:", current.lod_policy_decision.demand.desired_level,
                  "wanted:", _demand_for_current_camera(win).desired_level,
                  "session_id:", current.session_id)
            raise
        current = win.renderer._frame_session
        assert int(current.lod_policy_decision.demand.desired_level) < fit_desired, (
            "zoom-in did not re-derive a finer LOD demand: the session still "
            f"wants level {current.lod_policy_decision.demand.desired_level} "
            f"(fit was {fit_desired}) — the 2026-07-18 frozen-demand defect"
        )
    finally:
        win.close()
        restore_default_backend(settings)
