import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def process_events(qtbot, count=8):
    """``count`` x 10 ms of event loop, unconditionally.  Almost never right.

    This is a sleep, not a wait: it does not ask the product anything, so it is
    simultaneously too long on an idle machine and too short on a busy one.
    Six hundred and seventy-two call sites of it were ~90 s of static sleep per
    pass over the suite, and eleven of the fourteen tests that failed under
    three concurrent suites on 2026-08-01 were waits of this shape.

    Use :func:`settle_idle` or :func:`settle_quiescent` instead -- they wait on
    what the product publishes about its own progress, so they cost what the
    machine needs and fail only on a lost wakeup.

    What is left here is the one case those cannot express: a claim *about a
    time window*, where the product behaviour under test is itself a throttle.
    ``tests/ui/test_interactive_render_coalescing.py`` is the honest example and
    its module docstring gives the reasoning.  If a new site wants this, say in
    a comment which throttle it is bounded by.
    """

    for _ in range(count):
        qtbot.wait(10)


def auto_confirm_message_boxes(monkeypatch, *, answer=True):
    """Answer every ``QMessageBox.question`` without showing it.

    Destroying a stored definition asks first in both library managers, and an
    offscreen modal never gets an answer -- the run just stops. Tests that
    exercise a destructive button pick the answer up front.
    """

    from pyqtgraph.Qt import QtWidgets

    standard = QtWidgets.QMessageBox.StandardButton
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        staticmethod(lambda *args, **kwargs: standard.Yes if answer else standard.No),
    )


def clear_arrayscope_settings():
    from pyqtgraph.Qt import QtCore

    settings = QtCore.QSettings()
    settings.clear()
    settings.sync()


def view_action(win, text):
    for action in win.menuBar().actions():
        if action.text() == "View":
            for child in action.menu().actions():
                if child.text() == text:
                    return child
    raise AssertionError(f"View action not found: {text}")


def panel_body(panel):
    return panel.body


def assert_panel_invariants(win, name, expected_location):
    from pyqtgraph.Qt import QtWidgets

    from arrayscope.window.panels import PanelLocation

    panel = win.panel_manager._panels_by_name[name]
    assert panel.location == expected_location
    if expected_location == PanelLocation.HIDDEN:
        assert panel.dialog is None
        assert not panel.dock.isVisible()
        assert panel.body is not None
    elif expected_location == PanelLocation.DETACHED:
        assert panel.dialog is not None
        assert (
            panel.dialog.findChild(type(panel.body)) is panel.body
            or panel.body.parent() is not None
        )
        assert not panel.dock.isVisible()
    elif expected_location == PanelLocation.DOCKED:
        assert panel.dialog is None
        assert panel.dock.isVisible()
        assert QtWidgets.QDockWidget.widget(panel.dock) is panel.body


def _caller_label(depth=2):
    """Name the waiting line for a report, from inside a wrapper."""

    import inspect

    frame = inspect.currentframe()
    for _ in range(depth):
        if frame is None:
            return "settle"
        frame = frame.f_back
    return "settle" if frame is None else f"{frame.f_code.co_name}:{frame.f_lineno}"


def _refuse_closed_window(win) -> None:
    """A closed window settles instantly, so waiting on one is never a wait.

    ``closeEvent`` cancels the canvas-preserve chain, drops the dock-visibility
    queue and shuts the kernel down, so all three of the machines below answer
    "nothing outstanding" and every wait on that window returns at once. Four
    sites had this and none of them could fail: each closed its window, opened a
    *second* one to prove a setting survives a restart, and then settled the
    first -- so the window the test existed for was never waited on at all.

    That is a test bug rather than a product state, and it is invisible from the
    outside (the run is green and 141 ms faster), so it fails loudly here rather
    than being fixed one call site at a time.
    """

    if getattr(win, "_closing", False):
        raise AssertionError(
            "settle called on a closed window: it cannot wait, because closing "
            "cancels the preserve chain, empties the dock queue and shuts the "
            "kernel down. Settle the window the assertion is about -- usually "
            "the one opened after this one closed."
        )


def _window_settled(win) -> bool:
    """Everything the window was asked to do has been done.

    Three machines have to agree, because the fixed pauses this replaces sat
    after actions that drive any of them: the frame session (the render and its
    tiles), the queued dock-visibility requests, and the canvas-preserve retry
    chain.  Each publishes its own answer already; none of them is a clock.
    """

    manager = getattr(win, "layout_manager", None)
    if manager is not None:
        if manager.queued_dock_visibility_requests:
            return False
        if manager.canvas_preserver.diagnostics().active:
            return False
    return frame_session_settled(win)


def settle_layout(win, predicate=None, *, label=None):
    """Wait for the dock/panel machinery only, saying nothing about the frame.

    The weakest of the three: queued dock-visibility requests dispatched and
    the canvas-preserve retry chain finished.  It exists because a test may
    legitimately hide a dock in a state where the frame session is *not* going
    to settle -- the hidden-inspection ROI tests blank the committed frame on
    purpose -- and there ``settle_idle`` would be asking for something the test
    has arranged never to happen.
    """

    from tests.settle import deliver_posted_events, settle_window

    _refuse_closed_window(win)
    label = label or _caller_label()
    manager = win.layout_manager

    def settled():
        if manager.queued_dock_visibility_requests:
            return False
        if manager.canvas_preserver.diagnostics().active:
            return False
        return predicate is None or predicate()

    if settled():
        deliver_posted_events()
    settle_window(win, settled, label=label)


def settle_idle(win, predicate=None, *, label=None):
    """Wait until the window has finished what the last action asked of it.

    This is the replacement for ``process_events(qtbot, count=N)`` wherever the
    pause was covering a real hand-off: building the window, a render, an
    operation, a dock transition.  It waits on what the product publishes about
    its own progress, so it takes as long as the box needs and fails only on a
    lost wakeup.

    When the state already looks settled, one event-loop slice runs and the
    question is asked again.  That is not a pause -- it is the difference
    between "done" and "not started yet", and without it a wait placed
    immediately after an action whose continuation is still sitting in the
    event queue could only ever say yes.
    """

    from tests.settle import deliver_posted_events, settle_window

    _refuse_closed_window(win)
    label = label or _caller_label()

    def settled():
        return _window_settled(win) and (predicate is None or predicate())

    if settled():
        deliver_posted_events()
    settle_window(win, settled, label=label)


def settle_quiescent(win, predicate=None, *, label=None):
    """:func:`settle_idle`, and nothing is armed to do any more work.

    For claims of the shape "and that scheduled nothing further" -- a coalesced
    resize that must not emit a second range change, a retarget that must stay
    one transaction.  The fixed pause was giving a deferred continuation its
    chance to fire before the assertion looked; ``window_work_in_flight``
    answers the same question from the product, counting a pending render only
    while a timer is armed to flush it.
    """

    from tests.settle import deliver_posted_events, settle_window, window_work_in_flight

    _refuse_closed_window(win)
    label = label or _caller_label()

    def quiet():
        if window_work_in_flight(win):
            return False
        return _window_settled(win) and (predicate is None or predicate())

    if quiet():
        deliver_posted_events()
    settle_window(win, quiet, label=label)


def assert_size_close(actual, expected, tolerance=1):
    assert abs(actual.width() - expected.width()) <= tolerance
    assert abs(actual.height() - expected.height()) <= tolerance


# --- Live backend-window harness --------------------------------------------


def use_pyqtgraph_backend(extra_settings=None):
    """Point QSettings at the PyQtGraph backend; returns the settings object."""

    from pyqtgraph.Qt import QtCore

    from arrayscope.app.settings_state import ImageRenderingBackendChoice

    clear_arrayscope_settings()
    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.PYQTGRAPH.value)
    for key, value in (extra_settings or {}).items():
        settings.setValue(key, value)
    settings.sync()
    return settings


def use_wgpu_backend(extra_settings=None):
    """Point QSettings at the explicit wgpu backend; returns the settings object."""

    from pyqtgraph.Qt import QtCore

    from arrayscope.app.settings_state import ImageRenderingBackendChoice

    require_wgpu_adapter()
    clear_arrayscope_settings()
    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.WGPU.value)
    for key, value in (extra_settings or {}).items():
        settings.setValue(key, value)
    settings.sync()
    return settings


def restore_default_backend(settings):
    from arrayscope.app.settings_state import ImageRenderingBackendChoice

    settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.PYQTGRAPH.value)
    settings.sync()


def make_backend_window(qtbot, data, *, backend, require_gpu_atlas=False):
    """Build an ArrayScopeWindow on the given backend, refusing a fallback one.

    The environment question is already settled above this line:
    ``require_wgpu_adapter`` skips when the machine has no adapter, so *past*
    that gate the machine demonstrably has one. A window that then comes back
    on PyQtGraph is not an environment without a GPU — it is the factory having
    caught a failing surface build and degraded, which is a defect in whatever
    made the build fail.

    This used to skip, with the message ``wgpu backend unavailable in this Qt
    environment``, which is a sentence the run had just disproved. Being a
    call-phase skip it read as one more ``s`` among seventy and nothing
    surfaced it; it was observed on three different tests across eleven full
    passes, always under foreign load, and the cause was never once written
    down because the only record of it was a status-bar toast. So it fails now,
    and it fails carrying the traceback the factory recorded, because the run
    where it happens is the only run that can see why.
    """

    import pytest

    from arrayscope.display.backend_contract import image_view_backend_capabilities
    from arrayscope.window import ArrayScopeWindow

    if backend == "wgpu":
        require_wgpu_adapter()
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        refuse_fallback_backend(win.img_view, backend, context="this test")
    except pytest.fail.Exception:
        win.close()
        raise
    if require_gpu_atlas:
        assert image_view_backend_capabilities(win.img_view).tile_residency_kind == "gpu_atlas"
    return win


def refuse_fallback_backend(view, backend, *, context) -> None:
    """Fail -- never skip -- when a built surface is not the one asked for.

    The guard body of :func:`make_backend_window`, for the tests whose window
    construction that helper cannot own: a real ``QApplication`` harness with no
    ``qtbot``, a fixture that yields, a window that must be sized and shown
    before it is registered. Every caller must already be past an adapter gate
    -- ``require_wgpu_adapter`` here, or the autouse
    ``_require_parametrized_wgpu_adapter`` in ``tests/ui/conftest.py`` for a
    case parametrized ``backend="wgpu"``. Past one, PyQtGraph where wgpu was
    asked for is the factory having caught a failing surface build, not a
    machine without a GPU, so it is a defect and fails as one.

    Ten call sites hand-rolled this as ``pytest.skip`` and the argument against
    that is in :func:`make_backend_window`: it says an environment thing about a
    run that has already disproved it, and it says it as one more ``s`` in a
    summary nobody reads twice. Four of the ten were skipping for two reasons at
    once and only the gate above can separate them, so adding it is what made
    them safe to sharpen.

    ``pytest.fail(pytrace=False)`` rather than ``assert``: the message carries
    the factory's recorded traceback, which is the only account of why the
    build failed, and pytest's own would bury it under these three lines.
    """

    import pytest

    from arrayscope.display.backend_contract import require_image_view_backend

    try:
        require_image_view_backend(view, backend, context=context)
    except RuntimeError as exc:
        pytest.fail(str(exc), pytrace=False)


def require_wgpu_adapter() -> None:
    """Skip an explicit WGPU test when no Vulkan adapter exists locally."""

    import contextlib

    import pytest

    wgpu = pytest.importorskip("wgpu")
    from wgpu.backends.wgpu_native.extras import set_instance_extras

    with contextlib.suppress(RuntimeError):
        set_instance_extras(backends=["Vulkan"])
    try:
        wgpu.gpu.request_adapter_sync(power_preference="low-power")
    except Exception as exc:
        pytest.skip(f"no WGPU Vulkan adapter: {exc}")


def frame_session_settled(win) -> bool:
    """The settle triple every live gate builds on; committed-frame currency
    checks stay with the caller."""

    session = getattr(win.renderer, "_frame_session", None)
    if session is None:
        return False
    return bool(
        session.visible_plan_complete()
        and not win.montage_tile_evaluation_controller.is_busy()
        and session.required_target_settled()
    )


def apply_plane(win, index, *, reason):
    win._set_view_state(win.view_state.with_slice(0, index))
    win.render(reason=reason)


def plane_settled(win, index) -> bool:
    frame = getattr(win, "_committed_display_frame", None)
    if frame is None:
        return False
    if int(frame.geometry.view_state.slice_indices[0]) != int(index):
        return False
    return frame_session_settled(win)


def committed_value(win, view_x, view_y):
    """Probe the committed frame the way live hover does (render.py)."""

    geometry = win.renderer.display_geometry
    if geometry is None:
        return None
    context = geometry.context_for_view_point(float(view_x), float(view_y))
    if context is None:
        return None
    return win.renderer._hover_value_from_display(context.mapping)


def give_generous_work_area(win, width=4096, height=4096):
    """Pin a work area far larger than any fixture window.

    The offscreen QPA reports an 800x800 screen -- SMALLER than several of
    these fixtures' own windows.  The screen clamp (added so an opening dock
    can no longer push the window off the desktop) would therefore fire in
    situations a real desktop cannot produce, and the affected tests would be
    measuring the platform rather than the behaviour they exist to pin.

    Tests about canvas-preservation MECHANICS call this so the clamp stays
    out of the way.  The clamp's own behaviour is pinned separately, with a
    deliberately small work area, in test_dock_screen_fit.py.
    """

    from pyqtgraph.Qt import QtCore

    win.layout_manager._available_size_override = QtCore.QSize(int(width), int(height))
