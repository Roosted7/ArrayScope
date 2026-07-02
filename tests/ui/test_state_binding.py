"""The declarative ViewState binder (roadmap Y3)."""

from types import SimpleNamespace

from arrayscope.ui.state_binding import ViewStateBinder


class _Widget:
    def __init__(self):
        self.blocked = []
        self.value = None

    def blockSignals(self, flag):
        self.blocked.append(bool(flag))


def test_sync_applies_only_changed_values_with_signals_blocked():
    binder = ViewStateBinder()
    widget = _Widget()

    def apply(value):
        assert widget.blocked and widget.blocked[-1] is True
        widget.value = value

    binder.bind("value", read=lambda win: win.state, apply=apply, widgets=(widget,))
    window = SimpleNamespace(state=1)

    assert binder.sync(window) == 1
    assert widget.value == 1
    assert widget.blocked == [True, False]

    # Unchanged value: no re-apply.
    assert binder.sync(window) == 0
    window.state = 2
    assert binder.sync(window) == 1
    assert widget.value == 2


def test_named_sync_limits_the_pass():
    binder = ViewStateBinder()
    applied = []
    binder.bind("a", read=lambda win: win.a, apply=lambda value: applied.append(("a", value)))
    binder.bind("b", read=lambda win: win.b, apply=lambda value: applied.append(("b", value)))
    window = SimpleNamespace(a=1, b=1)

    assert binder.sync(window, names=("a",)) == 1
    assert applied == [("a", 1)]
    # The skipped binding still applies on the next full pass.
    assert binder.sync(window) == 1
    assert applied == [("a", 1), ("b", 1)]


def test_forget_forces_reapply_after_widget_side_drift():
    binder = ViewStateBinder()
    applied = []
    binder.bind("value", read=lambda win: win.state, apply=applied.append)
    window = SimpleNamespace(state=5)
    binder.sync(window)
    assert binder.sync(window) == 0
    binder.forget()
    assert binder.sync(window) == 1
    assert applied == [5, 5]


def test_rebinding_a_name_resets_its_change_detection():
    binder = ViewStateBinder()
    applied = []
    binder.bind("value", read=lambda win: win.state, apply=applied.append)
    window = SimpleNamespace(state=7)
    binder.sync(window)
    binder.bind("value", read=lambda win: win.state, apply=applied.append)
    assert binder.sync(window) == 1
    assert applied == [7, 7]


def test_dead_widget_does_not_break_sync():
    class DeadWidget:
        def blockSignals(self, flag):
            raise RuntimeError("wrapped C++ object deleted")

    binder = ViewStateBinder()
    applied = []
    binder.bind("value", read=lambda win: win.state, apply=applied.append, widgets=(DeadWidget(),))
    assert binder.sync(SimpleNamespace(state=3)) == 1
    assert applied == [3]


def test_on_demand_bindings_run_only_when_named():
    binder = ViewStateBinder()
    applied = []
    binder.bind("full", read=lambda win: win.state, apply=lambda value: applied.append(("full", value)))
    binder.bind(
        "fast",
        read=lambda win: win.state,
        apply=lambda value: applied.append(("fast", value)),
        on_demand=True,
    )
    window = SimpleNamespace(state=1)

    assert binder.sync(window) == 1
    assert applied == [("full", 1)]
    assert binder.sync(window, names=("fast",)) == 1
    assert applied == [("full", 1), ("fast", 1)]
