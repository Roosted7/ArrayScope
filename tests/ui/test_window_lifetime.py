"""A closed window must actually be freed, not merely hidden.

Measured 2026-08-02, offscreen: building windows in one process and closing
each one left ``QApplication.allWidgets()`` growing by 657 per build with no
drop.  Two separate causes had to hold at once.

The first is in the product and is what this file guards.  ``OperationStackDock``
builds five proxy buttons that are never added to a layout, so they had no Qt
parent -- each one a top-level window in its own right, which the destruction of
the ArrayScope window cannot reach.  Three of them connect ``clicked`` to a
lambda.  Qt drops a connection when a *QObject* receiver is destroyed, but a
lambda has no receiver: the sender owns it, from C++, where the cycle collector
cannot see it.  A leave-one-out over the twenty orphaned widgets named exactly
those three as the pin, and only those three -- the two proxies wired to bound
methods did not pin, and neither did any pyqtgraph menu.  So the closure held
the dock, the dock held the window, and nothing in Python could tell.

The second cause is in the *measurement*, and is why this test flushes by hand.
``QApplication.processEvents()`` does not deliver ``DeferredDelete``; only
exiting an event loop or ``sendPostedEvents(None, DeferredDelete)`` does.  A
probe that closes, calls ``deleteLater()`` and pumps ``processEvents()`` never
destroys the widget tree at all, so it reports a total leak whatever the product
does.  That is also what pytest-qt's ``addWidget`` teardown does today, which is
why the suite carries dead widgets even with the pin gone.

Both matter, and neither is sufficient: on this file's 32-test sibling, the pin
fix alone moved the end-of-run widget count 9982 -> 9812, the flush alone moved
it 9982 -> 9590, and the two together moved it to 0 with RSS 788 -> 562 MB.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.window import ArrayScopeWindow


def _flush():
    """Deliver every pending ``DeferredDelete``, whoever posted it.

    Global on purpose: it collects windows that an earlier test closed and
    pytest-qt's teardown left pending, which is why the count below takes its
    baseline *after* a flush rather than before.
    """

    app = QtWidgets.QApplication.instance()
    for _ in range(2):
        app.processEvents()
        # processEvents() alone never delivers this; see the module docstring.
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    gc.collect()


def _destroy(win):
    """Close ``win`` and give Qt and Python every chance to free it."""

    win.close()
    win.deleteLater()
    _flush()


def test_closed_window_is_freed(qtbot):
    """The window's Python object dies with it, so nothing it owns stays resident."""

    # Deliberately not registered with ``qtbot.addWidget``: this test destroys
    # the window itself, and pytest-qt's teardown would then call ``close()``
    # on a wrapper whose C++ object is already gone, burying the real failure
    # under a RuntimeError and an "item was not torn down properly" cascade.
    win = ArrayScopeWindow(np.arange(8 * 9, dtype=float).reshape(8, 9))
    alive = weakref.ref(win)

    _destroy(win)
    del win
    gc.collect()

    assert alive() is None


def test_operation_proxy_buttons_have_a_parent(qtbot):
    """The five layout-less proxies are owned by the dock, not by the process.

    Stated as ownership rather than as a widget count because that is the
    property the fix establishes: a parented widget is destroyed with the tree
    whether or not anything is connected to it, so this keeps holding if the
    lambdas above are ever rewritten as bound methods.
    """

    win = ArrayScopeWindow(np.arange(8 * 9, dtype=float).reshape(8, 9))
    qtbot.addWidget(win)
    dock = win.operation_dock

    proxies = {
        "save_button": dock.save_button,
        "load_button": dock.load_button,
        "export_button": dock.export_button,
        "save_view_button": dock.save_view_button,
        "load_view_button": dock.load_view_button,
    }
    orphans = sorted(name for name, button in proxies.items() if button.parent() is None)
    assert not orphans, f"proxy buttons with no parent outlive the window: {orphans}"
    assert all(dock.isAncestorOf(button) for button in proxies.values())


def test_no_widget_outlives_a_closed_window(qtbot):
    """Nothing at all survives -- the orphans held 274 widgets between them."""

    _flush()
    before = len(QtWidgets.QApplication.allWidgets())
    win = ArrayScopeWindow(np.arange(8 * 9, dtype=float).reshape(8, 9))  # owned here, as above
    assert len(QtWidgets.QApplication.allWidgets()) > before

    _destroy(win)
    del win
    gc.collect()

    assert len(QtWidgets.QApplication.allWidgets()) == before
