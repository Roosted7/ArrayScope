"""Applying the theme to the QApplication: what may be skipped, what may not.

``apply_theme_to_qapplication`` runs on every window construction, and
``QApplication.setStyleSheet`` restyles every live widget in the process --
so its cost grows with the number of open windows (measured 202 ms for the
first window and 567 ms for the fifth). It is therefore skipped when the
sheet it would install is the one already in force. These tests pin the two
halves of that bargain: the skip really happens, and nothing that must
restyle stops restyling.

The same bargain, and the same two halves, apply to the repaint that follows
it. ``_apply_theme_choice`` calls ``_retheme_presentation_surfaces``, which
clears the icon cache and then visits every widget and every action in the
process. That is a *theme switch's* job; on a window construction it can
never help the window being built, because ``__init__`` applies the theme
long before ``_build_window_ui`` exists to give it any surfaces. Measured
over a process building windows back to back, it grew a build from 116 ms to
180 ms by the fourteenth; guarded, the same builds stay flat at ~100 ms.
"""

import numpy as np


# Imported inside each test, not here, which is what every other file in
# ``tests/ui`` does (37 of them, none at module scope).  The reason is
# ``tests/app/test_theme_settings.py``: it loads ``arrayscope.app.theme`` and
# ``arrayscope.app.settings_state`` straight from their paths and REPLACES the
# ``sys.modules`` entries with fresh copies.  Any module that already bound
# names from the originals keeps the old classes, so two enum members print
# identically and compare unequal.  Importing the package at collection time
# put the originals in place before that loader ran and broke a sibling test
# that had nothing to do with this change.
def _theme():
    from arrayscope.app import theme

    return theme


def _data():
    return np.arange(32 * 32, dtype=float).reshape(32, 32)


def _sheet_writes(app, monkeypatch):
    """Count ``setStyleSheet`` calls on ``app`` without suppressing them."""
    calls = []
    original = type(app).setStyleSheet

    def counted(self, sheet):
        calls.append(sheet)
        return original(self, sheet)

    monkeypatch.setattr(type(app), "setStyleSheet", counted)
    return calls


def test_reapplying_the_same_theme_does_not_restyle_the_application(qt_app, monkeypatch):
    from arrayscope.app.theme import ThemeChoice

    _theme().apply_theme_to_qapplication(qt_app, ThemeChoice.DARK)
    installed = qt_app.styleSheet()
    assert installed, "dark theme should install a stylesheet"

    writes = _sheet_writes(qt_app, monkeypatch)
    _theme().apply_theme_to_qapplication(qt_app, ThemeChoice.DARK)

    assert writes == [], "an unchanged theme must not re-set the application stylesheet"
    assert qt_app.styleSheet() == installed
    # Assert on the sheet the QApplication carries rather than on the tokens
    # object it stores: the suite re-imports ``arrayscope.*`` in places, so two
    # copies of this module can be live at once and an identity check would
    # fall back to DARK_TOKENS and pass even when the theme is wrong. The hex
    # strings are equal across copies, and the sheet is what decides the look.
    assert _theme().DARK_TOKENS.window in installed
    assert _theme().LIGHT_TOKENS.window not in installed


def test_a_theme_change_still_restyles_the_application(qt_app, monkeypatch):
    from arrayscope.app.theme import ThemeChoice

    _theme().apply_theme_to_qapplication(qt_app, ThemeChoice.DARK)
    dark_sheet = qt_app.styleSheet()

    writes = _sheet_writes(qt_app, monkeypatch)
    _theme().apply_theme_to_qapplication(qt_app, ThemeChoice.LIGHT)

    assert len(writes) == 1, "a changed theme must re-set the application stylesheet"
    light_sheet = qt_app.styleSheet()
    assert light_sheet != dark_sheet
    assert _theme().LIGHT_TOKENS.window in light_sheet
    assert _theme().DARK_TOKENS.window not in light_sheet

    # ...and switching back restores the original sheet exactly, so the skip
    # cannot latch the application into whichever theme it saw first.
    _theme().apply_theme_to_qapplication(qt_app, ThemeChoice.DARK)
    assert qt_app.styleSheet() == dark_sheet


def test_a_cleared_stylesheet_is_repaired_by_the_next_apply(qt_app):
    from arrayscope.app.theme import ThemeChoice

    # The guard compares against the sheet the QApplication actually carries
    # rather than against the tokens it stores, so a sheet dropped by anything
    # else (another apply path, a test, a dialog) is reinstalled rather than
    # skipped as "already current".
    _theme().apply_theme_to_qapplication(qt_app, ThemeChoice.DARK)
    dark_sheet = qt_app.styleSheet()

    qt_app.setStyleSheet("")
    _theme().apply_theme_to_qapplication(qt_app, ThemeChoice.DARK)

    assert qt_app.styleSheet() == dark_sheet


def _icon_named_buttons(app):
    from pyqtgraph.Qt import QtWidgets

    from arrayscope.ui.icons import _ICON_NAME_PROPERTY

    return [
        widget
        for widget in app.allWidgets()
        if isinstance(widget, QtWidgets.QAbstractButton) and widget.property(_ICON_NAME_PROPERTY)
    ]


def _icon_image(button):
    return button.icon().pixmap(18, 18).toImage()


def test_a_second_window_does_not_retint_every_icon_in_the_process(qt_app, monkeypatch):
    from arrayscope.ui import icons as icons_module
    from arrayscope.ui.icons import _ICON_NAME_PROPERTY
    from arrayscope.window.main import ArrayScopeWindow

    first = ArrayScopeWindow(_data())
    try:
        by_name = {
            button.property(_ICON_NAME_PROPERTY): _icon_image(button)
            for button in _icon_named_buttons(qt_app)
        }
        assert by_name, "the first window should own icon-bearing buttons to compare against"

        retints = []
        original = icons_module.refresh_icon_tints
        monkeypatch.setattr(
            icons_module,
            "refresh_icon_tints",
            lambda: (retints.append(1), original())[1],
        )

        second = ArrayScopeWindow(_data())
        try:
            assert retints == [], (
                "building a window must not re-tint every icon in the process: the "
                "theme has not changed, and this window's own surfaces do not exist "
                "yet when the theme is applied"
            )
            # Same bargain as the stylesheet above: skipping is only legitimate
            # if the new window's icons come out right anyway, which they do
            # because the cache was left holding icons tinted for this palette.
            second_icons = {
                button.property(_ICON_NAME_PROPERTY): _icon_image(button)
                for button in _icon_named_buttons(qt_app)
                if button.window() is second
            }
            assert second_icons, "the second window should own icon-bearing buttons"
            shared = set(second_icons) & set(by_name)
            assert shared, "the two windows should share at least one icon name"
            for name in sorted(shared):
                assert second_icons[name] == by_name[name], f"icon {name!r} differs from window one"
        finally:
            second.close()
    finally:
        first.close()


def test_a_theme_change_still_retints_the_icons(qt_app):
    from arrayscope.app.theme import ThemeChoice
    from arrayscope.window.main import ArrayScopeWindow

    restore = qt_app.styleSheet()
    win = ArrayScopeWindow(_data())
    try:
        win._apply_theme_choice(ThemeChoice.DARK, persist=False)
        buttons = [button for button in _icon_named_buttons(qt_app) if button.window() is win]
        assert buttons, "the window should own icon-bearing buttons"
        dark = [_icon_image(button) for button in buttons]

        win._apply_theme_choice(ThemeChoice.LIGHT, persist=False)
        light = [_icon_image(button) for button in buttons]

        # Icons are baked pixmaps, so nothing but the re-tint changes them. Every
        # one of them, not merely some: a guard that let the repaint run for only
        # part of the process would leave dark glyphs on a light window.
        assert all(a != b for a, b in zip(dark, light, strict=True))

        # ...and back, so the skip cannot latch the process into the first theme
        # it happened to see.
        win._apply_theme_choice(ThemeChoice.DARK, persist=False)
        assert [_icon_image(button) for button in buttons] == dark
    finally:
        win.close()
        qt_app.setStyleSheet(restore)


def test_a_second_window_is_themed_without_reinstalling_the_stylesheet(qt_app, monkeypatch):
    from arrayscope.window.main import ArrayScopeWindow

    first = ArrayScopeWindow(_data())
    try:
        installed = qt_app.styleSheet()
        themed = first.palette().window().color().name()

        writes = _sheet_writes(qt_app, monkeypatch)
        second = ArrayScopeWindow(_data())
        try:
            assert writes == [], "a second window must not restyle the whole application"
            assert qt_app.styleSheet() == installed
            # The saving is only legitimate if the new window is themed anyway:
            # the sheet lives on the QApplication, so widgets built afterwards
            # resolve it when they are polished. Comparing against the first
            # window pins "same as a window that did install the sheet"; the
            # sheet check keeps that from passing on two identically unthemed
            # windows.
            assert second.palette().window().color().name() == themed
            assert themed in installed
        finally:
            second.close()
    finally:
        first.close()
