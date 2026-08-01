"""Known-harmless terminal noise is dropped; everything else still prints.

Covers arrayscope.app.log_noise: the wgpu-native message filter and the Qt
logging-category rules, plus the wiring through the one pre-QApplication
hook (launch._prepare_qt_environment) that every app path goes through.
"""

import importlib.util
import logging
import os
import subprocess
import sys

import pytest

from arrayscope.app.log_noise import (
    _WGPU_NOISE_FILTER,
    install_wgpu_log_noise_filter,
)


@pytest.fixture
def wgpu_logger():
    """The logger wgpu-py emits native records on, restored afterwards."""
    logger = logging.getLogger("wgpu")
    yield logger
    logger.removeFilter(_WGPU_NOISE_FILTER)


def test_wgpu_noise_filter_drops_only_the_known_messages(wgpu_logger, caplog):
    install_wgpu_log_noise_filter()

    with caplog.at_level(logging.WARNING, logger="wgpu"):
        wgpu_logger.warning("Unable to find extension: VK_EXT_physical_device_drm")
        wgpu_logger.warning(
            "Detected skylake derivative running on mesa i915. "
            "Clears to srgb textures will use manual shader clears."
        )
        # Same workaround notice, different chip family: the tail is what the
        # filter matches, so a machine that names another derivative is quiet too.
        wgpu_logger.warning(
            "Detected tigerlake derivative running on mesa i915. "
            "Clears to srgb textures will use manual shader clears."
        )
        wgpu_logger.warning("Unable to find extension: VK_KHR_swapchain")
        wgpu_logger.error("device lost")

    messages = [record.getMessage() for record in caplog.records]
    assert messages == ["Unable to find extension: VK_KHR_swapchain", "device lost"]


def test_wgpu_noise_filter_installs_once(wgpu_logger):
    baseline = len(wgpu_logger.filters)
    install_wgpu_log_noise_filter()
    install_wgpu_log_noise_filter()
    assert len(wgpu_logger.filters) == baseline + 1


@pytest.mark.skipif(
    importlib.util.find_spec("pyqtgraph") is None,
    reason="pyqtgraph is not installed",
)
def test_prepare_qt_environment_silences_the_known_noise():
    """The real hook disables the textinput category.

    It deliberately does NOT create the "wgpu" logger -- see
    ``test_prepare_qt_environment_leaves_import_wgpu_working``.

    Subprocess because logging-category rules are process-global mutable
    state; in-process the rule change would outlive the test.
    """
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.pop("QT_LOGGING_RULES", None)  # an outer rule would skew "before"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import logging\n"
            "from pyqtgraph.Qt import QtCore\n"
            "from arrayscope.app.launch import _prepare_qt_environment\n"
            "category = QtCore.QLoggingCategory('qt.qpa.wayland.textinput')\n"
            "print('before', category.isWarningEnabled())\n"
            "_prepare_qt_environment()\n"
            "print('after', category.isWarningEnabled())\n"
            "print('uncreated', 'wgpu' not in logging.Logger.manager.loggerDict)\n",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.stdout.splitlines()[-3:] == ["before True", "after False", "uncreated True"]


@pytest.mark.skipif(
    importlib.util.find_spec("wgpu") is None,
    reason="wgpu is not installed",
)
def test_prepare_qt_environment_leaves_import_wgpu_working():
    """The launch hook must not break ``import wgpu`` (2026-07-31 regression).

    wgpu-py installs its own ``Logger`` subclass by wrapping
    ``getLogger("wgpu")`` in ``setLoggerClass`` and then asserting the result
    is that subclass.  ``getLogger`` hands back an already-registered logger
    unchanged, so a hook that merely *touched* the name before the import
    left ``import wgpu`` raising ``AssertionError`` -- which the AUTO backend
    probe reported as "no usable wgpu device" on a perfectly good GPU.

    Subprocess because the damage is done at first import: once wgpu is
    imported in this process the ordering can never be reproduced again.
    """
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import logging\n"
            "from arrayscope.app.launch import _prepare_qt_environment\n"
            "_prepare_qt_environment()\n"
            "import wgpu\n"
            "from arrayscope.app.log_noise import install_wgpu_log_noise_filter\n"
            "print('imported', bool(wgpu))\n"
            "print('installs', install_wgpu_log_noise_filter())\n"
            "print('filtered', not logging.getLogger('wgpu').filter(\n"
            "    logging.LogRecord('wgpu', logging.WARNING, 'probe', 0,\n"
            "    'Unable to find extension: VK_EXT_physical_device_drm', None, None)))\n",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.stdout.splitlines()[-3:] == ["imported True", "installs True", "filtered True"]


@pytest.mark.skipif(
    importlib.util.find_spec("wgpu") is None,
    reason="wgpu is not installed",
)
def test_shared_device_installs_the_filter_before_requesting_an_adapter():
    """The deferred install lands on the path that emits the message.

    Nothing else would attach it: the launch hook now declines to create the
    logger, so the filter has to arrive with the import that creates it.
    """
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import logging\n"
            "from arrayscope.app.launch import _prepare_qt_environment\n"
            "_prepare_qt_environment()\n"
            "import arrayscope.display.wgpu_imageview2d as view\n"
            "try:\n"
            "    view._shared_wgpu_device()\n"
            "except Exception:\n"
            "    pass\n"
            "print('attached', any(\n"
            "    type(f).__name__ == '_KnownNoiseFilter'\n"
            "    for f in logging.getLogger('wgpu').filters))\n",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.stdout.splitlines()[-1] == "attached True"
