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


def test_wgpu_noise_filter_drops_only_the_known_message(wgpu_logger, caplog):
    install_wgpu_log_noise_filter()

    with caplog.at_level(logging.WARNING, logger="wgpu"):
        wgpu_logger.warning("Unable to find extension: VK_EXT_physical_device_drm")
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
    """The real hook disables the textinput category AND installs the filter.

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
            "print('filtered', not logging.getLogger('wgpu').filter(\n"
            "    logging.LogRecord('wgpu', logging.WARNING, 'probe', 0,\n"
            "    'Unable to find extension: VK_EXT_physical_device_drm', None, None)))\n",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.stdout.splitlines()[-3:] == ["before True", "after False", "filtered True"]
