"""The module histogram executor must never absorb kernel-declined work.

`_HISTOGRAM_EXECUTOR` is a non-daemon pool that ``concurrent.futures`` joins
at interpreter exit — work landing there after the kernel closes admission
keeps the whole process alive past ``kernel_shutdown`` complete. In the
application, a submitter is always installed and a terminal decline
("admission", i.e. the controller closed for shutdown) must drop the refresh
instead of rescheduling it outside the kernel's lifecycle. The executor
remains the compute path only for standalone widgets with no submitter.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.core.frame_targets import WorkStart


@pytest.fixture
def controller_factory(qt_app):
    from pyqtgraph.Qt import QtCore

    from arrayscope.display.histogram_controller import HistogramDisplayController

    owners = []

    def build(submitter):
        owner = QtCore.QObject()
        if submitter is not None:
            owner._submit_background_task = submitter
        owners.append(owner)
        return HistogramDisplayController(owner)

    yield build
    for owner in owners:
        owner.deleteLater()


def _request():
    from arrayscope.display.histogram_plot import HistogramPlotRequest

    data = np.zeros((512, 512), dtype=np.float32)
    return HistogramPlotRequest(
        data=data,
        source_identity=(id(data), data.shape, str(data.dtype)),
        histogram_bounds=(0.0, 1.0),
        visible_value_span=1.0,
        pixel_extent=200.0,
        generation=1,
        view_signature=("test",),
    )


def _capture_executor(monkeypatch):
    from arrayscope.display import histogram_controller

    submitted = []

    def submit(fn, *args, **kwargs):
        submitted.append(fn)
        return SimpleNamespace(
            add_done_callback=lambda callback: None,
            done=lambda: True,
            cancel=lambda: False,
        )

    monkeypatch.setattr(histogram_controller._HISTOGRAM_EXECUTOR, "submit", submit)
    return submitted


def test_terminally_declined_submission_never_reaches_module_executor(
    controller_factory, monkeypatch
):
    submitted = _capture_executor(monkeypatch)
    controller = controller_factory(
        lambda fn, *, on_done, key: WorkStart(False, "admission")
    )

    controller._schedule_histogram_job(_request())

    assert submitted == [], (
        "kernel-declined histogram work escaped to the non-daemon module "
        "executor; it would keep the process alive past kernel shutdown"
    )
    # The decline must also release the running slot so a later refresh can
    # schedule again instead of being wedged behind a job that never ran.
    assert controller._running_request_signature is None


def test_widget_without_submitter_still_computes_on_module_executor(
    controller_factory, monkeypatch
):
    submitted = _capture_executor(monkeypatch)
    controller = controller_factory(None)

    controller._schedule_histogram_job(_request())

    assert len(submitted) == 1
