"""Startup warm-up gating for the image-backend factory.

The GPU device init (~2 s) and shader compilation (~165 ms) are pre-built on a
background thread while a file loads, so the first image is not stalled behind
them (see ``warm_image_backend_async``).  These tests pin the *decision* logic
(which backends warrant a warm, and the one-shot/non-blocking spawn contract)
without touching a real GPU — the device build itself is stubbed out.
"""

from __future__ import annotations

import types

import pytest

from arrayscope.app.settings_state import ImageRenderingBackendChoice
from arrayscope.display import image_view_factory as fac


def _settings(choice: ImageRenderingBackendChoice):
    return types.SimpleNamespace(image_rendering_backend=choice)


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        (ImageRenderingBackendChoice.WGPU, True),  # explicit pin: always warm
        (ImageRenderingBackendChoice.PYQTGRAPH, False),  # never a wasted device
        (ImageRenderingBackendChoice.VISPY, False),
    ],
)
def test_should_warm_follows_explicit_pin(choice, expected):
    assert fac._should_warm_wgpu(_settings(choice)) is expected


def test_should_warm_auto_skips_offscreen(monkeypatch):
    monkeypatch.setattr(fac.platform, "system", lambda: "Linux")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    assert fac._should_warm_wgpu(_settings(ImageRenderingBackendChoice.AUTO)) is False


def test_should_warm_auto_skips_non_linux(monkeypatch):
    monkeypatch.setattr(fac.platform, "system", lambda: "Darwin")
    assert fac._should_warm_wgpu(_settings(ImageRenderingBackendChoice.AUTO)) is False


def test_should_warm_auto_on_linux_display(monkeypatch):
    monkeypatch.setattr(fac.platform, "system", lambda: "Linux")
    monkeypatch.setenv("QT_QPA_PLATFORM", "wayland")
    assert fac._should_warm_wgpu(_settings(ImageRenderingBackendChoice.AUTO)) is True


def test_warm_is_one_shot_and_non_blocking(monkeypatch):
    # Stub the real device build so no GPU is touched; count invocations.
    calls = []

    def fake_warm():
        calls.append(1)

    import arrayscope.display.wgpu_imageview2d as ivm

    monkeypatch.setattr(ivm, "warm_wgpu_backend", fake_warm)
    # Reset the one-shot guard for a clean run.
    monkeypatch.setattr(fac, "_warm_started", False)

    settings = _settings(ImageRenderingBackendChoice.WGPU)
    fac.warm_image_backend_async(settings)
    fac.warm_image_backend_async(settings)  # second call must not spawn again

    # Join any warm thread we spawned so the assertion is deterministic.
    for thread in list(__import__("threading").enumerate()):
        if thread.name == "arrayscope-wgpu-warm":
            thread.join(timeout=5)
    assert calls == [1]


def test_warm_skips_when_backend_not_wgpu(monkeypatch):
    called = []
    import arrayscope.display.wgpu_imageview2d as ivm

    monkeypatch.setattr(ivm, "warm_wgpu_backend", lambda: called.append(1))
    monkeypatch.setattr(fac, "_warm_started", False)

    fac.warm_image_backend_async(_settings(ImageRenderingBackendChoice.PYQTGRAPH))
    assert called == []
    assert fac._warm_started is False
