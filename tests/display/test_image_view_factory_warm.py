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

from arrayscope.app.settings_state import ImageRenderingBackendChoice, TextureCodecChoice
from arrayscope.display import image_view_factory as fac


def _settings(
    choice: ImageRenderingBackendChoice,
    texture_codec: TextureCodecChoice = TextureCodecChoice.OFF,
):
    return types.SimpleNamespace(image_rendering_backend=choice, texture_codec=texture_codec)


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        (ImageRenderingBackendChoice.WGPU, True),  # explicit pin: always warm
        (ImageRenderingBackendChoice.PYQTGRAPH, False),  # never a wasted device
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


def test_retired_persisted_choice_warms_like_auto(monkeypatch):
    """ADR 0061: a stale ``vispy`` string is AUTO input, not a non-wgpu pin.

    ``settings=None`` is the file-open path, which reads the persisted value
    straight from QSettings and never sees ``AppSettingsState``'s migration.
    Without normalizing here too, a migrated user silently loses the ~2 s
    device warm and compiles CPU display kernels for a wgpu session.
    """

    monkeypatch.setattr(fac.platform, "system", lambda: "Linux")
    monkeypatch.setenv("QT_QPA_PLATFORM", "wayland")
    stale = types.SimpleNamespace(
        image_rendering_backend="vispy", texture_codec=TextureCodecChoice.OFF
    )
    assert fac._image_backend_choice_value(stale) == ImageRenderingBackendChoice.AUTO.value
    assert fac._should_warm_wgpu(stale) is True
    assert fac.cpu_display_backend_likely(stale) is False


def test_auto_resolver_prefers_wgpu_and_caches_probe(monkeypatch):
    monkeypatch.setattr(fac, "_auto_resolution_cache", None)
    monkeypatch.setattr(fac.platform, "system", lambda: "Linux")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    calls = []

    def probe():
        calls.append("probe")
        return "test Vulkan adapter"

    monkeypatch.setattr(fac, "_probe_wgpu_device", probe)

    first = fac.resolve_auto_backend_choice()
    second = fac.resolve_auto_backend_choice()

    assert first == second
    assert first[0] is ImageRenderingBackendChoice.WGPU
    assert "wgpu device" in first[1]
    assert calls == ["probe"]


@pytest.mark.parametrize(
    ("platform_name", "qt_platform", "reason"),
    [
        ("Windows", "", "no reference performance traces"),
        ("Linux", "offscreen", "offscreen Qt platform"),
    ],
)
def test_auto_resolver_uses_pyqtgraph_without_a_supported_display(
    monkeypatch,
    platform_name,
    qt_platform,
    reason,
):
    monkeypatch.setattr(fac, "_auto_resolution_cache", None)
    monkeypatch.setattr(fac.platform, "system", lambda: platform_name)
    if qt_platform:
        monkeypatch.setenv("QT_QPA_PLATFORM", qt_platform)
    else:
        monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setattr(
        fac,
        "_probe_wgpu_device",
        lambda: (_ for _ in ()).throw(AssertionError("device probe must not run")),
    )

    choice, explanation = fac.resolve_auto_backend_choice()

    assert choice is ImageRenderingBackendChoice.PYQTGRAPH
    assert reason in explanation


def test_auto_resolver_falls_back_to_pyqtgraph_when_wgpu_probe_fails(monkeypatch):
    monkeypatch.setattr(fac, "_auto_resolution_cache", None)
    monkeypatch.setattr(fac.platform, "system", lambda: "Linux")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setattr(fac, "_probe_wgpu_device", lambda: None)

    choice, explanation = fac.resolve_auto_backend_choice()

    assert choice is ImageRenderingBackendChoice.PYQTGRAPH
    assert explanation == "no usable wgpu device"


def test_auto_wgpu_construction_failure_falls_back_to_pyqtgraph(qt_app, monkeypatch):
    from arrayscope.display.backends import wgpu as wgpu_backend

    monkeypatch.setattr(
        fac,
        "resolve_auto_backend_choice",
        lambda: (ImageRenderingBackendChoice.WGPU, "test device"),
    )

    class BrokenWgpuSurface:
        def __init__(self, **_kwargs):
            raise RuntimeError("injected device loss")

    monkeypatch.setattr(wgpu_backend, "WgpuSurface", BrokenWgpuSurface)
    messages = []
    view = fac.create_image_view(
        _settings(ImageRenderingBackendChoice.AUTO),
        notify=messages.append,
    )
    try:
        assert view.surface.capabilities.name == "pyqtgraph"
        assert messages == [
            "Image rendering backend: wgpu | test device",
            "wgpu renderer unavailable; using PyQtGraph (injected device loss)",
        ]
    finally:
        view.close()


def test_bc_bulk_warm_requires_wgpu_and_enabled_texture_codec():
    assert fac.wgpu_texture_compression_likely(
        _settings(ImageRenderingBackendChoice.WGPU, TextureCodecChoice.AUTO)
    )
    assert not fac.wgpu_texture_compression_likely(
        _settings(ImageRenderingBackendChoice.WGPU, TextureCodecChoice.OFF)
    )
    assert not fac.wgpu_texture_compression_likely(
        _settings(ImageRenderingBackendChoice.PYQTGRAPH, TextureCodecChoice.AUTO)
    )


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


def test_signatureless_and_settingsless_paths_never_enable_lossy_codec(monkeypatch):
    """No settings-less / attribute-less path may silently enable lossy BC pools.

    The G7 verdict (commit 4683bc3c) restored lossy texture compression to OFF
    by default. Two fallback sites must honour that even on a wgpu-capable
    machine (``bc_available=True``): the ``WgpuImageView2D.__init__`` signature
    default and the ``create_image_view`` construction default when ``settings``
    lacks a ``texture_codec`` attribute. Both must resolve to the raw "off"
    executor mode, never lossy "on"/"auto".
    """
    import inspect

    from arrayscope.app.settings_state import (
        normalize_texture_codec_choice,
        texture_codec_executor_mode,
    )
    from arrayscope.display.wgpu_imageview2d import WgpuImageView2D

    # (1) WgpuImageView2D.__init__ signature default.
    sig_default = inspect.signature(WgpuImageView2D.__init__).parameters["texture_codec"].default
    sig_choice = normalize_texture_codec_choice(sig_default)
    assert texture_codec_executor_mode(sig_choice, bc_available=True) == "off"

    # (2) create_image_view construction default: an explicit wgpu pin on a
    # settings object that has no ``texture_codec`` attribute at all. Stub the
    # surface so no GPU is touched; capture the codec value it is handed.
    captured = {}

    class _StubSurface:
        def __init__(self, *, texture_codec, **_kwargs):
            captured["texture_codec"] = texture_codec

    import arrayscope.display.backends.wgpu as wgpu_backend

    monkeypatch.setattr(wgpu_backend, "WgpuSurface", _StubSurface)

    settings = types.SimpleNamespace(image_rendering_backend=ImageRenderingBackendChoice.WGPU)
    fac.create_image_view(settings)
    factory_choice = normalize_texture_codec_choice(captured["texture_codec"])
    assert texture_codec_executor_mode(factory_choice, bc_available=True) == "off"


def test_warm_skips_when_backend_not_wgpu(monkeypatch):
    called = []
    import arrayscope.display.wgpu_imageview2d as ivm

    monkeypatch.setattr(ivm, "warm_wgpu_backend", lambda: called.append(1))
    monkeypatch.setattr(fac, "_warm_started", False)

    fac.warm_image_backend_async(_settings(ImageRenderingBackendChoice.PYQTGRAPH))
    assert called == []
    assert fac._warm_started is False
