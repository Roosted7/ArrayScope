import numpy as np

from tests.ui.helpers import clear_arrayscope_settings as _clear_arrayscope_settings
from tests.ui.helpers import process_events as _process_events


def _menu(win, text):
    for action in win.menuBar().actions():
        if action.text() == text:
            return action.menu()
    raise AssertionError(f"menu not found: {text}")


def _submenu_action(win, menu_text, submenu_text, action_text):
    menu = _menu(win, menu_text)
    for action in menu.actions():
        if action.text() == submenu_text:
            submenu = action.menu()
            for child in submenu.actions():
                if child.text() == action_text:
                    return child
    raise AssertionError(f"action not found: {menu_text}/{submenu_text}/{action_text}")


def _menu_action(win, menu_text, action_text):
    menu = _menu(win, menu_text)
    for action in menu.actions():
        if action.text() == action_text:
            return action
    raise AssertionError(f"action not found: {menu_text}/{action_text}")


def test_performance_menu_exists(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        assert _menu(win, "Performance") is not None
        assert _submenu_action(win, "Performance", "Memory Profile", "Balanced") is not None
        assert _submenu_action(win, "Performance", "Render Memory Budget", "128 MiB") is not None
        assert _menu_action(win, "Performance", "Use Less Memory") is not None
        assert _menu_action(win, "Performance", "Use More Memory") is not None
    finally:
        win.close()


def test_selecting_fft_workers_updates_settings(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import FFTWorkersChoice
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        _submenu_action(win, "Performance", "FFT Workers", "2").trigger()
        _process_events(qtbot)
        assert win.app_settings.fft_workers == FFTWorkersChoice.TWO
        assert win.compute_policy.fft_workers_visible == 2
        assert win.compute_policy.fft_workers_stage == 2
        assert win.compute_policy.fft_workers_tile == 1
        montage_workers = win.montage_tile_evaluation_controller.diagnostics().max_workers
        assert 1 <= montage_workers <= win.compute_policy.montage_tile_workers
        assert (
            win.stage_evaluation_controller.diagnostics().max_workers
            == win.compute_policy.stage_workers
        )
    finally:
        win.close()


def test_render_memory_budget_persists_through_settings(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import MemoryProfileChoice
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        _submenu_action(win, "Performance", "Render Memory Budget", "1024 MiB").trigger()
        _submenu_action(win, "Performance", "Memory Profile", "Custom").trigger()
        _process_events(qtbot)
        assert win.app_settings.render_memory_budget_mb == 1024
        assert win.app_settings.memory_profile == MemoryProfileChoice.CUSTOM
        assert win.renderer._memory_policy().visible_render_budget_bytes == 1024 * 1024 * 1024
    finally:
        win.close()

    second = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(second)
    try:
        _process_events(qtbot)
        assert second.app_settings.render_memory_budget_mb == 1024
        assert second.app_settings.memory_profile == MemoryProfileChoice.CUSTOM
    finally:
        second.close()


def test_selecting_pyfftw_backend_does_not_crash(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import FFTBackendChoice
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        _submenu_action(win, "Performance", "FFT Backend", "pyFFTW").trigger()
        _process_events(qtbot)
        assert win.app_settings.fft_backend == FFTBackendChoice.PYFFTW
    finally:
        win.close()


def test_selecting_memory_profile_recomputes_policy(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import MemoryProfileChoice
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        _submenu_action(win, "Performance", "Memory Profile", "Conservative").trigger()
        _process_events(qtbot)

        assert win.app_settings.memory_profile == MemoryProfileChoice.CONSERVATIVE
        assert win.renderer._memory_policy().profile == MemoryProfileChoice.CONSERVATIVE
    finally:
        win.close()


def test_memory_stress_actions_adjust_profile_budget_and_policy(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import MemoryProfileChoice
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        assert win.app_settings.render_memory_budget_mb == 512

        _menu_action(win, "Performance", "Use Less Memory").trigger()
        _process_events(qtbot)
        assert win.app_settings.memory_profile == MemoryProfileChoice.CONSERVATIVE
        assert win.app_settings.render_memory_budget_mb == 256
        assert win.renderer._memory_policy().profile == MemoryProfileChoice.CONSERVATIVE

        _menu_action(win, "Performance", "Decrease Render Budget").trigger()
        _process_events(qtbot)
        assert win.app_settings.render_memory_budget_mb == 128
        assert win._visible_render_budget_bytes() == 128 * 1024 * 1024

        _menu_action(win, "Performance", "Use More Memory").trigger()
        _process_events(qtbot)
        assert win.app_settings.memory_profile == MemoryProfileChoice.AGGRESSIVE
        assert win.app_settings.render_memory_budget_mb == 256

        _menu_action(win, "Performance", "Increase Render Budget").trigger()
        _process_events(qtbot)
        assert win.app_settings.render_memory_budget_mb == 512
    finally:
        win.close()


def test_montage_quality_policy_menu_defaults_and_switches(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import MontageQualityPolicyChoice
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        resident = _submenu_action(win, "Performance", "Montage LOD", "Resident (multi-resolution)")
        native = _submenu_action(win, "Performance", "Montage LOD", "Native only")
        # ADR 0050: resident is the default and the menu reflects it.
        assert win.app_settings.montage_quality_policy == MontageQualityPolicyChoice.RESIDENT
        assert resident.isChecked()
        assert not native.isChecked()

        native.trigger()
        _process_events(qtbot)
        assert win.app_settings.montage_quality_policy == MontageQualityPolicyChoice.NATIVE_ONLY
        assert win._settings.value("montage_quality_policy") == "native-only"
        assert native.isChecked()
        assert not resident.isChecked()

        resident.trigger()
        _process_events(qtbot)
        assert win.app_settings.montage_quality_policy == MontageQualityPolicyChoice.RESIDENT
        assert win._settings.value("montage_quality_policy") == "resident"
    finally:
        win.close()


def test_montage_quality_policy_change_applies_to_next_montage_session(qtbot):
    """A policy switch must take effect without an application restart."""

    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((3, 4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, indices=(0, 1, 2), text=":"))
        _process_events(qtbot)
        expected = win.renderer._montage_quality_policy_mode()
        session = getattr(win.renderer, "_frame_session", None)
        if session is not None:
            assert str(session.lod_policy_mode) == expected

        _submenu_action(win, "Performance", "Montage LOD", "Native only").trigger()
        _process_events(qtbot)
        session = getattr(win.renderer, "_frame_session", None)
        assert win.renderer._montage_quality_policy_mode() == "native-only"
        if session is not None:
            # The menu change replaced the session so the new policy is live.
            assert str(session.lod_policy_mode) == "native-only"
    finally:
        win.close()


def test_wgpu_present_method_menu_switches_and_persists(qtbot):
    """Performance → wgpu Presentation: Auto/Bitmap/Screen radio group.

    The submenu is enabled only while the wgpu backend is selected (the
    setting is a wgpu-backend concern); choices persist through QSettings
    and apply to newly opened windows like the backend choice itself.
    """

    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import (
        ImageRenderingBackendChoice,
        WgpuPresentMethodChoice,
    )
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        auto = _submenu_action(
            win, "Performance", "wgpu Presentation", "Auto (screen on native Wayland)"
        )
        bitmap = _submenu_action(
            win, "Performance", "wgpu Presentation", "Bitmap (readback compositing)"
        )
        screen = _submenu_action(
            win, "Performance", "wgpu Presentation", "Screen (native swapchain pin)"
        )
        # Bitmap is the default and the menu reflects it.
        assert win.app_settings.wgpu_present_method == WgpuPresentMethodChoice.BITMAP
        assert bitmap.isChecked()
        assert not auto.isChecked()
        assert not screen.isChecked()
        # Enabled under the default AUTO backend (AUTO resolves to wgpu on
        # GPU-capable devices) and under an explicit wgpu pin.
        assert win.app_settings.image_rendering_backend == ImageRenderingBackendChoice.AUTO
        assert win._wgpu_present_method_menu.isEnabled()

        _submenu_action(
            win,
            "Performance",
            "Image Rendering Backend",
            "wgpu (GPU compute)",
        ).trigger()
        _process_events(qtbot)
        assert win._wgpu_present_method_menu.isEnabled()

        auto.trigger()
        _process_events(qtbot)
        assert win.app_settings.wgpu_present_method == WgpuPresentMethodChoice.AUTO
        assert win._settings.value("wgpu_present_method") == "auto"
        assert auto.isChecked()
        assert not bitmap.isChecked()

        screen.trigger()
        _process_events(qtbot)
        assert win.app_settings.wgpu_present_method == WgpuPresentMethodChoice.SCREEN
        assert win._settings.value("wgpu_present_method") == "screen"

        # Switching the backend away greys the submenu but keeps the choice.
        _submenu_action(
            win, "Performance", "Image Rendering Backend", "PyQtGraph (CPU / remote)"
        ).trigger()
        _process_events(qtbot)
        assert not win._wgpu_present_method_menu.isEnabled()
        assert win.app_settings.wgpu_present_method == WgpuPresentMethodChoice.SCREEN
        assert win.app_settings.image_rendering_backend == ImageRenderingBackendChoice.PYQTGRAPH
    finally:
        win.close()


def test_texture_codec_menu_switches_and_persists(qtbot):
    """Performance → GPU Texture Compression: AUTO/OFF/BC radio group persists."""
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import TextureCodecChoice
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        assert win.app_settings.texture_codec == TextureCodecChoice.OFF
        off = _submenu_action(win, "Performance", "GPU Texture Compression", "Off (uncompressed)")
        off.trigger()
        _process_events(qtbot)
        assert win.app_settings.texture_codec == TextureCodecChoice.OFF
        assert win._settings.value("texture_codec") == "off"
        assert off.isChecked()
        bc = _submenu_action(win, "Performance", "GPU Texture Compression", "BC (force)")
        bc.trigger()
        _process_events(qtbot)
        assert win.app_settings.texture_codec == TextureCodecChoice.BC
        assert win._settings.value("texture_codec") == "bc"
    finally:
        win.close()


def test_host_cache_compression_menu_switches_and_persists(qtbot):
    """Performance → Host Cache Compression: RAW/ZFP/Blosc2 radio group persists."""
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import ChunkTransportCodecChoice
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        assert win.app_settings.chunk_transport_codec == ChunkTransportCodecChoice.RAW
        zfp = _submenu_action(win, "Performance", "Host Cache Compression", "ZFP (lossless)")
        zfp.trigger()
        _process_events(qtbot)
        assert win.app_settings.chunk_transport_codec == ChunkTransportCodecChoice.ZFP
        assert win._settings.value("chunk_transport_codec") == "zfp"
        assert zfp.isChecked()
    finally:
        win.close()


def test_host_cache_compression_greys_out_unavailable_codec(qtbot, monkeypatch):
    """A codec whose backing library is absent is disabled; the rest stay usable."""
    _clear_arrayscope_settings()
    import arrayscope.gpu.chunk_codec as chunk_codec
    from arrayscope.window import ArrayScopeWindow

    # Simulate an environment where zfpy is missing but blosc2 is importable.
    monkeypatch.setattr(chunk_codec, "available_codec_names", lambda: ("raw", "blosc2", "bitpack"))

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        sub = "Host Cache Compression"
        off = _submenu_action(win, "Performance", sub, "Off (uncompressed)")
        auto = _submenu_action(win, "Performance", sub, "Auto (experimental)")
        zfp = _submenu_action(win, "Performance", sub, "ZFP (lossless)")
        blosc2 = _submenu_action(win, "Performance", sub, "Blosc2 (lossless)")
        assert off.isEnabled()  # off is always available
        assert blosc2.isEnabled()  # blosc2 importable
        assert auto.isEnabled()  # at least one codec available
        assert not zfp.isEnabled()  # zfpy missing -> greyed
        assert "zfpy" in zfp.toolTip()
    finally:
        win.close()


def test_host_cache_compression_greys_out_auto_when_no_codec(qtbot, monkeypatch):
    """With no codec library importable, auto/zfp/blosc2 are all disabled."""
    _clear_arrayscope_settings()
    import arrayscope.gpu.chunk_codec as chunk_codec
    from arrayscope.window import ArrayScopeWindow

    monkeypatch.setattr(chunk_codec, "available_codec_names", lambda: ("raw", "bitpack"))

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        sub = "Host Cache Compression"
        off = _submenu_action(win, "Performance", sub, "Off (uncompressed)")
        auto = _submenu_action(win, "Performance", sub, "Auto (experimental)")
        zfp = _submenu_action(win, "Performance", sub, "ZFP (lossless)")
        blosc2 = _submenu_action(win, "Performance", sub, "Blosc2 (lossless)")
        assert off.isEnabled()  # off is always available
        assert not auto.isEnabled()
        assert not zfp.isEnabled()
        assert not blosc2.isEnabled()
    finally:
        win.close()


_RESIDENT_CROP_REBIND = "Resident Crop Rebind"


def test_resident_crop_rebind_menu_defaults_on_and_persists(qtbot):
    """Performance → Resident Crop Rebind: checkable, default ON, persists off."""
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        action = _menu_action(win, "Performance", _RESIDENT_CROP_REBIND)
        assert action.isCheckable()
        # Default ON: the rebind settles the same pixels and the same auto
        # levels as the ordinary evaluation, on raw and operation-pipeline
        # montages alike (tests/ui/test_resident_crop_rebind.py).
        assert action.isChecked()
        assert win.app_settings.resident_crop_rebind is True

        action.trigger()
        _process_events(qtbot)
        assert win.app_settings.resident_crop_rebind is False
        assert not action.isChecked()
        assert not _to_bool(win._settings.value("resident_crop_rebind"))
    finally:
        win.close()

    # A newly opened window restores the persisted toggle from QSettings.
    win2 = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win2)
    try:
        _process_events(qtbot)
        assert win2.app_settings.resident_crop_rebind is False
        assert not _menu_action(win2, "Performance", _RESIDENT_CROP_REBIND).isChecked()
    finally:
        win2.close()


def _to_bool(value):
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)
