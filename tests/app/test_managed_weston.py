from __future__ import annotations

import subprocess
from pathlib import Path


def test_managed_weston_runs_child_once_and_cleans_private_socket(tmp_path, monkeypatch):
    from arrayscope.tools.managed_weston import run_in_managed_weston

    runtime = tmp_path / "runtime"
    artifacts = tmp_path / "artifacts"
    runtime.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    monkeypatch.setattr(
        "arrayscope.tools.managed_weston.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    calls = []

    def run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        socket_arg = next(arg for arg in command if str(arg).startswith("--socket="))
        socket_name = str(socket_arg).split("=", 1)[1]
        (runtime / socket_name).touch()
        (runtime / f"{socket_name}.lock").touch()
        status_index = command.index("--status-file") + 1
        Path(command[status_index]).write_text("7\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("arrayscope.tools.managed_weston.subprocess.run", run)

    result = run_in_managed_weston(
        ("python", "-m", "arrayscope.tools.profile_montage_workflow", "--backend", "wgpu"),
        artifact_dir=artifacts,
        output_size=(1400, 940),
    )

    assert result == 7
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:3] == ("/usr/bin/weston", "--backend=wayland", "--shell=kiosk")
    assert "--width=1400" in command
    assert "--height=940" in command
    assert "--debug" in command
    assert command.count("--") == 2
    assert kwargs["env"]["ARRAYSCOPE_MANAGED_WESTON"] == "1"
    assert kwargs["env"]["QT_QPA_PLATFORM"] == "wayland"
    assert not tuple(runtime.glob("arrayscope-profile-*"))


def test_managed_weston_capture_moves_one_screenshot_and_cleans_temp(tmp_path, monkeypatch):
    from PIL import Image

    from arrayscope.tools.managed_weston import capture_managed_weston_screenshot

    calls = []

    def run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        capture = Path(kwargs["cwd"]) / "wayland-screenshot-test.png"
        Image.new("RGB", (40, 30), "magenta").save(capture)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "arrayscope.tools.managed_weston.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    monkeypatch.setattr("arrayscope.tools.managed_weston.subprocess.run", run)
    destination = tmp_path / "window.png"

    assert capture_managed_weston_screenshot(destination) == destination
    assert Image.open(destination).size == (40, 30)
    assert len(calls) == 1
    assert calls[0][0] == ("/usr/bin/weston-screenshooter",)
    assert not Path(calls[0][1]["cwd"]).exists()
