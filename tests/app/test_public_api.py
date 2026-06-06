def test_package_module_is_callable():
    import sys

    for name in list(sys.modules):
        if name == "arrayscope" or name.startswith("arrayscope."):
            del sys.modules[name]

    import arrayscope as asc

    assert callable(asc)


def test_calling_package_delegates_to_launch(monkeypatch):
    import sys

    for name in list(sys.modules):
        if name == "arrayscope" or name.startswith("arrayscope."):
            del sys.modules[name]

    import arrayscope as asc

    calls = []

    def fake_arrayscope(data, *args, **kwargs):
        calls.append((data, args, kwargs))
        return "window"

    monkeypatch.setattr(asc, "_arrayscope", fake_arrayscope)

    assert asc("data", title="demo") == "window"
    assert calls == [("data", (), {"title": "demo"})]


def test_cli_file_launch_blocks(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import numpy as np

    from arrayscope import __main__ as cli

    path = tmp_path / "subject.session.npy"
    path.write_bytes(b"placeholder")
    calls = []

    def fake_load_path(filepath):
        assert filepath == path
        return SimpleNamespace(data=np.zeros((2, 2)), metadata={})

    def fake_arrayscope(**kwargs):
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(cli, "load_path", fake_load_path)
    monkeypatch.setattr(cli, "arrayscope", fake_arrayscope)
    monkeypatch.setattr("sys.argv", ["arrayscope", str(path)])

    cli.main()

    assert calls
    assert calls[0]["block"] is True
    assert calls[0]["filepath"] == path
