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


def test_cli_multi_file_launches_valid_paths_before_final_block(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import numpy as np

    from arrayscope import __main__ as cli

    first = tmp_path / "first.npy"
    bad = tmp_path / "bad.npy"
    third = tmp_path / "third.npy"
    for path in (first, bad, third):
        path.write_bytes(b"placeholder")
    events = []

    def fake_load_path(filepath):
        events.append(("load", filepath.name))
        if filepath == bad:
            raise RuntimeError("broken file")
        return SimpleNamespace(data=np.zeros((2, 2)), metadata={})

    def fake_open_array_window(**kwargs):
        events.append(("open", kwargs["filepath"].name, kwargs["block"]))
        return object()

    def fake_run_loop():
        events.append(("loop",))

    monkeypatch.setattr(cli, "load_path", fake_load_path)
    monkeypatch.setattr(cli, "_open_array_window", fake_open_array_window)
    monkeypatch.setattr(cli, "_run_cli_event_loop", fake_run_loop)
    monkeypatch.setattr("sys.argv", ["arrayscope", str(first), str(bad), str(third)])

    cli.main()

    assert events == [
        ("load", "first.npy"),
        ("open", "first.npy", False),
        ("load", "bad.npy"),
        ("load", "third.npy"),
        ("open", "third.npy", False),
        ("loop",),
    ]


def test_cli_multi_path_selector_uses_nonblocking_view(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import numpy as np

    from arrayscope import __main__ as cli

    array_path = tmp_path / "array.npy"
    selector_path = tmp_path / "stack.npz"
    array_path.write_bytes(b"placeholder")
    selector_path.write_bytes(b"placeholder")
    events = []

    class FakeSelector:
        def __init__(self, filepath):
            self.filepath = filepath

        def requires_gui(self):
            return True

        def view(self, *, block=False):
            events.append(("selector", self.filepath.name, block))
            return True

    monkeypatch.setattr(cli, "load_path", lambda filepath: SimpleNamespace(data=np.zeros((2, 2)), metadata={}))
    monkeypatch.setattr(cli, "NpzDatasetSelector", FakeSelector)
    monkeypatch.setattr(
        cli,
        "_open_array_window",
        lambda **kwargs: events.append(("open", kwargs["filepath"].name, kwargs["block"])) or object(),
    )
    monkeypatch.setattr(cli, "_run_cli_event_loop", lambda: events.append(("loop",)))
    monkeypatch.setattr("sys.argv", ["arrayscope", str(array_path), str(selector_path)])

    cli.main()

    assert events == [
        ("open", "array.npy", False),
        ("selector", "stack.npz", False),
        ("loop",),
    ]


def test_cli_multi_path_single_dataset_selector_opens_inline(monkeypatch, tmp_path):
    import numpy as np

    from arrayscope import __main__ as cli

    selector_path = tmp_path / "single.npz"
    selector_path.write_bytes(b"placeholder")
    events = []

    class FakeSelector:
        def __init__(self, filepath):
            self.filepath = filepath

        def requires_gui(self):
            return False

        def get_single_data(self):
            return "arr", np.zeros((2, 2))

        def close(self):
            events.append(("close", self.filepath.name))

    monkeypatch.setattr(cli, "NpzDatasetSelector", FakeSelector)
    monkeypatch.setattr(
        cli,
        "_open_array_window",
        lambda **kwargs: events.append(
            (
                "open",
                kwargs["filepath"].name,
                kwargs["dataset_path"],
                kwargs["selector_class_name"],
                kwargs["block"],
            )
        )
        or object(),
    )
    monkeypatch.setattr(cli, "_run_cli_event_loop", lambda: events.append(("loop",)))
    monkeypatch.setattr("sys.argv", ["arrayscope", str(selector_path), str(selector_path)])

    cli.main()

    assert events == [
        ("close", "single.npz"),
        ("open", "single.npz", "arr", "FakeSelector", False),
        ("close", "single.npz"),
        ("open", "single.npz", "arr", "FakeSelector", False),
        ("loop",),
    ]
