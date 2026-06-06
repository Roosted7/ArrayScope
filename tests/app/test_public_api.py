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
