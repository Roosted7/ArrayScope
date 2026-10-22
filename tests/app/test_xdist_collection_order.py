"""Harness-only ordering stays identical across initial and restarted workers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests import conftest as harness
from tests import testmon_policy


class _Item:
    def __init__(self, nodeid):
        self.nodeid = nodeid

    def get_closest_marker(self, _name):
        return None


_DECLARATION = ("a.py::test_first", "a.py::test_second", "b.py::test_only", "c.py::test_only")


def _collect(config):
    items = [_Item(nodeid) for nodeid in _DECLARATION]
    wrapper = harness.pytest_collection_modifyitems(config, items)
    next(wrapper)
    # Another plugin, such as testmon, may reorder between the wrapper halves.
    items.reverse()
    with pytest.raises(StopIteration):
        next(wrapper)
    return tuple(item.nodeid for item in items)


def _node(controller):
    node = SimpleNamespace(config=controller, workerinput={})
    # Without our optional xdist hook, collection still runs, using worker-local
    # mutable timings. Keep that old behavior observable in this regression.
    configure = getattr(harness, "pytest_configure_node", lambda _node: None)
    configure(node)
    return node


def _worker(node, data):
    return SimpleNamespace(
        workerinput=dict(node.workerinput), testmon_data=data, testmon_config=object()
    )


@pytest.mark.parametrize(("numprocesses", "tx"), [(2, []), (None, ["popen"])])
def test_initial_and_restarted_workers_share_one_pre_run_timing_snapshot(
    monkeypatch, numprocesses, tx
):
    reports = {
        "a.py::test_first": {"duration": 1.0},
        "b.py::test_only": {"duration": 9.0},
        "c.py::test_only": {"duration": 0.1},
    }
    data = SimpleNamespace(all_tests=reports)
    controller = SimpleNamespace(
        testmon_data=data,
        testmon_config=None,
        option=SimpleNamespace(numprocesses=numprocesses, tx=tx),
    )
    calls = []
    read_seconds = testmon_policy.file_seconds

    def record_read(config):
        calls.append(config)
        return read_seconds(config)

    monkeypatch.setattr(testmon_policy, "file_seconds", record_read)
    # Exercise the controller's pre-node session hook without disturbing the
    # enclosing pytest run's reporting ledgers when this test runs serially.
    for name in (
        "_red_ledger",
        "_flake_ledger",
        "_skip_ledger",
        "_forced_reds",
        "_changed_since_recorded",
        "_flake_run_shape",
        "_flake_load_start",
    ):
        monkeypatch.setattr(harness, name, getattr(harness, name))
    harness.pytest_sessionstart(SimpleNamespace(config=controller))
    reports["a.py::test_first"]["duration"] = 20.0
    first = _node(controller)
    # A running worker records new durations before another worker (or its
    # crash replacement) has finished collection against the shared map.
    reports["a.py::test_first"]["duration"] = 30.0
    reports["c.py::test_only"]["duration"] = 2.0
    first_ids = _collect(_worker(first, data))
    second = _node(controller)
    reports["b.py::test_only"]["duration"] = 60.0
    reports["a.py::test_first"]["duration"] = 0.1
    second_ids = _collect(_worker(second, data))
    restarted = _node(controller)
    reports["c.py::test_only"]["duration"] = 90.0
    restarted_ids = _collect(_worker(restarted, data))

    expected = ("c.py::test_only", "b.py::test_only", "a.py::test_first", "a.py::test_second")
    assert first_ids == second_ids == restarted_ids == expected
    assert sorted(first_ids) == sorted(_DECLARATION), "ordering must never change selection"
    assert calls == [controller], "only the controller reads live timings, once per run"
    assert first.workerinput == second.workerinput == restarted.workerinput
    assert all(isinstance(value, tuple) for value in first.workerinput.values())


def test_a_worker_without_a_snapshot_keeps_deterministic_order_without_live_reads(monkeypatch):
    def forbidden_read(_config):
        pytest.fail("a worker must not read timings that another worker can change")

    monkeypatch.setattr(testmon_policy, "file_seconds", forbidden_read)
    config = SimpleNamespace(workerinput={}, testmon_config=object())

    assert _collect(config) == _DECLARATION


def test_serial_collection_keeps_using_its_current_timing_map():
    reports = {
        "a.py::test_first": {"duration": 1.0},
        "b.py::test_only": {"duration": 9.0},
        "c.py::test_only": {"duration": 0.1},
    }
    config = SimpleNamespace(
        testmon_data=SimpleNamespace(all_tests=reports), testmon_config=object()
    )

    assert _collect(config) == (
        "c.py::test_only",
        "b.py::test_only",
        "a.py::test_first",
        "a.py::test_second",
    )
    reports["a.py::test_first"]["duration"] = 30.0
    assert _collect(config) == (
        "c.py::test_only",
        "a.py::test_first",
        "a.py::test_second",
        "b.py::test_only",
    )


def test_worker_configuration_hook_is_optional_when_xdist_is_absent():
    manager = pytest.PytestPluginManager()
    manager.register(SimpleNamespace(pytest_configure_node=harness.pytest_configure_node))

    manager.check_pending()
