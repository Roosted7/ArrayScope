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


def _start_session(config, monkeypatch):
    # Exercise startup without disturbing the enclosing run's reporting state.
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
    harness.pytest_sessionstart(SimpleNamespace(config=config))


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
    _start_session(controller, monkeypatch)
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


class _Worker:
    def __init__(self, identity):
        self.gateway = SimpleNamespace(id=identity)
        self.shutting_down = False
        self.batches = []

    def send_runtest_some(self, indices):
        self.batches.append(tuple(indices))

    def shutdown(self):
        self.shutting_down = True


def _scheduled_file_units(config, collection):
    from xdist.scheduler.loadfile import LoadFileScheduling

    scheduler = LoadFileScheduling(config)
    nodes = [_Worker("gw0"), _Worker("gw1")]
    for node in nodes:
        scheduler.add_node(node)
        scheduler.add_node_collection(node, list(collection))
    scheduler.schedule()
    first_files = tuple(collection[node.batches[0][0]].split("::")[0] for node in nodes)
    while scheduler.has_pending:
        pending = [
            (node, nodeid)
            for node, units in scheduler.assigned_work.items()
            for unit in units.values()
            for nodeid, done in unit.items()
            if not done
        ]
        assert pending, "scheduler lost a selected test"
        for node, nodeid in pending:
            scheduler.mark_test_complete(node, collection.index(nodeid))
    batches = tuple(
        tuple(collection[index] for index in batch) for node in nodes for batch in node.batches
    )
    return first_files, batches


def test_actual_loadfile_scheduler_preserves_cost_order_and_every_selected_file_unit(monkeypatch):
    from _pytest.config.argparsing import Parser
    from xdist.plugin import pytest_addoption

    parser = Parser(_ispytest=True)
    pytest_addoption(parser)
    if not hasattr(parser.parse([]), "loadscopereorder"):
        pytest.skip("installed xdist does not expose configurable scope ordering")
    declaration = (
        *(f"fast_many.py::test_{index}" for index in range(8)),
        *(f"medium.py::test_{index}" for index in range(4)),
        *(f"slow_fft.py::test_{index}" for index in range(2)),
    )
    items = [_Item(nodeid) for nodeid in declaration]
    ordered = testmon_policy.order_for_makespan(
        items,
        {"fast_many.py": 1.0, "medium.py": 3.0, "slow_fft.py": 90.0},
        {nodeid: index for index, nodeid in enumerate(declaration)},
    )
    collection = [item.nodeid for item in ordered]

    def config(active):
        return SimpleNamespace(
            option=SimpleNamespace(numprocesses=2, dist="loadfile", loadscopereorder=True),
            testmon_config=SimpleNamespace(collect=False) if active else None,
            getvalue=lambda _name: ["popen", "popen"],
        )

    baseline = config(False)
    _start_session(baseline, monkeypatch)
    original_first, original_batches = _scheduled_file_units(baseline, collection)
    configured = config(True)
    _start_session(configured, monkeypatch)
    first, batches = _scheduled_file_units(configured, collection)

    assert original_first == ("fast_many.py", "medium.py")
    assert first == ("slow_fft.py", "medium.py")
    for dispatched in (original_batches, batches):
        assert sorted(nodeid for batch in dispatched for nodeid in batch) == sorted(declaration)
        assert all(len({nodeid.split("::")[0] for nodeid in batch}) == 1 for batch in dispatched)
        assert len(dispatched) == 3, "each file must stay one indivisible work unit"


@pytest.mark.parametrize(
    ("distribution", "cost_ordering", "has_option"),
    [("loadscope", True, True), ("loadfile", False, True), ("loadfile", True, False)],
)
def test_other_schedulers_and_older_xdist_keep_their_options(
    monkeypatch, distribution, cost_ordering, has_option
):
    options = SimpleNamespace(numprocesses=2, dist=distribution)
    if has_option:
        options.loadscopereorder = True
    config = SimpleNamespace(
        option=options,
        testmon_config=SimpleNamespace(collect=False) if cost_ordering else None,
    )

    _start_session(config, monkeypatch)

    assert hasattr(options, "loadscopereorder") is has_option
    if has_option:
        assert options.loadscopereorder is True
