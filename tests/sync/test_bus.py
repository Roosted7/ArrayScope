"""SyncBus transport tests.

These run broker and clients inside one process; QLocalServer/QLocalSocket
behave identically for same-process and cross-process connections, so this
exercises the real socket path (the cross-process case differs only in who
owns the event loop).
"""

import uuid

import pytest

from arrayscope.sync.bus import SYNC_SERVER_NAME_ENV, SyncBus, default_server_name
from arrayscope.sync.messages import state_message

pytest.importorskip("pytestqt")


@pytest.fixture
def server_name():
    return f"arrayscope-sync-test-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def make_bus(qtbot, server_name):
    buses = []

    def make():
        bus = SyncBus(server_name=server_name)
        buses.append(bus)
        return bus

    yield make
    for bus in buses:
        bus.stop()
    # Destroy deleteLater'd sockets now, while the buses are still referenced,
    # instead of letting them die inside a later test's event loop.
    from pyqtgraph.Qt import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is not None:
        for _ in range(3):
            app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
            app.processEvents()


def _received(qtbot, bus):
    messages = []
    bus.messageReceived.connect(messages.append)
    return messages


def test_first_bus_becomes_broker_second_becomes_client(make_bus):
    first = make_bus()
    second = make_bus()
    assert first.start() == "broker"
    assert second.start() == "client"


def test_publish_reaches_other_participants_but_not_self(qtbot, make_bus):
    broker = make_bus()
    client_a = make_bus()
    client_b = make_bus()
    broker.start()
    client_a.start()
    client_b.start()
    at_broker = _received(qtbot, broker)
    at_a = _received(qtbot, client_a)
    at_b = _received(qtbot, client_b)

    from_a = state_message("dims", "window-a", 1, {"shape": [4], "slice_indices": [1]})
    client_a.publish(from_a)
    qtbot.waitUntil(lambda: bool(at_broker and at_b), timeout=2000)
    assert at_broker == [from_a]
    assert at_b == [from_a]
    assert at_a == []

    from_broker = state_message("dims", "window-broker", 1, {"shape": [4], "slice_indices": [2]})
    broker.publish(from_broker)
    qtbot.waitUntil(lambda: len(at_a) == 1 and len(at_b) == 2, timeout=2000)
    assert at_a == [from_broker]
    assert at_b == [from_a, from_broker]
    assert at_broker == [from_a]


def test_broker_exit_triggers_reelection_and_messages_flow_again(qtbot, make_bus):
    broker = make_bus()
    client_a = make_bus()
    client_b = make_bus()
    broker.start()
    client_a.start()
    client_b.start()

    broker.stop()
    qtbot.waitUntil(
        lambda: {client_a.role, client_b.role} == {"broker", "client"},
        timeout=5000,
    )

    new_broker = client_a if client_a.role == "broker" else client_b
    survivor = client_b if new_broker is client_a else client_a
    at_survivor = _received(qtbot, survivor)
    # The survivor's role flips to "client" the instant its socket connects,
    # but the new broker registers it as a relay peer asynchronously (server
    # side newConnection). Wait for the relay path to be established before
    # publishing, otherwise the message races reconnection and is dropped.
    qtbot.waitUntil(lambda: new_broker.peer_count >= 1, timeout=2000)
    message = state_message("levels", "window-x", 5, {"levels": [0.0, 2.0], "window_mode": "absolute"})
    new_broker.publish(message)
    qtbot.waitUntil(lambda: at_survivor == [message], timeout=2000)


def test_stale_server_name_is_reclaimed(make_bus, server_name):
    # A crashed broker can leave a stale socket behind on Unix. Simulate by
    # listening and closing without removeServer via an aborted bus start.
    first = make_bus()
    assert first.start() == "broker"
    # Simulate crash: close the server object without cleanup.
    first._server.close()
    first._role = "stopped"

    second = make_bus()
    assert second.start() == "broker"


def test_default_server_name_env_override(monkeypatch):
    monkeypatch.setenv(SYNC_SERVER_NAME_ENV, "custom-group")
    assert default_server_name() == "custom-group"
    monkeypatch.delenv(SYNC_SERVER_NAME_ENV)
    assert default_server_name().startswith("arrayscope-sync-")
