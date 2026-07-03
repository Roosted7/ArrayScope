"""Cross-process sync transport over Qt local sockets.

One ArrayScope process per user acts as the broker (it owns the
``QLocalServer``); every other process connects as a client. Qt local
sockets map to named pipes on Windows and Unix domain sockets on Linux and
macOS, so the same code runs on all three platforms, needs no network
permissions, and stays on the local machine.

Roles are dynamic: the first bus to start becomes broker; when the broker
process exits, each surviving client retries after a short random delay and
the first one to bind becomes the new broker while the rest reconnect.
Messages are newline-delimited JSON envelopes (see ``sync.messages``); the
broker relays every message to all other participants and also delivers it
to its own window. A bus never delivers a message back to the socket it
arrived on, and never delivers its own published messages locally.
"""

from __future__ import annotations

import getpass
import importlib
import os
import random
import re

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

import pyqtgraph.Qt as Qt

# pyqtgraph's Qt shim does not re-export QtNetwork; load it from the binding
# pyqtgraph selected so both stay on the same Qt library.
QtNetwork = importlib.import_module(f"{Qt.QT_LIB}.QtNetwork")

from arrayscope.sync.messages import decode_lines, encode_message


SYNC_SERVER_NAME_ENV = "ARRAYSCOPE_SYNC_NAME"

_CONNECT_TIMEOUT_MS = 250
_RETRY_MIN_MS = 50
_RETRY_MAX_MS = 350


def default_server_name() -> str:
    """Per-user server name so separate users on one machine never share."""

    override = os.environ.get(SYNC_SERVER_NAME_ENV)
    if override:
        return str(override)
    try:
        user = getpass.getuser()
    except Exception:
        user = "unknown"
    user = re.sub(r"[^A-Za-z0-9._-]+", "-", str(user)) or "unknown"
    return f"arrayscope-sync-{user}"


class SyncBus(Qt.QtCore.QObject):
    """Broker-or-client message bus for linked ArrayScope windows."""

    messageReceived = Qt.QtCore.Signal(object)
    roleChanged = Qt.QtCore.Signal(str)  # "broker" | "client" | "connecting" | "stopped"

    def __init__(self, server_name: str | None = None, parent=None):
        super().__init__(parent)
        self.server_name = str(server_name) if server_name else default_server_name()
        self._server: QtNetwork.QLocalServer | None = None
        self._client: QtNetwork.QLocalSocket | None = None
        self._peers: list[QtNetwork.QLocalSocket] = []
        self._buffers: dict[QtNetwork.QLocalSocket, bytes] = {}
        self._retry_timer: Qt.QtCore.QTimer | None = None
        self._role = "stopped"
        self._stopping = False

    # ------------------------------------------------------------------
    # Lifecycle

    @property
    def role(self) -> str:
        return self._role

    def is_running(self) -> bool:
        return self._role in ("broker", "client")

    def start(self) -> str:
        """Connect to the group broker, or become the broker; returns role."""

        if self.is_running():
            return self._role
        self._stopping = False
        if self._connect_as_client():
            return self._role
        if self._listen_as_broker():
            return self._role
        # Lost a startup race in both directions; retry shortly.
        self._set_role("connecting")
        self._schedule_retry()
        return self._role

    def stop(self) -> None:
        self._stopping = True
        if self._retry_timer is not None:
            self._retry_timer.stop()
        if self._client is not None:
            client, self._client = self._client, None
            client.disconnected.disconnect(self._on_client_disconnected)
            client.abort()
            client.deleteLater()
        for peer in tuple(self._peers):
            peer.abort()
            peer.deleteLater()
        self._peers = []
        self._buffers = {}
        if self._server is not None:
            server, self._server = self._server, None
            server.close()
            server.deleteLater()
            QtNetwork.QLocalServer.removeServer(self.server_name)
        self._set_role("stopped")

    # ------------------------------------------------------------------
    # Publishing

    def publish(self, message) -> bool:
        """Send one envelope to every other participant in the group."""

        data = encode_message(message)
        if self._role == "broker":
            self._relay(data, exclude=None)
            return True
        if self._role == "client" and self._client is not None:
            self._client.write(data)
            self._client.flush()
            return True
        return False

    # ------------------------------------------------------------------
    # Broker role

    def _listen_as_broker(self) -> bool:
        server = QtNetwork.QLocalServer(self)
        try:
            server.setSocketOptions(QtNetwork.QLocalServer.SocketOption.UserAccessOption)
        except Exception:
            pass
        if not server.listen(self.server_name):
            # A stale socket file (crashed broker on Unix) blocks listen even
            # though nobody is serving. Connecting already failed, so it is
            # safe to remove and retry once.
            QtNetwork.QLocalServer.removeServer(self.server_name)
            if not server.listen(self.server_name):
                server.deleteLater()
                return False
        server.newConnection.connect(self._on_new_peer)
        self._server = server
        self._set_role("broker")
        return True

    def _on_new_peer(self) -> None:
        if self._server is None:
            return
        while True:
            peer = self._server.nextPendingConnection()
            if peer is None:
                return
            self._peers.append(peer)
            self._buffers[peer] = b""
            peer.readyRead.connect(lambda peer=peer: self._on_peer_ready_read(peer))
            peer.disconnected.connect(lambda peer=peer: self._on_peer_disconnected(peer))

    def _on_peer_ready_read(self, peer) -> None:
        buffer = self._buffers.get(peer, b"") + bytes(peer.readAll().data())
        messages, self._buffers[peer] = decode_lines(buffer)
        for message in messages:
            self._relay(encode_message(message), exclude=peer)
            self.messageReceived.emit(message)

    def _on_peer_disconnected(self, peer) -> None:
        if peer in self._peers:
            self._peers.remove(peer)
        self._buffers.pop(peer, None)
        peer.deleteLater()

    def _relay(self, data: bytes, *, exclude) -> None:
        for peer in tuple(self._peers):
            if peer is exclude:
                continue
            peer.write(data)
            peer.flush()

    # ------------------------------------------------------------------
    # Client role

    def _connect_as_client(self) -> bool:
        client = QtNetwork.QLocalSocket(self)
        client.connectToServer(self.server_name)
        if not client.waitForConnected(_CONNECT_TIMEOUT_MS):
            client.abort()
            client.deleteLater()
            return False
        self._client = client
        self._buffers[client] = b""
        client.readyRead.connect(self._on_client_ready_read)
        client.disconnected.connect(self._on_client_disconnected)
        self._set_role("client")
        return True

    def _on_client_ready_read(self) -> None:
        client = self._client
        if client is None:
            return
        buffer = self._buffers.get(client, b"") + bytes(client.readAll().data())
        messages, self._buffers[client] = decode_lines(buffer)
        for message in messages:
            self.messageReceived.emit(message)

    def _on_client_disconnected(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            self._buffers.pop(client, None)
            client.deleteLater()
        if self._stopping:
            return
        # Broker exited: re-elect. Jitter keeps surviving clients from
        # racing each other for the listen slot at the same instant.
        self._set_role("connecting")
        self._schedule_retry()

    # ------------------------------------------------------------------
    # Re-election

    def _schedule_retry(self) -> None:
        if self._retry_timer is None:
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._retry_start)
            self._retry_timer = timer
        self._retry_timer.start(random.randint(_RETRY_MIN_MS, _RETRY_MAX_MS))

    def _retry_start(self) -> None:
        if self._stopping or self.is_running():
            return
        self.start()

    def _set_role(self, role: str) -> None:
        if role == self._role:
            return
        self._role = role
        self.roleChanged.emit(role)
