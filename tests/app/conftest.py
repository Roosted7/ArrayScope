"""Shared fixtures for the harness's own tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests import conftest as harness


def write_map(path, mark: str, *, pages: int = 0) -> Path:
    """A real, tiny, WAL-mode SQLite map carrying ``mark``. Returns its path.

    Seeding and adoption used to be tested with placeholder bytes, which was
    honest while they were a file copy and stopped being honest the moment they
    became ``sqlite3.Connection.backup`` — a primitive that refuses anything
    that is not a database, which is the whole point of it. So the fixtures say
    it in SQLite: ``mark`` is what a test asserts arrived, in place of the
    byte string it used to compare.

    WAL because the real map is in WAL — see
    ``testmon_policy._snapshot_sqlite`` for why that is the entire story here.

    ``pages`` pads the file out to more than the header page, for tests that
    need somewhere to break: a one-page database has no b-tree to damage.
    """

    path = Path(path)
    connection = sqlite3.connect(str(path))
    try:
        connection.execute("PRAGMA journal_mode=wal")
        connection.execute("CREATE TABLE IF NOT EXISTS marker (mark TEXT)")
        connection.execute("DELETE FROM marker")
        connection.execute("INSERT INTO marker (mark) VALUES (?)", (mark,))
        if pages:
            connection.execute("CREATE TABLE IF NOT EXISTS padding (id INTEGER, name TEXT)")
            connection.execute("CREATE INDEX IF NOT EXISTS padding_name ON padding (name)")
            connection.executemany(
                "INSERT INTO padding (id, name) VALUES (?, ?)",
                [(index, f"tests/test_{index}.py::test_{index}") for index in range(pages * 40)],
            )
        connection.commit()
    finally:
        connection.close()
    return path


def break_a_page(path, page: int = 3) -> None:
    """Scribble over one interior page, the way a torn `cp` does.

    Not a truncation and not a header edit: those are their own failure modes
    with their own error messages. This is the shape the copy bug produced — a
    file whose header and schema are fine, so it opens, whose pages disagree,
    so the first real read of them raises ``database disk image is malformed``.
    """

    path = Path(path)
    with path.open("r+b") as handle:
        handle.seek(16)
        page_size = int.from_bytes(handle.read(2), "big") or 4096
        if page_size == 1:  # the header's spelling of 65536
            page_size = 65536
        handle.seek(page_size * (page - 1))
        handle.write(b"\xa5" * page_size)


def read_map(path) -> str:
    """The mark :func:`write_map` put in, read back through SQLite."""

    connection = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    try:
        return connection.execute("SELECT mark FROM marker").fetchone()[0]
    finally:
        connection.close()


@pytest.fixture
def as_the_controller(monkeypatch):
    """Run controller-only reporting code from a test that lives on a worker.

    ``_report_map_erosion``, ``_report_function_coverage``,
    ``_report_sweep_cadence`` and ``_refuse_since_on_unattributed_map`` all
    return immediately on an xdist worker, because one process must own the
    sidecar and the terminal. Tests of those paths execute *on* a worker, so
    without this they assert against a function that declined to act — passing
    under ``-n 0`` and failing under ``-n auto``, which is exactly the shape
    the suite's own docs warn about. Production never hits it: every one of
    those callers really is controller-only.
    """

    monkeypatch.setattr(harness, "_XDIST_WORKER", None)
