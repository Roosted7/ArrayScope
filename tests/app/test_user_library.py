"""The one storage protocol behind the colormap and operation libraries.

These behaviors used to be tested twice, near-identically, once per library
(``tests/display/test_colormap_library.py`` and
``tests/operations/test_operation_library.py``). ADR 0063 made them one
implementation, so they are one test file; each library's own file keeps only
what is genuinely its own.
"""

from __future__ import annotations

import json

import pytest

from arrayscope.app import user_library

# ---------------------------------------------------------------------------
# Listeners
# ---------------------------------------------------------------------------


def test_remove_listener_stops_notifications_and_is_idempotent():
    registry = user_library.ListenerRegistry("test")
    calls = []

    def listener():
        calls.append(1)

    # Removing before it was ever registered is a no-op...
    registry.remove(listener)
    registry.add(listener)
    registry.add(listener)  # registering twice must not double-notify
    registry.notify()
    assert calls == [1]

    registry.remove(listener)
    registry.notify()
    assert calls == [1]
    # ...and removing a second time must not raise.
    registry.remove(listener)


def test_notify_prunes_a_listener_bound_to_a_deleted_widget():
    # A window that closed without unregistering leaves a listener bound to a
    # deleted C++ object; invoking it raises RuntimeError. The registry must
    # self-heal (drop it) so a library mutation neither crashes nor keeps
    # calling the dead wrapper on every future mutation.
    registry = user_library.ListenerRegistry("test")

    class _DeletedWidgetListener:
        def __call__(self):
            raise RuntimeError("Internal C++ object (ArrayScopeWindow) already deleted.")

    dead = _DeletedWidgetListener()
    live_calls = []
    registry.add(dead)
    registry.add(lambda: live_calls.append(1))

    registry.notify()  # must not raise
    assert live_calls == [1]  # a sibling listener still ran
    registry.notify()
    assert live_calls == [1, 1]  # and keeps running once the corpse is gone


def test_notify_keeps_going_after_an_ordinary_listener_failure(caplog):
    registry = user_library.ListenerRegistry("test")
    calls = []
    registry.add(lambda: (_ for _ in ()).throw(ValueError("boom")))
    registry.add(lambda: calls.append(1))
    registry.notify()
    assert calls == [1]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _store(tmp_path, **kwargs):
    kwargs.setdefault("format_name", "arrayscope-test-layout")
    kwargs.setdefault("group_key", "map_groups")
    kwargs.setdefault("order_key", "map_order")
    return user_library.LayoutStore(lambda: str(tmp_path), **kwargs)


def test_headerless_layout_still_loads_and_the_next_save_upgrades_it(tmp_path):
    # The colormap library wrote header-less layouts for two years. Reading one
    # must keep working; writing must produce the shared format+version header,
    # which makes the upgrade one-way and invisible.
    (tmp_path / "layout.json").write_text(
        json.dumps({"group_order": ["Perceptual"], "map_groups": {"turbo": ""}, "map_order": {}})
    )
    store = _store(tmp_path)
    layout = store.load()
    assert layout["group_order"] == ["Perceptual"]
    assert layout["map_groups"] == {"turbo": ""}

    store.save(layout)
    written = json.loads((tmp_path / "layout.json").read_text())
    assert written["format"] == "arrayscope-test-layout"
    assert written["version"] == user_library.LAYOUT_VERSION
    assert written["group_order"] == ["Perceptual"]


def test_unset_optional_lists_are_not_persisted(tmp_path):
    # An unset presentation override must keep tracking the code default rather
    # than freezing today's value into the file.
    store = _store(tmp_path, optional_lists=("common_ids",))
    store.save({**store.empty(), "group_order": ["User"]})
    assert "common_ids" not in json.loads((tmp_path / "layout.json").read_text())

    store.save({**store.empty(), "common_ids": ["mean"]})
    assert json.loads((tmp_path / "layout.json").read_text())["common_ids"] == ["mean"]
    assert store.load()["common_ids"] == ["mean"]


def test_a_broken_layout_file_falls_back_to_the_defaults(tmp_path):
    (tmp_path / "layout.json").write_text("{not json")
    assert _store(tmp_path).load() == {"group_order": [], "map_groups": {}, "map_order": {}}


def test_rename_group_reassigns_members_and_the_order_entry(tmp_path):
    store = _store(tmp_path)
    store.save({**store.empty(), "group_order": ["Perceptual", "User"]})
    assert store.rename_group("Perceptual", "Mine", ["turbo", "viridis"])
    layout = store.load()
    assert layout["group_order"] == ["Mine", "User"]
    assert layout["map_groups"] == {"turbo": "Mine", "viridis": "Mine"}
    # A no-op rename never rewrites the file.
    assert not store.rename_group("Mine", "Mine", ["turbo"])
    assert not store.rename_group("Mine", "   ", ["turbo"])


def test_ordered_groups_prefers_persisted_then_default_then_alphabetical(tmp_path):
    store = _store(tmp_path)
    store.save({**store.empty(), "group_order": ["User"]})
    ordered = store.ordered_groups({"User", "CET", "Zed", "Perceptual"}, ("Perceptual", "CET"))
    assert ordered == ["User", "Perceptual", "CET", "Zed"]


def test_remove_discards_the_persisted_layout(tmp_path):
    store = _store(tmp_path)
    store.save({**store.empty(), "group_order": ["User"]})
    store.remove()
    assert not (tmp_path / "layout.json").exists()
    assert store.load()["group_order"] == []
    store.remove()  # removing what is not there is a no-op


# ---------------------------------------------------------------------------
# Hidden sets
# ---------------------------------------------------------------------------


def test_hidden_set_round_trips_and_notifies(tmp_path):
    calls = []
    listeners = user_library.ListenerRegistry("test")
    listeners.add(lambda: calls.append(1))
    hidden = user_library.HiddenSet(
        lambda: str(tmp_path / "nested"), file_name="hidden.json", listeners=listeners
    )
    assert hidden.values() == frozenset()
    hidden.set("turbo", True)
    assert hidden.values() == frozenset({"turbo"})
    hidden.set("turbo", False)
    assert hidden.values() == frozenset()
    assert calls == [1, 1]


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def test_atomic_write_leaves_no_temporary_behind(tmp_path):
    path = str(tmp_path / "deep" / "payload.json")
    user_library.write_json_atomically(path, {"a": 1})
    assert json.loads((tmp_path / "deep" / "payload.json").read_text()) == {"a": 1}
    assert [p.name for p in (tmp_path / "deep").iterdir()] == ["payload.json"]


def test_migration_copies_once_and_never_over_an_occupied_destination(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    (old / "mine.json").write_text('{"name": "mine"}')
    (old / "notes.txt").write_text("ignored")

    assert user_library.migrate_directory(str(old), str(new))
    assert (new / "mine.json").read_text() == '{"name": "mine"}'
    assert not (new / "notes.txt").exists()
    # The source survives: an older ArrayScope on the same machine keeps working.
    assert (old / "mine.json").exists()

    # A second run must not overwrite what the user has since done.
    (new / "mine.json").write_text('{"name": "edited"}')
    assert not user_library.migrate_directory(str(old), str(new))
    assert (new / "mine.json").read_text() == '{"name": "edited"}'


def test_migration_declines_when_there_is_nothing_to_move(tmp_path):
    assert not user_library.migrate_directory(str(tmp_path / "absent"), str(tmp_path / "new"))
    assert not user_library.migrate_directory(str(tmp_path), str(tmp_path))


# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected_payload", "expected_id"),
    [
        ('{"name": "mine", "stops": []}', {"name": "mine", "stops": []}, "mine"),
        ('{"stops": []}', {"stops": []}, None),
        ("[1, 2, 3]", None, None),
        ("{not json", None, None),
    ],
)
def test_problem_record_salvages_whatever_the_payload_still_offers(
    tmp_path, text, expected_payload, expected_id
):
    path = tmp_path / "thing.json"
    path.write_text(text)
    problem = user_library.problem_record(str(path), "could not read", id_field="name")
    assert problem.path == str(path)
    assert problem.message == "could not read"
    assert problem.payload == expected_payload
    assert problem.item_id == expected_id
