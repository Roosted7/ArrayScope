"""Recipe import policy validates authored intent before loading operations."""

import pytest

from arrayscope.operations import recipes


@pytest.mark.parametrize("enabled", ["false", "true", 0, 1, None, {}, []])
def test_version_two_enabled_requires_a_json_boolean(enabled):
    recipe = {"version": 2, "operations": [{"id": "reverse", "axis": 0, "enabled": enabled}]}

    with pytest.raises(ValueError, match="enabled must be a JSON boolean"):
        recipes.steps_from_recipe(recipe, (3, 4))


@pytest.mark.parametrize("version", [True, False, 1.0, 2.0, "2", None])
def test_recipe_versions_are_exact_integers(version):
    with pytest.raises(ValueError, match="unsupported recipe version"):
        recipes.steps_from_recipe({"version": version, "operations": []}, (3, 4))


@pytest.mark.parametrize("enabled", [True, False])
def test_valid_authored_enabled_state_round_trips(enabled):
    recipe = {"version": 2, "operations": [{"id": "reverse", "axis": 0, "enabled": enabled}]}

    steps = recipes.steps_from_recipe(recipe, (3, 4))

    assert steps[0].enabled is enabled
    restored = recipes.loads_recipe_steps(recipes.dumps_recipe(steps), (3, 4))
    assert restored[0].operation == steps[0].operation
    assert restored[0].enabled is enabled


def test_absent_enabled_defaults_to_enabled():
    steps = recipes.steps_from_recipe(
        {"version": 2, "operations": [{"id": "reverse", "axis": 0}]}, (3, 4)
    )

    assert steps[0].enabled is True


def test_policy_rejection_runs_before_lookup_or_import_quarantine(monkeypatch):
    def unexpected_lookup(*_args, **_kwargs):
        pytest.fail("the rejected operation must never reach lookup")

    def reject(operation_id):
        raise ValueError(f"Rejected {operation_id}")

    monkeypatch.setattr(recipes, "create_operation", unexpected_lookup)

    with pytest.raises(ValueError, match="Rejected user:command"):
        recipes.steps_from_recipe(
            {"version": 2, "operations": [{"id": "user:command"}]},
            (3, 4),
            imported=True,
            operation_policy=reject,
        )


def test_required_enabled_operation_cannot_silently_become_unavailable(monkeypatch):
    class UnavailableOperation:
        def current_unavailable_reason(self):
            return "Required reference is unavailable"

    monkeypatch.setattr(
        recipes, "create_operation", lambda *_args, **_kwargs: UnavailableOperation()
    )

    with pytest.raises(ValueError, match=r"operation 0.*Required reference is unavailable"):
        recipes.steps_from_recipe(
            {"version": 2, "operations": [{"id": "reverse", "axis": 0}]},
            (3, 4),
            require_available=True,
        )
