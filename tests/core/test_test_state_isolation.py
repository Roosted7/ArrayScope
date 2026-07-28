import sys
import types

import pytest

import arrayscope

_PROBE_NAME = "arrayscope._test_identity_probe"

# The pair below is the one place in the suite where a test observes what the
# previous one left behind — that is the property under test, since the autouse
# sys.modules restoration in tests/conftest.py runs in the teardown between
# them. The marker keeps them adjacent and in this order through the
# duration-based reordering that change selection applies; without it the sort
# separates them and the second reports a phantom teardown regression.
pytestmark = pytest.mark.coupled_order("arrayscope_module_identity")


def test_01_lazy_arrayscope_import_registers_one_module_object():
    """Ring 1: a module first imported in one test keeps one identity."""

    probe = types.ModuleType(_PROBE_NAME)
    sys.modules[_PROBE_NAME] = probe
    arrayscope._test_identity_probe = probe

    assert sys.modules[_PROBE_NAME] is arrayscope._test_identity_probe


def test_02_lazy_arrayscope_import_stays_consistent_across_test_teardown():
    """Ring 1: teardown must not split sys.modules from its parent package."""

    assert sys.modules[_PROBE_NAME] is arrayscope._test_identity_probe
