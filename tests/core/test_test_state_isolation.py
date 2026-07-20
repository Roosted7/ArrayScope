import sys
import types

import arrayscope

_PROBE_NAME = "arrayscope._test_identity_probe"


def test_01_lazy_arrayscope_import_registers_one_module_object():
    """Ring 1: a module first imported in one test keeps one identity."""

    probe = types.ModuleType(_PROBE_NAME)
    sys.modules[_PROBE_NAME] = probe
    arrayscope._test_identity_probe = probe

    assert sys.modules[_PROBE_NAME] is arrayscope._test_identity_probe


def test_02_lazy_arrayscope_import_stays_consistent_across_test_teardown():
    """Ring 1: teardown must not split sys.modules from its parent package."""

    assert sys.modules[_PROBE_NAME] is arrayscope._test_identity_probe
