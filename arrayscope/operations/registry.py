"""Registry for ArrayScope dimension operations."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass

from arrayscope.operations.pipeline import (
    CenteredFFT,
    CenteredIFFT,
    CombineRealImagAxis,
    Conjugate,
    Crop,
    FFTShift,
    Maximum,
    Mean,
    Minimum,
    ReverseAxis,
    RootSumSquares,
    SplitComplexAxis,
    Sum,
)


@dataclass(frozen=True)
class OperationParameter:
    name: str
    label: str
    kind: str = "int"
    # Richer metadata for form rendering / value coercion. All optional so
    # existing constructions (positional name/label/kind) keep working and older
    # recipes/packs that never set these fields behave exactly as before.
    default: int | float | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    step: int | float | None = None
    description: str = ""


# Group taxonomy for presenting operations in menus / palettes. Entries keep
# their real ``group``; "Common" is a *presentation* concern surfaced via
# ``COMMON_OPERATION_IDS`` / :func:`is_common`, not a group value an op carries.
DEFAULT_GROUP_ORDER: tuple[str, ...] = (
    "Common",
    "Reduce",
    "Transform",
    "Reshape",
    "Complex",
    "SigPy",
    "BART",
    "User",
    "Other",
)

# The handful of most-used ops a UI surface pins to the top. This is a
# presentation ordering, not a group membership -- each op keeps its real group.
COMMON_OPERATION_IDS: tuple[str, ...] = (
    "mean",
    "sum",
    "max",
    "min",
    "crop",
    "centered_fft",
    "centered_ifft",
)


def is_common(operation_id: str) -> bool:
    """Whether ``operation_id`` is one a UI surface pins to the top."""

    return operation_id in COMMON_OPERATION_IDS


@dataclass(frozen=True)
class OperationEntry:
    id: str
    label: str
    operation_type: type
    parameters: tuple[OperationParameter, ...] = ()
    changes_shape: bool = False
    requires_axis: bool = True
    # Presentation metadata (all defaulted so pack/plugin entries built without
    # these fields keep working). ``group`` places the op in the taxonomy above,
    # ``icon`` is the Material icon name the UI renders, ``description`` is a
    # one-line summary shown in tooltips / the operation manager.
    group: str = "Other"
    description: str = ""
    icon: str = "data_array"


OPERATION_REGISTRY = {
    "crop": OperationEntry(
        id="crop",
        label="Crop axis...",
        operation_type=Crop,
        parameters=(
            OperationParameter(
                "start", "Start", minimum=0, description="First index kept (inclusive)."
            ),
            OperationParameter(
                "stop", "Stop", minimum=1, description="Index one past the last kept (exclusive)."
            ),
        ),
        changes_shape=True,
        group="Transform",
        description="Keep a [start:stop] slice of one axis.",
        icon="crop",
    ),
    "reverse": OperationEntry(
        id="reverse",
        label="Reverse / flip axis",
        operation_type=ReverseAxis,
        group="Transform",
        description="Reverse the sample order along one axis.",
        icon="swap_horiz",
    ),
    "conjugate": OperationEntry(
        id="conjugate",
        label="Conjugate",
        operation_type=Conjugate,
        requires_axis=False,
        group="Complex",
        description="Complex-conjugate every sample.",
        icon="flip",
    ),
    "mean": OperationEntry(
        id="mean",
        label="Mean over axis",
        operation_type=Mean,
        changes_shape=True,
        group="Reduce",
        description="Average all samples along one axis.",
        icon="functions",
    ),
    "rss": OperationEntry(
        id="rss",
        label="Root-sum-squares over axis",
        operation_type=RootSumSquares,
        changes_shape=True,
        group="Reduce",
        description="Root-sum-of-squares combine along one axis (coil combine).",
        icon="analytics",
    ),
    "sum": OperationEntry(
        id="sum",
        label="Sum over axis",
        operation_type=Sum,
        changes_shape=True,
        group="Reduce",
        description="Sum all samples along one axis.",
        icon="functions",
    ),
    "max": OperationEntry(
        id="max",
        label="Maximum over axis",
        operation_type=Maximum,
        changes_shape=True,
        group="Reduce",
        description="Maximum-intensity projection along one axis.",
        icon="vertical_align_top",
    ),
    "min": OperationEntry(
        id="min",
        label="Minimum over axis",
        operation_type=Minimum,
        changes_shape=True,
        group="Reduce",
        description="Minimum-intensity projection along one axis.",
        icon="vertical_align_bottom",
    ),
    "centered_fft": OperationEntry(
        id="centered_fft",
        label="Centered FFT",
        operation_type=CenteredFFT,
        group="Transform",
        description="Centered forward FFT along one axis (image to k-space).",
        icon="waves",
    ),
    "centered_ifft": OperationEntry(
        id="centered_ifft",
        label="Centered iFFT",
        operation_type=CenteredIFFT,
        group="Transform",
        description="Centered inverse FFT along one axis (k-space to image).",
        icon="waves",
    ),
    "fftshift": OperationEntry(
        id="fftshift",
        label="FFT shift",
        operation_type=FFTShift,
        group="Transform",
        description="Swap the halves of one axis (DC to center).",
        icon="sync_alt",
    ),
    "combine_real_imag": OperationEntry(
        id="combine_real_imag",
        label="Combine real/imag axis",
        operation_type=CombineRealImagAxis,
        changes_shape=True,
        group="Complex",
        description="Fold a size-2 axis of real/imag parts into one complex axis.",
        icon="join_inner",
    ),
    "split_complex": OperationEntry(
        id="split_complex",
        label="Split complex axis",
        operation_type=SplitComplexAxis,
        changes_shape=True,
        group="Complex",
        description="Split a size-1 complex axis into a size-2 real/imag axis.",
        icon="call_split",
    ),
}


_LOGGER = logging.getLogger(__name__)

# First-party in-process operation packs.
#
# Unlike a third-party entry-point plugin (arrayscope.operations.plugins), a
# *pack* ships inside the ArrayScope tree and registers its
# ``PluginOperationSpec`` objects directly here.  A pack is optional: each pack
# module self-guards on its backend (e.g. a runnable ``bart`` binary) and
# contributes nothing when that backend is absent -- so ``import arrayscope``
# never touches the backend, and import-health stays green.  Packs reuse the same
# ``PluginOperation`` machinery as entry-point plugins, so a pack op flows
# through the identical opaque materialization / Tier-2 conformance gate.
_PACK_SPECS: dict[str, object] = {}
_PACKS_LOADED = False

# Pack modules that expose ``register()`` (each guards its own backend).
_PACK_MODULES: tuple[str, ...] = (
    "arrayscope.operations.packs.bart_pack",
    "arrayscope.operations.packs.sigpy_pack",
)


def register_pack_operation(spec) -> None:
    """Register one in-process pack operation spec (namespaced, collision-safe).

    Called by a pack module's ``register()``.  The id must be namespaced (carry
    the plugin ``:`` separator) and must not shadow a built-in id -- the same
    rules the entry-point plugin path enforces.
    """

    from arrayscope.operations.plugins import NAMESPACE_SEPARATOR

    operation_id = spec.id
    if NAMESPACE_SEPARATOR not in operation_id:
        raise ValueError(
            f"pack operation id {operation_id!r} must be namespaced "
            f"(contain {NAMESPACE_SEPARATOR!r})"
        )
    if operation_id in OPERATION_REGISTRY:
        raise ValueError(f"pack operation id {operation_id!r} shadows a built-in operation")
    _PACK_SPECS[operation_id] = spec


# User-defined operations (see arrayscope.operations.library).
#
# A *user op* is a third registration source parallel to the first-party packs
# above.  It is authored by the end user (a wrapper JSON + a python file next to
# their session config), so -- unlike a first-party pack -- registry code never
# scans disk for them itself: that would let one broken user file fail every
# ``all_operations()`` enumeration (including the smoke harness) on an unrelated
# machine.  Instead ``arrayscope.operations.library.refresh_user_operations``
# owns the disk scan and drives registration here via
# :func:`register_user_operation`, exactly the way a pack drives
# :func:`register_pack_operation`.  Ids must live in the ``user:`` namespace, so
# a user op can never shadow a built-in or a pack op.  Because the specs are the
# same ``PluginOperationSpec`` objects the packs use, a user op flows through the
# identical ``PluginOperation`` materialization / recipe round-trip path.
_USER_SPECS: dict[str, object] = {}

_USER_NAMESPACE_PREFIX = "user:"


def register_user_operation(spec) -> None:
    """Register one user-defined operation spec (``user:`` namespace, no shadow).

    Called by :func:`arrayscope.operations.library.refresh_user_operations`.
    The id must live in the ``user:`` namespace and must not shadow a built-in
    -- the same collision safety the pack / entry-point paths enforce.
    """

    operation_id = spec.id
    if not operation_id.startswith(_USER_NAMESPACE_PREFIX):
        raise ValueError(
            f"user operation id {operation_id!r} must start with {_USER_NAMESPACE_PREFIX!r}"
        )
    if operation_id in OPERATION_REGISTRY:
        raise ValueError(f"user operation id {operation_id!r} shadows a built-in operation")
    _USER_SPECS[operation_id] = spec


def _ensure_user_operations_for(operation_id: str) -> None:
    """Lazily scan the user ops dir when a ``user:`` id is not yet registered.

    A recipe / CLI can reference a ``user:`` op before any UI surface has
    triggered the library's disk scan (e.g. ``create_operation`` from a loaded
    recipe on a fresh process). Drive the scan on demand so the id resolves.
    The import is function-local: the library imports the registry, not the
    other way round, so this avoids an import cycle. The scan never raises.
    """

    if not operation_id.startswith(_USER_NAMESPACE_PREFIX) or operation_id in _USER_SPECS:
        return
    from arrayscope.operations import library

    library._ensure_user_operations()


def unregister_user_operation(operation_id: str) -> None:
    """Drop a single user-op registration (used when a wrapper is removed)."""

    _USER_SPECS.pop(operation_id, None)


def _reset_user_operations() -> None:
    """Clear all user-op registration (the library re-drives it on refresh)."""

    _USER_SPECS.clear()


def load_operation_packs() -> None:
    """Import first-party packs and let them register (idempotent, lazy).

    Importing a pack module is side-effect-free; registration is driven here by
    calling each module's ``register()``, which no-ops when its backend is
    absent.  This is reset-safe: after ``_reset_operation_packs`` the cached
    modules are re-asked to register.
    """

    global _PACKS_LOADED
    if _PACKS_LOADED:
        return
    _PACKS_LOADED = True
    for module_name in _PACK_MODULES:
        try:
            module = importlib.import_module(module_name)
            register = getattr(module, "register", None)
            if callable(register):
                register()
        except Exception:  # pragma: no cover - a broken pack must not break the app
            _LOGGER.exception("failed to load operation pack %s", module_name)


def _reset_operation_packs() -> None:
    """Clear pack registration (test seam for re-loading with a changed backend)."""

    global _PACKS_LOADED
    _PACK_SPECS.clear()
    _PACKS_LOADED = False


def _pack_operation_entry(spec) -> OperationEntry:
    from arrayscope.operations.plugins import PluginOperation

    return OperationEntry(
        id=spec.id,
        label=spec.label,
        operation_type=PluginOperation,
        parameters=tuple(spec.parameters),
        changes_shape=bool(spec.changes_shape),
        requires_axis=bool(spec.requires_axis),
        group=getattr(spec, "group", "Other"),
        description=getattr(spec, "description", ""),
        icon=getattr(spec, "icon", "data_array"),
    )


def operation_entries():
    """Built-in operation entries only (concrete dataclass operation types)."""

    return tuple(OPERATION_REGISTRY.values())


def all_operations() -> tuple[OperationEntry, ...]:
    """Every operation the dock can offer: built-ins + installed in-process packs.

    This is the enumeration the operation dock / command palette use so pack ops
    (e.g. the BART pack, and any future pack) are offered alongside the built-ins.
    ``operation_entries`` stays built-ins-only for callers that assume concrete
    dataclass operations.
    """

    load_operation_packs()
    return (
        *OPERATION_REGISTRY.values(),
        *(_pack_operation_entry(spec) for spec in _PACK_SPECS.values()),
        *(_pack_operation_entry(spec) for spec in _USER_SPECS.values()),
    )


def get_operation_entry(operation_id: str) -> OperationEntry:
    entry = OPERATION_REGISTRY.get(operation_id)
    if entry is not None:
        return entry

    load_operation_packs()
    pack_spec = _PACK_SPECS.get(operation_id)
    if pack_spec is not None:
        return _pack_operation_entry(pack_spec)

    _ensure_user_operations_for(operation_id)
    user_spec = _USER_SPECS.get(operation_id)
    if user_spec is not None:
        return _pack_operation_entry(user_spec)

    from arrayscope.operations import plugins

    if plugins.is_plugin_operation_id(operation_id):
        return plugins.plugin_operation_entry(operation_id)
    if plugins.NAMESPACE_SEPARATOR in operation_id:
        raise ValueError(
            f"unknown or uninstalled plugin operation: {operation_id} "
            "(is the providing package installed?)"
        )
    raise ValueError(f"unknown operation id: {operation_id}")


def create_operation(operation_id: str, axis=None, parameters: Mapping[str, object] | None = None):
    if operation_id not in OPERATION_REGISTRY:
        from arrayscope.operations import plugins

        load_operation_packs()
        _ensure_user_operations_for(operation_id)
        spec = _PACK_SPECS.get(operation_id) or _USER_SPECS.get(operation_id)
        if spec is not None:
            # Prime the plugin spec cache so PluginOperation resolution (create,
            # region-honor adjudication, recipe round-trip) works with no entry
            # point.  Re-primed each call -> robust to ``_reset_plugin_cache``.
            plugins._SPEC_CACHE[operation_id] = spec
            return plugins.create_plugin_operation(operation_id, axis=axis, parameters=parameters)

        if plugins.is_plugin_operation_id(operation_id):
            return plugins.create_plugin_operation(operation_id, axis=axis, parameters=parameters)

    entry = get_operation_entry(operation_id)
    parameters = dict(parameters or {})
    kwargs = {}

    if entry.requires_axis:
        if axis is None:
            raise ValueError(f"operation {operation_id} requires an axis")
        kwargs["axis"] = int(axis)

    for parameter in entry.parameters:
        if parameter.name not in parameters:
            # A declared default fills a missing value; a defaultless parameter
            # still raises -- recipes/CLI must not silently drop a required value.
            if parameter.default is not None:
                parameters[parameter.name] = parameter.default
            else:
                raise ValueError(f"operation {operation_id} requires parameter {parameter.name}")
        value = parameters[parameter.name]
        if parameter.kind == "int":
            value = int(value)
        elif parameter.kind == "float":
            value = float(value)
        kwargs[parameter.name] = value

    return entry.operation_type(**kwargs)


def operation_id_for(operation) -> str:
    from arrayscope.operations.plugins import PluginOperation

    if isinstance(operation, PluginOperation):
        return operation.plugin_id

    operation_type = type(operation)
    for entry in OPERATION_REGISTRY.values():
        if entry.operation_type is operation_type:
            return entry.id
    operation_module = getattr(operation_type, "__module__", "")
    operation_name = getattr(operation_type, "__name__", "")
    for entry in OPERATION_REGISTRY.values():
        entry_type = entry.operation_type
        if (
            getattr(entry_type, "__module__", "") == operation_module
            and getattr(entry_type, "__name__", "") == operation_name
        ):
            return entry.id
    raise ValueError(f"operation type is not registered: {operation_type.__name__}")


def operation_parameter_value(operation, name):
    """Read one parameter value off any operation.

    Built-in operations store each declared parameter as an attribute; a
    :class:`~arrayscope.operations.plugins.PluginOperation` keeps them in its
    opaque ``params`` mapping instead. Reading via ``getattr`` alone therefore
    raises ``AttributeError`` for a parameterized plugin op -- this normalizes
    both.
    """

    from arrayscope.operations.plugins import PluginOperation

    if isinstance(operation, PluginOperation):
        return dict(operation.params).get(name)
    return getattr(operation, name, None)


def describe_operation(operation) -> str:
    operation_id = operation_id_for(operation)
    entry = get_operation_entry(operation_id)
    label = entry.label.rstrip(".")
    parts = [label]
    if entry.requires_axis:
        axis = operation.axis
        # "Mean over axis" + 2 -> "Mean over axis 2", not "... axis axis 2".
        parts = [f"{label} {axis}"] if label.endswith(" over axis") else [f"{label} · axis {axis}"]
    for parameter in entry.parameters:
        parts.append(f"{parameter.name}={operation_parameter_value(operation, parameter.name)}")
    return " ".join(parts)
