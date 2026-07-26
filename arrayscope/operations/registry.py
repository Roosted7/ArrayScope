"""Registry for ArrayScope dimension operations."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass

from arrayscope.operations.input_slots import OperationInputSlot, ResolvedSlot, SlotBinding
from arrayscope.operations.pipeline import (
    CenteredFFT,
    CenteredIFFT,
    Clip,
    CombineRealImagAxis,
    Conjugate,
    Crop,
    CumulativeSum,
    Difference,
    FFTShift,
    Gradient,
    HardThreshold,
    ImaginaryPart,
    LogMagnitude,
    Magnitude,
    Maximum,
    Mean,
    Median,
    Minimum,
    Normalize,
    Offset,
    Pad,
    Percentile,
    Phase,
    Power,
    RealPart,
    Resample,
    ReverseAxis,
    Roll,
    RootSumSquares,
    Scale,
    SoftThreshold,
    SplitComplexAxis,
    Squeeze,
    StandardDeviation,
    Sum,
    Transpose,
    Variance,
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
    "Pointwise",
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
    "magnitude",
    "log_magnitude",
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
    unavailable_reason: str = ""
    input_slots: tuple[OperationInputSlot, ...] = ()


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
    "magnitude": OperationEntry(
        id="magnitude",
        label="Magnitude",
        operation_type=Magnitude,
        requires_axis=False,
        group="Complex",
        description="Take the absolute value of every sample.",
        icon="equalizer",
    ),
    "phase": OperationEntry(
        id="phase",
        label="Phase",
        operation_type=Phase,
        requires_axis=False,
        group="Complex",
        description="Return each sample's phase angle in radians.",
        icon="rotate_right",
    ),
    "real": OperationEntry(
        id="real",
        label="Real part",
        operation_type=RealPart,
        requires_axis=False,
        group="Complex",
        description="Select the real component of every sample.",
        icon="looks_one",
    ),
    "imag": OperationEntry(
        id="imag",
        label="Imaginary part",
        operation_type=ImaginaryPart,
        requires_axis=False,
        group="Complex",
        description="Select the imaginary component of every sample.",
        icon="looks_two",
    ),
    "log_magnitude": OperationEntry(
        id="log_magnitude",
        label="Log magnitude...",
        operation_type=LogMagnitude,
        parameters=(
            OperationParameter(
                "epsilon",
                "Epsilon",
                kind="float",
                default=1e-6,
                minimum=1e-12,
                maximum=1.0,
                step=1e-6,
                description="Positive floor applied before the natural logarithm.",
            ),
        ),
        requires_axis=False,
        group="Complex",
        description="Natural log of magnitude with a configurable positive floor.",
        icon="ssid_chart",
    ),
    "scale": OperationEntry(
        id="scale",
        label="Scale...",
        operation_type=Scale,
        parameters=(
            OperationParameter(
                "factor",
                "Factor",
                kind="float",
                default=1.0,
                minimum=-1e6,
                maximum=1e6,
                step=0.1,
                description="Real multiplier applied to every sample.",
            ),
        ),
        requires_axis=False,
        group="Pointwise",
        description="Multiply every sample by a real scalar without changing dtype.",
        icon="close",
    ),
    "offset": OperationEntry(
        id="offset",
        label="Offset...",
        operation_type=Offset,
        parameters=(
            OperationParameter(
                "value",
                "Offset",
                kind="float",
                default=0.0,
                minimum=-1e6,
                maximum=1e6,
                step=0.1,
                description="Real value added to every sample.",
            ),
        ),
        requires_axis=False,
        group="Pointwise",
        description="Add a real scalar to every sample without changing dtype.",
        icon="add",
    ),
    "power": OperationEntry(
        id="power",
        label="Power...",
        operation_type=Power,
        parameters=(
            OperationParameter(
                "exponent",
                "Exponent",
                kind="float",
                default=2.0,
                minimum=-16.0,
                maximum=16.0,
                step=0.1,
                description="Power; complex inputs use NumPy's principal branch.",
            ),
        ),
        requires_axis=False,
        group="Pointwise",
        description="Raise every sample to a power, preserving the input dtype.",
        icon="superscript",
    ),
    "clip": OperationEntry(
        id="clip",
        label="Clip...",
        operation_type=Clip,
        parameters=(
            OperationParameter(
                "minimum",
                "Minimum",
                kind="float",
                default=-1.0,
                minimum=-1e9,
                maximum=1e9,
                step=0.1,
                description="Lower inclusive bound.",
            ),
            OperationParameter(
                "maximum",
                "Maximum",
                kind="float",
                default=1.0,
                minimum=-1e9,
                maximum=1e9,
                step=0.1,
                description="Upper inclusive bound.",
            ),
        ),
        requires_axis=False,
        group="Pointwise",
        description="Clip real data, or real and imaginary components independently.",
        icon="compress",
    ),
    "soft_threshold": OperationEntry(
        id="soft_threshold",
        label="Soft threshold...",
        operation_type=SoftThreshold,
        parameters=(
            OperationParameter(
                "threshold",
                "Threshold",
                kind="float",
                default=0.1,
                minimum=0.0,
                maximum=1e6,
                step=0.01,
                description="Shrink magnitude by this non-negative amount.",
            ),
        ),
        requires_axis=False,
        group="Pointwise",
        description="Magnitude soft-threshold while preserving phase and dtype.",
        icon="filter_alt",
    ),
    "hard_threshold": OperationEntry(
        id="hard_threshold",
        label="Hard threshold...",
        operation_type=HardThreshold,
        parameters=(
            OperationParameter(
                "threshold",
                "Threshold",
                kind="float",
                default=0.1,
                minimum=0.0,
                maximum=1e6,
                step=0.01,
                description="Zero samples whose magnitude is below this value.",
            ),
        ),
        requires_axis=False,
        group="Pointwise",
        description="Magnitude hard-threshold while preserving phase and dtype.",
        icon="filter_alt_off",
    ),
    "normalize": OperationEntry(
        id="normalize",
        label="L2 normalize over axis",
        operation_type=Normalize,
        group="Transform",
        description="Divide each axis-line by its L2 norm; zero lines remain zero.",
        icon="straighten",
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
    "std": OperationEntry(
        id="std",
        label="Standard deviation over axis",
        operation_type=StandardDeviation,
        changes_shape=True,
        group="Reduce",
        description="Sample standard deviation along one axis (ddof=1).",
        icon="functions",
    ),
    "var": OperationEntry(
        id="var",
        label="Variance over axis",
        operation_type=Variance,
        changes_shape=True,
        group="Reduce",
        description="Sample variance along one axis (ddof=1).",
        icon="functions",
    ),
    "median": OperationEntry(
        id="median",
        label="Median over axis",
        operation_type=Median,
        changes_shape=True,
        group="Reduce",
        description="Median along one axis; complex values use NumPy ordering.",
        icon="align_vertical_center",
    ),
    "percentile": OperationEntry(
        id="percentile",
        label="Percentile over axis...",
        operation_type=Percentile,
        parameters=(
            OperationParameter(
                "q",
                "Percentile",
                kind="float",
                default=50.0,
                minimum=0.0,
                maximum=100.0,
                step=1.0,
                description="Percentile from 0 through 100, inclusive.",
            ),
        ),
        changes_shape=True,
        group="Reduce",
        description="Percentile along one axis; complex components reduce independently.",
        icon="percent",
    ),
    "roll": OperationEntry(
        id="roll",
        label="Roll / circular shift...",
        operation_type=Roll,
        parameters=(
            OperationParameter(
                "amount",
                "Amount",
                default=0,
                minimum=-1_000_000,
                maximum=1_000_000,
                step=1,
                description="Signed circular shift in samples.",
            ),
        ),
        group="Transform",
        description="Circularly shift one axis; negative amounts shift toward lower indices.",
        icon="360",
    ),
    "pad": OperationEntry(
        id="pad",
        label="Pad axis...",
        operation_type=Pad,
        parameters=(
            OperationParameter(
                "before",
                "Before",
                default=0,
                minimum=0,
                maximum=1_000_000,
                step=1,
                description="Samples added before the axis.",
            ),
            OperationParameter(
                "after",
                "After",
                default=0,
                minimum=0,
                maximum=1_000_000,
                step=1,
                description="Samples added after the axis.",
            ),
            OperationParameter(
                "mode",
                "Mode",
                default=0,
                minimum=0,
                maximum=2,
                step=1,
                description="0 = zero, 1 = edge, 2 = reflect.",
            ),
        ),
        changes_shape=True,
        group="Reshape",
        description="Pad one axis asymmetrically using zero, edge, or reflect values.",
        icon="border_outer",
    ),
    "resample": OperationEntry(
        id="resample",
        label="Resample axis...",
        operation_type=Resample,
        parameters=(
            OperationParameter(
                "factor",
                "Factor",
                kind="float",
                default=1.0,
                minimum=0.01,
                maximum=100.0,
                step=0.05,
                description="Fractional output/input length ratio.",
            ),
            OperationParameter(
                "order",
                "Spline order",
                default=1,
                minimum=0,
                maximum=3,
                step=1,
                description="Interpolation order from 0 (nearest) through 3 (cubic).",
            ),
            OperationParameter(
                "mode",
                "Boundary mode",
                default=2,
                minimum=0,
                maximum=2,
                step=1,
                description="0 = zero, 1 = edge, 2 = reflect.",
            ),
        ),
        changes_shape=True,
        group="Reshape",
        description="Fractionally resample one axis with sample-centred spline interpolation.",
        icon="aspect_ratio",
    ),
    "transpose": OperationEntry(
        id="transpose",
        label="Transpose / swap axes...",
        operation_type=Transpose,
        parameters=(
            OperationParameter(
                "other_axis",
                "Other axis",
                default=1,
                minimum=0,
                step=1,
                description="Second axis in the permutation.",
            ),
        ),
        changes_shape=True,
        group="Reshape",
        description="Permute an array by swapping the selected axis with another axis.",
        icon="swap_calls",
    ),
    "squeeze": OperationEntry(
        id="squeeze",
        label="Squeeze singleton axis",
        operation_type=Squeeze,
        changes_shape=True,
        group="Reshape",
        description="Remove one selected axis whose length is exactly one.",
        icon="unfold_less",
    ),
    "difference": OperationEntry(
        id="difference",
        label="Difference along axis",
        operation_type=Difference,
        changes_shape=True,
        group="Transform",
        description="First forward difference along one axis (output is one sample shorter).",
        icon="difference",
    ),
    "gradient": OperationEntry(
        id="gradient",
        label="Gradient along axis",
        operation_type=Gradient,
        group="Transform",
        description="First-order finite-difference gradient with the input shape preserved.",
        icon="show_chart",
    ),
    "cumulative_sum": OperationEntry(
        id="cumulative_sum",
        label="Cumulative sum along axis",
        operation_type=CumulativeSum,
        group="Transform",
        description="Running sum along one axis without dtype promotion.",
        icon="waterfall_chart",
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
        unavailable_reason=spec.current_unavailable_reason(),
        input_slots=tuple(getattr(spec, "input_slots", ()) or ()),
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


def create_operation(
    operation_id: str,
    axis=None,
    parameters: Mapping[str, object] | None = None,
    *,
    slot_bindings: Mapping[str, SlotBinding | Mapping[str, object]] | None = None,
    slot_resolver=None,
    resolved_slots: Mapping[str, ResolvedSlot] | None = None,
):
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
            return plugins.create_plugin_operation(
                operation_id,
                axis=axis,
                parameters=parameters,
                slot_bindings=slot_bindings,
                slot_resolver=slot_resolver,
                resolved_slots=resolved_slots,
            )

        if plugins.is_plugin_operation_id(operation_id):
            return plugins.create_plugin_operation(
                operation_id,
                axis=axis,
                parameters=parameters,
                slot_bindings=slot_bindings,
                slot_resolver=slot_resolver,
                resolved_slots=resolved_slots,
            )

    entry = get_operation_entry(operation_id)
    if slot_bindings or resolved_slots:
        raise ValueError(f"built-in operation {operation_id} does not declare input slots")
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


def operation_slot_bindings(operation) -> dict[str, SlotBinding]:
    """Read bound auxiliary sources without exposing process-local slot data."""

    from arrayscope.operations.plugins import PluginOperation

    if not isinstance(operation, PluginOperation):
        return {}
    return dict(operation.slot_bindings)


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
    for slot in entry.input_slots:
        binding = operation_slot_bindings(operation).get(slot.name)
        label = "" if binding is None else binding.label
        parts.append(f"{slot.name}={label or (binding.kind if binding is not None else 'unbound')}")
    return " ".join(parts)
