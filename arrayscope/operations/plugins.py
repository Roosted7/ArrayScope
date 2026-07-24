"""Tier-1 external-operation plugin registry.

Third-party pip packages contribute whole-array operations to ArrayScope by
advertising an entry point in the ``arrayscope.operations`` group.  Tier-1
semantics are deliberately narrow: a plugin op is **OPAQUE** (it materializes
the whole array on CPU), **whole-array** (it makes no region/partial claims --
those are Tier-2, out of scope), and **cache-stage-able** (its output is a
legitimate stage boundary).  A plugin contributes a pure
``fn(ndarray) -> ndarray`` plus a shape/dtype adapter; this module wraps that
into a :class:`PluginOperation` that satisfies the same pipeline-step interface
the built-in operations use, so it flows through the existing opaque
materialization path rather than a parallel one.

Discovery is lazy by construction.  Entry-point *names* are enumerated at
registry-build time (cheap metadata read, no import), but the plugin module is
only imported -- ``entry_point.load()`` -- on the first *actual use* of that op
(constructing it, applying it, or reconstructing it from a recipe).  The loaded
spec is cached thereafter.

Namespaced, collision-safe ids: a plugin op id must carry a namespace
separator (``:``) so a third-party op can never accidentally shadow a built-in
id such as ``crop``.  An entry point that is un-namespaced, or that collides
with a built-in id, is rejected loudly (logged) and ignored -- never silently
honored.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points

from arrayscope.operations.capabilities import (
    OperationCapabilities,
    OperationClass,
    OperationKind,
    default_chunkable_axes,
)
from arrayscope.operations.plugin_conformance import verify_region_conformance
from arrayscope.operations.regions import (
    AxisRegion,
    AxisRegionKind,
    RegionSpec,
    apply_region,
)
from arrayscope.operations.registry import OperationEntry, OperationParameter

Shape = tuple[int, ...]

PLUGIN_ENTRY_POINT_GROUP = "arrayscope.operations"

# A plugin op id must contain this separator so it lives in its own namespace
# and cannot collide with a flat built-in id.
NAMESPACE_SEPARATOR = ":"

_LOGGER = logging.getLogger(__name__)

# Cache of loaded specs keyed by plugin id.  Populated on first use only.
_SPEC_CACHE: dict[str, PluginOperationSpec] = {}

# Tier-2 conformance gate state.
#
# A region (Tier-2) claim is honored -- the op allowed to run per-region -- only
# after the conformance harness passes for that (id, axis, params).  We cache the
# verdict here so the property test runs once per bound op, and expose a stats
# counter so the honor/downgrade decision is observable (testing law #2: never
# silently drop a claim).  Keyed by (plugin_id, axis, sorted-params).
_REGION_HONOR: dict[tuple[str, int | None, tuple[tuple[str, object], ...]], bool] = {}
_REGION_STATS: dict[str, int] = {"verified": 0, "honored": 0, "rejected": 0}

# The synthetic probe the gate property-tests the claim against.  The verdict is
# generic (a claim is a claim about the op, not about one runtime array), so a
# fixed shape/dtype set + fixed seed makes the decision deterministic and
# reproducible -- two reconstructions of the same recipe reach the same verdict.
_REGION_PROBE_SHAPE = (5, 4, 3)
_REGION_PROBE_DTYPES: tuple[object, ...] = ("float64", "int32")
_REGION_PROBE_SEED = 0xA11CE  # fixed, arbitrary -> deterministic verdict
_REGION_PROBE_SAMPLES = 16


def region_conformance_stats() -> dict[str, int]:
    """Observable Tier-2 gate tally: verified / honored / rejected claims."""

    return dict(_REGION_STATS)


@dataclass(frozen=True)
class PluginOperationSpec:
    """What a plugin factory returns to describe one Tier-1 operation.

    The factory named by the entry point takes no arguments and returns one of
    these.  ``fn`` covers the common stateless case; ``build`` covers the
    parametric case (it receives the resolved axis and parameter mapping and
    returns the bound ``fn``).  Exactly one of ``fn``/``build`` must be given.
    ``output_shape``/``output_dtype`` are the shape/dtype adapter; both default
    to identity (shape- and dtype-preserving).
    """

    id: str
    label: str
    fn: Callable[..., object] | None = None
    build: Callable[[int | None, Mapping[str, object]], Callable[..., object]] | None = None
    output_shape: Callable[[Shape, int | None, Mapping[str, object]], Shape] | None = None
    output_dtype: Callable[[object], object] | None = None
    parameters: tuple[OperationParameter, ...] = ()
    requires_axis: bool = False
    changes_shape: bool = False
    # Presentation metadata mirrored onto the synthesized OperationEntry so pack
    # / entry-point ops carry the same group / description / icon the built-ins
    # do. All defaulted -> older specs keep working unchanged.
    group: str = "Other"
    description: str = ""
    icon: str = "data_array"
    # Tier-2 opt-in (default False keeps the op OPAQUE / Tier-1).  When True the
    # author *claims* the op is windowable: it commutes with sub-region reads,
    # i.e. ``fn(whole)[region] == fn(whole[region])`` on every axis, so the
    # engine may run it per-region instead of materializing the whole array.
    # A claim is only a claim: it is honored only after
    # :func:`arrayscope.operations.plugin_conformance.verify_region_conformance`
    # property-tests it (see the registry gate below).  A false claim would show
    # plausible-but-wrong pixels at interactive speed, so an unverified/failing
    # claim is downgraded to the OPAQUE whole-array path, never trusted.
    region_capable: bool = False

    def __post_init__(self) -> None:
        if (self.fn is None) == (self.build is None):
            raise ValueError(f"plugin operation {self.id!r} must declare exactly one of fn / build")

    def resolve_fn(self, axis: int | None, params: Mapping[str, object]) -> Callable[..., object]:
        if self.build is not None:
            built = self.build(axis, dict(params))
            if not callable(built):
                raise TypeError(f"plugin operation {self.id!r} build() did not return a callable")
            return built
        return self.fn  # type: ignore[return-value]

    def resolve_output_shape(
        self, shape: Shape, axis: int | None, params: Mapping[str, object]
    ) -> Shape:
        if self.output_shape is None:
            return tuple(int(size) for size in shape)
        return tuple(int(size) for size in self.output_shape(tuple(shape), axis, dict(params)))

    def resolve_output_dtype(self, input_dtype):
        if self.output_dtype is None:
            return input_dtype
        return self.output_dtype(input_dtype)


@dataclass(frozen=True)
class PluginOperation:
    """A Tier-1 plugin operation bound to a namespaced id, axis, and params.

    The instance stores only its identity (``plugin_id`` + ``axis`` +
    ``params``); the wrapped ``fn`` and adapters are resolved lazily from the
    cached spec.  Keeping the callables out of the dataclass fields makes two
    reconstructions of the same recipe compare equal, which is exactly what the
    recipe round-trip requires.
    """

    plugin_id: str
    axis: int | None = None
    params: tuple[tuple[str, object], ...] = ()

    def _spec(self) -> PluginOperationSpec:
        return load_plugin_spec(self.plugin_id)

    def _params(self) -> dict[str, object]:
        return dict(self.params)

    def _region_honored(self) -> bool:
        # A Tier-2 region claim is honored per (id, axis, params) only after the
        # conformance gate passes; otherwise this op behaves as Tier-1 OPAQUE.
        return is_region_honored(self.plugin_id, self.axis, self.params)

    @property
    def execution_class(self) -> OperationClass:
        # A verified windowable op is a shader-on-read candidate; an OPAQUE
        # (or downgraded) op always CPU-materializes the whole array.
        return OperationClass.SHADER_ON_READ if self._region_honored() else OperationClass.OPAQUE

    def apply(self, data):
        fn = self._spec().resolve_fn(self.axis, self._params())
        return fn(data)

    def output_shape(self, shape: Shape) -> Shape:
        return self._spec().resolve_output_shape(shape, self.axis, self._params())

    def output_dtype(self, input_dtype):
        return self._spec().resolve_output_dtype(input_dtype)

    def capabilities(self, input_shape: Shape, input_dtype=None) -> OperationCapabilities:
        ndim = len(tuple(input_shape))
        all_axes = tuple(range(ndim))
        if self._region_honored():
            # Tier-2 (verified windowable): a per-region ELEMENTWISE stage, the
            # same shape the built-in pointwise ops (Conjugate) use.  It blocks
            # and expands no axis, so a display-axis window shift is a subset of
            # its own input -- the fast path the conformance gate just proved.
            return OperationCapabilities(
                kind=OperationKind.ELEMENTWISE,
                blocking_axes=(),
                chunkable_axes=default_chunkable_axes(OperationKind.ELEMENTWISE, ndim=ndim),
                expands_request_axes=(),
                cache_stage=True,
                can_fuse=True,
            )
        # OPAQUE whole-array stage: it needs every input sample (blocks and
        # expands every input axis), is not chunkable, and is a legitimate
        # cache-stage boundary.  We describe it as a global TRANSFORM so the
        # region planner treats it as whole-axis work; the OPAQUE endpoint
        # classification is carried by ``execution_class`` above.
        return OperationCapabilities(
            kind=OperationKind.TRANSFORM,
            blocking_axes=all_axes,
            chunkable_axes=default_chunkable_axes(
                OperationKind.TRANSFORM, ndim=ndim, blocking_axes=all_axes
            ),
            expands_request_axes=all_axes,
            cache_stage=True,
            can_fuse=False,
        )

    def required_input_region(self, input_shape: Shape, output_region: RegionSpec) -> RegionSpec:
        if self._region_honored():
            # Windowable: producing the output sub-region needs exactly that
            # sub-region of the input (identity map, as elementwise ops declare).
            return output_region
        # Opaque whole-array: producing any output requires the whole input.
        ndim = len(tuple(input_shape))
        return RegionSpec(tuple(AxisRegion(AxisRegionKind.ALL) for _ in range(ndim)))

    def apply_to_region(
        self, data, *, input_region: RegionSpec, output_region: RegionSpec, evaluation_context=None
    ):
        del input_region, evaluation_context
        if self._region_honored():
            # ``data`` is already the requested sub-region (identity input map);
            # the verified windowable fn produces exactly that output sub-region.
            return self.apply(data)
        # ``data`` is the whole input (we requested ALL on every axis); apply
        # the opaque fn to the whole array, then take the requested output slab.
        return apply_region(self.apply(data), output_region)


def _builtin_operation_ids() -> frozenset[str]:
    # Imported lazily to avoid a registry <-> plugins import cycle.
    from arrayscope.operations.registry import OPERATION_REGISTRY

    return frozenset(OPERATION_REGISTRY)


def _discover_entry_points() -> tuple[EntryPoint, ...]:
    try:
        return tuple(entry_points(group=PLUGIN_ENTRY_POINT_GROUP))
    except Exception:  # pragma: no cover - importlib.metadata backend variance
        _LOGGER.exception("failed to enumerate %s entry points", PLUGIN_ENTRY_POINT_GROUP)
        return ()


def discover_plugin_entry_points() -> dict[str, EntryPoint]:
    """Map namespaced plugin op id -> EntryPoint, without importing any plugin.

    Rejects (and logs) entry points that are un-namespaced or that collide with
    a built-in id.  This only reads metadata; ``EntryPoint.load()`` is not
    called here.
    """

    builtin_ids = _builtin_operation_ids()
    discovered: dict[str, EntryPoint] = {}
    for entry_point in _discover_entry_points():
        name = entry_point.name
        if NAMESPACE_SEPARATOR not in name:
            _LOGGER.warning(
                "ignoring un-namespaced plugin operation %r from %r: id must contain %r",
                name,
                _entry_point_origin(entry_point),
                NAMESPACE_SEPARATOR,
            )
            continue
        if name in builtin_ids:
            _LOGGER.warning(
                "ignoring plugin operation %r from %r: it shadows a built-in operation id",
                name,
                _entry_point_origin(entry_point),
            )
            continue
        if name in discovered:
            _LOGGER.warning(
                "ignoring duplicate plugin operation %r from %r: id already provided by %r",
                name,
                _entry_point_origin(entry_point),
                _entry_point_origin(discovered[name]),
            )
            continue
        discovered[name] = entry_point
    return discovered


def plugin_operation_ids() -> tuple[str, ...]:
    """Namespaced ids of installed plugin ops (lazy: no plugin import)."""

    return tuple(sorted(discover_plugin_entry_points()))


def is_plugin_operation_id(operation_id: str) -> bool:
    return operation_id in discover_plugin_entry_points()


def load_plugin_spec(operation_id: str) -> PluginOperationSpec:
    """Load (and cache) the spec for a plugin op, importing its module now.

    Raises a clear :class:`ValueError` when the id is not an installed plugin
    op -- e.g. a recipe referencing an uninstalled third-party package.
    """

    cached = _SPEC_CACHE.get(operation_id)
    if cached is not None:
        return cached

    entry_point = discover_plugin_entry_points().get(operation_id)
    if entry_point is None:
        raise ValueError(
            f"unknown or uninstalled plugin operation: {operation_id!r} "
            f"(no entry point in group {PLUGIN_ENTRY_POINT_GROUP!r})"
        )

    factory = entry_point.load()
    spec = factory()
    if not isinstance(spec, PluginOperationSpec):
        raise TypeError(
            f"plugin operation {operation_id!r} factory returned {type(spec).__name__}, "
            f"expected PluginOperationSpec"
        )
    if spec.id != operation_id:
        raise ValueError(
            f"plugin operation entry point {operation_id!r} declares mismatched id {spec.id!r}"
        )
    _SPEC_CACHE[operation_id] = spec
    return spec


def plugin_operation_entry(operation_id: str) -> OperationEntry:
    """Synthesize an :class:`OperationEntry` for a plugin op (loads its spec)."""

    spec = load_plugin_spec(operation_id)
    return OperationEntry(
        id=spec.id,
        label=spec.label,
        operation_type=PluginOperation,
        parameters=tuple(spec.parameters),
        changes_shape=bool(spec.changes_shape),
        requires_axis=bool(spec.requires_axis),
        group=spec.group,
        description=spec.description,
        icon=spec.icon,
    )


def create_plugin_operation(
    operation_id: str, axis=None, parameters: Mapping[str, object] | None = None
) -> PluginOperation:
    """Build a bound :class:`PluginOperation` (loads the plugin spec)."""

    spec = load_plugin_spec(operation_id)
    parameters = dict(parameters or {})

    resolved_axis: int | None = None
    if spec.requires_axis:
        if axis is None:
            raise ValueError(f"plugin operation {operation_id} requires an axis")
        resolved_axis = int(axis)

    bound_params: list[tuple[str, object]] = []
    for parameter in spec.parameters:
        if parameter.name not in parameters:
            # A declared default fills a missing value; a defaultless parameter
            # still raises (recipes/CLI must not silently drop a required value).
            if parameter.default is not None:
                parameters[parameter.name] = parameter.default
            else:
                raise ValueError(
                    f"plugin operation {operation_id} requires parameter {parameter.name}"
                )
        value = parameters[parameter.name]
        if parameter.kind == "int":
            value = int(value)
        elif parameter.kind == "float":
            value = float(value)
        bound_params.append((parameter.name, value))

    operation = PluginOperation(
        plugin_id=operation_id, axis=resolved_axis, params=tuple(bound_params)
    )
    if spec.region_capable:
        # Adjudicate the Tier-2 claim up front so the honor/downgrade decision
        # (and its warning + stat) happens at construction, not lazily mid-render.
        is_region_honored(operation_id, operation.axis, operation.params)
    return operation


def recipe_item_for_plugin_operation(operation: PluginOperation, *, enabled: bool) -> dict:
    """Serialize a plugin-op step to a recipe item by its namespaced id."""

    item: dict[str, object] = {"id": operation.plugin_id}
    if operation.axis is not None:
        item["axis"] = int(operation.axis)
    if operation.params:
        item["parameters"] = dict(operation.params)
    item["enabled"] = bool(enabled)
    return item


def is_region_honored(
    plugin_id: str, axis: int | None = None, params: tuple[tuple[str, object], ...] = ()
) -> bool:
    """Whether this bound op's Tier-2 region claim passed conformance.

    The verdict is computed once (property-tested) and cached.  A spec that does
    not opt in (``region_capable=False``) is never honored -- it stays a Tier-1
    OPAQUE whole-array op.
    """

    import numpy as np

    spec = load_plugin_spec(plugin_id)
    if not spec.region_capable:
        return False

    key = (plugin_id, axis, tuple(params))
    cached = _REGION_HONOR.get(key)
    if cached is not None:
        return cached

    # Property-test the claim across the probe dtypes.  A claim is honored only
    # if it holds for EVERY probe (one counterexample downgrades it).
    param_map = dict(params)
    honored = True
    first_failure = None
    for dtype in _REGION_PROBE_DTYPES:
        result = verify_region_conformance(
            spec,
            _REGION_PROBE_SHAPE,
            dtype,
            rng=np.random.default_rng(_REGION_PROBE_SEED),
            axis=axis,
            params=param_map,
            samples=_REGION_PROBE_SAMPLES,
        )
        if not result.honored:
            honored = False
            first_failure = result
            break

    _REGION_HONOR[key] = honored
    _REGION_STATS["verified"] += 1
    if honored:
        _REGION_STATS["honored"] += 1
    else:
        _REGION_STATS["rejected"] += 1
        # Downgrade, don't refuse: the underlying fn is still a correct Tier-1
        # OPAQUE op, so we run it whole-array (correct, just not the fast path)
        # rather than break the user's pipeline over a performance annotation.
        # The refusal to trust the fast path is what matters, and it is loud.
        _LOGGER.warning(
            "plugin operation %r declared region_capable but FAILED conformance "
            "(%s); downgrading to OPAQUE whole-array. It will produce correct "
            "pixels but cannot run per-region.",
            plugin_id,
            first_failure.reason if first_failure is not None else "no detail",
        )
    return honored


def _entry_point_origin(entry_point: EntryPoint) -> str:
    dist = getattr(entry_point, "dist", None)
    dist_name = getattr(dist, "name", None)
    return str(dist_name) if dist_name else entry_point.value


def _reset_plugin_cache() -> None:
    """Clear the loaded-spec cache (test seam for re-discovery)."""

    _SPEC_CACHE.clear()
    _REGION_HONOR.clear()
    _REGION_STATS.update(verified=0, honored=0, rejected=0)
