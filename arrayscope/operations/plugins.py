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
materialization path rather than a parallel one. Output shape, dtype, and any
Tier-2 region claim are characterized together before planner use.

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
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from inspect import signature

import numpy as np

from arrayscope.core.axis_utils import (
    DEFAULT_AXIS_ARITY,
    normalize_axis_arity,
    normalize_axis_selection,
)
from arrayscope.operations.cancellation import EvaluationCancelled, call_with_cancellation
from arrayscope.operations.capabilities import (
    OperationCapabilities,
    OperationKind,
    default_chunkable_axes,
)
from arrayscope.operations.input_slots import (
    OperationInputSlot,
    ResolvedSlot,
    SlotBinding,
    unresolved_slot,
)
from arrayscope.operations.plugin_conformance import (
    CharacterizationMismatch,
    CharacterizationUnavailable,
    OperationCharacterization,
    characterize_operation,
    record_runtime_mismatch,
)
from arrayscope.operations.regions import (
    AxisRegion,
    AxisRegionKind,
    RegionSpec,
    apply_region,
    axis_in_region_result,
    replace_region_axis,
    take_axis_region,
)
from arrayscope.operations.registry import (
    OperationEntry,
    OperationParameter,
    _lift_axes_from_parameters,
    coerce_parameter_value,
)

Shape = tuple[int, ...]

PLUGIN_ENTRY_POINT_GROUP = "arrayscope.operations"

# A plugin op id must contain this separator so it lives in its own namespace
# and cannot collide with a flat built-in id.
NAMESPACE_SEPARATOR = ":"

_LOGGER = logging.getLogger(__name__)

# Cache of loaded specs keyed by plugin id.  Populated on first use only.
_SPEC_CACHE: dict[str, PluginOperationSpec] = {}

_REGION_PROBE_SHAPE = (5, 4, 3)


def _accepts_arguments(function, *arguments) -> bool:
    try:
        signature(function).bind(*arguments)
    except (TypeError, ValueError):
        return False
    return True


@dataclass(frozen=True)
class PluginOperationSpec:
    """What a plugin factory returns to describe one Tier-1 operation.

    The factory named by the entry point takes no arguments and returns one of
    these.  ``fn`` covers the common stateless case; ``build`` covers the
    parametric case (it receives the resolved axis *selection* and parameter
    mapping and returns the bound ``fn``).  Exactly one of ``fn``/``build`` must
    be given.  The selection is an ``int`` for a ``requires_axis`` spec and a
    sorted tuple for a ``requires_axes`` one; it is ``None`` for a spec that
    takes neither, and also for a ``maps_over_slabs`` spec, whose callable sees
    one slab at a time and never needs to know which axes that slab spans.
    ``output_shape``/``output_dtype`` are compatibility adapters; observed
    characterization is authoritative and both adapters default to identity.
    """

    id: str
    label: str
    fn: Callable[..., object] | None = None
    build: (
        Callable[
            [int | None, Mapping[str, object], Mapping[str, np.ndarray]],
            Callable[..., object],
        ]
        | None
    ) = None
    output_shape: Callable[[Shape, int | None, Mapping[str, object]], Shape] | None = None
    output_dtype: Callable[[object], object] | None = None
    parameters: tuple[OperationParameter, ...] = ()
    input_slots: tuple[OperationInputSlot, ...] = ()
    requires_axis: bool = False
    # The axis-SET counterpart of ``requires_axis`` (see
    # ``registry.OperationEntry``): the resolved selection reaches ``build`` as
    # the ``axes`` keyword and a plain ``fn`` through ``functools.partial``-free
    # wrapping below. Mutually exclusive with ``requires_axis``.
    requires_axes: bool = False
    axis_arity: tuple[int, int | None] = DEFAULT_AXIS_ARITY
    # "Map me over the rest." The callable is handed each SLAB spanning exactly
    # the selected axes -- a 2-D image when two axes are chosen -- and the
    # runtime broadcasts it over every other axis. Implies ``requires_axes``:
    # there is no slab without an axis set to define it.
    #
    # This is the half of subset-of-dims that an author cannot write themselves
    # today: a Tier-1 plugin receives the whole array and must loop by hand,
    # which also makes the whole array one indivisible cache stage. Declaring it
    # here instead lets the capability layer say blocking = the selected axes,
    # chunkable = the complement, so scrubbing a broadcast axis re-runs one slab
    # rather than the volume.
    maps_over_slabs: bool = False
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
    # A registered operation may be visible but deliberately non-runnable (an
    # unfinished template, unresolved environment, or imported command awaiting
    # review). ``availability`` is evaluated lazily so a repaired environment
    # is picked up without importing or executing the operation body.
    unavailable_reason: str = ""
    availability: Callable[[], str | None] | None = None
    # Shared declarative runtime metadata. It is exported by the operation
    # manager and copied verbatim when a command-backed system example is
    # duplicated into a user operation.
    runtime: str = "python"
    runtime_config: Mapping[str, object] | None = None
    environment_id: str = ""
    # Dynamic identity is part of the characterization key. Linked user files
    # provide a callable that returns their current mtime; command-backed specs
    # provide the command template text.
    source_identity: object = None

    def __post_init__(self) -> None:
        if (self.fn is None) == (self.build is None):
            raise ValueError(f"plugin operation {self.id!r} must declare exactly one of fn / build")
        if self.requires_axis and self.requires_axes:
            raise ValueError(
                f"plugin operation {self.id!r} declares both requires_axis and requires_axes; "
                "an operation takes one axis or an axis set, not both"
            )
        if self.maps_over_slabs and not self.requires_axes:
            raise ValueError(
                f"plugin operation {self.id!r} declares maps_over_slabs without requires_axes; "
                "a slab is defined by the axis set it spans"
            )
        object.__setattr__(
            self,
            "axis_arity",
            normalize_axis_arity(self.axis_arity) if self.requires_axes else DEFAULT_AXIS_ARITY,
        )
        slot_names = [slot.name for slot in self.input_slots]
        if len(slot_names) != len(set(slot_names)):
            raise ValueError(f"plugin operation {self.id!r} has duplicate input-slot names")
        parameter_names = {parameter.name for parameter in self.parameters}
        overlap = sorted(parameter_names & set(slot_names))
        if overlap:
            raise ValueError(
                f"plugin operation {self.id!r} reuses parameter names as input slots: {overlap}"
            )
        if self.input_slots and self.build is not None:
            try:
                signature(self.build).bind(None, {}, {})
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"plugin operation {self.id!r} declares input slots, so build() "
                    "must accept (axis, params, slots)"
                ) from exc

    def resolve_fn(
        self,
        axis: int | None,
        params: Mapping[str, object],
        slots: Mapping[str, np.ndarray] | None = None,
    ) -> Callable[..., object]:
        reason = self.current_unavailable_reason()
        if reason:
            raise RuntimeError(f"operation {self.id!r} is unavailable: {reason}")
        if self.build is not None:
            arguments = (axis, dict(params), dict(slots or {}))
            if not self.input_slots and not _accepts_arguments(self.build, *arguments):
                # Slotless third-party builders predate Bundle E's third
                # argument. Their two-argument contract remains valid; a
                # slot-bearing definition is validated above and never takes
                # this incumbent path.
                built = self.build(*arguments[:2])
            else:
                built = self.build(*arguments)
            if not callable(built):
                raise TypeError(f"plugin operation {self.id!r} build() did not return a callable")
            return built
        if slots:
            return lambda data: self.fn(data, **dict(slots))  # type: ignore[misc]
        return self.fn  # type: ignore[return-value]

    def current_unavailable_reason(self) -> str:
        """Why this operation cannot run right now, or ``""``.

        The ``availability`` callable is a *probe of the outside world* -- it
        resolves environments, renders command templates, and stats
        executables. Any of that can raise on a definition the user is allowed
        to author: a command operation combining a conda environment with
        shell execution makes ``build_command`` raise ValueError, and both
        halves of that combination are one click apart in the manager.

        This is the boundary where such a raise must stop. It is called from
        ``registry._pack_operation_entry``, so an escaping exception does not
        disable one operation -- it takes down ``all_operations()``, and with
        it the operation manager, the add popup, and the command palette, i.e.
        every surface the user would repair the definition from. Reporting the
        exception text as the unavailable reason keeps the definition listed,
        editable, and self-describing.
        """

        if self.unavailable_reason:
            return str(self.unavailable_reason)
        if self.availability is not None:
            try:
                return str(self.availability() or "")
            except Exception as exc:  # never let one definition break the listing
                return f"Availability check failed: {exc}"
        return ""

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
class _SlabSpec:
    """Presents a slab-mapped spec's callable to the characterizer at SLAB rank.

    The conformance harness is duck-typed on ``id`` / ``resolve_fn`` /
    ``source_identity`` / ``region_capable`` precisely so a second caller can
    reuse it, and this is that caller: a slab-mapped operation's contract is a
    contract about *one slab*, so the probes must be slab-shaped. Probing the
    broadcast wrapper instead would multiply every probe by the number of slabs
    in the probe array and still tell us nothing the slab rule does not.

    The id is namespaced apart from the operation's own so a slab
    characterization and a whole-array one can never collide in the shared
    cache. ``region_capable`` is False: a Tier-2 window claim is about the whole
    array and is not something a slab rule can establish.
    """

    spec: PluginOperationSpec
    region_capable: bool = False

    @property
    def id(self) -> str:
        return f"{self.spec.id}#slab"

    @property
    def source_identity(self):
        return self.spec.source_identity

    def resolve_fn(self, axis, params, slots=None):
        # The axis selection is deliberately dropped: a slab-mapped callable
        # sees a slab and is not told which axes it spans (see the field note
        # on ``maps_over_slabs``).
        del axis
        return self.spec.resolve_fn(None, params, slots)


@dataclass(frozen=True)
class SlabPlan:
    """What a slab-mapped operation does to one input signature."""

    axes: tuple[int, ...]
    slab_shape: Shape
    output_slab_shape: Shape
    output_shape: Shape
    output_dtype: np.dtype
    #: True when the callable returns a slab of exactly the shape it was given.
    #: Only then does the operation earn the broadcast-chunkable declaration.
    shape_preserving: bool


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
    # The axis SET, for a ``requires_axes`` spec. Sorted and unique by
    # construction (``normalize_axis_selection``), which is what lets two
    # reconstructions of the same recipe compare equal even if one wrote the
    # axes in the other order.
    axes: tuple[int, ...] = ()
    params: tuple[tuple[str, object], ...] = ()
    slot_bindings: tuple[tuple[str, SlotBinding], ...] = ()
    resolved_slots: tuple[tuple[str, ResolvedSlot], ...] = ()
    _shape_hint: Shape | None = field(default=None, compare=False, repr=False)
    _dtype_hint: object = field(default=None, compare=False, repr=False)
    _characterization_hint: OperationCharacterization | None = field(
        default=None, compare=False, repr=False
    )

    def _spec(self) -> PluginOperationSpec:
        return load_plugin_spec(self.plugin_id)

    def _params(self) -> dict[str, object]:
        return dict(self.params)

    def _axis_selection(self):
        """What ``build`` receives: an ``int``, a sorted tuple, or ``None``.

        ``None`` for a slab-mapped op even though it has an axis set: its
        callable sees a slab, and telling it which axes that slab spans would
        invite it to index by them -- against a slab whose numbering no longer
        matches the array's.
        """

        if self._spec().maps_over_slabs:
            return None
        return self.axes if self.axes else self.axis

    def _slot_inputs(self) -> dict[str, ResolvedSlot]:
        return dict(self.resolved_slots)

    def _slot_arrays(self) -> dict[str, np.ndarray | str]:
        """Each bound slot's materialized value.

        Array-backed slots produce an ``ndarray``; a ``file`` slot produces its
        path, which the operation opens itself.
        """

        reason = self.current_unavailable_reason()
        if reason:
            raise RuntimeError(f"operation {self.plugin_id!r} is unavailable: {reason}")
        return {name: source.materialize() for name, source in self.resolved_slots}

    def current_unavailable_reason(self) -> str:
        reason = self._spec().current_unavailable_reason()
        if reason:
            return reason
        for slot in self._spec().input_slots:
            source = self._slot_inputs().get(slot.name)
            if source is None:
                return f"Input “{slot.label}” is not bound."
            if source.unavailable_reason:
                return source.unavailable_reason
        return ""

    def _characterize(
        self, shape: Shape, dtype, *, exact_input=None, cancellation_token=None
    ) -> OperationCharacterization:
        characterization = characterize_operation(
            self._spec(),
            tuple(shape),
            np.dtype(dtype),
            axis=self._axis_selection(),
            params=self._params(),
            slots=self._slot_inputs(),
            exact_input=exact_input,
            cancellation_token=cancellation_token,
        )
        object.__setattr__(self, "_shape_hint", tuple(shape))
        object.__setattr__(self, "_dtype_hint", np.dtype(dtype))
        object.__setattr__(self, "_characterization_hint", characterization)
        return characterization

    # -- slab mapping ("apply me per slab, broadcast me over the rest") -------

    def _maps_over_slabs(self) -> bool:
        return bool(self._spec().maps_over_slabs)

    def _slab_axes(self, ndim: int) -> tuple[int, ...]:
        return normalize_axis_selection(
            self.axes, ndim=int(ndim), label=f"operation {self.plugin_id!r} axes"
        )

    def slab_plan(self, input_shape: Shape, input_dtype=None) -> SlabPlan:
        """Characterize the SLAB callable and project it onto the whole array.

        The projection is the whole trick: the operation is a map from a slab to
        a slab, so a rule fitted at slab rank predicts the whole-array output by
        substituting the predicted slab sizes back into the applied axis
        positions. Every other axis is untouched, which is exactly what
        "broadcast over the rest" means.
        """

        input_shape = tuple(int(size) for size in input_shape)
        dtype = np.dtype("float32" if input_dtype is None else input_dtype)
        axes = self._slab_axes(len(input_shape))
        slab_shape = tuple(input_shape[axis] for axis in axes)
        characterization = characterize_operation(
            _SlabSpec(self._spec()),
            slab_shape,
            dtype,
            axis=None,
            params=self._params(),
            slots=self._slot_inputs(),
        )
        output_slab_shape = characterization.output_shape
        if len(output_slab_shape) != len(axes):
            # A slab-mapped callable that changes the slab's *rank* has no
            # whole-array shape at all: there is no position to put the extra
            # (or missing) axis in that every slab would agree on.
            raise CharacterizationUnavailable(
                f"operation {self.plugin_id!r} maps over slabs but changed the slab rank "
                f"from {len(axes)} to {len(output_slab_shape)}"
            )
        output_shape = list(input_shape)
        for axis, size in zip(axes, output_slab_shape, strict=True):
            output_shape[axis] = int(size)
        return SlabPlan(
            axes=axes,
            slab_shape=slab_shape,
            output_slab_shape=tuple(int(size) for size in output_slab_shape),
            output_shape=tuple(output_shape),
            output_dtype=np.dtype(characterization.output_dtype),
            shape_preserving=bool(
                characterization.predictable and characterization.shape_rule.kind == "identity"
            ),
        )

    def _apply_over_slabs(self, data, axes, *, cancellation_token=None):
        """Run the callable once per slab and reassemble in the input's axis order."""

        array = np.asarray(data)
        fn = self._spec().resolve_fn(None, self._params(), self._slot_arrays())
        rank = array.ndim
        trailing = tuple(range(rank - len(axes), rank))
        moved = np.moveaxis(array, axes, trailing)
        broadcast_shape = moved.shape[: rank - len(axes)]
        slabs = [
            np.asarray(call_with_cancellation(fn, moved[index], cancellation_token))
            for index in np.ndindex(*broadcast_shape)
        ]
        output_slab_shape = tuple(slabs[0].shape)
        if any(tuple(slab.shape) != output_slab_shape for slab in slabs):
            # Every slab must agree, or the results do not form an array -- and
            # the shape rule fitted from one slab would be a lie for the others.
            raise CharacterizationMismatch(
                f"operation {self.plugin_id!r} returned differently shaped slabs; "
                "a slab-mapped operation must map every slab the same way"
            )
        stacked = np.stack(slabs).reshape(broadcast_shape + output_slab_shape)
        return np.moveaxis(stacked, trailing, axes)

    def _slab_region_input_shape(self, array, input_region: RegionSpec) -> Shape:
        """A stand-in input shape carrying the real size of every applied axis.

        Only the applied axes' sizes reach ``slab_plan``, and those axes are
        always ALL in the region we asked for, so their real sizes are readable
        off the region result. The rest are placeholders -- they exist so the
        rank matches, and nothing consults them.
        """

        ndim = len(input_region.axes)
        shape = [1] * ndim
        for axis in self._slab_axes(ndim):
            shape[axis] = int(np.shape(array)[axis_in_region_result(input_region, axis)])
        return tuple(shape)

    def _apply_slab_region(
        self, data, *, input_region: RegionSpec, output_region: RegionSpec, evaluation_context=None
    ):
        """Run the slab map on the sub-region the planner chunked out for us.

        This is where the broadcast pays: for a shape-preserving slab op the
        planner asked only for the applied axes whole, so ``data`` is one chunk
        of the complement -- often a single slab -- and the loop below runs the
        callable that many times instead of once per slab in the volume.
        """

        cancellation_token = getattr(evaluation_context, "cancellation_token", None)
        array = np.asarray(data)
        plan = self.slab_plan(self._slab_region_input_shape(array, input_region), array.dtype)
        if not plan.shape_preserving:
            # Whole-array branch: we asked for everything, so ``data`` is the
            # whole input and the requested output slab is taken afterwards.
            return apply_region(
                self.apply(array, evaluation_context=evaluation_context), output_region
            )
        axes = plan.axes
        # Every applied axis is ALL in the region we asked for, so none of them
        # is a POINT and each keeps a position in the region result.
        slab_axes = tuple(axis_in_region_result(input_region, axis) for axis in axes)
        result = self._apply_over_slabs(array, slab_axes, cancellation_token=cancellation_token)
        # Descending: taking a POINT output region drops that axis and shifts
        # every higher position down; the ones still to be used are all lower.
        for axis, slab_axis in sorted(zip(axes, slab_axes, strict=True), reverse=True):
            result = take_axis_region(
                result, output_region.axes[axis], result.shape[slab_axis], axis=slab_axis
            )
        return result

    def characterize_output(
        self, shape: Shape, dtype, *, exact_input=None
    ) -> tuple[Shape, np.dtype]:
        if self._maps_over_slabs():
            plan = self.slab_plan(shape, dtype)
            return plan.output_shape, plan.output_dtype
        characterization = self._characterize(shape, dtype, exact_input=exact_input)
        return characterization.output_shape, characterization.output_dtype

    def _region_honored(self, shape: Shape | None = None, dtype=None) -> bool:
        if self.slot_bindings or self._maps_over_slabs():
            # A slab-mapped operation is never Tier-2 windowable: it is applied
            # whole *within* each slab, so a sub-window of an applied axis is a
            # different computation, not a subset of this one. It gets its own
            # (stronger, declared rather than probed) region contract below.
            return False
        if shape is None:
            shape = self._shape_hint or _REGION_PROBE_SHAPE
        if dtype is None:
            dtype = self._dtype_hint or np.dtype("float64")
        return self._characterize(shape, dtype).region_honored

    def apply(self, data, *, evaluation_context=None):
        # The evaluation context is where a cancellation token lives, and an
        # out-of-process runtime is the one thing here that can run long enough
        # to need one. The token reaches the runtime wrapper only; the author's
        # own ``fn(array)`` never sees it (see ``call_with_cancellation``).
        cancellation_token = getattr(evaluation_context, "cancellation_token", None)
        if cancellation_token is not None and getattr(cancellation_token, "cancelled", False):
            # Characterization below runs the operation too -- five to nine
            # probes for a new signature -- so an already-cancelled request has
            # to be refused before it, not after.
            raise EvaluationCancelled()
        array = np.asarray(data)
        if self._maps_over_slabs():
            plan = self.slab_plan(array.shape, array.dtype)
            result = self._apply_over_slabs(array, plan.axes, cancellation_token=cancellation_token)
            if tuple(result.shape) != plan.output_shape or result.dtype != plan.output_dtype:
                raise CharacterizationMismatch(
                    f"operation {self.plugin_id!r} produced shape {tuple(result.shape)} / "
                    f"dtype {result.dtype} over slabs, but its slab characterization "
                    f"predicted {plan.output_shape} / {plan.output_dtype}; the result "
                    "was withheld"
                )
            return result
        characterization = self._characterize(
            array.shape, array.dtype, cancellation_token=cancellation_token
        )
        fn = self._spec().resolve_fn(self._axis_selection(), self._params(), self._slot_arrays())
        result = np.asarray(call_with_cancellation(fn, data, cancellation_token))
        predicted_shape = characterization.output_shape
        predicted_dtype = characterization.output_dtype
        if tuple(result.shape) != predicted_shape or result.dtype != predicted_dtype:
            record_runtime_mismatch(
                characterization,
                actual_shape=result.shape,
                actual_dtype=result.dtype,
            )
            raise CharacterizationMismatch(
                f"operation {self.plugin_id!r} returned shape {tuple(result.shape)} / "
                f"dtype {result.dtype}, but its cached characterization predicted "
                f"{predicted_shape} / {predicted_dtype}; the result was withheld and "
                "the operation was demoted to unpredictable"
            )
        return result

    def output_shape(self, shape: Shape) -> Shape:
        dtype = self._dtype_hint or np.dtype("float32")
        if self._maps_over_slabs():
            return self.slab_plan(shape, dtype).output_shape
        return self._characterize(shape, dtype).output_shape

    def output_dtype(self, input_dtype):
        shape = self._shape_hint or _REGION_PROBE_SHAPE
        if self._maps_over_slabs():
            return self.slab_plan(shape, input_dtype).output_dtype
        return self._characterize(shape, input_dtype).output_dtype

    def output_axis_sources(self, input_shape):
        """Per output axis, the input axis it derives from (``None`` if new).

        The characterized shape rule already answers this per output axis, so
        the planner's input-space -> output-space capability translation (see
        ``arrayscope.operations.capabilities.expanded_output_axes``) reads the
        same source of truth the shape prediction does.
        """

        if self._maps_over_slabs():
            # Rank-preserving by construction: a slab-mapped operation may
            # resize an applied axis but never adds or removes one.
            return tuple(range(len(tuple(input_shape))))
        dtype = self._dtype_hint or np.dtype("float32")
        rule = self._characterize(tuple(input_shape), dtype).shape_rule
        return tuple(axis_rule.source_axis for axis_rule in rule.axes)

    def output_axes(self, axes):
        from dataclasses import replace

        from arrayscope.core.axis_info import AxisInfo, renumber_default_axes

        input_shape = tuple(axis.size for axis in axes)
        dtype = self._dtype_hint or np.dtype("float32")
        if self._maps_over_slabs():
            plan = self.slab_plan(input_shape, dtype)
            applied = set(plan.axes)
            return tuple(
                replace(axis_info, size=size, spacing=None, origin=None)
                if index in applied and size != axis_info.size
                else replace(axis_info, size=size)
                for index, (axis_info, size) in enumerate(
                    zip(tuple(axes), plan.output_shape, strict=True)
                )
            )
        rule = self._characterize(input_shape, dtype).shape_rule
        output = []
        for output_axis, axis_rule in enumerate(rule.axes):
            if axis_rule.source_axis is None:
                output.append(
                    AxisInfo(
                        id=f"{self.plugin_id}:axis-{output_axis}",
                        label=f"Dim {output_axis}",
                        size=axis_rule.predict(input_shape),
                    )
                )
                continue
            source = axes[axis_rule.source_axis]
            size = axis_rule.predict(input_shape)
            if axis_rule.mode == "offset" and int(axis_rule.value) == 0:
                output.append(replace(source, size=size))
            else:
                output.append(replace(source, size=size, spacing=None, origin=None))
        # A default label and a default id are both *positional*, so carrying
        # them through a rank change duplicates them: a stack-at-axis-0 op
        # produced ['Dim 0', 'Dim 0', 'Dim 1'] because the new axis 0 and the
        # surviving input axis 0 both claimed position 0. Renumbering the
        # defaulted ones (and only those) restores uniqueness without touching
        # a label or coordinate the user actually chose.
        return renumber_default_axes(tuple(output))

    def capabilities(self, input_shape: Shape, input_dtype=None) -> OperationCapabilities:
        ndim = len(tuple(input_shape))
        all_axes = tuple(range(ndim))
        if self._maps_over_slabs():
            plan = self.slab_plan(input_shape, input_dtype)
            if plan.shape_preserving:
                # The declaration the whole feature exists for: only the applied
                # axes block, so the complement stays chunkable and scrubbing a
                # broadcast axis costs one slab instead of the volume.
                #
                # It is gated on the callable *preserving the slab shape*
                # deliberately. Chunked evaluation assembles output chunks at
                # positions derived from the input chunk's own extents; a
                # callable that resizes its slab makes those positions a
                # function of the shape rule, which is exactly the kind of
                # silent misplacement this codebase refuses to risk for speed.
                # A resizing slab op is still correct -- it just falls to the
                # whole-array branch below.
                return OperationCapabilities(
                    kind=OperationKind.TRANSFORM,
                    blocking_axes=plan.axes,
                    chunkable_axes=default_chunkable_axes(
                        OperationKind.TRANSFORM, ndim=ndim, blocking_axes=plan.axes
                    ),
                    expands_request_axes=plan.axes,
                    cache_stage=True,
                    can_fuse=False,
                    notes=("Applied per slab; broadcast over every other axis.",),
                )
            return OperationCapabilities(
                kind=OperationKind.TRANSFORM,
                blocking_axes=all_axes,
                chunkable_axes=(),
                expands_request_axes=all_axes,
                cache_stage=True,
                can_fuse=False,
                notes=("Slab-mapped but slab-resizing; evaluated whole-array.",),
            )
        characterization = self._characterize(
            input_shape, np.dtype("float32") if input_dtype is None else input_dtype
        )
        if characterization.region_honored and not self.slot_bindings:
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
        ndim = len(tuple(input_shape))
        if self._maps_over_slabs():
            plan = self.slab_plan(input_shape, self._dtype_hint)
            if plan.shape_preserving:
                # Whole applied axes, whatever was asked of the rest: exactly
                # the built-in axis-set transforms' contract.
                region = output_region
                for axis in plan.axes:
                    region = replace_region_axis(region, axis, AxisRegion(AxisRegionKind.ALL))
                return region
            return RegionSpec(tuple(AxisRegion(AxisRegionKind.ALL) for _ in range(ndim)))
        if self._region_honored(input_shape):
            # Windowable: producing the output sub-region needs exactly that
            # sub-region of the input (identity map, as elementwise ops declare).
            return output_region
        # Opaque whole-array: producing any output requires the whole input.
        return RegionSpec(tuple(AxisRegion(AxisRegionKind.ALL) for _ in range(ndim)))

    def apply_to_region(
        self, data, *, input_region: RegionSpec, output_region: RegionSpec, evaluation_context=None
    ):
        if self._maps_over_slabs():
            return self._apply_slab_region(
                data,
                input_region=input_region,
                output_region=output_region,
                evaluation_context=evaluation_context,
            )
        del input_region
        if self._region_honored():
            # ``data`` is already the requested sub-region (identity input map);
            # the verified windowable fn produces exactly that output sub-region.
            return self.apply(data, evaluation_context=evaluation_context)
        # ``data`` is the whole input (we requested ALL on every axis); apply
        # the opaque fn to the whole array, then take the requested output slab.
        return apply_region(self.apply(data, evaluation_context=evaluation_context), output_region)


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
        requires_axes=bool(spec.requires_axes),
        axis_arity=spec.axis_arity if spec.requires_axes else DEFAULT_AXIS_ARITY,
        group=spec.group,
        description=spec.description,
        icon=spec.icon,
        unavailable_reason=spec.current_unavailable_reason(),
        input_slots=tuple(spec.input_slots),
    )


def create_plugin_operation(
    operation_id: str,
    axis=None,
    parameters: Mapping[str, object] | None = None,
    *,
    axes=None,
    ndim: int | None = None,
    slot_bindings: Mapping[str, SlotBinding | Mapping[str, object]] | None = None,
    slot_resolver=None,
    resolved_slots: Mapping[str, ResolvedSlot] | None = None,
) -> PluginOperation:
    """Build a bound :class:`PluginOperation` (loads the plugin spec)."""

    spec = load_plugin_spec(operation_id)
    # A form-built mapping may carry the axis set under the reserved key; the
    # explicit keyword wins. See ``registry.AXES_VALUE_KEY``.
    parameters, axes = _lift_axes_from_parameters(parameters, axes)
    parameters = dict(parameters or {})

    resolved_axis: int | None = None
    resolved_axes: tuple[int, ...] = ()
    if spec.requires_axis:
        if axis is None:
            raise ValueError(f"plugin operation {operation_id} requires an axis")
        resolved_axis = int(axis)
    elif spec.requires_axes:
        if axes is None:
            raise ValueError(f"plugin operation {operation_id} requires an axis set")
        resolved_axes = normalize_axis_selection(
            axes,
            ndim=ndim,
            arity=spec.axis_arity,
            label=f"plugin operation {operation_id} axes",
        )

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
        bound_params.append(
            (
                parameter.name,
                coerce_parameter_value(
                    parameter.kind, parameters[parameter.name], choices=parameter.choices
                ),
            )
        )

    raw_bindings = dict(slot_bindings or {})
    supplied_resolved = dict(resolved_slots or {})
    declared_names = {slot.name for slot in spec.input_slots}
    extras = sorted((set(raw_bindings) | set(supplied_resolved)) - declared_names)
    if extras:
        raise ValueError(f"plugin operation {operation_id} has unknown input slots: {extras}")
    bound_slots: list[tuple[str, SlotBinding]] = []
    sources: list[tuple[str, ResolvedSlot]] = []
    for slot in spec.input_slots:
        binding = SlotBinding.from_payload(raw_bindings.get(slot.name, {}))
        if binding.kind and binding.kind not in slot.accepts:
            raise ValueError(
                f"input slot {slot.name!r} does not accept binding kind {binding.kind!r}"
            )
        source = supplied_resolved.get(slot.name)
        if source is None and slot_resolver is not None and binding.is_bound:
            try:
                source = slot_resolver(slot, binding)
            except Exception as exc:
                source = unresolved_slot(
                    binding,
                    f"Input “{slot.label}” cannot resolve: {exc}",
                )
        if source is None:
            reason = (
                f"Input “{slot.label}” is not bound."
                if not binding.is_bound
                else f"Input “{slot.label}” cannot resolve in this document."
            )
            source = unresolved_slot(binding, reason)
        bound_slots.append((slot.name, binding))
        sources.append((slot.name, source))

    return PluginOperation(
        plugin_id=operation_id,
        axis=resolved_axis,
        axes=resolved_axes,
        params=tuple(bound_params),
        slot_bindings=tuple(bound_slots),
        resolved_slots=tuple(sources),
    )


def recipe_item_for_plugin_operation(operation: PluginOperation, *, enabled: bool) -> dict:
    """Serialize a plugin-op step to a recipe item by its namespaced id."""

    item: dict[str, object] = {"id": operation.plugin_id}
    if operation.axis is not None:
        item["axis"] = int(operation.axis)
    if operation.axes:
        item["axes"] = [int(axis) for axis in operation.axes]
    if operation.params:
        item["parameters"] = dict(operation.params)
    if operation.slot_bindings:
        item["inputs"] = {name: binding.to_payload() for name, binding in operation.slot_bindings}
    item["enabled"] = bool(enabled)
    return item


def _entry_point_origin(entry_point: EntryPoint) -> str:
    dist = getattr(entry_point, "dist", None)
    dist_name = getattr(dist, "name", None)
    return str(dist_name) if dist_name else entry_point.value
