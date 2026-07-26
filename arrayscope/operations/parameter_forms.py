"""Qt-free parameter-form model for operation parameters.

This is the headless foundation the operation-parameter popup (a later UI chunk)
renders from: it turns an :class:`~arrayscope.operations.registry.OperationEntry`
plus the current array *context* (shape + chosen axis) into a
:class:`ParameterForm` of typed, bounded :class:`ParameterField` objects, exposes
read-only :class:`DerivedValue` info lines (e.g. the resulting output length),
applies the interdependence adjustments some ops need (editing crop ``start`` may
nudge ``stop``), and validates the current values.

Keeping this Qt-free means the whole parameter model is unit-testable without a
display, and any UI (dock popup, command palette, a future web view) renders the
same fields from the same source of truth. The default form is derived purely
from the entry's declared parameter metadata (``default`` seeds the value,
``minimum``/``maximum`` bound it); ops that need array-context awareness or
cross-field interdependence register a small provider below, keyed by op id.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from arrayscope.operations.input_slots import SlotBinding, SlotSourceOption
from arrayscope.operations.registry import OperationEntry, OperationParameter

Shape = tuple[int, ...]


@dataclass
class ParameterField:
    """One editable operation parameter, with bounds and a current value."""

    name: str
    label: str
    kind: str  # "int" | "float"
    value: int | float
    minimum: int | float | None = None
    maximum: int | float | None = None
    step: int | float | None = None
    description: str = ""
    read_only: bool = False


@dataclass(frozen=True)
class DerivedValue:
    """A read-only, computed info line (e.g. ``Output length: 42``)."""

    label: str
    text: str


@dataclass
class SlotField:
    """One required auxiliary source rendered beside ordinary parameters."""

    name: str
    label: str
    description: str
    accepts: tuple[str, ...]
    options: tuple[SlotSourceOption, ...] = ()
    binding: SlotBinding = field(default_factory=lambda: SlotBinding(""))


def _coerce(kind: str, value) -> int | float:
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    raise ValueError(f"unknown parameter kind: {kind!r}")


class ParameterForm:
    """A set of parameter fields with optional interdependence + derivation.

    ``adjust`` (if given) is called after every :meth:`set_value` with the form
    and the name of the field that changed, so an op can keep dependent fields
    consistent (crop ``start`` < ``stop``). ``derive`` returns the read-only info
    lines; ``validate`` adds op-specific checks on top of the default bounds
    check.
    """

    def __init__(
        self,
        fields: list[ParameterField],
        *,
        slot_fields: list[SlotField] | None = None,
        adjust: Callable[[ParameterForm, str], None] | None = None,
        derive: Callable[[ParameterForm], list[DerivedValue]] | None = None,
        validate: Callable[[ParameterForm], str | None] | None = None,
    ) -> None:
        self.fields = fields
        self.slot_fields = list(slot_fields or ())
        self._adjust = adjust
        self._derive = derive
        self._validate = validate

    def field(self, name: str) -> ParameterField:
        for candidate in self.fields:
            if candidate.name == name:
                return candidate
        raise KeyError(name)

    def values(self) -> dict[str, int | float]:
        """The parameter mapping to hand to ``create_operation``."""

        return {candidate.name: candidate.value for candidate in self.fields}

    def bindings(self) -> dict[str, SlotBinding]:
        return {candidate.name: candidate.binding for candidate in self.slot_fields}

    def set_binding(self, name: str, binding: SlotBinding) -> None:
        for candidate in self.slot_fields:
            if candidate.name == name:
                candidate.binding = SlotBinding.from_payload(binding)
                return
        raise KeyError(name)

    def set_value(self, name: str, value) -> None:
        """Set one field's value (coerced), then apply interdependence."""

        target = self.field(name)
        target.value = _coerce(target.kind, value)
        if self._adjust is not None:
            self._adjust(self, name)

    def derived(self) -> list[DerivedValue]:
        if self._derive is None:
            return []
        return self._derive(self)

    def validate(self) -> str | None:
        """Return ``None`` when valid, else a human-readable message."""

        for candidate in self.fields:
            if candidate.minimum is not None and candidate.value < candidate.minimum:
                return f"{candidate.label} must be at least {candidate.minimum}."
            if candidate.maximum is not None and candidate.value > candidate.maximum:
                return f"{candidate.label} must be at most {candidate.maximum}."
        for candidate in self.slot_fields:
            if not candidate.binding.is_bound:
                return f"{candidate.label} requires an input."
            if candidate.binding.kind not in candidate.accepts:
                return (
                    f"{candidate.label} does not accept "
                    f"{candidate.binding.kind.replace('-', ' ')} inputs."
                )
            selected = next(
                (option for option in candidate.options if option.binding == candidate.binding),
                None,
            )
            if selected is not None and not selected.available:
                return selected.unavailable_reason or f"{candidate.label} is unavailable."
        if self._validate is not None:
            return self._validate(self)
        return None


# --- default form (metadata-driven) ------------------------------------------


def _seed_value(parameter: OperationParameter) -> int | float:
    """Initial value for a field: the declared default, else the low bound, else 0."""

    if parameter.default is not None:
        source = parameter.default
    elif parameter.minimum is not None:
        source = parameter.minimum
    else:
        source = 0
    return _coerce(parameter.kind, source)


def _default_field(parameter: OperationParameter) -> ParameterField:
    return ParameterField(
        name=parameter.name,
        label=parameter.label,
        kind=parameter.kind,
        value=_seed_value(parameter),
        minimum=parameter.minimum,
        maximum=parameter.maximum,
        step=parameter.step,
        description=parameter.description,
    )


def _default_form(entry: OperationEntry, *, shape: Shape | None, axis: int | None) -> ParameterForm:
    del shape, axis  # the default form is purely metadata-driven
    return ParameterForm([_default_field(parameter) for parameter in entry.parameters])


# --- per-op form providers (context / interdependence aware) -----------------


def _axis_length(shape: Shape | None, axis: int | None) -> int | None:
    if shape is None or axis is None:
        return None
    resolved = axis % len(shape) if shape else axis
    if 0 <= resolved < len(shape):
        return int(shape[resolved])
    return None


def _crop_form(entry: OperationEntry, *, shape: Shape | None, axis: int | None) -> ParameterForm:
    """crop: start/stop bounded by axis length, kept start < stop, output length line.

    Editing one bound past the other nudges the *other* bound to preserve
    start < stop -- the canonical "changing one parameter changes another" case.
    """

    del entry
    length = _axis_length(shape, axis)
    start_max = (length - 1) if length is not None else None
    stop_max = length
    start = ParameterField(
        name="start",
        label="Start",
        kind="int",
        value=0,
        minimum=0,
        maximum=start_max,
        step=1,
        description="First index kept (inclusive).",
    )
    stop = ParameterField(
        name="stop",
        label="Stop",
        kind="int",
        value=length if length is not None else 1,
        minimum=1,
        maximum=stop_max,
        step=1,
        description="Index one past the last kept (exclusive).",
    )

    def adjust(form: ParameterForm, name: str) -> None:
        start_field = form.field("start")
        stop_field = form.field("stop")
        if start_field.value < stop_field.value:
            return
        if name == "start":
            # Editing start crossed stop: push stop up; if stop is pinned to its
            # ceiling, pull start back down instead so start < stop always holds.
            stop_field.value = start_field.value + 1
            if stop_max is not None and stop_field.value > stop_max:
                stop_field.value = stop_max
                start_field.value = stop_field.value - 1
        else:
            # Editing stop crossed start: pull start down; if start is at 0, push
            # stop back up.
            start_field.value = stop_field.value - 1
            if start_field.value < 0:
                start_field.value = 0
                stop_field.value = 1

    def derive(form: ParameterForm) -> list[DerivedValue]:
        span = form.field("stop").value - form.field("start").value
        return [DerivedValue("Output length", str(max(span, 0)))]

    return ParameterForm([start, stop], adjust=adjust, derive=derive)


def _percentile_form(
    entry: OperationEntry, *, shape: Shape | None, axis: int | None
) -> ParameterForm:
    """percentile: pin q to its mathematical bounds and show the sample count."""

    del entry
    length = _axis_length(shape, axis)
    q = ParameterField(
        name="q",
        label="Percentile",
        kind="float",
        value=50.0,
        minimum=0.0,
        maximum=100.0,
        step=1.0,
        description="Percentile from 0 through 100, inclusive.",
    )

    def derive(form: ParameterForm) -> list[DerivedValue]:
        del form
        if length is None:
            return []
        return [DerivedValue("Samples on axis", str(length))]

    return ParameterForm([q], derive=derive)


def _pad_form(entry: OperationEntry, *, shape: Shape | None, axis: int | None) -> ParameterForm:
    """pad: context defaults centre the axis in its next power-of-two length."""

    del entry
    length = _axis_length(shape, axis)
    if length is None:
        before_value = after_value = 0
    else:
        target = 1 << max(length - 1, 0).bit_length()
        total = target - length
        before_value = total // 2
        after_value = total - before_value
    before = ParameterField(
        name="before",
        label="Before",
        kind="int",
        value=before_value,
        minimum=0,
        maximum=1_000_000,
        step=1,
        description="Samples added before the axis.",
    )
    after = ParameterField(
        name="after",
        label="After",
        kind="int",
        value=after_value,
        minimum=0,
        maximum=1_000_000,
        step=1,
        description="Samples added after the axis.",
    )
    mode = ParameterField(
        name="mode",
        label="Mode",
        kind="int",
        value=0,
        minimum=0,
        maximum=2,
        step=1,
        description="0 = zero, 1 = edge, 2 = reflect.",
    )

    def derive(form: ParameterForm) -> list[DerivedValue]:
        if length is None:
            return []
        output = length + int(form.field("before").value) + int(form.field("after").value)
        return [
            DerivedValue("Current length", str(length)),
            DerivedValue("Output length", str(output)),
        ]

    return ParameterForm([before, after, mode], derive=derive)


def _resample_form(
    entry: OperationEntry, *, shape: Shape | None, axis: int | None
) -> ParameterForm:
    """resample: fractional factor plus exact context-derived output length."""

    del entry
    length = _axis_length(shape, axis)
    factor = ParameterField(
        name="factor",
        label="Factor",
        kind="float",
        value=1.0,
        minimum=0.01,
        maximum=100.0,
        step=0.05,
        description="Fractional output/input length ratio.",
    )
    order = ParameterField(
        name="order",
        label="Spline order",
        kind="int",
        value=1,
        minimum=0,
        maximum=3,
        step=1,
        description="Interpolation order from 0 (nearest) through 3 (cubic).",
    )
    mode = ParameterField(
        name="mode",
        label="Boundary mode",
        kind="int",
        value=2,
        minimum=0,
        maximum=2,
        step=1,
        description="0 = zero, 1 = nearest, 2 = reflect.",
    )

    def derive(form: ParameterForm) -> list[DerivedValue]:
        if length is None:
            return []
        value = max(1, int(length * float(form.field("factor").value) + 0.5))
        return [
            DerivedValue("Current length", str(length)),
            DerivedValue("Output length", str(value)),
        ]

    return ParameterForm([factor, order, mode], derive=derive)


def _transpose_form(
    entry: OperationEntry, *, shape: Shape | None, axis: int | None
) -> ParameterForm:
    """transpose: bound the partner axis and default to the next axis."""

    del entry
    ndim = len(shape) if shape is not None else None
    resolved_axis = None if axis is None or ndim is None else axis % ndim
    default = 0 if ndim is None else (int(resolved_axis or 0) + 1) % ndim
    other_axis = ParameterField(
        name="other_axis",
        label="Other axis",
        kind="int",
        value=default,
        minimum=0,
        maximum=None if ndim is None else ndim - 1,
        step=1,
        description="Second axis in the permutation.",
    )
    return ParameterForm([other_axis])


# Keyed by op id. An op absent here falls back to the metadata-driven default
# form, which already honors the parameter default / min / max / step / desc.
_FORM_PROVIDERS: dict[str, Callable[..., ParameterForm]] = {
    "crop": _crop_form,
    "percentile": _percentile_form,
    "pad": _pad_form,
    "resample": _resample_form,
    "transpose": _transpose_form,
}


def build_parameter_form(
    entry: OperationEntry,
    *,
    shape: Shape | None = None,
    axis: int | None = None,
    slot_options: dict[str, tuple[SlotSourceOption, ...]] | None = None,
    slot_bindings: dict[str, SlotBinding] | None = None,
) -> ParameterForm | None:
    """Build the parameter form for ``entry`` in the current array context.

    Returns ``None`` for a parameterless op (nothing to render). Ops with a
    registered provider get context/interdependence-aware fields; every other
    parameterized op gets the metadata-driven default form.
    """

    if not entry.parameters and not entry.input_slots:
        return None
    provider = _FORM_PROVIDERS.get(entry.id) if entry.parameters else None
    form = (
        provider(entry, shape=shape, axis=axis)
        if provider is not None
        else _default_form(entry, shape=shape, axis=axis)
    )
    options_by_name = dict(slot_options or {})
    bindings_by_name = dict(slot_bindings or {})
    form.slot_fields = [
        SlotField(
            name=slot.name,
            label=slot.label,
            description=slot.description,
            accepts=slot.accepts,
            options=tuple(options_by_name.get(slot.name, ())),
            binding=bindings_by_name.get(slot.name, SlotBinding("")),
        )
        for slot in entry.input_slots
    ]
    return form
