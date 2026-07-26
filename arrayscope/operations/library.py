"""First-class operation library: grouping / layout, hiding, and user ops.

This is the *presentation & configuration* layer over
:mod:`arrayscope.operations.registry` -- the analogue of
:mod:`arrayscope.display.colormap_library` for operations rather than
colormaps.  It answers "which operations exist, in what groups, in what order,
and which are hidden", and it owns the headline feature of this chunk:
**user-defined operations**.

Layering / imports: Qt-free by discipline (``QtCore``-only, and only
transitively via :mod:`arrayscope.app.user_dirs`), so it stays importable in
headless / worker contexts.  It depends on the registry; the registry never
depends on it (the registry exposes ``register_user_operation`` and the library
drives it, exactly the way a first-party pack drives ``register_pack_operation``).

User operations on disk
-----------------------
Each user op is a wrapper JSON ``<slug>.json`` in
:func:`user_operations_directory`, plus -- for *import*-mode ops -- a copied
``<slug>.py`` code file alongside it.  The wrapper schema (``version`` 1)::

    {
        "format": "arrayscope-operation",
        "version": 1,
        "id": "user:<slug>",
        "label": "...",
        "description": "...",
        "group": "User",
        "icon": "extension",
        "runtime": "python",
        "source": {
            "mode": "import" | "link",
            "path": "<file>.py",  # relative for import, absolute for link
            "callable": "function_name",
        },
        "requires_axis": true,
        "changes_shape": false,
        "parameters": [
            {
                "name": ...,
                "label": ...,
                "kind": "int" | "float",
                "default": ...,
                "minimum": ...,
                "maximum": ...,
                "step": ...,
                "description": ...,
            }
        ],
    }

``runtime`` values ``"python"``, ``"command"``, ``"julia"``, and ``"matlab"``
are concrete.  Command-backed operations use explicit tokenization plus an
``npy`` or ``cfl`` array handoff; Python can name an out-of-process execution
environment.  Shape and dtype are discovered by bounded characterization when
an input signature is first planned, so ``changes_shape`` is presentation
metadata rather than a declared adapter.
A broken wrapper or code file **never** breaks startup: it is caught, logged, recorded in
:func:`user_operation_problems`, and skipped, so the rest of the library loads.

The operation manager UI (a separate chunk) builds on this module: it renders
:func:`grouped_operations`, persists arrangements through
:func:`apply_library_layout`, toggles :func:`set_operation_hidden`, and calls
:func:`create_empty_user_operation` / :func:`duplicate_operation` /
:func:`update_user_operation_source` / :func:`remove_user_operation` /
:func:`update_user_operation`.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import inspect
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from arrayscope.app import user_dirs
from arrayscope.operations import command_runtime, environments, registry
from arrayscope.operations.plugins import PluginOperationSpec
from arrayscope.operations.registry import (
    COMMON_OPERATION_IDS,
    DEFAULT_GROUP_ORDER,
    OperationEntry,
    OperationParameter,
)

_LOGGER = logging.getLogger(__name__)

WRAPPER_FORMAT = "arrayscope-operation"
LAYOUT_FORMAT = "arrayscope-operation-layout"

SUPPORTED_RUNTIME = "python"
SUPPORTED_RUNTIMES: frozenset[str] = frozenset({"python", "command", "julia", "matlab"})
# Compatibility name retained for callers that used the old schema constant.
RESERVED_RUNTIMES: frozenset[str] = frozenset()

# Default "More" fold-out groups a UI surface tucks below the common ops. This is
# presentation policy the layout can override via ``more_groups``.
#
# Everything except the pinned "Common" section folds away by default. That
# looks aggressive but it is what keeps the add popup short: the native toolbox
# is 37 operations, and "Common" already holds the head of each group, so what
# remains in the groups *is* the less-used tail. The earlier default named the
# optional backend groups instead, which stopped partitioning anything the
# moment those packs were demoted -- the popup became one flat 37-row scroll
# with an empty fold-out, which is exactly what the fold-out exists to prevent.
#
# Anything a user reaches for often is one checkbox away from the pinned
# section in the operation manager, and the layout can override this list
# wholesale, so this default trades one click for a popup that opens small.
DEFAULT_MORE_GROUPS: tuple[str, ...] = (
    "Reduce",
    "Transform",
    "Complex",
    "Pointwise",
    "Reshape",
    "SigPy",
    "BART",
    "User",
    "Other",
)

# Python module namespace user code is imported under (keeps user modules from
# colliding with real packages and with each other).
_USER_MODULE_NAMESPACE = "arrayscope_user_ops"


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_layout_cache: dict | None = None
_listeners: list = []

# Whether the on-disk user ops have been scanned + registered this session.
_user_loaded = False
# (file, message) for every wrapper the loader skipped -- surfaced to the future
# manager UI so a broken user op is *visible*, not silently missing.
_user_problems: list[tuple[str, str]] = []
# Imported user modules cached by (abspath, mtime) so a "link"-mode edit is
# picked up (a bumped mtime is a fresh key -> re-import) while an unchanged file
# imports once.
_module_cache: dict[tuple[str, int], object] = {}


# ---------------------------------------------------------------------------
# Directory
# ---------------------------------------------------------------------------


def user_operations_directory() -> str:
    """Directory holding user-op wrapper JSON + code files (tests monkeypatch this)."""

    return str(user_dirs.user_operations_directory())


def execution_environments() -> tuple[environments.ExecutionEnvironment, ...]:
    """Named environments persisted beside user operation wrappers."""

    return environments.load_environments(user_operations_directory())


def update_execution_environment(**fields) -> environments.ExecutionEnvironment:
    record = environments.upsert_environment(user_operations_directory(), fields)
    refresh_user_operations()
    return record


def remove_execution_environment(environment_id: str) -> bool:
    removed = environments.remove_environment(user_operations_directory(), environment_id)
    if removed:
        refresh_user_operations()
    return removed


def resolve_execution_environment(environment_id: str):
    return environments.resolve_environment(user_operations_directory(), environment_id)


# ---------------------------------------------------------------------------
# Listeners (mirrors colormap_library, incl. dead-Qt-listener self-healing)
# ---------------------------------------------------------------------------


def add_library_listener(callback) -> None:
    if callable(callback) and callback not in _listeners:
        _listeners.append(callback)


def remove_library_listener(callback) -> None:
    """Unregister a listener added with :func:`add_library_listener` (idempotent).

    Widget-bound listeners MUST call this on teardown: the registry is
    process-global, so a listener that outlives its widget turns the next
    mutation into a call on a dead Qt object.
    """

    with contextlib.suppress(ValueError):
        _listeners.remove(callback)


def _notify() -> None:
    for callback in tuple(_listeners):
        try:
            callback()
        except RuntimeError:
            # Listener bound to a Qt widget whose C++ object is gone (window
            # closed without unregistering). Drop it so the registry self-heals
            # instead of retrying the dead wrapper on every mutation.
            remove_library_listener(callback)
        except Exception:
            # No repr of the callback: repr of a dead Qt wrapper raises
            # RuntimeError itself.
            _LOGGER.exception("operation library listener failed")


# ---------------------------------------------------------------------------
# Layout persistence
# ---------------------------------------------------------------------------


def _layout_file() -> str:
    return os.path.join(user_operations_directory(), "layout.json")


def _load_layout() -> dict:
    global _layout_cache
    if _layout_cache is None:
        try:
            with open(_layout_file()) as handle:
                payload = json.load(handle)
            _layout_cache = {
                "group_order": [str(g) for g in payload.get("group_order", [])],
                "op_groups": {str(k): str(v) for k, v in payload.get("op_groups", {}).items()},
                "op_order": {str(k): int(v) for k, v in payload.get("op_order", {}).items()},
                "common_ids": [str(i) for i in payload["common_ids"]]
                if "common_ids" in payload
                else None,
                "more_groups": [str(g) for g in payload["more_groups"]]
                if "more_groups" in payload
                else None,
            }
        except Exception:
            _layout_cache = {
                "group_order": [],
                "op_groups": {},
                "op_order": {},
                "common_ids": None,
                "more_groups": None,
            }
    return _layout_cache


def _save_layout(layout: dict) -> None:
    global _layout_cache
    os.makedirs(user_operations_directory(), exist_ok=True)
    payload: dict = {
        "format": LAYOUT_FORMAT,
        "version": 1,
        "group_order": [str(g) for g in layout.get("group_order", [])],
        "op_groups": {str(k): str(v) for k, v in layout.get("op_groups", {}).items()},
        "op_order": {str(k): int(v) for k, v in layout.get("op_order", {}).items()},
    }
    # Only persist the presentation overrides that were actually set, so an
    # unset override keeps tracking the code default rather than freezing today's.
    if layout.get("common_ids") is not None:
        payload["common_ids"] = [str(i) for i in layout["common_ids"]]
    if layout.get("more_groups") is not None:
        payload["more_groups"] = [str(g) for g in layout["more_groups"]]
    with open(_layout_file(), "w") as handle:
        json.dump(payload, handle, indent=2)
    _layout_cache = None
    _notify()


def effective_common_ids() -> tuple[str, ...]:
    """Ids a UI surface pins to the top -- layout override, else the code default."""

    override = _load_layout()["common_ids"]
    return tuple(override) if override is not None else tuple(COMMON_OPERATION_IDS)


def effective_more_groups() -> tuple[str, ...]:
    """Fold-out "More" groups -- layout override, else the code default."""

    override = _load_layout()["more_groups"]
    return tuple(override) if override is not None else DEFAULT_MORE_GROUPS


def effective_group(entry: OperationEntry) -> str:
    """Group ``entry`` lands in -- the layout override, else its declared group."""

    return _load_layout()["op_groups"].get(entry.id, entry.group)


def apply_library_layout(
    group_order=None,
    op_groups=None,
    op_order=None,
    common_ids=None,
    more_groups=None,
) -> None:
    """Persist a user arrangement.  Any argument left ``None`` keeps its current value."""

    current = _load_layout()
    _save_layout(
        {
            "group_order": current["group_order"] if group_order is None else group_order,
            "op_groups": current["op_groups"] if op_groups is None else op_groups,
            "op_order": current["op_order"] if op_order is None else op_order,
            "common_ids": current["common_ids"] if common_ids is None else common_ids,
            "more_groups": current["more_groups"] if more_groups is None else more_groups,
        }
    )


def reset_layout() -> None:
    """Discard the persisted layout (revert to code defaults)."""

    global _layout_cache
    path = _layout_file()
    if os.path.exists(path):
        os.remove(path)
    _layout_cache = None
    _notify()


# ---------------------------------------------------------------------------
# Hidden operations
# ---------------------------------------------------------------------------


def _hidden_file() -> str:
    return os.path.join(user_operations_directory(), "hidden-operations.json")


def hidden_operations() -> frozenset[str]:
    try:
        with open(_hidden_file()) as handle:
            return frozenset(str(op_id) for op_id in json.load(handle))
    except Exception:
        return frozenset()


def set_operation_hidden(operation_id: str, hidden: bool) -> None:
    ids = set(hidden_operations())
    if bool(hidden):
        ids.add(str(operation_id))
    else:
        ids.discard(str(operation_id))
    os.makedirs(user_operations_directory(), exist_ok=True)
    with open(_hidden_file(), "w") as handle:
        json.dump(sorted(ids), handle)
    _notify()


def reset_operation(operation_id: str) -> bool:
    """Un-hide ``operation_id``.

    For a built-in / pack op this restores it to the listing.  For a user op it
    is a no-op beyond un-hiding (a user op is removed with
    :func:`remove_user_operation`, not "reset").
    """

    if str(operation_id) in hidden_operations():
        set_operation_hidden(operation_id, False)
        return True
    return False


# ---------------------------------------------------------------------------
# Grouped listing
# ---------------------------------------------------------------------------


def _all_entries() -> tuple[OperationEntry, ...]:
    _ensure_user_operations()
    return registry.all_operations()


def grouped_operations(*, include_hidden: bool = False):
    """Ordered ``[(group, [entries])]`` honoring the persisted layout.

    Groups default to each op's declared ``group``; group order defaults to
    :data:`DEFAULT_GROUP_ORDER`; within a group the default order is registration
    order.  The persisted layout overrides any of these.
    """

    layout = _load_layout()
    entries = _all_entries()
    hidden = hidden_operations()
    default_positions = {entry.id: index for index, entry in enumerate(entries)}

    by_group: dict[str, list[OperationEntry]] = {}
    for entry in entries:
        if entry.id in hidden and not include_hidden:
            continue
        by_group.setdefault(effective_group(entry), []).append(entry)

    for group_entries in by_group.values():
        group_entries.sort(
            key=lambda entry: (
                layout["op_order"].get(entry.id, default_positions.get(entry.id, 10_000)),
                entry.label.lower(),
            )
        )

    ordered = [group for group in layout["group_order"] if group in by_group]
    ordered += [
        group for group in DEFAULT_GROUP_ORDER if group in by_group and group not in ordered
    ]
    ordered += [group for group in sorted(by_group) if group not in ordered]
    return [(group, by_group[group]) for group in ordered]


# ---------------------------------------------------------------------------
# User-op loading / registration
# ---------------------------------------------------------------------------


def _load_user_operations() -> None:
    """Scan the ops dir and (re)register every valid wrapper.  Never raises."""

    global _user_loaded, _user_problems
    registry._reset_user_operations()
    problems: list[tuple[str, str]] = []
    directory = user_operations_directory()
    if os.path.isdir(directory):
        for file_name in sorted(os.listdir(directory)):
            if not file_name.endswith(".json"):
                continue
            if file_name in (
                "layout.json",
                "hidden-operations.json",
                environments.ENVIRONMENTS_FILE,
            ):
                continue
            path = os.path.join(directory, file_name)
            try:
                spec = _spec_from_wrapper_file(path)
                registry.register_user_operation(spec)
            except Exception as exc:  # a broken user op must not break the app
                _LOGGER.warning("skipping user operation %s: %s", path, exc)
                problems.append((path, str(exc)))
    _user_problems = problems
    _user_loaded = True


def _ensure_user_operations() -> None:
    if not _user_loaded:
        _load_user_operations()


def refresh_user_operations() -> None:
    """Re-scan the ops dir, re-register, and notify listeners (force reload)."""

    _load_user_operations()
    _notify()


def user_operation_problems() -> list[tuple[str, str]]:
    """``(file, message)`` for every wrapper the loader skipped this session."""

    _ensure_user_operations()
    return list(_user_problems)


def _spec_from_wrapper_file(path: str) -> PluginOperationSpec:
    """Build a :class:`PluginOperationSpec` from one wrapper JSON (validates)."""

    with open(path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("wrapper is not a JSON object")
    if payload.get("format") not in (WRAPPER_FORMAT, None):
        raise ValueError(f"unrecognized wrapper format: {payload.get('format')!r}")

    operation_id = str(payload.get("id") or "")
    if not operation_id.startswith("user:"):
        raise ValueError(f"user operation id must start with 'user:', got {operation_id!r}")

    runtime = str(payload.get("runtime") or SUPPORTED_RUNTIME)
    if runtime not in SUPPORTED_RUNTIMES:
        raise ValueError(f"unknown runtime: {runtime}")

    slug = operation_id[len("user:") :]
    parameters = _parameters_from_payload(payload.get("parameters", ()))
    requires_axis = bool(payload.get("requires_axis", False))
    changes_shape = bool(payload.get("changes_shape", False))

    template = payload.get("template")
    unavailable_reason = ""
    if isinstance(template, dict):
        unavailable_reason = str(template.get("reason") or "")
    review = payload.get("review")
    if isinstance(review, dict) and bool(review.get("required")):
        unavailable_reason = str(
            review.get("reason")
            or "Imported command operation is unavailable until reviewed in the operation manager."
        )

    environment_id = str(payload.get("environment") or "")
    source = payload.get("source")
    runtime_config: dict[str, object] = {}
    if runtime == "python":
        if not isinstance(source, dict):
            raise ValueError("wrapper is missing a 'source' object")
        mode = str(source.get("mode") or "import")
        if mode not in ("import", "link"):
            raise ValueError(f"unknown source mode: {mode}")
        rel_or_abs = str(source.get("path") or "")
        if not rel_or_abs:
            raise ValueError("source is missing a 'path'")
        callable_name = str(source.get("callable") or "")
        if not callable_name:
            raise ValueError("source is missing a 'callable'")
        directory = user_operations_directory()
        code_path = rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(directory, rel_or_abs)
        try:
            available = {info.name for info in introspect_python_source(code_path)}
        except FileNotFoundError as exc:
            raise ValueError(f"source file not found: {code_path}") from exc
        except SyntaxError as exc:
            raise ValueError(f"source file has a syntax error: {exc}") from exc
        if callable_name not in available:
            raise ValueError(
                f"source file {code_path!r} has no top-level function {callable_name!r}"
            )
        if environment_id:
            runtime_config = _runtime_config(payload)
            build = _make_python_environment_build(
                code_path,
                callable_name,
                environment_id=environment_id,
                handoff=str(runtime_config["handoff"]),
                timeout=runtime_config["timeout_s"],
            )
        else:
            build = _make_user_build(code_path, slug, callable_name)
    else:
        runtime_config = _runtime_config(payload)
        command_problem = _command_definition_problem(runtime, runtime_config, parameters)
        if command_problem and not unavailable_reason:
            unavailable_reason = command_problem
        build = _make_command_build(
            runtime,
            runtime_config,
            environment_id=environment_id,
        )

    return PluginOperationSpec(
        id=operation_id,
        label=str(payload.get("label") or slug),
        build=build,
        parameters=parameters,
        requires_axis=requires_axis,
        changes_shape=changes_shape,
        group=str(payload.get("group") or "User"),
        description=str(payload.get("description") or ""),
        icon=str(payload.get("icon") or "extension"),
        # User ops are Tier-1 OPAQUE: no region claim, so no conformance gate.
        region_capable=False,
        unavailable_reason=unavailable_reason,
        availability=_availability_check(
            runtime,
            runtime_config,
            environment_id=environment_id,
        ),
        runtime=runtime,
        runtime_config=runtime_config,
        environment_id=environment_id,
        source_identity=lambda: _python_source_identity(code_path),
    )


def _runtime_config(payload: dict) -> dict[str, object]:
    timeout = payload.get("timeout_s", command_runtime.DEFAULT_TIMEOUT_S)
    if timeout is not None:
        try:
            timeout = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout_s must be a number or null") from exc
        if timeout <= 0:
            timeout = None
    handoff = str(payload.get("handoff") or "npy")
    command_runtime.array_handoff(handoff)
    return {
        "command_template": str(payload.get("command_template") or ""),
        "handoff": handoff,
        "timeout_s": timeout,
        "shell": bool(payload.get("shell", False)),
    }


def _command_definition_problem(
    runtime: str,
    config: dict[str, object],
    parameters: tuple[OperationParameter, ...],
) -> str:
    command_template = str(config.get("command_template") or "")
    if not command_template.strip():
        return "Command template is empty."
    fields = command_runtime.template_fields(command_template)
    required = {"in", "out", *(parameter.name for parameter in parameters)}
    missing = sorted(required - fields)
    if missing:
        return f"Command template is missing placeholders for: {', '.join(missing)}"
    if runtime in {"julia", "matlab"} and bool(config.get("shell")):
        return f"{runtime} runtime does not support shell execution."
    return ""


def _resolve_runtime_environment(environment_id: str):
    if not environment_id:
        return None, None
    return resolve_execution_environment(environment_id)


def _runtime_prefix(runtime: str, resolved) -> tuple[str, ...]:
    conda_prefix = tuple(getattr(resolved, "command_prefix", ()) or ())
    if runtime == "command":
        return conda_prefix
    default = "julia" if runtime == "julia" else "matlab"
    interpreter = getattr(resolved, "interpreter", None) if resolved is not None else None
    if interpreter == "python":
        interpreter = default
    return (*conda_prefix, str(interpreter or default))


def _availability_check(runtime: str, config: dict[str, object], *, environment_id: str):
    def check() -> str | None:
        resolved, reason = _resolve_runtime_environment(environment_id)
        if reason:
            return reason
        if runtime == "python":
            if not environment_id:
                return None
            if resolved is None or not resolved.interpreter:
                return f"Execution environment {environment_id!r} has no Python interpreter."
            return None
        prefix = _runtime_prefix(runtime, resolved)
        values = {"in": "input", "out": "output"}
        command = command_runtime.build_command(
            str(config["command_template"]),
            values
            | {
                field: "0"
                for field in command_runtime.template_fields(str(config["command_template"]))
                if field not in values
            },
            shell=bool(config["shell"]),
            prefix=prefix,
        )
        executable = command_runtime.command_executable(command, shell=bool(config["shell"]))
        effective_env = None if resolved is None else resolved.env
        if executable is None or not command_runtime.is_executable(executable, env=effective_env):
            return f"Command executable is not available: {executable or '(empty command)'}"
        return None

    return check


def _make_command_build(
    runtime: str,
    config: dict[str, object],
    *,
    environment_id: str,
):
    def build(axis, params):
        values = dict(params)
        if axis is not None:
            values["axis"] = axis

        def bound(data):
            resolved, reason = _resolve_runtime_environment(environment_id)
            if reason:
                raise RuntimeError(reason)
            return command_runtime.run_command_template(
                str(config["command_template"]),
                data,
                parameters=values,
                handoff=str(config["handoff"]),
                shell=bool(config["shell"]),
                prefix=_runtime_prefix(runtime, resolved),
                env=None if resolved is None else resolved.env,
                cwd=None if resolved is None else resolved.cwd,
                timeout=config["timeout_s"],
            )

        return bound

    return build


def _make_python_environment_build(
    code_path: str,
    callable_name: str,
    *,
    environment_id: str,
    handoff: str,
    timeout,
):
    worker = str(Path(__file__).with_name("python_environment_worker.py"))

    def build(axis, params):
        parameters_json = json.dumps(dict(params), separators=(",", ":"))
        axis_json = json.dumps(axis)

        def bound(data):
            resolved, reason = resolve_execution_environment(environment_id)
            if reason:
                raise RuntimeError(reason)
            if resolved is None or not resolved.interpreter:
                raise RuntimeError(
                    f"Execution environment {environment_id!r} has no Python interpreter."
                )
            prefix = tuple(resolved.command_prefix)
            command = [
                *prefix,
                resolved.interpreter,
                worker,
                "--source",
                code_path,
                "--callable",
                callable_name,
                "--handoff",
                handoff,
                "--parameters",
                parameters_json,
                "--axis",
                axis_json,
                "{in}",
                "{out}",
            ]
            return command_runtime.run_array_command(
                command,
                data,
                handoff=handoff,
                env=resolved.env,
                cwd=resolved.cwd,
                timeout=timeout,
            )

        return bound

    return build


def _parameters_from_payload(raw) -> tuple[OperationParameter, ...]:
    parameters: list[OperationParameter] = []
    for item in raw or ():
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if not name:
            continue
        kind = str(item.get("kind") or "float")
        if kind not in ("int", "float"):
            # An unknown kind (e.g. "str") would survive load only to crash
            # later in form building / value coercion, so reject the wrapper
            # here -> recorded as a problem, skipped, rest of the library loads.
            raise ValueError(f"unsupported parameter kind {kind!r} for parameter {name!r}")
        parameters.append(
            OperationParameter(
                name=name,
                label=str(item.get("label") or name.replace("_", " ").title()),
                kind=kind,
                default=item.get("default"),
                minimum=item.get("minimum"),
                maximum=item.get("maximum"),
                step=item.get("step"),
                description=str(item.get("description") or ""),
            )
        )
    return tuple(parameters)


def _load_user_module(path: str, slug: str):
    """Import the user code file, caching by (abspath, mtime) for link-edit pickup."""

    abspath = os.path.abspath(path)
    mtime_ns = os.stat(abspath).st_mtime_ns
    key = (abspath, mtime_ns)
    cached = _module_cache.get(key)
    if cached is not None:
        return cached
    module_name = f"{_USER_MODULE_NAMESPACE}.{slug}"
    spec = importlib.util.spec_from_file_location(module_name, abspath)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load user operation module from {abspath}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _module_cache[key] = module
    return module


def _python_source_identity(path: str) -> tuple[str, str, int | None]:
    abspath = os.path.abspath(path)
    try:
        mtime_ns = os.stat(abspath).st_mtime_ns
    except OSError:
        mtime_ns = None
    return "python-file", abspath, mtime_ns


@dataclass(frozen=True)
class _AcceptedArgs:
    names: frozenset[str]
    var_keyword: bool


def _accepted_args(fn) -> _AcceptedArgs:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        # A builtin / C callable with no introspectable signature: assume it
        # takes only the array (safest -- pass nothing extra).
        return _AcceptedArgs(frozenset(), False)
    names = set()
    var_keyword = False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            var_keyword = True
        elif parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            names.add(parameter.name)
    return _AcceptedArgs(frozenset(names), var_keyword)


def _make_user_build(path: str, slug: str, callable_name: str):
    """Return a ``build(axis, params) -> fn(data)`` that lazily imports the code.

    Call adaptation (introspected from the live function each import): the array
    is always the first positional arg; ``axis`` is passed only when the function
    accepts an ``axis`` parameter (or ``**kwargs``); each declared parameter is
    passed by name only when the function accepts that name (or ``**kwargs``).
    This supports ``f(data)``, ``f(data, axis)``, ``f(data, **params)`` and
    ``f(data, axis, **params)`` uniformly.
    """

    def build(axis, params):
        module = _load_user_module(path, slug)
        fn = getattr(module, callable_name, None)
        if not callable(fn):
            raise AttributeError(
                f"user operation module {path!r} has no callable {callable_name!r}"
            )
        accepted = _accepted_args(fn)

        def bound(data):
            kwargs: dict[str, object] = {
                name: value
                for name, value in dict(params).items()
                if name in accepted.names or accepted.var_keyword
            }
            if axis is not None and ("axis" in accepted.names or accepted.var_keyword):
                kwargs["axis"] = axis
            return fn(data, **kwargs)

        return bound

    return build


# ---------------------------------------------------------------------------
# AST introspection (never imports / executes user code)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallableParam:
    name: str
    kind: str  # "int" | "float"
    default: object | None = None


@dataclass(frozen=True)
class CallableInfo:
    name: str
    doc: str
    params: tuple[CallableParam, ...] = ()
    has_axis: bool = False


def introspect_python_source(path: str) -> list[CallableInfo]:
    """List top-level functions in ``path`` via ``ast`` (no import / execution).

    Parsing rather than importing is deliberate: it is safe (no user side effects
    run) and it works on a file that would fail to import (a top-level ``raise``,
    a missing dependency) so the manager UI can still show its callables.

    For each function the first positional argument is treated as the array;
    ``has_axis`` reports an ``axis`` parameter; ``params`` are the remaining
    non-data, non-axis parameters (positional-or-keyword and keyword-only), with
    ``kind`` guessed from annotation then default (fallback ``"float"``).
    """

    with open(path) as handle:
        source = handle.read()
    tree = ast.parse(source)
    infos: list[CallableInfo] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        infos.append(_callable_info_from_node(node))
    return infos


def _callable_info_from_node(node) -> CallableInfo:
    doc = (ast.get_docstring(node) or "").strip().splitlines()
    first_line = doc[0].strip() if doc else ""

    positional = list(node.args.posonlyargs) + list(node.args.args)
    # Align defaults to the tail of the positional args.
    pos_defaults: list = list(node.args.defaults)
    default_by_arg: dict[str, ast.AST] = {}
    if pos_defaults:
        for arg, default in zip(positional[-len(pos_defaults) :], pos_defaults, strict=False):
            default_by_arg[arg.arg] = default
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=False):
        if default is not None:
            default_by_arg[arg.arg] = default

    all_args = positional + list(node.args.kwonlyargs)
    has_axis = any(arg.arg == "axis" for arg in all_args)

    params: list[CallableParam] = []
    for index, arg in enumerate(all_args):
        name = arg.arg
        if index == 0 and arg in positional:
            # The array argument.
            continue
        if name == "axis":
            continue
        default_node = default_by_arg.get(name)
        default_value = _literal_from_node(default_node)
        kind = _kind_from_annotation(arg.annotation)
        if kind is None:
            kind = _kind_from_value(default_value)
        params.append(CallableParam(name=name, kind=kind, default=default_value))

    return CallableInfo(
        name=node.name,
        doc=first_line,
        params=tuple(params),
        has_axis=has_axis,
    )


def _kind_from_annotation(annotation) -> str | None:
    if annotation is None:
        return None
    name = None
    if isinstance(annotation, ast.Name):
        name = annotation.id
    elif isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        name = annotation.value
    if name in ("int",):
        return "int"
    if name in ("float",):
        return "float"
    return None


def _kind_from_value(value) -> str:
    if isinstance(value, bool):
        return "int"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "float"


def _literal_from_node(node):
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Import / link / remove / update
# ---------------------------------------------------------------------------


def _safe_slug(name: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name).strip().lower())
    slug = slug.strip("_-")
    return slug or "operation"


def _unique_slug(directory: str, base: str) -> str:
    """A slug whose ``.json`` and ``.py`` and ``user:`` id are all free."""

    def taken(slug: str) -> bool:
        return (
            os.path.exists(os.path.join(directory, f"{slug}.json"))
            or os.path.exists(os.path.join(directory, f"{slug}.py"))
            or f"user:{slug}" in registry._USER_SPECS
        )

    slug = base
    suffix = 2
    while taken(slug):
        slug = f"{base}-{suffix}"
        suffix += 1
    return slug


def _wrapper_path_for_id(operation_id: str) -> str | None:
    """Locate the wrapper JSON whose ``id`` field equals ``operation_id``."""

    directory = user_operations_directory()
    if not os.path.isdir(directory):
        return None
    for file_name in sorted(os.listdir(directory)):
        if not file_name.endswith(".json") or file_name in (
            "layout.json",
            "hidden-operations.json",
        ):
            continue
        path = os.path.join(directory, file_name)
        try:
            with open(path) as handle:
                if str(json.load(handle).get("id") or "") == str(operation_id):
                    return path
        except Exception:
            continue
    return None


def user_operation_wrapper(operation_id: str) -> dict | None:
    """Return a detached copy of a user operation's wrapper payload."""

    path = _wrapper_path_for_id(operation_id)
    if path is None:
        return None
    with open(path) as handle:
        payload = json.load(handle)
    return dict(payload) if isinstance(payload, dict) else None


def user_operation_source_path(operation_id: str) -> str | None:
    """Absolute path to the Python source backing a user operation."""

    wrapper = user_operation_wrapper(operation_id)
    if wrapper is None:
        return None
    source = wrapper.get("source")
    if not isinstance(source, dict):
        return None
    rel_or_abs = str(source.get("path") or "")
    if not rel_or_abs:
        return None
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(user_operations_directory(), rel_or_abs)


def create_empty_user_operation() -> str:
    """Create and register an unfinished user operation for in-manager editing."""

    from arrayscope.operations.operation_definitions import empty_template_source

    directory = user_operations_directory()
    os.makedirs(directory, exist_ok=True)
    label = _unique_copy_label("New operation", copy_suffix=False)
    slug = _unique_slug(directory, _safe_slug(label))
    operation_id = f"user:{slug}"
    callable_name = _callable_name(slug)
    source_name = f"{slug}.py"
    source_path = os.path.join(directory, source_name)
    with open(source_path, "w") as handle:
        handle.write(empty_template_source(callable_name))
    wrapper = {
        "format": WRAPPER_FORMAT,
        "version": 1,
        "id": operation_id,
        "label": label,
        "description": "",
        "group": "User",
        "icon": "extension",
        "runtime": SUPPORTED_RUNTIME,
        "source": {"mode": "import", "path": source_name, "callable": callable_name},
        "requires_axis": False,
        "changes_shape": False,
        "parameters": [],
        "template": {
            "kind": "empty",
            "message": "Empty template — open the code file and implement it before running.",
            "reason": "Empty template — open the code file and implement it before running.",
        },
    }
    _write_wrapper(os.path.join(directory, f"{slug}.json"), wrapper)
    refresh_user_operations()
    return operation_id


def duplicate_operation(operation_id: str) -> str:
    """Duplicate any operation into a selected, editable ``user:`` copy.

    Native implementations are copied into a standalone function. Pack and
    entry-point operations get a working adapter that names their provider
    dependency. Shape-changing copies use the same discovery contract as any
    other user operation.
    """

    from arrayscope.operations.operation_definitions import (
        adapter_template_source,
        export_operation_definition,
        native_editable_source,
    )

    entry = registry.get_operation_entry(operation_id)
    definition = export_operation_definition(entry)
    directory = user_operations_directory()
    os.makedirs(directory, exist_ok=True)
    label = _unique_copy_label(entry.label)
    slug = _unique_slug(directory, _safe_slug(label))
    duplicate_id = f"user:{slug}"
    source_name = f"{slug}.py"
    callable_name = _callable_name(slug)
    tier = str(definition.get("tier") or "")
    runtime = str(definition.get("runtime") or SUPPORTED_RUNTIME)

    if runtime in {"command", "julia", "matlab"}:
        wrapper = {
            key: value
            for key, value in definition.items()
            if key
            in {
                "description",
                "group",
                "icon",
                "runtime",
                "command_template",
                "handoff",
                "timeout_s",
                "shell",
                "environment",
                "requires_axis",
                "parameters",
            }
        }
        wrapper.update(
            {
                "format": WRAPPER_FORMAT,
                "version": 1,
                "id": duplicate_id,
                "label": label,
                "group": effective_group(entry),
                "changes_shape": False,
                "template": {
                    "kind": "command-copy",
                    "source_id": entry.id,
                    "message": "Editable copy of the command-template definition.",
                },
            }
        )
        _write_wrapper(os.path.join(directory, f"{slug}.json"), wrapper)
        refresh_user_operations()
        return duplicate_id

    template: dict[str, str]
    if tier == "builtin":
        source_text = native_editable_source(entry, callable_name)
        template = {
            "kind": "native-copy",
            "source_id": entry.id,
            "message": "Editable copy of the native implementation.",
        }
    elif tier in ("pack", "plugin"):
        source_text = adapter_template_source(entry, callable_name)
        provider = definition.get("source", {}).get("provider") or definition.get("source", {}).get(
            "distribution"
        )
        template = {
            "kind": "external-adapter",
            "source_id": entry.id,
            "message": (
                f"Working adapter template — it still depends on {provider or entry.id}. "
                "Replace the function body to make it independent."
            ),
        }
    elif tier == "user":
        source_path = user_operation_source_path(operation_id)
        if source_path is None:
            raise ValueError(f"user operation has no source path: {operation_id}")
        with open(source_path) as handle:
            source_text = handle.read()
        source = definition.get("source") or {}
        callable_name = str(source.get("callable") or callable_name)
        template = {
            "kind": "user-copy",
            "source_id": entry.id,
            "message": "Independent copy of the user operation and its source file.",
        }
    else:
        raise ValueError(f"cannot duplicate operation tier {tier!r}")

    with open(os.path.join(directory, source_name), "w") as handle:
        handle.write(source_text)
    wrapper = {
        "format": WRAPPER_FORMAT,
        "version": 1,
        "id": duplicate_id,
        "label": label,
        "description": definition.get("description", ""),
        "group": effective_group(entry),
        "icon": definition.get("icon", "extension"),
        "runtime": SUPPORTED_RUNTIME,
        "source": {"mode": "import", "path": source_name, "callable": callable_name},
        "requires_axis": bool(definition.get("requires_axis", False)),
        "changes_shape": bool(definition.get("changes_shape", False)),
        "parameters": list(definition.get("parameters") or ()),
        "template": template,
    }
    _write_wrapper(os.path.join(directory, f"{slug}.json"), wrapper)
    refresh_user_operations()
    return duplicate_id


def update_user_operation_source(
    operation_id: str,
    py_path: str,
    callable_name: str,
    *,
    link: bool,
    infer: bool = True,
) -> bool:
    """Retarget a user op's source safely, optionally applying AST inference.

    Introspection remains AST-only.  When ``infer`` is true every inferred field
    is written into the ordinary editable wrapper fields so the manager can show
    it immediately; subsequent edits are never hidden behind inference.
    """

    infos = {info.name: info for info in introspect_python_source(py_path)}
    info = infos.get(callable_name)
    if info is None:
        raise ValueError(f"{py_path!r} has no top-level function {callable_name!r}")
    wrapper_path = _wrapper_path_for_id(operation_id)
    if wrapper_path is None:
        return False
    with open(wrapper_path) as handle:
        payload = json.load(handle)

    directory = user_operations_directory()
    slug = operation_id[len("user:") :]
    if link:
        stored_path = os.path.abspath(py_path)
    else:
        stored_path = f"{slug}.py"
        destination = os.path.join(directory, stored_path)
        if os.path.abspath(py_path) != os.path.abspath(destination):
            shutil.copyfile(py_path, destination)

    payload["source"] = {
        "mode": "link" if link else "import",
        "path": stored_path,
        "callable": callable_name,
    }
    payload.pop("template", None)
    if infer:
        payload.update(
            label=callable_name.replace("_", " ").title(),
            description=info.doc,
            requires_axis=info.has_axis,
            parameters=[_parameter_payload(parameter) for parameter in info.params],
        )
    _write_wrapper(wrapper_path, payload)
    refresh_user_operations()
    return True


def _unique_copy_label(base: str, *, copy_suffix: bool = True) -> str:
    root = str(base).rstrip(".").strip()
    candidate = f"{root} copy" if copy_suffix else root
    labels = {entry.label.casefold() for entry in registry.all_operations()}
    if candidate.casefold() not in labels:
        return candidate
    suffix = 2
    while f"{candidate} {suffix}".casefold() in labels:
        suffix += 1
    return f"{candidate} {suffix}"


def _callable_name(slug: str) -> str:
    name = slug.replace("-", "_")
    if name and name[0].isdigit():
        name = f"operation_{name}"
    return name or "operation"


def _write_wrapper(path: str, payload: dict) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def import_custom_operation(
    py_path: str,
    callable_name: str,
    *,
    link: bool = False,
    label: str | None = None,
    description: str | None = None,
    group: str = "User",
    icon: str = "extension",
    changes_shape: bool = False,
) -> str:
    """Register a user op wrapping ``callable_name`` in ``py_path``; return its id.

    ``link=False`` (import): the ``.py`` is copied into the ops dir so the op is
    self-contained.  ``link=True``: the wrapper stores the absolute path and
    edits to the original file are picked up live (mtime-keyed re-import).

    The wrapper is auto-filled from an ``ast`` introspection of the target
    function: ``requires_axis`` from an ``axis`` parameter, ``parameters`` from
    the remaining non-data args (default + annotation-guessed kind), ``label``
    from the function name, ``description`` from the docstring.
    """

    infos = {info.name: info for info in introspect_python_source(py_path)}
    info = infos.get(callable_name)
    if info is None:
        raise ValueError(f"{py_path!r} has no top-level function {callable_name!r}")

    directory = user_operations_directory()
    os.makedirs(directory, exist_ok=True)
    base_slug = _safe_slug(label or callable_name)
    slug = _unique_slug(directory, base_slug)
    operation_id = f"user:{slug}"

    if link:
        source_path = os.path.abspath(py_path)
    else:
        source_path = f"{slug}.py"
        shutil.copyfile(py_path, os.path.join(directory, source_path))

    wrapper = {
        "format": WRAPPER_FORMAT,
        "version": 1,
        "id": operation_id,
        "label": label or callable_name.replace("_", " ").title(),
        "description": description if description is not None else info.doc,
        "group": group,
        "icon": icon,
        "runtime": SUPPORTED_RUNTIME,
        "source": {
            "mode": "link" if link else "import",
            "path": source_path,
            "callable": callable_name,
        },
        "requires_axis": info.has_axis,
        "changes_shape": bool(changes_shape),
        "parameters": [_parameter_payload(param) for param in info.params],
    }
    _write_wrapper(os.path.join(directory, f"{slug}.json"), wrapper)

    refresh_user_operations()
    return operation_id


def _parameter_payload(param: CallableParam) -> dict:
    payload: dict = {
        "name": param.name,
        "label": param.name.replace("_", " ").title(),
        "kind": param.kind,
    }
    if param.default is not None:
        payload["default"] = param.default
    return payload


def remove_user_operation(operation_id: str, *, delete_files: bool = True) -> bool:
    """Delete a user op's wrapper (and, for import-mode, its copied code file)."""

    path = _wrapper_path_for_id(operation_id)
    if path is None:
        return False
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except Exception:
        payload = {}
    source = payload.get("source") if isinstance(payload, dict) else None
    os.remove(path)
    if delete_files and isinstance(source, dict) and source.get("mode") == "import":
        rel = str(source.get("path") or "")
        if rel and not os.path.isabs(rel):
            code_path = os.path.join(user_operations_directory(), rel)
            if os.path.exists(code_path):
                os.remove(code_path)
    refresh_user_operations()
    return True


def update_user_operation(operation_id: str, **wrapper_fields) -> bool:
    """Rewrite a user op's wrapper with ``wrapper_fields``, then refresh.

    Field values are merged over the existing wrapper (``source`` is merged
    shallowly).  ``id`` cannot be changed here.
    """

    path = _wrapper_path_for_id(operation_id)
    if path is None:
        return False
    with open(path) as handle:
        payload = json.load(handle)
    wrapper_fields.pop("id", None)
    source_update = wrapper_fields.pop("source", None)
    if isinstance(source_update, dict):
        merged_source = dict(payload.get("source") or {})
        merged_source.update(source_update)
        payload["source"] = merged_source
    payload.update(wrapper_fields)
    _write_wrapper(path, payload)
    refresh_user_operations()
    return True


def operation_runtime(operation_id: str) -> str:
    """Declarative runtime for one operation without executing its body."""

    wrapper = user_operation_wrapper(operation_id)
    if wrapper is not None:
        return str(wrapper.get("runtime") or SUPPORTED_RUNTIME)
    try:
        from arrayscope.operations.operation_definitions import export_operation_definition

        return str(export_operation_definition(operation_id).get("runtime") or "python")
    except Exception:
        return "python"


def quarantine_imported_command(operation_id: str) -> bool:
    """Persist the ADR 0060 imported-recipe review boundary for a command op."""

    if operation_runtime(operation_id) not in {"command", "julia", "matlab"}:
        return False
    path = _wrapper_path_for_id(operation_id)
    if path is None:
        return False
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    review = payload.get("review")
    if isinstance(review, dict) and review.get("required"):
        return True
    payload["review"] = {
        "required": True,
        "reason": (
            "Imported recipe command — review this definition in the operation "
            "manager before allowing it to run."
        ),
    }
    _write_wrapper(path, payload)
    refresh_user_operations()
    return True


def review_user_operation(operation_id: str) -> bool:
    """Mark an imported command definition reviewed and recompute availability."""

    path = _wrapper_path_for_id(operation_id)
    if path is None:
        return False
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if "review" not in payload:
        return False
    payload.pop("review", None)
    _write_wrapper(path, payload)
    refresh_user_operations()
    return True
