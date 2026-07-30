"""Production definitions that no production caller reaches.

Two classes, both failures:

* **Unreachable** — nothing names it. Not ``arrayscope/``, not ``tests/``, not
  the repo-root product trees.
* **Test-only** — the only files naming it live under ``tests/``. This is the
  class that rots silently: the function still *executes*, so a coverage
  number reports it healthy, while no shipped code path reaches it.

This is a lint, not a test, and it is wired the way ruff is: the gate is
``.githooks/pre-commit`` via ``tools/dead_code.py``. Running the tree scan
again under pytest would buy nothing and cost ~3.7 s on every suite run, so
``tests/app/test_dead_code.py`` keeps only the synthetic proof that these
rules can actually fail.

``tests/coverage_map.py`` cannot answer either half, so this does not build on
it. Its ``uncovered`` set means "no recorded test entered this function": a
live function nobody tests is uncovered, and a test-only function is
*covered*. Reachability is a static question about who **names** a definition,
which is why it is asked statically here.

Scope is module-level ``def``/``class`` only. Methods and nested functions are
reached through instances, dataclass fields, and duck-typed protocols, so a
name scan cannot adjudicate them; including them would trade a short, trusted
report for a long, ignored one.

A *reference* is any ``Name``, attribute name, import alias, or **string
constant** anywhere in the tree. Counting bare strings is what keeps dynamic
access honest: ``getattr(module, "name")``, the PEP 562 lazy re-export maps,
Qt ``connect()`` by slot name, and entry-point values all reach their target
through a string, and every one of them is credited as a caller here.

Two references deliberately do *not* count:

* A name's own definition — recursion is not a caller.
* An ``__all__`` entry — a declaration is not a caller. This is the rule that
  exposes rot: ``display_output_is_composited_rgb`` kept an ``__all__`` line
  and four test assertions for the whole time it had no production caller.

AST over ``arrayscope/`` and the repo-root product trees; a plain word scan
over ``tests/``, where over-approximating can only relabel a finding and never
hide one. Nothing is imported and no Qt object is built: ~1.9 s.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

ROOT = Path(__file__).parents[1]

#: Repo-root trees that ship or drive the product rather than test it. The
#: gallery in ``tools/`` is a review surface, so a name it reaches is live.
_NON_TEST_ROOTS = ("tools", "wrappers", "packaging", "experiments")

_DEFINITION = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

_STRUCTURAL_BASES = {"Protocol", "ABC"}

_WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


# Real entry points with no in-tree caller by construction. Everything else is
# either deleted or listed below with its reason -- the rule is never widened
# silently, because a guard that is wrong once gets deleted.
#
# Format: (path, name, reason).
_ALLOWLIST = (
    # --- Test/benchmark seams a production caller must never have -----------
    (
        "arrayscope/display/shader_kernels.py",
        "prewarm_blocking",
        "synchronous compile for tests/benchmarks; production prewarms off-thread",
    ),
    (
        "arrayscope/gpu/device_topology.py",
        "reset_topology_cache",
        "cache-drop seam for adapter/device lifecycle churn; production caches for the process",
    ),
    (
        "arrayscope/operations/plugins.py",
        "_reset_plugin_cache",
        "re-discovery seam; production discovers entry points once",
    ),
    # --- Oracles: the CPU mirror IS the reference, the shader is the caller --
    (
        "arrayscope/display/shader_mapping.py",
        "wgpu_nan_hatch_rgba",
        "RGBA8 mirror of the WGSL nan_marker (A2); WGSL is the caller-side twin",
    ),
    (
        "arrayscope/display/shader_mapping.py",
        "wgpu_missing_hatch_rgba",
        "RGBA8 mirror of the WGSL missing_marker (A3)",
    ),
    (
        "arrayscope/display/shader_mapping.py",
        "wgpu_clip_rgb",
        "RGB8 mirror of the WGSL clip marker (A4)",
    ),
    (
        "arrayscope/display/shader_mapping.py",
        "wgpu_pixel_grid_darken",
        "mirror of the WGSL pixel_grid darkening (A1)",
    ),
    (
        "arrayscope/display/shader_mapping.py",
        "wgpu_minification_tap_count",
        "mirror of the C1 minification box filter's tap count",
    ),
    (
        "arrayscope/display/shader_mapping.py",
        "wgpu_minification_taps",
        "mirror of the C1 box-filter tap coordinates",
    ),
    (
        "arrayscope/display/pyramid.py",
        "reduce_source_grid",
        "direct source-grid oracle every reducer family is checked against",
    ),
    # --- Benchmark entry points, driven by their own ring or by a CLI -------
    (
        "arrayscope/operations/benchmarks.py",
        "run_foundation_benchmarks",
        "benchmark ring entry point; no product path may run a benchmark suite",
    ),
)

# Test-only definitions this sweep FOUND but did not adjudicate (2026-07-30).
#
# Every one is real: no shipped code path names it. They are not exempt and not
# excused -- they are named debt, because "delete it" and "restore a caller" are
# different answers here and the difference is a product decision:
#
# * **Superseded twins.** A public entry replaced by an internal one that now
#   carries production: ``evaluate_slab`` over ``evaluate_slab_from_plan``,
#   ``cache_policy.decide_texture_codec`` over the executor's own
#   ``_decide_codec_family``, ``plugin_operation_ids`` over
#   ``discover_plugin_entry_points``. Deleting the twin means migrating its
#   tests onto the survivor, which changes what they exercise.
# * **A producer whose consumer is still live.** ``evaluate_shared_preview`` is
#   the only thing that builds shared-preview cohort rows, and nothing calls it
#   -- while ``window/frame_effects.py`` still consumes those rows through
#   ``_looks_like_shared_preview_rows``, ``_admit_preview_cohort_level_evidence``
#   and the ``preview-cohort-pending`` round-level source. Deleting it would
#   erase the evidence for what may be a render regression, not clean up rot.
#
# The count below is a ceiling, so this list can only shrink.
_PENDING_ADJUDICATION = (
    ("arrayscope/core/array_source.py", "is_lazy_source_array"),
    ("arrayscope/core/cache_status.py", "cache_status_prefetching"),
    ("arrayscope/core/cache_status.py", "cache_status_stale_ignored"),
    ("arrayscope/core/compare.py", "compatible_roi_shape"),
    ("arrayscope/core/dimension_roles.py", "DimensionRoles"),
    ("arrayscope/core/gui_callback_budget.py", "should_yield_after_item"),
    ("arrayscope/core/memory_budget.py", "estimate_montage_tile_grid_bytes"),
    ("arrayscope/core/numba_runtime.py", "get_group"),
    ("arrayscope/core/numba_runtime.py", "registered_names"),
    ("arrayscope/core/window_levels.py", "choose_window_levels"),
    ("arrayscope/display/_numba_pyramid.py", "is_ready"),
    (
        "arrayscope/display/backends/pyqtgraph/histogram_adapter.py",
        "installed_histogram_lut_api_facts",
    ),
    ("arrayscope/display/backends/pyqtgraph/tiles.py", "_assemble_page_backed_payload"),
    ("arrayscope/display/colormap_library.py", "colormaps_for_family"),
    ("arrayscope/display/colormap_library.py", "refresh_user_colormaps"),
    ("arrayscope/display/histogram_controller.py", "adaptive_histogram_for_view"),
    ("arrayscope/display/lod.py", "inner_uv_for_gutter"),
    ("arrayscope/display/model/montage_levels.py", "_finite_bounds"),
    ("arrayscope/display/model/montage_levels.py", "_finite_sample"),
    ("arrayscope/display/model/tiled_histogram_identity.py", "tiled_histogram_key"),
    ("arrayscope/display/model/tiled_histogram_identity.py", "tiled_semantic_histogram_identity"),
    ("arrayscope/display/montage.py", "tile_status_at_global_point"),
    ("arrayscope/display/pyramid.py", "_reduce_sample"),
    ("arrayscope/display/pyramid.py", "partition_source_grid_pages"),
    ("arrayscope/display/pyramid.py", "reduce_source_grid_mean"),
    ("arrayscope/display/pyramid.py", "reduction_xy_to_yx"),
    ("arrayscope/display/region_source.py", "EagerDisplayRegionSource"),
    ("arrayscope/display/shader_mapping.py", "phase_lut_indices"),
    ("arrayscope/display/shader_mapping.py", "shader_component_uniform"),
    ("arrayscope/display/viewport.py", "coerce_viewport_policy"),
    ("arrayscope/gpu/bc_codec.py", "bc4_plan"),
    ("arrayscope/gpu/cache_policy.py", "decide_texture_codec"),
    ("arrayscope/gpu/chunk_codec.py", "CompressedChunkCache"),
    ("arrayscope/gpu/chunk_codec.py", "gpu_decodable_codec_names"),
    ("arrayscope/gpu/chunk_texture_codec.py", "encode_complex_tile"),
    ("arrayscope/gpu/chunk_texture_codec.py", "encode_scalar_tile"),
    ("arrayscope/io/lazy_sources.py", "supports_memmap_source"),
    ("arrayscope/io/numpy_save.py", "estimate_nbytes"),
    ("arrayscope/operations/_numba_native_ops.py", "is_ready"),
    ("arrayscope/operations/_numba_reductions.py", "is_ready"),
    ("arrayscope/operations/capabilities.py", "operation_execution_class"),
    ("arrayscope/operations/chunked.py", "evaluate_image_snapshot_chunked"),
    ("arrayscope/operations/dim_ops.py", "undo_fftshift"),
    ("arrayscope/operations/fft_backend.py", "available_fft_backends"),
    ("arrayscope/operations/packs/sigpy_pack.py", "sigpy_available"),
    ("arrayscope/operations/planner.py", "candidate_stage_cache_points"),
    ("arrayscope/operations/plugin_conformance.py", "verify_region_conformance"),
    ("arrayscope/operations/plugins.py", "is_region_honored"),
    ("arrayscope/operations/plugins.py", "plugin_operation_ids"),
    ("arrayscope/operations/plugins.py", "region_conformance_stats"),
    ("arrayscope/operations/recipes.py", "operations_from_recipe"),
    ("arrayscope/operations/regions.py", "region_axis_kinds"),
    ("arrayscope/operations/slabs.py", "evaluate_slab"),
    ("arrayscope/operations/stack.py", "delete_operation"),
    ("arrayscope/operations/stack.py", "move_operation"),
    ("arrayscope/render/effects.py", "chunk_level_stats_for_pages"),
    ("arrayscope/render/effects.py", "evaluate_shared_preview"),
    ("arrayscope/render/effects.py", "preview_claim_key"),
    ("arrayscope/render/effects.py", "reduce_display_payload_axes"),
    ("arrayscope/render/lod.py", "_page_set_complete"),
    ("arrayscope/render/lod.py", "admit_retained_preview_level"),
    ("arrayscope/render/lod.py", "plan_lod_page_targets"),
    ("arrayscope/window/frame_controller.py", "_should_defer_montage_side_panels"),
    ("arrayscope/window/montage_viewport.py", "remap_montage_roi_geometry"),
    ("arrayscope/window/render_contract.py", "session_is_current"),
)

#: Ceiling, never a target. Lower it as entries are resolved; raising it needs a
#: reason in the same commit.
_PENDING_CEILING = 65


#: ``arrayscope/tools/`` is the tracing/benchmark/oracle harness (docs/areas.md
#: routes it to ``tests/app`` + ``tests/stress``). Being called only from tests
#: is that package's design, so it is exempt from the *test-only* class alone --
#: a definition nothing there reaches at all still fails as unreachable.
_HARNESS_PACKAGE = "arrayscope/tools/"


@dataclass(frozen=True)
class Definition:
    relative: str
    name: str
    lineno: int

    def __str__(self) -> str:
        return f"{self.relative}:{self.lineno} {self.name}"


def _dunder_all_strings(tree: ast.Module) -> set[int]:
    """``id()`` of every string inside a module-level ``__all__`` assignment."""

    marked: set[int] = set()
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            continue
        if node.value is None:
            continue
        marked.update(
            id(child)
            for child in ast.walk(node.value)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        )
    return marked


def _names_referenced(node: ast.AST, declared: set[int]) -> set[str]:
    found: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            found.add(child.id)
        elif isinstance(child, ast.Attribute):
            found.add(child.attr)
        elif isinstance(child, ast.Constant):
            if isinstance(child.value, str) and id(child) not in declared:
                found.add(child.value)
        elif isinstance(child, ast.ImportFrom):
            for alias in child.names:
                found.add(alias.name)
                found.add(alias.asname or alias.name)
        elif isinstance(child, ast.Import):
            for alias in child.names:
                found.add(alias.name.split(".")[0])
                if alias.asname:
                    found.add(alias.asname)
    return found


def _structurally_exempt(node: ast.AST, relative: str) -> bool:
    """Entry points Python itself reaches, so no in-tree name can exist."""

    name = node.name
    # PEP 562 module ``__getattr__`` and every other dunder: called by protocol.
    if name.startswith("__") and name.endswith("__"):
        return True
    # pytest calls its hooks by name.
    if name.startswith("pytest_"):
        return True
    # ``[project.scripts] arrayscope = "arrayscope.__main__:main"``.
    if relative.endswith("__main__.py"):
        return True
    if isinstance(node, ast.ClassDef):
        # Structural types: an implementation never names its Protocol.
        for base in node.bases:
            base_name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
            if base_name in _STRUCTURAL_BASES:
                return True
        for decorator in node.decorator_list:
            decorated = decorator.func if isinstance(decorator, ast.Call) else decorator
            label = (
                decorated.attr
                if isinstance(decorated, ast.Attribute)
                else getattr(decorated, "id", "")
            )
            if label == "runtime_checkable":
                return True
    return False


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py"))


def _words_under(directory: Path) -> set[str]:
    """Every identifier-shaped word in a tree, without parsing it."""

    words: set[str] = set()
    if not directory.is_dir():
        return words
    for path in _python_files(directory):
        # This module names every allowlisted and pending definition. Counting
        # itself would let an entry keep its own subject looking
        # test-referenced forever, which is what the staleness check must see.
        if path.name == Path(__file__).name:
            continue
        words.update(_WORD.findall(path.read_text(encoding="utf-8")))
    return words


@cache
def unreferenced_definitions(root: Path) -> tuple[tuple[Definition, ...], tuple[Definition, ...]]:
    """``(unreachable, test_only)`` module-level definitions under ``root``.

    ``root`` is the repository root; the package scanned is ``root/arrayscope``.
    Taking the repo root (rather than the package) is what lets the synthetic
    fixture below drive the whole rule set, allowlist excluded.

    Cached because both guards below ask the same question of the same tree,
    and one parse of ``arrayscope/`` plus ``tests/`` is the whole cost here.
    """

    package = root / "arrayscope"
    definitions: list[tuple[Definition, ast.AST]] = []
    # name -> number of distinct owners naming it; an owner is one top-level
    # definition, or one file's module level.
    package_references: dict[str, int] = {}
    owner_references: dict[Definition, set[str]] = {}

    def credit(names: set[str]) -> None:
        for name in names:
            package_references[name] = package_references.get(name, 0) + 1

    for path in _python_files(package):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        relative = path.relative_to(root).as_posix()
        declared = _dunder_all_strings(tree)
        module_level: set[str] = set()
        for node in tree.body:
            if isinstance(node, _DEFINITION):
                definition = Definition(relative, node.name, node.lineno)
                definitions.append((definition, node))
                owned = _names_referenced(node, declared) - {node.name}
                owner_references[definition] = owned
                credit(owned)
                # A decorator sits outside the body it decorates, so it is
                # module-level evidence -- a registration decorator naming a
                # sibling really does reach it.
                for decorator in node.decorator_list:
                    module_level |= _names_referenced(decorator, declared)
            else:
                module_level |= _names_referenced(node, declared)
        credit(module_level)

    def outside(roots: tuple[str, ...]) -> set[str]:
        names: set[str] = set()
        for name in roots:
            directory = root / name
            if not directory.is_dir():
                continue
            for path in _python_files(directory):
                # ``tools/dead_code.py`` is this module's CLI. A product-root
                # reference SUPPRESSES a finding, so if the CLI ever grew a
                # mention of an allowlisted name it would silently hide it.
                if path.name == Path(__file__).name:
                    continue
                tree = ast.parse(path.read_text(encoding="utf-8"))
                names |= _names_referenced(tree, _dunder_all_strings(tree))
        return names

    # ``tests/`` is read with a word scan rather than a parse, which is most of
    # this guard's wall clock. It over-approximates -- a name in a test comment
    # counts -- and that is safe in exactly one direction: the test side only
    # ever decides which *label* a finding gets, never whether it is reported.
    # ``product_names`` does suppress findings, so those roots stay parsed.
    test_names = _words_under(root / "tests")
    product_names = outside(_NON_TEST_ROOTS)

    unreachable: list[Definition] = []
    test_only: list[Definition] = []
    for definition, node in definitions:
        if _structurally_exempt(node, definition.relative):
            continue
        owners = package_references.get(definition.name, 0)
        if definition.name in owner_references[definition]:
            owners -= 1
        if owners or definition.name in product_names:
            continue
        if definition.name in test_names:
            if not definition.relative.startswith(_HARNESS_PACKAGE):
                test_only.append(definition)
        else:
            unreachable.append(definition)
    return tuple(unreachable), tuple(test_only)


def _allowed(definition: Definition) -> bool:
    return any(
        definition.relative == path and definition.name == name for path, name, _ in _ALLOWLIST
    )


def _pending(definition: Definition) -> bool:
    return (definition.relative, definition.name) in _PENDING_ADJUDICATION


def problems(root: Path = ROOT) -> list[str]:
    """Every reason this tree should be refused, most actionable first."""

    unreachable, test_only = unreferenced_definitions(root)

    found = [
        f"unreachable (nothing names it):   {definition}"
        for definition in unreachable
        if not _allowed(definition) and not _pending(definition)
    ]
    found += [
        f"test-only (only tests/ names it): {definition}"
        for definition in test_only
        if not _allowed(definition) and not _pending(definition)
    ]
    if found:
        found.append("  -> delete it with its assertions, or restore a real caller;")
        found.append(f"     a genuine entry point goes in _ALLOWLIST in {Path(__file__).name}.")

    flagged = {(definition.relative, definition.name) for definition in unreachable + test_only}
    # A stale entry is how the rule gets silently widened, so it is a failure
    # in its own right rather than a line nobody ever deletes.
    stale = [
        f"stale allowlist entry (no longer flagged): {path}::{name}"
        for path, name, _ in _ALLOWLIST
        if (path, name) not in flagged
    ]
    stale += [
        f"resolved pending entry (no longer flagged): {path}::{name}"
        for path, name in _PENDING_ADJUDICATION
        if (path, name) not in flagged
    ]

    ceiling = []
    if len(_PENDING_ADJUDICATION) > _PENDING_CEILING:
        ceiling.append(
            f"{len(_PENDING_ADJUDICATION)} pending entries against a ceiling of "
            f"{_PENDING_CEILING}: adjudicate one instead of raising the ceiling."
        )
    return found + stale + ceiling
