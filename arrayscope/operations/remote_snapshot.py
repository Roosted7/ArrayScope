"""The tile/image evaluation body as ``(entry, request)`` (ADR 0064 family 6).

``evaluate_image_snapshot`` is the montage's per-tile body: read the planned
base region, run the operation transitions over it, build the display image.
It is already pure over ``(document, view_state)`` **except** when the region
plan carries stage-cache candidates, because then it participates in a
parent-owned ``StageCache`` — reads it, claims it, writes it.

So the migration seam is exactly that condition. :func:`run_image_snapshot`
keeps the call site's signature, and only when the plan declares no cache
candidates does it re-express the call as a ``RemoteCall``: an FFT-prefix tile
that shares a ~123 MB stage slab keeps running in-process, where the cache it
depends on lives, and a source-direct tile goes to a worker process. That is
the family-3 rule ("the parent admits results") applied by refusing to move the
cases that would have to write one.

The document a child evaluates is rebuilt, not pickled: a source descriptor
(``operations/source_transport.py``) plus the ordinary recipe wire format. That
keeps arrays out of the pipe and makes the round trip prove itself — the
equality pin runs the entry against the same fixture the inline call did.
"""

from __future__ import annotations

import json
import threading
import weakref
from dataclasses import dataclass, replace

import numpy as np

from arrayscope.kernel.remote_call import RemoteCall
from arrayscope.operations.evaluator import evaluate_image_snapshot
from arrayscope.operations.pipeline import ArrayDocument
from arrayscope.operations.recipes import recipe_from_steps, steps_from_recipe
from arrayscope.operations.slabs import plan_slab, request_for_image
from arrayscope.operations.source_transport import base_data_token, describe_base_data

#: How many rebuilt documents a process keeps. One montage session evaluates
#: every tile against one document, so this only has to survive the moment a
#: session hands over to its successor.
DOCUMENT_CACHE_ENTRIES = 4

#: One montage's worth of tile requests, so a plan-order pass over a wide
#: viewport re-decides nothing; beyond that the memo is not worth its own size.
STAGE_DECISION_MEMO_ENTRIES = 512

_stage_free_decisions: dict[object, bool] = {}
_decision_lock = threading.Lock()


@dataclass(frozen=True)
class ImageSnapshotRequest:
    """One image/tile evaluation as data a fresh interpreter can act on.

    ``identity`` is what both sides key the rebuilt document on; it is derived
    from the source token, the recipe and the revision, so two tiles of the
    same montage rebuild the document once per process, not once per tile.
    """

    identity: object
    source: object
    recipe: object
    revision: int
    base_axes: tuple
    view_state: object
    colormap_lut: object
    shader_display: bool
    provisional_histogram: bool
    canonical_orientation: bool
    evaluation_context: object


def evaluate_image_snapshot_request(request: ImageSnapshotRequest, token=None):
    """Evaluate one image snapshot from its descriptor — the moved body.

    Pure over ``request``: no session, no stage cache, no lifecycle. A stage
    cache is not merely withheld here, it is proven absent by the gate in
    :func:`portable_image_snapshot_request`.
    """

    document = document_for_request(request)
    context = request.evaluation_context
    if context is not None:
        context = replace(context, cancellation_token=token)
    return evaluate_image_snapshot(
        document,
        request.view_state,
        colormap_lut=request.colormap_lut,
        cancellation_token=token,
        shader_display=bool(request.shader_display),
        provisional_histogram=bool(request.provisional_histogram),
        stage_cache=None,
        stage_document_key=None,
        evaluation_context=context,
        canonical_orientation=bool(request.canonical_orientation),
    )


#: Derived from the function rather than typed out, so the entry a child
#: resolves cannot drift from the body this module actually defines.
IMAGE_SNAPSHOT_ENTRY = f"{__name__}:{evaluate_image_snapshot_request.__name__}"


def run_image_snapshot(
    document,
    view_state,
    *,
    colormap_lut=None,
    cancellation_token=None,
    shader_display: bool = False,
    provisional_histogram: bool = False,
    stage_cache=None,
    stage_document_key=None,
    evaluation_context=None,
    canonical_orientation: bool = False,
):
    """Evaluate one image snapshot, off-process when the body is portable."""

    request = portable_image_snapshot_request(
        document,
        view_state,
        colormap_lut=colormap_lut,
        shader_display=shader_display,
        provisional_histogram=provisional_histogram,
        stage_cache=stage_cache,
        evaluation_context=evaluation_context,
        canonical_orientation=canonical_orientation,
    )
    if request is None:
        return evaluate_image_snapshot(
            document,
            view_state,
            colormap_lut=colormap_lut,
            cancellation_token=cancellation_token,
            shader_display=bool(shader_display),
            provisional_histogram=bool(provisional_histogram),
            stage_cache=stage_cache,
            stage_document_key=stage_document_key,
            evaluation_context=evaluation_context,
            canonical_orientation=bool(canonical_orientation),
        )
    # Seeding first is what keeps the in-process path free: a worker backend
    # without a pool (or an inline backend under test) resolves the identity to
    # the very document it was handed, so nothing is reopened or re-planned.
    remember_document(request.identity, document, weak=True)
    return RemoteCall(IMAGE_SNAPSHOT_ENTRY, request)(cancellation_token)


def portable_image_snapshot_request(
    document,
    view_state,
    *,
    colormap_lut=None,
    shader_display: bool = False,
    provisional_histogram: bool = False,
    stage_cache=None,
    evaluation_context=None,
    canonical_orientation: bool = False,
) -> ImageSnapshotRequest | None:
    """Build the request, or ``None`` when this evaluation must stay in-process."""

    if document is None or view_state is None:
        return None
    if not _stage_free(document, view_state, stage_cache):
        return None
    recipe = portable_recipe_text(document)
    if recipe is None:
        return None
    source = describe_base_data(document.base_data, publish=True)
    if source is None:
        return None
    context = (
        None if evaluation_context is None else replace(evaluation_context, cancellation_token=None)
    )
    return ImageSnapshotRequest(
        identity=(base_data_token(document.base_data), recipe, int(document.revision)),
        source=source,
        recipe=recipe,
        revision=int(document.revision),
        base_axes=tuple(document.base_axes),
        view_state=view_state,
        colormap_lut=None if colormap_lut is None else np.asarray(colormap_lut),
        shader_display=bool(shader_display),
        provisional_histogram=bool(provisional_histogram),
        canonical_orientation=bool(canonical_orientation),
        evaluation_context=context,
    )


def _stage_free(document, view_state, stage_cache) -> bool:
    """Would this evaluation touch the stage cache at all?

    ``evaluate_slab_from_plan`` takes its pure branch exactly when there is no
    cache or no candidate, so asking the planner is not a heuristic — it is the
    same predicate the evaluator itself branches on. A document with no enabled
    operations has no stages to declare, which skips the planning entirely.
    """

    if stage_cache is None:
        return True
    if not document.enabled_operations:
        return True
    request = request_for_image(view_state)
    # Planning an FFT-prefix montage tile costs ~0.2 ms, and the answer depends
    # on nothing but the planner's own inputs — the base signature, the steps
    # and the request — so a bounded memo pays for every tile after the first
    # instead of re-deciding the same refusal per tile.
    key = (
        base_data_token(document.base_data),
        tuple(np.shape(document.base_data)),
        str(getattr(document.base_data, "dtype", None)),
        document.steps,
        request,
    )
    decided = _stage_free_decisions.get(key)
    if decided is not None:
        return bool(decided)
    try:
        plan = plan_slab(document, request)
    except Exception:
        return False
    decided = not tuple(getattr(plan.region_plan, "cache_candidates", ()) or ())
    with _decision_lock:
        _stage_free_decisions[key] = decided
        while len(_stage_free_decisions) > STAGE_DECISION_MEMO_ENTRIES:
            _stage_free_decisions.pop(next(iter(_stage_free_decisions)))
    return bool(decided)


def portable_recipe_text(document) -> str | None:
    """The document's steps as recipe JSON a fresh interpreter can re-resolve.

    Only built-in operations qualify. A plugin or user-pack operation resolves
    through state the GUI process loaded (installed distributions, the user
    library on disk); a spawned worker has none of it, and a tile that failed
    to re-resolve its own pipeline would be a wrong tile, not a slow one.

    JSON text, not the dict: the recipe is half of the identity that keys the
    rebuilt document, and a dict is neither hashable nor order-stable.
    """

    from arrayscope.operations.registry import OPERATION_REGISTRY, operation_id_for

    steps = tuple(document.steps)
    if not steps:
        return ""
    try:
        for step in steps:
            if operation_id_for(step.operation) not in OPERATION_REGISTRY:
                return None
        return json.dumps(recipe_from_steps(steps), sort_keys=True)
    except Exception:
        return None


# ------------------------------------------------------- document rebuilding

_documents: dict[object, tuple] = {}
_document_lock = threading.Lock()


def remember_document(identity, document, *, weak: bool = False) -> None:
    """Publish a document under the identity its requests carry.

    The GUI process publishes ``weak``: this cache exists to spare a child the
    rebuild, and a strong entry here would keep a closed window's base array —
    and its whole operation chain — alive for the life of the process.
    """

    entry: tuple
    if weak:
        try:
            entry = (None, weakref.ref(document))
        except TypeError:
            return
    else:
        entry = (document, None)
    with _document_lock:
        _documents.pop(identity, None)
        _documents[identity] = entry
        while len(_documents) > DOCUMENT_CACHE_ENTRIES:
            _documents.pop(next(iter(_documents)))


def document_for_request(request: ImageSnapshotRequest):
    """The document this request names — the live one, or a rebuilt one."""

    with _document_lock:
        entry = _documents.get(request.identity)
    if entry is not None:
        strong, reference = entry
        document = strong if reference is None else reference()
        if document is not None:
            return document
    base_data = request.source.reconstruct()
    recipe = json.loads(request.recipe) if request.recipe else None
    steps = (
        ()
        if not recipe
        else steps_from_recipe(recipe, tuple(int(size) for size in np.shape(base_data)))
    )
    document = ArrayDocument(
        base_data,
        steps=steps,
        revision=int(request.revision),
        axes=tuple(request.base_axes) or None,
    )
    remember_document(request.identity, document)
    return document


__all__ = [
    "DOCUMENT_CACHE_ENTRIES",
    "IMAGE_SNAPSHOT_ENTRY",
    "ImageSnapshotRequest",
    "document_for_request",
    "evaluate_image_snapshot_request",
    "portable_image_snapshot_request",
    "portable_recipe_text",
    "remember_document",
    "run_image_snapshot",
]
