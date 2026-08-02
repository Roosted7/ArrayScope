"""The tile/image evaluation body as ``(entry, request)`` (ADR 0064 family 6).

``evaluate_image_snapshot`` is the montage's per-tile body: read the planned
base region, run the operation transitions over it, build the display image.
It is already pure over ``(document, view_state)`` **except** when the region
plan carries stage-cache candidates, because then it participates in a
parent-owned ``StageCache`` — reads it, claims it, writes it.

So the migration seam is exactly that condition. :func:`run_image_snapshot`
keeps the call site's signature, and re-expresses the call as a ``RemoteCall``
whenever the child can be handed everything the parent's cache would have
served it.

That used to mean "no cache candidates at all", which left the whole FFT-prefix
workload — the expensive one — on the in-process threads.
``operations/stage_segments.py`` removes the reason: a stored stage value is
published into its own shared-memory segment, so a request can *name* the
~234 MB slab instead of needing the cache object that owns it. The gate is now
the honest one: every candidate this plan would look up must resolve the same
way in the child as in the parent — resident candidates come along as segments,
and a candidate that is resident but unpublished refuses the move, because a
child that silently missed it would recompute an FFT instead of reading one.
A stage nobody has materialized yet still stays parent-side: single-flight,
admission and R2/R3 live there (the family-3 rule, "the parent admits results").

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
from arrayscope.operations.evaluator import evaluate_image_snapshot, stage_document_key
from arrayscope.operations.pipeline import ArrayDocument
from arrayscope.operations.recipes import recipe_from_steps, steps_from_recipe
from arrayscope.operations.slabs import plan_slab, request_for_image, stage_key_for_candidate
from arrayscope.operations.source_transport import base_data_token, describe_base_data
from arrayscope.operations.stage_segments import (
    AttachedStageCache,
    close_retired_attachments,
    segment_for_value,
)

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

    ``stage_segments`` are the parent's cached stages this evaluation may read,
    as names a child maps; ``stage_cache_max_bytes`` is the publishing cache's
    envelope, which the evaluation reads to decide which stage a transition
    writes its output region for.
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
    stage_segments: tuple = ()
    stage_cache_max_bytes: int = 0


def evaluate_image_snapshot_request(request: ImageSnapshotRequest, token=None):
    """Evaluate one image snapshot from its descriptor — the moved body.

    Pure over ``request``: no session, no lifecycle, and no cache this process
    owns. The stage values it may read are exactly the ones the request names,
    served out of shared memory by :class:`AttachedStageCache`; an evaluation
    that would have needed anything else never became a request.
    """

    segments = tuple(request.stage_segments or ())
    if segments:
        # Between tasks, never during one: a segment this process retired may
        # still back the slab of the task that retired it.
        close_retired_attachments()
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
        stage_cache=(
            None
            if not segments
            else AttachedStageCache(segments, max_bytes=int(request.stage_cache_max_bytes))
        ),
        # The attached cache matches on a stage's position and shape, never on
        # the document key — which is just as well, because the child's own is
        # the only one it could compute.
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
        stage_document_key=stage_document_key,
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
    stage_document_key=None,
    evaluation_context=None,
    canonical_orientation: bool = False,
) -> ImageSnapshotRequest | None:
    """Build the request, or ``None`` when this evaluation must stay in-process."""

    if document is None or view_state is None:
        return None
    segments: tuple = ()
    if not _stage_free(document, view_state, stage_cache):
        segments = _resident_stage_segments(document, view_state, stage_cache, stage_document_key)
        if not segments:
            return None
    return _snapshot_request(
        document,
        view_state,
        colormap_lut=colormap_lut,
        shader_display=shader_display,
        provisional_histogram=provisional_histogram,
        evaluation_context=evaluation_context,
        canonical_orientation=canonical_orientation,
        stage_segments=segments,
        stage_cache_max_bytes=int(getattr(stage_cache, "max_bytes", 0) or 0),
    )


def run_stage_backed_snapshot(
    document,
    view_state,
    *,
    stage_value,
    colormap_lut=None,
    cancellation_token=None,
    shader_display: bool = False,
    provisional_histogram: bool = False,
    stage_cache_max_bytes: int = 0,
    evaluation_context=None,
    canonical_orientation: bool = False,
):
    """One tile off a stage the caller already holds, or ``None`` to stay here.

    The montage fan-in hands a tile its stage value directly, so this path never
    consults the cache at all: the value either has a published segment or it
    does not. ``None`` means the caller keeps its own in-process composition —
    which is the whole behaviour when the process pool is off, at the cost of
    one dictionary lookup.
    """

    segment = segment_for_value(stage_value)
    if segment is None:
        return None
    request = _snapshot_request(
        document,
        view_state,
        colormap_lut=colormap_lut,
        shader_display=shader_display,
        provisional_histogram=provisional_histogram,
        evaluation_context=evaluation_context,
        canonical_orientation=canonical_orientation,
        stage_segments=(segment,),
        stage_cache_max_bytes=int(stage_cache_max_bytes or 0),
    )
    if request is None:
        return None
    remember_document(request.identity, document, weak=True)
    return RemoteCall(IMAGE_SNAPSHOT_ENTRY, request)(cancellation_token)


def _snapshot_request(
    document,
    view_state,
    *,
    colormap_lut,
    shader_display: bool,
    provisional_histogram: bool,
    evaluation_context,
    canonical_orientation: bool,
    stage_segments: tuple,
    stage_cache_max_bytes: int,
) -> ImageSnapshotRequest | None:
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
        stage_segments=tuple(stage_segments),
        stage_cache_max_bytes=int(stage_cache_max_bytes),
    )


def _resident_stage_segments(document, view_state, stage_cache, document_key) -> tuple:
    """The published segments that answer this plan's candidates, or ``()``.

    The child must reach the *same* stage decision the parent would. Its
    lookup walks the candidates in one fixed order and takes the first hit, so
    it agrees with the parent exactly when the segments it is given are the
    candidates the parent's cache holds — no more, no fewer. Hence the two
    refusals: nothing resident (the stage is not materialized yet, and
    materializing it is parent-side work), or something resident that is not
    published (the child would skip past it and recompute an FFT prefix that
    already exists).
    """

    probe = getattr(stage_cache, "peek_containing_resident", None)
    if not callable(probe):
        return ()
    try:
        plan = plan_slab(document, request_for_image(view_state))
    except Exception:
        return ()
    candidates = tuple(getattr(plan.region_plan, "cache_candidates", ()) or ())
    if not candidates:
        return ()
    if document_key is None:
        document_key = stage_document_key(document)
    segments: list = []
    seen: set = set()
    for candidate in candidates:
        value = probe(stage_key_for_candidate(document_key, candidate))
        if value is None:
            continue
        segment = segment_for_value(value)
        if segment is None:
            return ()
        if segment.name in seen:
            continue
        seen.add(segment.name)
        segments.append(segment)
    return tuple(segments)


def _stage_free(document, view_state, stage_cache) -> bool:
    """Would this evaluation touch the stage cache at all?

    ``evaluate_slab_from_plan`` takes its pure branch exactly when there is no
    cache or no candidate, so asking the planner is not a heuristic — it is the
    same predicate the evaluator itself branches on. A document with no enabled
    operations has no stages to declare, which skips the planning entirely.

    A ``False`` is no longer a refusal, only a second question:
    :func:`_resident_stage_segments` asks whether the cache's answer can travel.
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
    "run_stage_backed_snapshot",
]
