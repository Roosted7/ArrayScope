"""The prepared presentation transaction and its keep-latest mailbox.

ADR 0065 stage 2 slice C. The GUI thread's montage commit becomes: take a
prepared transaction, validate it, submit it, run the Qt residue, post the
acknowledgement inputs back. Everything the take needs — the delta, the level
decisions, the governor's render-pass verdict, the presentation decision, the
snapshots the residue reads — is frozen here by the presentation owner before
the GUI ever wakes up.

Three things are load-bearing and none of them is the dataclass:

- **Keep-latest by generation, exact-key take.** Same contract as
  :class:`~arrayscope.presentation.prepared_uploads.PreparedUploadMailbox`, and
  correct here for the same reason: two transactions under one key supersede
  each other, and only the newest can pass the take's validation anyway. A key
  that is not present simply was not prepared, and the caller's inline path
  covers it. *A miss is a normal outcome.*
- **Closed counter equations.** ``published == taken + replaced + discarded +
  cleared + pending``. A slice that always falls back inline is a no-op wearing
  a success face, and only the hit/miss/inline split can say which one shipped.
- **Validation is identity, never content.** The tuple in
  :class:`TransactionIdentity` generalises ``_prepared_atomic_transaction_current``
  to every transaction. Both ``session_id`` *and* ``id(session)`` are in it:
  an index-window retarget advances the generation on the *same* object, so
  either half alone lets a predecessor's transaction impersonate its successor
  — which is ``fired-generation-stale`` re-expressed as a wrong-transaction
  take.

Nothing in this module imports Qt, and nothing in it may: the owner thread
builds every value here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TransactionIdentity:
    """Everything a take compares before it trusts a prepared transaction.

    Read the fields as five independent questions, because they fail
    independently:

    ``session_id``/``session_object_id``
        Whose session is this? Both halves, per the module docstring.
    ``level_revision``/``base_revision``
        Does the delta still chain onto the live presentation state? The base
        revision is the one continuity check that catches a commit having
        landed between prepare and take.
    ``scope``
        The tiles the transaction claims. A live required scope that has grown
        past it is not fatal (the next transaction covers the remainder); one
        that no longer *contains* it is, because the delta names tiles the
        session has stopped requiring.
    ``capabilities_id``
        The backend surface identity. A rebuilt image view is a different
        submit target with a different upload contract.
    ``eviction_stamp``
        Decision 5's residency guard: ``bind`` only adds coverage, so an
        unchanged stamp proves nothing the prepare trusted has been unbound.
    """

    session_id: int
    session_object_id: int
    session_key: Any
    level_revision: int
    base_revision: int
    render_round_id: str
    scope: frozenset
    capabilities_id: int
    eviction_stamp: tuple | None
    payload_markers: dict = field(default_factory=dict)

    def matches_session(self, session) -> bool:
        return int(getattr(session, "session_id", -1) or 0) == int(self.session_id) and id(
            session
        ) == int(self.session_object_id)


@dataclass(frozen=True)
class PrepTimings:
    """The owner's half of the per-commit timing record (decision 12).

    Frozen because it crosses a thread. Merged with :class:`SubmitTimings`
    into one :class:`CommitTimings` published by a single attribute swap, so a
    diagnostics reader can never see the prepare block of one commit beside the
    submit block of another — which is exactly what twenty independent
    ``_last_montage_*`` attributes guaranteed once the two halves ran on
    different threads.
    """

    payload_build_ms: float = 0.0
    payload_seed_ms: float = 0.0
    payload_level_ms: float = 0.0
    payload_state_ms: float = 0.0
    payload_build_call_ms: float = 0.0
    payload_priority_ms: float = 0.0
    payload_active_ms: float = 0.0
    prepare_apply_ms: float = 0.0
    prepare_stats_ms: float = 0.0
    prepare_metadata_ms: float = 0.0
    prepare_source_ms: float = 0.0
    prepare_histogram_ms: float = 0.0


@dataclass(frozen=True)
class SubmitTimings:
    """The GUI's half: everything measured from the take to the ack post."""

    tile_commit_ms: float = 0.0
    tile_layer_apply_ms: float = 0.0
    set_image_ms: float = 0.0
    submit_span_ms: float = 0.0
    acknowledge_ms: float = 0.0
    retained_store_ms: float = 0.0
    state_publish_ms: float = 0.0
    geometry_sync_ms: float = 0.0
    overlay_update_ms: float = 0.0
    identity_check_ms: float = 0.0
    followup_ms: float = 0.0


@dataclass(frozen=True)
class CommitTimings:
    """One commit's whole timing story, published atomically.

    ``submit_span_ms`` is the redefinition decision 6 makes exactly once: see
    ``FramePipelineEffects._record_commit_feedback``, which is the one place
    the definition lives.
    """

    prep: PrepTimings = PrepTimings()
    submit: SubmitTimings = SubmitTimings()


@dataclass(frozen=True)
class PreparedPresentationTransaction:
    """One ready-to-submit montage commit.

    Built by :meth:`FramePipelineEffects._prepare_tile_layer_transaction`,
    either on the presentation owner (published here) or inline on the GUI
    thread (handed straight to the submit). The two producers are the same
    code, which is what makes the inline path a *correctness anchor* rather
    than a second implementation to keep in sync.

    Field groups, in the order the consumers use them:

    - identity + presentation state: ``identity``, ``tile_state``,
      ``base_tile_state``, ``tile_delta``, ``active_payloads``,
      ``tile_states``, ``dirty_tiles``;
    - the submit's own inputs: ``display_image``, ``geometry``,
      ``capabilities``, ``semantic_source``, ``applied_level_source``,
      ``histogram_plot_data``, ``requested_levels``, ``commit_kind``;
    - the atomic-successor block: ``atomic_successor`` and friends,
      ``requires_hidden_warm``, ``warm_levels``, ``prepared_atomic``;
    - the governor block: ``limits``, ``pass_budget_ms``,
      ``governor_details``, ``render_pass_token``, ``pass_kind``,
      ``structural_key``, ``representation_key``;
    - the decision hoisted out of ``_commit_direct_delta``:
      ``presentation_decision`` with the ``previous_frame`` it was decided
      against;
    - what the residue and the ack need: ``evidence_obligation``,
      ``publish_*`` flags, ``coverage_phase_before``,
      ``presented_before_commit``, ``prep``.
    """

    identity: TransactionIdentity
    display_image: Any
    geometry: Any
    tile_state: Any
    base_tile_state: Any
    tile_delta: Any
    active_payloads: dict
    tile_states: tuple = ()
    dirty_tiles: tuple = ()

    capabilities: Any = None
    semantic_source: Any = None
    applied_level_source: Any = None
    histogram_plot_data: Any = None
    requested_levels: Any = None
    commit_kind: Any = None

    #: How many upserts the delta carried. Cheap to recompute and carried
    #: anyway: the take mirrors it onto the renderer before the submit, and a
    #: consumer that reached for ``tile_delta.upserts`` there would be walking
    #: a dict the owner built for a different question.
    delta_upserts: int = 0

    first_display_commit: bool = False
    explicit_auto: bool = False
    decision_force_auto: bool = False
    semantic_commit: bool = False
    frame_commit: bool = False

    atomic_successor: bool = False
    cpu_atomic_successor: bool = False
    shader_atomic_successor: bool = False
    atomic_successor_pending_before: bool = False
    atomic_prepared_reused: bool = False
    atomic_fast_built: bool = False
    atomic_fast_reject_reason: str = ""
    requires_hidden_warm: bool = False
    warm_levels: Any = None
    prepared_atomic: Any = None

    limits: dict = field(default_factory=dict)
    max_upserts: int = 0
    unbounded_reason: str = ""
    pass_budget_ms: Any = None
    governor_details: tuple = ()
    render_pass_token: Any = None
    pass_kind: str = ""
    structural_key: Any = None
    representation_key: Any = None
    frame_critical_at_prepare: bool = False
    interactive_at_prepare: bool = False

    presentation_decision: Any = None
    previous_frame: Any = None
    render_context: Any = None
    direct_delta_eligible: bool = False

    evidence_obligation: tuple | None = None
    evidence_required: bool = False
    evidence_deferred: bool = False
    publish_metadata: bool = False
    publish_first_pass_histogram: bool = False
    publish_histogram_plot: bool = False
    histogram_metadata_pending: bool = False
    level_decision: dict = field(default_factory=dict)

    coverage_phase_before: bool = False
    presented_before_commit: frozenset = frozenset()
    dirty_before: int = 0
    pending_before: int = 0
    presented_before: int = 0
    fast_drain: bool = False

    prep: PrepTimings = PrepTimings()
    #: Set when the owner prepared this; ``False`` for the inline path. Only
    #: diagnostics — the take validates identity, never provenance.
    prepared_off_thread: bool = False

    @property
    def key(self):
        """The mailbox key: session identity plus the level round."""

        return (
            int(self.identity.session_id),
            int(self.identity.session_object_id),
            int(self.identity.level_revision),
        )


@dataclass(frozen=True)
class AckInputs:
    """What the GUI observed at submit, frozen for the owner's ack handler.

    ``resident_histogram_evidence`` is ONE bool for both of its consumers on
    purpose (decision 8): the probe is a live backend read, and two calls
    inside one commit can legitimately disagree, which would let the coverage
    barrier and the evidence-selection branch describe different commits.
    """

    transaction: PreparedPresentationTransaction
    report: Any = None
    committed_levels: Any = None
    shader_windowing: bool = False
    resident_histogram_evidence: bool = False
    residency_snapshot: Any = None
    committed_frame: Any = None
    continuity_range: Any = None
    submit: SubmitTimings = SubmitTimings()
    draw_pending_at_submit: bool = False
    interactive_at_submit: bool = False
    first_pass_histogram_published: bool = False
    applied: bool = True
    commit_start: float = 0.0


__all__ = [
    "AckInputs",
    "CommitTimings",
    "PrepTimings",
    "PreparedPresentationTransaction",
    "SubmitTimings",
    "TransactionIdentity",
]
