"""Verify final visible-tile presentation invariants in an event trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MAX_IDENTICAL_ACKS = 25
MAX_IDENTICAL_COMMIT_BAILS = 25


def _commit_bail_signature(event: dict[str, object]) -> tuple[tuple[str, object], ...]:
    """Stable semantic signature for one commit deferral.

    Sequence and timestamp identify occurrences, not progress.  Every other
    field is part of the reason/state contract emitted by ``commit_bail``;
    canonical JSON keeps nested details hashable without weakening that
    contract to a hand-maintained subset.
    """

    return tuple(
        (key, json.dumps(value, sort_keys=True, separators=(",", ":"), default=repr))
        for key, value in sorted(event.items())
        if key not in {"sequence", "ts_ns"}
    )


def _ack_satisfies_target(ack: dict[str, object], target: dict[str, object]) -> bool:
    if ack.get("source_index") != target.get("source_index"):
        return False
    level = ack.get("level")
    target_level = target.get("target_level")
    if level is None or target_level is None:
        return False
    quality = str(ack.get("quality", "") or "exact")
    # Mirror lifecycle settlement (`_quality_lod_satisfies_target`): exact
    # pixels satisfy their level or any coarser demand; retained
    # fallback/preview pixels satisfy only a *strictly* coarser demand — an
    # equal-level fallback still owes exact target work.
    if quality in {"fallback", "preview"}:
        return int(level) < int(target_level)
    return int(level) <= int(target_level)


def verify_trace(
    path: str | Path,
    *,
    expect_targets: int | None = None,
    max_identical_acks: int = MAX_IDENTICAL_ACKS,
    max_identical_commit_bails: int = MAX_IDENTICAL_COMMIT_BAILS,
) -> dict[str, object]:
    """Replay lifecycle scope and prove every final target reached exact ack.

    ``expect_targets`` guards against vacuous passes: a trace whose lifecycle
    edges were never emitted (renamed event kind, broken emitter, wrong file)
    replays as an empty scope and would otherwise verify clean.  Scenario
    harnesses know their tile count and must pass it.

    ``max_identical_acks`` catches livelock: the stall watchdog (V3) sees
    deadlocks — nothing happening — but a presentation loop re-acknowledging
    the same payload forever keeps its gate armed and stays invisible to it.
    A healthy tile acknowledges an identity once, maybe a handful of times
    across level rebinding; hundreds of identical accepted acks mean the
    presentation layer is spinning (observed: ~5,500 identical acks per tile
    per minute at idle).

    ``max_identical_commit_bails`` catches the complementary no-progress
    loop: a commit barrier repeatedly defers the same transaction state while
    its named wakeup produces no change.  Healthy pacing changes at least one
    signature field as work drains; dozens of byte-identical semantic bails
    mean the waited-for complement has no live owner.

    Every ``commit_batch``/``backend_complete`` event must report an empty
    ``identity_rejected``: a rejected upsert is a payload the session queued
    against a target its typed identity can never satisfy (the silent
    re-emit loop behind the 2026-07-16 stale/empty-tile stall), which is a
    defect even when the run otherwise converges.

    A backend commit must also report no exact upsert inside an open preview
    pass. Retaining already-presented exact pixels is valid stronger coverage;
    introducing new exact pixels while another slot still awaits its preview
    is the mixed-quality race the plan-wide first-pixel pass forbids.
    """

    targets: dict[int, dict[str, object]] = {}
    acknowledgements: dict[int, dict[str, object]] = {}
    first_ack_sequences: dict[int, int] = {}
    identical_ack_counts: dict[tuple[object, ...], int] = {}
    identical_commit_bail_counts: dict[tuple[tuple[str, object], ...], int] = {}
    stalls: list[dict[str, object]] = []
    identity_rejected_commits: list[dict[str, object]] = []
    lifecycle_events = 0
    event_count = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event_count += 1
        event = json.loads(line)
        kind = str(event.get("kind", ""))
        if kind == "commit_bail":
            signature = _commit_bail_signature(event)
            identical_commit_bail_counts[signature] = (
                identical_commit_bail_counts.get(signature, 0) + 1
            )
            continue
        if kind == "stall":
            stalls.append(event)
            continue
        if kind == "commit_batch":
            # A backend_complete commit reporting identity-rejected upserts
            # means the session queued a payload whose typed identity can
            # never satisfy its own target — the silent re-emit loop behind
            # the 2026-07-16 stale/empty-tile stall.  Healthy replays commit
            # zero such payloads even when every target eventually settles.
            if (
                str(event.get("phase", "")) == "backend_complete"
                and tuple(event.get("identity_rejected", ()) or ())
            ):
                identity_rejected_commits.append(event)
            continue
        if kind == "lifecycle":
            lifecycle_events += 1
            edge = str(event.get("edge", ""))
            tile = int(event.get("tile", -1))
            if edge == "target_required":
                targets[tile] = {
                    "source_index": event.get("source_index"),
                    "target_level": event.get("target_level"),
                    "sequence": int(event.get("sequence", 0) or 0),
                }
                # Backend acknowledgement is idempotent: a retarget that a
                # tile's committed payload already satisfies is deliberately
                # not re-acknowledged.  Retain the prior ack exactly when it
                # satisfies the new target; anything else must earn a fresh
                # acknowledgement.
                retained = acknowledgements.get(tile)
                if retained is not None and not _ack_satisfies_target(retained, targets[tile]):
                    acknowledgements.pop(tile, None)
                    first_ack_sequences.pop(tile, None)
            elif edge == "target_satisfied_retained":
                # Production closed this target with an already-acknowledged
                # compatible payload (idempotent backends do not re-upload or
                # re-ack).  The emitter publishes the retained payload's
                # source/level/quality; re-judge them here so a buggy emitter
                # cannot vouch for a payload the settlement rule rejects.
                target = targets.get(tile)
                if (
                    target is not None
                    and int(event.get("sequence", 0) or 0) >= int(target["sequence"])
                    and _ack_satisfies_target(event, target)
                ):
                    acknowledgements[tile] = event
                    first_ack_sequences.setdefault(
                        tile, int(event.get("sequence", 0) or 0)
                    )
            elif edge == "target_released":
                targets.pop(tile, None)
                acknowledgements.pop(tile, None)
                first_ack_sequences.pop(tile, None)
            continue
        if kind != "backend_ack" or not bool(event.get("accepted", False)):
            continue
        ack_signature = (
            event.get("tile"),
            event.get("source_index"),
            event.get("level"),
            event.get("quality"),
        )
        identical_ack_counts[ack_signature] = identical_ack_counts.get(ack_signature, 0) + 1
        tile = int(event.get("tile", -1))
        target = targets.get(tile)
        if target is None:
            continue
        if int(event.get("sequence", 0) or 0) < int(target["sequence"]):
            continue
        source_matches = event.get("source_index") == target["source_index"]
        quality = str(event.get("quality", "") or "")
        level = event.get("level")
        target_level = target["target_level"]
        level_matches = (
            level is not None
            and target_level is not None
            and int(level) <= int(target_level)
        )
        if source_matches and quality not in {"fallback", "preview", ""} and level_matches:
            acknowledgements[tile] = event
            first_ack_sequences.setdefault(tile, int(event.get("sequence", 0) or 0))

    missing = tuple(sorted(set(targets).difference(acknowledgements)))
    violations = [
        {
            "invariant": "final_required_target_acknowledged",
            "tile": tile,
            "target": targets[tile],
        }
        for tile in missing
    ]
    violations.extend(
        {
            "invariant": "no_stall_events",
            "session_id": event.get("session_id"),
            "owner_chain": event.get("owner_chain"),
        }
        for event in stalls
    )
    violations.extend(
        {
            "invariant": "no_identity_rejected_commits",
            "session_id": event.get("session_id"),
            "revision": event.get("revision"),
            "identity_rejected": tuple(event.get("identity_rejected") or ()),
        }
        for event in identity_rejected_commits
    )
    if event_count == 0:
        violations.append({"invariant": "trace_not_empty"})
    elif lifecycle_events == 0:
        violations.append({"invariant": "lifecycle_events_present"})
    if expect_targets is not None and len(targets) != int(expect_targets):
        violations.append(
            {
                "invariant": "required_target_count",
                "expected": int(expect_targets),
                "observed": len(targets),
            }
        )
    if max_identical_acks > 0:
        violations.extend(
            {
                "invariant": "no_acknowledgement_churn",
                "tile": signature[0],
                "source_index": signature[1],
                "level": signature[2],
                "quality": signature[3],
                "identical_acks": count,
                "limit": int(max_identical_acks),
            }
            for signature, count in sorted(
                identical_ack_counts.items(), key=lambda item: -item[1]
            )
            if count > int(max_identical_acks)
        )
    if max_identical_commit_bails > 0:
        violations.extend(
            {
                "invariant": "no_identical_commit_bail_loop",
                "outcome": json.loads(dict(signature).get("outcome", '""')),
                "wakeup": json.loads(dict(signature).get("wakeup", '""')),
                "session_id": json.loads(dict(signature).get("session_id", "null")),
                "identical_commit_bails": count,
                "limit": int(max_identical_commit_bails),
            }
            for signature, count in sorted(
                identical_commit_bail_counts.items(), key=lambda item: -item[1]
            )
            if count > int(max_identical_commit_bails)
        )
    return {
        "ok": not violations,
        "event_count": event_count,
        "lifecycle_events": lifecycle_events,
        "identity_rejected_commits": len(identity_rejected_commits),
        "required_targets": len(targets),
        "acknowledged_targets": len(acknowledgements),
        "acknowledgement_order": tuple(
            tile for tile, _sequence in sorted(first_ack_sequences.items(), key=lambda item: item[1])
        ),
        "violations": tuple(violations),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify final visible-tile invariants in an ArrayScope trace"
    )
    parser.add_argument("trace")
    parser.add_argument(
        "--expect-targets",
        type=int,
        default=None,
        help="Exact final required-target count; guards against vacuously clean traces",
    )
    parser.add_argument(
        "--max-identical-acks",
        type=int,
        default=MAX_IDENTICAL_ACKS,
        help="Identical accepted acks per identity before flagging livelock churn; 0 disables",
    )
    parser.add_argument(
        "--max-identical-commit-bails",
        type=int,
        default=MAX_IDENTICAL_COMMIT_BAILS,
        help="Identical semantic commit bails before flagging a no-owner loop; 0 disables",
    )
    args = parser.parse_args(argv)
    result = verify_trace(
        args.trace,
        expect_targets=args.expect_targets,
        max_identical_acks=args.max_identical_acks,
        max_identical_commit_bails=args.max_identical_commit_bails,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if bool(result["ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
