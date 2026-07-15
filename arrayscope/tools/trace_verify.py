"""Verify final visible-tile presentation invariants in an event trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MAX_IDENTICAL_ACKS = 25


def _ack_satisfies_target(ack: dict[str, object], target: dict[str, object]) -> bool:
    if ack.get("source_index") != target.get("source_index"):
        return False
    level = ack.get("level")
    target_level = target.get("target_level")
    return (
        level is not None
        and target_level is not None
        and int(level) <= int(target_level)
    )


def verify_trace(
    path: str | Path,
    *,
    expect_targets: int | None = None,
    max_identical_acks: int = MAX_IDENTICAL_ACKS,
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
    """

    targets: dict[int, dict[str, object]] = {}
    acknowledgements: dict[int, dict[str, object]] = {}
    first_ack_sequences: dict[int, int] = {}
    identical_ack_counts: dict[tuple[object, ...], int] = {}
    stalls: list[dict[str, object]] = []
    lifecycle_events = 0
    event_count = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event_count += 1
        event = json.loads(line)
        kind = str(event.get("kind", ""))
        if kind == "stall":
            stalls.append(event)
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
                # re-ack).  Contract: the emitter must only publish this edge
                # for a payload whose source/level satisfy the current target.
                target = targets.get(tile)
                if target is not None and int(event.get("sequence", 0) or 0) >= int(
                    target["sequence"]
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
    return {
        "ok": not violations,
        "event_count": event_count,
        "lifecycle_events": lifecycle_events,
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
    args = parser.parse_args(argv)
    result = verify_trace(
        args.trace,
        expect_targets=args.expect_targets,
        max_identical_acks=args.max_identical_acks,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if bool(result["ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
