"""Verify final visible-tile presentation invariants in an event trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify_trace(path: str | Path) -> dict[str, object]:
    """Replay lifecycle scope and prove every final target reached exact ack."""

    targets: dict[int, dict[str, object]] = {}
    acknowledgements: dict[int, dict[str, object]] = {}
    event_count = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event_count += 1
        event = json.loads(line)
        kind = str(event.get("kind", ""))
        if kind == "lifecycle":
            edge = str(event.get("edge", ""))
            tile = int(event.get("tile", -1))
            if edge == "target_required":
                targets[tile] = {
                    "source_index": event.get("source_index"),
                    "target_level": event.get("target_level"),
                    "sequence": int(event.get("sequence", 0) or 0),
                }
                acknowledgements.pop(tile, None)
            elif edge == "target_released":
                targets.pop(tile, None)
                acknowledgements.pop(tile, None)
            continue
        if kind != "backend_ack" or not bool(event.get("accepted", False)):
            continue
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

    missing = tuple(sorted(set(targets).difference(acknowledgements)))
    violations = tuple(
        {
            "invariant": "final_required_target_acknowledged",
            "tile": tile,
            "target": targets[tile],
        }
        for tile in missing
    )
    return {
        "ok": not violations,
        "event_count": event_count,
        "required_targets": len(targets),
        "acknowledged_targets": len(acknowledgements),
        "violations": violations,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify final visible-tile invariants in an ArrayScope trace"
    )
    parser.add_argument("trace")
    args = parser.parse_args(argv)
    result = verify_trace(args.trace)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if bool(result["ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
