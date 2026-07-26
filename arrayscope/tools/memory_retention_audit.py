"""Inspect ArrayScope's configured retained-memory envelope.

This is an ownership audit, not an RSS predictor. It applies the production
``MemoryPolicy`` and lists each independently budgeted owner that can retain
array bytes: source, display materialization, ROI demand regions, profile,
reusable stages, LOD pages, retained cross-session payloads, and the selected
backend's physical residency. Aliases and allocator behavior mean the sum can
exceed observed RSS; conversely driver memory, Python objects, and third-party
libraries remain outside it.

Use the report before proposing compression. A cold tier should replace bytes
at an owner whose misses are expensive, not create another unaccounted cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arrayscope.core.memory_policy import (
    MiB,
    SystemMemorySnapshot,
    compute_memory_policy,
    sample_system_memory,
)

GiB = 1024 * MiB


def run_audit(
    *,
    input_bytes: int,
    profile: str,
    render_cap_mb: int,
    source_mode: str,
    backend: str = "wgpu",
    system: SystemMemorySnapshot | None = None,
) -> dict:
    system = sample_system_memory() if system is None else system
    policy = compute_memory_policy(
        profile=profile,
        render_cap_mb=render_cap_mb,
        input_nbytes=max(0, int(input_bytes)),
        system=system,
    )
    tile_residency = min(
        int(policy.visible_render_budget_bytes),
        max(64 * MiB, int(policy.display_cache_budget_bytes) // 2),
        int(policy.user_render_cap_bytes),
    )
    lod_pages = max(256 * MiB, int(policy.display_cache_budget_bytes) // 2)
    source_resident = int(input_bytes) if source_mode == "eager" else 0
    backend = str(backend).strip().lower()
    if backend not in {"pyqtgraph", "wgpu"}:
        raise ValueError(f"unknown backend {backend!r}")
    owners = [
        _owner(
            "source",
            source_resident,
            "authoritative eager ndarray"
            if source_mode == "eager"
            else "lazy/mapped; logical bytes only",
            compress_at="file/chunk-store boundary",
        ),
        _owner(
            "display_materialization",
            policy.display_cache_budget_bytes,
            "upstream CPU display payloads; feeds every backend",
            compress_at="usually do not; misses are often cheap derivatives",
        ),
        _owner(
            "roi_region_demand",
            policy.display_cache_budget_bytes,
            "exact ROI subregions; independent demand-analysis LRU",
            compress_at="usually do not; first measure reuse and recompute cost",
        ),
        _owner(
            "profile_scalar",
            policy.profile_cache_budget_bytes,
            "small exact inspection results",
            compress_at="no",
        ),
        _owner(
            "stage",
            policy.stage_cache_budget_bytes,
            "exact reusable operation prefixes; misses may repeat FFT/IFFT",
            compress_at="best cold-tier candidate after measured pressure",
        ),
        _owner(
            "lod_pages",
            lod_pages,
            "derived presentation pages across reductions/reducers",
            compress_at="source-provided or GPU-native only",
        ),
        _owner(
            "retained_payload_refs",
            tile_residency,
            "cross-session acknowledged payloads; byte-bounded, may alias display/page arrays",
            compress_at="no separate tier; retain by value and physical residency",
        ),
    ]
    if backend == "pyqtgraph":
        owners.append(
            _owner(
                "pyqtgraph_raster_residency",
                tile_residency,
                "CPU ImageItem/tile backing for the software display backend",
                compress_at="no; evict/rebuild under the residency budget",
            )
        )
    else:
        owners.append(
            _owner(
                f"{backend}_gpu_page_residency",
                tile_residency,
                "device texture/page residency; not the evaluator ROI region cache",
                compress_at="GPU-native/pre-encoded only after a measured capacity gate",
                address_space="device",
            )
        )
    retained_upper_bound = sum(
        int(owner["budget_bytes"]) for owner in owners if owner["address_space"] == "host"
    )
    device_upper_bound = sum(
        int(owner["budget_bytes"]) for owner in owners if owner["address_space"] == "device"
    )
    transient = int(policy.visible_render_budget_bytes) + int(policy.prefetch_budget_bytes)
    return {
        "schema": "arrayscope.memory-retention-audit.v1",
        "profile": policy.profile.value,
        "source_mode": source_mode,
        "backend": backend,
        "input_logical_bytes": int(input_bytes),
        "system_total_bytes": int(policy.system_total_bytes),
        "system_available_bytes": int(policy.system_available_bytes),
        "process_rss_bytes": int(policy.process_rss_bytes),
        "owners": owners,
        "configured_retained_upper_bound_bytes": retained_upper_bound,
        "configured_device_upper_bound_bytes": device_upper_bound,
        "visible_and_prefetch_transient_bytes": transient,
        "configured_plus_transient_bytes": retained_upper_bound + transient,
        "notes": [
            "Configured maxima are independent guardrails, not a global process cap.",
            "Host and device address spaces are reported separately.",
            "The host sum intentionally double-counts possible aliases to expose retention authority.",
            "The display cache is upstream of every backend; roi_region_demand is ROI analysis, not GPU page storage.",
            "Driver overhead, Python objects, and third-party allocator retention are extra.",
        ],
    }


def _owner(
    name: str,
    budget_bytes: int,
    purpose: str,
    *,
    compress_at: str,
    address_space: str = "host",
) -> dict:
    return {
        "name": name,
        "budget_bytes": max(0, int(budget_bytes)),
        "purpose": purpose,
        "compression_posture": compress_at,
        "address_space": address_space,
    }


def _format(result: dict) -> str:
    gib = float(GiB)
    lines = [
        f"profile={result['profile']} source={result['source_mode']} "
        f"backend={result['backend']} "
        f"input={result['input_logical_bytes'] / gib:.2f} GiB",
        "owner                         budget GiB  purpose",
    ]
    lines.extend(
        f"{owner['name']:<29}{owner['budget_bytes'] / gib:>10.2f}  "
        f"{owner['address_space']}: {owner['purpose']}"
        for owner in result["owners"]
    )
    lines.extend(
        (
            f"retained upper bound: {result['configured_retained_upper_bound_bytes'] / gib:.2f} GiB",
            f"device residency bound: "
            f"{result['configured_device_upper_bound_bytes'] / gib:.2f} GiB",
            f"+ visible/prefetch transient: "
            f"{result['visible_and_prefetch_transient_bytes'] / gib:.2f} GiB",
            "This is an ownership envelope, not measured RSS; see notes in JSON output.",
        )
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-gib", type=float, default=4.0)
    parser.add_argument(
        "--profile",
        choices=("conservative", "balanced", "aggressive", "custom"),
        default="balanced",
    )
    parser.add_argument("--render-cap-mb", type=int, default=512)
    parser.add_argument("--source-mode", choices=("lazy", "eager"), default="lazy")
    parser.add_argument("--backend", choices=("pyqtgraph", "wgpu"), default="wgpu")
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: tuple[str, ...] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_audit(
        input_bytes=max(0, int(float(args.input_gib) * GiB)),
        profile=args.profile,
        render_cap_mb=args.render_cap_mb,
        source_mode=args.source_mode,
        backend=args.backend,
    )
    print(_format(result))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
