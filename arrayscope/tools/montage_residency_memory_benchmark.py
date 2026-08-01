"""Does a montage viewport walk hold its memory, and inside whose budget?

This used to live in ``tests/app/test_memory_stress.py`` as absolute assertions
on process RSS and on cache-bytes-versus-installed-cap.  It could not stay
there.  Every threshold it compared against is derived, at the moment of the
run, from ``psutil.virtual_memory().available`` — the memory policy in
``arrayscope.core.memory_policy`` sizes the display, page, profile and stage
budgets as fractions of whatever the box happens to have free.  So a second
workload on the machine shrinks the caps under a working set that was already
resident, and the gate fails with something like ``assert 2948096 <= 2423808``:
a true statement about a contended machine and no statement at all about this
code.  Measured 2026-08-01 with three concurrent suites.

What remains in pytest is the part that does not move with the machine: walking
the same semantic windows twice must not grow the accounted working set, which
is a cache-keying property and is asserted relatively.

What this tool adds back is the measurement, taken where a measurement means
something: one process, a quiet box, every input recorded next to every output.
It reports the policy inputs it ran under, so a number from a loaded machine is
identifiable as such rather than silently comparable with a good one.

    tools/env python -m arrayscope.tools.montage_residency_memory_benchmark
    tools/env python -m arrayscope.tools.montage_residency_memory_benchmark --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

MiB = 1024 * 1024

#: The walk: three montage rows, twice.  The second pass revisits the exact
#: semantic windows of the first, which is what makes growth attributable to
#: cache keying rather than to a larger working set.
ROWS = (1, 3, 5)
PASSES = 2
TILE = 256


@dataclass(frozen=True)
class Sample:
    """One settled viewport, measured."""

    label: str
    rss_bytes: int
    page_cache_used: int
    page_cache_cap: int
    display_cache_used: int
    display_cache_cap: int

    @property
    def accounted_bytes(self) -> int:
        return int(self.page_cache_used) + int(self.display_cache_used)

    @property
    def owner_cap_bytes(self) -> int:
        return int(self.page_cache_cap) + int(self.display_cache_cap)

    @property
    def within_caps(self) -> bool:
        return (
            self.page_cache_used <= self.page_cache_cap
            and self.display_cache_used <= self.display_cache_cap
        )


def _memory_inputs() -> dict[str, int | None]:
    """What the policy sized itself from, recorded beside the result.

    Without this a run is uninterpretable: the caps below are fractions of
    ``available``, so two runs with different free memory are measuring
    different products.
    """

    try:
        import psutil
    except ImportError:
        return {"available_bytes": None, "total_bytes": None, "load_average": None}
    virtual = psutil.virtual_memory()
    try:
        load = os.getloadavg()[0]
    except (OSError, AttributeError):
        load = None
    return {
        "available_bytes": int(virtual.available),
        "total_bytes": int(virtual.total),
        "load_average": load,
    }


def _settled_sample(window, process, label: str) -> Sample:
    session = window._frame_session
    page_cache = session.lod_page_cache
    display_cache = window.operation_evaluator.display_cache_diagnostics()
    return Sample(
        label=label,
        rss_bytes=int(process.memory_info().rss),
        page_cache_used=int(page_cache.bytes_used),
        page_cache_cap=int(page_cache.max_bytes),
        display_cache_used=int(display_cache.bytes_used),
        display_cache_cap=int(display_cache.max_bytes),
    )


def run_benchmark(*, sources: int = 64, columns: int = 8) -> dict:
    """Open a montage, walk it twice, and report residency against its budgets."""

    import psutil
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets

    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.core.memory_policy import MIN_RENDER_CAP_MB
    from arrayscope.tools.presentation_settlement import (
        presentation_is_settled,
        presentation_settlement_diagnostic,
        presentation_target_token,
    )
    from arrayscope.window import ArrayScopeWindow

    app = pg.mkQApp()
    process = psutil.Process()
    data = np.zeros((TILE, TILE, sources), dtype=np.float32)
    window = ArrayScopeWindow(data)
    # The smallest render budget the application will actually adopt: the page
    # cache is governed by this cap, the display cache by its own policy budget,
    # and they are reported separately because they are separate owners.
    window.app_settings = AppSettingsState(
        theme=window.app_settings.theme,
        prefetch_nearby_slices=window.app_settings.prefetch_nearby_slices,
        panel_resize_behavior=window.app_settings.panel_resize_behavior,
        fft_backend=window.app_settings.fft_backend,
        fft_workers=window.app_settings.fft_workers,
        memory_profile=window.app_settings.memory_profile,
        render_memory_budget_mb=MIN_RENDER_CAP_MB,
    )

    def settle(previous_target=None, *, label: str) -> None:
        # No deadline: this is a measurement run on a machine that is supposed
        # to be quiet, and a wall-clock cap here would silently truncate the
        # walk on the very machine whose numbers are least trustworthy.  A
        # genuine hang is visible as a tool that does not return.
        while True:
            if previous_target is None:
                if presentation_is_settled(window, require_quiescent=True):
                    return
            else:
                current = presentation_target_token(window)
                if (
                    current is not None
                    and current != previous_target
                    and presentation_is_settled(
                        window, expected_target=current, require_quiescent=True
                    )
                ):
                    return
            loop = QtWidgets.QApplication.instance()
            if loop is None:
                raise RuntimeError(f"no Qt application while settling {label}")
            app.processEvents()

    samples: list[Sample] = []
    try:
        window.show()
        settle(label="startup")
        # Backend construction is a one-time process high-water mark unrelated
        # to montage residency; the first settled presentation is the boundary.
        baseline_rss = int(process.memory_info().rss)

        initial_target = presentation_target_token(window)
        window._set_view_state(
            window.view_state.with_montage_axis(
                2, columns=columns, indices=tuple(range(sources)), text=":"
            )
        )
        window.render(reason="residency-benchmark")
        settle(initial_target, label="montage")
        samples.append(_settled_sample(window, process, "montage"))

        for index in range(PASSES):
            for row in ROWS:
                previous_target = presentation_target_token(window)
                y0 = row * (TILE + 1)
                window.img_view.getView().setRange(
                    xRange=(0, TILE), yRange=(y0, y0 + TILE), padding=0
                )
                window.update_image_view()
                settle(previous_target, label=f"pass{index}-row{row}")
                samples.append(_settled_sample(window, process, f"pass{index}-row{row}"))
        diagnostic = presentation_settlement_diagnostic(window)
    finally:
        window.close()

    walk = samples[1:]
    first_walk = [s.accounted_bytes for s in walk[: len(ROWS)]]
    second_walk = [s.accounted_bytes for s in walk[len(ROWS) :]]
    rss = [s.rss_bytes for s in samples]
    return {
        "memory_inputs": _memory_inputs(),
        "baseline_rss_bytes": baseline_rss,
        "samples": [asdict(s) for s in samples],
        "accounted_peak_bytes": max(s.accounted_bytes for s in samples),
        "owner_cap_peak_bytes": max(s.owner_cap_bytes for s in samples),
        "every_sample_within_caps": all(s.within_caps for s in samples),
        "first_walk_peak_bytes": max(first_walk) if first_walk else 0,
        "second_walk_peak_bytes": max(second_walk) if second_walk else 0,
        "rss_growth_bytes": max(rss) - baseline_rss,
        "rss_spread_bytes": max(rss) - min(rss),
        "final_settlement": diagnostic,
    }


def _format(result: dict) -> str:
    inputs = result["memory_inputs"]
    available = inputs.get("available_bytes")
    lines = [
        "montage residency memory",
        f"  policy inputs   available={_mib(available)} total={_mib(inputs.get('total_bytes'))} "
        f"load={inputs.get('load_average')}",
        f"  installed caps  {_mib(result['owner_cap_peak_bytes'])} (page + display, peak)",
        f"  accounted peak  {_mib(result['accounted_peak_bytes'])}",
        f"  within caps     {result['every_sample_within_caps']}",
        f"  walk peaks      first={_mib(result['first_walk_peak_bytes'])} "
        f"second={_mib(result['second_walk_peak_bytes'])}",
        f"  rss growth      {_mib(result['rss_growth_bytes'])} over the settled baseline",
        f"  rss spread      {_mib(result['rss_spread_bytes'])} across the walk",
    ]
    if inputs.get("load_average") is not None and inputs["load_average"] > 2.0:
        lines.append(
            "  NOTE: the caps above are fractions of free memory on a machine that was "
            "not idle. Re-run quiet before comparing these numbers with anything."
        )
    return "\n".join(lines)


def _mib(value) -> str:
    if value is None:
        return "n/a"
    return f"{int(value) / MiB:.1f} MiB"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", type=int, default=64, help="montage tile count")
    parser.add_argument("--columns", type=int, default=8, help="montage columns")
    parser.add_argument("--json", type=Path, default=None, help="also write the raw result here")
    return parser


def main(argv: tuple[str, ...] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_benchmark(sources=max(1, int(args.sources)), columns=max(1, int(args.columns)))
    print(_format(result))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
