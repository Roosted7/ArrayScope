"""Montage -> single slice -> montage, without a reload (WGPU).

The user's report: "I went from the full montage, to a single slice, then back
to the full montage (WITHOUT a reload) — many tiles were missing and needed to
be slowly added. They should have been present already, so only one single
atlas change away."

The field trace (`arrayscope-diagnostics-20260803-160840.jsonl`, t=56.9-63,
`session_id` 148 -> 149 -> 150) shows the montage rebuilding its presented set
71 -> 82 -> 135 -> 163 -> 249 -> 272 over 2.6 s while `wgpu_uploads_total` and
`wgpu_upload_bytes_total` stay flat.  That reads like a governed commit drip and
it is not one: at every sample of that ramp `dirty_payload_tiles` and
`pending_payload_upserts` are 0-12 out of 272, so the commit path is holding
nothing back.  What moves in lockstep with the ramp is
`tile_lod_preview_reduced_scheduled` (1002 -> 1202, i.e. **200 fresh floor-rung
producer tasks**) and `tile_lod_preview_presentations` (0 -> 22 -> 63 -> 97 ->
180 -> 200).  The refill is PRODUCTION-bound, not admission-bound.

The two tests below separate the fact from the defect:

* the return leg uploads nothing — so whatever those producers computed, the
  backend already had.  That is the evidence the work is redundant, and it must
  not regress into "we fixed the drip by uploading everything again";
* the return leg re-produces a preview for nearly every tile, which is the
  defect itself and is therefore an expected failure until it is fixed.

The 72-of-272 asymmetry in the field is the same statement from the other side:
the tiles whose native ``RenderedTile`` survived in the window-scoped display
cache came back in ONE commit (71 of them at t=60.1); only the tiles whose
wrapper died with the session dripped.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.settle import settle_window
from tests.ui.helpers import (
    frame_session_settled,
    make_backend_window,
    restore_default_backend,
    use_wgpu_backend,
)

SOURCES = 64
COLUMNS = 8
TILE = 160


def _montage_data() -> np.ndarray:
    """Per-source gradients: a tile shown in the wrong slot is a pixel difference."""

    yy = np.arange(TILE, dtype=np.float32)[:, None, None]
    xx = np.arange(TILE, dtype=np.float32)[None, :, None]
    zz = np.arange(SOURCES, dtype=np.float32)[None, None, :]
    return np.ascontiguousarray(yy * 3.0 + xx * 1.0 + zz * 5000.0)


def _montage_on(win) -> None:
    win._set_view_state(
        win.view_state.with_montage_axis(
            2, columns=COLUMNS, indices=tuple(range(SOURCES)), text=":"
        )
    )
    win.update_image_view()
    settle_window(win, lambda: frame_session_settled(win), label="montage on")


def _montage_off(win) -> None:
    win._set_view_state(
        win.view_state.with_montage_axis(None, columns=None, indices=None, text=None)
    )
    win.update_image_view()
    settle_window(win, lambda: frame_session_settled(win), label="montage off")


def _counters(win) -> dict[str, int]:
    session = win.renderer._frame_session
    stats = dict(win.img_view.presentation_diagnostics())
    return {
        "session_id": int(session.session_id),
        "presented": len(session.lifecycle.presented_tiles),
        "preview_tasks": int(getattr(win.renderer, "_montage_preview_reduced_scheduled", 0) or 0),
        "uploads": int(stats.get("wgpu_uploads_total", 0) or 0),
        "upload_bytes": int(stats.get("wgpu_upload_bytes_total", 0) or 0),
    }


def _cycle(qtbot):
    """Open the montage, leave for one slice, come back.  Returns the counters."""

    settings = use_wgpu_backend(extra_settings={"montage_quality_policy": "resident"})
    win = make_backend_window(qtbot, _montage_data(), backend="wgpu", require_gpu_atlas=True)
    try:
        win.resize(900, 700)
        win.show()
        _montage_on(win)
        first = _counters(win)
        _montage_off(win)
        away = _counters(win)
        _montage_on(win)
        back = _counters(win)
        # The cycle has to be a real rebirth, or the assertions below measure
        # nothing: an index-window retarget keeps the session object and its
        # whole presentation scope, which is a different (already working) path.
        assert away["session_id"] != first["session_id"]
        assert back["session_id"] != away["session_id"]
        assert first["presented"] == SOURCES
        assert back["presented"] == SOURCES
        return first, away, back
    finally:
        win.close()
        restore_default_backend(settings)


def test_a_montage_return_costs_no_gpu_upload(qtbot):
    """Coming back re-shows pixels the GPU never stopped holding.

    This is the load-bearing half of the field evidence: the 2.6 s the user
    waits buys *no* new bytes on the device.  Whatever a fix does about the
    redundant CPU production, it must not buy the return leg by re-uploading —
    that would trade a 2.6 s drip for a pool-pressure stall.
    """

    _first, away, back = _cycle(qtbot)

    assert back["uploads"] == away["uploads"]
    assert back["upload_bytes"] == away["upload_bytes"]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Session rebirth discards the LOD residency evidence and the floor payload "
        "wrappers, so the returning montage plans a producing FLOOR rung ('cold "
        "floor fill', presentation_only=False) for every tile whose native "
        "RenderedTile did not survive in the window-scoped display cache.  "
        "Measured at the returning session's first ladder plan: resident_levels=() "
        "and floor_available=False on 64 of 64 tiles."
    ),
)
def test_a_montage_return_re_presents_instead_of_re_producing(qtbot):
    """The property a fix has to establish, stated where it can be measured.

    A montage returning to an unchanged document, operation chain and index
    window must re-present what it already showed.  Re-producing it is forbidden
    work under R2 ("a tile that already meets the floor is skipped in that
    pass — never re-produced"), and the trace shows the re-production is not
    only forbidden but *inert*: it uploads nothing.

    The bound is deliberately loose (a handful of tiles, not zero).  The claim
    is "the return is not a refill", and a fix that leaves a few genuinely
    evicted tiles to re-produce still satisfies it; a fix that only shrinks the
    batch does not.
    """

    _first, away, back = _cycle(qtbot)

    assert back["preview_tasks"] - away["preview_tasks"] <= SOURCES // 8
