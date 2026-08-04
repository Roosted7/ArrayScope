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

The two tests below hold the two halves apart, because a fix can satisfy either
one alone and be wrong:

* the return leg uploads nothing — the evidence that the CPU production was
  redundant, and the guard against "we fixed the drip by uploading everything
  again";
* the return leg does not re-produce what it is about to show.

The 72-of-272 asymmetry in the field is the same statement from the other side:
the tiles whose native ``RenderedTile`` survived in the window-scoped display
cache came back in ONE commit (71 of them at t=60.1); only the tiles whose
naming died with the session dripped.  ``render.lod.ResidentFloorIndex``
(2026-08-04) gives the reduced floors the same window-scoped survival the
native results already had.
"""

from __future__ import annotations

import numpy as np

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


def _cycle(qtbot, *, on_return=None):
    """Open the montage, leave for one slice, come back.  Returns the counters.

    ``on_return`` wraps only the return leg, so a probe sees the transition the
    user reported and not the cold first entry.
    """

    settings = use_wgpu_backend(extra_settings={"montage_quality_policy": "resident"})
    win = make_backend_window(qtbot, _montage_data(), backend="wgpu", require_gpu_atlas=True)
    try:
        win.resize(900, 700)
        win.show()
        _montage_on(win)
        first = _counters(win)
        _montage_off(win)
        away = _counters(win)
        if on_return is not None:
            on_return(win)
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


def test_a_montage_return_re_presents_instead_of_re_producing(qtbot):
    """The property the fix establishes, stated where it can be measured.

    A montage returning to an unchanged document, operation chain and index
    window must re-present what it already showed.  Re-producing it is forbidden
    work under R2 ("a tile that already meets the floor is skipped in that
    pass — never re-produced"), and the trace shows the re-production is not
    only forbidden but *inert*: it uploads nothing.

    This was ``xfail(strict=True)`` until 2026-08-04.  What it pinned:
    ``FrameSession`` rebirth empties the ``TileLifecycle`` whose
    ``record.levels`` was the only record of WHICH ``LodPageSetKey`` names the
    reduced pages a tile already owns — while the pages themselves live on in
    the window-scoped ``LodPageCache``.  ``best_floor_key`` therefore returned
    None, ``TileLodState.floor_available`` was False, and ``LodLadder.plan_tile``
    took its ``cold_floor`` branch: measured at the returning session's first
    ladder plan, ``resident_levels=()`` and ``floor_available=False`` on 64 of
    64 tiles, and 63 of 64 planned ``FLOOR ... presentation_only=False,
    reason="cold floor fill"``.  ``render.lod.ResidentFloorIndex`` is that
    missing name, kept beside the pages it names.

    The bound was deliberately loose (a handful of tiles, not zero) while the
    claim was only "the return is not a refill".  It is now 1, for the reason
    recorded at the assertion.
    """

    _first, away, back = _cycle(qtbot)

    # Measured 2026-08-04, live wgpu, both settled and mid-fill predecessors:
    # ZERO.  The bound was ``<= SOURCES // 8`` while the claim was only "the
    # return is not a refill"; the claim it can now carry is the stronger one
    # the user actually reported — nothing pops back in, because nothing is
    # re-made.  A tile whose pages the pyramid genuinely evicted still has to
    # re-produce, and that is what the ``+ 1`` leaves room for: eviction is a
    # budget event this test does not control, and pinning exactly 0 would make
    # an unrelated cache-size change look like this defect returning.
    assert back["preview_tasks"] - away["preview_tasks"] <= 1


def test_a_montage_return_is_one_render_entry(qtbot):
    """The return leg blocks the GUI once, not twice.

    The other half of the same transition, and it is not about tiles at all.
    Trace 7 records the return (t=148.1, ``reason=slice-range``) as a single
    689.4 ms ``render_entry`` of which ``viewport_plan_ms`` claimed 353.5 —
    against a settled baseline for that same label of 1.3 ms.  The plan had not
    become 270x slower.  The label was billing a second, complete
    ``update_image_view``:

    ``_maybe_auto_fit_montage_tiles`` moves the camera from inside the entry,
    Qt delivers ``sigRangeChanged`` synchronously, ``ViewportBridge`` answered a
    programmatic montage range change with an immediate
    ``retarget_montage_viewport()``, and that cannot take its viewport-only path
    because the live session is still keyed to the OUTGOING view state -- so it
    fell through to a whole nested render entry, inside the timed span, whose
    own phase attributes the outer entry then overwrote.  The duplicate was
    therefore invisible in the bill of materials and visible only as an
    inflated ``viewport_plan_ms``.

    Measured on a 144-tile montage return, 7 returns per arm, median: before,
    2 entries (one nested) and ``viewport_plan_ms`` 22.4 ms of a 23.1 ms
    callback; after, 1 entry and 1.9 ms of 20.3 ms.  The callback only drops
    12% because the nested entry was doing real work the outer one then had to
    do itself — what is removed is the DUPLICATION and the mis-attribution,
    not a third of a second of pure waste.

    The assertion is on the ENTRY COUNT, deliberately, and on neither duration.
    Re-entrancy is the defect; the milliseconds are its consequence, and a
    millisecond bound here would fail on a busy machine for reasons that have
    nothing to do with this.
    """

    entries: list[int] = []
    depth = {"value": 0}

    def instrument(win) -> None:
        renderer = win.renderer
        original = type(renderer).update_image_view

        def counted(self, **kwargs):
            depth["value"] += 1
            entries.append(depth["value"])
            try:
                return original(self, **kwargs)
            finally:
                depth["value"] -= 1

        # Bound to the instance: nothing here outlives this window.
        renderer.update_image_view = counted.__get__(renderer, type(renderer))

    _first, _away, _back = _cycle(qtbot, on_return=instrument)

    assert entries, "the return leg ran no render entry at all"
    assert not [d for d in entries if d > 1], (
        f"the montage return re-entered update_image_view (depths {entries}); "
        "a camera change made by a render entry started a second complete one"
    )
