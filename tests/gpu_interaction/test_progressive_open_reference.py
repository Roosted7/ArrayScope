"""Real-display progressive-open publication-correctness gate (ring 4).

Pins the residual of the "Progressive-load publication correctness" standing
lane (a50247e0): a real-Wayland visual open of a progressively-filled source
must, at completion, show ZERO unread (zero-fill) regions and the correct
final pixels/levels on the first-class backends.  ``ProgressiveArraySource``
(arrayscope/io/progressive.py) already publishes atomic detached region
snapshots and the OFFSCREEN gate is green (tests/io/test_progressive.py); by
testing law #1 a torn/unread-tile failure is a live-render failure and must be
pinned by the ring that can SEE it -- real GL/Qt raster, this ring.

Two oracles, deliberately layered:

* the physical-pixel oracle (arrayscope/tools/framebuffer_reference.py) proves
  the framebuffer/Qt-raster faithfully shows the *committed payloads*;
* a truth-anchored full-coverage gate proves every committed payload equals
  the KNOWN final data for its source frame -- i.e. no region is still
  zero-fill.

The layering is not redundant, and this test proves it (testing law #5 --
an oracle that has never failed on an injected fault is unproven):

  Fault injection.  With only PART of the source written (the tail frames
  still zero), the payload-based physical oracle passes VACUOUSLY -- for an
  unwritten tile the framebuffer shows zero and the committed payload is zero,
  so they agree -- while the truth-anchored coverage gate goes RED on exactly
  the unwritten tiles.  Completing the write turns the coverage gate green
  again.  This is why the coverage gate, not the pixel oracle alone, is the
  load-bearing acceptance for progressive publication.

Ring: tests/gpu_interaction only (real display, real GL / real Qt raster).
The offscreen sanity for the underlying source lives in
tests/io/test_progressive.py and is never acceptance for a rendering claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.gpu_interaction.conftest import COUNT, TILE, Harness


def gradient_montage_data() -> np.ndarray:
    """(64, 64, 36): frame ``k`` = ``10*k`` + a smooth in-tile gradient.

    Distinct per-frame offsets make an unread (zero-fill) tile obvious --
    every real frame has non-zero content, so a still-zero tile can only be an
    unpublished region, never legitimate data.
    """

    yy, xx = np.mgrid[0:TILE, 0:TILE].astype(np.float32)
    gradient = (yy + xx) * (8.0 / (2.0 * (TILE - 1)))
    frames = np.arange(COUNT, dtype=np.float32)[:, None, None] * 10.0 + gradient[None]
    return frames.transpose(1, 2, 0).copy()


def _make_progressive_window(app, backend_choice, final):
    """Open a production window over a ZERO-filled ProgressiveArraySource.

    Mirrors the async-open flow (arrayscope/app/open_flow.py): the streaming
    probe hands the viewer a ``LazySourceArray`` over a reader-owned zero
    destination *before the bytes arrive*, and the loader fills it in place.
    We drive that fill explicitly so the test is deterministic.
    """

    from pyqtgraph.Qt import QtCore

    from arrayscope.core.array_source import LazySourceArray
    from arrayscope.io.progressive import ProgressiveArraySource
    from arrayscope.window import ArrayScopeWindow

    settings = QtCore.QSettings()
    settings.clear()
    settings.setValue("image_rendering_backend", backend_choice.value)
    settings.setValue("first_run_hints_dismissed", True)
    settings.sync()

    # Reader-owned zero destination, exactly as load_*_progressive allocates it
    # (``data.fill(0)`` then ``ProgressiveArraySource(data)``).  budget=None so
    # every evaluation read goes through ``read_region`` -- never a cached full
    # materialization that would hide later writes.
    source = ProgressiveArraySource(np.zeros_like(final), label="progressive-open-test")
    lazy = LazySourceArray(source, materialize_budget_bytes=None)
    win = ArrayScopeWindow(lazy)
    win.show()
    harness = Harness(app, win)
    harness.pump(0.3)
    win._set_view_state(win.view_state.with_montage_axis(2, text=":"))
    win.render(reason="progressive-open-montage")
    assert harness.wait_settled(), (
        f"montage never settled over the empty source: {harness.settlement_diagnostics()}"
    )
    return harness, source


def _commit_write(harness, source, values, index_spec):
    """Publish a region into the source and refresh the viewer.

    Uses the same cache-invalidating refresh the async-open flow uses when
    loaded regions land (``open_flow._refresh_viewer_data`` ->
    ``notify_data_changed(force_autolevel=True)``), then settles the render on
    the real settle condition -- no fixed sleeps (testing law #4).
    """

    with source.write_transaction() as array:
        array[index_spec] = values
    harness.win.notify_data_changed(force_autolevel=True)
    harness.fit_plan_view()
    harness.pump(0.3)
    assert harness.wait_settled(), (
        f"scene never settled after publishing a region: {harness.settlement_diagnostics()}"
    )


def _assert_native_regime(harness) -> None:
    """Regime guard (strategy law 3): pin the native-resolution scalar regime
    so the exact payload-vs-truth equality below is meaningful (a reduced LOD
    would resample and defeat exact comparison)."""

    session = harness.session
    for number in session.required_tile_numbers():
        payload = session.display_tile_payloads[int(number)]
        level = 0 if payload.lod is None else int(payload.lod.level)
        assert level == 0, (
            f"tile {number} presented LOD level {level}; this gate pins the "
            "native-resolution regime so payload-vs-source equality is exact"
        )


def committed_coverage_failures(harness, final) -> list[tuple[int, int, bool]]:
    """Tiles whose committed payload does not equal the known final frame.

    The truth anchor: for every required tile, the committed semantic payload
    must equal ``final[:, :, source_index]`` exactly (native regime, so the
    read is a verbatim copy).  A still-zero payload is an unread/zero-fill
    region.  Returns ``(tile_number, source_index, payload_is_all_zero)`` for
    every failing tile.
    """

    session = harness.session
    required = sorted(int(number) for number in session.required_tile_numbers())
    if not required or len(required) != COUNT:
        raise AssertionError(
            f"progressive coverage gate needs the full montage: required={required}"
        )
    failures: list[tuple[int, int, bool]] = []
    for number in required:
        payload = session.display_tile_payloads[number]
        source_index = int(payload.source_index)
        semantic = np.asarray(payload.semantic_data)
        expected = final[:, :, source_index]
        if semantic.shape != expected.shape or not np.array_equal(semantic, expected):
            failures.append((number, source_index, not bool(np.any(semantic))))
    return failures


def assert_full_coverage(harness, final) -> None:
    """Every required tile shows its correct final frame -- zero unread."""

    failures = committed_coverage_failures(harness, final)
    if failures:
        raise AssertionError(
            "progressive full-coverage gate: "
            f"{len(failures)}/{COUNT} required tiles are unread/zero-fill or "
            f"diverge from the known final data: {failures[:8]}"
        )


BACKENDS = ("vispy", "pyqtgraph")


@pytest.fixture(params=BACKENDS)
def progressive_window(request, qt_app):
    """Production window over a progressive source, one per first-class backend.

    Skips (never fails) when the requested backend is unavailable in this Qt
    environment -- the same contract as tests/ui/helpers.make_backend_window.
    """

    from arrayscope.app.qt_binding import prefer_pyside6

    prefer_pyside6()
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore

    from arrayscope.app.launch import _prepare_qt_environment
    from arrayscope.app.settings_state import ImageRenderingBackendChoice
    from arrayscope.display.backend_contract import image_view_backend_capabilities

    backend = request.param
    choice = (
        ImageRenderingBackendChoice.VISPY
        if backend == "vispy"
        else ImageRenderingBackendChoice.PYQTGRAPH
    )

    _prepare_qt_environment()
    app = pg.mkQApp()
    previous_names = (str(app.organizationName()), str(app.applicationName()))
    app.setOrganizationName("ArrayScope")
    app.setApplicationName("ArrayScopeProgressiveOpenHarness")

    final = gradient_montage_data()
    harness = source = None
    try:
        harness, source = _make_progressive_window(app, choice, final)
        capabilities = image_view_backend_capabilities(harness.win.img_view)
        if capabilities.name != backend:
            pytest.skip(f"{backend} backend unavailable in this Qt environment")
        yield harness, source, final
    finally:
        settings = QtCore.QSettings()
        if harness is not None:
            harness.win.close()
            for _ in range(50):
                app.processEvents()
        settings.clear()
        settings.sync()
        app.setOrganizationName(previous_names[0])
        app.setApplicationName(previous_names[1])


def test_progressive_full_commit_shows_final_pixels(progressive_window):
    """Ring 4: after every region is published, the physical frame shows the
    correct final pixels over the full required tile set with zero unread."""

    harness, source, final = progressive_window

    # Open observed zeros (nothing published yet): a truthful mid-load frame,
    # and the baseline the completion must move away from.
    open_failures = committed_coverage_failures(harness, final)
    assert len(open_failures) == COUNT, (
        "empty-source open must present every tile as unread, else the "
        f"progressive fill below proves nothing: {open_failures[:4]}"
    )
    assert all(is_zero for *_, is_zero in open_failures), (
        f"empty-source open tiles must be genuinely zero-fill: {open_failures[:4]}"
    )

    # Publish the whole array (loader completion) and refresh.
    _commit_write(harness, source, final, (slice(None), slice(None), slice(None)))
    _assert_native_regime(harness)

    # Truth-anchored coverage: zero unread, correct final data everywhere.
    assert_full_coverage(harness, final)

    # Ring-4 physical-pixel oracle: the framebuffer / Qt raster faithfully
    # shows those committed payloads over the exact required tile set.
    report = harness.assert_tile_matches_cpu_reference()
    required = {int(number) for number in harness.session.required_tile_numbers()}
    assert {tile.tile_number for tile in report.tiles} == required
    assert len(report.tiles) == COUNT
    assert all(tile.samples >= report.min_samples_per_tile for tile in report.tiles), (
        "physical oracle sample floor not met -- comparison would be vacuous"
    )


def test_partial_fill_fails_coverage_gate_and_completion_recovers(progressive_window):
    """Fault injection (testing law #5): a torn frame (tail still zero) FAILS
    the full-coverage gate on exactly the unpublished tiles, while the
    payload-based physical oracle passes vacuously -- proving the coverage gate
    is the load-bearing acceptance.  Completing the write recovers it."""

    harness, source, final = progressive_window

    half = COUNT // 2
    # Publish only the leading half of the frames; the tail stays zero-fill.
    _commit_write(harness, source, final[:, :, :half], (slice(None), slice(None), slice(0, half)))
    _assert_native_regime(harness)

    failures = committed_coverage_failures(harness, final)
    failing_indices = sorted(source_index for _tile, source_index, _zero in failures)
    assert failing_indices == list(range(half, COUNT)), (
        f"torn frame must fail exactly the unpublished tail tiles: {failures}"
    )
    assert all(is_zero for *_, is_zero in failures), (
        "every failing tile must be genuinely unread (all-zero), not merely wrong"
    )
    with pytest.raises(AssertionError, match="unread/zero-fill or"):
        assert_full_coverage(harness, final)

    # The payload-based physical oracle CANNOT see this class: for a still-zero
    # tile the framebuffer shows zero and the committed payload is zero, so they
    # agree.  Its passing here is exactly why the truth-anchored gate exists.
    harness.assert_tile_matches_cpu_reference()

    # Completion: publish the remaining tail.  The coverage gate recovers,
    # proving the red above was the injected unread regions and nothing else.
    _commit_write(
        harness, source, final[:, :, half:], (slice(None), slice(None), slice(half, COUNT))
    )
    _assert_native_regime(harness)
    assert_full_coverage(harness, final)
    harness.assert_tile_matches_cpu_reference()
