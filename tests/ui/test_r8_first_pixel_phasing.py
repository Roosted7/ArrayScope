from __future__ import annotations

from collections import Counter

import numpy as np

from tests.ui.helpers import clear_arrayscope_settings


def _event_index(events, name: str) -> int:
    return next(index for index, event in enumerate(events) if event[0] == name)


def test_vispy_complex_first_pass_levels_precede_physical_draw_and_refinement(qtbot, monkeypatch):
    """R8B.2: rough payload evidence, draw, histogram, and refinement are phased.

    This drives the production preview + target ladder with deterministic
    complex sources whose magnitudes are nowhere near the widget fallback
    ``(0, 1)``.  The instrumentation records only semantic phase boundaries;
    it does not change scheduling, payloads, batching, or backend results.
    """

    clear_arrayscope_settings()

    from pyqtgraph.Qt import QtCore

    from arrayscope.display.backends.vispy.tiles import GpuWindowedTileVisual
    from arrayscope.display.model.montage_levels import LevelEvidenceQuality
    from arrayscope.display.shader_mapping import TexturePlaneKind
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.kernel import Kernel
    from arrayscope.render.level_stats import LevelStatsService
    from arrayscope.window import ArrayScopeWindow
    from arrayscope.window.frame_effects import FramePipelineEffects

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", "vispy")
    settings.setValue("montage_quality_policy", "resident")
    settings.sync()

    events: list[tuple] = []
    evidence_attempts = Counter()
    box = {}

    original_prepared = LevelStatsService._update_montage_level_bounds_from_prepared
    original_rendered = LevelStatsService._update_montage_level_bounds_from_rendered
    original_set_levels = GpuWindowedTileVisual.set_levels
    original_present = VisPyImageView2D.setTiledPresentation
    original_submit_speculative = Kernel.submit_speculative_batch
    original_admit_reduced = FramePipelineEffects._admit_reduced_display_payload
    original_admit_target = FramePipelineEffects._admit_evaluation_result

    def record_merge(service, level_key, source_index: int, before_sources):
        summary = service._montage_level_tracker().summary_for(level_key)
        after_sources = frozenset() if summary is None else frozenset(summary.source_indices)
        if int(source_index) not in after_sources or after_sources == before_sources:
            return
        quality = LevelEvidenceQuality(int(getattr(summary, "evidence_quality", 0) or 0))
        phase = "rough sample merged" if quality <= LevelEvidenceQuality.ROUGH_TARGET else "refined sample merged"
        events.append((phase, int(source_index), int(quality), tuple(getattr(summary, "bounds", ()) or ())))

    def update_prepared(service, level_key, rendered, **kwargs):
        summary = service._montage_level_tracker().summary_for(level_key)
        before = frozenset() if summary is None else frozenset(summary.source_indices)
        result = original_prepared(service, level_key, rendered, **kwargs)
        quality = kwargs.get("evidence_quality", LevelEvidenceQuality.ROUGH_PREVIEW)
        evidence_attempts[int(quality)] += 1
        record_merge(
            service,
            level_key,
            int(rendered.tile.source_index),
            before,
        )
        return result

    def update_rendered(service, level_key, rendered, **kwargs):
        summary = service._montage_level_tracker().summary_for(level_key)
        before = frozenset() if summary is None else frozenset(summary.source_indices)
        result = original_rendered(service, level_key, rendered, **kwargs)
        requested = kwargs.get("evidence_quality")
        if requested is None:
            requested = (
                LevelEvidenceQuality.REFINED
                if bool(kwargs.get("refined", False))
                else LevelEvidenceQuality.ROUGH_TARGET
            )
        evidence_attempts[int(requested)] += 1
        record_merge(
            service,
            level_key,
            int(rendered.tile.source_index),
            before,
        )
        return result

    def set_levels(visual, levels):
        normalized = tuple(float(value) for value in levels)
        changed = original_set_levels(visual, levels)
        if changed:
            events.append(("shader levels applied", normalized))
        return changed

    def present(view, **kwargs):
        delta = kwargs["tile_delta"]
        state = kwargs["tile_state"]
        active = tuple(int(tile) for tile in tuple(delta.active_tiles))
        payloads = state.active_payloads(delta)
        complex_upserts = {
            int(tile): payload
            for tile, payload in dict(delta.upserts or {}).items()
            if getattr(payload, "texture_kind", None) == TexturePlaneKind.COMPLEX_RG32F
        }
        qualities = {
            str(getattr(payload, "quality", "exact") or "exact")
            for payload in payloads.values()
        }
        complete_preview_payload = bool(
            active
            and set(active) <= set(payloads)
            and qualities == {"preview"}
        )
        complete_target_payload = bool(
            active
            and set(active) <= set(payloads)
            and qualities == {"exact"}
        )
        if kwargs.get("histogramPlotData") is not None:
            session = getattr(getattr(box.get("win"), "renderer", None), "_frame_session", None)
            summary = None
            if session is not None:
                summary = box["win"].renderer._montage_level_tracker().summary_for(session.level_key)
            quality = int(getattr(summary, "evidence_quality", 0) or 0)
            name = (
                "refined levels/histogram publication"
                if quality >= int(LevelEvidenceQuality.REFINED)
                else "rough histogram publication"
            )
            events.append((name, quality))
        report = original_present(view, **kwargs)
        if complex_upserts and getattr(report, "committed_upserts", None):
            for tile_number in tuple(report.committed_upserts):
                payload = complex_upserts[int(tile_number)]
                assert payload.tile_identity is not None
                assert payload.tile_identity.complex_mapping is not None
                assert payload.presentation_identity is not None
                assert payload.presentation_identity.levels_generation > 0
                assert payload.presentation_identity.levels == tuple(
                    float(value) for value in kwargs["levels"]
                )
            events.append(
                (
                    "backend physical draw acknowledgement",
                    tuple(sorted(int(tile) for tile in report.committed_upserts)),
                    tuple(float(value) for value in kwargs["levels"]),
                )
            )
        if (
            complete_preview_payload
            and set(active) <= set(getattr(report, "presented_tiles", ()))
            and not any(event[0] == "rough pass complete" for event in events)
        ):
            events.append(("rough pass complete", tuple(sorted(active))))
        if (
            complete_target_payload
            and set(active) <= set(getattr(report, "presented_tiles", ()))
            and not any(event[0] == "target pass complete" for event in events)
        ):
            events.append(("target pass complete", tuple(sorted(active))))
        return report

    def submit_speculative(kernel, **kwargs):
        if kwargs.get("kind") in {"semantic-level-evidence", "montage-refined-level-stats"}:
            session = getattr(getattr(box.get("win"), "renderer", None), "_frame_session", None)
            settled = bool(session is not None and session.required_target_settled())
            events.append(("refined evidence start", kwargs.get("kind"), settled))
        return original_submit_speculative(kernel, **kwargs)

    def admit_reduced(effects, step, tile_number, payload, *, quality=None):
        resolved = str(quality or ("exact" if step is not None and int(step.rung) == 2 else "preview"))
        if resolved != "preview" and not any(event[0] == "target pass start" for event in events):
            events.append(("target pass start", "reduced"))
        return original_admit_reduced(effects, step, tile_number, payload, quality=quality)

    def admit_target(effects, tile, result):
        if not any(event[0] == "target pass start" for event in events):
            events.append(("target pass start", "native"))
        return original_admit_target(effects, tile, result)

    monkeypatch.setattr(LevelStatsService, "_update_montage_level_bounds_from_prepared", update_prepared)
    monkeypatch.setattr(LevelStatsService, "_update_montage_level_bounds_from_rendered", update_rendered)
    monkeypatch.setattr(GpuWindowedTileVisual, "set_levels", set_levels)
    monkeypatch.setattr(VisPyImageView2D, "setTiledPresentation", present)
    monkeypatch.setattr(Kernel, "submit_speculative_batch", submit_speculative)
    monkeypatch.setattr(FramePipelineEffects, "_admit_reduced_display_payload", admit_reduced)
    monkeypatch.setattr(FramePipelineEffects, "_admit_evaluation_result", admit_target)

    yy, xx = np.mgrid[:96, :160]
    data = np.empty((96, 160, 6), dtype=np.complex64)
    for source_index in range(data.shape[2]):
        magnitude = 250.0 + source_index * 300.0 + 0.7 * xx + 0.3 * yy
        phase = 0.2 * source_index + xx / 37.0 - yy / 53.0
        data[..., source_index] = magnitude * np.exp(1j * phase)

    win = ArrayScopeWindow(data)
    box["win"] = win
    win.resize(760, 520)
    win.show()
    qtbot.addWidget(win)
    try:
        state = (
            win.view_state.with_channel("complex")
            .with_montage_axis(2, columns=3, indices=tuple(range(6)), text=":")
        )
        win._set_view_state(state)
        win.render(reason="test-r8-first-pixel-phasing")
        qtbot.waitUntil(
            lambda: (
                win.renderer._frame_session is not None
                and win.renderer._frame_session.required_target_settled()
                and not win.renderer._frame_session.pending_refined_level_tiles
                and win.renderer._montage_level_tracker()
                .summary_for(win.renderer._frame_session.level_key)
                .refined
            ),
            timeout=20_000,
        )
        qtbot.wait(250)

        names = [event[0] for event in events]
        assert "rough pass complete" in names, events
        assert "target pass start" in names, events
        assert "target pass complete" in names, events
        assert "refined evidence start" in names, events
        assert "refined levels/histogram publication" in names, "\n".join(
            (
                *(map(str, events)),
                f"decision={win.renderer._last_montage_level_decision!r}",
                f"flush={win.renderer._frame_session.flush_pending!r}",
                f"final={win.renderer._frame_session.final_commit_pending!r}",
            )
        )

        first_draw = _event_index(events, "backend physical draw acknowledgement")
        assert _event_index(events, "rough sample merged") < first_draw
        assert _event_index(events, "shader levels applied") < first_draw
        assert events[first_draw][2] != (0.0, 1.0)
        assert all(
            event[2] != (0.0, 1.0)
            for event in events
            if event[0] == "backend physical draw acknowledgement"
        )

        rough_complete = _event_index(events, "rough pass complete")
        rough_histogram = _event_index(events, "rough histogram publication")
        target_start = _event_index(events, "target pass start")
        target_complete = _event_index(events, "target pass complete")
        refined_start = _event_index(events, "refined evidence start")
        refined_publication = _event_index(events, "refined levels/histogram publication")
        assert rough_complete < rough_histogram < target_start
        assert target_complete < refined_start < refined_publication

        rough_level_updates = {
            event[1]
            for index, event in enumerate(events)
            if event[0] == "shader levels applied" and index < rough_complete
        }
        assert rough_level_updates, events
        assert sum(
            1 for event in events if event[0] == "rough sample merged"
        ) >= 2

        assert all(event[2] is True for event in events if event[0] == "refined evidence start")
    finally:
        win.close()
        settings.setValue("image_rendering_backend", "pyqtgraph")
        settings.setValue("montage_quality_policy", "resident")
        settings.sync()
