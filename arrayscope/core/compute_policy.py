"""Lane-specific compute and FFT worker policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os

from arrayscope.app.settings_state import FFTWorkersChoice, normalize_fft_workers_choice
from arrayscope.operations import fft_backend


class ComputeLane(Enum):
    VISIBLE = "visible"
    STAGE = "stage"
    MONTAGE_TILE = "montage_tile"
    PREFETCH = "prefetch"
    PROFILE = "profile"
    ROI = "roi"
    PIXEL = "pixel"


@dataclass(frozen=True)
class ComputePolicy:
    visible_workers: int
    montage_tile_workers: int
    stage_workers: int
    prefetch_workers: int
    profile_workers: int
    roi_workers: int
    pixel_workers: int
    fft_workers_visible: int
    fft_workers_stage: int
    fft_workers_tile: int
    fft_workers_prefetch: int
    fft_workers_profile: int
    fft_workers_roi: int
    fft_workers_pixel: int

    def workers_for_lane(self, lane: ComputeLane) -> int:
        return {
            ComputeLane.VISIBLE: self.visible_workers,
            ComputeLane.STAGE: self.stage_workers,
            ComputeLane.MONTAGE_TILE: self.montage_tile_workers,
            ComputeLane.PREFETCH: self.prefetch_workers,
            ComputeLane.PROFILE: self.profile_workers,
            ComputeLane.ROI: self.roi_workers,
            ComputeLane.PIXEL: self.pixel_workers,
        }[ComputeLane(lane)]

    def fft_workers_for_lane(self, lane: ComputeLane) -> int:
        return {
            ComputeLane.VISIBLE: self.fft_workers_visible,
            ComputeLane.STAGE: self.fft_workers_stage,
            ComputeLane.MONTAGE_TILE: self.fft_workers_tile,
            ComputeLane.PREFETCH: self.fft_workers_prefetch,
            ComputeLane.PROFILE: self.fft_workers_profile,
            ComputeLane.ROI: self.fft_workers_roi,
            ComputeLane.PIXEL: self.fft_workers_pixel,
        }[ComputeLane(lane)]


@dataclass(frozen=True)
class EvaluationContext:
    lane: ComputeLane
    cancellation_token: object | None
    fft_workers: int
    memory_policy: object


def compute_policy_from_settings(settings, *, cpu_count: int | None = None) -> ComputePolicy:
    count = max(1, int(cpu_count if cpu_count is not None else (os.cpu_count() or 1)))
    choice = normalize_fft_workers_choice(getattr(settings, "fft_workers", FFTWorkersChoice.AUTO))
    resolved = int(fft_backend.resolve_fft_workers(choice.value, cpu_count=count))
    visible_fft = max(1, min(4, resolved))
    stage_fft = max(1, min(4, resolved))
    explicit_aggressive = choice == FFTWorkersChoice.ALL_MINUS_ONE
    tile_workers = 2
    tile_fft = 1
    product_limit = max(2, count // 2)
    if not explicit_aggressive:
        while tile_workers * tile_fft > product_limit and tile_workers > 1:
            tile_workers -= 1
    return ComputePolicy(
        visible_workers=1,
        montage_tile_workers=max(1, tile_workers),
        stage_workers=1,
        prefetch_workers=1,
        profile_workers=1,
        roi_workers=1,
        pixel_workers=1,
        fft_workers_visible=visible_fft,
        fft_workers_stage=stage_fft,
        fft_workers_tile=tile_fft,
        fft_workers_prefetch=1,
        fft_workers_profile=1,
        fft_workers_roi=1,
        fft_workers_pixel=1,
    )
