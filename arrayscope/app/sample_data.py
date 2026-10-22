"""Small deterministic examples for exploring ArrayScope without a data file.

Only metadata is created at import. Arrays are generated on demand and remain
under 4 MiB, so trying an example needs no download, cache, or worker lifetime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arrayscope.core.axis_info import AxisInfo


@dataclass(frozen=True)
class SampleSpec:
    id: str
    title: str
    description: str
    hint: str


SAMPLES = (
    SampleSpec(
        "volume",
        "Explore a volume",
        "An asymmetric 3D phantom with fine spatial detail.",
        "Move the Slice control, then draw a rectangle to inspect values.",
    ),
    SampleSpec(
        "time-series",
        "Follow a time series",
        "A moving signal across 24 time points.",
        "Move Time, or enter : in its selection to see a montage.",
    ),
    SampleSpec(
        "complex",
        "Inspect a complex field",
        "Magnitude and phase in a synthetic wave field.",
        "Switch between Magnitude and Phase, then try a centered FFT operation.",
    ),
)


def make_sample(sample_id: str) -> tuple[np.ndarray, tuple[AxisInfo, ...]]:
    """Return a fresh example with meaningful dimension labels and units."""
    if sample_id not in {sample.id for sample in SAMPLES}:
        raise ValueError(f"Unknown example: {sample_id}")
    y, x = np.meshgrid(
        np.linspace(-1, 1, 128, dtype=np.float32),
        np.linspace(-1, 1, 128, dtype=np.float32),
        indexing="ij",
    )
    if sample_id == "volume":
        z = np.linspace(-1, 1, 24, dtype=np.float32)
        radius = (x[..., None] / 0.8) ** 2 + (y[..., None] / 0.65) ** 2 + (z / 0.9) ** 2
        main = np.exp(-3 * radius)
        feature = np.exp(
            -35 * ((x[..., None] - 0.3) ** 2 + (y[..., None] + 0.2) ** 2 + (z - 0.1) ** 2)
        )
        detail = 0.12 * np.cos(28 * x[..., None]) * np.exp(-8 * radius)
        data = main + 0.8 * feature + detail
        third = AxisInfo("axis-2", "Slice", 24, unit="mm", spacing=2, source_index=2)
    elif sample_id == "time-series":
        time = np.linspace(0, 2 * np.pi, 24, endpoint=False, dtype=np.float32)
        cx = 0.4 * np.cos(time)
        cy = 0.35 * np.sin(time)
        data = np.exp(-22 * ((x[..., None] - cx) ** 2 + (y[..., None] - cy) ** 2))
        data += (
            0.15 * (1 + np.sin(9 * x[..., None] + time)) * np.exp(-2 * (x * x + y * y))[..., None]
        )
        third = AxisInfo("axis-2", "Time", 24, unit="s", spacing=0.1, source_index=2)
    else:
        phase = 8 * np.pi * x + 4 * np.pi * y + 2 * np.arctan2(y, x)
        envelope = np.exp(-3 * (x * x + y * y)) + 0.35 * np.exp(
            -30 * ((x - 0.45) ** 2 + (y + 0.2) ** 2)
        )
        data = envelope * np.exp(1j * phase)
        third = None
    data = np.asarray(data, dtype=np.complex64 if sample_id == "complex" else np.float32)
    axes = (
        AxisInfo("axis-0", "Y", 128, unit="mm", spacing=0.5, source_index=0),
        AxisInfo("axis-1", "X", 128, unit="mm", spacing=0.5, source_index=1),
    )
    return data, axes + ((third,) if third is not None else ())
