"""Frozen physical-residency truth, captured GUI-side and read anywhere.

ADR 0065 stage 2, decision 5. The crop-rebind seed asks the backend one
question — "are this binding's pages physically resident?" — through
``img_view.tiledPayloadResident``, and that probe walks the live wgpu page
table. Once the ladder retarget runs on the presentation owner (slice B) the
probe is a cross-thread read of a dict the GUI thread mutates, and
``tuple(dict)`` against a concurrent writer is a ``RuntimeError``, not a torn
read. So the GUI captures the answer set instead, at the one moment it is
authoritative: the tail of a commit's acknowledgement.

Two things make a *snapshot* of residency sound rather than merely convenient:

- ``bind`` only ever adds coverage. A key that was resident at capture is
  still resident unless ``unbind`` ran, and ``unbind`` is the single path that
  bumps ``PageTable.eviction_stamp`` (``gpu/page_table.py``). So an unchanged
  stamp proves nothing the snapshot trusted has been lost.
- Keys bound *since* the capture are missing from the set, so the predicate
  can only under-report. A declined rebind costs an evaluation; a wrongly
  granted one presents pages that are not there. The asymmetry is why this is
  conservative by construction and not by policy.

The mirror in ``TileLifecycle`` is deliberately not reused for this: it
answers "was this exact payload acknowledged", and a crop rebind mints a new
source id over pages that were never acknowledged under it. The two diverge
precisely on the case this exists for.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class PhysicalResidencyProbe:
    """A ``tiledPayloadResident`` predicate over a frozen key set.

    Built by the backend view (it owns how a payload becomes page keys) and
    evaluated by whoever holds the snapshot. Everything it reads is either
    immutable or captured; the one live read is
    :attr:`_live_stamp`, and that is guarded — see :meth:`valid`.
    """

    __slots__ = ("_binding", "_live_stamp", "eviction_stamp", "resident_keys")

    def __init__(
        self,
        *,
        resident_keys: frozenset,
        eviction_stamp: tuple[int, int],
        binding: Callable[[Any, frozenset], Any],
        live_stamp: Callable[[], tuple[int, int]],
    ) -> None:
        self.resident_keys = resident_keys
        self.eviction_stamp = tuple(eviction_stamp)
        #: ``(payload, resident_keys) -> binding | None``. Pure with respect to
        #: the view: it derives representation/mapping from the payload alone
        #: and takes the key set as an argument, so it touches no page table.
        self._binding = binding
        self._live_stamp = live_stamp

    def valid(self) -> bool:
        """Whether nothing has been evicted since the capture.

        The live read is ``PageTable.eviction_stamp``, a pair whose epoch half
        is fixed for the life of one table instance — so within an instance
        this is a single ``int`` load, and a torn read across instances can
        only *disagree*, which invalidates. There is no reading that turns a
        changed stamp into an unchanged one, which is the only direction that
        would be unsafe.
        """

        try:
            return tuple(self._live_stamp()) == self.eviction_stamp
        except Exception:
            # A torn-down surface has no stamp to compare; an unanswerable
            # question is not a "yes".
            return False

    def __call__(self, payload) -> bool:
        binding = self._binding(payload, self.resident_keys)
        if binding is None:
            return False
        keys = self.resident_keys
        return all(key in keys for key in binding.page_keys)


@dataclass(frozen=True)
class ResidencySnapshot:
    """What an off-GUI ladder retarget needs instead of live display reads.

    ``committed_frame`` is the second half and not an afterthought: the
    crop-rebind seed clones its predecessor wrappers out of
    ``win._committed_display_frame``, and that attribute and the page table
    have to agree about which commit they describe or the seed rebinds a
    wrapper whose pages it never verified.
    """

    committed_frame: Any = None
    probe: PhysicalResidencyProbe | None = None
    #: Identity of the session generation whose commit produced this. Purely
    #: diagnostic here; slice C's transaction take validates against it.
    session_identity: tuple[int, int] | None = None

    def physical_resident_fn(self) -> Callable[[Any], bool] | None:
        """The probe if it still describes the backend, else ``None``.

        ``None`` is the existing "no residency seam" branch
        (``frame_session._zero_byte_retarget_tiles``,
        ``rebind_resident_crop_tiles``): the logical remap set answers instead.
        That branch predates this file and already defines the degraded
        behaviour, which is why an invalidated snapshot needs no new staleness
        concept — decision "no new staleness concepts" holds.
        """

        probe = self.probe
        if probe is None or not probe.valid():
            return None
        return probe


def capture_residency_snapshot(win) -> ResidencySnapshot | None:
    """Freeze the backend's residency truth. GUI thread only.

    Called at the tail of a commit's acknowledgement, where the page table has
    just stopped changing and ``_committed_display_frame`` has just been
    republished. Backends with no page table (pyqtgraph) return ``None`` and
    their off-thread readers take the logical branch.
    """

    if win is None:
        return None
    img_view = getattr(win, "img_view", None)
    snapshot_fn = getattr(img_view, "tiledResidencySnapshot", None)
    probe = snapshot_fn() if callable(snapshot_fn) else None
    if probe is None:
        return None
    return ResidencySnapshot(
        committed_frame=getattr(win, "_committed_display_frame", None),
        probe=probe,
        session_identity=None,
    )


__all__ = [
    "PhysicalResidencyProbe",
    "ResidencySnapshot",
    "capture_residency_snapshot",
]
