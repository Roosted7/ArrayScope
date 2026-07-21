"""Progressive/streaming reader behaviour: equality, progress, cancel."""

import threading

import numpy as np
import pytest

from arrayscope.io.file_interpreters import load_path
from arrayscope.io.progressive import (
    LoadCancelled,
    load_cfl_progressive,
    load_npy_progressive,
)


def _fractions(events):
    return [e.fraction for e in events if e.stage == "reading" and e.fraction is not None]


# --- .npy ------------------------------------------------------------------


@pytest.mark.parametrize("order", ["C", "F"])
def test_npy_progressive_matches_np_load(tmp_path, order):
    arr = np.random.default_rng(0).normal(size=(13, 7, 5)).astype(np.float32)
    arr = np.asfortranarray(arr) if order == "F" else np.ascontiguousarray(arr)
    path = tmp_path / "data.npy"
    np.save(path, arr)

    events = []
    loaded = load_npy_progressive(path, progress=events.append)

    np.testing.assert_array_equal(loaded, np.load(path))
    assert loaded.flags["F_CONTIGUOUS" if order == "F" else "C_CONTIGUOUS"]
    fractions = _fractions(events)
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0
    assert events[-1].stage == "finalizing"


def test_npy_progressive_publishes_detached_region_reads(tmp_path):
    arr = np.arange(24, dtype=np.int64).reshape(4, 6)
    path = tmp_path / "data.npy"
    np.save(path, arr)

    probes = []
    initial = []

    def capture_probe(probe):
        probes.append(probe)
        initial.append(probe.data.read_region((slice(None), slice(None))))

    loaded = load_npy_progressive(path, on_streaming_probe=capture_probe)

    assert len(probes) == 1
    probe = probes[0]
    assert probe.data.shape == (4, 6)
    assert probe.metadata["detected_format"] == "numpy"
    assert not np.any(initial[0])
    assert not np.shares_memory(initial[0], loaded)
    np.testing.assert_array_equal(np.asarray(probe.data), arr)


def test_progressive_source_read_cannot_observe_an_inflight_write():
    from arrayscope.io.progressive import ProgressiveArraySource

    source = ProgressiveArraySource(np.zeros(4, dtype=np.int32))
    reader_started = threading.Event()
    reader_finished = threading.Event()
    observed = []

    def read():
        reader_started.set()
        observed.append(source.read_region((slice(None),)))
        reader_finished.set()

    with source.write_transaction() as destination:
        destination[:2] = 1
        thread = threading.Thread(target=read)
        thread.start()
        assert reader_started.wait(2.0)
        assert not reader_finished.is_set()
        destination[2:] = 1

    assert reader_finished.wait(2.0)
    thread.join()
    np.testing.assert_array_equal(observed[0], np.ones(4, dtype=np.int32))


def test_progressive_source_never_silently_drops_a_noncontiguous_write():
    """A non-contiguous destination must be visible-or-refused, never a silent no-op.

    ``write_flat``/``write_bytes`` mutate through ``ravel(order="K")``, which
    returns a *copy* for a non-contiguous array -- so on unfixed code the write
    lands in a throwaway buffer and ``read_region`` still returns all zeros
    (silent data loss). The contract is: either the source refuses the backing
    array at construction, or the write is genuinely visible afterwards. This
    test fails on the pre-fix code (the ValueError is not raised and the write
    is lost) and passes once the gap is closed.
    """
    from arrayscope.io.progressive import ProgressiveArraySource

    backing = np.zeros((4, 4))[:, ::2]  # non-contiguous view (a strided slice)
    assert not backing.flags.forc

    try:
        source = ProgressiveArraySource(backing)
    except ValueError:
        return  # refused loudly at construction -- acceptable, no silent loss

    # If construction was allowed, the write MUST be visible (not a silent no-op).
    source.write_flat(0, np.arange(8))
    result = source.read_region((slice(None), slice(None)))
    assert np.any(result), "non-contiguous write was silently discarded"


def test_npy_progressive_rejects_truncated_file(tmp_path):
    arr = np.zeros((64, 64), dtype=np.float64)
    path = tmp_path / "data.npy"
    np.save(path, arr)
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])

    with pytest.raises(ValueError, match="truncated"):
        load_npy_progressive(path)


def test_npy_progressive_cancel_raises(tmp_path):
    arr = np.zeros((32, 32), dtype=np.float32)
    path = tmp_path / "data.npy"
    np.save(path, arr)

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(LoadCancelled):
        load_npy_progressive(path, cancel=cancel)


# --- .cfl ------------------------------------------------------------------


def _write_cfl(tmp_path, arr):
    path = tmp_path / "kspace.cfl"
    with open(tmp_path / "kspace.hdr", "w") as header:
        header.write("# Dimensions\n" + " ".join(str(s) for s in arr.shape) + "\n")
    np.asfortranarray(arr).T.reshape(-1).tofile(path)
    return path


def test_cfl_progressive_matches_legacy_loader(tmp_path):
    rng = np.random.default_rng(1)
    arr = (rng.normal(size=(6, 5, 3, 1)) + 1j * rng.normal(size=(6, 5, 3, 1))).astype(np.complex64)
    path = _write_cfl(tmp_path, arr)

    events = []
    probes = []
    loaded = load_cfl_progressive(path, progress=events.append, on_streaming_probe=probes.append)

    np.testing.assert_array_equal(loaded, arr.reshape(6, 5, 3))
    assert probes[0].data.shape == (6, 5, 3)
    np.testing.assert_array_equal(np.asarray(probes[0].data), loaded)
    fractions = _fractions(events)
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0


def test_load_path_cfl_reports_progress(tmp_path):
    arr = np.ones((4, 4), dtype=np.complex64)
    path = _write_cfl(tmp_path, arr)

    events = []
    loaded = load_path(path, progress=events.append)

    np.testing.assert_array_equal(loaded.data, arr)
    assert loaded.metadata["detected_format"] == "cfl"
    assert _fractions(events)[-1] == 1.0


# --- Philips .rec ----------------------------------------------------------


def _write_rec_pair(tmp_path, *, size=4, n_slices=2):
    """Minimal magnitude-only XML/REC pair with identity rescaling."""
    image_infos = [
        f"""
    <Image_Info>
      <Key>
        <Attribute Name="Slice" Type="Int32">{slice_index}</Attribute>
        <Attribute Name="Echo" Type="Int32">1</Attribute>
        <Attribute Name="Grad Orient" Type="Int32">1</Attribute>
        <Attribute Name="BValue" Type="Int32">1</Attribute>
        <Attribute Name="Phase" Type="Int32">1</Attribute>
        <Attribute Name="Dynamic" Type="Int32">1</Attribute>
      </Key>
      <Attribute Name="Resolution X" Type="Int32">{size}</Attribute>
      <Attribute Name="Resolution Y" Type="Int32">{size}</Attribute>
      <Attribute Name="Pixel Size" Type="Int32">16</Attribute>
      <Attribute Name="Rescale Intercept" Type="Float">0.0</Attribute>
      <Attribute Name="Rescale Slope" Type="Float">1.0</Attribute>
      <Attribute Name="Scale Slope" Type="Float">1.0</Attribute>
      <Attribute Name="Type" Type="String">M</Attribute>
    </Image_Info>"""
        for slice_index in range(1, n_slices + 1)
    ]
    xml_text = f"""<PRIDE_V5>
  <Series_Info>
    <Attribute Name="Max No Slices" Type="Int32">{n_slices}</Attribute>
    <Attribute Name="Max No Echoes" Type="Int32">1</Attribute>
    <Attribute Name="Max No Gradient Orients" Type="Int32">1</Attribute>
    <Attribute Name="Max No B Values" Type="Int32">1</Attribute>
    <Attribute Name="Max No Phases" Type="Int32">1</Attribute>
    <Attribute Name="Max No Dynamics" Type="Int32">1</Attribute>
  </Series_Info>
  {"".join(image_infos)}
</PRIDE_V5>
"""
    (tmp_path / "scan.xml").write_text(xml_text)
    slices = [
        np.arange(size * size, dtype=np.uint16) + 100 * slice_index
        for slice_index in range(n_slices)
    ]
    (tmp_path / "scan.rec").write_bytes(b"".join(s.tobytes() for s in slices))
    expected = np.zeros((size, size, n_slices), dtype=np.float32)
    for slice_index, raw in enumerate(slices):
        expected[:, :, slice_index] = raw.reshape(size, size).astype(np.float32)
    return tmp_path / "scan.rec", expected


def test_rec_load_path_streams_and_reports_slice_progress(tmp_path):
    path, expected = _write_rec_pair(tmp_path, size=4, n_slices=3)

    events = []
    probes = []
    loaded = load_path(path, progress=events.append, on_streaming_probe=probes.append)

    np.testing.assert_array_equal(loaded.data, expected)
    assert loaded.metadata["detected_format"] == "rec"
    assert len(probes) == 1
    assert probes[0].data.shape == expected.shape
    np.testing.assert_array_equal(np.asarray(probes[0].data), loaded.data)
    assert probes[0].axes is not None
    fractions = _fractions(events)
    assert fractions == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_rec_cancel_mid_load_leaves_partial_buffer(tmp_path):
    path, expected = _write_rec_pair(tmp_path, size=4, n_slices=3)

    cancel = threading.Event()
    probes = []
    events = []

    def cancel_after_first_slice(event):
        events.append(event)
        if event.stage == "reading" and event.fraction >= 1 / 3:
            cancel.set()

    with pytest.raises(LoadCancelled):
        load_path(
            path, progress=cancel_after_first_slice, cancel=cancel, on_streaming_probe=probes.append
        )

    partial = np.asarray(probes[0].data)
    np.testing.assert_array_equal(partial[:, :, 0], expected[:, :, 0])
    assert np.all(partial[:, :, 2] == 0)
