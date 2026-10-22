"""Shared-memory attachment honors the publishing process's unlink ownership."""

from multiprocessing import shared_memory

import numpy as np
import pytest

from arrayscope.kernel.shm_slots import SlotPool


def test_slot_attacher_closes_without_unlinking_the_owning_pool():
    owner = SlotPool(slot_bytes=64, slots=1)
    name = owner.handle().name
    try:
        attached = SlotPool.attach(owner.handle())
        lease = attached.lease()
        descriptor = lease.publish(np.arange(8, dtype=np.int64))
        attached.close()
        actual = owner.take(descriptor)
        np.testing.assert_array_equal(actual, np.arange(8, dtype=np.int64))
        del actual
        owner.release(descriptor)
        probe = shared_memory.SharedMemory(name=name)
        probe.close()
    finally:
        owner.close()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)


@pytest.mark.parametrize("supports_track", [False, True])
def test_attach_uses_track_only_when_the_constructor_supports_it(monkeypatch, supports_track):
    from arrayscope.core.shared_memory import attach_shared_memory

    sentinel = object()
    calls = []

    def legacy(*, name, create=False, size=0):
        calls.append((name, create, size))
        return sentinel

    def modern(*, name, create=False, size=0, track=True):
        calls.append((name, create, size, track))
        return sentinel

    monkeypatch.setattr(shared_memory, "SharedMemory", modern if supports_track else legacy)
    assert attach_shared_memory("publisher-owned") is sentinel
    assert calls == [
        ("publisher-owned", False, 0, False) if supports_track else ("publisher-owned", False, 0)
    ]


def test_constructor_type_errors_are_not_treated_as_legacy_support(monkeypatch):
    from arrayscope.core.shared_memory import attach_shared_memory

    calls = []

    def broken(*, name, create=False, track=True):
        calls.append((name, create, track))
        raise TypeError("constructor failed after accepting track")

    monkeypatch.setattr(shared_memory, "SharedMemory", broken)
    with pytest.raises(TypeError, match="constructor failed after accepting track"):
        attach_shared_memory("publisher-owned")
    assert calls == [("publisher-owned", False, False)]


@pytest.mark.parametrize("legacy_signature", [False, True])
def test_spawned_attacher_exit_does_not_unregister_or_unlink_publishers(legacy_signature):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    environment = dict(os.environ)
    environment.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--ownership-probe",
            str(int(legacy_signature)),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stderr == "", "resource tracker or child cleanup reported an ownership error"
    report = json.loads(completed.stdout)
    assert report == {
        "child_values": [[0, 1, 2, 3, 4, 5, 6, 7]] * 3,
        "publisher_names_survived": True,
        "publisher_unlinked_names": True,
    }


def _child_owner_probe(
    handle, descriptor, source_leaf, stage_descriptor, connection, legacy_signature
):
    from arrayscope.operations import source_transport, stage_segments

    if legacy_signature:
        constructor = shared_memory.SharedMemory

        def legacy(name=None, create=False, size=0):
            return constructor(name=name, create=create, size=size)

        shared_memory.SharedMemory = legacy
    slots = SlotPool.attach(handle)
    try:
        slot_values = slots.take(descriptor)
        source_values = source_leaf.reconstruct()
        stage_values = stage_segments.attach_stage_array(stage_descriptor)
        connection.send([slot_values.tolist(), source_values.tolist(), stage_values.tolist()])
        del slot_values, source_values, stage_values
    finally:
        source_transport._release_published_segments()
        stage_segments.close_all_attachments()
        slots.close()
        connection.close()


def _ownership_probe(legacy_signature):
    import json
    import multiprocessing

    from arrayscope.core.shared_memory import attach_shared_memory
    from arrayscope.operations.regions import region_from_index_spec
    from arrayscope.operations.source_transport import SharedMemoryLeaf
    from arrayscope.operations.stage_segments import StageSegment

    context = multiprocessing.get_context("spawn")
    owner = SlotPool(slot_bytes=64, slots=1, context=context)
    source = shared_memory.SharedMemory(create=True, size=64)
    stage = shared_memory.SharedMemory(create=True, size=64)
    names = [owner.handle().name, source.name, stage.name]
    process = None
    parent_connection, child_connection = context.Pipe()
    values = np.arange(8, dtype=np.int64)
    try:
        descriptor = owner.lease().publish(values)
        np.ndarray(values.shape, values.dtype, buffer=source.buf)[:] = values
        np.ndarray(values.shape, values.dtype, buffer=stage.buf)[:] = values
        source_leaf = SharedMemoryLeaf(
            source.name, values.shape, values.dtype.str, "C", values.nbytes
        )
        stage_descriptor = StageSegment(
            stage.name,
            values.shape,
            values.dtype.str,
            "C",
            values.nbytes,
            1,
            values.shape,
            values.dtype.str,
            region_from_index_spec(values.shape, (slice(None),)),
            "highest",
            1.0,
            True,
            False,
        )
        process = context.Process(
            target=_child_owner_probe,
            args=(
                owner.handle(),
                descriptor,
                source_leaf,
                stage_descriptor,
                child_connection,
                legacy_signature,
            ),
        )
        process.start()
        child_connection.close()
        assert parent_connection.poll(10), "spawned child did not publish its attachment probe"
        child_values = parent_connection.recv()
        process.join(10)
        assert process.exitcode == 0
        for name in names:
            probe = attach_shared_memory(name)
            np.testing.assert_array_equal(
                np.ndarray(values.shape, values.dtype, buffer=probe.buf), values
            )
            probe.close()
        owner.release(descriptor)
        # A child has exited, but the owner can still allocate and publish.
        descriptor = owner.lease().publish(values + 1)
        actual = owner.take(descriptor)
        np.testing.assert_array_equal(actual, values + 1)
        del actual
        owner.release(descriptor)
    finally:
        parent_connection.close()
        child_connection.close()
        if process is not None and process.is_alive():
            process.terminate()
            process.join(10)
        owner.close()
        source.close()
        source.unlink()
        stage.close()
        stage.unlink()
    for name in names:
        try:
            probe = attach_shared_memory(name)
        except FileNotFoundError:
            continue
        probe.close()
        raise AssertionError("publisher failed to unlink its segment")
    print(
        json.dumps(
            {
                "child_values": child_values,
                "publisher_names_survived": True,
                "publisher_unlinked_names": True,
            }
        )
    )


if __name__ == "__main__":
    import sys

    if sys.argv[1] != "--ownership-probe":
        raise SystemExit("unknown shared-memory probe")
    _ownership_probe(bool(int(sys.argv[2])))
