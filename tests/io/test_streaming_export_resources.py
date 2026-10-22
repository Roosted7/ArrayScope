"""Linux process limits prove streaming needs less address space than its output.

This is an address-space/allocation oracle, not a hard resident-memory claim.
The subprocess reports RSS separately and never maps or loads its whole output.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires Linux RLIMIT_AS")
def test_streaming_export_succeeds_when_a_full_output_allocation_actually_fails(tmp_path):
    resource = pytest.importorskip("resource", reason="address-space limits are unavailable")
    if not hasattr(resource, "RLIMIT_AS"):
        pytest.skip("this platform cannot limit process address space")
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("PYTEST_") and name != "TESTMON_DATAFILE"
    }
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    completed = subprocess.run(
        [
            sys.executable,
            "-X",
            "faulthandler",
            str(Path(__file__)),
            "--constrained-probe",
            str(tmp_path),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    if "resource_limit_unavailable" in report:
        pytest.skip(report["resource_limit_unavailable"])
    assert report["full_allocation_failed"] is True
    assert report["output_bytes"] == 256 * 1024**2
    assert report["output_bytes"] > report["address_space_headroom_bytes"]
    assert 64 * 1024**2 <= report["address_space_headroom_bytes"] <= 96 * 1024**2
    assert report["streamed_output_shape"] == [32768, 2048]
    assert report["logical_source_region_bytes"] == report["output_bytes"]
    assert report["largest_source_region_bytes"] <= 4 * 1024**2
    assert report["verified_bytes"] == report["output_bytes"]
    assert report["actual_sha256"] == report["independent_sha256"]
    assert report["cancelled_after_payload_write"] is True
    assert report["cancelled_source_reads"] >= 1
    assert 0 < report["cancelled_source_region_bytes"] <= 4 * 1024**2
    assert report["previous_output_preserved"] is True
    assert report["temporary_files_removed"] is True


def _constrained_probe(directory):
    """Also callable directly for a standalone JSON resource-evidence artifact."""
    import errno
    import hashlib
    import resource
    from types import SimpleNamespace

    import h5py
    import numpy as np
    import psutil

    # Direct script execution has tests/io on sys.path; use this checkout's
    # ordinary package imports, with no duplicate-module loading machinery.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from arrayscope.io.hdf5_source import HDF5ArraySource, open_hdf5_source
    from arrayscope.io.region_export import RegionExportRequest
    from arrayscope.io.streaming_export import export_streaming_region, plan_streaming_export
    from arrayscope.operations.cancellation import EvaluationCancelled
    from arrayscope.operations.pipeline import ArrayDocument, ReverseAxis, Scale

    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / "sparse-source.h5"
    output = directory / "streamed.npy"
    cancelled_output = directory / "cancelled.npy"
    shape = (32768, 2048)
    dtype = np.dtype(np.float32)
    output_bytes = shape[0] * shape[1] * dtype.itemsize
    changed_start, changed_rows = 16384, 8
    changed = np.arange(changed_rows * shape[1], dtype=dtype).reshape(changed_rows, shape[1])
    changed += 100
    with h5py.File(source_path, "w") as handle:
        dataset = handle.create_dataset(
            "signal", shape=shape, dtype=dtype, chunks=(changed_rows, shape[1]), fillvalue=3.25
        )
        dataset[changed_start : changed_start + changed_rows] = changed
    del changed, dataset
    assert source_path.stat().st_size < 2 * 1024**2
    source = open_hdf5_source(source_path, "signal")
    document = ArrayDocument(source, operations=(ReverseAxis(0), Scale(2)))
    chunk_target = 4 * 1024**2
    request = RegionExportRequest(
        document, output, tuple((0, size) for size in shape), squeeze=False
    )

    # Import and warm every exercised owner before imposing the process limit.
    # A new library mapping cannot be mistaken for the export's working data.
    warm_output = directory / "warm.npy"
    warm_request = RegionExportRequest(
        document, warm_output, ((0, 2), (0, shape[1])), squeeze=False
    )
    warm_plan = plan_streaming_export(warm_request, chunk_target_bytes=chunk_target)
    export_streaming_region(warm_request, prepared=warm_plan)
    np.testing.assert_array_equal(np.arange(4), np.arange(4))

    def read_header(handle):
        readers = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
        }
        return readers[np.lib.format.read_magic(handle)](handle)

    with warm_output.open("rb") as handle:
        read_header(handle)
        np.fromfile(handle, dtype=dtype, count=shape[1])
    warm_output.unlink()
    del warm_request, warm_plan
    process = psutil.Process()
    baseline_vms = process.memory_info().vms
    baseline_rss = process.memory_info().rss
    baseline_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    headroom = 96 * 1024**2
    _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
    limit = baseline_vms + headroom
    if hard_limit != resource.RLIM_INFINITY:
        limit = min(limit, hard_limit)
    assert limit - baseline_vms >= 64 * 1024**2, "inherited limit leaves too little test headroom"
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard_limit))
    except OSError as error:
        if error.errno not in {errno.EPERM, errno.EACCES, errno.ENOSYS}:
            raise
        source.close()
        print(json.dumps({"resource_limit_unavailable": f"RLIMIT_AS setup was denied: {error}"}))
        return

    failed = False
    try:
        whole_output = np.empty(shape, dtype=dtype)
    except MemoryError:
        failed = True
    else:
        del whole_output
    assert failed, "control unexpectedly allocated the whole 256 MiB output under the limit"

    reads = []
    original_read = HDF5ArraySource.read_region

    def counted_read(self, index_spec, *, cancellation_token=None):
        array = original_read(self, index_spec, cancellation_token=cancellation_token)
        reads.append(int(array.nbytes))
        return array

    HDF5ArraySource.read_region = counted_read
    try:
        plan = plan_streaming_export(request, chunk_target_bytes=chunk_target)
        assert plan.output_bytes == output_bytes
        result = export_streaming_region(request, prepared=plan)
        assert result.shape == shape
        assert result.dtype == dtype
        source_read_bytes = sum(reads)
        largest_read = max(reads)
        assert source_read_bytes == output_bytes
        assert largest_read <= chunk_target

        # Read the standard NPY stream in small owned arrays. Mapping the full
        # file would itself exceed RLIMIT_AS and invalidate the resource test.
        actual_hash, expected_hash = hashlib.sha256(), hashlib.sha256()
        verified_bytes = 0
        rows_per_check = 128
        with output.open("rb") as handle:
            saved_shape, fortran_order, saved_dtype = read_header(handle)
            assert saved_shape == shape
            assert saved_dtype == dtype
            assert not fortran_order
            for start in range(0, shape[0], rows_per_check):
                stop = min(shape[0], start + rows_per_check)
                actual = np.fromfile(handle, dtype=dtype, count=(stop - start) * shape[1])
                actual = actual.reshape(stop - start, shape[1])
                expected = np.full(actual.shape, 6.5, dtype=dtype)
                source_rows = shape[0] - 1 - np.arange(start, stop)
                affected = (source_rows >= changed_start) & (
                    source_rows < changed_start + changed_rows
                )
                local_rows = source_rows[affected] - changed_start
                expected[affected] = (
                    local_rows[:, None] * shape[1] + np.arange(shape[1]) + 100
                ) * 2
                np.testing.assert_array_equal(actual, expected)
                actual_hash.update(memoryview(actual).cast("B"))
                expected_hash.update(memoryview(expected).cast("B"))
                verified_bytes += actual.nbytes
                del actual, expected
            assert handle.read(1) == b"", "unexpected trailing NPY payload"

        previous = b"previous complete scientific export"
        cancelled_output.write_bytes(previous)
        token = SimpleNamespace(cancelled=False)
        writes = []
        reads.clear()

        def cancel_after_payload(progress):
            if progress.phase == "writing" and (progress.completed_bytes or 0) > 1024**2:
                writes.append(progress.completed_bytes)
                token.cancelled = True

        cancelled_request = RegionExportRequest(
            document, cancelled_output, tuple((0, size) for size in shape), squeeze=False
        )
        try:
            export_streaming_region(
                cancelled_request,
                chunk_target_bytes=chunk_target,
                cancellation_token=token,
                progress_callback=cancel_after_payload,
            )
        except EvaluationCancelled:
            pass
        else:
            raise AssertionError("cancellation after a real payload write did not stop export")
        assert writes
        assert cancelled_output.read_bytes() == previous
        expected_files = {source_path.name, output.name, cancelled_output.name}
        assert {path.name for path in directory.iterdir()} == expected_files
        report = {
            "output_bytes": output_bytes,
            "streamed_output_shape": list(result.shape),
            "full_allocation_failed": failed,
            "baseline_vms_bytes": baseline_vms,
            "address_space_limit_bytes": limit,
            "address_space_headroom_bytes": limit - baseline_vms,
            "baseline_rss_bytes": baseline_rss,
            "baseline_peak_rss_bytes": baseline_peak_rss,
            "final_rss_bytes": process.memory_info().rss,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "logical_source_region_bytes": source_read_bytes,
            "largest_source_region_bytes": largest_read,
            "verified_bytes": verified_bytes,
            "actual_sha256": actual_hash.hexdigest(),
            "independent_sha256": expected_hash.hexdigest(),
            "cancelled_after_payload_write": bool(writes),
            "cancelled_source_reads": len(reads),
            "cancelled_source_region_bytes": sum(reads),
            "previous_output_preserved": True,
            "temporary_files_removed": True,
        }
    finally:
        HDF5ArraySource.read_region = original_read
        source.close()
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    assert sys.argv[1] == "--constrained-probe"
    _constrained_probe(Path(sys.argv[2]))
