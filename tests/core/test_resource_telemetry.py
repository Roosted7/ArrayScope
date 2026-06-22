from arrayscope.core.resource_telemetry import sample_resource_snapshot


class _VirtualMemory:
    total = 16 * 1024**3
    available = 8 * 1024**3


class _MemoryInfo:
    rss = 123


class _Process:
    instances = 0

    def __init__(self):
        type(self).instances += 1

    def memory_info(self):
        return _MemoryInfo()

    def cpu_percent(self, interval=None):
        assert interval is None
        return 12.5


class _Psutil:
    @staticmethod
    def virtual_memory():
        return _VirtualMemory()

    @staticmethod
    def Process():
        return _Process()

    @staticmethod
    def cpu_percent(interval=None):
        assert interval is None
        return 25.0


def test_resource_snapshot_uses_nonblocking_psutil_cpu():
    _Process.instances = 0
    psutil = _Psutil()
    snapshot = sample_resource_snapshot(psutil_module=psutil, cpu_count=16)

    assert snapshot.memory.total_bytes == 16 * 1024**3
    assert snapshot.memory.process_rss_bytes == 123
    assert snapshot.cpu.logical_count == 16
    assert snapshot.cpu.system_cpu_percent == 25.0
    assert snapshot.cpu.process_cpu_percent == 12.5
    assert snapshot.cpu.source == "psutil"
    sample_resource_snapshot(psutil_module=psutil, cpu_count=16)
    # One Process is used for reusable CPU-percent sampling; memory sampling
    # still takes its own RSS Process snapshot each call.
    assert _Process.instances == 3


def test_resource_snapshot_falls_back_when_psutil_unavailable():
    snapshot = sample_resource_snapshot(psutil_module=False, cpu_count=4)

    assert snapshot.cpu.logical_count == 4
    assert snapshot.cpu.system_cpu_percent is None
    assert snapshot.cpu.process_cpu_percent is None
    assert snapshot.cpu.source == "fallback"
    assert snapshot.cpu.warnings
