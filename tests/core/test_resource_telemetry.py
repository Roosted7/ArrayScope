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


class _WarmUpProcess(_Process):
    """A process sampler with psutil's real first-call-returns-zero behaviour."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def cpu_percent(self, interval=None):
        assert interval is None
        self.calls += 1
        # psutil's delta sampler has no previous reading on the first call.
        return 0.0 if self.calls == 1 else 12.5


class _WarmUpPsutil(_Psutil):
    def __init__(self):
        self.calls = 0
        self.process = _WarmUpProcess()

    def Process(self):
        return self.process

    def cpu_percent(self, interval=None):
        assert interval is None
        self.calls += 1
        return 0.0 if self.calls == 1 else 25.0


def test_first_cpu_sample_is_live_not_the_psutil_warm_up_zero():
    # psutil's `cpu_percent(interval=None)` returns 0.0 on its first call by
    # construction. The governor maps `system_cpu <= 0.0` onto a neutral 0.5
    # headroom, so an un-warmed sampler silently disables CPU pressure
    # entirely. The sampler must burn that first reading at construction.
    psutil = _WarmUpPsutil()

    snapshot = sample_resource_snapshot(psutil_module=psutil, cpu_count=16)

    assert snapshot.cpu.system_cpu_percent == 25.0
    assert snapshot.cpu.process_cpu_percent == 12.5
    assert snapshot.cpu.source == "psutil"
    # Two calls each: the discarded warm-up, then the reading returned above.
    assert psutil.calls == 2
    assert psutil.process.calls == 2


def test_resource_snapshot_falls_back_when_psutil_unavailable():
    snapshot = sample_resource_snapshot(psutil_module=False, cpu_count=4)

    assert snapshot.cpu.logical_count == 4
    assert snapshot.cpu.system_cpu_percent is None
    assert snapshot.cpu.process_cpu_percent is None
    assert snapshot.cpu.source == "fallback"
    assert snapshot.cpu.warnings
