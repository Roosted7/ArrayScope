"""Load diagnostics must not prevent platforms without Unix resource from testing."""

import json
import subprocess
import sys


def test_load_reporting_imports_and_samples_without_unix_resource():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys; sys.modules['resource']=None; "
            "from tests import load_report; "
            "start=load_report.LoadWindow.sample(); end=load_report.LoadWindow.sample(); "
            "assert start.load is None and end.cpu_seconds >= start.cpu_seconds; "
            "assert load_report.opening_line(start) is None; "
            "assert load_report.closing_line(start,end) is None; "
            "print(json.dumps(load_report.observation_context(start)))",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {}
