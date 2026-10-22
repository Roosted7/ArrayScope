"""Per-process paths for Qt settings and test artifacts across concurrent runs."""

from pathlib import Path
from uuid import uuid4


def isolated_process_root(temp_dir, *, worker, run_uid, process_id):
    # A nested pytest can inherit the outer worker's xdist environment. The
    # process id separates it even then; ordinary application children do not
    # import conftest and keep inheriting the parent's XDG_CONFIG_HOME.
    invocation = run_uid or uuid4().hex
    role = worker or "serial"
    return Path(temp_dir) / f"arrayscope-test-{invocation}-{role}-{process_id}"


def install_artifact_directory(environ, process_root):
    inherited = environ.get("ARRAYSCOPE_ARTIFACT_DIR")
    managed = environ.get("_ARRAYSCOPE_TEST_ARTIFACT_DIR")
    if inherited is None or inherited == managed:
        # Replace an outer pytest's generated path, preserving an explicit
        # caller override. Application children inherit it unchanged.
        artifacts = str(Path(process_root) / "artifacts")
        environ["ARRAYSCOPE_ARTIFACT_DIR"] = artifacts
        environ["_ARRAYSCOPE_TEST_ARTIFACT_DIR"] = artifacts
