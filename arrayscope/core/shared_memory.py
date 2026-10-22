"""Version-compatible attachment to publisher-owned shared memory.

Only the publisher unlinks a segment. Python 3.13 introduced ``track=False``
for attachers; older spawn children share their parent's resource tracker and
must retain its registration. Unregistering an attachment there would remove
the publisher's entry from the same tracker namespace.
"""

from __future__ import annotations

from inspect import signature


def attach_shared_memory(name):
    """Attach without adopting unlink ownership where the runtime supports it.

    Legacy constructors omit the newer keyword. Their ordinary registration
    is safe for the related multiprocessing children used by our transport;
    closing an attachment never unregisters or unlinks its publisher's name.
    Constructor failures propagate unchanged rather than triggering a retry.
    """
    from multiprocessing import shared_memory

    options = {"name": name, "create": False}
    if "track" in signature(shared_memory.SharedMemory).parameters:
        options["track"] = False
    return shared_memory.SharedMemory(**options)
