"""Suppression of known-harmless third-party terminal noise.

Three messages show up on ordinary desktop runs and mean nothing actionable
to anyone (2026-07-31, extended 2026-08-01):

- ``Unable to find extension: VK_EXT_physical_device_drm`` — wgpu-native's
  Vulkan backend probes this *optional* extension (it only adds DRM device
  identification) and warns when the driver does not expose it; the
  adapter works unaffected.  wgpu-py forwards native logs onto the Python
  ``"wgpu"`` logger, and with no handler configured logging's last-resort
  handler prints every WARNING to stderr.
- ``Detected skylake derivative running on mesa i915. Clears to srgb
  textures will use manual shader clears.`` — wgpu-hal reporting that it
  has *already* worked around a known Intel/mesa driver bug for this
  adapter.  It names the workaround it took, not a decision anyone can
  make: nothing in the application selects that path, and every render is
  correct either way.  Matched on the invariant tail so the same
  workaround notice is dropped whatever chip family the driver names.
- ``qt.qpa.wayland.textinput: ... Got leave event for surface 0x0 with
  focusing surface ...`` — QtWayland's text-input-v3 client warns when the
  compositor sends a leave for a surface Qt no longer tracks (focus
  handoffs between windows/popups).  Known upstream QtWayland noise;
  input methods keep working.

Neither can be *fixed* from here — one is a driver capability, the other an
upstream Qt/compositor disagreement — so each is silenced exactly at its
own width and no wider:

- The wgpu filter drops only records containing the known message; every
  other wgpu warning/error still prints.
- The Qt rule disables only the ``qt.qpa.wayland.textinput`` category's
  warnings.  Qt applies an explicit ``QT_LOGGING_RULES`` environment
  variable *after* rules set in code, so a user debugging input-method
  behavior can re-enable the category without touching ArrayScope.

Both hooks run from ``launch._prepare_qt_environment()`` — the one
pre-QApplication hook every app path goes through — before any adapter
request or Wayland surface exists.
"""

from __future__ import annotations

import logging

# Exact-substring allowlist of wgpu-native messages to drop.  Substring, not
# equality: wgpu-hal occasionally decorates messages with context prefixes.
_WGPU_NOISE_SUBSTRINGS = (
    "Unable to find extension: VK_EXT_physical_device_drm",
    "Clears to srgb textures will use manual shader clears",
)

# ``QLoggingCategory.setFilterRules`` REPLACES all rules previously set from
# code, so every code-side rule this application wants must live in this one
# string (semicolon- or newline-separated).
QT_LOGGING_NOISE_RULES = "qt.qpa.wayland.textinput.warning=false"


class _KnownNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(noise in message for noise in _WGPU_NOISE_SUBSTRINGS)


# One shared instance so repeated installation stays idempotent:
# ``Logger.addFilter`` refuses a filter that is already present.
_WGPU_NOISE_FILTER = _KnownNoiseFilter()


def install_wgpu_log_noise_filter() -> bool:
    """Drop the known-harmless wgpu-native messages; keep everything else.

    Attached to the ``"wgpu"`` logger itself (the one wgpu-py emits native
    records on), so the drop happens before propagation and applies whatever
    handlers the process ends up with.  Idempotent; returns whether the filter
    is now attached.

    NEVER creates that logger.  wgpu-py installs its own ``Logger`` subclass
    at import time by wrapping ``getLogger("wgpu")`` in ``setLoggerClass``,
    and ``getLogger`` returns an already-registered logger unchanged -- so
    pre-creating a plain ``Logger`` here left wgpu's own
    ``assert isinstance(logger, WGPULogger)`` failing, ``import wgpu`` raising
    ``AssertionError``, and the AUTO backend probe (which catches everything
    around device creation) reporting "no usable wgpu device" on a machine
    whose GPU was fine.  Calling this before wgpu is imported was the one
    thing that was never safe.

    So the install is deferred instead: nothing to attach to means nothing to
    filter yet, and :func:`arrayscope.gpu.wgpu_adapter.shared_wgpu_adapter`
    calls this again immediately after ``import wgpu`` -- before the adapter
    request, which is what emits the messages.  That is the process's only
    adapter request, so one deferred install covers every path.
    """

    # A PlaceHolder (left by a child logger such as "wgpu.foo") is not a
    # logger and materializing it through getLogger would recreate the bug.
    existing = logging.Logger.manager.loggerDict.get("wgpu")
    if not isinstance(existing, logging.Logger):
        return False
    existing.addFilter(_WGPU_NOISE_FILTER)
    return True


def apply_qt_logging_noise_rules() -> None:
    """Disable the known-noisy Qt logging categories (see module docstring).

    Callable before QApplication creation; Qt applies the rules to matching
    categories as they are registered.
    """
    from pyqtgraph.Qt import QtCore

    QtCore.QLoggingCategory.setFilterRules(QT_LOGGING_NOISE_RULES)
