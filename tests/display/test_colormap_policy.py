from arrayscope.core.view_state import ChannelMode
from arrayscope.display.colormap_policy import default_colormap_name, resolved_colormap_name


def test_channel_defaults_are_semantic_and_backend_independent():
    assert default_colormap_name(ChannelMode.COMPLEX) == "PAL-relaxed"
    assert default_colormap_name(ChannelMode.ANGLE) == "PAL-relaxed"
    assert default_colormap_name(ChannelMode.REAL) == "gray"
    assert default_colormap_name("imag") == "gray"


def test_explicit_colormap_survives_channel_changes():
    assert resolved_colormap_name(ChannelMode.REAL, "viridis", user_selected=True) == "viridis"
    assert resolved_colormap_name(ChannelMode.COMPLEX, "viridis", user_selected=True) == "viridis"


def test_automatic_colormap_tracks_channel_default():
    assert resolved_colormap_name(ChannelMode.REAL, "viridis", user_selected=False) == "gray"
    assert resolved_colormap_name(ChannelMode.COMPLEX, None, user_selected=False) == "PAL-relaxed"
