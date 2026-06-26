def __getattr__(name: str):
    if name == "VisPySurface":
        from arrayscope.display.backends.vispy.surface import VisPySurface

        return VisPySurface
    raise AttributeError(name)

__all__ = ["VisPySurface"]
