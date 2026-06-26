def __getattr__(name: str):
    if name == "PyQtGraphSurface":
        from arrayscope.display.backends.pyqtgraph.surface import PyQtGraphSurface

        return PyQtGraphSurface
    raise AttributeError(name)

__all__ = ["PyQtGraphSurface"]
