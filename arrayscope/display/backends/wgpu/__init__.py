def __getattr__(name: str):
    if name == "WgpuSurface":
        from arrayscope.display.backends.wgpu.surface import WgpuSurface

        return WgpuSurface
    raise AttributeError(name)

__all__ = ["WgpuSurface"]
