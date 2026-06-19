"""VisPy raster image visual and texture preparation helpers."""

from __future__ import annotations

import numpy as np

from arrayscope.display.shader_mapping import (
    ShaderDisplayMode,
    ShaderScale,
    TexturePlaneKind,
    default_phase_lut,
    pack_texture_data,
    shader_component_uniform,
)

try:
    from vispy.visuals import Visual
except Exception:  # pragma: no cover - optional dependency import path
    Visual = object


class GpuMappedImageVisual(Visual):
    _vertex_shader = """
    attribute vec2 a_position;
    attribute vec2 a_texcoord;
    varying vec2 v_texcoord;

    void main() {
        v_texcoord = a_texcoord;
        gl_Position = $transform(vec4(a_position, 0.0, 1.0));
    }
    """

    _fragment_shader = """
    uniform sampler2D u_color_texture;
    uniform sampler2D u_scalar_texture;
    uniform sampler2D u_lut_texture;
    uniform vec2 u_levels;
    uniform float u_mode;
    uniform float u_scale_mode;
    uniform float u_symlog_constant;
    uniform float u_component_mode;
    varying vec2 v_texcoord;

    float complex_component(vec2 z) {
        if (u_component_mode > 2.5) {
            return atan(z.y, z.x);
        }
        if (u_component_mode > 1.5) {
            return length(z);
        }
        if (u_component_mode > 0.5) {
            return z.y;
        }
        return z.x;
    }

    float map_scale(float value) {
        if (u_scale_mode > 1.5) {
            return sign(value) * log(1.0 + abs(value) / pow(10.0, u_symlog_constant)) / log(10.0);
        }
        if (u_scale_mode > 0.5) {
            return log(max(value, 0.0)) / log(10.0);
        }
        return value;
    }

    void main() {
        vec4 scalar_sample = texture2D(u_scalar_texture, v_texcoord);
        float scalar = scalar_sample.r;
        vec3 color = texture2D(u_color_texture, v_texcoord).rgb;
        if (u_mode > 2.5) {
            vec2 z = scalar_sample.rg;
            scalar = complex_component(z);
            float phase = atan(z.y, z.x);
            float phase_index = clamp((phase + 3.141592653589793) / 6.283185307179586, 0.0, 1.0);
            color = texture2D(u_lut_texture, vec2(phase_index, 0.5)).rgb;
        } else if (u_mode > 1.5) {
            scalar = complex_component(scalar_sample.rg);
            color = vec3(1.0, 1.0, 1.0);
        } else if (u_mode > 0.5) {
            color = vec3(1.0, 1.0, 1.0);
        }
        scalar = map_scale(scalar);
        float span = max(u_levels.y - u_levels.x, 1e-12);
        float intensity = clamp((scalar - u_levels.x) / span, 0.0, 1.0);
        if (scalar != scalar) {
            discard;
        }
        gl_FragColor = vec4(color * intensity, 1.0);
    }
    """

    def __init__(self, **kwargs):
        from vispy import gloo

        self._gloo = gloo
        self._vertices = gloo.VertexBuffer(np.zeros((6, 2), dtype=np.float32))
        self._texcoords = gloo.VertexBuffer(
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            )
        )
        self._color_texture = None
        self._scalar_texture = None
        self._lut_texture = None
        self._shape = (0, 0)
        self._mode = 0.0
        self._scale_mode = 0.0
        self._symlog_constant = 0.0
        self._component_mode = 0.0
        self._levels = (0.0, 1.0)
        self.color_source_id = None
        self.scalar_source_id = None
        self.upload_count = 0
        self.level_update_count = 0
        super().__init__(vcode=self._vertex_shader, fcode=self._fragment_shader, **kwargs)
        self.set_gl_state(depth_test=False, cull_face=False, blend=False)
        self._draw_mode = "triangles"
        self.freeze()

    @property
    def levels(self) -> tuple[float, float]:
        return self._levels

    def set_data(
        self,
        color,
        scalar,
        *,
        levels,
        color_source_id=None,
        scalar_source_id=None,
        copy: bool = False,
    ) -> None:
        color_array = _contiguous_color_texture(color, copy=copy)
        scalar_array = _contiguous_scalar(scalar, copy=copy)
        if tuple(color_array.shape[:2]) != tuple(scalar_array.shape[:2]):
            raise ValueError("windowed RGB color and scalar textures must have matching image shape")
        self._set_vertices(tuple(color_array.shape[:2]))
        scalar_plane = scalar_array[..., np.newaxis]
        if self._color_texture is None or tuple(self._shape) != tuple(color_array.shape[:2]):
            self._color_texture = self._gloo.Texture2D(
                color_array,
                format="rgb",
                internalformat="rgb8",
                interpolation="nearest",
                wrapping="clamp_to_edge",
            )
            self._scalar_texture = self._gloo.Texture2D(
                scalar_plane,
                format="red",
                internalformat="r32f",
                interpolation="nearest",
                wrapping="clamp_to_edge",
            )
        else:
            self._color_texture.set_data(color_array, copy=False)
            self._scalar_texture.set_data(scalar_plane, copy=False)
        self._shape = tuple(int(size) for size in color_array.shape[:2])
        self._mode = 0.0
        self._scale_mode = 0.0
        self._symlog_constant = 0.0
        self._component_mode = 0.0
        self.color_source_id = color_source_id
        self.scalar_source_id = scalar_source_id
        self.upload_count += 1
        self.set_levels(levels, count=False)

    def set_mapped_data(
        self,
        data,
        *,
        texture_kind: TexturePlaneKind,
        levels,
        source_id=None,
        shader_mapping=None,
        copy: bool = False,
    ) -> None:
        texture_kind = _coerce_texture_kind(texture_kind)
        if texture_kind == TexturePlaneKind.COMPLEX_RG32F:
            texture_format = "rg"
            internal_format = "rg32f"
            display_mode = getattr(shader_mapping, "display_mode", None)
            display_mode = getattr(display_mode, "value", display_mode)
            mode = 3.0 if display_mode == ShaderDisplayMode.PHASE_COLOR.value else 2.0
        else:
            texture_format = "red"
            internal_format = "r32f"
            mode = 1.0
        if self._scalar_texture is not None and self.scalar_source_id == source_id and float(self._mode) == mode:
            self._scale_mode = _shader_scale_uniform(getattr(shader_mapping, "scale", None))
            self._symlog_constant = float(getattr(shader_mapping, "symlog_constant", 0.0) or 0.0)
            self._component_mode = shader_component_uniform(getattr(shader_mapping, "component", None))
            self._set_lut_texture(getattr(shader_mapping, "lut_data", None))
            self.set_levels(levels, count=False)
            return
        data_array = np.array(data, copy=True) if copy else np.asarray(data)
        if texture_kind == TexturePlaneKind.COMPLEX_RG32F:
            data_array = pack_texture_data(data_array, TexturePlaneKind.COMPLEX_RG32F)
        else:
            data_array = pack_texture_data(data_array, TexturePlaneKind.SCALAR_R32F)[..., np.newaxis]
        self._set_vertices(tuple(data_array.shape[:2]))
        shape_changed = tuple(self._shape) != tuple(data_array.shape[:2]) or float(self._mode) != mode
        if self._color_texture is None:
            self._color_texture = self._gloo.Texture2D(
                np.ones((1, 1, 3), dtype=np.float32),
                format="rgb",
                internalformat="rgb32f",
                interpolation="nearest",
                wrapping="clamp_to_edge",
            )
        if self._scalar_texture is None or shape_changed:
            self._scalar_texture = self._gloo.Texture2D(
                data_array,
                format=texture_format,
                internalformat=internal_format,
                interpolation="nearest",
                wrapping="clamp_to_edge",
            )
        else:
            self._scalar_texture.set_data(data_array, copy=False)
        self._shape = tuple(int(size) for size in data_array.shape[:2])
        self._mode = mode
        self._scale_mode = _shader_scale_uniform(getattr(shader_mapping, "scale", None))
        self._symlog_constant = float(getattr(shader_mapping, "symlog_constant", 0.0) or 0.0)
        self._component_mode = shader_component_uniform(getattr(shader_mapping, "component", None))
        self.color_source_id = None
        self.scalar_source_id = source_id
        self._set_lut_texture(getattr(shader_mapping, "lut_data", None))
        self.upload_count += 1
        self.set_levels(levels, count=False)

    def _set_lut_texture(self, lut_data) -> None:
        lut = default_phase_lut() if lut_data is None else np.asarray(lut_data)
        if lut.ndim != 2 or lut.shape[0] < 1 or lut.shape[1] < 3:
            lut = default_phase_lut()
        lut = lut[:, :3]
        if lut.dtype != np.uint8:
            if np.issubdtype(lut.dtype, np.floating) and lut.size and float(np.nanmax(lut)) <= 1.0:
                lut = lut * 255.0
            lut = np.clip(lut, 0, 255).astype(np.uint8)
        lut = np.ascontiguousarray(lut.reshape((1, lut.shape[0], 3)))
        if self._lut_texture is None or tuple(getattr(self._lut_texture, "shape", ())) != tuple(lut.shape):
            self._lut_texture = self._gloo.Texture2D(
                lut,
                format="rgb",
                internalformat="rgb8",
                interpolation="linear",
                wrapping="clamp_to_edge",
            )
        else:
            self._lut_texture.set_data(lut, copy=False)

    def set_levels(self, levels, *, count: bool = True) -> None:
        self._levels = _normalize_levels(levels, self._levels)
        if count:
            self.level_update_count += 1
        self.update()

    def _set_vertices(self, shape: tuple[int, int]) -> None:
        height, width = (int(shape[0]), int(shape[1]))
        vertices = np.array(
            [
                [0.0, 0.0],
                [float(width), 0.0],
                [float(width), float(height)],
                [0.0, 0.0],
                [float(width), float(height)],
                [0.0, float(height)],
            ],
            dtype=np.float32,
        )
        self._vertices.set_data(vertices)

    def _prepare_transforms(self, view) -> None:
        view.view_program.vert["transform"] = view.transforms.get_transform()

    def _prepare_draw(self, view):
        if self._color_texture is None or self._scalar_texture is None:
            return False
        if self._lut_texture is None:
            self._set_lut_texture(None)
        program = view.view_program
        program["a_position"] = self._vertices
        program["a_texcoord"] = self._texcoords
        program["u_color_texture"] = self._color_texture
        program["u_scalar_texture"] = self._scalar_texture
        program["u_lut_texture"] = self._lut_texture
        program["u_levels"] = tuple(float(value) for value in self._levels)
        program["u_mode"] = float(self._mode)
        program["u_scale_mode"] = float(self._scale_mode)
        program["u_symlog_constant"] = float(self._symlog_constant)
        program["u_component_mode"] = float(self._component_mode)
        return True

    def _bounds(self, axis, view):
        del view
        if axis == 0:
            return (0.0, float(self._shape[1]))
        if axis == 1:
            return (0.0, float(self._shape[0]))
        return (0.0, 0.0)


def _normalize_levels(levels, fallback):
    if levels is None:
        levels = fallback
    low, high = levels
    low = float(low)
    high = float(high)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return (0.0, 1.0)
    return (low, high)


def _coerce_texture_kind(texture_kind):
    if texture_kind is None:
        return None
    if isinstance(texture_kind, TexturePlaneKind):
        return texture_kind
    if hasattr(texture_kind, "value"):
        texture_kind = texture_kind.value
    return TexturePlaneKind(texture_kind)


def _shader_scale_uniform(scale) -> float:
    if scale is None:
        return 0.0
    if isinstance(scale, ShaderScale):
        value = scale.value
    else:
        value = getattr(scale, "value", scale)
    if value == ShaderScale.LOG.value:
        return 1.0
    if value == ShaderScale.SYMLOG.value:
        return 2.0
    return 0.0


def _contiguous_display(data):
    arr = np.asarray(data)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def _contiguous_color_texture(data, *, copy: bool = False):
    arr = np.array(data, copy=True) if copy else np.asarray(data)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    if np.issubdtype(arr.dtype, np.floating) and arr.size and float(np.nanmax(arr)) > 1.0:
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def _contiguous_scalar(data, *, copy: bool = False):
    arr = np.array(data, dtype=np.float32, copy=True) if copy else np.asarray(data, dtype=np.float32)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr
