"""Batched VisPy montage tile rendering.

The renderer keeps tile identity, GPU residency, and draw visibility separate.
Presentation commits may change the set of drawn tiles without invalidating
unchanged resident texture slots.  Level-only changes update uniforms only.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from arrayscope.window.display_frame import DisplayTilePayload

try:
    from vispy.visuals import Visual
except Exception:  # pragma: no cover - optional dependency import path
    Visual = object


@dataclass(frozen=True)
class GpuMontageLayerStats:
    visible_items: int = 0
    resident_items: int = 0
    atlas_capacity: int = 0
    atlas_rebuilds: int = 0
    atlas_evictions: int = 0
    texture_uploads: int = 0
    texture_upload_bytes: int = 0
    vertex_uploads: int = 0
    items_updated: int = 0
    items_skipped: int = 0
    level_updates: int = 0
    upload_ms: float = 0.0


class AtlasCapacityError(RuntimeError):
    """Raised when one atlas page cannot contain the requested tile set."""


class TextureAtlasPool:
    """One-page texture atlas with stable, LRU-managed tile residency.

    ``source_id`` is treated as an immutable content identity.  A clean commit
    can therefore reuse a resident slot even when it is the first commit of a
    new viewport session.  Callers must change the source id or mark a tile
    dirty when its pixels change.
    """

    def __init__(self, gloo, *, max_texture_size: int = 8192):
        self._gloo = gloo
        self.max_texture_size = max(1, int(max_texture_size))
        self.tile_shape: tuple[int, int] | None = None
        self.capacity = 0
        self.columns = 1
        self.rows = 1
        self.atlas_shape: tuple[int, int] = (1, 1)
        self.scalar_texture = None
        self.color_texture = None
        self.slots: dict[int, int] = {}
        self.slot_owners: list[int | None] = []
        self.source_ids: dict[int, object] = {}
        self.last_used: dict[int, int] = {}
        self.serial = 0
        self.rebuild_count = 0
        self.eviction_count = 0
        self._clock = 0

    @property
    def resident_count(self) -> int:
        return len(self.source_ids)

    @property
    def cpu_shadow_bytes(self) -> int:
        # Atlas storage is allocated by shape on the GPU.  Only per-tile
        # staging arrays exist during a sub-upload.
        return 0

    def ensure_layout(self, *, tile_shape: tuple[int, int], count: int) -> bool:
        tile_h, tile_w = (max(1, int(tile_shape[0])), max(1, int(tile_shape[1])))
        requested = max(1, int(count))
        max_columns = self.max_texture_size // tile_w
        max_rows = self.max_texture_size // tile_h
        if max_columns < 1 or max_rows < 1:
            raise AtlasCapacityError(
                f"tile shape {(tile_h, tile_w)} exceeds atlas texture limit {self.max_texture_size}"
            )
        max_slots = int(max_columns * max_rows)
        if requested > max_slots:
            raise AtlasCapacityError(
                f"{requested} tiles of shape {(tile_h, tile_w)} require more than one "
                f"{self.max_texture_size}x{self.max_texture_size} atlas page (capacity {max_slots})"
            )

        shape_changed = self.tile_shape != (tile_h, tile_w)
        if not shape_changed and requested <= self.capacity:
            return False

        if shape_changed or self.capacity < 1:
            target_capacity = requested
        else:
            # Amortise growth when no complete visibility estimate was
            # available.  Normal montage commits pass a reserve count and hit
            # the final size in one allocation.
            target_capacity = max(requested, int(np.ceil(self.capacity * 1.5)))
        target_capacity = min(max_slots, max(1, target_capacity))
        columns, rows = _atlas_grid(
            tile_shape=(tile_h, tile_w),
            capacity=target_capacity,
            max_texture_size=self.max_texture_size,
        )
        atlas_shape = (int(rows * tile_h), int(columns * tile_w))

        self.tile_shape = (tile_h, tile_w)
        self.capacity = int(target_capacity)
        self.columns = int(columns)
        self.rows = int(rows)
        self.atlas_shape = atlas_shape
        # Shape-only allocation avoids a full scalar plus RGB CPU shadow copy.
        self.scalar_texture = self._gloo.Texture2D(
            shape=atlas_shape + (1,),
            format="red",
            internalformat="r32f",
            interpolation="nearest",
            wrapping="clamp_to_edge",
        )
        self.color_texture = self._gloo.Texture2D(
            shape=atlas_shape + (3,),
            format="rgb",
            internalformat="rgb8",
            interpolation="nearest",
            wrapping="clamp_to_edge",
        )
        self.slots.clear()
        self.slot_owners = [None] * self.capacity
        self.source_ids.clear()
        self.last_used.clear()
        self.serial += 1
        self.rebuild_count += 1
        return True

    def update_payloads(
        self,
        payloads: dict[int, DisplayTilePayload],
        *,
        tile_shape: tuple[int, int],
        dirty_tiles: tuple[int, ...] | None,
        rgb_already_windowed: bool,
        reserve_count: int | None = None,
    ) -> tuple[dict[int, tuple[float, float, float, float]], GpuMontageLayerStats]:
        start = perf_counter()
        payload_items = tuple(sorted((int(key), value) for key, value in payloads.items()))
        requested_capacity = max(len(payload_items), int(reserve_count or 0), 1)
        rebuilt = self.ensure_layout(tile_shape=tile_shape, count=requested_capacity)
        dirty = set() if dirty_tiles is None else {int(tile) for tile in dirty_tiles}
        active = {int(tile_number) for tile_number, _payload in payload_items}
        uvs: dict[int, tuple[float, float, float, float]] = {}
        uploads = 0
        upload_bytes = 0
        updated = 0
        skipped = 0
        evictions_before = self.eviction_count
        atlas_h, atlas_w = self.atlas_shape
        tile_h, tile_w = self.tile_shape or tile_shape

        for tile_number, payload in payload_items:
            slot, newly_assigned = self._slot_for(tile_number, active_tiles=active)
            self._touch(tile_number)
            row = slot // self.columns
            col = slot % self.columns
            y0 = row * tile_h
            x0 = col * tile_w
            y1 = y0 + tile_h
            x1 = x0 + tile_w
            source_changed = self.source_ids.get(tile_number) != payload.source_id
            should_upload = bool(newly_assigned or source_changed or tile_number in dirty)
            uvs[tile_number] = (x0 / atlas_w, y0 / atlas_h, x1 / atlas_w, y1 / atlas_h)
            if not should_upload:
                skipped += 1
                continue

            scalar, color = _payload_textures(
                payload,
                tile_shape=(tile_h, tile_w),
                rgb_already_windowed=rgb_already_windowed,
            )
            self.scalar_texture.set_data(scalar, offset=(int(y0), int(x0)), copy=False)
            self.color_texture.set_data(color, offset=(int(y0), int(x0)), copy=False)
            self.source_ids[tile_number] = payload.source_id
            updated += 1
            uploads += 2
            upload_bytes += int(scalar.nbytes + color.nbytes)

        elapsed = (perf_counter() - start) * 1000.0 if updated or rebuilt else 0.0
        return uvs, GpuMontageLayerStats(
            visible_items=len(payload_items),
            resident_items=self.resident_count,
            atlas_capacity=self.capacity,
            atlas_rebuilds=int(rebuilt),
            atlas_evictions=self.eviction_count - evictions_before,
            texture_uploads=uploads,
            texture_upload_bytes=upload_bytes,
            items_updated=updated,
            items_skipped=skipped,
            upload_ms=elapsed,
        )

    def _slot_for(self, tile_number: int, *, active_tiles: set[int]) -> tuple[int, bool]:
        current = self.slots.get(int(tile_number))
        if current is not None:
            return int(current), False

        try:
            slot = self.slot_owners.index(None)
        except ValueError:
            candidates = [
                owner
                for owner in self.slot_owners
                if owner is not None and int(owner) not in active_tiles
            ]
            if not candidates:
                raise AtlasCapacityError(
                    f"atlas has {self.capacity} slots but {len(active_tiles)} active tiles require residency"
                )
            victim = min(candidates, key=lambda owner: self.last_used.get(int(owner), -1))
            slot = int(self.slots.pop(int(victim)))
            self.source_ids.pop(int(victim), None)
            self.last_used.pop(int(victim), None)
            self.eviction_count += 1

        previous = self.slot_owners[int(slot)]
        if previous is not None and int(previous) != int(tile_number):
            self.slots.pop(int(previous), None)
            self.source_ids.pop(int(previous), None)
            self.last_used.pop(int(previous), None)
        self.slot_owners[int(slot)] = int(tile_number)
        self.slots[int(tile_number)] = int(slot)
        return int(slot), True

    def _touch(self, tile_number: int) -> None:
        self._clock += 1
        self.last_used[int(tile_number)] = int(self._clock)


class GpuMontageLayer:
    def __init__(self, *, scene, visuals, gloo, transforms, parent):
        self._scene = scene
        self._visuals = visuals
        self._gloo = gloo
        self._transforms = transforms
        self._pool = TextureAtlasPool(gloo)
        self._visual = scene.visuals.create_visual_node(GpuWindowedTileVisual)(parent=parent)
        self._visual.visible = False
        self._geometry_key: tuple[object, ...] | None = None
        self._levels: tuple[float, float] = (0.0, 1.0)
        self._visible_items = 0
        self._last_stats = GpuMontageLayerStats()

    @property
    def visual(self):
        return self._visual

    @property
    def last_stats(self) -> GpuMontageLayerStats:
        return self._last_stats

    def clear(self) -> None:
        # Hiding a layer must not discard useful GPU residency.  A later
        # viewport/session can reuse the same source identities.
        self._visual.visible = False
        self._geometry_key = None
        self._visible_items = 0

    def set_levels(self, levels) -> GpuMontageLayerStats:
        levels = _normalize_levels(levels, self._levels)
        changed = levels != self._levels
        self._levels = levels
        if changed:
            self._visual.set_levels(levels)
        self._last_stats = GpuMontageLayerStats(
            visible_items=self._visible_items,
            resident_items=self._pool.resident_count,
            atlas_capacity=self._pool.capacity,
            level_updates=int(changed),
            items_skipped=self._visible_items,
        )
        return self._last_stats

    def update(
        self,
        *,
        payloads: dict[int, DisplayTilePayload],
        geometry,
        levels,
        dirty_tiles: tuple[int, ...] | None,
        rgb_already_windowed: bool,
    ) -> GpuMontageLayerStats:
        montage = getattr(geometry, "montage", None)
        if montage is None:
            self.clear()
            return GpuMontageLayerStats()
        payloads = {int(key): value for key, value in dict(payloads or {}).items()}
        reserve_count = _atlas_reserve_count(geometry, minimum=len(payloads))
        uvs, texture_stats = self._pool.update_payloads(
            payloads,
            tile_shape=(int(montage.tile_height), int(montage.tile_width)),
            dirty_tiles=dirty_tiles,
            rgb_already_windowed=rgb_already_windowed,
            reserve_count=reserve_count,
        )
        geometry_key = (
            tuple(
                (
                    int(key),
                    int(self._pool.slots[int(key)]),
                    _payload_mode(payload, rgb_already_windowed=rgb_already_windowed),
                )
                for key, payload in sorted(payloads.items())
            ),
            int(montage.tile_height),
            int(montage.tile_width),
            int(montage.columns),
            int(montage.rows),
            int(montage.gap),
            self._pool.serial,
        )
        vertex_uploads = 0
        if geometry_key != self._geometry_key:
            vertices, texcoords, modes = _quad_buffers(
                montage,
                payloads,
                uvs,
                rgb_already_windowed=rgb_already_windowed,
            )
            self._visual.set_geometry(vertices, texcoords, modes)
            self._geometry_key = geometry_key
            vertex_uploads = 1
        levels = _normalize_levels(levels, self._levels)
        level_updates = 0
        if levels != self._levels:
            self._levels = levels
            self._visual.set_levels(levels)
            level_updates = 1
        if self._pool.scalar_texture is not None and self._pool.color_texture is not None:
            self._visual.set_textures(self._pool.scalar_texture, self._pool.color_texture)
        self._visible_items = len(payloads)
        self._visual.visible = bool(payloads)
        self._last_stats = GpuMontageLayerStats(
            visible_items=texture_stats.visible_items,
            resident_items=texture_stats.resident_items,
            atlas_capacity=texture_stats.atlas_capacity,
            atlas_rebuilds=texture_stats.atlas_rebuilds,
            atlas_evictions=texture_stats.atlas_evictions,
            texture_uploads=texture_stats.texture_uploads,
            texture_upload_bytes=texture_stats.texture_upload_bytes,
            vertex_uploads=vertex_uploads,
            items_updated=texture_stats.items_updated,
            items_skipped=texture_stats.items_skipped,
            level_updates=level_updates,
            upload_ms=texture_stats.upload_ms,
        )
        return self._last_stats


class GpuWindowedTileVisual(Visual):
    _vertex_shader = """
    attribute vec2 a_position;
    attribute vec2 a_texcoord;
    attribute float a_mode;
    varying vec2 v_texcoord;
    varying float v_mode;

    void main() {
        v_texcoord = a_texcoord;
        v_mode = a_mode;
        gl_Position = $transform(vec4(a_position, 0.0, 1.0));
    }
    """

    _fragment_shader = """
    uniform sampler2D u_scalar_texture;
    uniform sampler2D u_color_texture;
    uniform vec2 u_levels;
    varying vec2 v_texcoord;
    varying float v_mode;

    void main() {
        float scalar = texture2D(u_scalar_texture, v_texcoord).r;
        vec3 color = texture2D(u_color_texture, v_texcoord).rgb;
        float span = max(u_levels.y - u_levels.x, 1e-12);
        float intensity = clamp((scalar - u_levels.x) / span, 0.0, 1.0);
        float alpha = 1.0;
        if (scalar != scalar) {
            intensity = 0.0;
            alpha = 0.0;
        }
        if (v_mode < 0.5) {
            gl_FragColor = vec4(vec3(intensity), alpha);
        } else if (v_mode < 1.5) {
            gl_FragColor = vec4(color * intensity, alpha);
        } else {
            gl_FragColor = vec4(color, alpha);
        }
    }
    """

    def __init__(self, **kwargs):
        from vispy import gloo
        super().__init__(vcode=self._vertex_shader, fcode=self._fragment_shader, **kwargs)
        self._vertices = gloo.VertexBuffer(np.zeros((0, 2), dtype=np.float32))
        self._texcoords = gloo.VertexBuffer(np.zeros((0, 2), dtype=np.float32))
        self._modes = gloo.VertexBuffer(np.zeros((0,), dtype=np.float32))
        self.vertex_data = np.zeros((0, 2), dtype=np.float32)
        self.texcoord_data = np.zeros((0, 2), dtype=np.float32)
        self.mode_data = np.zeros((0,), dtype=np.float32)
        self._bounds_xy = (0.0, 0.0, 0.0, 0.0)
        self._scalar_texture = None
        self._color_texture = None
        self._levels = (0.0, 1.0)
        self.set_gl_state("translucent", cull_face=False)
        self._draw_mode = "triangles"
        self.freeze()

    def set_geometry(self, vertices, texcoords, modes) -> None:
        vertices = np.asarray(vertices, dtype=np.float32).reshape((-1, 2))
        texcoords = np.asarray(texcoords, dtype=np.float32).reshape((-1, 2))
        modes = np.asarray(modes, dtype=np.float32).reshape((-1,))
        self.vertex_data = vertices
        self.texcoord_data = texcoords
        self.mode_data = modes
        self._vertices.set_data(vertices)
        self._texcoords.set_data(texcoords)
        self._modes.set_data(modes)
        if len(vertices):
            self._bounds_xy = (
                float(np.nanmin(vertices[:, 0])),
                float(np.nanmax(vertices[:, 0])),
                float(np.nanmin(vertices[:, 1])),
                float(np.nanmax(vertices[:, 1])),
            )
        else:
            self._bounds_xy = (0.0, 0.0, 0.0, 0.0)
        self.update()

    def set_textures(self, scalar_texture, color_texture) -> None:
        if scalar_texture is self._scalar_texture and color_texture is self._color_texture:
            return
        self._scalar_texture = scalar_texture
        self._color_texture = color_texture
        self.update()

    def set_levels(self, levels) -> None:
        self._levels = _normalize_levels(levels, self._levels)
        self.update()

    def _prepare_transforms(self, view) -> None:
        view.view_program.vert["transform"] = view.transforms.get_transform()

    def _prepare_draw(self, view):
        if self._scalar_texture is None or self._color_texture is None:
            return False
        program = view.view_program
        program["a_position"] = self._vertices
        program["a_texcoord"] = self._texcoords
        program["a_mode"] = self._modes
        program["u_scalar_texture"] = self._scalar_texture
        program["u_color_texture"] = self._color_texture
        program["u_levels"] = tuple(float(value) for value in self._levels)
        return True

    def _bounds(self, axis, view):
        del view
        if axis == 0:
            return self._bounds_xy[0], self._bounds_xy[1]
        if axis == 1:
            return self._bounds_xy[2], self._bounds_xy[3]
        return (0.0, 0.0)


def create_gpu_montage_layer(*, scene, visuals, gloo, transforms, parent) -> GpuMontageLayer:
    return GpuMontageLayer(scene=scene, visuals=visuals, gloo=gloo, transforms=transforms, parent=parent)


def _payload_textures(payload: DisplayTilePayload, *, tile_shape: tuple[int, int], rgb_already_windowed: bool):
    tile_h, tile_w = (int(tile_shape[0]), int(tile_shape[1]))
    image = np.asarray(payload.image)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        color = _fit_color(image[..., :3], (tile_h, tile_w))
        if rgb_already_windowed:
            scalar = np.ones((tile_h, tile_w), dtype=np.float32)
        else:
            source = payload.histogram_data
            if source is None:
                source = _luminance(color)
            scalar = _fit_scalar(source, (tile_h, tile_w))
        return scalar, color
    scalar = _fit_scalar(image, (tile_h, tile_w))
    return scalar, np.zeros((tile_h, tile_w, 3), dtype=np.uint8)


def _fit_scalar(data, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    arr = np.asarray(data, dtype=np.float32)
    height = min(shape[0], int(arr.shape[0]))
    width = min(shape[1], int(arr.shape[1]))
    if height > 0 and width > 0:
        out[:height, :width] = arr[:height, :width]
    return out


def _fit_color(data, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape + (3,), dtype=np.uint8)
    arr = np.asarray(data)
    if np.issubdtype(arr.dtype, np.floating):
        if arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = np.clip(arr * 255.0, 0.0, 255.0)
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8, copy=False)
    height = min(shape[0], int(arr.shape[0]))
    width = min(shape[1], int(arr.shape[1]))
    if height > 0 and width > 0:
        out[:height, :width, :] = arr[:height, :width, :3]
    return out


def _quad_buffers(montage, payloads, uvs, *, rgb_already_windowed: bool):
    vertices = []
    texcoords = []
    modes = []
    tile_w = int(montage.tile_width)
    tile_h = int(montage.tile_height)
    stride_x = tile_w + int(montage.gap)
    stride_y = tile_h + int(montage.gap)
    for tile_number, source_index in enumerate(tuple(montage.indices)):
        payload = payloads.get(int(tile_number))
        uv = uvs.get(int(tile_number))
        if payload is None or uv is None:
            continue
        row = int(tile_number) // int(montage.columns)
        col = int(tile_number) % int(montage.columns)
        x0 = float(col * stride_x)
        y0 = float(row * stride_y)
        x1 = x0 + tile_w
        y1 = y0 + tile_h
        u0, v0, u1, v1 = uv
        vertices.extend(((x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)))
        texcoords.extend(((u0, v0), (u1, v0), (u1, v1), (u0, v0), (u1, v1), (u0, v1)))
        image = np.asarray(payload.image)
        if image.ndim == 3 and image.shape[-1] in (3, 4):
            mode = 2.0 if rgb_already_windowed else 1.0
        else:
            mode = 0.0
        modes.extend((mode,) * 6)
    return (
        np.asarray(vertices, dtype=np.float32).reshape((-1, 2)),
        np.asarray(texcoords, dtype=np.float32).reshape((-1, 2)),
        np.asarray(modes, dtype=np.float32).reshape((-1,)),
    )


def _luminance(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


def _normalize_levels(levels, fallback):
    if levels is None:
        return tuple(float(value) for value in fallback)
    low, high = levels
    low = float(low)
    high = float(high)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return tuple(float(value) for value in fallback)
    return low, high


def _atlas_grid(*, tile_shape: tuple[int, int], capacity: int, max_texture_size: int) -> tuple[int, int]:
    tile_h, tile_w = (int(tile_shape[0]), int(tile_shape[1]))
    capacity = max(1, int(capacity))
    max_columns = max_texture_size // tile_w
    max_rows = max_texture_size // tile_h
    # Aim for an approximately square atlas in pixels, not in slot count.
    ideal_columns = int(np.ceil(np.sqrt(capacity * tile_h / tile_w)))
    columns = max(1, min(max_columns, ideal_columns))
    rows = int(np.ceil(capacity / columns))
    if rows > max_rows:
        columns = int(np.ceil(capacity / max_rows))
        rows = int(np.ceil(capacity / columns))
    if columns > max_columns or rows > max_rows:
        raise AtlasCapacityError(
            f"atlas grid {columns}x{rows} exceeds texture limit {max_texture_size} for tile shape {tile_shape}"
        )
    return columns, rows


def _atlas_reserve_count(geometry, *, minimum: int) -> int:
    states = tuple(getattr(geometry, "montage_tile_states", ()) or ())
    if not states:
        return max(1, int(minimum))
    active = 0
    for state in states:
        value = str(getattr(state, "value", state))
        if value in {"loaded", "loading"}:
            active += 1
    return max(1, int(minimum), int(active))


def _payload_mode(payload: DisplayTilePayload, *, rgb_already_windowed: bool) -> int:
    image = np.asarray(payload.image)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        return 2 if rgb_already_windowed else 1
    return 0
