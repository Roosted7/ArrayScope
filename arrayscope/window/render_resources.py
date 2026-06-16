"""Memory-policy and render-budget helpers for ArrayScope windows."""

from __future__ import annotations

import numpy as np

from arrayscope.core.memory_budget import estimate_display_image_bytes
from arrayscope.core.memory_policy import apply_policy_hysteresis, compute_memory_policy, input_nbytes_for
from arrayscope.core.view_state import ChannelMode


class RenderResourceMixin:
    def _estimated_image_display_bytes(self, view_state):
        if view_state.image_axes is None:
            return 0
        shape = []
        for axis in view_state.image_axes:
            indices = view_state.axis_range_indices[axis]
            shape.append(len(indices) if indices is not None else view_state.shape[axis])
        dtypes = self.operation_coordinator.operation_dtype_estimates()
        dtype = dtypes[-1] if dtypes else getattr(self.document.base_data, "dtype", np.dtype(float))
        rgb = view_state.channel == ChannelMode.COMPLEX
        return estimate_display_image_bytes(tuple(shape), dtype, rgb=rgb, histogram=rgb)

    def _visible_render_budget_bytes(self) -> int:
        return int(self._memory_policy().visible_render_budget_bytes)

    def _montage_canvas_budget_bytes(self) -> int:
        return int(self._memory_policy().montage_canvas_budget_bytes)

    def _single_montage_tile_budget_bytes(self) -> int:
        return int(self._memory_policy().single_tile_budget_bytes)

    def _prefetch_budget_bytes(self) -> int:
        return int(self._memory_policy().prefetch_budget_bytes)

    def _memory_policy(self):
        policy = getattr(self, "_current_memory_policy", None)
        if policy is None:
            policy = self._refresh_memory_policy()
        return policy

    def _refresh_memory_policy(self, *, active_render: bool = False):
        current = compute_memory_policy(
            profile=getattr(getattr(self, "app_settings", None), "memory_profile", "balanced"),
            render_cap_mb=getattr(getattr(self, "app_settings", None), "render_memory_budget_mb", 512),
            input_nbytes=input_nbytes_for(getattr(self, "base_data", None)),
        )
        policy = apply_policy_hysteresis(
            getattr(self, "_current_memory_policy", None),
            current,
            active_render=bool(active_render),
        )
        self._current_memory_policy = policy
        self._apply_memory_policy_to_caches(policy)
        return policy

    def _apply_memory_policy_to_caches(self, policy) -> None:
        evaluator = getattr(self, "operation_evaluator", None)
        if evaluator is not None and hasattr(evaluator, "apply_memory_policy"):
            evaluator.apply_memory_policy(policy)

    def _montage_render_active(self) -> bool:
        session = getattr(self, "_montage_session", None)
        return bool(session is not None and (session.pending_tiles or session.loading_tiles))
    
