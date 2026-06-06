"""Pure helpers for saving displayed arrays as NumPy files."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def default_numpy_filename(source_filepath=None):
    if source_filepath is None:
        return "arrayscope.npy"

    source_path = Path(source_filepath)
    source_name = source_path.name
    if source_name.lower().endswith(".nii.gz"):
        return source_name[:-7] + ".npy"
    return f"{source_path.stem}.npy"


def selected_numpy_data(data, ranges, *, squeeze=True):
    sliced_data = data[tuple(slice(start, stop) for start, stop in ranges)]
    return np.squeeze(sliced_data) if squeeze else sliced_data


def estimate_nbytes(shape, dtype):
    return int(np.prod(tuple(int(size) for size in shape), dtype=np.int64)) * np.dtype(dtype).itemsize


def save_derived_array(path, data, *, recipe_json=None, view_recipe_json=None, sidecar=True):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        payload = {"array": data}
        if recipe_json is not None:
            payload["recipe_json"] = np.array(recipe_json)
        if view_recipe_json is not None:
            payload["view_recipe_json"] = np.array(view_recipe_json)
        np.savez(path, **payload)
        return (path,)

    if suffix != ".npy":
        path = path.with_suffix(".npy")
    np.save(path, data)
    written = [path]
    if sidecar and recipe_json is not None:
        recipe_path = path.with_suffix(".recipe.json")
        recipe_path.write_text(recipe_json.rstrip() + "\n", encoding="utf-8")
        written.append(recipe_path)
    if sidecar and view_recipe_json is not None:
        view_path = path.with_suffix(".view.json")
        view_path.write_text(view_recipe_json.rstrip() + "\n", encoding="utf-8")
        written.append(view_path)
    return tuple(written)
