"""Immutable HDF5 metadata shared by browsing and scientific lazy reads."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from pathlib import Path
from stat import S_ISREG

import numpy as np


class HDF5SourceChanged(RuntimeError):
    """The file or dataset no longer matches its captured metadata."""


def file_identity(path):
    stat = Path(path).stat()
    if not S_ISREG(stat.st_mode):
        raise ValueError("HDF5 input must be a regular file")
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def dataset_identity(dataset):
    """Describe storage and layout without accessing any dataset payload."""
    import h5py

    plist = dataset.id.get_create_plist()
    filters = tuple(plist.get_filter(index) for index in range(plist.get_nfilters()))
    return (
        dataset.shape,
        dataset.dtype.str,
        repr(dataset.dtype.descr),
        dataset.chunks,
        dataset.maxshape,
        dataset.compression,
        repr(dataset.compression_opts),
        dataset.scaleoffset,
        dataset.shuffle,
        dataset.fletcher32,
        tuple(dataset.external or ()),
        bool(dataset.is_virtual),
        int(h5py.h5o.get_info(dataset.id).addr),
        filters,
    )


@dataclass(frozen=True)
class HDF5DatasetDescriptor:
    key: str
    dataset_path: str
    shape: tuple[int, ...] | None
    dtype: np.dtype
    storage_shape: tuple[int, ...] | None
    storage_dtype: np.dtype
    encoding: str
    field_name: str | None
    real_field: str | None
    imag_field: str | None
    chunk_shape: tuple[int, ...] | None
    compression: str | None
    filter_ids: tuple[int, ...]
    lazy_supported: bool
    eager_supported: bool
    unsupported_reason: str
    file_identity: tuple
    dataset_identity: tuple

    @property
    def nbytes(self):
        return None if self.shape is None else prod(self.shape) * self.dtype.itemsize


@dataclass(frozen=True)
class HDF5Catalog:
    path: Path
    file_identity: tuple
    groups: tuple[str, ...]
    datasets: tuple[HDF5DatasetDescriptor, ...]

    def dataset(self, key):
        key = str(key).lstrip("/")
        for descriptor in self.datasets:
            if descriptor.key == key:
                return descriptor
        raise ValueError(f"HDF5 dataset {key!r} was not found")


def _descriptors(dataset, key, identity):
    shape = dataset.shape
    dtype = dataset.dtype
    signature = dataset_identity(dataset)
    reason = ""
    if dataset.is_virtual:
        reason = "Virtual HDF5 datasets are not supported"
    elif dataset.external:
        reason = "HDF5 datasets with external storage are not supported"
    elif shape is None:
        reason = "Null HDF5 datasets have no array shape"
    base = HDF5DatasetDescriptor(
        key=key,
        dataset_path="/" + key,
        shape=shape,
        dtype=dtype,
        storage_shape=shape,
        storage_dtype=dtype,
        encoding="plain" if not reason else "unsupported",
        field_name=None,
        real_field=None,
        imag_field=None,
        chunk_shape=dataset.chunks,
        compression=dataset.compression,
        filter_ids=tuple(item[0] for item in signature[-1]),
        lazy_supported=not reason and dtype.kind in "biufc",
        eager_supported=not reason and dtype.kind in "biufc",
        unsupported_reason=reason or ("" if dtype.kind in "biufc" else "Not a numeric array"),
        file_identity=identity,
        dataset_identity=signature,
    )
    if reason or not dtype.names:
        return (base,)

    real = next((name for name in ("real", "realdata", "r") if name in dtype.names), None)
    imag = next((name for name in ("imag", "imagdata", "i") if name in dtype.names), None)
    if real is not None and imag is not None:
        real_dtype, imag_dtype = dtype.fields[real][0], dtype.fields[imag][0]
        if all(item.base.kind in "biufc" for item in (real_dtype, imag_dtype)):
            try:
                complex_shape = np.broadcast_shapes(
                    shape + real_dtype.shape, shape + imag_dtype.shape
                )
            except ValueError:
                pass
            else:
                complex_dtype = (
                    np.empty(0, dtype=real_dtype.base) + 1j * np.empty(0, dtype=imag_dtype.base)
                ).dtype
                subarray = real_dtype.subdtype is not None or imag_dtype.subdtype is not None
                base = replace(
                    base,
                    shape=complex_shape,
                    dtype=complex_dtype,
                    encoding="complex",
                    real_field=real,
                    imag_field=imag,
                    lazy_supported=not subarray,
                    eager_supported=True,
                    unsupported_reason=(
                        "Open this compound complex subarray using Load all in the HDF5 dataset browser"
                        if subarray
                        else ""
                    ),
                )
    fields = []
    for name in dtype.names:
        field_dtype = dtype.fields[name][0]
        subarray = field_dtype.subdtype is not None
        field_shape = shape
        if subarray:
            field_shape = (() if shape == (1,) else shape) + field_dtype.shape
        numeric = field_dtype.base.kind in "biufc"
        fields.append(
            replace(
                base,
                key=f"{key}/{name}",
                shape=field_shape,
                dtype=field_dtype.base,
                encoding="field",
                field_name=name,
                real_field=None,
                imag_field=None,
                lazy_supported=numeric and not subarray,
                eager_supported=numeric,
                unsupported_reason=(
                    "Open this compound subarray field using Load all in the HDF5 dataset browser"
                    if subarray and numeric
                    else ""
                    if numeric
                    else "Not a numeric array"
                ),
            )
        )
    return (base, *fields)


def _resolve_selection(handle, key):
    """Resolve only local hard links, stopping before any compound field tail."""
    import h5py

    parts = str(key).lstrip("/").split("/")
    current = handle
    for position, name in enumerate(parts):
        if not name or not isinstance(current, h5py.Group):
            raise ValueError(f"HDF5 dataset {key!r} was not found")
        link = current.get(name, getlink=True)
        if link is None:
            raise ValueError(f"HDF5 dataset {key!r} was not found")
        if not isinstance(link, h5py.HardLink):
            kind = "External" if isinstance(link, h5py.ExternalLink) else "Soft"
            raise ValueError(f"{kind} HDF5 links are not supported")
        current = current[name]
        if isinstance(current, h5py.Dataset):
            dataset_key = "/".join(parts[: position + 1])
            field = "/".join(parts[position + 1 :]) or None
            return current, dataset_key, field
    raise ValueError(f"HDF5 path {key!r} names a group, not a dataset")


def describe_hdf5_dataset(path, key):
    """Describe one explicit selection without scanning unrelated metadata."""
    import h5py

    path = Path(path).expanduser().absolute()
    identity = file_identity(path)
    with h5py.File(path, "r") as handle:
        dataset, dataset_key, field = _resolve_selection(handle, key)
        candidates = _descriptors(dataset, dataset_key, identity)
        descriptor = next((item for item in candidates if item.field_name == field), None)
        if descriptor is None:
            raise ValueError(f"HDF5 dataset or field {key!r} was not found")
    if file_identity(path) != identity:
        raise HDF5SourceChanged("HDF5 source changed while reading metadata; open it again")
    return descriptor


def catalog_hdf5(path):
    """List local hard-linked metadata; never traverse soft or external links."""
    import h5py

    path = Path(path).expanduser().absolute()
    identity = file_identity(path)
    groups = []
    datasets = []
    visited_groups = set()
    with h5py.File(path, "r") as handle:

        def visit(group, prefix=""):
            address = int(h5py.h5o.get_info(group.id).addr)
            if address in visited_groups:
                return
            visited_groups.add(address)
            for name in sorted(group):
                key = f"{prefix}/{name}" if prefix else name
                link = group.get(name, getlink=True)
                if not isinstance(link, h5py.HardLink):
                    datasets.append(
                        HDF5DatasetDescriptor(
                            key,
                            "/" + key,
                            None,
                            np.dtype("O"),
                            None,
                            np.dtype("O"),
                            "unsupported",
                            None,
                            None,
                            None,
                            None,
                            None,
                            (),
                            False,
                            False,
                            "External HDF5 links are not supported"
                            if isinstance(link, h5py.ExternalLink)
                            else "Soft HDF5 links are not supported",
                            identity,
                            (),
                        )
                    )
                    continue
                obj = group[name]
                if isinstance(obj, h5py.Group):
                    groups.append(key)
                    visit(obj, key)
                elif isinstance(obj, h5py.Dataset):
                    datasets.extend(_descriptors(obj, key, identity))

        visit(handle)
    if file_identity(path) != identity:
        raise HDF5SourceChanged("HDF5 source changed while reading its catalog; open it again")
    return HDF5Catalog(path, identity, tuple(groups), tuple(datasets))
