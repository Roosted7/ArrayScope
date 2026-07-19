"""File-type catalogue shared by all desktop-integration backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FileType:
    """One associable family of files.

    ``owned`` means ArrayScope defines the MIME type itself (no widely
    deployed owner exists) and may claim to be the default handler. For
    types with an established owner (DICOM, HDF5) ArrayScope only adds
    itself as an "Open with" candidate.
    """

    key: str
    extensions: tuple
    mime: str
    description: str
    owned: bool


FILE_TYPES = (
    FileType("npy", (".npy",), "application/x-numpy-array", "NumPy array", True),
    FileType("npz", (".npz",), "application/x-numpy-archive", "NumPy array archive", True),
    FileType("cfl", (".cfl",), "application/x-bart-cfl", "BART complex-float array", True),
    FileType("rec", (".rec",), "application/x-philips-rec", "Philips XML/REC image", True),
    FileType("nii", (".nii", ".nii.gz"), "application/x-nifti", "NIfTI neuroimaging data", True),
    FileType("mat", (".mat",), "application/x-matlab-data", "MATLAB data file", True),
    FileType("dcm", (".dcm",), "application/dicom", "DICOM image", False),
    FileType("h5", (".h5", ".hdf5"), "application/x-hdf", "HDF5 data file", False),
)


def owned_types():
    return tuple(t for t in FILE_TYPES if t.owned)


def all_mime_types():
    return tuple(t.mime for t in FILE_TYPES)
