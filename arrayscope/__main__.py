#!/usr/bin/env python3
"""
Command-line interface for arrayscope.
"""
import argparse
import numpy as np
from pathlib import Path
from arrayscope.app.launch import arrayscope
from arrayscope.io.selectors import H5DatasetSelector, NpzDatasetSelector, MatDatasetSelector
from arrayscope.io.file_interpreters import data_file_suffix, load_path


_CLI_WINDOWS = []


def _open_array_window(*, data, title, block, filepath, dataset_path=None, selector_class_name=None):
    if block:
        return arrayscope(
            data=data,
            title=title,
            block=True,
            filepath=filepath,
            dataset_path=dataset_path,
            selector_class_name=selector_class_name,
        )
    from arrayscope.app.launch import _create_window

    _app, win = _create_window(
        data,
        title=title,
        filepath=filepath,
        dataset_path=dataset_path,
        selector_class_name=selector_class_name,
    )
    _CLI_WINDOWS.append(win)
    return win


def _run_cli_event_loop():
    import pyqtgraph as pg

    return pg.mkQApp().exec()


def _selector_for_suffix(filepath, suffix):
    if suffix in ['.h5', '.hdf5']:
        return H5DatasetSelector(filepath)
    if suffix == '.npz':
        return NpzDatasetSelector(filepath)
    if suffix == '.mat':
        return MatDatasetSelector(filepath)
    return None


def _open_loaded_file(filepath: Path, *, block: bool) -> bool:
    loaded = load_path(filepath)
    title = filepath.name or str(filepath)
    detected_format = loaded.metadata.get('detected_format')
    if detected_format:
        title = f"{title} [{detected_format}]"
    _open_array_window(data=loaded.data, title=title, block=block, filepath=filepath)
    return not block


def _open_selector_file(filepath: Path, selector, *, block: bool) -> bool:
    if block:
        if not selector.view(block=True):
            print(f"No compatible datasets found in {filepath}")
        return False
    if not selector.requires_gui():
        result = selector.get_single_data()
        if not result:
            selector.close()
            print(f"No compatible datasets found in {filepath}")
            return False
        name, data = result
        selector.close()
        _open_array_window(
            data=data,
            title=f"{filepath.name} - {name}",
            block=False,
            filepath=filepath,
            dataset_path=name,
            selector_class_name=selector.__class__.__name__,
        )
        return True
    if not selector.view(block=False):
        print(f"No compatible datasets found in {filepath}")
    return False


def main():
    parser = argparse.ArgumentParser(
        prog='arrayscope',
        description='Interactive N-dimensional array viewer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  arrayscope data.npy                      # View single file
  arrayscope data.h5 data2.npy data3.npz   # View multiple files
  arrayscope scan.REC                      # View Philips REC/XML pair
  arrayscope ref.cfl                       # View BART CFL/HDR pair
    arrayscope dicomdir/                     # Convert DICOM directory via dcm2niix, then view
  arrayscope scan.dcm                      # View DICOM file
  arrayscope scan.nii                      # View NIfTI file
  arrayscope data.txt                      # View text file with numeric data
  
For files with multiple datasets (HDF5, NPZ, MAT), a GUI selector will automatically appear.
        """
    )
    parser.add_argument('files', type=str, nargs='+', 
                        help='Path(s) to data files or DICOM directories')
    
    args = parser.parse_args()
    
    block_each = len(args.files) == 1
    needs_event_loop = False

    for file_arg in args.files:
        filepath = Path(file_arg)
        
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            continue
        
        try:
            suffix = data_file_suffix(filepath)
            # Single-dataset formats and DICOM directories are handled by file_interpreters.load_path
            if filepath.is_dir() or suffix in ['.npy', '.rec', '.cfl', '.dcm', '.nii', '.nii.gz', '.txt']:
                needs_event_loop = _open_loaded_file(filepath, block=block_each) or needs_event_loop
                continue
            
            # Multi-dataset formats - use selectors
            selector = _selector_for_suffix(filepath, suffix)
            if selector is None:
                print(f"Unsupported file type: {suffix}. Supported types: directories with DICOM .dcm files, .h5, .hdf5, .npy, .npz, .mat, .REC, .cfl, .dcm, .nii, .nii.gz, .txt")
                continue
            
            # Select and view dataset (shows GUI if multiple datasets)
            needs_event_loop = _open_selector_file(filepath, selector, block=block_each) or needs_event_loop
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not block_each and needs_event_loop:
        _run_cli_event_loop()


if __name__ == '__main__':
    main()
