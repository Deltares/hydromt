#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Small utility to create vrt files."""
from __future__ import annotations

import glob
import os
from os.path import dirname

from hydromt import _compat

if _compat.HAS_RIO_VRT:
    import rio_vrt


def create_vrt(
    vrt_path: str,
    files: list = None,
    files_path: str = None,
):
    r"""Create a .vrt file from a list op raster datasets.

    Either a list of files (`files`) or a path containing wildcards
    (`files_path`) to infer the list of files is required.

    Parameters
    ----------
    vrt_path : str
        Path of the output vrt
    files : list, optional
        List of raster datasets filenames, by default None
    files_path : str, optional
        Unix style path containing a pattern using wildcards (*)
        n.b. this is without an extension
        e.g. c:\\temp\\*\\*.tif for all tif files in subfolders of 'c:\temp'
    """
    if not _compat.HAS_RIO_VRT:
        raise ImportError(
            "rio-vrt is required for execution, install with 'pip install rio-vrt'"
        )

    if files is None and files_path is None:
        raise ValueError("Either 'files' or 'files_path' is required")

    if files is None and files_path is not None:
        files = glob.glob(files_path)
        if len(files) == 0:
            raise IOError(f"No files found at {files_path}")

    outdir = dirname(vrt_path)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    rio_vrt.build_vrt(vrt_path, files=files, relative=True)
    return None
