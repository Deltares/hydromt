"""Utility functions for data adapters."""
import os
from os.path import isdir, join
from pathlib import Path
from typing import List, Optional, Tuple

import xarray as xr


def netcdf_writer(
    obj: xr.Dataset | xr.DataArray,
    data_root: str | Path,
    data_name: str,
    variables: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """Utiliy function for writing a xarray dataset/data array to file.

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        Dataset.
    data_root : str | Path
        root to write the data to.
    data_name : str
        filename to write to.
    variables : Optional[List[str]], optional
        list of dataset variables to write, by default None

    Returns
    -------
    fn_out: str
        Absolute path to output file
    driver: str
        Name of driver to read data with
    """
    driver = "netcdf"
    dvars = [obj.name] if isinstance(obj, xr.DataArray) else obj.data_vars
    if variables is None:
        encoding = {k: {"zlib": True} for k in dvars}
        fn_out = join(data_root, f"{data_name}.nc")
        obj.to_netcdf(fn_out, encoding=encoding)
    else:  # save per variable
        if not isdir(join(data_root, data_name)):
            os.makedirs(join(data_root, data_name))
        for var in dvars:
            fn_out = join(data_root, data_name, f"{var}.nc")
            obj[var].to_netcdf(fn_out, encoding={var: {"zlib": True}})
        fn_out = join(data_root, data_name, "{variable}.nc")
    return fn_out, driver
