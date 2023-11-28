"""Utility functions for data adapters."""
import logging
import os
from os.path import isdir, join
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr


def netcdf_writer(
    obj: Union[xr.Dataset, xr.DataArray],
    data_root: Union[str, Path],
    data_name: str,
    variables: Optional[List[str]] = None,
    encoder: str = "zlib",
) -> str:
    """Utiliy function for writing a xarray dataset/data array to a netcdf file.

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        Dataset.
    data_root : str | Path
        root to write the data to.
    data_name : str
        filename to write to.
    variables : Optional[List[str]]
        list of dataset variables to write, by default None

    Returns
    -------
    fn_out: str
        Absolute path to output file
    """
    dvars = [obj.name] if isinstance(obj, xr.DataArray) else obj.data_vars
    if variables is None:
        encoding = {k: {encoder: True} for k in dvars}
        fn_out = join(data_root, f"{data_name}.nc")
        obj.to_netcdf(fn_out, encoding=encoding)
    else:  # save per variable
        if not isdir(join(data_root, data_name)):
            os.makedirs(join(data_root, data_name))
        for var in dvars:
            fn_out = join(data_root, data_name, f"{var}.nc")
            obj[var].to_netcdf(fn_out, encoding={var: {encoder: True}})
        fn_out = join(data_root, data_name, "{variable}.nc")
    return fn_out


def zarr_writer(
    obj: Union[xr.Dataset, xr.DataArray],
    data_root: Union[str, Path],
    data_name: str,
    **kwargs,
) -> str:
    """Utiliy function for writing a xarray dataset/data array to a netcdf file.

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        Dataset.
    data_root : str | Path
        root to write the data to.
    data_name : str
        filename to write to.

    Returns
    -------
    fn_out: str
        Absolute path to output file
    """
    fn_out = join(data_root, f"{data_name}.zarr")
    if isinstance(obj, xr.DataArray):
        obj = obj.to_dataset()
    obj.to_zarr(fn_out, **kwargs)
    return fn_out


def shift_dataset_time(
    dt: int, ds: xr.Dataset, logger: logging.Logger, time_unit: str = "s"
) -> xr.Dataset:
    """Shifts time of a xarray dataset.

    Parameters
    ----------
    dt : int
        time delta to shift the time of the dataset
    ds : xr.Dataset
        xarray dataset
    logger : logging.Logger
        logger

    Returns
    -------
    xr.Dataset
        time shifted dataset
    """
    if (
        dt != 0
        and "time" in ds.dims
        and ds["time"].size > 1
        and np.issubdtype(ds["time"].dtype, np.datetime64)
    ):
        logger.debug(f"Shifting time labels with {dt} {time_unit}.")
        ds["time"] = ds["time"] + pd.to_timedelta(dt, unit=time_unit)
    elif dt != 0:
        logger.warning("Time shift not applied, time dimension not found.")
    return ds
