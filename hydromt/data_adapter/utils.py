"""Utility functions for data adapters."""

from logging import Logger
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from hydromt._typing import Data
from hydromt._typing.type_def import Variables


def shift_dataset_time(
    dt: int, ds: Data, logger: Logger, time_unit: str = "s"
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


def has_no_data(
    data: Optional[Union[pd.DataFrame, gpd.GeoDataFrame, xr.Dataset, xr.DataArray]],
) -> bool:
    """Check whether various data containers are empty."""
    if data is None:
        return True
    elif isinstance(data, xr.Dataset):
        return all([v.size == 0 for v in data.data_vars.values()])
    else:
        return len(data) == 0


def _single_var_as_array(
    maybe_ds: Optional[xr.Dataset],
    single_var_as_array: bool,
    variable_name: Optional[Variables] = None,
) -> Optional[xr.Dataset]:
    if maybe_ds is None:
        return None
    else:
        ds = maybe_ds
    # return data array if single variable dataset
    dvars = list(ds.data_vars.keys())
    if single_var_as_array and len(dvars) == 1:
        da = ds[dvars[0]]
        if isinstance(variable_name, list) and len(variable_name) == 1:
            da.name = variable_name[0]
        elif isinstance(variable_name, str):
            da.name = variable_name
        return da
    else:
        return ds
