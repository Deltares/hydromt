"""Utility functions for data adapters."""

from logging import Logger, getLogger
from typing import Dict, Optional, Union, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from hydromt._typing import SourceMetadata, TimeRange, Variables

logger = getLogger(__name__)


def shift_dataset_time(
    dt: int, ds: Optional[xr.Dataset], logger: Logger, time_unit: str = "s"
) -> Optional[xr.Dataset]:
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
    if ds is None:
        return None

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
    if isinstance(maybe_ds, xr.DataArray):
        return maybe_ds
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


def _set_vector_nodata(
    ds: Optional[xr.Dataset], metadata: "SourceMetadata"
) -> Optional[xr.Dataset]:
    if ds is None:
        return None
    if metadata.nodata is not None:
        if not isinstance(metadata.nodata, dict):
            nodata = {k: metadata.nodata for k in ds.data_vars.keys()}
        else:
            nodata = metadata.nodata
        for k in ds.data_vars:
            mv = nodata.get(k, None)
            if mv is not None and ds[k].vector.nodata is None:
                ds[k].vector.set_nodata(mv)
    return ds


def _set_raster_nodata(
    ds: Optional[xr.Dataset], metadata: "SourceMetadata"
) -> Optional[xr.Dataset]:
    if ds is None:
        return None
    if metadata.nodata is not None:
        if not isinstance(metadata.nodata, dict):
            nodata = {k: metadata.nodata for k in ds.data_vars.keys()}
        else:
            nodata = metadata.nodata
        for k in ds.data_vars:
            mv = nodata.get(k, None)
            if mv is not None and ds[k].raster.nodata is None:
                ds[k].raster.set_nodata(mv)
    return ds


def _slice_temporal_dimension(
    ds: Optional[xr.Dataset],
    time_range: Optional[TimeRange],
    logger: Logger = logger,
    # TODO: https://github.com/Deltares/hydromt/issues/802
) -> Optional[xr.Dataset]:
    if ds is None:
        return None
    else:
        if (
            "time" in ds.dims
            and ds["time"].size > 1
            and np.issubdtype(ds["time"].dtype, np.datetime64)
        ):
            logger.debug(f"Slicing time dim {time_range}")
            ds = ds.sel(time=slice(*time_range))
        if has_no_data(ds):
            return None
        else:
            return ds


def _set_metadata(
    ds: Optional[xr.Dataset], metadata: "SourceMetadata"
) -> Optional[xr.Dataset]:
    if ds is None:
        return None
    elif metadata.attrs:
        if isinstance(ds, xr.DataArray):
            name = cast(str, ds.name)
            ds.attrs.update(metadata.attrs[name])
        else:
            for k in metadata.attrs:
                ds[k].attrs.update(metadata.attrs[k])

    ds.attrs.update(metadata)
    return ds


def _rename_vars(
    ds: Optional[xr.Dataset], rename: Dict[str, str]
) -> Optional[xr.Dataset]:
    if ds is None:
        return None
    rm = {k: v for k, v in rename.items() if k in ds}
    ds = ds.rename(rm)
    return ds
