import typing
from logging import getLogger
from typing import Dict, Optional, cast

import numpy as np
import pandas as pd
import xarray as xr

if typing.TYPE_CHECKING:
    from pandas._libs.tslibs.timedeltas import TimeDeltaUnitChoices

from hydromt._typing.metadata import SourceMetadata
from hydromt._typing.type_def import TimeRange, Variables
from hydromt._utils.nodata import _has_no_data

logger = getLogger(__name__)

__all__ = [
    "_set_metadata",
    "_shift_dataset_time",
    "_slice_temporal_dimension",
    "_rename_vars",
]


def _shift_dataset_time(
    dt: int,
    ds: Optional[xr.Dataset],
    time_unit: "TimeDeltaUnitChoices" = "s",
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


def _slice_temporal_dimension(
    ds: Optional[xr.Dataset], time_range: TimeRange
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
        if _has_no_data(ds):
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

    ds.attrs.update(metadata.model_dump(exclude_unset=True, exclude={"attrs"}))
    return ds


def _rename_vars(
    ds: Optional[xr.Dataset], rename: Dict[str, str]
) -> Optional[xr.Dataset]:
    if ds is None:
        return None
    rm = {k: v for k, v in rename.items() if k in ds}
    ds = ds.rename(rm)
    return ds


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
