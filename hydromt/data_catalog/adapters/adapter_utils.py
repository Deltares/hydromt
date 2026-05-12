"""Utility functions for data catalog adapters."""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from hydromt._utils import (
    _has_no_data,
)
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.typing import (
    TimeRange,
)

logger = logging.getLogger(__name__)


def _create_time_slice(
    data: xr.Dataset | pd.DataFrame,
    tstart: pd.Timestamp | datetime | str,
    tstop: pd.Timestamp | datetime | str,
    inclusive: bool = True,
    handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
) -> slice | None:
    """Create a slice object for the time dimension of data that includes the requested time range.

    Parameters
    ----------
    data : xr.Dataset | pd.DataFrame
        The dataset or dataframe to check the time extent against.
        For a DataFrame, the index must be a DatetimeIndex.
    tstart : pd.Timestamp | datetime | str
        The start time of the requested time range.
    tstop : pd.Timestamp | datetime | str
        The stop time of the requested time range.
    inclusive : bool, optional
        Whether to include the start and end times in the slice, by default True
    handle_nodata : NoDataStrategy, optional
        The strategy to handle the case where no data is left after slicing, by default NoDataStrategy.RAISE

    Returns
    -------
    slice | None
        A slice object that can be used to index the time dimension of data.
        If no data is left after slicing, the behavior depends on handle_nodata:
        - If handle_nodata is NoDataStrategy.RAISE, a NoDataException is raised.
        - If handle_nodata is NoDataStrategy.IGNORE | NoDataStrategy.WARN, None is returned.
    """
    pd_tstart = pd.Timestamp(tstart)
    pd_tstop = pd.Timestamp(tstop)

    if isinstance(data, pd.DataFrame):
        time_index = data.index
        if not isinstance(time_index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")
        data_tstart = time_index[0]
        data_tstop = time_index[-1]
    else:
        timevar = data["time"]
        xr_tstartstop = pd.to_datetime(timevar.isel(time=[0, -1]).to_series())
        data_tstart = xr_tstartstop.index[0]
        data_tstop = xr_tstartstop.index[-1]

    if pd_tstart < data_tstart:
        exec_nodata_strat(
            f"Requested tstart {pd_tstart} outside of available range '{data_tstart}' to '{data_tstop}'.",
            handle_nodata,
        )
        return None
    if pd_tstop > data_tstop:
        exec_nodata_strat(
            f"Requested tstop {pd_tstop} outside of available range '{data_tstart}' to '{data_tstop}'.",
            handle_nodata,
        )
        return None

    if isinstance(data, pd.DataFrame):
        time_index = data.index
        if inclusive:
            # pad: last value <= tstart
            start_idx = time_index.searchsorted(pd_tstart, side="right") - 1
            # backfill: first value >= tstop
            stop_idx = time_index.searchsorted(pd_tstop, side="left")
        else:
            # backfill: first value >= tstart
            start_idx = time_index.searchsorted(pd_tstart, side="left")
            # pad: last value <= tstop
            stop_idx = time_index.searchsorted(pd_tstop, side="right") - 1
        start_idx = max(0, start_idx)
        stop_idx = min(len(time_index) - 1, stop_idx)
        return slice(time_index[start_idx], time_index[stop_idx])
    else:
        if inclusive:
            minimum = data.sel(time=pd_tstart, method="pad")
            maximum = data.sel(time=pd_tstop, method="backfill")
        else:
            minimum = data.sel(time=pd_tstart, method="backfill")
            maximum = data.sel(time=pd_tstop, method="pad")
        return slice(minimum.time.values, maximum.time.values)


def _slice_temporal_dimension(
    ds: xr.Dataset,
    time_range: TimeRange,
    handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    inclusive: bool = True,
) -> xr.Dataset | None:
    """Slice the time dimension of the dataset to the requested time range, if it exists.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to slice.
    time_range : TimeRange
        The time range to slice to.
    handle_nodata : NoDataStrategy, optional
        The strategy to handle the case where no data is left after slicing, by default NoDataStrategy.RAISE
    inclusive : bool, optional
        Whether to include the start and end times in the slice, by default True
    """
    if (
        "time" in ds.dims
        and ds["time"].size > 1
        and np.issubdtype(ds["time"].dtype, np.datetime64)
    ):
        logger.debug(f"Slicing time dim {time_range}")
        time_slice = _create_time_slice(
            ds,
            time_range.start,
            time_range.end,
            inclusive=inclusive,
            handle_nodata=handle_nodata,
        )
        if time_slice is None:
            return None
        ds = ds.sel(time=time_slice)
    if _has_no_data(ds):
        exec_nodata_strat("No data left after temporal slicing.", handle_nodata)
        return None

    return ds
