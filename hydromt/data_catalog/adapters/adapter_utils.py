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


def _check_time_overlap(
    pd_tstart: pd.Timestamp,
    pd_tstop: pd.Timestamp,
    data_tstart: pd.Timestamp,
    data_tstop: pd.Timestamp,
    handle_nodata: NoDataStrategy,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Check overlap and clamp the requested range to the data extent.

    Returns the clamped (true_tstart, true_tstop) or None if there is no overlap.
    """
    if pd_tstop < data_tstart or pd_tstart > data_tstop:
        exec_nodata_strat(
            f"Requested time range ({pd_tstart}, {pd_tstop}) has no overlap with available range '{data_tstart}' to '{data_tstop}'.",
            handle_nodata,
        )
        return None

    true_tstart = max(pd_tstart, data_tstart)
    true_tstop = min(pd_tstop, data_tstop)

    if true_tstart != pd_tstart or true_tstop != pd_tstop:
        logger.warning(
            f"Requested time range ({pd_tstart}, {pd_tstop}) partially overlaps with available range '{data_tstart}' to '{data_tstop}'. Clamping to ({true_tstart}, {true_tstop})."
        )

    return true_tstart, true_tstop


def _create_dataset_time_slice(
    ds: xr.Dataset,
    *,
    tstart: pd.Timestamp | datetime | str,
    tstop: pd.Timestamp | datetime | str,
    inclusive: bool = False,
    handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
) -> slice | None:
    """Create a time slice for an xr.Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to check the time extent against.
    tstart : pd.Timestamp | datetime | str
        The start time of the requested time range.
    tstop : pd.Timestamp | datetime | str
        The stop time of the requested time range.
    inclusive : bool, optional
        Whether to include the start and end times in the slice, by default False.
    handle_nodata : NoDataStrategy, optional
        The strategy to handle the case where no data is left after slicing, by default NoDataStrategy.RAISE.

    Returns
    -------
    slice | None
        A slice object for the time dimension, or None if no overlap exists.
    """
    pd_tstart = pd.Timestamp(tstart)
    pd_tstop = pd.Timestamp(tstop)

    time_values = pd.to_datetime(ds["time"].values)
    if not time_values.is_monotonic_increasing:
        logger.info("Dataset time dimension is not sorted; sorting for slicing.")
        ds = ds.sortby("time")
        time_values = pd.to_datetime(ds["time"].values)

    result = _check_time_overlap(
        pd_tstart, pd_tstop, time_values[0], time_values[-1], handle_nodata
    )
    if result is None:
        return None
    true_tstart, true_tstop = result

    if inclusive:
        minimum = ds.sel(time=true_tstart, method="pad")
        maximum = ds.sel(time=true_tstop, method="backfill")
    else:
        minimum = ds.sel(time=true_tstart, method="backfill")
        maximum = ds.sel(time=true_tstop, method="pad")
    return slice(minimum.time.values, maximum.time.values)


def _create_dataframe_time_slice(
    df: pd.DataFrame,
    *,
    tstart: pd.Timestamp | datetime | str,
    tstop: pd.Timestamp | datetime | str,
    inclusive: bool = False,
    handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
) -> slice | None:
    """Create a time slice for a pd.DataFrame with a DatetimeIndex.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to check the time extent against. Index must be a DatetimeIndex.
    tstart : pd.Timestamp | datetime | str
        The start time of the requested time range.
    tstop : pd.Timestamp | datetime | str
        The stop time of the requested time range.
    inclusive : bool, optional
        Whether to include the start and end times in the slice, by default False.
    handle_nodata : NoDataStrategy, optional
        The strategy to handle the case where no data is left after slicing, by default NoDataStrategy.RAISE.

    Returns
    -------
    slice | None
        A slice object for the DataFrame index, or None if no overlap exists.
    """
    pd_tstart = pd.Timestamp(tstart)
    pd_tstop = pd.Timestamp(tstop)

    time_index = df.index
    if not isinstance(time_index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    if not time_index.is_monotonic_increasing:
        logger.info("DataFrame DatetimeIndex is not sorted; sorting for slicing.")
        df = df.sort_index()
        time_index = df.index

    result = _check_time_overlap(
        pd_tstart, pd_tstop, time_index[0], time_index[-1], handle_nodata
    )
    if result is None:
        return None
    true_tstart, true_tstop = result

    if inclusive:
        # pad: last value <= tstart
        start_idx = time_index.searchsorted(true_tstart, side="right") - 1
        # backfill: first value >= tstop
        stop_idx = time_index.searchsorted(true_tstop, side="left")
    else:
        # backfill: first value >= tstart
        start_idx = time_index.searchsorted(true_tstart, side="left")
        # pad: last value <= tstop
        stop_idx = time_index.searchsorted(true_tstop, side="right") - 1
    start_idx = max(0, start_idx)
    stop_idx = min(len(time_index) - 1, stop_idx)
    return slice(time_index[start_idx], time_index[stop_idx])


def _create_time_slice(
    data: xr.Dataset | pd.DataFrame,
    *,
    tstart: pd.Timestamp | datetime | str,
    tstop: pd.Timestamp | datetime | str,
    inclusive: bool = False,
    handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
) -> slice | None:
    """Create a slice object for the time dimension of data that includes the requested time range.

    Dispatches to `_create_dataset_time_slice` or `_create_dataframe_time_slice`
    based on the type of `data`.
    """
    if isinstance(data, pd.DataFrame):
        return _create_dataframe_time_slice(
            data,
            tstart=tstart,
            tstop=tstop,
            inclusive=inclusive,
            handle_nodata=handle_nodata,
        )
    return _create_dataset_time_slice(
        data,
        tstart=tstart,
        tstop=tstop,
        inclusive=inclusive,
        handle_nodata=handle_nodata,
    )


def _slice_temporal_dimension(
    ds: xr.Dataset,
    time_range: TimeRange,
    handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
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
    """
    if (
        "time" in ds.dims
        and ds["time"].size > 1
        and np.issubdtype(ds["time"].dtype, np.datetime64)
    ):
        logger.debug(f"Slicing time dim {time_range}")
        time_slice = _create_time_slice(
            ds,
            tstart=time_range.start,
            tstop=time_range.end,
            inclusive=time_range.inclusive,
            handle_nodata=handle_nodata,
        )
        if time_slice is None:
            return None
        ds = ds.sel(time=time_slice)
    if _has_no_data(ds):
        exec_nodata_strat("No data left after temporal slicing.", handle_nodata)
        return None

    return ds
