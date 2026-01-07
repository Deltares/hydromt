"""Utility functions for data catalog adapters."""

import logging
from typing import Optional

import numpy as np
import xarray as xr

from hydromt._utils import (
    _has_no_data,
)
from hydromt.error import NoDataStrategy, exec_nodata_strat
from hydromt.typing import (
    TimeRange,
)

logger = logging.getLogger(__name__)


def _slice_temporal_dimension(
    ds: xr.Dataset,
    time_range: TimeRange,
    handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
) -> Optional[xr.Dataset]:
    time_dims = ["time", "valid_time"]
    for time_dim in time_dims:
        if (
            time_dim in ds.dims
            and ds[time_dim].size > 1
            and np.issubdtype(ds[time_dim].dtype, np.datetime64)
        ):
            logger.debug(f"Slicing time dim {time_dim} by {time_range}")
            ds = ds.sel({time_dim: slice(time_range.start, time_range.end)})
            if _has_no_data(ds):
                exec_nodata_strat("No data left after temporal slicing.", handle_nodata)
                return None
    return ds
