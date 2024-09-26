"""Common routines for preprocessing."""

from typing import Callable, Dict

import numpy as np
import xarray as xr


def _round_latlon(ds: xr.Dataset, decimals: int = 5) -> xr.Dataset:
    """Round the x and y dimensions to latlon."""
    x_dim = ds.raster.x_dim
    y_dim = ds.raster.y_dim
    ds[x_dim] = np.round(ds[x_dim], decimals=decimals)
    ds[y_dim] = np.round(ds[y_dim], decimals=decimals)
    return ds


def _to_datetimeindex(ds: xr.Dataset) -> xr.Dataset:
    """Convert the 'time' index to datetimeindex."""
    if ds.indexes["time"].dtype == "O":
        ds["time"] = ds.indexes["time"].to_datetimeindex()
    return ds


def _remove_duplicates(ds: xr.Dataset) -> xr.Dataset:
    """Remove duplicates from the 'time' index."""
    return ds.sel(time=~ds.get_index("time").duplicated())


def _harmonise_dims(ds: xr.Dataset) -> xr.Dataset:
    """Harmonise lon-lat-time dimensions.

    Where needed:
        - lon: Convert longitude coordinates from 0-360 to -180-180
        - lat: Do N->S orientation instead of S->N
        - time: Convert to datetimeindex.

    Parameters
    ----------
    ds: xr.DataSet
        DataSet with dims to harmonise

    Returns
    -------
    ds: xr.DataSet
        DataSet with harmonised longitude-latitude-time dimensions
    """
    # Longitude
    x_dim = ds.raster.x_dim
    lons = ds[x_dim].values
    if np.any(lons > 180):
        ds[x_dim] = xr.Variable(x_dim, np.where(lons > 180, lons - 360, lons))
        ds = ds.sortby(x_dim)
    # Latitude
    y_dim = ds.raster.y_dim
    if np.diff(ds[y_dim].values)[0] > 0:
        ds = ds.reindex({y_dim: ds[y_dim][::-1]})
    # Final check for lat-lon
    assert (
        np.diff(ds[y_dim].values)[0] < 0
    ), "orientation not N->S after get_data preprocess set_lon_lat_axis"
    assert (
        np.diff(ds[x_dim].values)[0] > 0
    ), "orientation not W->E after get_data preprocess set_lon_lat_axis"
    # Time
    if ds.indexes["time"].dtype == "O":
        ds = _to_datetimeindex(ds)

    return ds


PREPROCESSORS: Dict[str, Callable] = {
    "round_latlon": _round_latlon,
    "to_datetimeindex": _to_datetimeindex,
    "remove_duplicates": _remove_duplicates,
    "harmonise_dims": _harmonise_dims,
}
