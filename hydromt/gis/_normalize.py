import pandas as pd
import xarray as xr


def normalize_xarray_dtypes(obj: xr.Dataset | xr.DataArray):
    """
    Normalize xarray objects by removing pandas extension dtypes.

    Ensures compatibility with dask, netCDF, and Zarr.
    """
    if isinstance(obj, xr.DataArray):
        if isinstance(obj.dtype, pd.StringDtype):
            obj = obj.astype(object)
        for name, coord in obj.coords.items():
            if isinstance(coord.dtype, pd.StringDtype):
                obj = obj.assign_coords({name: coord.astype(object)})
        return obj

    for name, var in obj.variables.items():
        if isinstance(var.dtype, pd.StringDtype):
            obj[name] = var.astype(object)

    return obj
