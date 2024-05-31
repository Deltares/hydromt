from typing import Optional, Union

import geopandas as gpd
import pandas as pd
import xarray as xr

from hydromt._typing import SourceMetadata

__all__ = ["_has_no_data", "_set_vector_nodata", "_set_raster_nodata"]


def _has_no_data(
    data: Optional[Union[pd.DataFrame, gpd.GeoDataFrame, xr.Dataset, xr.DataArray]],
) -> bool:
    """Check whether various data containers are empty."""
    if data is None:
        return True
    elif isinstance(data, xr.Dataset):
        return all([v.size == 0 for v in data.data_vars.values()])
    else:
        return len(data) == 0


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
