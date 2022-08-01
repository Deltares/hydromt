import numpy as np
import pandas as pd
import xarray as xr
import logging

logger = logging.getLogger(__name__)

__all__ = ["grid_maptable", "vector_to_grid"]


RESAMPLING = {"grid_maptable": "nearest"}
DTYPES = {"grid_maptable": np.int16}


def grid_maptable(da, ds_like, fn_map, logger=logger, params=None):
    """Returns RasterData map and related parameter maps.
    The parameter maps are prepared based on the initial raster map and
    mapping table as provided in fn_map.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray containing classification.
    ds_like : xarray.DataArray
        Dataset at model resolution.
    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing parameters maps at model resolution and region
    """
    # read csv with remapping values
    df = pd.read_csv(
        fn_map, index_col=0, sep=",|;", engine="python", dtype=DTYPES
    )  # TODO make dtype flexible
    # limit dtypes to avoid gdal errors downstream
    ddict = {"float64": np.float32, "int64": np.int32}
    dtypes = {c: ddict.get(str(df[c].dtype), df[c].dtype) for c in df.columns}
    df = pd.read_csv(fn_map, index_col=0, sep=",|;", engine="python", dtype=dtypes)
    keys = df.index.values
    if params is None:
        params = [p for p in df.columns if p != "description"]
    elif not np.all(np.isin(params, df.columns)):
        missing = [p for p in params if p not in df.columns]
        raise ValueError(f"Parameter(s) missing in mapping file: {missing}")
    # setup ds out
    ds_out = xr.Dataset(coords=ds_like.raster.coords)
    # setup reclass method
    def reclass(x):
        return np.vectorize(d.get)(x, nodata)

    da = da.raster.interpolate_na(method="nearest")  # TODO: make this flexible
    # apply for each parameter
    for param in params:
        method = RESAMPLING.get(param, "average")  # TODO: make this flexible
        values = df[param].values
        nodata = values[-1]  # NOTE values is set in last row
        d = dict(zip(keys, values))  # NOTE global param in reclass method
        logger.info(f"Deriving {param} using {method} resampling (nodata={nodata}).")
        da_param = xr.apply_ufunc(
            reclass, da, dask="parallelized", output_dtypes=[values.dtype]
        )
        da_param.attrs.update(_FillValue=nodata)  # first set new nodata values
        ds_out[param] = da_param.raster.reproject_like(
            ds_like, method=method
        )  # then resample

    return ds_out


def vector_to_grid(
    gdf,
    ds_like,
    col_name="",
    method="value",
    mask_name="msk",
    logger=logger,
):
    """Returns gridded data from vector.
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing data.
    ds_like : xarray.DataArray
        Dataset at model resolution.
    method : str {'value', 'fraction'}
        Method for rasterizing.
    mask_name : str
        Name of a mask array in ds_like.
    Returns
    -------
    da_out : xarray.Dataarray
        Dataarray containing gridded map at model resolution over region.
    """
    if method == "value":
        da_out = ds_like.raster.rasterize(
            gdf,
            col_name=col_name,
            nodata=0,
            all_touched=True,
            dtype=None,
            sindex=False,
        )
    elif method == "fraction":
        # Using the vectors directly (long computation time but most accurate)
        # Create vector grid (for calculating fraction and storage per grid cell)
        logger.debug(
            "Creating vector grid for calculating coverage fraction per grid cell"
        )
        gdf["geometry"] = gdf.geometry.buffer(0)  # fix potential geometry errors
        msktn = ds_like[mask_name]
        idx_valid = np.where(msktn.values.flatten() != msktn.raster.nodata)[0]
        gdf_grid = ds_like.raster.vector_grid().loc[idx_valid]
        gdf_grid["coverfrac"] = np.zeros(len(idx_valid))
        gdf_grid["area"] = gdf_grid.to_crs(
            3857
        ).area  # area calculation in projected crs   #TODO: ask why 3857 and not mod.crs

        # Calculate fraction per (vector) grid cell
        # Looping over each vector shape
        for i in range(len(gdf)):
            shape = gdf.iloc[i]
            gridded_shape = gdf_grid.intersection(shape.geometry)
            gridded_shape = gridded_shape.loc[~gridded_shape.is_empty]
            idxs = gridded_shape.index
            if np.any(idxs):
                # area calculation needs projected crs
                sharea_cell = gridded_shape.to_crs(3857).area
                gdf_grid.loc[idxs, "coverfrac"] += (
                    sharea_cell / gdf_grid.loc[idxs, "area"]
                )
        # Create the rasterized coverage fraction map
        da_out = ds_like.raster.rasterize(
            gdf_grid,
            col_name="coverfrac",
            nodata=0,
            all_touched=False,
            dtype=None,
            sindex=False,
        )

    return da_out
