"""Implementation for grid based workflows."""
import logging
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Polygon

import hydromt

logger = logging.getLogger(__name__)

__all__ = [
    "grid_from_constant",
    "grid_from_rasterdataset",
    "grid_from_raster_reclass",
    "grid_from_geodataframe",
    "rotated_grid",
]


def grid_from_constant(
    grid_like: Union[xr.DataArray, xr.Dataset],
    constant: Union[int, float],
    name: str,
    dtype: Optional[str] = "float32",
    nodata: Optional[Union[int, float]] = None,
    mask_name: Optional[str] = "mask",
) -> xr.DataArray:
    """Prepare a grid based on a constant value.

    Parameters
    ----------
    grid_like: xr.DataArray, xr.Dataset
        Grid to copy metadata from.
    constant: int, float
        Constant value to fill grid with.
    name: str
        Name of grid.
    dtype: str, optional
        Data type of grid. By default 'float32'.
    nodata: int, float, optional
        Nodata value. By default infered from dtype.
    mask_name: str, optional
        Name of mask in self.grid to use for masking raster_fn. By default 'mask'.
        Use None to disable masking.

    Returns
    -------
    da: xr.DataArray
        Grid with constant value.
    """
    da = hydromt.raster.full(
        coords=grid_like.raster.coords,
        nodata=constant,
        dtype=dtype,
        name=name,
        attrs={},
        crs=grid_like.raster.crs,
        lazy=False,
    )
    # Set nodata value
    da.raster.set_nodata(nodata)
    # Masking
    if mask_name is not None:
        if mask_name in grid_like:
            da = da.raster.mask(grid_like[mask_name])

    return da


def grid_from_rasterdataset(
    grid_like: Union[xr.DataArray, xr.Dataset],
    ds: Union[xr.DataArray, xr.Dataset],
    variables: Optional[List] = None,
    fill_method: Optional[str] = None,
    reproject_method: Optional[Union[List, str]] = "nearest",
    mask_name: Optional[str] = "mask",
    rename: Optional[Dict] = dict(),
) -> xr.Dataset:
    """Prepare data by resampling ds to grid_like.

    If raster is a dataset, all variables will be added unless
    ``variables`` list is specified.

    Parameters
    ----------
    grid_like: xr.DataArray, xr.Dataset
        Grid to copy metadata from.
    ds: xr.DataArray, xr.Dataset
        Dataset with raster data.
    variables: list, optional
        List of variables to add to grid from raster_fn. By default all.
    fill_method : str, optional
        If specified, fills nodata values using fill_nodata method.
        Available methods are {'linear', 'nearest', 'cubic', 'rio_idw'}.
    reproject_method: list, str, optional
        See rasterio.warp.reproject for existing methods, by default 'nearest'.
        Can provide a list corresponding to ``variables``.
    mask_name: str, optional
        Name of mask in self.grid to use for masking raster_fn. By default 'mask'.
        Use None to disable masking.
    rename: dict, optional
        Dictionary to rename variable names in raster_fn before adding to grid
        {'name_in_raster_fn': 'name_in_grid'}. By default empty.

    Returns
    -------
    ds_out: xr.Dataset
        Dataset with data from ds resampled to grid_like
    """
    if variables is not None:
        ds = ds[variables]
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
    # Fill nodata
    if fill_method is not None:
        ds = ds.raster.interpolate_na(method=fill_method)
    # Reprojection
    # one reproject method for all variables
    reproject_method = np.atleast_1d(reproject_method)
    if len(reproject_method) == 1:
        ds_out = ds.raster.reproject_like(grid_like, method=reproject_method[0])
    # one reproject method per variable
    elif len(reproject_method) == len(variables):
        ds_list = []
        for var, method in zip(variables, reproject_method):
            ds_list.append(ds[var].raster.reproject_like(grid_like, method=method))
        ds_out = xr.merge(ds_list)
    else:
        raise ValueError(f"reproject_method should have length 1 or {len(variables)}")
    # Masking
    if mask_name is not None:
        if mask_name in grid_like:
            ds_out = ds_out.raster.mask(grid_like[mask_name])
    # Rename

    return ds_out.rename(rename)


def grid_from_raster_reclass(
    grid_like: Union[xr.DataArray, xr.Dataset],
    da: xr.DataArray,
    reclass_table: pd.DataFrame,
    reclass_variables: List,
    fill_method: Optional[str] = None,
    reproject_method: Optional[Union[List, str]] = "nearest",
    mask_name: Optional[str] = "mask",
    rename: Optional[Dict] = dict(),
) -> xr.Dataset:
    """Prepare data variable(s) resampled to grid_like object by reclassifying the data in ``da`` based on ``reclass_table``.

    Parameters
    ----------
    grid_like: xr.DataArray, xr.Dataset
        Grid to copy metadata from.
    da: xr.DataArray
        DataArray with classification raster data.
    reclass_table: pd.DataFrame
        Tabular pandas dataframe object for the reclassification table of `da`.
    reclass_variables: list
        List of reclass_variables from reclass_table_fn table to add to maps.
        Index column should match values in `raster_fn`.
    fill_method : str, optional
        If specified, fills nodata values in `raster_fn` using fill_nodata method
        before reclassifying. Available methods are
        {'linear', 'nearest', 'cubic', 'rio_idw'}.
    reproject_method: str, optional
        See rasterio.warp.reproject for existing methods, by default "nearest".
        Can provide a list corresponding to ``reclass_variables``.
    mask_name: str, optional
        Name of mask in self.grid to use for masking raster_fn. By default 'mask'.
        Use None to disable masking.
    rename: dict, optional
        Dictionary to rename variable names in reclass_variables before adding to grid
        {'name_in_reclass_table': 'name_in_grid'}. By default empty.

    Returns
    -------
    ds_out: xr.Dataset
        Dataset with reclassified data from reclass_table to da resampled to grid_like.
    """  # noqa: E501
    if not isinstance(da, xr.DataArray):
        raise ValueError("da should be a single variable.")
    if reclass_variables is not None:
        reclass_table = reclass_table[reclass_variables]
    # Fill nodata
    if fill_method is not None:
        da = da.raster.interpolate_na(method=fill_method)
    # Mapping function
    ds_out = da.raster.reclassify(reclass_table=reclass_table, method="exact")
    # Reprojection
    # one reproject method for all variables
    reproject_method = np.atleast_1d(reproject_method)
    if len(reproject_method) == 1:
        ds_out = ds_out.raster.reproject_like(grid_like, method=reproject_method[0])
    # one reproject method per variable
    elif len(reproject_method) == len(reclass_variables):
        ds_list = []
        for var, method in zip(reclass_variables, reproject_method):
            ds_list.append(ds_out[var].raster.reproject_like(grid_like, method=method))
        ds_out = xr.merge(ds_list)
    else:
        raise ValueError(
            f"reproject_method should have length 1 or {len(reclass_variables)}"
        )
    # Masking
    if mask_name is not None:
        if mask_name in grid_like:
            ds_out = ds_out.raster.mask(grid_like[mask_name])
    # Rename
    return ds_out.rename(rename)


def grid_from_geodataframe(
    grid_like: Union[xr.DataArray, xr.Dataset],
    gdf: gpd.GeoDataFrame,
    variables: Optional[Union[List, str]] = None,
    nodata: Optional[Union[List, int, float]] = -1,
    rasterize_method: Optional[str] = "value",
    mask_name: Optional[str] = "mask",
    rename: Optional[Union[Dict, str]] = dict(),
    all_touched: Optional[bool] = True,
) -> xr.Dataset:
    """Prepare data variable(s) resampled to grid_like object by rasterizing the data from ``gdf``.

    Rasterizes the data from ``gdf``. Several type of rasterization are possible:
        * "fraction": returns the fraction of the grid cell covered by the gdf shape.
        * "area": Returns the area of the grid cell covered by the gdf shape.
        * "value": the value from the variables columns of gdf are used.
            If this is used, variables must be specified.

    Parameters
    ----------
    grid_like: xr.DataArray, xr.Dataset
        Grid to copy metadata from.
    gdf : gpd.GeoDataFrame
        geopandas object to rasterize.
    variables : List, str, optional
        List of variables to add to grid from vector_fn. Required if
        rasterize_method is "value", by default None.
    nodata : List, int, float, optional
        No data value to use for rasterization, by default -1. If a list is provided,
        it should have the same length has variables.
    rasterize_method : str, optional
        Method to rasterize the vector data. Either {"value", "fraction", "area"}.
        If "value", the value from the variables columns in vector_fn are used
        directly in the raster. If "fraction", the fraction of the grid cell covered
        by the vector file is returned. If "area", the area of the grid cell covered
        by the vector file is returned.
    mask_name: str, optional
        Name of mask in self.grid to use for masking raster_fn. By default 'mask'.
        Use None to disable masking.
    rename: dict or str, optional
        Dictionary to rename variable names in variables before adding to grid
        {'name_in_variables': 'name_in_grid'}. To rename with method fraction
        or area give directly 'name_in_grid' string. By default empty.
    all_touched : bool, optional
        If True (default), all pixels touched by geometries will be burned in.
        If false, only pixels whose center is within the polygon or
        that are selected by Bresenham's line algorithm will be burned in.

    Returns
    -------
    ds_out: xr.Dataset
        Dataset with data from vector_fn resampled to grid_like.
    """  # noqa: E501
    # Check which method is used
    if rasterize_method == "value":
        ds_lst = []
        vars = np.atleast_1d(variables)
        nodata = np.atleast_1d(nodata)
        # Check length of nodata
        if len(nodata) != len(vars):
            if len(nodata) == 1:
                nodata = np.repeat(nodata, len(vars))
            else:
                raise ValueError(
                    f"Length of nodata ({len(nodata)}) should be equal to 1 "
                    + f"or length of variables ({len(vars)})."
                )
        # Loop of variables and nodata
        for var, nd in zip(vars, nodata):
            # Rasterize
            da = grid_like.raster.rasterize(
                gdf=gdf,
                col_name=var,
                nodata=nd,
                all_touched=all_touched,
            )
            # Rename
            if var in rename.keys():
                var = rename[var]
            # Masking
            if mask_name is not None:
                if mask_name in grid_like:
                    da = da.raster.mask(grid_like[mask_name])
            ds_lst.append(da.rename(var))
        # Merge
        ds_out = xr.merge(ds_lst)

    elif rasterize_method in ["fraction", "area"]:
        # Rasterize
        da = grid_like.raster.rasterize_geometry(
            gdf=gdf,
            method=rasterize_method,
            mask_name=None,
            name=rename,
            nodata=nodata,
        )
        # Masking
        if mask_name is not None:
            if mask_name in grid_like:
                da = da.raster.mask(grid_like[mask_name])
        ds_out = da.to_dataset()

    else:
        raise ValueError(
            f"rasterize_method {rasterize_method} not recognized."
            + "Use one of {'value', 'fraction', 'area'}."
        )

    return ds_out


def rotated_grid(
    pol: Polygon, res: float, dec_origin=0, dec_rotation=3
) -> Tuple[float, float, int, int, float]:
    """Return the origin (x0, y0), shape (mmax, nmax) and rotation of the rotated grid.

    The grid is fitted to the minimum rotated rectangle around the
    area of interest (pol). The grid shape is defined by the resolution (res).

    Parameters
    ----------
    pol : Polygon
        Polygon of the area of interest
    res : float
        Resolution of the grid
    dec_origin:
        Number of signifiant numbers to round the origin points to.
    dec_rotation:
        Number of signifiant numbers to round the rotation to.
    """

    def _azimuth(point1, point2):
        """Azimuth between 2 points (interval 0 - 180)."""
        angle = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
        return round(np.degrees(angle), dec_rotation)

    def _dist(a, b):
        """Distance between points."""
        return np.hypot(b[0] - a[0], b[1] - a[1])

    mrr = pol.minimum_rotated_rectangle
    coords = np.asarray(mrr.exterior.coords)[:-1, :]  # get coordinates of all corners
    # get origin based on the corner with the smallest distance to origin
    # after translation to account for possible negative coordinates
    ib = np.argmin(
        np.hypot(coords[:, 0] - coords[:, 0].min(), coords[:, 1] - coords[:, 1].min())
    )
    ir = (ib + 1) % 4
    il = (ib + 3) % 4
    x0, y0 = coords[ib, :]
    x0, y0 = round(x0, dec_origin), round(y0, dec_origin)
    az1 = _azimuth((x0, y0), coords[ir, :])
    az2 = _azimuth((x0, y0), coords[il, :])
    axis1 = _dist((x0, y0), coords[ir, :])
    axis2 = _dist((x0, y0), coords[il, :])
    if az2 < az1:
        rot = az2
        mmax = int(np.ceil(axis2 / res))
        nmax = int(np.ceil(axis1 / res))
    else:
        rot = az1
        mmax = int(np.ceil(axis1 / res))
        nmax = int(np.ceil(axis2 / res))

    return x0, y0, mmax, nmax, rot
