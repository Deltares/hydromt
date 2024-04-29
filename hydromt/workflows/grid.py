"""Implementation for grid based workflows."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
from pyproj import CRS
from shapely.geometry import Polygon

from hydromt._typing.error import NoDataStrategy, _exec_nodata_strat
from hydromt.data_catalog import DataCatalog
from hydromt.gis import raster
from hydromt.gis.raster import full

logger = logging.getLogger(__name__)

__all__ = [
    "grid_from_constant",
    "grid_from_rasterdataset",
    "grid_from_raster_reclass",
    "grid_from_geodataframe",
    "rotated_grid",
]


def create_grid_from_region(
    kind: str,
    geom: gpd.GeoDataFrame,
    *,
    logger: logging.Logger = logger,
    data_catalog: Optional[DataCatalog] = None,
    res: Optional[float] = None,
    crs: Optional[int] = None,
    rotated: bool = False,
    hydrography_fn: Optional[str] = None,
    add_mask: bool = True,
    align: bool = True,
    dec_origin: int = 0,
    dec_rotation: int = 3,
) -> xr.Dataset:
    # Derive xcoords, ycoords and geom for the different kind options
    # Generate grid based on res for region bbox
    if kind in ["bbox", "geom"]:
        if not isinstance(res, (int, float)):
            raise ValueError("res argument required for kind 'bbox', 'geom'")
        input_crs = CRS.from_user_input(crs) if crs is not None else geom.crs
        if input_crs.is_geographic and res > 1:
            logger.warning(
                f"The expected CRS {crs} is geographic and resolution is {res} degree ie more than 111*111 km2. Check if this is correct."
            )
        if input_crs.is_projected and res < 1:
            logger.warning(
                f"The expected CRS {crs} is projected and resolution is {res} meter only. Check if this is correct."
            )

        if not rotated:
            xmin, ymin, xmax, ymax = geom.total_bounds
            res = abs(res)
            if align:
                xmin = round(xmin / res) * res
                ymin = round(ymin / res) * res
                xmax = round(xmax / res) * res
                ymax = round(ymax / res) * res
            xcoords = np.linspace(
                xmin + res / 2,
                xmax - res / 2,
                num=round((xmax - xmin) / res),
                endpoint=True,
            )
            ycoords = np.flip(
                np.linspace(
                    ymin + res / 2,
                    ymax - res / 2,
                    num=round((ymax - ymin) / res),
                    endpoint=True,
                )
            )
        else:  # rotated
            geomu = geom.unary_union
            x0, y0, mmax, nmax, rot = rotated_grid(
                geomu, res, dec_origin=dec_origin, dec_rotation=dec_rotation
            )
            transform = (
                Affine.translation(x0, y0)
                * Affine.rotation(rot)
                * Affine.scale(res, res)
            )
    elif kind in ["basin", "subbasin", "interbasin"]:
        # get ds_hyd but clipped to geom, one variable is enough
        assert hydrography_fn is not None
        assert data_catalog is not None
        da_hyd = data_catalog.get_rasterdataset(hydrography_fn, geom=geom)
        if da_hyd is None:
            _exec_nodata_strat(
                f"No data available in hydrography_fn '{hydrography_fn}' on variable 'flwdir'",
                NoDataStrategy.RAISE,
                logger,
            )
        assert da_hyd is not None
        if not isinstance(res, (int, float)):
            logger.info(
                "res argument not defined, using resolution of "
                f"hydrography_fn {da_hyd.raster.res}"
            )
            res = da_hyd.raster.res
        # Reproject da_hyd based on crs and grid and align, method is not important
        # only coords will be used
        input_crs = CRS.from_user_input(crs) if crs is not None else da_hyd.raster.crs
        if input_crs.is_geographic and res > 1:
            logger.warning(
                f"The expected CRS {crs} is geographic and resolution is {res} degree ie more than 111*111 km2. Check if this is correct."
            )
        if input_crs.is_projected and res < 1:
            logger.warning(
                f"The expected CRS {crs} is projected and resolution is {res} meter only. Check if this is correct."
            )

        if res != da_hyd.raster.res and crs != da_hyd.raster.crs:
            da_hyd = da_hyd.raster.reproject(dst_crs=crs, dst_res=res, align=align)
            assert da_hyd is not None
        # Get xycoords, geom
        xcoords = da_hyd.raster.xcoords.values
        ycoords = da_hyd.raster.ycoords.values
    elif kind == "grid":
        # Get xycoords
        xcoords = geom.raster.xcoords.values
        ycoords = geom.raster.ycoords.values
        if res is not None:
            logger.warning(
                "For region kind 'grid', the grid's res is used and not"
                f" user-defined res {res}"
            )

    # Instantiate grid object
    # Generate grid using hydromt full method
    if not rotated:
        coords = {"y": ycoords, "x": xcoords}
        grid = raster.full(
            coords=coords,
            nodata=1,
            dtype=np.uint8,
            name="mask",
            attrs={},
            crs=geom.crs,
            lazy=False,
        )
    else:
        grid = raster.full_from_transform(
            transform,
            shape=(mmax, nmax),
            nodata=1,
            dtype=np.uint8,
            name="mask",
            attrs={},
            crs=geom.crs,
            lazy=False,
        )
    # Create geometry_mask with geom
    if add_mask:
        grid = grid.raster.geometry_mask(geom, all_touched=True)
        grid.name = "mask"
    # Remove mask variable mask from grid if not add_mask
    else:
        grid = grid.to_dataset()
        grid = grid.drop_vars("mask")
    return grid


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
    da = full(
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
    rename: Optional[Dict] = None,
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
    rename = rename or dict()
    variables = ds or ds[variables]
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
    rename: Optional[Dict] = None,
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
    rename = rename or dict()
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
    rename: Optional[Union[Dict, str]] = None,
    all_touched: Optional[bool] = True,
) -> xr.Dataset:
    """Prepare data variable(s) resampled to grid_like object by rasterizing the data from ``gdf``.

    Several type of rasterization are possible:
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
    rename = rename or dict()
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
