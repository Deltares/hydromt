"""Implementation for grid based workflows."""

from logging import Logger, getLogger
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
from pyproj import CRS, Transformer
from shapely.geometry import Polygon

from hydromt._typing.type_def import Number
from hydromt.data_catalog import DataCatalog
from hydromt.gis import _gis_utils, raster
from hydromt.model.processes.region import (
    parse_region_basin,
    parse_region_bbox,
    parse_region_geom,
    parse_region_grid,
)

logger: Logger = getLogger(__name__)

__all__ = [
    "create_grid_from_region",
    "create_rotated_grid_from_geom",
    "grid_from_constant",
    "grid_from_rasterdataset",
    "grid_from_raster_reclass",
    "grid_from_geodataframe",
    "rotated_grid",
]


def create_grid_from_region(
    region: Dict[str, Any],
    *,
    data_catalog: Optional[DataCatalog] = None,
    res: Optional[Number] = None,
    crs: Optional[Union[int, str]] = None,
    region_crs: int = 4326,
    rotated: bool = False,
    hydrography_path: Optional[str] = None,
    basin_index_path: Optional[str] = None,
    add_mask: bool = True,
    align: bool = True,
    dec_origin: int = 0,
    dec_rotation: int = 3,
) -> xr.DataArray:
    """Create a 2D regular grid or reads an existing grid.

    A 2D regular grid will be created from a geometry (geom_fn) or bbox. If an
    existing grid is given, then no new grid will be generated.

    Parameters
    ----------
    region : dict
        Dictionary describing region of interest, e.g.:
        * {'bbox': [xmin, ymin, xmax, ymax]}
        * {'geom': 'path/to/polygon_geometry'}
        * {'grid': 'path/to/grid_file'}
        * {'basin': [x, y]}

        Region must be of kind [grid, bbox, geom, basin, subbasin, interbasin].
    data_catalog : DataCatalog, optional
        If the data_catalog is None, a new DataCatalog will be created, by default None.
    res: float or int, optional
        Resolution used to generate 2D grid [unit of the CRS], required if region
        is not based on 'grid'.
    crs : int, or str optional
        EPSG code of the grid to create, or 'utm'. if crs is 'utm' the closest utm grid will be
        guessed at.
    region_crs : int, optional
        EPSG code of the region geometry, by default 4326. Only applies if
        region is of kind 'bbox' or if geom crs is not defined in the file itself.
    rotated : bool
        if True, a minimum rotated rectangular grid is fitted around the region,
        by default False. Only applies if region is of kind 'bbox', 'geom'
    hydrography_fn : str, optional
        Name of data source for hydrography data. Required if region is of kind
            'basin', 'subbasin' or 'interbasin'.

        * Required variables: ['flwdir'] and any other 'snapping' variable required
            to define the region.

        * Optional variables: ['basins'] if the `region` is based on a
            (sub)(inter)basins without a 'bounds' argument.

    basin_index_path: str, optional
        Name of data source with basin (bounding box) geometries associated with
        the 'basins' layer of `hydrography_fn`. Only required if the `region` is
        based on a (sub)(inter)basins without a 'bounds' argument.
    add_mask : bool
        Add mask variable to grid object, by default True.
    align : bool
        If True (default), align target transform to resolution.
    dec_origin : int, optional
        number of decimals to round the origin coordinates, by default 0
    dec_rotation : int, optional
        number of decimals to round the rotation angle, by default 3
    logger : Logger
        Logger object, by default a module level logger is used.

    Returns
    -------
    grid : xr.DataArray
        Generated grid mask.
    """
    data_catalog = data_catalog or DataCatalog()

    kind = next(iter(region))
    if kind in ["bbox", "geom"]:
        if not res:
            raise ValueError("res argument required for kind 'bbox', 'geom'")

        if kind == "geom":
            geom = parse_region_geom(
                region,
                crs=region_crs,
                data_catalog=data_catalog,
            )
        else:
            geom = parse_region_bbox(region, crs=region_crs)

        if crs is not None:
            # bbox needs to be 4326 to find correct UTM zone
            # we'll transform the bbox manually to avoid having to
            # reproject the entire geom twice

            # Create a transformer from the original CRS to EPSG:4326
            transformer = Transformer.from_crs(geom.crs, "EPSG:4326", always_xy=True)

            # Output transformed bounds
            bounds_4326 = transformer.transform_bounds(*geom.total_bounds)
            crs = _gis_utils._parse_crs(crs, bbox=bounds_4326)
            geom = geom.to_crs(crs)

        if rotated:
            grid = create_rotated_grid_from_geom(
                geom, res=res, dec_origin=dec_origin, dec_rotation=dec_rotation
            )
        else:
            xcoords, ycoords = _extract_coords_from_geom(geom, res=res, align=align)
            grid = _create_non_rotated_grid(xcoords, ycoords, crs=geom.crs)
    elif kind == "grid":
        if rotated:
            logger.warning(
                "Ignoring rotated argument when creating grid with kind grid"
            )

        if res is None:
            logger.warning("Ignoring res argument when creating grid with kind grid")

        if crs is None:
            logger.warning("Ignoring crs argument when creating grid with kind grid")

        da = parse_region_grid(region, data_catalog=data_catalog)
        xcoords = da.raster.xcoords.values
        ycoords = da.raster.ycoords.values
        geom = da.raster.box
        grid = _create_non_rotated_grid(xcoords, ycoords, crs=da.raster.crs)
    elif kind in ["basin", "interbasin", "subbasin"]:
        if rotated:
            logger.warning("Cannot create a rotated grid from a basin region")

        geom = parse_region_basin(
            region,
            data_catalog=data_catalog,
            hydrography_path=hydrography_path,
            basin_index_path=basin_index_path,
        )
        xcoords, ycoords, crs = _extract_coords_from_basin(
            geom,
            hydrography_fn=hydrography_path,
            res=res,
            crs=crs,
            align=align,
            data_catalog=data_catalog,
        )
        grid = _create_non_rotated_grid(xcoords, ycoords, crs=crs)
    else:
        raise ValueError(
            f"Region for grid must be of kind [grid, bbox, geom, basin, subbasin,"
            f" interbasin], kind {kind} not understood."
        )

    if add_mask:
        grid = grid.raster.geometry_mask(geom, all_touched=True)
        grid.name = "mask"
        return grid
    else:
        grid = grid.to_dataset()
        return grid.drop_vars("mask")


def _create_non_rotated_grid(
    xcoords: np.typing.ArrayLike, ycoords: np.typing.ArrayLike, *, crs: int
) -> xr.DataArray:
    """Create a grid that is not rotated based on x and y coordinates.

    Parameters
    ----------
    xcoords : np.typing.ArrayLike
        An array of x coordinates.
    ycoords : np.typing.ArrayLike
        An array of y coordinates.
    crs : int
        The crs that the coordinates are in.

    Returns
    -------
    xr.DataArray
        A grid that is not rotated.
    """
    coords = {"x": xcoords, "y": ycoords}
    return raster.full(
        coords=coords,
        nodata=1,
        dtype=np.uint8,
        name="mask",
        attrs={},
        crs=crs,
        lazy=False,
    )


def create_rotated_grid_from_geom(
    geom: gpd.GeoDataFrame, *, res: float, dec_origin: int, dec_rotation: int
) -> xr.DataArray:
    """Create a rotated grid based on a geometry.

    Parameters
    ----------
    geom : gpd.GeoDataFrame
        The geometry to create the grid from.
    res : float
        The resolution of the grid.
    dec_origin : int
        The number of significant numbers to round the origin points to.
    dec_rotation : int
        The number of significant numbers to round the rotation to.

    Returns
    -------
    xr.DataArray
        A rotated grid based on the geometry.
    """
    geomu = geom.union_all()
    x0, y0, mmax, nmax, rot = rotated_grid(
        geomu, res, dec_origin=dec_origin, dec_rotation=dec_rotation
    )
    transform = (
        Affine.translation(x0, y0) * Affine.rotation(rot) * Affine.scale(res, res)
    )
    return raster.full_from_transform(
        transform,
        shape=(nmax, mmax),
        nodata=1,
        dtype=np.uint8,
        name="mask",
        attrs={},
        crs=geom.crs,
        lazy=False,
    )


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
        Name of mask in self.grid to use for masking raster_data. By default 'mask'.
        Use None to disable masking.

    Returns
    -------
    da: xr.DataArray
        Grid with constant value.
    """
    da = raster.full(
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
    variables: Optional[List[str]] = None,
    fill_method: Optional[str] = None,
    reproject_method: Optional[Union[List[str], str]] = "nearest",
    mask_name: Optional[str] = "mask",
    rename: Optional[Dict[str, str]] = None,
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
        List of variables to add to grid from raster_data. By default all.
    fill_method : str, optional
        If specified, fills nodata values using fill_nodata method.
        Available methods are {'linear', 'nearest', 'cubic', 'rio_idw'}.
    reproject_method: list, str, optional
        See rasterio.warp.reproject for existing methods, by default 'nearest'.
        Can provide a list corresponding to ``variables``.
    mask_name: str, optional
        Name of mask in self.grid to use for masking raster_data. By default 'mask'.
        Use None to disable masking.
    rename: dict, optional
        Dictionary to rename variable names in raster_data before adding to grid
        {'name_in_raster_data': 'name_in_grid'}. By default empty.

    Returns
    -------
    ds_out: xr.Dataset
        Dataset with data from ds resampled to grid_like
    """
    rename = rename or dict()
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
    variables = ds or ds[variables]
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
        List of reclass_variables from reclass_table_data table to add to maps.
        Index column should match values in `raster_data`.
    fill_method : str, optional
        If specified, fills nodata values in `raster_data` using fill_nodata method
        before reclassifying. Available methods are
        {'linear', 'nearest', 'cubic', 'rio_idw'}.
    reproject_method: str, optional
        See rasterio.warp.reproject for existing methods, by default "nearest".
        Can provide a list corresponding to ``reclass_variables``.
    mask_name: str, optional
        Name of mask in self.grid to use for masking raster_data. By default 'mask'.
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
    if mask_name is not None and mask_name in grid_like:
        ds_out = ds_out.raster.mask(grid_like[mask_name])
    # Rename
    return ds_out.rename(rename)


def grid_from_geodataframe(
    grid_like: Union[xr.DataArray, xr.Dataset],
    gdf: gpd.GeoDataFrame,
    variables: Optional[Union[List[str], str]] = None,
    nodata: Optional[Union[List[Union[int, float]], int, float]] = -1,
    rasterize_method: Optional[str] = "value",
    mask_name: Optional[str] = "mask",
    rename: Optional[Union[Dict[str, str], str]] = None,
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
        List of variables to add to grid from vector_data. Required if
        rasterize_method is "value", by default None.
    nodata : List, int, float, optional
        No data value to use for rasterization, by default -1. If a list is provided,
        it should have the same length has variables.
    rasterize_method : str, optional
        Method to rasterize the vector data. Either {"value", "fraction", "area"}.
        If "value", the value from the variables columns in vector_data are used
        directly in the raster. If "fraction", the fraction of the grid cell covered
        by the vector file is returned. If "area", the area of the grid cell covered
        by the vector file is returned.
    mask_name: str, optional
        Name of mask in self.grid to use for masking raster_data. By default 'mask'.
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
        Dataset with data from vector_data resampled to grid_like.
    """  # noqa: E501
    # Check which method is used
    rename = rename or dict()
    if rasterize_method == "value":
        ds_lst = []
        variables = np.atleast_1d(variables)
        nodata = np.atleast_1d(nodata)
        # Check length of nodata
        if len(nodata) != len(variables):
            if len(nodata) == 1:
                nodata = np.repeat(nodata, len(variables))
            else:
                raise ValueError(
                    f"Length of nodata ({len(nodata)}) should be equal to 1 "
                    + f"or length of variables ({len(variables)})."
                )
        # Loop of variables and nodata
        for var, nd in zip(variables, nodata):
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
            if mask_name is not None and mask_name in grid_like:
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


def _extract_coords_from_geom(
    geom: gpd.GeoDataFrame, *, res: float, align: bool
) -> Tuple[np.typing.ArrayLike, np.typing.ArrayLike]:
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
    return xcoords, ycoords


def _extract_coords_from_basin(
    geom: gpd.GeoDataFrame,
    *,
    hydrography_fn: str,
    res: Optional[float],
    crs: Optional[int],
    align: bool,
    data_catalog: DataCatalog,
) -> Tuple[np.typing.ArrayLike, np.typing.ArrayLike, CRS]:
    da_hyd = data_catalog.get_rasterdataset(hydrography_fn, geom=geom)

    if not res:
        logger.info(
            "res argument not defined, using resolution of ",
            f"hydrography data {da_hyd.raster.res}",
        )
        res = da_hyd.raster.res
    if res != da_hyd.raster.res:
        if crs is not None and crs != da_hyd.raster.crs:
            crs = _gis_utils._parse_crs(crs, da_hyd.raster.bounds)
        else:
            crs = da_hyd.raster.crs
        da_hyd = da_hyd.raster.reproject(dst_crs=crs, dst_res=res, align=align)

    xcoords = da_hyd.raster.xcoords.values
    ycoords = da_hyd.raster.ycoords.values

    return xcoords, ycoords, crs
