"""Implementation for mesh based workflows."""

from logging import Logger, getLogger
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu
from pyproj import CRS
from shapely.geometry import box

from hydromt.data_catalog import DataCatalog
from hydromt.gis import gis_utils
from hydromt.gis.raster import GEO_MAP_COORD
from hydromt.model.processes.region import (
    parse_region_bbox,
    parse_region_geom,
    parse_region_mesh,
)

logger: Logger = getLogger(__name__)

__all__ = [
    "create_mesh2d_from_region",
    "create_mesh2d_from_mesh",
    "create_mesh2d_from_geom",
    "mesh2d_from_rasterdataset",
    "mesh2d_from_raster_reclass",
]


def create_mesh2d_from_region(
    region: Dict[str, Any],
    *,
    crs: Optional[int] = None,
    region_crs: int = 4326,
    res: Optional[float] = None,
    align: bool = True,
    data_catalog: Optional[DataCatalog] = None,
):
    """HYDROMT CORE METHOD: Create an 2D unstructured mesh or reads an existing 2D mesh according UGRID conventions.

    Grids are read according to UGRID conventions. An 2D unstructured mesh
    will be created as 2D rectangular grid from a geometry (geom_filename) or bbox.
    If an existing 2D mesh is given, then no new mesh will be generated but an extent
    can be extracted using the `bounds` argument of region.

    Note Only existing meshed with only 2D grid can be read.

    Parameters
    ----------
    region : dict
        Dictionary describing region of interest, bounds can be provided for type 'mesh'.
        In case of 'mesh', if the file includes several grids, the specific 2D grid can
        be selected using the 'grid_name' argument.
        CRS for 'bbox' and 'bounds' should be 4326; e.g.:

        * {'bbox': [xmin, ymin, xmax, ymax]}
        * {'geom': 'path/to/polygon_geometry'}
        * {'mesh': 'path/to/2dmesh_file'}
        * {'mesh': 'path/to/mesh_file', 'grid_name': 'mesh2d', 'bounds': [xmin, ymin, xmax, ymax]}

    res : float, optional
        Resolution used to generate 2D mesh [unit of the CRS], required if region
        is not based on 'mesh'.
    crs : int, optional
        Optional EPSG code of the model.
        If None using the one from region, and else 4326.
    region_crs : int, optional
        EPSG code of the region geometry, by default None. Only applies if region is
        of kind 'bbox'or if geom crs is not defined in the file itself.
    align : bool, default True
        Align the mesh to the resolution.
        Required for 'bbox' and 'geom' region types.
    logger : logging.Logger
        Logger object, a default module logger is used if not specified.
    data_catalog : DataCatalog, optional
        Optional data catalog to use for reading data.
        Required if region is based on 'geom'.

    Returns
    -------
    mesh2d : xu.UgridDataset
        Generated mesh2d.
    """
    kind = next(iter(region))
    if kind == "mesh":
        uds = parse_region_mesh(region)
        return create_mesh2d_from_mesh(
            uds,
            grid_name=region.get("grid_name", None),
            crs=crs,
            bounds=region.get("bounds", None),
        )

    if kind in ["bbox", "geom"]:
        if not res:
            raise ValueError(f"res argument required for kind {kind}")

        geom = (
            parse_region_bbox(region, crs=region_crs)
            if kind == "bbox"
            else parse_region_geom(region, crs=region_crs, data_catalog=data_catalog)
        )
        clip_to_geom = kind == "geom"

        if crs is not None:
            crs = gis_utils.parse_crs(crs, bbox=geom.total_bounds)
            geom = geom.to_crs(crs)

        return create_mesh2d_from_geom(
            geom, res=res, align=align, clip_to_geom=clip_to_geom
        )

    raise ValueError(f"Unsupported region kind '{kind}' found in grid creation.")


def create_mesh2d_from_mesh(
    uds: xu.UgridDataset,
    *,
    grid_name: Optional[str],
    crs: Optional[int],
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> xu.UgridDataset:
    """
    Create a 2D mesh from another mesh.

    The 2D mesh can be extracted from a multi grid mesh using the grid_name argument.
    If bounds are provided, the mesh is clipped to the bounds.

    Parameters
    ----------
    uds : xu.UgridDataset
        Mesh dataset.
    grid_name : str, optional
        Name of the grid to extract from the mesh.
    crs : int, optional
        Optional EPSG code of the mesh if cannot be found.
    bounds : tuple, optional
        Bounding box to clip the mesh.
    logger : logging.Logger
        Logger object, a default module logger is used if not specified.

    Returns
    -------
    mesh2d : xu.UgridDataset
        Generated mesh2d.
    """
    grids = dict()
    for grid in uds.grids:
        grids[grid.name] = grid
    # Select specific topology if given
    if grid_name is not None:
        if grid_name not in grids:
            raise ValueError(f"Mesh file does not contain grid {grid_name}.")
        grid = grids[grid_name]
    elif len(grids) == 1:
        grid = next(iter(grids.values()))
    else:
        raise ValueError(
            "Mesh file contains several grids. "
            "Use grid_name argument of region to select a single one."
        )
    # Check if 2d mesh file else throw error
    if grid.topology_dimension != 2:
        raise ValueError("Grid in mesh file for create_mesh2d is not 2D.")
    # Continues with a 2D grid
    mesh2d = xu.UgridDataset(grid.to_dataset())

    # Check crs and reproject to model crs
    grid_crs = grid.crs
    if crs is not None:
        bbox = None
        if grid_crs is not None:
            if grid_crs.to_epsg() == 4326:
                bbox = mesh2d.ugrid.grid.bounds
        crs = gis_utils.parse_crs(crs, bbox=bbox)
    else:
        crs = CRS.from_user_input(4326)
    if grid_crs is not None:  # parse crs
        mesh2d.ugrid.grid.set_crs(grid_crs)
    else:
        # Assume model crs
        logger.warning(
            "Mesh data from mesh file doesn't have a CRS." f" Assuming crs option {crs}"
        )
        mesh2d.ugrid.grid.set_crs(crs)
    mesh2d = mesh2d.drop_vars(GEO_MAP_COORD, errors="ignore")

    # If bounds are provided in region, extract mesh for bounds
    if bounds is not None:
        bounds_geom = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=4326)
        xmin, ymin, xmax, ymax = bounds_geom.to_crs(mesh2d.ugrid.grid.crs).total_bounds
        subset = mesh2d.ugrid.sel(y=slice(ymin, ymax), x=slice(xmin, xmax))
        # Check if still cells after clipping
        err = (
            "MeshDataset: No data within model region."
            "Check that bounds were given in the correct crs 4326"
        )
        subset = subset.ugrid.assign_node_coords()
        if subset.ugrid.grid.node_x.size == 0 or subset.ugrid.grid.node_y.size == 0:
            raise IndexError(err)
        mesh2d = subset

    return mesh2d


def create_mesh2d_from_geom(
    geom: gpd.GeoDataFrame,
    *,
    res: float,
    align: bool,
    clip_to_geom: bool,
) -> xu.UgridDataset:
    """
    Create a regular 2D mesh from a boundary geometry.

    Parameters
    ----------
    geom : gpd.GeoDataFrame
        Boundary geometry.
    res : float
        Resolution of the mesh.
    align : bool
        Align the mesh to the resolution.
    clip_to_geom : bool
        Mask inactive cells of the mesh to the boundary geometry.

    Returns
    -------
    mesh2d : xu.UgridDataset
        Generated mesh2d.
    """
    xmin, ymin, xmax, ymax = geom.total_bounds
    if align:
        xmin = round(xmin / res) * res
        ymin = round(ymin / res) * res
        xmax = round(xmax / res) * res
        ymax = round(ymax / res) * res
    # note we flood the number of faces within bounds
    ncol = round((xmax - xmin) / res)  # int((xmax - xmin) // res)
    nrow = round((ymax - ymin) / res)  # int((ymax - ymin) // res)
    dx, dy = res, -res
    faces = []
    for i in range(nrow):
        top = ymax + i * dy
        bottom = ymax + (i + 1) * dy
        for j in range(ncol):
            left = xmin + j * dx
            right = xmin + (j + 1) * dx
            faces.append(box(left, bottom, right, top))
    grid = gpd.GeoDataFrame(geometry=faces, crs=geom.crs)
    # clip to geom
    if clip_to_geom:
        grid = grid.loc[
            gpd.sjoin(
                grid, geom, how="left", predicate="intersects"
            ).index_right.notna()
        ].reset_index()
    # Create mesh from grid
    grid.index.name = "mesh2d_nFaces"
    mesh2d = xu.UgridDataset.from_geodataframe(grid)
    mesh2d = mesh2d.ugrid.assign_face_coords()
    mesh2d.ugrid.grid.set_crs(grid.crs)
    return mesh2d


def mesh2d_from_rasterdataset(
    ds: Union[xr.DataArray, xr.Dataset],
    mesh2d: Union[xu.UgridDataArray, xu.Ugrid2d],
    variables: Optional[List] = None,
    fill_method: Optional[str] = None,
    resampling_method: Optional[Union[str, List]] = "centroid",
    rename: Optional[Dict] = None,
) -> xu.UgridDataset:
    """
    Resamples data in ds to mesh2d.

    Raster data is interpolated to the mesh using the ``resampling_method``.

    Parameters
    ----------
    ds: xr.DataArray, xr.Dataset
        Raster xarray data object.
    mesh2d: xu.UgridDataArray, xu.Ugrid2d
        Mesh2d grid to resample to.
    variables: list, optional
        List of variables to resample. By default all variables in ds.
    fill_method : str, optional
        If specified, fills no data values using fill_nodata method.
        Available methods are {'linear', 'nearest', 'cubic', 'rio_idw'}.
    resampling_method: str, optional
        Method to sample from raster data to mesh. By default mean. Options include
        {"centroid", "barycentric", "mean", "harmonic_mean", "geometric_mean", "sum",
        "minimum", "maximum", "mode", "median", "max_overlap"}. If centroid, will use
        :py:meth:`xugrid.CentroidLocatorRegridder` method. If barycentric, will use
        :py:meth:`xugrid.BarycentricInterpolator` method. If any other, will use
        :py:meth:`xugrid.OverlapRegridder` method.
        Can provide a list corresponding to ``variables``.
    rename: dict, optional
        Dictionary to rename variable names in ds
        {'name_ds': 'name_in_uds_out'}. By default empty.

    Returns
    -------
    uds_out: xu.UgridDataset
        Resampled data on mesh2d.
    """
    rename = rename or {}
    if variables is not None:
        ds = ds[variables]
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()

    if fill_method is not None:
        ds = ds.raster.interpolate_na(method=fill_method)

    # check resampling method
    resampling_method = np.atleast_1d(resampling_method)
    if len(resampling_method) == 1:
        resampling_method = np.repeat(resampling_method, len(ds.data_vars))
    # one reproject method per variable
    elif len(resampling_method) != len(variables):
        raise ValueError(
            f"resampling_method should have length 1 or {len(ds.data_vars)}"
        )

    # Prepare regridder
    regridder = dict()
    # Get one variable name in ds to simplify to da
    var = [v for v in ds.data_vars][0]
    uda = xu.UgridDataArray.from_structured(
        ds[var], x=ds.raster.xcoords.name, y=ds.raster.ycoords.name
    )
    uda.ugrid.set_crs(ds.raster.crs)

    # Need to reproject before calling regridder
    if hasattr(mesh2d, "ugrid"):
        mesh2d_crs = mesh2d.ugrid.grid.crs
    else:  # ugrid2d
        mesh2d_crs = mesh2d.crs
    if mesh2d_crs != uda.ugrid.grid.crs:
        uda = uda.ugrid.to_crs(mesh2d_crs)

    for method in np.unique(resampling_method):
        logger.info(f"Preparing regridder for {method} method")
        if method == "centroid":
            regridder[method] = xu.CentroidLocatorRegridder(uda, mesh2d)
        elif method == "barycentric":
            regridder[method] = xu.BarycentricInterpolator(uda, mesh2d)
        else:
            regridder[method] = xu.OverlapRegridder(uda, mesh2d, method=method)

    # Convert ds to xugrid
    for i, var in enumerate(ds.data_vars):
        logger.info(f"Resampling {var} to mesh2d using {resampling_method[i]} method")
        uda = xu.UgridDataArray.from_structured(
            ds[var].rename({ds.raster.x_dim: "x", ds.raster.y_dim: "y"})
        )
        uda.ugrid.set_crs(ds.raster.crs)
        # Reproject
        if mesh2d_crs != uda.ugrid.grid.crs:
            uda = uda.ugrid.to_crs(mesh2d_crs)
        # Interpolate
        method = resampling_method[i]
        # Interpolate
        uda_out = regridder[method].regrid(uda)
        # Add to uds_out
        if i == 0:
            uds_out = uda_out.to_dataset()
        else:
            uds_out[var] = uda_out

    # Rename variables
    if rename is not None:
        uds_out = uds_out.rename(rename)

    return uds_out


def mesh2d_from_raster_reclass(
    da: xr.DataArray,
    df_vars: pd.DataFrame,
    mesh2d: Union[xu.UgridDataArray, xu.Ugrid2d],
    reclass_variables: list,
    fill_method: Optional[str] = None,
    resampling_method: Optional[Union[str, list]] = "centroid",
    rename: Optional[Dict] = None,
) -> List[str]:
    """Resample data to ``mesh2d`` grid by reclassifying the data in ``da`` based on ``df_vars``.

    The reclassified raster data
    are subsequently interpolated to the mesh using `resampling_method`.

    Parameters
    ----------
    da : xr.DataArray
        Raster xarray DataArray object.
    df_vars : pd.DataFrame
        Tabular pandas dataframe object for the reclassification table of `da`.
    mesh2d: xu.UgridDataArray, xu.Ugrid2d
        Mesh2d grid to resample to.
    reclass_variables : list
        List of reclass_variables from the df_vars table to add to the
        mesh. The index column should match values in da.
    fill_method : str, optional
        If specified, fills nodata values in `da` using the `fill_method`
        method before reclassifying. Available methods are
        {'linear', 'nearest', 'cubic', 'rio_idw'}.
    resampling_method: str, list, optional
        Method to sample from raster data to mesh. By default mean. Options include
        {"centroid", "barycentric", "mean", "harmonic_mean", "geometric_mean", "sum",
        "minimum", "maximum", "mode", "median", "max_overlap"}. If centroid, will use
        :py:meth:`xugrid.CentroidLocatorRegridder` method. If barycentric, will use
        :py:meth:`xugrid.BarycentricInterpolator` method. If any other, will use
        :py:meth:`xugrid.OverlapRegridder` method.
        Can provide a list corresponding to ``reclass_variables``.
    rename : dict, optional
        Dictionary to rename variable names in `reclass_variables` before adding
        them to the mesh. The dictionary should have the form
        {'name_in_reclass_table': 'name_in_uds_out'}. By default, an empty dictionary.

    Returns
    -------
    uds_out : xu.UgridDataset
        Resampled data on mesh2d.

    See Also
    --------
    mesh2d_from_rasterdataset
    """  # noqa: E501
    rename = rename or {}

    if fill_method is not None:
        da = da.raster.interpolate_na(method=fill_method)

    # Mapping function
    ds_vars = da.raster.reclassify(reclass_table=df_vars, method="exact")

    uds_out = mesh2d_from_rasterdataset(
        ds_vars,
        mesh2d,
        variables=reclass_variables,
        fill_method=None,
        resampling_method=resampling_method,
        rename=rename,
    )

    return uds_out
