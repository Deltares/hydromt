"""parse a region from a dict. See parse_region for information on usage."""

from logging import Logger, getLogger
from os import makedirs
from os.path import basename, exists, join
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast

import geopandas as gpd
import numpy as np
import xarray as xr
import xugrid as xu
from genericpath import isdir, isfile
from pyproj import CRS
from shapely import box

from hydromt._typing.type_def import StrPath
from hydromt.data_catalog import DataCatalog
from hydromt.gis import utils as gis_utils
from hydromt.plugins import PLUGINS
from hydromt.root import ModelRoot
from hydromt.workflows.basin_mask import get_basin_geometry

logger = getLogger(__name__)


def parse_region(
    region: dict,
    *,
    crs: Optional[int],
    logger: Logger = logger,
    hydrography_fn: Optional[str] = None,
    basin_index_fn: Optional[str] = None,
    data_catalog: Optional[DataCatalog] = None,
) -> gpd.GeoDataFrame:
    """Parse a region and return the GeoDataFrame.

    Parameters
    ----------
    region : dict
        Dictionary describing region of interest.

        For an exact clip of the region:

        * {'bbox': [xmin, ymin, xmax, ymax]}

        * {'geom': /path/to/polygon_geometry}

        For a region based of another models grid:

        * {'<model_name>': root}

        For a region based on a mesh grid of a mesh file:

        * {'mesh': /path/to/mesh}

        For a grid region:

        * {'grid': /path/to/grid}

        Entire basin can be defined based on an ID, one or multiple point location
        (x, y), or a region of interest (bounding box or geometry) for which the
        basin IDs are looked up. The basins withint the area of interest can be further
        filtered to only include basins with their outlet within the area of interest
        ('outlets': true) of stream threshold arguments (e.g.: 'uparea': 1000).

        Common use-cases include:

        * {'basin': ID}

        * {'basin': [ID1, ID2, ..]}

        * {'basin': [x, y]}

        * {'basin': [[x1, x2, ..], [y1, y2, ..]]}

        * {'basin': /path/to/point_geometry}

        * {'basin': [xmin, ymin, xmax, ymax]}

        * {'basin': [xmin, ymin, xmax, ymax], 'outlets': true}

        * {'basin': [xmin, ymin, xmax, ymax], '<variable>': threshold}

        Subbasins are defined by its outlet locations and include all area upstream
        from these points. The outlet locations can be passed as xy coordinate pairs,
        but also derived from the most downstream cell(s) within a area of interest
        defined by a bounding box or geometry, optionally refined by stream threshold
        arguments.

        The method can be speed up by providing an additional ``bounds`` argument which
        should contain all upstream cell. If cells upstream of the subbasin are not
        within the provide bounds a warning will be raised. Common use-cases include:

        * {'subbasin': [x, y], '<variable>': threshold}

        * {
            'subbasin': [[x1, x2, ..], [y1, y2, ..]],
            '<variable>': threshold, 'bounds': [xmin, ymin, xmax, ymax]
            }

        * {'subbasin': /path/to/point_geometry, '<variable>': threshold}

        * {'subbasin': [xmin, ymin, xmax, ymax], '<variable>': threshold}

        * {'subbasin': /path/to/polygon_geometry, '<variable>': threshold}

        Interbasins are similar to subbasins but are bounded by a bounding box or
        geometry and do not include all upstream area. Common use-cases include:

        * {'interbasin': [xmin, ymin, xmax, ymax], '<variable>': threshold}

        * {'interbasin': [xmin, ymin, xmax, ymax], 'xy': [x, y]}

        * {'interbasin': /path/to/polygon_geometry, 'outlets': true}
    """
    kwargs = region.copy()
    # NOTE: the order is important to prioritize the arguments
    options = {
        "basin": ["basid", "geom", "bbox", "xy"],
        "subbasin": ["geom", "bbox", "xy"],
        "interbasin": ["geom", "bbox", "xy"],
        "geom": ["geom"],
        "bbox": ["bbox"],
        "mesh": ["UgridDataArray"],
        "grid": ["raster"],
    }
    kind = next(iter(kwargs))  # first key of region
    value0 = kwargs.pop(kind)

    if kind in PLUGINS.model_plugins:
        model_class = PLUGINS.model_plugins[kind]
        other_model = model_class(root=value0, mode="r", logger=logger)
        value0 = other_model.region
        kwargs = dict(geom=other_model.region)
        kind = "geom"
    elif kind == "mesh":
        if isinstance(value0, (str, Path)) and isfile(value0):
            kwarg = dict(mesh=xu.open_dataset(value0))
        elif isinstance(value0, (xu.UgridDataset, xu.UgridDataArray)):
            kwarg = dict(mesh=value0)
        elif isinstance(value0, (xu.Ugrid1d, xu.Ugrid2d)):
            kwarg = dict(
                mesh=xu.UgridDataset(value0.to_dataset(optional_attributes=True))
            )
        else:
            raise ValueError(
                f"Unrecognized type {type(value0)}."
                "Should be a path, data catalog key or xugrid object."
            )
        kwargs.update(kwarg)
    elif kind not in options:
        k_lst = '", "'.join(list(options.keys()))
        raise ValueError(f'Region key "{kind}" not understood, select from "{k_lst}"')

    kwarg = _parse_region_value(value0, data_catalog=data_catalog)
    if len(kwarg) == 0 or next(iter(kwarg)) not in options[kind]:
        v_lst = '", "'.join(list(options[kind]))
        raise ValueError(
            f'Region value "{value0}" for kind={kind} not understood, '
            f'provide one of "{v_lst}"'
        )
    kwargs.update(kwarg)

    if kind in ["basin", "subbasin", "interbasin"]:
        # retrieve global hydrography data (lazy!)
        assert data_catalog is not None
        assert hydrography_fn is not None
        assert basin_index_fn is not None
        ds_org = data_catalog.get_rasterdataset(hydrography_fn)
        if "bounds" not in kwargs:
            kwargs.update(basin_index=data_catalog.get_source(basin_index_fn))
        # get basin geometry
        geom, _ = get_basin_geometry(ds=ds_org, kind=kind, logger=logger, **kwargs)
        _update_crs(geom, crs)
    elif kind == "bbox":
        bbox = kwargs["bbox"]
        geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
        _update_crs(geom, crs)
    elif kind == "geom":
        geom = kwargs["geom"]
        if geom.crs is None:
            raise ValueError('Model region "geom" has no CRS')
        _update_crs(geom, crs)
    elif kind == "grid":
        assert data_catalog is not None
        ds = data_catalog.get_rasterdataset(value0, kwargs=kwargs)
        assert ds is not None
        if crs is not None:
            logger.warning(
                "For region kind 'grid', the grid's crs is used and not"
                f" user-defined crs '{crs}'"
            )
        geom = ds.raster.box

    kwargs_str = dict()
    for k, v in kwargs.items():
        if isinstance(v, gpd.GeoDataFrame):
            v = f"GeoDataFrame {v.total_bounds} (crs = {v.crs})"
        elif isinstance(v, xr.DataArray):
            v = f"DataArray {v.raster.bounds} (crs = {v.raster.crs})"
        kwargs_str.update({k: v})
    logger.debug(f"Parsed region (kind={kind}): {str(kwargs_str)}")

    return geom


def write_region(
    region: gpd.GeoDataFrame,
    *,
    filename: StrPath,
    logger: Logger = logger,
    root: ModelRoot,
    to_wgs84=False,
    **write_kwargs,
):
    """Write the model region to a file."""
    write_path = join(root.path, filename)

    if exists(write_path) and not root.is_override_mode():
        raise OSError(
            f"Model dir already exists and cannot be overwritten: {write_path}"
        )
    base_name = basename(write_path)
    if not exists(base_name):
        makedirs(base_name, exist_ok=True)

    if region is None:
        logger.info("No region data found. skipping writing...")
    else:
        logger.info(f"writing region data to {write_path}")
        gdf = cast(gpd.GeoDataFrame, region.copy())

        if to_wgs84 and (
            write_kwargs.get("driver") == "GeoJSON"
            or str(filename).lower().endswith(".geojson")
        ):
            gdf = gdf.to_crs(4326)

        gdf.to_file(write_path, **write_kwargs)


def _update_crs(
    geom: Optional[gpd.GeoDataFrame], crs: Optional[Union[CRS, int]]
) -> Optional[gpd.GeoDataFrame]:
    if crs is not None and geom is not None:
        crs = gis_utils.parse_crs(crs, bbox=geom.total_bounds)
        return geom.to_crs(crs)
    return geom


def _parse_region_value(
    value: Any, *, data_catalog: Optional[DataCatalog]
) -> Dict[str, Any]:
    kwarg = {}
    if isinstance(value, np.ndarray):
        value = value.tolist()  # array to list

    if isinstance(value, list):
        if np.all([isinstance(p0, int) and abs(p0) > 180 for p0 in value]):  # all int
            kwarg = dict(basid=value)
        elif len(value) == 4:  # 4 floats
            kwarg = dict(bbox=value)
        elif len(value) == 2:  # 2 floats
            kwarg = dict(xy=value)
    elif isinstance(value, tuple) and len(value) == 2:  # tuple of x and y coords
        kwarg = dict(xy=value)
    elif isinstance(value, int):  # single int
        kwarg = dict(basid=value)
    elif isinstance(value, (str, Path)) and isdir(value):
        kwarg = dict(root=value)
    elif isinstance(value, (str, Path)):
        assert data_catalog is not None
        geom = data_catalog.get_geodataframe(value)
        kwarg = dict(geom=geom)
    elif isinstance(value, gpd.GeoDataFrame):  # geometry
        kwarg = dict(geom=value)
    else:
        raise ValueError(f"Region value {value} not understood.")

    if "geom" in kwarg and np.all(kwarg["geom"].geometry.type == "Point"):
        xy = (
            kwarg["geom"].geometry.x.values,
            kwarg["geom"].geometry.y.values,
        )
        kwarg = dict(xy=xy)
    return kwarg
