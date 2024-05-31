"""parse a region from a dict. See parse_region for information on usage."""

from logging import Logger, getLogger
from os.path import isdir, isfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import geopandas as gpd
import numpy as np
import xarray as xr
import xugrid as xu
from pyproj import CRS
from shapely import box

from hydromt._typing import NoDataStrategy
from hydromt._typing.type_def import StrPath
from hydromt.data_catalog import DataCatalog
from hydromt.gis import utils as gis_utils
from hydromt.model.processes.basin_mask import get_basin_geometry
from hydromt.plugins import PLUGINS

if TYPE_CHECKING:
    from hydromt.model.model import Model

_logger = getLogger(__name__)

__all__ = [
    "parse_region_basin",
    "parse_region_bbox",
    "parse_region_geom",
    "parse_region_grid",
    "parse_region_other_model",
    "parse_region_mesh",
]


def parse_region_basin(
    region: dict,
    *,
    data_catalog: DataCatalog,
    hydrography_fn: StrPath,
    basin_index_fn: Optional[StrPath] = None,
    logger: Logger = _logger,
) -> gpd.GeoDataFrame:
    """Parse a basin /subbasin / interbasin region and return the GeoDataFrame.

    Parameters
    ----------
    region : dict
        Dictionary describing region of interest.
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
    data_catalog : DataCatalog
        DataCatalog object containing the data sources.
    hydrography_fn : strPath
        Path of the hydrography raster dataset in the data catalog.
    basin_index_fn : strPath, optional
        Path of the basin index raster dataset in the data catalog.
    logger : Logger, optional
        Logger object.
    """
    var_thresh_kwargs = region.copy()
    kind = next(iter(region))
    value0 = var_thresh_kwargs.pop(kind)

    _assert_parse_key(kind, "basin", "interbasin", "subbasin")

    # TODO: Make this very specific to basin.
    kwargs = _parse_region_value(value0, data_catalog=data_catalog)
    kwargs.update(var_thresh_kwargs)

    expected_keys = (
        ["basid", "geom", "bbox", "xy"] if kind == "basin" else ["geom", "bbox", "xy"]
    )
    _assert_parsed_values(
        key=next(iter(kwargs)), region_value=value0, kind=kind, expected=expected_keys
    )
    kwargs_str = dict()
    for k, v in kwargs.items():
        if isinstance(v, gpd.GeoDataFrame):
            v = f"GeoDataFrame {v.total_bounds} (crs = {v.crs})"
        kwargs_str.update({k: v})
    logger.debug(f"Parsed region (kind={kind}): {str(kwargs_str)}")

    ds_org = data_catalog.get_rasterdataset(hydrography_fn)
    if "bounds" not in kwargs and basin_index_fn is not None:
        kwargs.update(basin_index=data_catalog.get_source(str(basin_index_fn)))
    # get basin geometry
    geom, _ = get_basin_geometry(ds=ds_org, kind=kind, logger=logger, **kwargs)

    return geom


def parse_region_bbox(
    region: dict,
    *,
    crs: int = 4326,
    logger: Logger = _logger,
) -> gpd.GeoDataFrame:
    """Parse a region of kind bbox and return the GeoDataFrame.

    Parameters
    ----------
    region : dict
        Dictionary describing region of interest.
        For an exact clip of the region:
    * {'bbox': [xmin, ymin, xmax, ymax]}
    crs : CRS, optional
        CRS of the bounding box coordinates. By default EPSG 4326.
    logger : Logger, optional
        Logger object.
    """
    kwargs = region.copy()
    kind = next(iter(region))
    value0 = kwargs.pop(kind)

    _assert_parse_key(kind, "bbox")

    # TODO: Make this very specific to bbox
    kwargs.update(_parse_region_value(value0, data_catalog=None))

    _assert_parsed_values(
        key=next(iter(kwargs)), region_value=value0, kind="bbox", expected=["bbox"]
    )
    logger.debug(f"Parsed region (kind={kind}): {str(kwargs)}")

    bbox = kwargs["bbox"]
    geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=crs)
    return geom


def parse_region_geom(
    region: dict,
    *,
    crs: Optional[int] = None,
    data_catalog: Optional[DataCatalog] = None,
    logger: Logger = _logger,
) -> gpd.GeoDataFrame:
    """Parse a region and return the GeoDataFrame.

    Parameters
    ----------
    region : dict
        Dictionary describing region of interest.
        For an exact clip of the region:
        * {'geom': /path/to/polygon_geometry}
    crs : CRS, optional
        CRS of the geometry in case it cannot be derived from the geometry itself.
    data_catalog : DataCatalog, optional
        DataCatalog object containing the data sources.
    logger : Logger, optional
        Logger object.
    """
    kwargs = region.copy()
    kind = next(iter(region))
    value0 = kwargs.pop(kind)

    _assert_parse_key(kind, "geom")

    # TODO: Make this very specific to geom
    kwargs.update(_parse_region_value(value0, data_catalog=data_catalog))

    _assert_parsed_values(
        key=next(iter(kwargs)), region_value=value0, kind="geom", expected=["geom"]
    )
    kwargs_str = dict()
    for k, v in kwargs.items():
        if isinstance(v, gpd.GeoDataFrame):
            v = f"GeoDataFrame {v.total_bounds} (crs = {v.crs})"
        kwargs_str.update({k: v})
    logger.debug(f"Parsed region (kind={kind}): {str(kwargs_str)}")

    geom = kwargs["geom"]
    if geom.crs is None:
        logger.debug(f'Model region "geom" has no CRS, setting to {crs}')
        _update_crs(geom, crs)

    return geom


def parse_region_grid(
    region: dict,
    *,
    data_catalog: Optional[DataCatalog],
    logger: Logger = _logger,
) -> Union[xr.DataArray, xr.Dataset]:
    """Parse a region of kind grid and return the corresponding xarray object.

    Note that additional arguments from the region can be passed to the
    ``driver_kwargs`` argument of the hydromt.data_catalog.DataCatalog.get_rasterdataset
    method.

    Parameters
    ----------
    region : dict
        Dictionary describing region of interest.
        For a region based on a grid:

        * {'grid': /path/to/grid}
    data_catalog : DataCatalog
        DataCatalog object containing the data sources.
    logger : Logger, optional
        Logger object.

    Returns
    -------
    xr.DataArray or xr.Dataset
        The parsed xarray object.
    """
    kwargs = region.copy()
    kind = next(iter(region))
    value0 = kwargs.pop(kind)

    _assert_parse_key(kind, "grid")

    kwargs_str = dict()
    for k, v in kwargs.items():
        if isinstance(v, xr.DataArray):
            v = f"DataArray {v.raster.bounds} (crs = {v.raster.crs})"
        kwargs_str.update({k: v})
    logger.debug(f"Parsed region (kind={kind}): {str(kwargs_str)}")

    data_catalog = data_catalog or DataCatalog()
    da = data_catalog.get_rasterdataset(
        value0,
        handle_nodata=NoDataStrategy.RAISE,
        single_var_as_array=True,
        driver_kwargs=kwargs,
    )

    return da


def parse_region_other_model(region: dict, *, logger: Logger = _logger) -> "Model":
    """Parse a region with a model path and return that whole Model in read mode.

    Parameters
    ----------
    region : dict
        Dictionary describing region of interest.
        For a region based of another models grid:
        * {'<model_name>': root}
    logger : Logger, optional
        Logger object.
    """
    kwargs = region.copy()
    kind = next(iter(region))
    value0 = kwargs.pop(kind)

    _assert_parse_key(kind, *PLUGINS.model_plugins.keys())
    logger.debug(f"Parsed region (kind={kind}): {str(kwargs)}")

    model_class = PLUGINS.model_plugins[kind]
    return model_class(root=value0, mode="r", logger=logger)


def parse_region_mesh(region: dict) -> xu.UgridDataset:
    """Parse a region with a mesh path and return that mesh in read mode.

    Parameters
    ----------
    region : dict
        Dictionary describing region of interest.
        For a region based of a mesh grid of a mesh file:
        * {'mesh': /path/to/mesh}
        * {'mesh': UgridDataArray}
        * {'mesh': UgridDataset}
        * {'mesh': Ugrid1d}
        * {'mesh': Ugrid2d}
    """
    kwargs = region.copy()
    kind = next(iter(region))
    value0 = kwargs.pop(kind)

    _assert_parse_key(kind, "mesh")

    if isinstance(value0, (str, Path)) and isfile(value0):
        return xu.open_dataset(value0)
    elif isinstance(value0, (xu.UgridDataset, xu.UgridDataArray)):
        return value0
    elif isinstance(value0, (xu.Ugrid1d, xu.Ugrid2d)):
        return xu.UgridDataset(value0.to_dataset(optional_attributes=True))
    else:
        raise ValueError(
            f"Unrecognized type {type(value0)}." "Should be a path or xugrid object."
        )


def _update_crs(
    geom: gpd.GeoDataFrame, crs: Optional[Union[CRS, int]]
) -> Optional[gpd.GeoDataFrame]:
    if crs is not None:
        crs = gis_utils.parse_crs(crs, bbox=geom.total_bounds)
        return geom.to_crs(crs)
    return geom


def _parse_region_value(
    value: Any, *, data_catalog: Optional[DataCatalog]
) -> Dict[str, Any]:
    kwarg: Dict[str, Any] = {}
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
        data_catalog = data_catalog or DataCatalog()
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


def _assert_parse_key(key: str, *expected: str) -> None:
    if key not in expected:
        raise KeyError(f"Expected key in '{', '.join(expected)}' but got '{key}'.")


def _assert_parsed_values(
    *, key: str, region_value: Any, kind: str, expected: List[str]
) -> None:
    if key not in expected:
        raise ValueError(
            f"Region value '{region_value}' for kind={kind} not understood, "
            f"provide one of {','.join(expected)}"
        )
