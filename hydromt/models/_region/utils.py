import logging
from os.path import isdir, isfile
from pathlib import Path
from typing import Any, Dict, Tuple

import geopandas as gpd
import numpy as np
import xarray as xr
import xugrid as xu

from hydromt import _compat
from hydromt.data_catalog import DataCatalog
from hydromt.models import MODELS

logger = logging.getLogger(__name__)


def _parse_region(
    region, data_catalog=None, logger=logger
) -> Tuple[str, Dict[str, Any]]:
    """Check and return parsed region arguments.

    Parameters
    ----------
    region : dict
        Dictionary describing region of interest.

        For an exact clip of the region:

        * {'bbox': [xmin, ymin, xmax, ymax]}

        * {'geom': /path/to/polygon_geometry}

        For a region based of another models grid:

        * {'<model_name>': root}

        For a region based of the grid of a raster file:

        * {'grid': /path/to/raster}

        For a region based on a mesh grid of a mesh file:

        * {'mesh': /path/to/mesh}

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
    logger:
        The logger to use.

    Returns
    -------
    kind : {'basin', 'subbasin', 'interbasin', 'geom', 'bbox', 'grid'}
        region kind
    kwargs : dict
        parsed region json
    """
    if data_catalog is None:
        data_catalog = DataCatalog()
    kwargs = region.copy()
    # NOTE: the order is important to prioritize the arguments
    options = {
        "basin": ["basid", "geom", "bbox", "xy"],
        "subbasin": ["geom", "bbox", "xy"],
        "interbasin": ["geom", "bbox", "xy"],  # FIXME remove interbasin & xy combi?
        "outlet": ["geom", "bbox"],  # deprecated!
        "geom": ["geom"],
        "bbox": ["bbox"],
        "grid": ["RasterDataArray"],
        "mesh": ["UgridDataArray"],
    }
    kind = next(iter(kwargs))  # first key of region
    value0 = kwargs.pop(kind)
    if kind in MODELS:
        model_class = MODELS.load(kind)
        kwargs = dict(mod=model_class(root=value0, mode="r", logger=logger))
        kind = "model"
    elif kind == "grid":
        kwargs = {"grid": data_catalog.get_rasterdataset(value0, driver_kwargs=kwargs)}
    elif kind == "mesh":
        if _compat.HAS_XUGRID:
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
                    f"Unrecognised type {type(value0)}."
                    "Should be a path, data catalog key or xugrid object."
                )
            kwargs.update(kwarg)
        else:
            raise ImportError("xugrid is required to read mesh files.")
    elif kind not in options:
        k_lst = '", "'.join(list(options.keys()) + list(MODELS))
        raise ValueError(f'Region key "{kind}" not understood, select from "{k_lst}"')
    else:
        kwarg = _parse_region_value(value0, data_catalog=data_catalog)
        if len(kwarg) == 0 or next(iter(kwarg)) not in options[kind]:
            v_lst = '", "'.join(list(options[kind]))
            raise ValueError(
                f'Region value "{value0}" for kind={kind} not understood, '
                f'provide one of "{v_lst}"'
            )
        kwargs.update(kwarg)
    kwargs_str = dict()
    for k, v in kwargs.items():
        if isinstance(v, gpd.GeoDataFrame):
            v = f"GeoDataFrame {v.total_bounds} (crs = {v.crs})"
        elif isinstance(v, xr.DataArray):
            v = f"DataArray {v.raster.bounds} (crs = {v.raster.crs})"
        kwargs_str.update({k: v})
    logger.debug(f"Parsed region (kind={kind}): {str(kwargs_str)}")
    return kind, kwargs


def _parse_region_value(value, data_catalog):
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


def _check_size(ds, logger=logger, threshold=12e3**2):
    # warning for large domain
    if (
        np.multiply(*ds.raster.shape) > threshold
    ):  # 12e3 ** 2 > 10x10 degree at 3 arcsec
        logger.warning(
            "Loading very large spatial domain to derive a subbasin. "
            "Provide initial 'bounds' if this takes too long."
        )
