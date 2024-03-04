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

logger = logging.getLogger(__name__)


def _parse_region(
    region, data_catalog=None, logger=logger
) -> Tuple[str, Dict[str, Any]]:
    if data_catalog is None:
        data_catalog = DataCatalog()
    kwargs = region.copy()
    # NOTE: the order is important to prioritize the arguments
    options = {
        "basin": ["basid", "geom", "bbox", "xy"],
        "subbasin": ["geom", "bbox", "xy"],
        "interbasin": ["geom", "bbox", "xy"],
        "geom": ["geom"],
        "bbox": ["bbox"],
        "grid": ["RasterDataArray"],
        "mesh": ["UgridDataArray"],
    }
    kind = next(iter(kwargs))  # first key of region
    value0 = kwargs.pop(kind)
    from hydromt.models import MODELS

    if kind in MODELS:
        model_class = MODELS.load(kind)
        kwargs = dict(mod=model_class.__init__(root=value0, mode="r", logger=logger))
        kind = "model"

    if kind == "grid":
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
        k_lst = '", "'.join(list(options.keys()))
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
