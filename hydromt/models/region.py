# -*- coding: utf-8 -*-
"""Parse region argument of model.setup_basemaps methods
"""

from os.path import join, isfile, isdir
import geopandas as gpd
import logging
import numpy as np


logger = logging.getLogger(__name__)


def parse_region(region, logger=logger):
    """Checks and returns parsed region arguments.

    Parameters
    ----------
    region : dict
        Dictionary describing region of interest.


        For an exact clip of the region:

        * {'bbox': [xmin, ymin, xmax, ymax]}

        * {'geom': (path to) geopandas.GeoDataFrame[Polygon]}

        For an copy of another models grid:

        * {wflow/sfincs/..: root}

        For basins/outlets intersecting with the region:

        * {'basin': [xmin, ymin, xmax, ymax]}

        * {'basin': (path to) geopandas.GeoDataFrame[Polygon]}

        * {'outlet': [xmin, ymin, xmax, ymax]}

        * {'outlet': (path to) geopandas.GeoDataFrame[Polygon]}


        For basin with ID or at (x, y):

        * {'basin': ID}

        * {'basin': [ID1, ID2, ..]}

        * {'basin': [x, y]}

        * {'basin': [[x1, x2, ..], [y1, y2, ..]]}

        * {'basin': (path to) geopandas.GeoDataFrame[Point]}

        Subbasins are defined by its outlet locations a confining bounding box or geometry.
        Optional stream arguments can be provided to defined a stream mask
        on which the subbasin outlet must be located based on `variable: threshold` pairs,
        e.g.: uparea: 30. Additionally, `outlets: True` includes all outlets in the stream mask

        In case the subbasin is defined by its outlet locations, the stream mask is used
        to snap the locations to the nearest downstream stream cell. The process can be
        speed up by providing an additional bounding bbox argument.

        In case the subbasin is defined by a bounding box or geometry the `within: True`
        option forces the region to be limited to subbasins which are completely
        inside the bounding box or polygon.

        In any case if cell upstream of the subbasin are not within the provide bounding
        box or geometry a warning will be raised.

        * {'subbasin': [x, y], 'variable': threshold}

        * {'subbasin': [[x1, x2, ..], [y1, y2, ..]], 'variable': threshold}

        * {'subbasin': (path to) geopandas.GeoDataFrame[Point], 'variable': threshold}

        * {'subbasin': [xmin, ymin, xmax, ymax], 'variable': threshold, outlets: bool, within: bool}

        * {'subbasin': (path to) geopandas.GeoDataFrame[Polygon], 'variable': threshold, outlets: bool, within: bool}

    Returns
    -------
    kwargs : dict
        parsed region
    """
    # import within function to avoid circular ref
    from . import MODELS  # global list of models

    kwargs = region.copy()
    # NOTE: the order is important to prioritize the arguments
    options = {
        "basin": ["basid", "geom", "bbox", "xy"],
        "subbasin": ["geom", "bbox", "xy"],
        "outlet": ["geom", "bbox"],
        "geom": ["geom"],
        "bbox": ["bbox"],
    }
    kind = next(iter(kwargs))
    value0 = kwargs.pop(kind)
    if kind in MODELS:
        kwargs = dict(mod=MODELS[kind](root=value0, mode="r", logger=logger))
        kind = "model"
    elif kind not in options:
        k_lst = '", "'.join(list(options.keys()))
        raise ValueError(f'Region key "{kind}" not understood, select from "{k_lst}"')
    else:
        kwarg = _parse_region_value(value0)
        if len(kwarg) == 0 or next(iter(kwarg)) not in options[kind]:
            v_lst = '", "'.join(list(options[kind]))
            raise ValueError(
                f'Region value "{value0}" for kind={kind} not understood, '
                f'provide one of "{v_lst}"'
            )
        kwargs.update(kwarg)
    kwargs_str = kwargs.copy()
    for k, v in kwargs.items():
        if isinstance(v, gpd.GeoDataFrame):
            v = f"GeoDataFrame ({v.index.size} rows)"
            kwargs_str.update({k: v})
    logger.debug(f"Parsed region (kind={kind}): {str(kwargs_str)}")
    return kind, kwargs


def _parse_region_value(value):
    kwarg = {}
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
    elif isinstance(value, str) and isfile(value):
        kwarg = dict(geom=gpd.read_file(value))
    elif isinstance(value, gpd.GeoDataFrame):  # geometry
        kwarg = dict(geom=value)
    elif isinstance(value, str) and isdir(value):
        kwarg = dict(root=value)
    if "geom" in kwarg and np.all(kwarg["geom"].geometry.type == "Point"):
        xy = (
            kwarg["geom"].geometry.x.values,
            kwarg["geom"].geometry.y.values,
        )
        kwarg = dict(xy=xy)
    return kwarg
