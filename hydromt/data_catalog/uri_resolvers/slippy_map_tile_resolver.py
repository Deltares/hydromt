"""
Resolver for Slippy Map Tiles.

Should be able to load in Slippy map tiles using any convention.
https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames.
"""

from logging import Logger, getLogger
from typing import Dict, Optional, Tuple, Union

import rasterio
from pyproj import CRS

from hydromt._typing import Geom, StrPath
from hydromt.gis.raster_utils import cellres

logger: Logger = getLogger(__name__)


# TODO: fully implement in https://github.com/Deltares/hydromt/issues/875
def _get_zoom_levels_and_crs(
    uri: StrPath, logger=logger
) -> Tuple[int, Dict[int, float]]:
    """Get zoom levels and crs from adapter or detect from tif file if missing."""
    zoom_levels = {}
    crs = None
    try:
        with rasterio.open(uri) as src:
            res = abs(src.res[0])
            crs = src.crs
            overviews = [src.overviews(i) for i in src.indexes]
            if len(overviews[0]) > 0:  # check overviews for band 0
                # check if identical
                if not all([o == overviews[0] for o in overviews]):
                    raise ValueError("Overviews are not identical across bands")
                # dict with overview level and corresponding resolution
                zls = [1] + overviews[0]
                zoom_levels = {i: res * zl for i, zl in enumerate(zls)}
    except rasterio.RasterioIOError as e:
        logger.warning(f"IO error while detecting zoom levels: {e}")
    return zoom_levels, crs


def _parse_zoom_level(
    zoom_level: Union[int, Tuple[Union[int, float], str]],
    geom: Optional[Geom] = None,
    zls_dict: Optional[Dict[int, float]] = None,
    dst_crs: Optional[CRS] = None,
    logger=logger,
) -> Optional[int]:
    """Return overview level of data corresponding to zoom level.

    Parameters
    ----------
    zoom_level: int or tuple
        overview level or tuple with resolution and unit
    geom: gpd.GeoSeries, optional
        geometry to determine res if zoom_level or source in degree
    zls_dict: dict, optional
        dictionary with overview levels and corresponding resolution
    dst_crs: pyproj.CRS, optional
        destination crs to determine res if zoom_level tuple is provided
        with different unit than dst_crs
    """
    # check zoom level
    if zls_dict is None or len(zls_dict) == 0 or zoom_level is None:
        return None
    elif isinstance(zoom_level, int):
        if zoom_level not in zls_dict:
            raise ValueError(
                f"Zoom level {zoom_level} not defined." f"Select from {zls_dict}."
            )
        zl = zoom_level
        dst_res = zls_dict[zoom_level]
    elif (
        isinstance(zoom_level, tuple)
        and isinstance(zoom_level[0], (int, float))
        and isinstance(zoom_level[1], str)
        and len(zoom_level) == 2
        and dst_crs is not None
    ):
        src_res, src_res_unit = zoom_level
        # convert res if different unit than crs
        dst_crs = CRS.from_user_input(dst_crs)
        dst_crs_unit = dst_crs.axis_info[0].unit_name
        dst_res = src_res
        if dst_crs_unit != src_res_unit:
            known_units = ["degree", "metre", "US survey foot", "meter", "foot"]
            if src_res_unit not in known_units:
                raise TypeError(
                    f"zoom_level unit {src_res_unit} not understood;"
                    f" should be one of {known_units}"
                )
            if dst_crs_unit not in known_units:
                raise NotImplementedError(
                    f"no conversion available for {src_res_unit} to {dst_crs_unit}"
                )
            conversions = {
                "foot": 0.3048,
                "metre": 1,  # official pyproj units
                "US survey foot": 0.3048,  # official pyproj units
            }  # to meter
            if src_res_unit == "degree" or dst_crs_unit == "degree":
                lat = 0
                if geom is not None:
                    lat = geom.to_crs(4326).centroid.y.item()
                conversions["degree"] = cellres(lat=lat)[1]
            fsrc = conversions.get(src_res_unit, 1)
            fdst = conversions.get(dst_crs_unit, 1)
            dst_res = src_res * fsrc / fdst
        # find nearest zoom level
        res = list(zls_dict.values())[0] / 2
        zls = list(zls_dict.keys())
        smaller = [x < (dst_res + res * 0.01) for x in zls_dict.values()]
        zl = zls[-1] if all(smaller) else zls[max(smaller.index(False) - 1, 0)]
    elif dst_crs is None:
        raise ValueError("No CRS defined, hence no zoom level can be determined.")
    else:
        raise TypeError(f"zoom_level not understood: {type(zoom_level)}")
    logger.debug(f"Using zoom level {zl} ({dst_res:.2f})")
    return zl
