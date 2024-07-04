"""
Resolver for Slippy Map Tiles.

Should be able to load in Slippy map tiles using any convention.
https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames.
"""

from datetime import datetime
from logging import Logger, getLogger
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import rasterio
from fsspec import AbstractFileSystem

from hydromt._typing import StrPath, Zoom
from hydromt._typing.error import NoDataStrategy
from hydromt.data_catalog.uri_resolvers import ConventionResolver

logger: Logger = getLogger(__name__)


class SlippyMapTileResolver(ConventionResolver):
    """Resolves slippy map tiles."""

    def resolve(
        self,
        uri: str,
        fs: AbstractFileSystem,
        *,
        time_range: Optional[Tuple[datetime]] = None,
        mask: Union[gpd.GeoDataFrame, gpd.GeoSeries, None] = None,
        variables: Optional[List[str]] = None,
        zoom: Optional[Zoom] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Resolves the uri."""  # noqa
        zoom_levels, crs = _get_zoom_levels_and_crs(uri)


def _get_zoom_levels_and_crs(uri: StrPath) -> Tuple[Dict[int, float], int]:
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
