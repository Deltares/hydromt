"""MetaDataResolver for raster tindex files."""

from logging import Logger, getLogger
from os.path import abspath, dirname, join
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import geopandas as gpd
from fsspec import AbstractFileSystem

from hydromt._typing import NoDataStrategy, TimeRange, ZoomLevel
from hydromt.metadata_resolver.metadata_resolver import MetaDataResolver

logger: Logger = getLogger(__name__)


class RasterTindexResolver(MetaDataResolver):
    """Implementation of the MetaDataResolver for raster tindex files."""

    def resolve(
        self,
        uri: str,
        fs: AbstractFileSystem,
        mask: Optional[gpd.GeoDataFrame] = None,
        *,
        time_range: Optional[TimeRange] = None,
        zoom_level: Optional[ZoomLevel] = None,
        variables: Union[int, tuple[float, str], None] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Optional[Logger] = logger,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Resolve URIs of a raster tindex file."""
        gdf = gpd.read_file(uri)
        gdf = gdf.iloc[gdf.sindex.query(mask.to_crs(gdf.crs).unary_union)]
        tileindex: Optional[str] = options.get("tileindex")
        if tileindex is None:
            raise ValueError(
                f"{self.__class__.__name__} needs options specifying 'tileindex'"
            )
        if gdf.index.size == 0:
            raise IOError("No intersecting tiles found.")
        elif tileindex not in gdf.columns:
            raise IOError(
                f'Tile index "{tileindex}" column missing in tile index file.'
            )
        else:
            root = dirname(uri)
            paths = []
            for fn in gdf[tileindex]:
                path = Path(str(fn))
                if not path.is_absolute():
                    paths.append(str(Path(abspath(join(root, fn)))))
        return paths
