"""URIResolver for raster tindex files."""

from logging import Logger, getLogger
from os.path import abspath, dirname, join
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import geopandas as gpd
from fsspec import AbstractFileSystem

from hydromt._typing import NoDataStrategy, TimeRange, ZoomLevel, exec_nodata_strat
from hydromt.data_catalog.uri_resolvers.uri_resolver import URIResolver

logger: Logger = getLogger(__name__)


class RasterTindexResolver(URIResolver):
    """Implementation of the URIResolver for raster tindex files."""

    name = "raster_tindex"

    def resolve(
        self,
        uri: str,
        fs: AbstractFileSystem,
        *,
        time_range: Optional[TimeRange] = None,
        zoom_level: Optional[ZoomLevel] = None,
        mask: Optional[gpd.GeoDataFrame] = None,
        variables: Union[int, tuple[float, str], None] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Resolve URIs of a raster tindex file.

        Parameters
        ----------
        uri : str
            Unique Resource Identifier
        fs : AbstractFileSystem
            fsspec filesystem used to resolve wildcards in the uri
        time_range : Optional[TimeRange], optional
            left-inclusive start end time of the data, by default None
        mask : Optional[Geom], optional
            A geometry defining the area of interest, by default None
        zoom_level : Optional[ZoomLevel], optional
            zoom_level of the dataset, by default None
        variables : Optional[List[str]], optional
            Names of variables to return, or all if None, by default None
        handle_nodata : NoDataStrategy, optional
            how to react when no data is found, by default NoDataStrategy.RAISE
        options : Optional[Dict[str, Any]], optional
            extra options for this resolver, by default None

        Returns
        -------
        List[str]
            a list of expanded uris

        Raises
        ------
        NoDataException
            when no data is found and `handle_nodata` is `NoDataStrategy.RAISE`
        """
        if mask is None:
            raise ValueError(f"Resolver {self.name} needs a mask")
        gdf = gpd.read_file(uri)
        gdf = gdf.iloc[gdf.sindex.query(mask.to_crs(gdf.crs).union_all())]
        tileindex: Optional[str] = options.get("tileindex")
        if tileindex is None:
            raise ValueError(
                f"{self.__class__.__name__} needs options specifying 'tileindex'"
            )
        if gdf.index.size == 0:
            exec_nodata_strat(
                f"resolver '{self.name}' found no intersecting tiles.",
                strategy=handle_nodata,
            )
            return []  # in case of ignore

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
