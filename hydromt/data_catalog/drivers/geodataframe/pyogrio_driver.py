"""Driver to read geodataframes using Pyogrio."""

from logging import Logger, getLogger
from os.path import splitext
from typing import ClassVar, List, Optional, Set, Union

import geopandas as gpd
import pandas as pd
from pyogrio import read_dataframe, read_info, write_dataframe
from pyproj import CRS

from hydromt._typing import Bbox, Geom, SourceMetadata, StrPath
from hydromt._typing.error import NoDataStrategy, exec_nodata_strat
from hydromt.data_catalog.drivers.geodataframe.geodataframe_driver import (
    GeoDataFrameDriver,
)

logger: Logger = getLogger(__name__)


class PyogrioDriver(GeoDataFrameDriver):
    """Driver to read GeoDataFrames using the `pyogrio` package."""

    name = "pyogrio"
    supports_writing = True
    _supported_extensions: ClassVar[Set[str]] = {".gpkg", ".shp", ".geojson", ".fgb"}

    def read(
        self,
        uris: List[str],
        *,
        mask: Optional[Geom] = None,
        variables: Optional[List[str]] = None,
        predicate: Optional[str] = None,
        metadata: Optional[SourceMetadata] = None,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
    ) -> gpd.GeoDataFrame:
        """
        Read data using pyogrio.

        Warning
        -------
        The 'predicate' keyword argument is unused in this method and is only present
        in the method's signature for compatibility with other functions.
        """
        if len(uris) > 1:
            raise ValueError(
                "DataFrame: Reading multiple files with the "
                f"{self.__class__.__name__} driver is not supported."
            )
        elif len(uris) == 0:
            gdf = gpd.GeoDataFrame()
        else:
            _uri = uris[0]
            if mask is not None:
                bbox = _bbox_from_file_and_mask(_uri, mask=mask)
            else:
                bbox = None
            gdf: Union[pd.DataFrame, gpd.GeoDataFrame] = read_dataframe(
                _uri, bbox=bbox, columns=variables, **self.options
            )
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise IOError(f"DataFrame from uri: '{_uri}' contains no geometry column.")

        crs = None
        if metadata is not None:
            crs = metadata.crs
        if crs is not None:
            if gdf.crs is not None:
                logger.warning(f"Overwriting crs of GeoDataFrame to {crs}")
            gdf.set_crs(crs)

        if gdf.index.size == 0:
            exec_nodata_strat(f"No data from driver {self}'.", strategy=handle_nodata)
        return gdf

    def write(
        self,
        path: StrPath,
        gdf: gpd.GeoDataFrame,
        **kwargs,
    ) -> str:
        """
        Write out a GeoDataFrame to file using pyogrio.

        args:
        """
        no_ext, ext = splitext(path)
        if ext not in self._supported_extensions:
            logger.warning(
                f"driver {self.name} has no support for extension {ext}"
                "switching to .fgb."
            )
            path = no_ext + ".fgb"

        write_dataframe(gdf, path, **kwargs)

        return str(path)


def _bbox_from_file_and_mask(
    uri: str,
    mask: Geom,
) -> Optional[Bbox]:
    """Create a bbox from the file metadata and mask given.

    Pyogrio's mask or bbox arguments require a mask or bbox in the same CRS as the data.
    This function takes the mask filter and crs of the input data
    and returns a bbox in the same crs as the data based on the input filters.

    Parameters
    ----------
    uri: str,
        URI of the data.
    mask: GeoDataFrame | GeoSeries | BaseGeometry
        mask to filter the data while reading.
    """
    source_crs = None
    if source_crs_str := read_info(uri).get("crs"):
        source_crs = CRS.from_user_input(source_crs_str)

    if not source_crs:
        logger.warning(
            f"Reading from uri: '{uri}' without CRS definition. Filtering with crs:"
            f" {mask.crs}, cannot compare crs."
        )
    elif mask.crs != source_crs:
        mask = mask.to_crs(source_crs)

    return tuple(mask.total_bounds)
