"""Generic DataSource for GeoDataFrames."""

from logging import Logger
from typing import Iterable

import geopandas as gpd

from hydromt.nodata import NoDataStrategy

from .data_source import DataSource


class GeoDataFrameDataSource(DataSource):
    """
    DataSource for GeoDataFrames.

    Reads and validates DataCatalog entries.
    """

    def read_data(
        self,
        uri: str | Iterable[str],
        bbox: list[float],
        mask: gpd.GeoDataFrame,
        buffer: float,
        predicate: str,
        handle_nodata: NoDataStrategy,
        logger: Logger,
    ) -> gpd.GeoDataFrame:
        """Use initialize driver to read data."""
        pass
