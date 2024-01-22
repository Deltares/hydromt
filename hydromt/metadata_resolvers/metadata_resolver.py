"""Metadata Resolver responsible for finding the data using the URI in the Data Catalog."""
from abc import ABC, abstractmethod

import geopandas as gpd
from pydantic import BaseModel

from hydromt.data_sources.data_source import DataSource
from hydromt.nodata import NoDataStrategy
from hydromt.typing import Bbox, TimeRange


class MetaDataResolver(BaseModel, ABC):
    """Metadata Resolver responsible for finding the data using the URI in the Data Catalog."""

    @abstractmethod
    def resolve_uri(
        self,
        uri: str,
        source: DataSource,
        *,
        timerange: TimeRange | None = None,
        bbox: Bbox | None = None,
        geom: gpd.GeoDataFrame | None = None,
        buffer: float = 0.0,
        predicate: str = "intersects",
        variables: list[str] | None = None,
        zoom_level: int = 0,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        **kwargs,
    ) -> list[str]:
        """Resolve metadata of data behind a single URI."""
        ...
