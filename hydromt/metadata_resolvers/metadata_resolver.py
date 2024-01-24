"""Metadata Resolver responsible for finding the data using the URI in the Data Catalog."""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import geopandas as gpd
from pydantic import BaseModel

from hydromt.nodata import NoDataStrategy
from hydromt.typing import Bbox, TimeRange

if TYPE_CHECKING:
    from hydromt.data_sources.data_source import DataSource


class MetaDataResolver(BaseModel, ABC):
    """Metadata Resolver responsible for finding the data using the URI in the Data Catalog."""

    @abstractmethod
    def resolve(
        self,
        source: "DataSource",
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
