"""Metadata Resolver responsible for finding the data using the URI in the Data Catalog."""
from abc import ABC, abstractmethod
from logging import Logger
from typing import TYPE_CHECKING, List, Optional

import geopandas as gpd
from pydantic import BaseModel

from hydromt._typing import Bbox, NoDataStrategy, Predicate, TimeRange

if TYPE_CHECKING:
    from hydromt.data_sources.data_source import DataSource


class MetaDataResolver(BaseModel, ABC):
    """Metadata Resolver responsible for finding the data using the URI in the Data Catalog."""

    @abstractmethod
    def resolve(
        self,
        source: "DataSource",
        *,
        timerange: Optional[TimeRange] = None,
        bbox: Optional[Bbox] = None,
        geom: Optional[gpd.GeoDataFrame] = None,
        buffer: float = 0.0,
        predicate: Predicate = "intersects",
        variables: Optional[List[str]] = None,
        zoom_level: int = 0,
        handle_nodata: NoDataStrategy = NoDataStrategy.RAISE,
        logger: Optional[Logger] = None,
        **kwargs,
    ) -> List[str]:
        """Resolve metadata of data behind a single URI."""
        ...
